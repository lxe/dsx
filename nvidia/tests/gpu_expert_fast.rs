#![cfg(feature = "cuda")]
//! Fast expert kernels (expert_fast.cuh): repacked GEMV vs the CPU block decoders, and moe_*_fast vs moe_gu / moe_down.
use dsx::experts::{KS, ExpCfg, GpuExperts, RawExpert, repack, repacked_id};
use dsx::formats::dequant_row;
use dsx::gguf::GType;
use dsx::gpu::*;

fn rng(seed: &mut u64) -> u64 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    *seed
}

/// random blocks with sane fp16 scales
fn random_rows(ty: GType, rows: usize, cols: usize, seed: &mut u64) -> Vec<u8> {
    let (w, b) = ty.block();
    let nb = rows * cols / w;
    let mut v: Vec<u8> = (0..nb * b).map(|_| rng(seed) as u8).collect();
    let at = match ty {
        GType::Q8_0 | GType::Q2_0 | GType::Iq2Xxs | GType::Iq1S | GType::Iq2Xs | GType::Iq2S => 0,
        GType::Ptq1_0 => 26,
        GType::Q3K => 108,
        GType::Q2K => 80,
        GType::Iq3Xxs => 0,
        _ => unreachable!(),
    };
    let scale = if ty == GType::Q8_0 { 1e-3 } else { 0.01 };
    for blk in 0..nb {
        let s = half::f16::from_f32(scale * (1.0 + (rng(seed) % 100) as f32 * 0.01)).to_le_bytes();
        v[blk * b + at..blk * b + at + 2].copy_from_slice(&s);
        if ty == GType::Q2K { v[blk*b+82..blk*b+84].copy_from_slice(&half::f16::from_f32(0.003).to_le_bytes()); }
    }
    v
}

fn random_x(n: usize, outlier: f32, seed: &mut u64) -> Vec<f32> {
    // in [-1, 1], every 97th value scaled by `outlier`
    (0..n).map(|i| ((rng(seed) % 2001) as f32 / 1000.0 - 1.0) * if i % 97 == 0 { outlier } else { 1.0 }).collect()
}

/// quant_act on the CPU: per 256 d = amax / 127, q = round(x / d); returns q * d
fn quantized(x: &[f32]) -> Vec<f32> {
    let mut o = vec![0f32; x.len()];
    for (xb, ob) in x.chunks(256).zip(o.chunks_mut(256)) {
        let d = xb.iter().fold(0f32, |a, &v| a.max(v.abs())) / 127.0;
        let id = if d > 0.0 { 1.0 / d } else { 0.0 };
        for (o, &v) in ob.iter_mut().zip(xb) {
            *o = (v * id).round_ties_even() * d;
        }
    }
    o
}

fn rms(v: &[f32]) -> f32 {
    (v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>() / v.len() as f64).sqrt() as f32
}

/// max |a - b| / rms(a)
fn err(want: &[f32], got: &[f32]) -> f32 {
    want.iter().zip(got).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max) / rms(want)
}

/// rms(a - b) / rms(a)
fn err_rms(want: &[f32], got: &[f32]) -> f32 {
    rms(&want.iter().zip(got).map(|(a, b)| a - b).collect::<Vec<_>>()) / rms(want)
}

struct QBufs {
    q: Buf,
    d: Buf,
    s: Buf,
}

fn quant(g: &Gpu, x: u64, k: usize, nvec: usize) -> QBufs {
    let b = QBufs { q: g.alloc(nvec * k).unwrap(), d: g.alloc(nvec * k / 64).unwrap(), s: g.alloc(nvec * k / 32).unwrap() };
    g.launch(g.kernel("quant_act").unwrap(), ((k / 256) as u32, nvec as u32, 1), 256, 0,
        &[Arg::P(x), Arg::I(k as i32), Arg::P(b.q.ptr), Arg::P(b.d.ptr), Arg::P(b.s.ptr), Arg::P(0), Arg::P(0), Arg::I(0), Arg::I(0), Arg::I(0), Arg::I(0), Arg::P(0), Arg::I(0)]).unwrap();
    b
}

#[test]
fn fast_gemv_matches_cpu() {
    let g = Gpu::new(0).unwrap();
    let mut seed = 0x1234_5678_9abc_def0u64;
    for ty in [GType::Q2K, GType::Iq3Xxs, GType::Q8_0, GType::Q2_0, GType::Ptq1_0, GType::Q3K, GType::Iq2Xxs, GType::Iq1S, GType::Iq2Xs, GType::Iq2S] {
        for (rows, k) in [(200usize, 5120usize), (136, 2304)] {
            let w = random_rows(ty, rows, k, &mut seed);
            let rb = ty.row_bytes(k);
            let mut wf = vec![0f32; rows * k];
            for r in 0..rows {
                dequant_row(ty, &w[r * rb..(r + 1) * rb], &mut wf[r * k..(r + 1) * k]);
            }
            let src = g.upload(&w).unwrap();
            let dst = g.alloc(w.len()).unwrap();
            repack(&g, ty, src.ptr, dst.ptr, rows, k).unwrap();
            let id = repacked_id(ty).unwrap();
            // the prefill path reads the repacked store through dequant_f16
            let w16 = g.alloc(rows * k * 2).unwrap();
            g.launch(g.kernel("dequant_f16").unwrap(), ((k as u32).div_ceil(256), rows as u32, 1), 256, 0,
                &[Arg::I(id), Arg::P(dst.ptr), Arg::P(0), Arg::P(w16.ptr), Arg::I(rows as i32), Arg::I(k as i32)]).unwrap();
            let got16: Vec<half::f16> = g.download(w16.ptr, rows * k).unwrap();
            let bad = wf.iter().zip(&got16).filter(|(a, b)| (**a - b.to_f32()).abs() > 1e-3 * a.abs() + 1e-7).count();
            assert_eq!(bad, 0, "{ty:?} dequant_f16 of the repacked layout");
            // and of the raw GGUF blocks (experts streamed from RAM in prefill)
            g.launch(g.kernel("dequant_f16").unwrap(), ((k as u32).div_ceil(256), rows as u32, 1), 256, 0,
                &[Arg::I(ty as u32 as i32), Arg::P(src.ptr), Arg::P(0), Arg::P(w16.ptr), Arg::I(rows as i32), Arg::I(k as i32)]).unwrap();
            let got16: Vec<half::f16> = g.download(w16.ptr, rows * k).unwrap();
            let bad = wf.iter().zip(&got16).filter(|(a, b)| (**a - b.to_f32()).abs() > 1e-3 * a.abs() + 1e-7).count();
            assert_eq!(bad, 0, "{ty:?} dequant_f16 of the raw blocks");
            for t in [1usize, 3, 6] {
                let x = random_x(t * k, 1.0, &mut seed);
                let xq = quantized(&x);
                let dot = |xs: &[f32]| -> Vec<f32> {
                    let mut y = vec![0f32; t * rows];
                    for tt in 0..t {
                        for r in 0..rows {
                            y[tt * rows + r] = wf[r * k..(r + 1) * k].iter().zip(&xs[tt * k..(tt + 1) * k]).map(|(a, b)| *a as f64 * *b as f64).sum::<f64>() as f32;
                        }
                    }
                    y
                };
                let (want, want_q) = (dot(&x), dot(&xq));
                let dx = g.upload(&x).unwrap();
                let qb = quant(&g, dx.ptr, k, t);
                let dy = g.alloc(t * rows * 4).unwrap();
                g.launch(g.kernel("xgemv").unwrap(), ((rows as u32).div_ceil(16 * dsx::experts::ROWT), 1, 1), 32 * KS, dsx::experts::tile_smem(ty),
                    &[Arg::I(id), Arg::P(dst.ptr), Arg::I(rows as i32), Arg::I(k as i32), Arg::I(t as i32), Arg::P(qb.q.ptr), Arg::P(qb.d.ptr), Arg::P(qb.s.ptr), Arg::P(dy.ptr)]).unwrap();
                let got: Vec<f32> = g.download(dy.ptr, t * rows).unwrap();
                let (eq, ef) = (err(&want_q, &got), err(&want, &got));
                eprintln!("{ty:?} {rows}x{k} T={t}: vs int8-x reference {eq:.1e}, vs f32 reference: max {ef:.2e}, rms {:.2e} (of rms)", err_rms(&want, &got));
                assert!(eq < 1e-5, "{ty:?} {rows}x{k} T={t}: kernel differs from its exact reference");
                assert!(ef < 2e-2, "{ty:?} {rows}x{k} T={t}: int8 activation error");
            }
        }
    }
}

fn bf16r(x: f32) -> f32 {
    half::bf16::from_f32(x).to_f32()
}

/// rows of a matrix (raw GGUF) dotted with each of `xs` (f64 accumulation)
fn mat_dots(ty: GType, raw: &[u8], rows: usize, k: usize, xs: &[&[f32]]) -> Vec<Vec<f32>> {
    let rb = ty.row_bytes(k);
    let mut w = vec![0f32; k];
    let mut out = vec![vec![0f32; rows]; xs.len()];
    for r in 0..rows {
        dequant_row(ty, &raw[r * rb..(r + 1) * rb], &mut w);
        for (o, x) in out.iter_mut().zip(xs) {
            o[r] = w.iter().zip(*x).map(|(a, b)| *a as f64 * *b as f64).sum::<f64>() as f32;
        }
    }
    out
}

#[test]
fn moe_fast_matches_moe_gu_down() {
    let g = Gpu::new(0).unwrap();
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let c = ExpCfg { n_layers: 2, n_experts: 14, inter: 2304, dim: 5120, lim: 10.0, rot: 0 };
    let kinds = [(GType::Q2_0, GType::Q2_0), (GType::Ptq1_0, GType::Q2_0), (GType::Q3K, GType::Q3K), (GType::Q2_0, GType::Q3K), (GType::Ptq1_0, GType::Ptq1_0), (GType::Q8_0, GType::Q8_0)];
    // layer 1: experts 0..10 resident, 10..14 not
    let mats: Vec<(GType, GType, Vec<u8>, Vec<u8>, Vec<u8>)> = (0..10)
        .map(|e| {
            let (gu, dn) = kinds[e % kinds.len()];
            (gu, dn, random_rows(gu, c.inter, c.dim, &mut seed), random_rows(gu, c.inter, c.dim, &mut seed), random_rows(dn, c.dim, c.inter, &mut seed))
        })
        .collect();
    let items: Vec<((usize, usize), RawExpert)> =
        mats.iter().enumerate().map(|(e, m)| ((1, e), RawExpert { gate: &m.2, up: &m.3, down: &m.4, gu: m.0, dn: m.1 })).collect();
    let old = GpuExperts::build(&g, c, &items, false).unwrap();
    let new = GpuExperts::build(&g, c, &items, true).unwrap();
    assert!(new.is_fast() && !old.is_fast());
    let k = 6;
    for t in [1usize, 3, 6] {
        // picks from 14 experts: tokens share some, a few are not resident
        let mut sel = vec![0i32; t * k];
        for tt in 0..t {
            let mut picked = vec![];
            while picked.len() < k {
                let e = (rng(&mut seed) % 14) as i32;
                if !picked.contains(&e) {
                    picked.push(e);
                }
            }
            sel[tt * k..(tt + 1) * k].copy_from_slice(&picked);
        }
        let wts: Vec<f32> = (0..t * k).map(|_| 0.05 + (rng(&mut seed) % 100) as f32 * 0.003).collect();
        let x: Vec<f32> = random_x(t * c.dim, 1.0, &mut seed);
        let mut want = vec![0f32; t * c.dim];
        for tt in 0..t {
            g.htod_sync(old.sel.ptr, &sel[tt * k..(tt + 1) * k]).unwrap();
            g.htod_sync(old.wts.ptr, &wts[tt * k..(tt + 1) * k]).unwrap();
            g.htod_sync(old.x.ptr, &x[tt * c.dim..(tt + 1) * c.dim]).unwrap();
            old.run_device(&g, 1, old.sel.ptr, old.wts.ptr, k, old.x.ptr, old.out.ptr).unwrap();
            let o: Vec<f32> = g.download(old.out.ptr, c.dim).unwrap();
            want[tt * c.dim..(tt + 1) * c.dim].copy_from_slice(&o);
        }
        g.htod_sync(new.sel.ptr, &sel).unwrap();
        g.htod_sync(new.wts.ptr, &wts).unwrap();
        g.htod_sync(new.x.ptr, &x).unwrap();
        new.run_device_fast(&g, 1, new.sel.ptr, new.wts.ptr, k, t, new.x.ptr, new.out.ptr).unwrap();
        let got: Vec<f32> = g.download(new.out.ptr, t * c.dim).unwrap();
        for tt in 0..t {
            let r = tt * c.dim..(tt + 1) * c.dim;
            let e = err(&want[r.clone()], &got[r.clone()]);
            let res = sel[tt * k..(tt + 1) * k].iter().filter(|&&e| e < 10).count();
            eprintln!("moe T={t} token {tt} ({res} resident picks) vs moe_gu + moe_down: max err {e:.2e}, rms err {:.2e} (of rms)", err_rms(&want[r.clone()], &got[r.clone()]));
            assert!(e < 0.1 && err_rms(&want[r.clone()], &got[r.clone()]) < 2.5e-2, "token {tt}");
        }
        // exact semantics: the same computation on the CPU with x and h quantized like quant_act
        let mut emu = vec![0f32; t * c.dim];
        let xq: Vec<Vec<f32>> = (0..t).map(|tt| quantized(&x[tt * c.dim..(tt + 1) * c.dim])).collect();
        for (e, m) in mats.iter().enumerate() {
            let users: Vec<(usize, usize)> = (0..t).filter_map(|tt| sel[tt * k..(tt + 1) * k].iter().position(|&s| s == e as i32).map(|j| (tt, j))).collect();
            if users.is_empty() {
                continue;
            }
            let xs: Vec<&[f32]> = users.iter().map(|&(tt, _)| xq[tt].as_slice()).collect();
            let (gt, ut) = (mat_dots(m.0, &m.2, c.inter, c.dim, &xs), mat_dots(m.0, &m.3, c.inter, c.dim, &xs));
            let hs: Vec<Vec<f32>> = users
                .iter()
                .enumerate()
                .map(|(n, &(tt, j))| {
                    let h: Vec<f32> = (0..c.inter)
                        .map(|r| {
                            let (gv, uv) = (gt[n][r].min(c.lim), ut[n][r].clamp(-c.lim, c.lim));
                            bf16r(gv / (1.0 + (-gv).exp()) * uv * wts[tt * k + j])
                        })
                        .collect();
                    quantized(&h)
                })
                .collect();
            let hr: Vec<&[f32]> = hs.iter().map(|v| v.as_slice()).collect();
            let o = mat_dots(m.1, &m.4, c.dim, c.inter, &hr);
            for (n, &(tt, _)) in users.iter().enumerate() {
                for r in 0..c.dim {
                    emu[tt * c.dim + r] += o[n][r];
                }
            }
        }
        let ee = err(&emu, &got);
        eprintln!("moe T={t}: vs the CPU emulation of quant_act + moe_*_fast: max err {ee:.1e} (of rms)");
        assert!(ee < 1e-3, "moe_*_fast semantics");
        // a layer with nothing resident gives zeros
        new.run_device_fast(&g, 0, new.sel.ptr, new.wts.ptr, k, t, new.x.ptr, new.out.ptr).unwrap();
        let z: Vec<f32> = g.download(new.out.ptr, t * c.dim).unwrap();
        assert!(z.iter().all(|&v| v == 0.0));
    }
}

/// moe_*_fast over a store that mixes the codebook formats with scalar ones (codebook tables in dynamic shared memory)
/// against the CPU emulation of quant_act + moe_*_fast; the old moe_gu / moe_down have no IQ kernels.
#[test]
fn moe_fast_iq_matches_cpu() {
    let g = Gpu::new(0).unwrap();
    let mut seed = 0x51ed_270b_27b9_d1a3u64;
    let c = ExpCfg { n_layers: 2, n_experts: 14, inter: 2304, dim: 5120, lim: 10.0, rot: 0 };
    let kinds = [(GType::Iq1S, GType::Q2K), (GType::Iq2S, GType::Iq3Xxs), (GType::Iq3Xxs, GType::Q2K), (GType::Iq1S, GType::Iq2Xxs), (GType::Iq2Xxs, GType::Iq2S), (GType::Iq2Xs, GType::Iq2Xs), (GType::Iq1S, GType::Q2_0), (GType::Ptq1_0, GType::Iq2S)];
    let mats: Vec<(GType, GType, Vec<u8>, Vec<u8>, Vec<u8>)> = (0..10)
        .map(|e| {
            let (gu, dn) = kinds[e % kinds.len()];
            (gu, dn, random_rows(gu, c.inter, c.dim, &mut seed), random_rows(gu, c.inter, c.dim, &mut seed), random_rows(dn, c.dim, c.inter, &mut seed))
        })
        .collect();
    let items: Vec<((usize, usize), RawExpert)> =
        mats.iter().enumerate().map(|(e, m)| ((1, e), RawExpert { gate: &m.2, up: &m.3, down: &m.4, gu: m.0, dn: m.1 })).collect();
    let new = GpuExperts::build(&g, c, &items, true).unwrap();
    let k = 6;
    for t in [1usize, 3, 6] {
        let mut sel = vec![0i32; t * k];
        for tt in 0..t {
            let mut picked = vec![];
            while picked.len() < k {
                let e = (rng(&mut seed) % 14) as i32;
                if !picked.contains(&e) {
                    picked.push(e);
                }
            }
            sel[tt * k..(tt + 1) * k].copy_from_slice(&picked);
        }
        let wts: Vec<f32> = (0..t * k).map(|_| 0.05 + (rng(&mut seed) % 100) as f32 * 0.003).collect();
        let x: Vec<f32> = random_x(t * c.dim, 1.0, &mut seed);
        g.htod_sync(new.sel.ptr, &sel).unwrap();
        g.htod_sync(new.wts.ptr, &wts).unwrap();
        g.htod_sync(new.x.ptr, &x).unwrap();
        new.run_device_fast(&g, 1, new.sel.ptr, new.wts.ptr, k, t, new.x.ptr, new.out.ptr).unwrap();
        let got: Vec<f32> = g.download(new.out.ptr, t * c.dim).unwrap();
        let mut emu = vec![0f32; t * c.dim];
        let xq: Vec<Vec<f32>> = (0..t).map(|tt| quantized(&x[tt * c.dim..(tt + 1) * c.dim])).collect();
        for (e, m) in mats.iter().enumerate() {
            let users: Vec<(usize, usize)> = (0..t).filter_map(|tt| sel[tt * k..(tt + 1) * k].iter().position(|&s| s == e as i32).map(|j| (tt, j))).collect();
            if users.is_empty() {
                continue;
            }
            let xs: Vec<&[f32]> = users.iter().map(|&(tt, _)| xq[tt].as_slice()).collect();
            let (gt, ut) = (mat_dots(m.0, &m.2, c.inter, c.dim, &xs), mat_dots(m.0, &m.3, c.inter, c.dim, &xs));
            let hs: Vec<Vec<f32>> = users
                .iter()
                .enumerate()
                .map(|(n, &(tt, j))| {
                    let h: Vec<f32> = (0..c.inter)
                        .map(|r| {
                            let (gv, uv) = (gt[n][r].min(c.lim), ut[n][r].clamp(-c.lim, c.lim));
                            bf16r(gv / (1.0 + (-gv).exp()) * uv * wts[tt * k + j])
                        })
                        .collect();
                    quantized(&h)
                })
                .collect();
            let hr: Vec<&[f32]> = hs.iter().map(|v| v.as_slice()).collect();
            let o = mat_dots(m.1, &m.4, c.dim, c.inter, &hr);
            for (n, &(tt, _)) in users.iter().enumerate() {
                for r in 0..c.dim {
                    emu[tt * c.dim + r] += o[n][r];
                }
            }
        }
        let ee = err(&emu, &got);
        eprintln!("moe IQ T={t}: vs the CPU emulation of quant_act + moe_*_fast: max err {ee:.1e} (of rms)");
        assert!(ee < 1e-3, "moe_*_fast semantics with IQ types");
    }
}
