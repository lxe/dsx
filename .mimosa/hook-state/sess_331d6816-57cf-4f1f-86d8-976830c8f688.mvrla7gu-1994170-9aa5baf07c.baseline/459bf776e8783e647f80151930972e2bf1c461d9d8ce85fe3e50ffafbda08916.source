//! Kernel self-tests against CPU implementations (dsx_core::numerics), on synthetic inputs: no model needed.

use crate::hip::{Arg, Gpu};
use anyhow::Result;
use dsx_core::numerics::*;

struct Rng(u64);
impl Rng {
    fn f(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

fn bf16r(v: &[f32]) -> Vec<u16> {
    v.iter().map(|&x| half::bf16::from_f32(x).to_bits()).collect()
}
fn from_bf16(v: &[u16]) -> Vec<f32> {
    v.iter().map(|&b| half::bf16::from_bits(b).to_f32()).collect()
}

fn report(name: &str, got: &[f32], want: &[f32]) {
    let md = got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    let n: f32 = want.iter().map(|x| x * x).sum::<f32>().sqrt();
    let d: f32 = got.iter().zip(want).map(|(a, b)| (a - b) * (a - b)).sum::<f32>().sqrt();
    let worst = got.iter().zip(want).enumerate().max_by(|a, b| (a.1.0 - a.1.1).abs().partial_cmp(&(b.1.0 - b.1.1).abs()).unwrap()).map(|(i, (a, b))| (i, *a, *b));
    println!("{name:<28} max|d| {md:.3e}  rel {:.3e}  worst {:?}", d / n.max(1e-30), worst);
}

pub fn run() -> Result<()> {
    let image: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/dsxa.hsaco"));
    let alt = std::env::var("DSXA_HSACO").ok().map(std::fs::read).transpose()?;
    let g = Gpu::new(alt.as_deref().unwrap_or(image))?;
    let mut r = Rng(0x9E3779B97F4A7C15);
    let (hd, rd) = (512usize, 64usize);
    let rope = Rope::new(rd, 0, 10000.0, 16.0, 32.0, 1.0);
    let ropec = Rope::new(rd, 65536, 160000.0, 16.0, 32.0, 1.0);
    let rf = g.upload(&rope.freqs)?;
    let rfc = g.upload(&ropec.freqs)?;
    let pos = 37usize;
    let st = g.upload(&[pos as i32, 0, 0, 0])?;

    // ---- attn_kv window path: rmsnorm(kv_norm) -> rope(pos) -> fake_fp8 -> bf16 ring slot
    {
        let kv: Vec<f32> = (0..hd).map(|_| r.f() * 3.0).collect();
        let w: Vec<f32> = (0..hd).map(|_| 0.5 + r.f().abs()).collect();
        let dkv = g.upload(&kv)?;
        let dw = g.upload(&w)?;
        let ring = g.alloc_zeroed(128 * hd * 2)?;
        g.launch(
            "attn_kv",
            (1, 1, 1),
            512,
            0,
            &[Arg::P(st.ptr), Arg::P(dkv.ptr), Arg::P(dw.ptr), Arg::P(ring.ptr), Arg::P(rf.ptr), Arg::I(0), Arg::P(0), Arg::P(0), Arg::P(0), Arg::P(0), Arg::P(0), Arg::P(0), Arg::P(0), Arg::P(0), Arg::P(rfc.ptr), Arg::F(1e-20)],
        )?;
        let got: Vec<u16> = g.dtoh(ring.at(pos % 128 * hd * 2), hd)?;
        let mut want = kv.clone();
        rmsnorm_inplace(&mut want, &w, 1e-20);
        rope.apply(&mut want[hd - rd..], pos, false);
        fake_fp8(&mut want, 32);
        report("attn_kv window row", &from_bf16(&got), &want);
    }

    // ---- sparse attention: NW window rows + NC compressed rows, 64 heads, sink; merge with inverse rope + fp8
    for &(nw, nc) in &[(1usize, 0usize), (2, 0), (38, 0), (38, 19), (128, 512)] {
        let t = nw - 1 + if nw == 128 { 200 } else { 0 };
        let st2 = g.upload(&[t as i32, 0, 0, 0])?;
        let nh = 64;
        let q: Vec<f32> = (0..nh * hd).map(|_| r.f() * 0.2).collect();
        let ring: Vec<f32> = (0..128 * hd).map(|_| r.f()).collect();
        let comp: Vec<f32> = (0..600 * hd).map(|_| r.f()).collect();
        let ringb = bf16r(&ring);
        let compb = bf16r(&comp);
        let cidx: Vec<i32> = (0..512).map(|i| if i < nc { (i * 3 % 600) as i32 + 128 } else { -1 }).collect();
        let sinks: Vec<f32> = (0..nh).map(|_| r.f() * 2.0).collect();
        let (dq, dring, dcomp, dcidx, dnc, dsink) = (g.upload(&q)?, g.upload(&ringb)?, g.upload(&compb)?, g.upload(&cidx)?, g.upload(&[nc as i32])?, g.upload(&sinks)?);
        let nchunk = (128 + 512usize).div_ceil(16);
        let part = g.alloc(nchunk * nh * (hd + 2) * 4)?;
        let oq = g.alloc(nh * hd)?;
        let os = g.alloc(nh * hd / 32 * 4)?;
        let scale = 1.0 / (hd as f32).sqrt();
        g.launch("attn_part", (nchunk as u32, 4, 1), 256, 0, &[Arg::P(st2.ptr), Arg::P(dq.ptr), Arg::P(dring.ptr), Arg::P(dcomp.ptr), Arg::P(dcidx.ptr), Arg::P(dnc.ptr), Arg::I(128), Arg::F(scale), Arg::P(part.ptr)])?;
        g.launch("attn_merge", (nh as u32, 1, 1), 512, 0, &[Arg::P(st2.ptr), Arg::P(part.ptr), Arg::I(nchunk as i32), Arg::P(dsink.ptr), Arg::P(rf.ptr), Arg::P(oq.ptr), Arg::P(os.ptr)])?;
        if nw == 128 {
            let us = g.time(|| {
                for _ in 0..50 {
                    g.launch("attn_part", (nchunk as u32, 4, 1), 256, 0, &[Arg::P(st2.ptr), Arg::P(dq.ptr), Arg::P(dring.ptr), Arg::P(dcomp.ptr), Arg::P(dcidx.ptr), Arg::P(dnc.ptr), Arg::I(128), Arg::F(scale), Arg::P(part.ptr)])?;
                }
                Ok(())
            })?;
            let us2 = g.time(|| {
                for _ in 0..50 {
                    g.launch("attn_merge", (nh as u32, 1, 1), 512, 0, &[Arg::P(st2.ptr), Arg::P(part.ptr), Arg::I(nchunk as i32), Arg::P(dsink.ptr), Arg::P(rf.ptr), Arg::P(oq.ptr), Arg::P(os.ptr)])?;
                }
                Ok(())
            })?;
            println!("attn_part {:.1} us, attn_merge {:.1} us (stream, isolated)", us / 50.0, us2 / 50.0);
            if std::env::var_os("ATT_TIMING").is_some() {
                let tb = g.alloc(64 * 64 * 514 * 4 + 4096 * 40)?;
                g.launch("attn_part", (nchunk as u32, 4, 1), 256, 0, &[Arg::P(st2.ptr), Arg::P(dq.ptr), Arg::P(dring.ptr), Arg::P(dcomp.ptr), Arg::P(dcidx.ptr), Arg::P(dnc.ptr), Arg::I(128), Arg::F(scale), Arg::P(tb.ptr)])?;
                let t: Vec<i64> = g.dtoh(tb.ptr + (64 * 64 * 514 * 4) as u64, nchunk * 4 * 5)?;
                let t0 = t.chunks(5).map(|c| c[0]).min().unwrap();
                let mut ph = [0f64; 4];
                let mut start_spread = 0i64;
                let mut end = 0i64;
                for c in t.chunks(5) {
                    for i in 0..4 { ph[i] += (c[i + 1] - c[i]) as f64 * 10.0 / (t.len() / 5) as f64; }
                    start_spread = start_spread.max(c[0] - t0);
                    end = end.max(c[4] - t0);
                }
                println!("attn_part phases (ns, mean over blocks): rowsrc {:.0} rows {:.0} scores {:.0} out {:.0}; last block starts at {} ns, ends at {} ns", ph[0], ph[1], ph[2], ph[3], start_spread * 10, end * 10);
            }
        }
        let codes: Vec<u8> = g.dtoh(oq.ptr, nh * hd)?;
        let scales: Vec<f32> = g.dtoh(os.ptr, nh * hd / 32)?;
        let got: Vec<f32> = codes.iter().enumerate().map(|(i, &c)| fnuz(c) * scales[i / 32]).collect();
        // CPU
        let ringf = from_bf16(&ringb);
        let compf = from_bf16(&compb);
        let oldest = if t + 1 > 128 { (t + 1) % 128 } else { 0 };
        let mut rows: Vec<&[f32]> = (0..nw).map(|i| &ringf[((oldest + i) % 128) * hd..][..hd]).collect();
        for &ci in &cidx[..nc] {
            rows.push(&compf[(ci as usize - 128) * hd..][..hd]);
        }
        let mut want = vec![0f32; nh * hd];
        for h in 0..nh {
            let qh = &q[h * hd..(h + 1) * hd];
            let sc: Vec<f32> = rows.iter().map(|row| qh.iter().zip(row.iter()).map(|(a, b)| a * b).sum::<f32>() * scale).collect();
            let mx = sc.iter().cloned().fold(-1e30f32, f32::max);
            let ex: Vec<f32> = sc.iter().map(|s| (s - mx).exp()).collect();
            let den: f32 = ex.iter().sum::<f32>() + (sinks[h] - mx).exp();
            let oh = &mut want[h * hd..(h + 1) * hd];
            for (e, row) in ex.iter().zip(&rows) {
                for j in 0..hd {
                    oh[j] += e / den * row[j];
                }
            }
            round_bf16_slice(oh);
            rope.apply(&mut oh[hd - rd..], t, true);
        }
        fake_fp8(&mut want, 32);
        report(&format!("attention w{nw} c{nc} (t={t})"), &got, &want);
    }
    // ---- q_b GEMV RoPE epilogue: epi=1 output == rope(epi=0 output) per 512-dim head
    {
        let (n, k) = (2048usize, 1280usize);
        let w: Vec<u8> = (0..n * k).map(|_| { let b = ((r.f() + 1.0) * 127.0) as u8; if b & 0x7f == 0x7f { 0x30 } else { b } }).collect();
        let sc: Vec<u8> = (0..(n / 32) * (k / 32)).map(|_| 120u8).collect();
        let wt = crate::load::tile_fp8(&w, n, k);
        let (dw, dsc) = (g.upload(&wt)?, g.upload(&sc)?);
        let x: Vec<f32> = (0..k).map(|_| r.f()).collect();
        let dx = g.upload(&x)?;
        let y0 = g.alloc_zeroed(n * 4)?;
        let y1 = g.alloc_zeroed(n * 4)?;
        crate::decode::test_gemv(&g, dw.ptr, dsc.ptr, n, k, dx.ptr, y0.ptr, 0, 0, st.ptr, rf.ptr, rfc.ptr)?;
        crate::decode::test_gemv(&g, dw.ptr, dsc.ptr, n, k, dx.ptr, y1.ptr, 1, 0, st.ptr, rf.ptr, rfc.ptr)?;
        let a: Vec<f32> = g.dtoh(y0.ptr, n)?;
        let b: Vec<f32> = g.dtoh(y1.ptr, n)?;
        let mut want = a.clone();
        for h in 0..n / hd {
            rope.apply(&mut want[h * hd + hd - rd..(h + 1) * hd], pos, false);
        }
        report("q_b rope epilogue", &b, &want);
        // gemv2 on an FP8 activation (codes with scale 1): epilogue rope consistency + agreement with gemv_jobs
        let codes: Vec<u8> = (0..k).map(|i| [0x38u8, 0x40, 0xC4, 0x30, 0xB8, 0x48][i % 6]).collect();
        let xs1 = vec![1f32; k / 32];
        let (dq, dxs) = (g.upload(&codes)?, g.upload(&xs1)?);
        let z0 = g.alloc_zeroed(n * 4)?;
        let z1 = g.alloc_zeroed(n * 4)?;
        crate::decode::test_gemv2(&g, dw.ptr, dsc.ptr, n, k, dq.ptr, dxs.ptr, z0.ptr, 0, st.ptr, rf.ptr, rfc.ptr)?;
        crate::decode::test_gemv2(&g, dw.ptr, dsc.ptr, n, k, dq.ptr, dxs.ptr, z1.ptr, 1, st.ptr, rf.ptr, rfc.ptr)?;
        let a2: Vec<f32> = g.dtoh(z0.ptr, n)?;
        let b2: Vec<f32> = g.dtoh(z1.ptr, n)?;
        let mut want2 = a2.clone();
        for h in 0..n / hd {
            rope.apply(&mut want2[h * hd + hd - rd..(h + 1) * hd], pos, false);
        }
        report("gemv2 rope epilogue", &b2, &want2);
        let xv: Vec<f32> = codes.iter().map(|&c| fnuz(c)).collect();
        let cpu2: Vec<f32> = (0..n).map(|i| (0..k).map(|c| dsx_core::model::fp8_e4m3_to_f32(w[i * k + c]) * f32::from_bits((sc[(i / 32) * (k / 32) + c / 32] as u32) << 23) * xv[c]).sum()).collect();
        report("gemv2 fp8 vs cpu", &a2, &cpu2);
        // and the GEMV itself against the CPU (FP8 OCP weights x fake_fp8(x))
        let mut xq = x.clone();
        fake_fp8(&mut xq, 32);
        let cpu: Vec<f32> = (0..n).map(|i| (0..k).map(|c| dsx_core::model::fp8_e4m3_to_f32(w[i * k + c]) * f32::from_bits((sc[(i / 32) * (k / 32) + c / 32] as u32) << 23) * xq[c]).sum()).collect();
        report("fp8 gemv vs cpu", &a, &cpu);
    }
    // ---- MXFP4 expert GEMV (exact table) vs CPU dequant, through the expert table (expert 0)
    {
        let (n, k) = (256usize, 1024usize);
        let w: Vec<u8> = (0..n * k / 2).map(|_| ((r.f() + 1.0) * 127.9) as u8).collect();
        let sc: Vec<u8> = (0..n * k / 32).map(|_| 118 + ((r.f() + 1.0) * 3.0) as u8).collect();
        let (codes, scales) = crate::load::tile_fp4(&w, &sc, n, k);
        let arena = g.alloc(codes.len() + scales.len())?;
        g.htod(arena.ptr, &codes)?;
        g.htod(arena.at(codes.len()), &scales)?;
        let mut em = crate::decode::ExpertMat::default();
        em.woff = 0;
        em.soff = codes.len() as i64;
        em.mul = 1.0;
        em.fmt = crate::decode::E_FP4;
        let tab = g.upload(&[em, em, em])?;
        let x: Vec<f32> = (0..k).map(|_| r.f()).collect();
        let dx = g.upload(&x)?;
        let y = g.alloc_zeroed(n * 4)?;
        crate::decode::test_job(&g, crate::decode::W_EXPERT, arena.ptr, 0, n, k, dx.ptr, y.ptr, 0, 1, tab.ptr)?;
        let got: Vec<f32> = g.dtoh(y.ptr, n)?;
        let mut xq = x.clone();
        fake_fp8(&mut xq, 32);
        let mut row = vec![0f32; k];
        let cpu: Vec<f32> = (0..n)
            .map(|i| {
                dsx_core::formats::dequant_row_scaled(dsx_core::gguf::GType::Fp4, &w[i * k / 2..(i + 1) * k / 2], &sc[i * k / 32..(i + 1) * k / 32], &mut row);
                row.iter().zip(&xq).map(|(a, b)| a * b).sum()
            })
            .collect();
        report("mxfp4 expert gemv vs cpu", &got, &cpu);
        // timing on a full-size expert matrix (2304 x 5120), replicated across 1.5 GB so it streams from HBM
        if std::env::var_os("SELFTEST_TIMING").is_some() {
        let (n, k) = (2304usize, 5120usize);
        let w: Vec<u8> = (0..n * k / 2).map(|_| ((r.f() + 1.0) * 127.9) as u8).collect();
        let sc: Vec<u8> = (0..n * k / 32).map(|_| 118 + ((r.f() + 1.0) * 3.0) as u8).collect();
        let (codes, scales) = crate::load::tile_fp4(&w, &sc, n, k);
        let per = codes.len() + scales.len();
        let copies = 240;
        let arena = g.alloc(per * copies)?;
        for c in 0..copies {
            g.htod(arena.at(c * per), &codes)?;
            g.htod(arena.at(c * per + codes.len()), &scales)?;
        }
        let tabs: Vec<crate::hip::Buf> = (0..copies)
            .map(|c| {
                let mut em = crate::decode::ExpertMat::default();
                em.woff = (c * per) as i64;
                em.soff = (c * per + codes.len()) as i64;
                em.mul = 1.0;
                g.upload(&[em, em, em]).unwrap()
            })
            .collect();
        let x: Vec<f32> = (0..k).map(|_| r.f()).collect();
        let dx = g.upload(&x)?;
        let y = g.alloc_zeroed(n * 4)?;
        for split in [1usize, 2, 4, 8] {
            let us = g.time(|| {
                for c in 0..copies {
                    crate::decode::test_job_nosync(&g, crate::decode::W_EXPERT, arena.ptr, 0, n, k, dx.ptr, y.ptr, 0, split, tabs[c].ptr)?;
                }
                Ok(())
            })? / copies as f32;
            println!("mxfp4 expert 2304x5120 split {split}: {us:.1} us, {:.0} GB/s", per as f32 / us / 1e3);
        }
        }
    }
    // ---- BF16 GEMV with the hi/lo activation split vs CPU f32
    {
        let (n, k) = (512usize, 5120usize);
        let w: Vec<f32> = (0..n * k).map(|_| r.f() * 0.05).collect();
        let wb = bf16r(&w);
        let wbytes: Vec<u8> = wb.iter().flat_map(|v| v.to_le_bytes()).collect();
        let wt = crate::load::tile_bf16(&wbytes, n, k);
        let dw = g.upload(&wt)?;
        let x: Vec<f32> = (0..k).map(|_| r.f() * 3.0).collect();
        let dx = g.upload(&x)?;
        let y = g.alloc_zeroed(n * 4)?;
        crate::decode::test_job(&g, crate::decode::W_BF16, dw.ptr, 0, n, k, dx.ptr, y.ptr, 1, 1, 0)?;
        let got: Vec<f32> = g.dtoh(y.ptr, n)?;
        let wf = from_bf16(&wb);
        let cpu: Vec<f32> = (0..n).map(|i| (0..k).map(|c| wf[i * k + c] * x[c]).sum()).collect();
        report("bf16 gemv (hi/lo) vs cpu", &got, &cpu);
    }
    // ---- hc_step: post (previous sub-block) + pre + next mix dots vs the reference's hc_mixes / hc_post / hc_pre
    {
        let (d, hc) = (5120usize, 4usize);
        let cfg_hc_eps = 1e-6f32;
        let h: Vec<f32> = (0..hc * d).map(|_| half::bf16::from_f32(r.f() * 4.0).to_f32()).collect();
        let y: Vec<f32> = (0..d).map(|_| r.f() * 2.0).collect();
        let fn_prev: Vec<f32> = (0..24 * hc * d).map(|_| r.f() * 0.02).collect();
        let fn_next: Vec<f32> = (0..24 * hc * d).map(|_| r.f() * 0.02).collect();
        let mut pa = crate::decode::HcArgs::default();
        let mut na = crate::decode::HcArgs::default();
        for i in 0..3 { pa.scale[i] = 0.5 + r.f().abs(); na.scale[i] = 0.5 + r.f().abs(); }
        for i in 0..24 { pa.base[i] = r.f(); na.base[i] = r.f(); }
        // raw mixes of the previous sub-block on h: dots + sum of squares
        let mut pm = vec![0f32; 80 * 32]; // per-block partials: put the totals in block 0
        for j in 0..24 { pm[j] = (0..hc * d).map(|i| fn_prev[j * hc * d + i] * h[i]).sum(); }
        pm[24] = h.iter().map(|v| v * v).sum();
        let (dh, dy, dpm, dfn) = (g.upload(&h)?, g.upload(&y)?, g.upload(&pm)?, g.upload(&fn_next)?);
        let nmix = g.alloc_zeroed(80 * 32 * 4)?;
        let xr = g.alloc_zeroed(d * 4)?;
        let xss = g.alloc_zeroed(80 * 4)?;
        crate::decode::test_hc_step(&g, dh.ptr, dy.ptr, dpm.ptr, &pa, dpm.ptr, &pa, dfn.ptr, nmix.ptr, xr.ptr, xss.ptr)?;
        let gh: Vec<f32> = g.dtoh(dh.ptr, hc * d)?;
        let gx: Vec<f32> = g.dtoh(xr.ptr, d)?;
        let gp: Vec<f32> = g.dtoh(nmix.ptr, 80 * 32)?;
        let gm: Vec<f32> = (0..25).map(|j| (0..80).map(|b| gp[b * 32 + j]).sum()).collect();
        // CPU: model.rs hc_mixes(h, fn_prev) -> (pre, post, comb); h' = hc_post(bf16(y), h, post, comb); x = hc_pre(h', pre)
        let c = dsx_core::model::Config { hc, dim: d, eps: 1e-20, hc_eps: cfg_hc_eps, sinkhorn_iters: 20, ..fake_cfg() };
        let (pre, post, comb) = dsx_core::model::hc_mixes_pub(&c, &h, &fn_prev, &pa.scale, &pa.base);
        let yb: Vec<f32> = y.iter().map(|&v| round_bf16(v)).collect();
        let h2 = dsx_core::model::hc_post_pub(&c, &yb, &h, &post, &comb);
        let x = dsx_core::model::hc_pre_pub(&c, &h2, &pre);
        let us = g.time(|| {
            for _ in 0..100 {
                crate::decode::test_hc_step_nosync(&g, dh.ptr, dy.ptr, dpm.ptr, &pa, dpm.ptr, &pa, dfn.ptr, nmix.ptr, xr.ptr, xss.ptr)?;
            }
            Ok(())
        })?;
        println!("hc_step: {:.2} us per launch (stream)", us / 100.0);
        report("hc_step h (post)", &gh, &h2);
        report("hc_step x (pre)", &gx, &x);
        let mut nm = vec![0f32; 25];
        for j in 0..24 { nm[j] = (0..hc * d).map(|i| fn_next[j * hc * d + i] * h2[i]).sum(); }
        nm[24] = h2.iter().map(|v| v * v).sum();
        report("hc_step next mix dots", &gm, &nm);
    }
    // ---- prompt-path mm: FP8 dense, MXFP4, BF16 over 37 tokens vs CPU
    {
        let (n, k, nt) = if std::env::var_os("MM_BIG").is_some() { (5120usize, 8192usize, 3usize) } else { (128usize, 512usize, 37usize) };
        let codes: Vec<u8> = (0..nt * k).map(|i| [0x38u8, 0x40, 0xC4, 0x30, 0xB8, 0x48, 0x00, 0xBC][(i * 7 + i / 5) % 8]).collect();
        let xsv: Vec<f32> = (0..nt * k / 32).map(|i| [1.0f32, 0.5, 2.0, 0.25][i % 4]).collect();
        let xv: Vec<f32> = (0..nt * k).map(|i| fnuz(codes[i]) * xsv[i / 32]).collect();
        let (dq, dxs) = (g.upload(&codes)?, g.upload(&xsv)?);
        let y = g.alloc_zeroed(nt * n * 4)?;
        // FP8 dense
        let w: Vec<u8> = (0..n * k).map(|_| { let b = ((r.f() + 1.0) * 127.0) as u8; if b & 0x7f == 0x7f { 0x30 } else { b } }).collect();
        let sc: Vec<u8> = (0..(n / 32) * (k / 32)).map(|i| 118 + (i % 5) as u8).collect();
        let (dw, dsc) = (g.upload(&crate::load::tile_fp8(&w, n, k))?, g.upload(&sc)?);
        crate::prefill::test_mm(&g, 0, dw.ptr, dsc.ptr, 0, 0, n, k, dq.ptr, dxs.ptr, 0, nt, y.ptr, (0, 0, 0))?;
        let got: Vec<f32> = g.dtoh(y.ptr, nt * n)?;
        let cpu: Vec<f32> = (0..nt * n).map(|o| { let (t, i) = (o / n, o % n); (0..k).map(|c| dsx_core::model::fp8_e4m3_to_f32(w[i * k + c]) * f32::from_bits((sc[(i / 32) * (k / 32) + c / 32] as u32) << 23) * xv[t * k + c]).sum() }).collect();
        report("mm fp8 dense (37 tok)", &got, &cpu);
        let xbf = g.upload(&bf16r(&xv))?;
        crate::prefill::test_mm(&g, 0, dw.ptr, dsc.ptr, 0, 0, n, k, dq.ptr, dxs.ptr, xbf.ptr, nt, y.ptr, (0, 0, 0))?;
        let got2: Vec<f32> = g.dtoh(y.ptr, nt * n)?;
        report("mm2 fp8 dense (37 tok)", &got2, &cpu);
        // vgemv over the first 5 token rows (split 2: atomics)
        g.memset(y.ptr, 0, nt * n * 4)?;
        crate::verify::test_vgemv(&g, 0, dw.ptr, dsc.ptr, 0, n, k, dq.ptr, dxs.ptr, 0, 5, y.ptr, 2)?;
        let gv: Vec<f32> = g.dtoh(y.ptr, 5 * n)?;
        report("vgemv fp8 (5 tok)", &gv, &cpu[..5 * n]);
        // MXFP4
        let w4: Vec<u8> = (0..n * k / 2).map(|_| ((r.f() + 1.0) * 127.9) as u8).collect();
        let s4: Vec<u8> = (0..n * k / 32).map(|_| 118 + ((r.f() + 1.0) * 3.0) as u8).collect();
        let (c4, t4) = crate::load::tile_fp4(&w4, &s4, n, k);
        let ar = g.alloc(c4.len() + t4.len())?;
        g.htod(ar.ptr, &c4)?;
        g.htod(ar.at(c4.len()), &t4)?;
        crate::prefill::test_mm(&g, 1, ar.ptr, 0, 0, c4.len() as i64, n, k, dq.ptr, dxs.ptr, 0, nt, y.ptr, (0, 0, 0))?;
        let got: Vec<f32> = g.dtoh(y.ptr, nt * n)?;
        let mut row = vec![0f32; k];
        let wd: Vec<Vec<f32>> = (0..n).map(|i| { dsx_core::formats::dequant_row_scaled(dsx_core::gguf::GType::Fp4, &w4[i * k / 2..(i + 1) * k / 2], &s4[i * k / 32..(i + 1) * k / 32], &mut row); row.clone() }).collect();
        let cpu: Vec<f32> = (0..nt * n).map(|o| { let (t, i) = (o / n, o % n); (0..k).map(|c| wd[i][c] * xv[t * k + c]).sum() }).collect();
        report("mm mxfp4 (37 tok)", &got, &cpu);
        crate::prefill::test_mm(&g, 1, ar.ptr, 0, 0, c4.len() as i64, n, k, dq.ptr, dxs.ptr, xbf.ptr, nt, y.ptr, (0, 0, 0))?;
        let got2: Vec<f32> = g.dtoh(y.ptr, nt * n)?;
        report("mm2 mxfp4 (37 tok)", &got2, &cpu);
        g.memset(y.ptr, 0, nt * n * 4)?;
        crate::verify::test_vgemv(&g, 1, ar.ptr, 0, c4.len() as i64, n, k, dq.ptr, dxs.ptr, 0, 5, y.ptr, 1)?;
        let gv: Vec<f32> = g.dtoh(y.ptr, 5 * n)?;
        report("vgemv mxfp4 (5 tok)", &gv, &cpu[..5 * n]);
        // BF16 weights x bf16 activations (xv is bf16-exact)
        let wb: Vec<f32> = (0..n * k).map(|_| r.f() * 0.1).collect();
        let wbb = bf16r(&wb);
        let wbytes: Vec<u8> = wbb.iter().flat_map(|v| v.to_le_bytes()).collect();
        let dwb = g.upload(&crate::load::tile_bf16(&wbytes, n, k))?;
        let xb = g.upload(&bf16r(&xv))?;
        crate::prefill::test_mm(&g, 4, dwb.ptr, 0, 0, 0, n, k, 0, 0, xb.ptr, nt, y.ptr, (0, 0, 0))?;
        let got: Vec<f32> = g.dtoh(y.ptr, nt * n)?;
        let wf = from_bf16(&wbb);
        let cpu: Vec<f32> = (0..nt * n).map(|o| { let (t, i) = (o / n, o % n); (0..k).map(|c| wf[i * k + c] * xv[t * k + c]).sum() }).collect();
        report("mm bf16 (37 tok)", &got, &cpu);
        unsafe { std::env::set_var("TEST_MM2", "1") };
        crate::prefill::test_mm(&g, 4, dwb.ptr, 0, 0, 0, n, k, 0, 0, xb.ptr, nt, y.ptr, (0, 0, 0))?;
        unsafe { std::env::remove_var("TEST_MM2") };
        let got2: Vec<f32> = g.dtoh(y.ptr, nt * n)?;
        report("mm2 bf16 (37 tok)", &got2, &cpu);
    }
    // ---- router top-6 vs the reference's sort
    {
        let mut bad = 0;
        for trial in 0..200 {
            let lg: Vec<f32> = (0..384).map(|_| r.f() * 4.0).collect();
            let bias: Vec<f32> = (0..384).map(|_| r.f() * 0.5).collect();
            let (dl, db) = (g.upload(&lg)?, g.upload(&bias)?);
            let out = g.alloc_zeroed(64)?;
            g.launch("test_route", (1, 1, 1), 64, 0, &[Arg::P(dl.ptr), Arg::P(db.ptr), Arg::F(1.5), Arg::P(out.ptr)])?;
            let ids: Vec<i32> = g.dtoh(out.ptr, 8)?;
            let sc: Vec<f32> = lg.iter().map(|&l| softplus(l).sqrt()).collect();
            let mut order: Vec<usize> = (0..384).collect();
            order.sort_by(|&a, &b| (sc[b] + bias[b]).partial_cmp(&(sc[a] + bias[a])).unwrap());
            let mut want: Vec<i32> = order[..6].iter().map(|&e| e as i32).collect();
            let mut got = ids[..6].to_vec();
            want.sort();
            got.sort();
            if want != got {
                bad += 1;
                if bad < 3 {
                    println!("route trial {trial}: gpu {got:?} cpu {want:?}");
                }
            }
        }
        println!("router top-6: {bad}/200 trials differ");
    }
    Ok(())
}

fn fake_cfg() -> dsx_core::model::Config {
    dsx_core::model::Config {
        vocab: 0, dim: 0, inter: 0, n_layers: 0, n_heads: 0, head_dim: 0, rope_dim: 0, q_lora: 0, o_lora: 0, o_groups: 0,
        n_experts: 0, top_k: 0, window: 0, idx_heads: 0, idx_dim: 0, idx_topk: 0, cand_layer: 0, cand_blocks: 0,
        cand_block: 0, hc: 0, sinkhorn_iters: 0, hc_eps: 0.0, eps: 0.0, swiglu_limit: 0.0, route_scale: 0.0,
        rope_theta: 0.0, compress_rope_theta: 0.0, rope_factor: 0.0, beta_fast: 0.0, beta_slow: 0.0,
        original_seq_len: 0, compress_ratios: vec![], kv_sources: vec![], index_sources: vec![], engram_layers: vec![],
        engram_rows: vec![], engram_heads: 0, engram_dim: 0, engram_ngram: 0, engram_pad: 0, engram_token_map: vec![],
        engram_primes: vec![], engram_multipliers: vec![], dspark_targets: vec![], expert_rot: vec![],
    }
}

fn fnuz(b: u8) -> f32 {
    if b == 0x80 {
        return f32::NAN;
    }
    let s = b >> 7;
    let e = ((b >> 3) & 15) as i32;
    let m = (b & 7) as f32;
    let x = if e == 0 { m / 8.0 * 2f32.powi(-7) } else { (1.0 + m / 8.0) * 2f32.powi(e - 8) };
    if s == 1 { -x } else { x }
}
