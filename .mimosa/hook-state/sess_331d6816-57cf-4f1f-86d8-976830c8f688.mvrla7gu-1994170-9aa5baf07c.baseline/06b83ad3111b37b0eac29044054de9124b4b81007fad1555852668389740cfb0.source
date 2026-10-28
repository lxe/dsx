#![cfg(feature = "cuda")]
//! Expert-format GEMV speed as decode runs them, old GGUF-block kernels vs the repacked fast kernels
//! (cargo test --release --features cuda --test gpu_expbench -- --ignored --nocapture).
use dsx::experts::{KS, ExpCfg, GpuExperts, RawExpert, repack, repacked_id};
use dsx::gguf::GType;
use dsx::gpu::*;
use std::time::Instant;

fn random_bytes(n: usize, seed: &mut u64) -> Vec<u8> {
    (0..n).map(|_| { *seed ^= *seed << 13; *seed ^= *seed >> 7; *seed ^= *seed << 17; *seed as u8 }).collect()
}

/// sane fp16 scales (random bytes there make inf/NaN, which costs nothing but reads badly)
fn fix_scales(ty: GType, w: &mut [u8]) {
    let (_, b) = ty.block();
    let at = match ty { GType::Ptq1_0 => 26, GType::Q3K => 108, _ => 0 }; // Q2_0, Q8_0, IQ2_XXS: d first
    let h = half::f16::from_f32(0.01).to_le_bytes();
    for blk in w.chunks_exact_mut(b) {
        blk[at..at + 2].copy_from_slice(&h);
    }
}

fn time(g: &Gpu, n: usize, mut f: impl FnMut()) -> f64 {
    f();
    g.sync().unwrap();
    let t = Instant::now();
    for _ in 0..n {
        f();
    }
    g.sync().unwrap();
    t.elapsed().as_secs_f64() / n as f64
}

const FORMATS: [(GType, Option<&str>); 7] =
    [(GType::Q2_0, Some("gemv_q2_0")), (GType::Ptq1_0, Some("gemv_ptq1_0")), (GType::Q3K, Some("gemv_q3_k")), (GType::Iq2Xxs, None), (GType::Iq1S, None), (GType::Iq2Xs, None), (GType::Iq2S, None)];

#[test]
#[ignore]
fn expert_gemv_speed() {
    let g = Gpu::new(0).unwrap();
    let mut seed = 1u64;
    eprintln!("{} ({} SMs): GB/s of weight bytes, old GGUF-block kernel vs repacked fast kernel (int8 x, dp4a)", g.name, g.sms);
    // six experts' gate rows back to back; six experts' down rows
    for (rows, k, what) in [(2304usize * 6, 5120usize, "gate x6"), (5120 * 6, 2304, "down x6")] {
        for (ty, old) in FORMATS {
            // DSX_BENCH_FMT=Iq2Xs, DSX_BENCH_SHAPE=gate: one format / shape (e.g. to profile it alone under ncu)
            if std::env::var("DSX_BENCH_FMT").is_ok_and(|f| f != format!("{ty:?}")) || std::env::var("DSX_BENCH_SHAPE").is_ok_and(|w| !what.starts_with(w.as_str())) {
                continue;
            }
            let bytes = ty.row_bytes(k) * rows;
            let mut w = random_bytes(bytes, &mut seed);
            fix_scales(ty, &mut w);
            let x: Vec<f32> = (0..8 * k).map(|i| (i % 100) as f32 * 0.01 - 0.5).collect();
            // copies to cycle through: >= 256 MB, so no run finds its weights in L2 (6 MB on a 3090, 96 MB on a 5090)
            let nc = (256e6 / bytes as f64).ceil() as usize;
            let (dx, dy) = (g.upload(&x).unwrap(), g.alloc(8 * rows * 4).unwrap());
            let dw: Vec<Buf> = (0..nc).map(|_| g.upload(&w).unwrap()).collect();
            let dr: Vec<Buf> = dw.iter().map(|s| {
                let d = g.alloc(bytes).unwrap();
                repack(&g, ty, s.ptr, d.ptr, rows, k).unwrap();
                d
            }).collect();
            let mut it = 0usize;
            let (xq, xd, xs) = (g.alloc(8 * k).unwrap(), g.alloc(8 * k / 64).unwrap(), g.alloc(8 * k / 32).unwrap());
            let (ko, kq, kf) = (old.map(|o| g.kernel(o).unwrap()), g.kernel("quant_act").unwrap(), g.kernel("xgemv").unwrap());
            let mut line = format!("{what:8} {:7} {:5.1} MB:", format!("{ty:?}"), bytes as f64 / 1e6);
            for t in [1usize, 6] {
                // IQ2_XXS has no old GGUF-block kernel
                let old_s = ko.map_or(f64::NAN, |ko| time(&g, 20, || {
                    it += 1;
                    g.launch(ko, ((rows as u32).div_ceil(4), 1, 1), 128, 0, &[Arg::P(dw[it % nc].ptr), Arg::P(dx.ptr), Arg::P(dy.ptr), Arg::I(rows as i32), Arg::I(k as i32), Arg::I(t as i32), Arg::I(k as i32), Arg::I(rows as i32)]).unwrap()
                }));
                let q_s = time(&g, 20, || {
                    g.launch(kq, ((k / 256) as u32, t as u32, 1), 256, 0, &[Arg::P(dx.ptr), Arg::I(k as i32), Arg::P(xq.ptr), Arg::P(xd.ptr), Arg::P(xs.ptr), Arg::P(0), Arg::P(0), Arg::I(0), Arg::I(0), Arg::I(0), Arg::I(0), Arg::P(0), Arg::I(0)]).unwrap()
                });
                let new_s = time(&g, 20, || {
                    it += 1;
                    g.launch(kf, ((rows as u32).div_ceil(16 * dsx::experts::ROWT), 1, 1), 32 * KS, dsx::experts::tile_smem(ty), &[Arg::I(repacked_id(ty).unwrap()), Arg::P(dr[it % nc].ptr), Arg::I(rows as i32), Arg::I(k as i32), Arg::I(t as i32), Arg::P(xq.ptr), Arg::P(xd.ptr), Arg::P(xs.ptr), Arg::P(dy.ptr)]).unwrap()
                });
                let gbs = |s: f64| bytes as f64 / s / 1e9;
                line += &format!("  T={t}: old {:4.0} us {:4.0} GB/s, new {:4.0} us {:4.0} GB/s (+quant {:.0} us)", old_s * 1e6, gbs(old_s), new_s * 1e6, gbs(new_s), q_s * 1e6);
            }
            eprintln!("{line}");
        }
    }
}

/// A layer's six resident experts end to end: old moe_gu + moe_down (one call per token) vs run_device_fast
/// (all T tokens in one call; here every token picks the same six, so each expert is read once).
#[test]
#[ignore]
fn expert_moe_speed() {
    let g = Gpu::new(0).unwrap();
    let mut seed = 7u64;
    // the same six experts in enough layers (>= 256 MB) that cycling through them defeats L2
    let nl = 6;
    let c = ExpCfg { n_layers: nl, n_experts: 8, inter: 2304, dim: 5120, lim: 10.0, rot: 0 };
    eprintln!("{}: 6 resident experts (gate+up+down), us per layer and GB/s of expert bytes read", g.name);
    for (gu, dn) in [(GType::Q2_0, GType::Q2_0), (GType::Ptq1_0, GType::Ptq1_0), (GType::Q3K, GType::Q3K), (GType::Iq2Xxs, GType::Iq2Xxs), (GType::Iq1S, GType::Iq1S), (GType::Iq2Xs, GType::Iq2Xs), (GType::Iq2S, GType::Iq2S)] {
        if std::env::var("DSX_BENCH_FMT").is_ok_and(|f| f != format!("{gu:?}")) {
            continue;
        }
        let mats: Vec<[Vec<u8>; 3]> = (0..6)
            .map(|_| {
                let mut m = [random_bytes(gu.row_bytes(c.dim) * c.inter, &mut seed), random_bytes(gu.row_bytes(c.dim) * c.inter, &mut seed), random_bytes(dn.row_bytes(c.inter) * c.dim, &mut seed)];
                fix_scales(gu, &mut m[0]);
                fix_scales(gu, &mut m[1]);
                fix_scales(dn, &mut m[2]);
                m
            })
            .collect();
        let items: Vec<((usize, usize), RawExpert)> =
            (0..nl).flat_map(|l| mats.iter().enumerate().map(move |(e, m)| ((l, e), RawExpert { gate: &m[0], up: &m[1], down: &m[2], gu, dn }))).collect();
        let bytes: usize = mats.iter().map(|m| m.iter().map(|v| v.len()).sum::<usize>()).sum();
        let old = GpuExperts::build(&g, c, &items, false).unwrap();
        let new = GpuExperts::build(&g, c, &items, true).unwrap();
        let x: Vec<f32> = (0..8 * c.dim).map(|i| (i % 100) as f32 * 0.01 - 0.5).collect();
        let sel: Vec<i32> = (0..8).flat_map(|_| 0..6).collect();
        let wts = vec![0.1f32; 48];
        for s in [&old, &new] {
            g.htod_sync(s.sel.ptr, &sel).unwrap();
            g.htod_sync(s.wts.ptr, &wts).unwrap();
            g.htod_sync(s.x.ptr, &x).unwrap();
        }
        let mut line = format!("{gu:?}/{dn:?} {:.0} MB:", bytes as f64 / 1e6);
        for t in [1usize, 6] {
            let mut l = 0;
            let old_s = time(&g, 24, || {
                l = (l + 1) % nl;
                for _ in 0..t {
                    old.run_device(&g, l, old.sel.ptr, old.wts.ptr, 6, old.x.ptr, old.out.ptr).unwrap()
                }
            });
            let new_s = time(&g, 24, || {
                l = (l + 1) % nl;
                new.run_device_fast(&g, l, new.sel.ptr, new.wts.ptr, 6, t, new.x.ptr, new.out.ptr).unwrap()
            });
            let gbs = |s: f64| bytes as f64 / s / 1e9;
            // old reads the experts once per token
            line += &format!("  T={t}: old {:5.0} us ({:4.0} GB/s), new {:4.0} us ({:4.0} GB/s)", old_s * 1e6, gbs(old_s) * t as f64, new_s * 1e6, gbs(new_s));
        }
        eprintln!("{line}");
    }
}
