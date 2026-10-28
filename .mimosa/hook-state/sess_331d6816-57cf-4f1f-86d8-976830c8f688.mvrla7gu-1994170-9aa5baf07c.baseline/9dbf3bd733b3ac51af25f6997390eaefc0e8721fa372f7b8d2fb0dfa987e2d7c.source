#![cfg(feature = "cuda")]
//! 6-bit dense planes: GPU GEMV vs the CPU decoder, and speed next to the repacked Q8 GEMV.
use dsx::gpu::*;
use std::time::Instant;

#[test]
fn q6_gemv() {
    let g = Gpu::new(0).unwrap();
    for (rows, k) in [(256usize, 1280usize), (32768, 1280), (5120, 8192)] {
        let x: Vec<f32> = (0..rows * k).map(|i| ((i * 2654435761usize) % 2001) as f32 / 1000.0 - 1.0).collect();
        let (mut lo, mut hi, mut sc, mut d) = (vec![0u8; rows * k / 2], vec![0u8; rows * k / 4], vec![0i8; rows * k / 16], vec![0u16; rows * k / 256]);
        for r in 0..rows {
            dsx::q6::quantize_row(&x[r * k..(r + 1) * k], &mut lo[r * k / 2..(r + 1) * k / 2], &mut hi[r * k / 4..(r + 1) * k / 4], &mut sc[r * k / 16..(r + 1) * k / 16], &mut d[r * k / 256..(r + 1) * k / 256]);
        }
        let v: Vec<f32> = (0..k).map(|i| (i % 13) as f32 * 0.1 - 0.6).collect();
        let (dl, dh, ds, dd, dv, dy) = (g.upload(&lo).unwrap(), g.upload(&hi).unwrap(), g.upload(&sc).unwrap(), g.upload(&d).unwrap(), g.upload(&v).unwrap(), g.alloc(rows * 4).unwrap());
        for (name, rpb) in [("gemv_q6p", 4u32), ("gemv_q6g4", 16), ("gemv_q6g8", 32)] {
        let kk = g.kernel(name).unwrap();
        let args = [Arg::P(dl.ptr), Arg::P(dv.ptr), Arg::P(dy.ptr), Arg::I(rows as i32), Arg::I(k as i32), Arg::I(1), Arg::I(k as i32), Arg::I(rows as i32), Arg::P(dh.ptr), Arg::P(ds.ptr), Arg::P(dd.ptr), Arg::I(rows as i32), Arg::I(0)];
        g.launch(kk, ((rows as u32).div_ceil(rpb), 1, 1), 128, 0, &args).unwrap();
        let got: Vec<f32> = g.download(dy.ptr, rows).unwrap();
        for r in (0..rows).step_by((rows / 8).max(1)) {
            let want: f64 = (0..k).map(|c| dsx::q6::dequant(&lo, &hi, &sc, &d, r * k + c) as f64 * v[c] as f64).sum();
            assert!((got[r] as f64 - want).abs() < 1e-3 * want.abs().max(1.0), "row {r}: {} vs {want}", got[r]);
        }
        let t = Instant::now();
        for _ in 0..50 { g.launch(kk, ((rows as u32).div_ceil(rpb), 1, 1), 128, 0, &args).unwrap(); }
        g.sync().unwrap();
        let us = t.elapsed().as_secs_f64() / 50.0 * 1e6;
        eprintln!("{name} {rows}x{k}: {us:.1} us, {:.0} GB/s ({:.0} G weights/s)", rows as f64 * k as f64 * 0.8203 / us / 1e3, (rows * k) as f64 / us / 1e3);
        }
    }
}

#[test]
fn q5k_gemv_and_dequant() {
    // random llama.cpp Q5_K rows repacked to 5-bit planes: GEMV (T = 3) and the f16 dequant vs the CPU decoder
    let g = Gpu::new(0).unwrap();
    for (rows, k) in [(2304usize, 4096usize), (4096, 2304)] {
        let mut seed = 0x9e3779b97f4a7c15u64;
        let mut raw: Vec<u8> = (0..rows * k / 256 * 176)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                seed as u8
            })
            .collect();
        for b in 0..rows * k / 256 {
            raw[b * 176..b * 176 + 2].copy_from_slice(&half::f16::from_f32(0.001).to_le_bytes());
            raw[b * 176 + 2..b * 176 + 4].copy_from_slice(&half::f16::from_f32(0.0004).to_le_bytes());
        }
        let (mut lo, mut hb, mut sm, mut dd) = (vec![0u8; rows * k / 2], vec![0u8; rows * k / 8], vec![0u8; rows * k / 16], vec![0u16; rows * k / 128]);
        let rb = k / 256 * 176;
        for r in 0..rows {
            dsx::q6::from_q5k_row(&raw[r * rb..(r + 1) * rb], &mut lo[r * k / 2..(r + 1) * k / 2], &mut hb[r * k / 8..(r + 1) * k / 8], &mut sm[r * k / 16..(r + 1) * k / 16], &mut dd[r * k / 128..(r + 1) * k / 128]);
        }
        let t = 3;
        let v: Vec<f32> = (0..t * k).map(|i| (i % 13) as f32 * 0.1 - 0.6 + (i / k) as f32 * 0.05).collect();
        let (dl, dh, ds, ddd, dv, dy, dw) = (g.upload(&lo).unwrap(), g.upload(&hb).unwrap(), g.upload(&sm).unwrap(), g.upload(&dd).unwrap(), g.upload(&v).unwrap(), g.alloc(t * rows * 4).unwrap(), g.alloc(rows * k * 2).unwrap());
        for (name, rpb) in [("gemv_q5p", 4u32), ("gemv_q5p4", 16)] {
        let kk = g.kernel(name).unwrap();
        let args = [Arg::P(dl.ptr), Arg::P(dv.ptr), Arg::P(dy.ptr), Arg::I(rows as i32), Arg::I(k as i32), Arg::I(t as i32), Arg::I(k as i32), Arg::I(rows as i32), Arg::P(dh.ptr), Arg::P(ds.ptr), Arg::P(ddd.ptr)];
        g.launch(kk, ((rows as u32).div_ceil(rpb), 1, 1), 128, 0, &args).unwrap();
        let got: Vec<f32> = g.download(dy.ptr, t * rows).unwrap();
        for tt in 0..t {
            for r in (0..rows).step_by(rows / 8) {
                let want: f64 = (0..k).map(|c| dsx::q6::dequant5(&lo, &hb, &sm, &dd, r * k + c) as f64 * v[tt * k + c] as f64).sum();
                let gr = got[tt * rows + r] as f64;
                assert!((gr - want).abs() < 1e-3 * want.abs().max(1.0), "{name} t {tt} row {r}: {gr} vs {want}");
            }
        }
        let args1 = [Arg::P(dl.ptr), Arg::P(dv.ptr), Arg::P(dy.ptr), Arg::I(rows as i32), Arg::I(k as i32), Arg::I(1), Arg::I(k as i32), Arg::I(rows as i32), Arg::P(dh.ptr), Arg::P(ds.ptr), Arg::P(ddd.ptr)];
        let tm = Instant::now();
        for _ in 0..50 { g.launch(kk, ((rows as u32).div_ceil(rpb), 1, 1), 128, 0, &args1).unwrap(); }
        g.sync().unwrap();
        let us = tm.elapsed().as_secs_f64() / 50.0 * 1e6;
        eprintln!("{name} {rows}x{k}: {us:.1} us, {:.0} GB/s", rows as f64 * k as f64 * 0.703 / us / 1e3);
        }
        g.launch(g.kernel("dequant_q5p_f16").unwrap(), ((k as u32).div_ceil(256), rows as u32, 1), 256, 0, &[Arg::P(dl.ptr), Arg::P(dh.ptr), Arg::P(ds.ptr), Arg::P(ddd.ptr), Arg::P(dw.ptr), Arg::I(rows as i32), Arg::I(k as i32)]).unwrap();
        let w16: Vec<half::f16> = g.download(dw.ptr, rows * k).unwrap();
        for i in (0..rows * k).step_by(997) {
            assert_eq!(w16[i], half::f16::from_f32(dsx::q6::dequant5(&lo, &hb, &sm, &dd, i)), "weight {i}");
        }
    }
}

/// The as-is UD file's Q6_K dense shapes on this GPU: gemv_q6p (1 row per warp; o_a as 8 group launches) vs the grouped
/// 4 / 8-row gemv_q6g, next to the repacked Q8 GEMV on the same shape (the speed the Q8 dense builds get).
/// cargo test --release --features cuda --test gpu_q6 q6_model_shapes -- --ignored --nocapture
#[test]
#[ignore]
fn q6_model_shapes() {
    let g = Gpu::new(0).unwrap();
    eprintln!("{}: us per call (GB/s of weight bytes), T=1", g.name);
    // (name, groups, rows per group, k)
    for (what, ng, rpg, k) in [("kv", 1usize, 512usize, 5120usize), ("q_a", 1, 1280, 5120), ("q_b", 1, 32768, 1280), ("o_a", 8, 1024, 4096), ("o_b", 1, 5120, 8192), ("head", 1, 129280, 5120)] {
        let rows = ng * rpg;
        let mut seed = 0x2545f4914f6cdd1du64;
        let mut rnd = || { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; seed };
        let lo: Vec<u8> = (0..rows * k / 2).map(|_| rnd() as u8).collect();
        let hi: Vec<u8> = (0..rows * k / 4).map(|_| rnd() as u8).collect();
        let sc: Vec<i8> = (0..rows * k / 16).map(|_| (rnd() % 64) as i8 - 32).collect();
        let d: Vec<u16> = (0..rows * k / 256).map(|_| half::f16::from_f32(0.001).to_bits()).collect();
        let v: Vec<f32> = (0..ng * k).map(|i| (i % 13) as f32 * 0.1 - 0.6).collect();
        let (dl, dh, ds, dd, dv, dy) = (g.upload(&lo).unwrap(), g.upload(&hi).unwrap(), g.upload(&sc).unwrap(), g.upload(&d).unwrap(), g.upload(&v).unwrap(), g.alloc(rows * 4).unwrap());
        // correctness of the grouped kernels on a few rows
        let want = |r: usize| -> f64 { let gi = r / rpg; (0..k).map(|c| dsx::q6::dequant(&lo, &hi, &sc, &d, r * k + c) as f64 * v[gi * k + c] as f64).sum() };
        let bytes = rows as f64 * k as f64 * 0.8203;
        let mut line = format!("{what:5} {ng}x{rpg}x{k}:");
        let time = |f: &dyn Fn()| { f(); g.sync().unwrap(); let t = Instant::now(); for _ in 0..30 { f(); } g.sync().unwrap(); t.elapsed().as_secs_f64() / 30.0 * 1e6 };
        let k1 = g.kernel("gemv_q6p").unwrap();
        let us = time(&|| for gi in 0..ng {
            g.launch(k1, ((rpg as u32).div_ceil(4), 1, 1), 128, 0, &[Arg::P(dl.ptr + (gi * rpg * k / 2) as u64), Arg::P(dv.ptr + (gi * k * 4) as u64), Arg::P(dy.ptr + (gi * rpg * 4) as u64), Arg::I(rpg as i32), Arg::I(k as i32), Arg::I(1), Arg::I(k as i32), Arg::I(rpg as i32), Arg::P(dh.ptr + (gi * rpg * k / 4) as u64), Arg::P(ds.ptr + (gi * rpg * k / 16) as u64), Arg::P(dd.ptr + (gi * rpg * k / 256 * 2) as u64)]).unwrap();
        });
        line += &format!("  q6p {us:6.1} ({:4.0})", bytes / us / 1e3);
        for (name, prw) in [("gemv_q6g4", 4u32), ("gemv_q6g8", 8), ("gemv_q6s8", 16)] {
            let kk = g.kernel(name).unwrap();
            let args = [Arg::P(dl.ptr), Arg::P(dv.ptr), Arg::P(dy.ptr), Arg::I(rows as i32), Arg::I(k as i32), Arg::I(1), Arg::I(k as i32), Arg::I(rows as i32), Arg::P(dh.ptr), Arg::P(ds.ptr), Arg::P(dd.ptr), Arg::I(rpg as i32), Arg::I(k as i32)];
            g.launch(kk, ((rows as u32).div_ceil(4 * prw), 1, 1), 128, 0, &args).unwrap();
            let got: Vec<f32> = g.download(dy.ptr, rows).unwrap();
            for r in [0, rpg - 1, rows / 2 + 3, rows - 1] {
                let w = want(r);
                assert!((got[r] as f64 - w).abs() < 1e-3 * w.abs().max(1.0), "{name} {what} row {r}: {} vs {w}", got[r]);
            }
            let us = time(&|| g.launch(kk, ((rows as u32).div_ceil(4 * prw), 1, 1), 128, 0, &args).unwrap());
            line += &format!("  {} {us:6.1} ({:4.0})", &name[5..], bytes / us / 1e3);
        }
        // Q8 (repacked: int8 [rows][k], f16 scale per 32) on the same shape, ungrouped
        let q8: Vec<i8> = (0..rows * k).map(|_| rnd() as i8).collect();
        let s8: Vec<u16> = (0..rows * k / 32).map(|_| half::f16::from_f32(0.001).to_bits()).collect();
        let (dq, dsc) = (g.upload(&q8).unwrap(), g.upload(&s8).unwrap());
        let kq = g.kernel("gemv_q8r").unwrap();
        let vv: Vec<f32> = (0..k).map(|i| (i % 7) as f32 * 0.1).collect();
        let dvv = g.upload(&vv).unwrap();
        let us = time(&|| g.launch(kq, ((rows as u32).div_ceil(4), 1, 1), 128, 0, &[Arg::P(dq.ptr), Arg::P(dvv.ptr), Arg::P(dy.ptr), Arg::I(rows as i32), Arg::I(k as i32), Arg::I(1), Arg::I(k as i32), Arg::I(rows as i32), Arg::P(dsc.ptr)]).unwrap());
        line += &format!("  | q8r {us:6.1} ({:4.0} GB/s of its {:.0} MB)", rows as f64 * k as f64 * 1.0625 / us / 1e3, rows as f64 * k as f64 * 1.0625 / 1e6);
        eprintln!("{line}");
    }
}
