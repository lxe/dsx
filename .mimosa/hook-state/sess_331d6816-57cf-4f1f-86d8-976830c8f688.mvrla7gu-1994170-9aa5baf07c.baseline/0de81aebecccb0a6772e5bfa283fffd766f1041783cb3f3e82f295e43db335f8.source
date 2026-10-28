#![cfg(feature = "cuda")]
//! Repacked Q8_0 GEMV variants: same results, and their speed on a big matrix (cargo test -- --ignored --nocapture).
use dsx::gpu::*;
use std::time::Instant;

#[test]
#[ignore]
fn q8_variants() {
    let g = Gpu::new(0).unwrap();
    for (rows, k) in [(32768usize, 1280usize), (5120, 8192), (8192, 4096), (129280, 5120), (512, 5120)] {
        let mut seed = 12345u64;
        let mut rnd = || { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; seed };
        let qs: Vec<i8> = (0..rows * k).map(|_| rnd() as i8).collect();
        let d: Vec<half::f16> = (0..rows * k / 32).map(|_| half::f16::from_f32(0.01)).collect();
        let x: Vec<f32> = (0..k).map(|i| ((i * 7919) % 1000) as f32 / 1000.0 - 0.5).collect();
        let (dq, dd, dx) = (g.upload(&qs).unwrap(), g.upload(&d).unwrap(), g.upload(&x).unwrap());
        let mut outs = vec![];
        for (name, per) in [("gemv_q8r", 4u32), ("gemv_q8u", 4), ("gemv_q8r4", 16)] {
            let y = g.alloc(rows * 4).unwrap();
            let kk = g.kernel(name).unwrap();
            let args = [Arg::P(dq.ptr), Arg::P(dx.ptr), Arg::P(y.ptr), Arg::I(rows as i32), Arg::I(k as i32), Arg::I(1), Arg::I(k as i32), Arg::I(rows as i32), Arg::P(dd.ptr)];
            g.launch(kk, ((rows as u32).div_ceil(per), 1, 1), 128, 0, &args).unwrap();
            g.sync().unwrap();
            let t = Instant::now();
            for _ in 0..20 {
                g.launch(kk, ((rows as u32).div_ceil(per), 1, 1), 128, 0, &args).unwrap();
            }
            g.sync().unwrap();
            let s = t.elapsed().as_secs_f64() / 20.0;
            eprintln!("{rows}x{k} {name}: {:.1} us, {:.0} GB/s", s * 1e6, (rows * k) as f64 * 1.0625 / s / 1e9);
            outs.push(g.download::<f32>(y.ptr, rows).unwrap());
        }
        for o in &outs[1..] {
            let md = o.iter().zip(&outs[0]).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            assert!(md < 1e-2, "variant differs by {md}");
        }
    }
}
