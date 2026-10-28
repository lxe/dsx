#![cfg(feature = "cuda")]
//! Small-row GEMVs and the hc dots: one-warp-per-row vs split-K, decode shapes (cargo test -- --ignored --nocapture).
use dsx::gpu::*;
use std::time::Instant;

fn time(g: &Gpu, k: Kernel, grid: (u32, u32, u32), block: u32, args: &[Arg]) -> f64 {
    g.launch(k, grid, block, 0, args).unwrap();
    g.sync().unwrap();
    let t = Instant::now();
    for _ in 0..200 {
        g.launch(k, grid, block, 0, args).unwrap();
    }
    g.sync().unwrap();
    t.elapsed().as_secs_f64() / 200.0 * 1e6
}

#[test]
#[ignore]
fn small_gemv() {
    let g = Gpu::new(0).unwrap();
    let x = g.upload(&vec![0.5f32; 8 * 32768]).unwrap();
    let y = g.alloc(8 * 65536 * 4).unwrap();
    for (name, rows, k, esz) in [("bf16", 384usize, 5120usize, 2usize), ("f16", 512, 5120, 2), ("f16", 32, 5120, 2), ("f16", 128, 512, 2), ("f16", 4096, 1280, 2)] {
        let w = g.upload(&vec![0x3c00u16; rows * k]).unwrap();
        let a = [Arg::P(w.ptr), Arg::P(x.ptr), Arg::P(y.ptr), Arg::I(rows as i32), Arg::I(k as i32), Arg::I(1), Arg::I(k as i32), Arg::I(rows as i32)];
        let t0 = time(&g, g.kernel(&format!("gemv_{name}")).unwrap(), ((rows as u32).div_ceil(4), 1, 1), 128, &a);
        let t1 = time(&g, g.kernel(&format!("gemv_{name}_sk")).unwrap(), (rows as u32, 1, 1), 128, &a);
        eprintln!("{name} {rows}x{k}: warp/row {t0:.1} us, split-K {t1:.1} us ({:.0} -> {:.0} GB/s)", (rows * k * esz) as f64 / t0 / 1e3, (rows * k * esz) as f64 / t1 / 1e3);
    }
    for (rows, k) in [(512usize, 5120usize), (1280, 5120)] {
        let qs = g.upload(&vec![1i8; rows * k]).unwrap();
        let d = g.upload(&vec![0x2000u16; rows * k / 32]).unwrap();
        let a = [Arg::P(qs.ptr), Arg::P(x.ptr), Arg::P(y.ptr), Arg::I(rows as i32), Arg::I(k as i32), Arg::I(1), Arg::I(k as i32), Arg::I(rows as i32), Arg::P(d.ptr)];
        let t0 = time(&g, g.kernel("gemv_q8r").unwrap(), ((rows as u32).div_ceil(4), 1, 1), 128, &a);
        let t1 = time(&g, g.kernel("gemv_q8r_sk").unwrap(), (rows as u32, 1, 1), 128, &a);
        eprintln!("q8r {rows}x{k}: warp/row {t0:.1} us, split-K {t1:.1} us");
    }
    // hc dots
    let hcd = 20480usize;
    let fnw = g.upload(&vec![0.01f32; 24 * hcd]).unwrap();
    let h = g.upload(&vec![0.5f32; hcd]).unwrap();
    let mix = g.alloc(24 * 4).unwrap();
    let part = g.alloc(25 * 16 * 4).unwrap();
    let t0 = time(&g, g.kernel("hc_dots").unwrap(), (24, 1, 1), 256, &[Arg::P(fnw.ptr), Arg::P(h.ptr), Arg::P(mix.ptr), Arg::I(hcd as i32), Arg::F(1e-20)]);
    for s in [4u32, 8, 16] {
        let t1 = time(&g, g.kernel("hc_part").unwrap(), (25, 1, s), 256, &[Arg::P(fnw.ptr), Arg::P(h.ptr), Arg::P(part.ptr), Arg::I(hcd as i32), Arg::I(s as i32)]);
        let t2 = time(&g, g.kernel("hc_reduce").unwrap(), (1, 1, 1), 32, &[Arg::P(part.ptr), Arg::P(mix.ptr), Arg::I(hcd as i32), Arg::I(s as i32), Arg::F(1e-20)]);
        eprintln!("hc dots: one block/row {t0:.1} us; split {s}: {t1:.1} + reduce {t2:.1} us");
    }
}
