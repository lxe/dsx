#![cfg(feature = "cuda")]
//! GPU GEMV kernels vs the CPU block decoders on random blocks.
use dsx::formats::dequant_row;
use dsx::gguf::GType;
use dsx::gpu::*;

fn rng(seed: &mut u64) -> u64 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    *seed
}

fn random_rows(ty: GType, rows: usize, cols: usize, seed: &mut u64) -> Vec<u8> {
    let (w, b) = ty.block();
    let nb = rows * cols / w;
    let mut v = vec![0u8; nb * b];
    for x in v.iter_mut() {
        *x = rng(seed) as u8;
    }
    // sane fp16 scales
    let scale_at: &[usize] = match ty {
        GType::Q8_0 | GType::Q2_0 => &[0],
        GType::Ptq1_0 => &[26],
        GType::Q3K => &[108],
        _ => &[],
    };
    for blk in 0..nb {
        for &o in scale_at {
            let s = half::f16::from_f32(0.01 + (rng(seed) % 100) as f32 * 1e-4).to_le_bytes();
            v[blk * b + o..blk * b + o + 2].copy_from_slice(&s);
        }
        if ty == GType::Ptq1_0 {
            // valid base-3 bytes only (<= 242 after the ceil mapping is any byte; all bytes decode, keep as is)
        }
    }
    v
}

#[test]
fn gemv_kernels_match_cpu() {
    let g = Gpu::new(0).unwrap();
    let mut seed = 0x1234_5678_9abc_def0u64;
    for (ty, kname) in [(GType::Q8_0, "gemv_q8_0"), (GType::Q2_0, "gemv_q2_0"), (GType::Ptq1_0, "gemv_ptq1_0"), (GType::Q3K, "gemv_q3_k")] {
        let (rows, cols, t) = (96usize, 5120usize, 3usize);
        let w = random_rows(ty, rows, cols, &mut seed);
        let x: Vec<f32> = (0..t * cols).map(|_| (rng(&mut seed) % 2001) as f32 / 1000.0 - 1.0).collect();
        let mut want = vec![0f32; t * rows];
        let rb = ty.row_bytes(cols);
        let mut buf = vec![0f32; cols];
        for r in 0..rows {
            dequant_row(ty, &w[r * rb..(r + 1) * rb], &mut buf);
            for tt in 0..t {
                want[tt * rows + r] = buf.iter().zip(&x[tt * cols..(tt + 1) * cols]).map(|(a, b)| (*a as f64) * (*b as f64)).sum::<f64>() as f32;
            }
        }
        let (dw, dx, dy) = (g.upload(&w).unwrap(), g.upload(&x).unwrap(), g.alloc(t * rows * 4).unwrap());
        g.launch(g.kernel(kname).unwrap(), ((rows as u32).div_ceil(4), 1, 1), 128, 0,
            &[Arg::P(dw.ptr), Arg::P(dx.ptr), Arg::P(dy.ptr), Arg::I(rows as i32), Arg::I(cols as i32), Arg::I(t as i32), Arg::I(cols as i32), Arg::I(rows as i32)]).unwrap();
        let got: Vec<f32> = g.download(dy.ptr, t * rows).unwrap();
        let maxrel = want.iter().zip(&got).map(|(a, b)| (a - b).abs() / a.abs().max(1e-2)).fold(0f32, f32::max);
        eprintln!("{kname}: max rel err {maxrel:.2e}");
        assert!(maxrel < 1e-3, "{kname}");
    }
}

#[test]
fn dequant_gemm_matches_gemv() {
    let g = Gpu::new(0).unwrap();
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    for (ty, kname, id) in [(GType::Q8_0, "gemv_q8_0", 8), (GType::Q2_0, "gemv_q2_0", 42), (GType::Ptq1_0, "gemv_ptq1_0", 143), (GType::Q3K, "gemv_q3_k", 11)] {
        let (rows, cols, t) = (256usize, 5120usize, 8usize);
        let w = random_rows(ty, rows, cols, &mut seed);
        let x: Vec<f32> = (0..t * cols).map(|_| ((rng(&mut seed) % 2001) as f32 / 1000.0 - 1.0) * 0.5).collect();
        let (dw, dx) = (g.upload(&w).unwrap(), g.upload(&x).unwrap());
        let (y1, y2) = (g.alloc(t * rows * 4).unwrap(), g.alloc(t * rows * 4).unwrap());
        g.launch(g.kernel(kname).unwrap(), ((rows as u32).div_ceil(4), 1, 1), 128, 0,
            &[Arg::P(dw.ptr), Arg::P(dx.ptr), Arg::P(y1.ptr), Arg::I(rows as i32), Arg::I(cols as i32), Arg::I(t as i32), Arg::I(cols as i32), Arg::I(rows as i32)]).unwrap();
        let (wh, xh) = (g.alloc(rows * cols * 2).unwrap(), g.alloc(t * cols * 2).unwrap());
        g.launch(g.kernel("dequant_f16").unwrap(), ((cols as u32).div_ceil(256), rows as u32, 1), 256, 0,
            &[Arg::I(id), Arg::P(dw.ptr), Arg::P(0), Arg::P(wh.ptr), Arg::I(rows as i32), Arg::I(cols as i32)]).unwrap();
        g.launch(g.kernel("to_f16").unwrap(), (((t * cols) as u32).div_ceil(256), 1, 1), 256, 0, &[Arg::P(dx.ptr), Arg::P(xh.ptr), Arg::I((t * cols) as i32)]).unwrap();
        g.gemm_f16(xh.ptr, wh.ptr, y2.ptr, t, cols, rows, rows).unwrap();
        let (a, b): (Vec<f32>, Vec<f32>) = (g.download(y1.ptr, t * rows).unwrap(), g.download(y2.ptr, t * rows).unwrap());
        let scale = a.iter().fold(0f32, |m, v| m.max(v.abs()));
        let maxd = a.iter().zip(&b).map(|(p, q)| (p - q).abs()).fold(0f32, f32::max) / scale;
        eprintln!("{kname} vs dequant+cuBLAS: max |diff| / max |y| = {maxd:.2e}");
        assert!(maxd < 5e-3, "{kname}");
    }
}
