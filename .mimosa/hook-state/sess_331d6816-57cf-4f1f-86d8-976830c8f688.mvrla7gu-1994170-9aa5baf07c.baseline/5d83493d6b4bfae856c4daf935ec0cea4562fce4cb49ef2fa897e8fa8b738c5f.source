#![cfg(feature = "cuda")]
//! DSX_ENGRAM_FP8: the engram wkv GEMV on FP8 + 32x32 E8M0 scales (converted exactly from the F16 copy) against
//! gemv_f16 on the same matrix, at the engram shape (25600 x 6144). DSX_ENGRAM_HF=dir with L1.weight.bin /
//! L1.scale.bin (the checkpoint's layers.1.engram.wkv weight and scale bytes) uses the real matrix; otherwise random
//! FP8 codes and scales. The speed test is --ignored:
//!   cargo test --release --features cuda --test gpu_engram_fp8 -- --ignored --nocapture
use dsx::gpu::*;

const ROWS: usize = 25600;
const COLS: usize = 6144;

fn rng(seed: &mut u64) -> u64 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    *seed
}

/// the F16 matrix dsx's GGUF holds: FP8 codes times their block scales, rounded to f16
fn f16_matrix() -> Vec<half::f16> {
    let (q, sc) = match std::env::var("DSX_ENGRAM_HF") {
        Ok(d) => (std::fs::read(format!("{d}/L1.weight.bin")).unwrap(), std::fs::read(format!("{d}/L1.scale.bin")).unwrap()),
        Err(_) => {
            let mut seed = 0x5eed_e17a_0f8b_32u64;
            let q = (0..ROWS * COLS).map(|_| rng(&mut seed) as u8).map(|b| if b & 0x7F == 0x7F { 0x7E } else { b }).collect();
            let sc = (0..ROWS / 32 * COLS / 32).map(|_| 109 + (rng(&mut seed) % 8) as u8).collect();
            (q, sc)
        }
    };
    assert_eq!((q.len(), sc.len()), (ROWS * COLS, ROWS * COLS / 1024));
    let nb = COLS / 32;
    (0..ROWS * COLS).map(|i| half::f16::from_f32(dsx::model::fp8_e4m3_to_f32(q[i]) * dsx::model::e8m0_to_f32(sc[i / COLS / 32 * nb + i % COLS / 32]))).collect()
}

struct Setup {
    g: Gpu,
    w16: Buf,
    w8: Buf,
    sc: Buf,
    x: Buf,
    y16: Buf,
    y8: Buf,
}

fn setup() -> Setup {
    let g = Gpu::new(0).unwrap();
    let h = f16_matrix();
    let (q, sc) = dsx::numerics::f16_to_fp8b32(&h, ROWS, COLS).expect("the F16 matrix converts exactly");
    let mut seed = 0x1234_5678u64;
    let x: Vec<f32> = (0..COLS).map(|_| (rng(&mut seed) % 2001) as f32 / 1000.0 - 1.0).collect();
    Setup { w16: g.upload(&h).unwrap(), w8: g.upload(&q).unwrap(), sc: g.upload(&sc).unwrap(), x: g.upload(&x).unwrap(), y16: g.alloc(ROWS * 4).unwrap(), y8: g.alloc(ROWS * 4).unwrap(), g }
}

/// 0: gemv_f16 on the F16 copy, 1: gemv_fp8b32 (4 rows a warp), 2: gemv_fp8b32_o (gemv_f16's order)
const KERNELS: [(&str, u32); 3] = [("gemv_f16", 4), ("gemv_fp8b32", 16), ("gemv_fp8b32_o", 4)];

fn launch_k(s: &Setup, which: usize, y: u64) {
    let (name, per) = KERNELS[which];
    let w = if which == 0 { s.w16.ptr } else { s.w8.ptr };
    let c = COLS as i32;
    s.g.launch(s.g.kernel(name).unwrap(), ((ROWS as u32).div_ceil(per), 1, 1), 128, 0, &[Arg::P(w), Arg::P(s.x.ptr), Arg::P(y), Arg::I(ROWS as i32), Arg::I(c), Arg::I(1), Arg::I(c), Arg::I(ROWS as i32), Arg::P(s.sc.ptr)]).unwrap();
}

fn launch(s: &Setup, fp8: bool) {
    launch_k(s, fp8 as usize, if fp8 { s.y8.ptr } else { s.y16.ptr });
}

#[test]
fn fp8_gemv_in_f16_order_is_bit_identical() {
    let s = setup();
    launch_k(&s, 0, s.y16.ptr);
    launch_k(&s, 2, s.y8.ptr);
    s.g.sync().unwrap();
    let a: Vec<f32> = s.g.download(s.y16.ptr, ROWS).unwrap();
    let b: Vec<f32> = s.g.download(s.y8.ptr, ROWS).unwrap();
    let diff = a.iter().zip(&b).filter(|(p, q)| p.to_bits() != q.to_bits()).count();
    println!("gemv_fp8b32_o vs gemv_f16: {diff} of {ROWS} outputs differ in any bit");
    assert_eq!(diff, 0);
}

#[test]
fn fp8_gemv_matches_f16() {
    let s = setup();
    launch(&s, false);
    launch(&s, true);
    s.g.sync().unwrap();
    let a: Vec<f32> = s.g.download(s.y16.ptr, ROWS).unwrap();
    let b: Vec<f32> = s.g.download(s.y8.ptr, ROWS).unwrap();
    let scale = a.iter().fold(0f32, |m, v| m.max(v.abs()));
    let err = a.iter().zip(&b).fold(0f32, |m, (p, q)| m.max((p - q).abs()));
    println!("engram wkv GEMV: max |fp8 - f16| {err:.3e}, max |y| {scale:.3e}");
    // the same weights; only the summation order differs
    assert!(err <= 1e-5 * scale.max(1.0), "fp8 vs f16 GEMV: {err} (max |y| {scale})");
}

#[test]
#[ignore]
fn fp8_gemv_speed() {
    let s = setup();
    let (e0, e1) = (s.g.timed_event().unwrap(), s.g.timed_event().unwrap());
    let mut t = [vec![], vec![], vec![]];
    // interleaved, median of 31 each
    for _ in 0..31 {
        for (i, tt) in t.iter_mut().enumerate() {
            s.g.record(e0, false).unwrap();
            launch_k(&s, i, s.y8.ptr);
            s.g.record(e1, false).unwrap();
            s.g.sync().unwrap();
            tt.push(s.g.elapsed_ms(e0, e1).unwrap());
        }
    }
    for (i, (name, _)) in KERNELS.iter().enumerate() {
        t[i].sort_by(f32::total_cmp);
        let ms = t[i][t[i].len() / 2];
        let bytes = if i == 0 { ROWS * COLS * 2 } else { ROWS * COLS + ROWS * COLS / 1024 };
        println!("{name}: {:.1} us median ({:.1}..{:.1}), {:.0} GB/s", ms * 1e3, t[i][0] * 1e3, t[i][t[i].len() - 1] * 1e3, bytes as f64 / ms as f64 / 1e6);
    }
}

/// gemv_fp8b32 off the engram shape: rows not a multiple of 16, k not a multiple of 512, T = 3 with strides,
/// against an f64 reference on the dequantized weights
#[test]
fn fp8_gemv_tail_and_tokens() {
    let g = Gpu::new(0).unwrap();
    let (rows, k, t, ldx, ldy) = (1000usize, 1056usize, 3usize, 1088usize, 1016usize);
    let mut seed = 0xfeed_beefu64;
    let q: Vec<u8> = (0..rows * k).map(|_| rng(&mut seed) as u8).map(|b| if b & 0x7F == 0x7F { 0x7E } else { b }).collect();
    let nb = k / 32;
    let sc: Vec<u8> = (0..rows.div_ceil(32) * nb).map(|_| 112 + (rng(&mut seed) % 10) as u8).collect();
    let x: Vec<f32> = (0..t * ldx).map(|_| (rng(&mut seed) % 2001) as f32 / 1000.0 - 1.0).collect();
    let (wb, sb, xb, yb) = (g.upload(&q).unwrap(), g.upload(&sc).unwrap(), g.upload(&x).unwrap(), g.alloc_zeroed(t * ldy * 4).unwrap());
    g.launch(g.kernel("gemv_fp8b32").unwrap(), ((rows as u32).div_ceil(16), 1, 1), 128, 0, &[Arg::P(wb.ptr), Arg::P(xb.ptr), Arg::P(yb.ptr), Arg::I(rows as i32), Arg::I(k as i32), Arg::I(t as i32), Arg::I(ldx as i32), Arg::I(ldy as i32), Arg::P(sb.ptr)]).unwrap();
    g.sync().unwrap();
    let y: Vec<f32> = g.download(yb.ptr, t * ldy).unwrap();
    for ti in 0..t {
        for r in 0..rows {
            let want: f64 = (0..k).map(|c| (dsx::model::fp8_e4m3_to_f32(q[r * k + c]) * dsx::model::e8m0_to_f32(sc[r / 32 * nb + c / 32])) as f64 * x[ti * ldx + c] as f64).sum();
            let got = y[ti * ldy + r] as f64;
            assert!((got - want).abs() <= 1e-5 * want.abs().max(1.0), "t {ti} row {r}: {got} vs {want}");
        }
        assert!(y[ti * ldy + rows..(ti + 1) * ldy].iter().all(|&v| v == 0.0), "wrote past rows");
    }
}
