//! Scalar numerics shared by the reference forward: norms, the training-time fake quantizers
//! (FP8 E4M3 with power-of-two scales, FP4 E2M1 with E8M0 or E4M3 scales), RoPE with YaRN.

use half::bf16;

pub fn rmsnorm(x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let r = 1.0 / (ms + eps).sqrt();
    for i in 0..x.len() {
        out[i] = w[i] * (x[i] * r);
    }
}

pub fn rmsnorm_inplace(x: &mut [f32], w: &[f32], eps: f32) {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let r = 1.0 / (ms + eps).sqrt();
    for i in 0..x.len() {
        x[i] = w[i] * (x[i] * r);
    }
}

#[inline]
pub fn round_bf16(x: f32) -> f32 {
    bf16::from_f32(x).to_f32()
}

pub fn round_bf16_slice(x: &mut [f32]) {
    for v in x {
        *v = round_bf16(*v);
    }
}

#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
pub fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

#[inline]
pub fn softplus(x: f32) -> f32 {
    // torch: x > 20 -> x
    if x > 20.0 { x } else { x.exp().ln_1p() }
}

/// round half to even on an f32 grid of `step` (step a power of two)
#[inline]
fn rne(v: f32, step: f32) -> f32 {
    let q = v / step;
    let r = q.round();
    let r = if (q - q.trunc()).abs() == 0.5 { 2.0 * (q / 2.0).round() } else { r };
    r * step
}

/// f32 -> nearest float8_e4m3fn (RNE, saturating at 448)
pub fn e4m3(v: f32) -> f32 {
    if v == 0.0 || !v.is_finite() {
        return if v.is_nan() { v } else if v == 0.0 { 0.0 } else { v.signum() * 448.0 };
    }
    let a = v.abs();
    let min_normal = 2f32.powi(-6);
    let q = if a < min_normal {
        rne(a, 2f32.powi(-9))
    } else {
        let e = a.log2().floor() as i32;
        rne(a, 2f32.powi(e - 3))
    };
    v.signum() * q.min(448.0)
}

const E2M1: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// f32 (already scaled, |v| <= 6) -> nearest E2M1, ties to the even mantissa
pub fn e2m1(v: f32) -> f32 {
    let a = v.abs().min(6.0);
    let mut best = 0usize;
    for i in 1..8 {
        let d = (a - E2M1[i]).abs();
        let db = (a - E2M1[best]).abs();
        if d < db || (d == db && i % 2 == 0 && best % 2 == 1) {
            best = i;
        }
    }
    v.signum() * E2M1[best]
}

/// 2^ceil(log2(x)), matching kernel.py fast_round_scale's bit trick
#[inline]
pub fn pow2_ceil(x: f32) -> f32 {
    let b = x.to_bits();
    let e = ((b >> 23) & 0xFF) as i32 - 127 + if b & 0x7F_FFFF != 0 { 1 } else { 0 };
    f32::from_bits(((e + 127) as u32) << 23)
}

/// kernel.py act_quant(inplace=True, scale_fmt=ue8m0): FP8 fake quant per block, power-of-two scale
pub fn fake_fp8(x: &mut [f32], block: usize) {
    for b in x.chunks_mut(block) {
        let amax = b.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-4);
        let s = pow2_ceil(amax / 448.0);
        for v in b.iter_mut() {
            *v = e4m3((*v / s).clamp(-448.0, 448.0)) * s;
        }
    }
}

/// kernel.py fp4_act_quant(inplace=True): E2M1 per block with an E8M0 (power of two) or E4M3 scale
pub fn fake_fp4(x: &mut [f32], block: usize, e4m3_scale: bool) {
    for b in x.chunks_mut(block) {
        let amax0 = b.iter().fold(0f32, |m, v| m.max(v.abs()));
        let s = if e4m3_scale {
            e4m3(amax0.max(6.0 * 2f32.powi(-9)) / 6.0)
        } else {
            pow2_ceil(amax0.max(6.0 * 2f32.powi(-126)) / 6.0)
        };
        for v in b.iter_mut() {
            *v = e2m1((*v / s).clamp(-6.0, 6.0)) * s;
        }
    }
}

/// RoPE tables: (cos, sin) per position and pair, model.py precompute_freqs_cis
pub struct Rope {
    pub freqs: Vec<f32>, // [dim/2]
}

impl Rope {
    pub fn new(dim: usize, original_seq_len: usize, base: f32, factor: f32, beta_fast: f32, beta_slow: f32) -> Rope {
        let half = dim / 2;
        let mut freqs: Vec<f32> = (0..half).map(|i| 1.0 / base.powf((2 * i) as f32 / dim as f32)).collect();
        if original_seq_len > 0 {
            let corrected = |rot: f32| -> f32 {
                dim as f32 * ((original_seq_len as f32) / (rot * 2.0 * std::f32::consts::PI)).ln() / (2.0 * base.ln())
            };
            let low = corrected(beta_fast).floor().max(0.0);
            let high = corrected(beta_slow).ceil().min((dim - 1) as f32);
            for (i, f) in freqs.iter_mut().enumerate() {
                let ramp = ((i as f32 - low) / (high - low).max(1e-3)).clamp(0.0, 1.0);
                let smooth = 1.0 - ramp;
                *f = *f / factor * (1.0 - smooth) + *f * smooth;
            }
        }
        Rope { freqs }
    }

    /// rotate adjacent pairs of `x` (len = dim) for position `pos`; inverse conjugates
    pub fn apply(&self, x: &mut [f32], pos: usize, inverse: bool) {
        for (i, f) in self.freqs.iter().enumerate() {
            // torch.polar in f32: angle = pos * f computed in f32 by torch.outer
            let ang = pos as f32 * f;
            let (s, c) = (ang.sin(), ang.cos());
            let s = if inverse { -s } else { s };
            let (a, b) = (x[2 * i], x[2 * i + 1]);
            x[2 * i] = a * c - b * s;
            x[2 * i + 1] = a * s + b * c;
        }
    }
}

/// The exact float8_e4m3fn byte for `x`, or None if `x` is not on the e4m3 grid (or beyond 448)
fn e4m3_code(x: f32) -> Option<u8> {
    let sign = if x.is_sign_negative() { 0x80u8 } else { 0 };
    let a = x.abs();
    if a == 0.0 {
        return Some(sign);
    }
    if !(a <= 448.0) {
        return None;
    }
    let b = a.to_bits();
    let e = ((b >> 23) & 0xFF) as i32 - 127;
    if e >= -6 {
        // normal: 3 mantissa bits, the other 20 zero
        (b & 0xF_FFFF == 0).then(|| sign | (((e + 7) as u8) << 3) | ((b >> 20) & 7) as u8)
    } else {
        // subnormal: a multiple of 2^-9 below 2^-6
        let m = a * 512.0;
        (m == m.trunc()).then(|| sign | m as u8)
    }
}

/// F16 matrix [rows][cols] -> FP8 e4m3 bytes [rows][cols] + E8M0 scales [rows/32][cols/32] (the HF checkpoint's
/// Fp8B32 layout), exactly: each 32x32 block takes the smallest power of two that keeps it within 448, and every value
/// must land on the e4m3 grid. None if any value does not (or rows / cols are not multiples of 32). A matrix that was
/// dequantized from such FP8 to F16 converts back losslessly.
pub fn f16_to_fp8b32(h: &[half::f16], rows: usize, cols: usize) -> Option<(Vec<u8>, Vec<u8>)> {
    use rayon::prelude::*;
    if rows % 32 != 0 || cols % 32 != 0 || h.len() != rows * cols {
        return None;
    }
    let nb = cols / 32;
    let mut q = vec![0u8; rows * cols];
    let mut sc = vec![0u8; rows / 32 * nb];
    let ok = q.par_chunks_mut(32 * cols).zip(sc.par_chunks_mut(nb)).enumerate().all(|(rb, (q, sc))| {
        let h = &h[rb * 32 * cols..(rb + 1) * 32 * cols];
        (0..nb).all(|b| {
            let amax = (0..32).flat_map(|r| &h[r * cols + b * 32..r * cols + b * 32 + 32]).fold(0f32, |m, v| m.max(v.to_f32().abs()));
            if !amax.is_finite() {
                return false;
            }
            let mut s = if amax == 0.0 { 0 } else { ((pow2_ceil(amax / 448.0).to_bits() >> 23) as i32) - 127 };
            if amax * 2f32.powi(-s) > 448.0 {
                s += 1;
            }
            if !(-127..=127).contains(&s) {
                return false;
            }
            sc[b] = (s + 127) as u8;
            let inv = 2f32.powi(-s);
            (0..32).all(|r| {
                (0..32).all(|c| {
                    let i = r * cols + b * 32 + c;
                    e4m3_code(h[i].to_f32() * inv).map(|v| q[i] = v).is_some()
                })
            })
        })
    });
    ok.then_some((q, sc))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e4m3_grid() {
        assert_eq!(e4m3(1.0), 1.0);
        assert_eq!(e4m3(1.0625), 1.0); // tie -> even
        assert_eq!(e4m3(1.1875), 1.25); // tie 1.125|1.25 -> even mantissa (1.25 = 1.010)
        assert_eq!(e4m3(500.0), 448.0);
        assert_eq!(e4m3(2f32.powi(-10)), 0.0); // tie between 0 and 2^-9 -> 0
        assert_eq!(e4m3(3.0 * 2f32.powi(-10)), 2.0 * 2f32.powi(-9)); // 1.5 steps -> 2 (even)
    }

    #[test]
    fn e2m1_ties() {
        assert_eq!(e2m1(0.25), 0.0);
        assert_eq!(e2m1(0.75), 1.0);
        assert_eq!(e2m1(1.25), 1.0);
        assert_eq!(e2m1(2.5), 2.0);
        assert_eq!(e2m1(5.0), 4.0);
        assert_eq!(e2m1(-3.4), -3.0);
    }

    #[test]
    fn pow2() {
        assert_eq!(pow2_ceil(1.0), 1.0);
        assert_eq!(pow2_ceil(1.5), 2.0);
        assert_eq!(pow2_ceil(0.3), 0.5);
    }

    #[test]
    fn f16_fp8b32_roundtrip() {
        // every e4m3 code times block scales from 2^-20 to 2^0, through F16 and back
        let (rows, cols) = (64, 96);
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let codes: Vec<u8> = (0..rows * cols).map(|i| if i < 254 { (i as u8).wrapping_add(if i >= 127 { 1 } else { 0 }) } else { (next() % 254) as u8 }).map(|b| if b & 0x7F == 0x7F { 0x7E } else { b }).collect();
        let scales: Vec<u8> = (0..rows / 32 * cols / 32).map(|i| 107 + (i as u8 * 7) % 21).collect();
        let w: Vec<f32> = (0..rows * cols).map(|i| crate::model::fp8_e4m3_to_f32(codes[i]) * crate::model::e8m0_to_f32(scales[(i / cols / 32) * (cols / 32) + (i % cols) / 32])).collect();
        let h: Vec<half::f16> = w.iter().map(|&v| half::f16::from_f32(v)).collect();
        let (q, sc) = f16_to_fp8b32(&h, rows, cols).expect("exact");
        for i in 0..rows * cols {
            let back = crate::model::fp8_e4m3_to_f32(q[i]) * crate::model::e8m0_to_f32(sc[(i / cols / 32) * (cols / 32) + (i % cols) / 32]);
            assert_eq!(back.to_bits(), h[i].to_f32().to_bits(), "value {i}");
        }
        // off the grid: block (0, 0) holding only 1 and 1 + 2^-4 (scale 2^-8: 256 and 272, between e4m3's 256 and 288)
        let mut h2 = h.clone();
        (0..32).for_each(|r| h2[r * cols..r * cols + 32].fill(half::f16::ZERO));
        h2[0] = half::f16::from_f32(1.0);
        assert!(f16_to_fp8b32(&h2, rows, cols).is_some());
        h2[1] = half::f16::from_f32(1.0625);
        assert!(f16_to_fp8b32(&h2, rows, cols).is_none());
        assert!(f16_to_fp8b32(&h[..rows * cols - cols], rows - 1, cols).is_none());
    }
}

/// Expert input rotation: an orthonormal Sylvester Hadamard of size `ROT` applied to each `ROT`-wide block of `x`
/// (`x.len()` a multiple of ROT). Its own inverse. Experts quantized in the rotated basis (W R) see R x.
pub const ROT: usize = 128;

pub fn fwht_blocks(x: &mut [f32]) {
    assert!(x.len() % ROT == 0);
    let s = 1.0 / (ROT as f32).sqrt();
    for b in x.chunks_exact_mut(ROT) {
        let mut h = 1;
        while h < ROT {
            for i in (0..ROT).step_by(2 * h) {
                for j in i..i + h {
                    let (a, c) = (b[j], b[j + h]);
                    b[j] = a + c;
                    b[j + h] = a - c;
                }
            }
            h *= 2;
        }
        for v in b.iter_mut() {
            *v *= s;
        }
    }
}

#[cfg(test)]
mod fwht_tests {
    use super::*;
    #[test]
    fn involution_and_norm() {
        let x: Vec<f32> = (0..2 * ROT).map(|i| ((i * 37 % 11) as f32 - 5.0) * 0.3).collect();
        let mut y = x.clone();
        fwht_blocks(&mut y);
        let (n0, n1): (f32, f32) = (x.iter().map(|v| v * v).sum(), y.iter().map(|v| v * v).sum());
        assert!((n0 - n1).abs() < 1e-3 * n0);
        fwht_blocks(&mut y);
        for (a, b) in x.iter().zip(&y) {
            assert!((a - b).abs() < 1e-4);
        }
    }
}
