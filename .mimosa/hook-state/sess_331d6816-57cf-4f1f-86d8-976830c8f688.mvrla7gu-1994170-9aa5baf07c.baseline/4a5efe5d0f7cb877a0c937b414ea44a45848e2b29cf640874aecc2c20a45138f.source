//! Checkpoint tensors -> device layouts (see kernels/gemv.h for the tile formats).
//!
//! Dense FP8 weights keep the checkpoint's values and E8M0 32x32 block scales; their bytes are reinterpreted as
//! gfx942's FNUZ fp8 (same bits = OCP value / 2, so the kernels multiply by 2). Routed experts stay MXFP4 (exact)
//! or are re-quantized on the GPU to a 3-bit / 2-bit codebook with one e4m3 scale per 32 weights.

use crate::hip::{Buf, Gpu};
use anyhow::{Context, Result, bail};
use crate::ck::Ck as SafeTensors;
use rayon::prelude::*;

pub fn bf16_to_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect()
}

/// A tensor as f32 (BF16 / F32 / F16 checkpoint dtypes)
pub fn f32_of(st: &SafeTensors, name: &str) -> Result<Vec<f32>> {
    let t = st.get(name)?;
    let b = st.bytes(t);
    Ok(match t.dtype.as_str() {
        "BF16" => bf16_to_f32(b),
        "F32" => b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
        "F16" => b.chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
        d => bail!("{name}: dtype {d} as f32"),
    })
}

pub fn shape2(st: &SafeTensors, name: &str) -> Result<(usize, usize)> {
    let t = st.get(name)?;
    match t.shape.len() {
        2 => Ok((t.shape[0], t.shape[1])),
        1 => Ok((1, t.shape[0])),
        _ => bail!("{name}: shape {:?}", t.shape),
    }
}

/// FP8 [n][k] (OCP bytes) -> tiles [n/16][k/64][64 lanes][16 B], FNUZ bytes (0x80 = -0 -> 0)
pub fn tile_fp8(w: &[u8], n: usize, k: usize) -> Vec<u8> {
    assert!(n % 16 == 0 && k % 64 == 0, "fp8 tile {n}x{k}");
    let kp = k / 64;
    let mut out = vec![0u8; n * k];
    out.par_chunks_mut(kp * 1024).enumerate().for_each(|(rt, o)| {
        for p in 0..kp {
            for l in 0..64 {
                let r = rt * 16 + l % 16;
                for i in 0..16 {
                    let c = p * 64 + (i / 8) * 32 + 8 * (l / 16) + i % 8;
                    let b = w[r * k + c];
                    o[(p * 64 + l) * 16 + i] = if b == 0x80 { 0 } else { b };
                }
            }
        }
    });
    out
}

/// BF16 [n][k] -> tiles [n/16][k/32][64][8 bf16]
pub fn tile_bf16(w: &[u8], n: usize, k: usize) -> Vec<u8> {
    assert!(n % 16 == 0 && k % 32 == 0, "bf16 tile {n}x{k}");
    let kq = k / 32;
    let mut out = vec![0u8; n * k * 2];
    out.par_chunks_mut(kq * 1024).enumerate().for_each(|(rt, o)| {
        for q in 0..kq {
            for l in 0..64 {
                let r = rt * 16 + l % 16;
                for i in 0..8 {
                    let c = q * 32 + (i / 4) * 16 + 4 * (l / 16) + i % 4;
                    let s = (r * k + c) * 2;
                    let d = ((q * 64 + l) * 8 + i) * 2;
                    o[d] = w[s];
                    o[d + 1] = w[s + 1];
                }
            }
        }
    });
    out
}

/// MXFP4 [n][k/2] (low nibble = even column) + E8M0 [n][k/32] -> codes [n/16][k/128][64][4 dwords] and scales
/// [n/16][k/128][16 rows][4]. Negative zero (code 8) becomes 0 (FNUZ has no -0).
pub fn tile_fp4(w: &[u8], sc: &[u8], n: usize, k: usize) -> (Vec<u8>, Vec<u8>) {
    assert!(n % 16 == 0 && k % 128 == 0, "fp4 tile {n}x{k}");
    let ku = k / 128;
    let mut codes = vec![0u8; n * k / 2];
    let mut scales = vec![0u8; n * k / 32];
    let code = |r: usize, c: usize| -> u32 {
        let b = w[r * k / 2 + c / 2];
        let v = if c % 2 == 0 { b & 15 } else { b >> 4 };
        if v == 8 { 0 } else { v as u32 }
    };
    codes.par_chunks_mut(ku * 1024).enumerate().for_each(|(rt, o)| {
        for u in 0..ku {
            for l in 0..64 {
                let r = rt * 16 + l % 16;
                for m in 0..4 {
                    let c0 = (4 * u + m) * 32 + 8 * (l / 16);
                    let mut d = 0u32;
                    for j in 0..8 {
                        let nib = if j < 4 { 2 * j } else { 2 * (j - 4) + 1 };
                        d |= code(r, c0 + j) << (4 * nib);
                    }
                    o[((u * 64 + l) * 4 + m) * 4..][..4].copy_from_slice(&d.to_le_bytes());
                }
            }
        }
    });
    scales.par_chunks_mut(ku * 64).enumerate().for_each(|(rt, o)| {
        for u in 0..ku {
            for r16 in 0..16 {
                for m in 0..4 {
                    o[(u * 16 + r16) * 4 + m] = sc[(rt * 16 + r16) * (k / 32) + 4 * u + m];
                }
            }
        }
    });
    (codes, scales)
}

/// Device matrix in one of the tile formats
pub struct DMat {
    pub w: Buf,
    pub sc: Option<Buf>,
    pub n: usize,
    pub k: usize,
}

pub fn up_fp8(g: &Gpu, st: &SafeTensors, name: &str) -> Result<DMat> {
    let (n, k) = shape2(st, name)?;
    let t = st.get(name)?;
    if t.dtype != "F8_E4M3" {
        bail!("{name}: {} not FP8", t.dtype);
    }
    let w = tile_fp8(st.bytes(t), n, k);
    let sname = name.strip_suffix(".weight").unwrap().to_string() + ".scale";
    let s = st.bytes(st.get(&sname)?);
    if s.len() != (n / 32) * (k / 32) {
        bail!("{sname}: {} scale bytes for {n}x{k}", s.len());
    }
    Ok(DMat { w: g.upload(&w)?, sc: Some(g.upload(s)?), n, k })
}

pub fn up_bf16(g: &Gpu, st: &SafeTensors, name: &str) -> Result<DMat> {
    let (n, k) = shape2(st, name)?;
    let t = st.get(name)?;
    if t.dtype != "BF16" {
        bail!("{name}: {} not BF16", t.dtype);
    }
    let w = tile_bf16(st.bytes(t), n, k);
    Ok(DMat { w: g.upload(&w)?, sc: None, n, k })
}

pub fn up_f32(g: &Gpu, st: &SafeTensors, name: &str) -> Result<Buf> {
    g.upload(&f32_of(st, name).with_context(|| name.to_string())?)
}

/// raw bytes of a tensor to the device (BF16 tables read row-wise: embeddings, indexer wk)
pub fn up_raw(g: &Gpu, st: &SafeTensors, name: &str) -> Result<Buf> {
    g.upload(st.bytes(st.get(name)?))
}

/// BF16 [n][k] -> FP8 (OCP e4m3 bytes) with E8M0 scales per 32x32 block (pow2_ceil(amax / 448)), tiled as up_fp8
pub fn up_bf16_as_fp8(g: &Gpu, st: &SafeTensors, name: &str) -> Result<DMat> {
    let (n, k) = shape2(st, name)?;
    let w = bf16_to_f32(st.bytes(st.get(name)?));
    let (nb, kb) = (n / 32, k / 32);
    let mut sc = vec![0u8; nb * kb];
    sc.par_chunks_mut(kb).enumerate().for_each(|(rb, srow)| {
        for (cb, sv) in srow.iter_mut().enumerate() {
            let mut a = 0f32;
            for r in 0..32 {
                for c in 0..32 {
                    a = a.max(w[(rb * 32 + r) * k + cb * 32 + c].abs());
                }
            }
            let s = dsx_core::numerics::pow2_ceil((a / 448.0).max(f32::MIN_POSITIVE));
            *sv = ((s.to_bits() >> 23) & 0xFF) as u8;
        }
    });
    let mut q = vec![0u8; n * k];
    q.par_chunks_mut(k).enumerate().for_each(|(r, qrow)| {
        for c in 0..k {
            let s = f32::from_bits((sc[(r / 32) * kb + c / 32] as u32) << 23);
            qrow[c] = e4m3_byte(dsx_core::numerics::e4m3(w[r * k + c] / s));
        }
    });
    let t = tile_fp8(&q, n, k);
    Ok(DMat { w: g.upload(&t)?, sc: Some(g.upload(&sc)?), n, k })
}

/// OCP e4m3 byte of a value already on the e4m3 grid
fn e4m3_byte(v: f32) -> u8 {
    if v == 0.0 {
        return 0;
    }
    let s = if v < 0.0 { 0x80u8 } else { 0 };
    let a = v.abs();
    let e = a.log2().floor() as i32;
    if e < -6 {
        // subnormal: m * 2^-9
        return s | ((a / 2f32.powi(-9)).round() as u8 & 7);
    }
    let m = ((a / 2f32.powi(e) - 1.0) * 8.0).round() as i32;
    let (e, m) = if m == 8 { (e + 1, 0) } else { (e, m) };
    s | (((e + 7) as u8) << 3) | m as u8
}
