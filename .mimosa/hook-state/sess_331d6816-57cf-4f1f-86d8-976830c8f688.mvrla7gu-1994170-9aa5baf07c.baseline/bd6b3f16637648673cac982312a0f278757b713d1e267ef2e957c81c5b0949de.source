//! T16 routed experts: exllamav3's bitshift trellis with the T16 codebook (docs/t16.md).
//!
//! One expert matrix W (inputs `k`, outputs `n`) is stored as one byte slice, `trellis | suh | svh`:
//! - trellis: one circular bit stream of 256 K bits (32 K bytes) per 16 x 16 tile, each tile's bytes in stream order
//!   (byte 0 holds bits 0-7, MSB first), tiles ordered by tile column: tile (kb, nb) (16 inputs kb, 16 outputs nb) at
//!   index nb * k/16 + kb. Position `t` of a tile holds input `t % 16` and output `t / 16` of that tile (column-major
//!   positions), and its 16-bit state is the window of the stream ending after its own bits. This is the layout the
//!   CPU kernel walks; the GPU store repacks to exllamav3's own order (`from_exl3` is the conversion from it).
//! - suh: f16 `[k]`, svh: f16 `[n]`.
//!
//! The expert computes `y = had128(had128(x * suh) @ W_inner) * svh`, with orthonormal 128-wide Walsh-Hadamard blocks.
//! A weight's value is `fp16(fma(fp16(1024 + b), fp16 0x20e6, fp16 0xca1e))`, where `b` is the byte sum of
//! `low16(state * 0x3A2F)`: 0.009567260742 b - 2.4375, rounded to fp16.

use crate::gguf::GType;
use half::f16;

pub const MUL: u32 = 0x3A2F;
const K_INV: u16 = 0x20e6;
const K_BIAS: u16 = 0xca1e;

/// twice the bits per weight (3, 4, 5) of a T16 type, or None
pub fn k2(ty: GType) -> Option<usize> {
    match ty {
        GType::T16K15 => Some(3),
        GType::T16K2 => Some(4),
        GType::T16K25 => Some(5),
        _ => None,
    }
}

pub fn ty_for_k2(k2: usize) -> Option<GType> {
    match k2 {
        3 => Some(GType::T16K15),
        4 => Some(GType::T16K2),
        5 => Some(GType::T16K25),
        _ => None,
    }
}

/// bytes of the trellis of a k x n matrix
pub fn trellis_bytes(k2: usize, k: usize, n: usize) -> usize {
    assert!(k % 16 == 0 && n % 16 == 0, "T16: {k} x {n} is not a whole number of 16 x 16 tiles");
    (k / 16) * (n / 16) * 16 * k2 // 8 k2 u16 per tile
}

/// bytes of one stored matrix: trellis, then suh (f16 per input), then svh (f16 per output)
pub fn matrix_bytes(ty: GType, k: usize, n: usize) -> usize {
    let k2 = k2(ty).expect("not a T16 type");
    trellis_bytes(k2, k, n) + 2 * k + 2 * n
}

/// bytes per tile
pub fn tile_bytes(k2: usize) -> usize {
    16 * k2
}

/// exllamav3's trellis (u16 `[k/16][n/16][8 k2]` as little-endian bytes: each tile is u32 words read MSB first) to
/// dsx's stored layout (tiles by column, each tile's bytes in stream order)
pub fn from_exl3(exl3: &[u8], k2: usize, k: usize, n: usize) -> Vec<u8> {
    assert_eq!(exl3.len(), trellis_bytes(k2, k, n));
    let (kt, nt, tb) = (k / 16, n / 16, tile_bytes(k2));
    let mut out = vec![0u8; exl3.len()];
    for kb in 0..kt {
        for nb in 0..nt {
            let src = &exl3[(kb * nt + nb) * tb..][..tb];
            let dst = &mut out[(nb * kt + kb) * tb..][..tb];
            for (d, s) in dst.chunks_exact_mut(4).zip(src.chunks_exact(4)) {
                d.copy_from_slice(&[s[3], s[2], s[1], s[0]]);
            }
        }
    }
    out
}

/// the value of a 16-bit state, exactly as exllamav3's T16 decode computes it
pub fn value(state: u32) -> f32 {
    let p = (state.wrapping_mul(MUL)) & 0xFFFF;
    let b = (p & 0xFF) + (p >> 8);
    let a = 1024.0 + b as f64; // exact in fp16 (b <= 510)
    let r = a * f16::from_bits(K_INV).to_f64() + f16::from_bits(K_BIAS).to_f64(); // the fma's exact product and sum
    f16::from_f64(r).to_f32()
}

/// the 511 values a weight can take, by byte sum b
pub fn value_of_bytesum(b: u32) -> f32 {
    let a = 1024.0 + b as f64;
    f16::from_f64(a * f16::from_bits(K_INV).to_f64() + f16::from_bits(K_BIAS).to_f64()).to_f32()
}

/// end bit (exclusive) of position t's bits in a tile of k2/2 bits per weight; half-integer K gives even positions
/// k2/2 rounded down and odd positions one more
fn bit_end(k2: usize, t: usize) -> usize {
    if k2 % 2 == 1 { (t + 1) * (k2 / 2) + (t + 1) / 2 } else { (t + 1) * (k2 / 2) }
}

/// the 16-bit state of position t of one tile (`tile`: its 16 k2 bytes, stream order)
pub fn state(tile: &[u8], k2: usize, t: usize) -> u32 {
    let nb = 128 * k2;
    let b0 = (bit_end(k2, t) + nb - 16) % nb;
    let mut st = 0u32;
    for i in 0..16 {
        let b = (b0 + i) % nb;
        st = (st << 1) | ((tile[b >> 3] as u32 >> (7 - (b & 7))) & 1);
    }
    st
}

fn u16s(b: &[u8]) -> impl Iterator<Item = u16> + '_ {
    b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]]))
}

/// the inner trellis matrix as f32 `[k][n]` (row = input)
pub fn decode_inner(trellis: &[u8], k2: usize, k: usize, n: usize) -> Vec<f32> {
    assert_eq!(trellis.len(), trellis_bytes(k2, k, n));
    let tb = tile_bytes(k2);
    let (kt, nt) = (k / 16, n / 16);
    let mut w = vec![0f32; k * n];
    for ti in 0..kt {
        for tj in 0..nt {
            let tile = &trellis[(tj * kt + ti) * tb..][..tb];
            for t in 0..256 {
                let (r, c) = (t % 16, t / 16);
                w[(16 * ti + r) * n + 16 * tj + c] = value(state(tile, k2, t));
            }
        }
    }
    w
}

/// split one stored matrix into (trellis, suh, svh)
pub fn parts(bytes: &[u8], ty: GType, k: usize, n: usize) -> (&[u8], Vec<f32>, Vec<f32>) {
    let k2 = k2(ty).expect("not a T16 type");
    assert_eq!(bytes.len(), matrix_bytes(ty, k, n), "T16 {k} x {n}: stored bytes");
    let tb = trellis_bytes(k2, k, n);
    let f = |b: &[u8]| u16s(b).map(|h| f16::from_bits(h).to_f32()).collect::<Vec<f32>>();
    (&bytes[..tb], f(&bytes[tb..tb + 2 * k]), f(&bytes[tb + 2 * k..]))
}

/// the effective matrix `diag(svh) H W_inner^T H diag(suh)` as f32 `[n][k]` (row = output), so `y = W_eff x`
pub fn dequant_effective(bytes: &[u8], ty: GType, k: usize, n: usize) -> Vec<f32> {
    use crate::numerics::fwht_blocks;
    let k2 = k2(ty).unwrap();
    let (tr, suh, svh) = parts(bytes, ty, k, n);
    let inner = decode_inner(tr, k2, k, n); // [k][n]
    // M = H_k W_inner H_n as a k x n matrix: rows (inputs) then columns (outputs)
    let mut m = inner;
    // H on the output side: each row of [k][n] holds n outputs
    for row in m.chunks_exact_mut(n) {
        fwht_blocks(row);
    }
    // H on the input side: transpose, transform, keep [n][k]
    let mut e = vec![0f32; n * k];
    for i in 0..k {
        for j in 0..n {
            e[j * k + i] = m[i * n + j];
        }
    }
    for (j, row) in e.chunks_exact_mut(k).enumerate() {
        fwht_blocks(row);
        for (i, v) in row.iter_mut().enumerate() {
            *v *= suh[i] * svh[j];
        }
    }
    e
}

/// y = had128(had128(x * suh) @ W_inner) * svh, the way the expert is defined (reference, slow)
pub fn forward(bytes: &[u8], ty: GType, k: usize, n: usize, x: &[f32]) -> Vec<f32> {
    use crate::numerics::fwht_blocks;
    let k2 = k2(ty).unwrap();
    let (tr, suh, svh) = parts(bytes, ty, k, n);
    let inner = decode_inner(tr, k2, k, n);
    let mut xs: Vec<f32> = x.iter().zip(&suh).map(|(a, s)| a * s).collect();
    fwht_blocks(&mut xs);
    let mut y = vec![0f32; n];
    for i in 0..k {
        let xi = xs[i];
        for (j, yj) in y.iter_mut().enumerate() {
            *yj += xi * inner[i * n + j];
        }
    }
    fwht_blocks(&mut y);
    y.iter_mut().zip(&svh).for_each(|(v, s)| *v *= s);
    y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn values_match_the_affine_form() {
        // 0.009567260742 b - 2.4375, within half an fp16 ulp
        for b in 0..=510u32 {
            let v = value_of_bytesum(b);
            let lin = 0.009567260742f64 * b as f64 - 2.4375;
            let ulp = (f16::from_f32(v).to_f32() - f16::from_bits(f16::from_f32(v).to_bits() + 1).to_f32()).abs();
            assert!((v as f64 - lin).abs() <= ulp as f64, "b {b}: {v} vs {lin}");
        }
    }

    #[test]
    fn integer_k_positions_are_k_bits_apart() {
        // with every bit of the stream set, any window is all ones; with a single set bit at position j, exactly the
        // windows covering it see it
        let k2 = 4;
        let mut w = vec![0u8; tile_bytes(k2)];
        w[0] = 0x80; // the MSB of byte 0 is bit 0 of the stream
        let hits: Vec<usize> = (0..256).filter(|&t| state(&w, k2, t) != 0).collect();
        // bit 0 is covered by windows ending at bits 1..=16, i.e. positions whose end bit (t+1)*2 is in 1..=16
        assert_eq!(hits, (0..8).collect::<Vec<_>>());
        assert_eq!(state(&w, k2, 7), 0x8000); // the window of bits 0..16: bit 0 is its MSB
    }

    #[test]
    fn effective_matches_forward() {
        let (k, n) = (256, 128);
        for k2 in 3..=5 {
            let ty = ty_for_k2(k2).unwrap();
            let mut bytes = vec![0u8; matrix_bytes(ty, k, n)];
            let mut s = 0x9e3779b9u32;
            let tb = trellis_bytes(k2, k, n);
            for b in bytes[..tb].iter_mut() {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                *b = s as u8;
            }
            for (i, c) in bytes[tb..].chunks_exact_mut(2).enumerate() {
                let v = f16::from_f32(0.5 + (i % 7) as f32 * 0.1);
                c.copy_from_slice(&v.to_bits().to_le_bytes());
            }
            let x: Vec<f32> = (0..k).map(|i| ((i * 37 % 19) as f32 - 9.0) / 9.0).collect();
            let y = forward(&bytes, ty, k, n, &x);
            let e = dequant_effective(&bytes, ty, k, n);
            for j in 0..n {
                let ye: f32 = (0..k).map(|i| e[j * k + i] * x[i]).sum();
                assert!((ye - y[j]).abs() < 1e-3 * (1.0 + y[j].abs()), "k2 {k2} out {j}: {ye} vs {}", y[j]);
            }
        }
    }
}
