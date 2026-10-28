//! Dense weights requantized to 6 bits at load (Q8_0 -> ggml's Q6_K algorithm), stored in planes the GPU reads
//! with aligned vector loads: per row, low nibbles [k/2], high 2-bit codes [k/4], int8 sub-scales per 16 [k/16],
//! fp16 super-scale per 256 [k/256]. 6.56 bits/weight, like Q6_K.
//!
//! Code order (`put` / `code`): each 32-bit word of `lo` holds 8 weights, byte j = weights j (low nibble) and j + 4
//! (high nibble); each word of `hi` holds 16, byte j = the high 2 bits of weights j, j + 4, j + 8, j + 12 (bits 0, 2,
//! 4, 6). So `lo_word & 0x0f0f0f0f | (hi_word << 4) & 0x30303030` is 4 whole codes in byte order: the GPU builds 4
//! codes in 3 instructions and turns each into a float with a byte permute instead of an int conversion.

/// store 6-bit code c of weight g (lo, hi zeroed first)
#[inline]
pub fn put(lo: &mut [u8], hi: &mut [u8], g: usize, c: u8) {
    lo[(g >> 3) * 4 + (g & 3)] |= (c & 15) << (4 * ((g >> 2) & 1));
    hi[(g >> 4) * 4 + (g & 3)] |= (c >> 4) << (2 * ((g >> 2) & 3));
}

/// 6-bit code of weight g
#[inline]
pub fn code(lo: &[u8], hi: &[u8], g: usize) -> u8 {
    ((lo[(g >> 3) * 4 + (g & 3)] >> (4 * ((g >> 2) & 1))) & 15) | (((hi[(g >> 4) * 4 + (g & 3)] >> (2 * ((g >> 2) & 3))) & 3) << 4)
}

fn nearest_int(f: f32) -> i32 {
    f.round_ties_even() as i32
}

/// ggml make_qx_quants(n, nmax, x, L, rmse_type = 1, NULL)
fn make_qx_quants(x: &[f32], nmax: i32, l: &mut [i8]) -> f32 {
    let (mut max, mut amax) = (0f32, 0f32);
    for &v in x {
        if v.abs() > amax {
            amax = v.abs();
            max = v;
        }
    }
    if amax < 1e-15 {
        l.iter_mut().for_each(|q| *q = 0);
        return 0.0;
    }
    let quant = |iscale: f32, l: &mut [i8]| -> (f32, f32) {
        let (mut sumlx, mut suml2) = (0f32, 0f32);
        for (i, &v) in x.iter().enumerate() {
            let q = nearest_int(iscale * v).clamp(-nmax, nmax - 1);
            l[i] = (q + nmax) as i8;
            let w = v * v;
            sumlx += w * v * q as f32;
            suml2 += w * (q * q) as f32;
        }
        (sumlx, suml2)
    };
    let mut iscale = -(nmax as f32) / max;
    let (sumlx, suml2) = quant(iscale, l);
    let mut scale = if suml2 != 0.0 { sumlx / suml2 } else { 0.0 };
    let mut best = scale * sumlx;
    let mut tmp = vec![0i8; x.len()];
    for is in -9..=9 {
        if is == 0 {
            continue;
        }
        iscale = -(nmax as f32 + 0.1 * is as f32) / max;
        let (slx, sl2) = quant(iscale, &mut tmp);
        if sl2 > 0.0 && slx * slx > best * sl2 {
            l.copy_from_slice(&tmp);
            scale = slx / sl2;
            best = scale * slx;
        }
    }
    scale
}

/// One row (k % 256 == 0) -> (lo nibbles, hi 2-bit codes, sub-scales, super-scales f16 bits)
pub fn quantize_row(x: &[f32], lo: &mut [u8], hi: &mut [u8], sc: &mut [i8], d: &mut [u16]) {
    let k = x.len();
    let mut l = [0i8; 256];
    let mut scales = [0f32; 16];
    for b in 0..k / 256 {
        let xb = &x[b * 256..(b + 1) * 256];
        let (mut max_scale, mut max_abs) = (0f32, 0f32);
        for ib in 0..16 {
            let s = make_qx_quants(&xb[16 * ib..16 * ib + 16], 32, &mut l[16 * ib..16 * ib + 16]);
            scales[ib] = s;
            if s.abs() > max_abs {
                max_abs = s.abs();
                max_scale = s;
            }
        }
        let mut codes = [32u8; 256];
        if max_abs < 1e-15 {
            d[b] = 0;
            for ib in 0..16 {
                sc[b * 16 + ib] = 0;
            }
        } else {
            let iscale = -128.0 / max_scale;
            let dh = half::f16::from_f32(1.0 / iscale);
            d[b] = dh.to_bits();
            for ib in 0..16 {
                sc[b * 16 + ib] = nearest_int(iscale * scales[ib]).min(127) as i8;
            }
            for j in 0..16 {
                let dd = dh.to_f32() * sc[b * 16 + j] as f32;
                if dd == 0.0 {
                    continue;
                }
                for ii in 0..16 {
                    let q = nearest_int(xb[16 * j + ii] / dd).clamp(-32, 31);
                    codes[16 * j + ii] = (q + 32) as u8;
                }
            }
        }
        lo[b * 128..b * 128 + 128].fill(0);
        hi[b * 64..b * 64 + 64].fill(0);
        for (i, &c) in codes.iter().enumerate() {
            put(lo, hi, b * 256 + i, c);
        }
    }
}

/// One row of llama.cpp Q6_K blocks (k % 256 == 0) -> the same planes, exactly: Q6_K stores code q (0..63, value
/// d * scale * (q - 32)) as low nibbles in ql (weights l, l + 32 of each 128-half; high nibbles weights l + 64, l + 96)
/// and 2-bit pairs in qh, int8 scales per 16 consecutive weights, fp16 d per 256.
pub fn from_q6k_row(raw: &[u8], lo: &mut [u8], hi: &mut [u8], sc: &mut [i8], d: &mut [u16]) {
    lo.fill(0);
    hi.fill(0);
    for (b, blk) in raw.chunks_exact(210).enumerate() {
        let (ql, qh) = (&blk[0..128], &blk[128..192]);
        for (i, s) in blk[192..208].iter().enumerate() {
            sc[b * 16 + i] = *s as i8;
        }
        d[b] = u16::from_le_bytes([blk[208], blk[209]]);
        for half in 0..2 {
            let (ql, qh) = (&ql[64 * half..], &qh[32 * half..]);
            for l in 0..32 {
                let codes = [
                    (ql[l] & 0xF) | ((qh[l] & 3) << 4),
                    (ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4),
                    (ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4),
                    (ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4),
                ];
                for (m, &c) in codes.iter().enumerate() {
                    let g = b * 256 + 128 * half + 32 * m + l;
                    put(lo, hi, g, c);
                }
            }
        }
    }
}

/// One row of llama.cpp Q5_K blocks (k % 256 == 0) -> 5-bit planes, exactly: lo nibbles (k/2), the 5th bit (k/8,
/// bit g % 8 of byte g / 8), a (scale, min) byte pair per 32 weights (k/16) and the fp16 (d, dmin) pair per 256 (k/256 u16
/// pairs); weight = d * scale * code - dmin * min, as ggml's dequantize_row_q5_K.
pub fn from_q5k_row(raw: &[u8], lo: &mut [u8], hb: &mut [u8], sm: &mut [u8], dd: &mut [u16]) {
    lo.fill(0);
    hb.fill(0);
    for (b, blk) in raw.chunks_exact(176).enumerate() {
        dd[2 * b] = u16::from_le_bytes([blk[0], blk[1]]);
        dd[2 * b + 1] = u16::from_le_bytes([blk[2], blk[3]]);
        let (sc, qh, qs) = (&blk[4..16], &blk[16..48], &blk[48..176]);
        for j in 0..8 {
            let (s, m) = if j < 4 { (sc[j] & 63, sc[j + 4] & 63) } else { ((sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4), (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4)) };
            sm[2 * (b * 8 + j)] = s;
            sm[2 * (b * 8 + j) + 1] = m;
        }
        for c in 0..4 {
            for l in 0..32 {
                let codes = [(qs[32 * c + l] & 0xF) | (((qh[l] >> (2 * c)) & 1) << 4), (qs[32 * c + l] >> 4) | (((qh[l] >> (2 * c + 1)) & 1) << 4)];
                for (h, &q) in codes.iter().enumerate() {
                    let g = b * 256 + 64 * c + 32 * h + l;
                    lo[g / 2] |= (q & 15) << (4 * (g % 2));
                    hb[g / 8] |= (q >> 4) << (g % 8);
                }
            }
        }
    }
}

/// Decoded weight of the 5-bit planes (reference for tests)
pub fn dequant5(lo: &[u8], hb: &[u8], sm: &[u8], dd: &[u16], g: usize) -> f32 {
    let q = ((lo[g / 2] >> (4 * (g % 2))) & 15) | (((hb[g / 8] >> (g % 8)) & 1) << 4);
    let (d, dmin) = (half::f16::from_bits(dd[g / 256 * 2]).to_f32(), half::f16::from_bits(dd[g / 256 * 2 + 1]).to_f32());
    d * sm[g / 32 * 2] as f32 * q as f32 - dmin * sm[g / 32 * 2 + 1] as f32
}

/// Decoded weight (reference for tests and the CPU side)
pub fn dequant(lo: &[u8], hi: &[u8], sc: &[i8], d: &[u16], g: usize) -> f32 {
    let c = code(lo, hi, g);
    half::f16::from_bits(d[g / 256]).to_f32() * sc[g / 16] as f32 * (c as i32 - 32) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q6k_repack_is_exact() {
        // random Q6_K blocks: the planes decode to exactly what ggml's dequantize_row_q6_K gives
        let k = 512;
        let mut seed = 0x2545f4914f6cdd1du64;
        let mut raw: Vec<u8> = (0..k / 256 * 210)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                seed as u8
            })
            .collect();
        for b in 0..k / 256 {
            raw[b * 210 + 208..b * 210 + 210].copy_from_slice(&half::f16::from_f32(0.01).to_le_bytes());
        }
        let (mut lo, mut hi, mut sc, mut d) = (vec![0u8; k / 2], vec![0u8; k / 4], vec![0i8; k / 16], vec![0u16; k / 256]);
        from_q6k_row(&raw, &mut lo, &mut hi, &mut sc, &mut d);
        let mut want = vec![0f32; k];
        crate::formats::dequant_row(crate::gguf::GType::Q6K, &raw, &mut want);
        for g in 0..k {
            assert_eq!(dequant(&lo, &hi, &sc, &d, g), want[g], "weight {g}");
        }
    }

    #[test]
    fn q5k_repack_is_exact() {
        let k = 512;
        let mut seed = 0x9e3779b97f4a7c15u64;
        let mut raw: Vec<u8> = (0..k / 256 * 176)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                seed as u8
            })
            .collect();
        for b in 0..k / 256 {
            raw[b * 176..b * 176 + 2].copy_from_slice(&half::f16::from_f32(0.01).to_le_bytes());
            raw[b * 176 + 2..b * 176 + 4].copy_from_slice(&half::f16::from_f32(0.003).to_le_bytes());
        }
        let (mut lo, mut hb, mut sm, mut dd) = (vec![0u8; k / 2], vec![0u8; k / 8], vec![0u8; k / 16], vec![0u16; k / 128]);
        from_q5k_row(&raw, &mut lo, &mut hb, &mut sm, &mut dd);
        let mut want = vec![0f32; k];
        crate::formats::dequant_row(crate::gguf::GType::Q5K, &raw, &mut want);
        for g in 0..k {
            assert_eq!(dequant5(&lo, &hb, &sm, &dd, g), want[g], "weight {g}");
        }
    }

    #[test]
    fn q6_roundtrip_error() {
        let k = 1024;
        let x: Vec<f32> = (0..k).map(|i| ((i * 7919 % 1000) as f32 / 500.0 - 1.0) * if i % 97 == 0 { 8.0 } else { 1.0 }).collect();
        let (mut lo, mut hi, mut sc, mut d) = (vec![0u8; k / 2], vec![0u8; k / 4], vec![0i8; k / 16], vec![0u16; k / 256]);
        quantize_row(&x, &mut lo, &mut hi, &mut sc, &mut d);
        let rms = (x.iter().map(|v| v * v).sum::<f32>() / k as f32).sqrt();
        let err = (0..k).map(|g| (dequant(&lo, &hi, &sc, &d, g) - x[g]).powi(2)).sum::<f32>() / k as f32;
        let rel = err.sqrt() / rms;
        assert!(rel < 0.03, "relative rms error {rel}");
    }
}
