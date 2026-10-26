//! CPU expert matvec: int8 activations (256-blocks) x GGUF weight rows, AVX2 with AVX-VNNI, AVX-512 VNNI or neither
//! (`isa`, picked once at run time).
//! Rows are read once and used for up to 8 token vectors (verify batches).
//!
//! Per-format activation layouts (built once per token, so weights need no shuffles):
//!   Q2_0  : per 128 (two blocks) byte 32k + 16j + m = q[64j + 4m + k], so bit-plane k of the
//!           two blocks' qs (loaded as one ymm) lines up with plane k of the activations
//!   PTQ1_0: per 128-block 160 bytes, plane n (32 bytes) = q at the weights of trit n of block bytes 0..32
//!           (bytes 26..32 and the 5th trit of qh are zero slots)
//!   Q3_K, Q8_0, IQ2_XXS, IQ2_XS, IQ2_S, IQ1_S: natural order
//! The offset formats (code - 1, trit - 1, Q3_K's -4) get the subtrahend from per-dword sums (`q2n`, `t1n`,
//! stored negated so they seed the accumulator) or per-16 sums (`s16`).

use crate::formats::dequant_row;
use crate::gguf::GType;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub const QK: usize = 256;
/// tokens per kernel pass (larger batches are split)
pub const MAX_T: usize = 8;

const fn q2_idx() -> [u8; 128] {
    let mut t = [0u8; 128];
    let mut k = 0;
    while k < 4 {
        let mut j = 0;
        while j < 2 {
            let mut m = 0;
            while m < 16 {
                t[32 * k + 16 * j + m] = (64 * j + 4 * m + k) as u8;
                m += 1;
            }
            j += 1;
        }
        k += 1;
    }
    t
}
const fn t1_idx() -> [u8; 160] {
    let mut t = [255u8; 160];
    let mut n = 0;
    while n < 5 {
        let mut j = 0;
        while j < 32 {
            t[32 * n + j] = if j < 16 {
                (16 * n + j) as u8
            } else if j < 24 {
                (80 + 8 * n + j - 16) as u8
            } else if j < 26 && n < 4 {
                (120 + 2 * n + j - 24) as u8
            } else {
                255
            };
            j += 1;
        }
        n += 1;
    }
    t
}
static Q2_IDX: [u8; 128] = q2_idx();
static T1_IDX: [u8; 160] = t1_idx();
/// IQ2_XXS codebook rows with their signs applied: entry 128 * grid + pattern holds ggml's iq2xxs_grid[grid] as i8,
/// negated where ksigns_iq2xs[pattern] has the weight's bit set (256 KB, built once).
fn iq2_signed() -> &'static [u64] {
    static T: std::sync::OnceLock<Vec<u64>> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let mut t = vec![0u64; 256 * 128];
        for (gi, &g) in crate::iqgrid::IQ2XXS_GRID.iter().enumerate() {
            for (si, &sb) in crate::iqgrid::KSIGNS_IQ2XS.iter().enumerate() {
                let b: [u8; 8] = std::array::from_fn(|j| {
                    let m = (g >> (8 * j)) as u8 as i8;
                    (if sb >> j & 1 != 0 { -m } else { m }) as u8
                });
                t[gi << 7 | si] = u64::from_le_bytes(b);
            }
        }
        t
    })
}

const L_Q2: u8 = 1;
const L_T1: u8 = 2;

fn grow<V: Default + Clone>(v: &mut Vec<V>, len: usize) {
    if v.len() < len {
        v.resize(len, V::default())
    }
}

fn layout_bit(t: GType) -> u8 {
    match t {
        GType::Q2_0 => L_Q2,
        GType::Ptq1_0 => L_T1,
        GType::Q2K | GType::Iq3Xxs | GType::Q3K | GType::Q8_0 | GType::Iq2Xxs | GType::Iq2Xs | GType::Iq2S | GType::Iq1S => 0,
        _ => panic!("{t:?}: no CPU kernel"),
    }
}

/// T quantized activation vectors of n (multiple of 256). Buffers are reused across `reset`s.
#[derive(Default)]
pub struct QAct {
    pub t: usize,
    pub n: usize,
    layouts: u8,
    d: Vec<f32>,   // [t][n/256]
    q: Vec<i8>,    // [t][n]
    s16: Vec<i16>, // [t][n/16]
    q2: Vec<i8>,   // [t][n]
    q2n: Vec<i32>, // [t][n/128][8]
    t1: Vec<i8>,   // [t][n/128*160]
    t1n: Vec<i32>, // [t][n/128][8]
}

/// Raw view of a QAct's buffers so several threads can quantize disjoint blocks.
#[derive(Clone, Copy)]
pub struct QActRaw {
    n: usize,
    layouts: u8,
    d: *mut f32,
    q: *mut i8,
    s16: *mut i16,
    q2: *mut i8,
    q2n: *mut i32,
    t1: *mut i8,
    t1n: *mut i32,
}
unsafe impl Send for QActRaw {}
unsafe impl Sync for QActRaw {}

impl QAct {
    pub fn new(x: &[f32], t: usize, n: usize, tys: &[GType]) -> QAct {
        let mut a = QAct::default();
        a.reset(t, n, tys);
        a.quantize(x);
        a
    }

    /// Size for t vectors of n with the layouts the weight types `tys` need.
    pub fn reset(&mut self, t: usize, n: usize, tys: &[GType]) {
        assert!(n % QK == 0, "activation length {n} is not a multiple of {QK}");
        self.t = t;
        self.n = n;
        self.layouts = tys.iter().fold(0, |a, &ty| a | layout_bit(ty));
        grow(&mut self.d, t * n / QK);
        grow(&mut self.q, t * n);
        grow(&mut self.s16, t * n / 16);
        if self.layouts & L_Q2 != 0 {
            grow(&mut self.q2, t * n);
            grow(&mut self.q2n, t * n / 16);
        }
        if self.layouts & L_T1 != 0 {
            grow(&mut self.t1, t * n / 128 * 160);
            grow(&mut self.t1n, t * n / 16);
        }
    }

    pub fn has(&self, ty: GType) -> bool {
        let b = layout_bit(ty);
        self.layouts & b == b
    }

    /// x: [t][n]
    pub fn quantize(&mut self, x: &[f32]) {
        assert_eq!(x.len(), self.t * self.n);
        let r = self.raw();
        for tok in 0..self.t {
            for b in 0..self.n / QK {
                unsafe { r.quantize_block(tok, b, &x[tok * self.n + b * QK..][..QK]) };
            }
        }
    }

    pub fn raw(&mut self) -> QActRaw {
        QActRaw {
            n: self.n,
            layouts: self.layouts,
            d: self.d.as_mut_ptr(),
            q: self.q.as_mut_ptr(),
            s16: self.s16.as_mut_ptr(),
            q2: self.q2.as_mut_ptr(),
            q2n: self.q2n.as_mut_ptr(),
            t1: self.t1.as_mut_ptr(),
            t1n: self.t1n.as_mut_ptr(),
        }
    }

    pub fn scale(&self, tok: usize, blk: usize) -> f32 {
        self.d[tok * self.n / QK + blk]
    }
    pub fn qs(&self, tok: usize) -> &[i8] {
        &self.q[tok * self.n..][..self.n]
    }
    /// Dequantized vector (q * d).
    pub fn dequant(&self, tok: usize) -> Vec<f32> {
        self.qs(tok)
            .iter()
            .enumerate()
            .map(|(i, &q)| q as f32 * self.scale(tok, i / QK))
            .collect()
    }

    fn tok(&self, tok: usize, ty: GType) -> Tok {
        assert!(self.has(ty), "activation lacks the {ty:?} layout");
        let n = self.n;
        unsafe {
            match ty {
                GType::Q2_0 => Tok {
                    x: self.q2.as_ptr().add(tok * n),
                    o: self.q2n.as_ptr().add(tok * n / 16),
                    s: std::ptr::null(),
                    d: self.d.as_ptr().add(tok * n / QK),
                },
                GType::Ptq1_0 => Tok {
                    x: self.t1.as_ptr().add(tok * n / 128 * 160),
                    o: self.t1n.as_ptr().add(tok * n / 16),
                    s: std::ptr::null(),
                    d: self.d.as_ptr().add(tok * n / QK),
                },
                _ => Tok {
                    x: self.q.as_ptr().add(tok * n),
                    o: std::ptr::null(),
                    s: self.s16.as_ptr().add(tok * n / 16),
                    d: self.d.as_ptr().add(tok * n / QK),
                },
            }
        }
    }
}

impl QActRaw {
    /// Quantize 256 values into block `blk` of token `tok`: d = amax / 127, q = round(x / d) (ties to even).
    /// Safety: buffers sized by `reset`; no other thread touches this (tok, blk).
    pub unsafe fn quantize_block(&self, tok: usize, blk: usize, x: &[f32]) {
        let x: &[f32; QK] = x.try_into().unwrap();
        #[cfg(target_arch = "x86_64")]
        match isa() {
            Isa::Vnni => return unsafe { vnni::quantize_block(self, tok, blk, x) },
            Isa::Vnni512 => return unsafe { vnni512::quantize_block(self, tok, blk, x) },
            Isa::Avx2 => return unsafe { avx2::quantize_block(self, tok, blk, x) },
            Isa::Scalar => {}
        }
        unsafe { self.quantize_block_scalar(tok, blk, x) }
    }

    unsafe fn quantize_block_scalar(&self, tok: usize, blk: usize, x: &[f32; QK]) {
        let amax = x.iter().fold(0f32, |a, &v| a.max(v.abs()));
        let d = amax / 127.0;
        let id = if d > 0.0 { 1.0 / d } else { 0.0 };
        let mut q = [0i8; QK];
        for (q, &v) in q.iter_mut().zip(x) {
            *q = (v * id).round_ties_even() as i8;
        }
        let n = self.n;
        let e0 = tok * n + blk * QK; // element offset
        unsafe {
            *self.d.add(e0 / QK) = d;
            std::ptr::copy_nonoverlapping(q.as_ptr(), self.q.add(e0), QK);
            for g in 0..16 {
                *self.s16.add(e0 / 16 + g) = q[16 * g..16 * g + 16].iter().map(|&v| v as i16).sum();
            }
            if self.layouts & L_Q2 != 0 {
                for h in 0..2 {
                    let src = &q[128 * h..128 * h + 128];
                    let dst = std::slice::from_raw_parts_mut(self.q2.add(e0 + 128 * h), 128);
                    for (o, &i) in dst.iter_mut().zip(&Q2_IDX) {
                        *o = src[i as usize];
                    }
                    let neg = std::slice::from_raw_parts_mut(self.q2n.add((e0 + 128 * h) / 16), 8);
                    dword_neg_sums(dst, neg);
                }
            }
            if self.layouts & L_T1 != 0 {
                for h in 0..2 {
                    let src = &q[128 * h..128 * h + 128];
                    let b128 = (e0 + 128 * h) / 128;
                    let dst = std::slice::from_raw_parts_mut(self.t1.add(b128 * 160), 160);
                    for (o, &i) in dst.iter_mut().zip(&T1_IDX) {
                        *o = if i == 255 { 0 } else { src[i as usize] };
                    }
                    let neg = std::slice::from_raw_parts_mut(self.t1n.add(b128 * 8), 8);
                    dword_neg_sums(dst, neg);
                }
            }
        }
    }
}

/// neg[l] = -(sum of bytes 4l..4l+4 of every 32-byte plane)
fn dword_neg_sums(planes: &[i8], neg: &mut [i32]) {
    neg.fill(0);
    for p in planes.chunks_exact(32) {
        for (l, o) in neg.iter_mut().enumerate() {
            *o -= p[4 * l..4 * l + 4].iter().map(|&v| v as i32).sum::<i32>();
        }
    }
}

#[derive(Clone, Copy)]
struct Tok {
    x: *const i8,
    o: *const i32,
    s: *const i16,
    d: *const f32,
}

/// The CPU kernels' instruction set.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Isa {
    /// AVX2 + AVX-VNNI (Alder Lake and later desktops)
    Vnni,
    /// AVX2 + AVX-512 VNNI on ymm (Ice Lake and later Xeons, Zen 4/5)
    Vnni512,
    /// AVX2 alone (Zen 2/3, Haswell to Comet Lake)
    Avx2,
    Scalar,
}

impl Isa {
    pub fn supported(self) -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            let base = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") && is_x86_feature_detected!("f16c");
            match self {
                Isa::Vnni => base && is_x86_feature_detected!("avxvnni") && is_x86_feature_detected!("bmi2"),
                Isa::Vnni512 => base && is_x86_feature_detected!("avx512vnni") && is_x86_feature_detected!("avx512vl") && is_x86_feature_detected!("bmi2"),
                Isa::Avx2 => base,
                Isa::Scalar => true,
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        matches!(self, Isa::Scalar)
    }
}

/// The best supported instruction set; DSX_CPU_ISA=vnni|avx512vnni|avx2|scalar picks a lower one.
pub fn isa() -> Isa {
    use std::sync::OnceLock;
    static ISA: OnceLock<Isa> = OnceLock::new();
    *ISA.get_or_init(|| {
        let all = [Isa::Vnni, Isa::Vnni512, Isa::Avx2, Isa::Scalar];
        let want = match std::env::var("DSX_CPU_ISA").as_deref() {
            Ok("vnni") => Some(Isa::Vnni),
            Ok("avx512vnni") => Some(Isa::Vnni512),
            Ok("avx2") => Some(Isa::Avx2),
            Ok("scalar") => Some(Isa::Scalar),
            _ => None,
        };
        let isa = want.filter(|i| i.supported()).unwrap_or_else(|| all.into_iter().find(|i| i.supported()).unwrap());
        if isa != Isa::Vnni {
            eprintln!("cpu kernels: {isa:?}");
        }
        isa
    })
}

pub fn simd_ok() -> bool {
    isa() != Isa::Scalar
}

/// out[j][r] = row r . token j, for all act.t tokens; rows = contiguous rows of `cols` weights.
pub fn dot_rows(ty: GType, rows: &[u8], cols: usize, act: &QAct, out: &mut [f32]) {
    let nr = rows.len() / ty.row_bytes(cols);
    let toks: Vec<usize> = (0..act.t).collect();
    dot_rows_sel(ty, rows, cols, act, &toks, out, nr);
}

/// out[j * ostride + r] = row r . token toks[j].
pub fn dot_rows_sel(ty: GType, rows: &[u8], cols: usize, act: &QAct, toks: &[usize], out: &mut [f32], ostride: usize) {
    dot_rows_isa(isa(), ty, rows, cols, act, toks, out, ostride)
}

#[allow(clippy::too_many_arguments)]
fn dot_rows_isa(isa: Isa, ty: GType, rows: &[u8], cols: usize, act: &QAct, toks: &[usize], out: &mut [f32], ostride: usize) {
    let rb = ty.row_bytes(cols);
    assert!(rows.len() % rb == 0 && act.n == cols);
    let nr = rows.len() / rb;
    if nr == 0 || toks.is_empty() {
        return;
    }
    assert!(out.len() >= (toks.len() - 1) * ostride + nr && ostride >= nr);
    if isa == Isa::Scalar {
        return dot_rows_scalar(ty, rows, cols, act, toks, out, ostride);
    }
    #[cfg(target_arch = "x86_64")]
    for (c, tc) in toks.chunks(MAX_T).enumerate() {
        let mut tp = [Tok {
            x: std::ptr::null(),
            o: std::ptr::null(),
            s: std::ptr::null(),
            d: std::ptr::null(),
        }; MAX_T];
        for (p, &t) in tp.iter_mut().zip(tc) {
            *p = act.tok(t, ty);
        }
        let o = out[c * MAX_T * ostride..].as_mut_ptr();
        macro_rules! run {
            ($m:ident, $f:ident) => {
                unsafe {
                    match tc.len() {
                        1 => $m::$f::<1>(rows, nr, cols, &tp, o, ostride),
                        2 => $m::$f::<2>(rows, nr, cols, &tp, o, ostride),
                        3 => $m::$f::<3>(rows, nr, cols, &tp, o, ostride),
                        4 => $m::$f::<4>(rows, nr, cols, &tp, o, ostride),
                        5 => $m::$f::<5>(rows, nr, cols, &tp, o, ostride),
                        6 => $m::$f::<6>(rows, nr, cols, &tp, o, ostride),
                        7 => $m::$f::<7>(rows, nr, cols, &tp, o, ostride),
                        _ => $m::$f::<8>(rows, nr, cols, &tp, o, ostride),
                    }
                }
            };
        }
        macro_rules! go {
            ($f:ident) => {
                match isa {
                    Isa::Vnni => run!(vnni, $f),
                    Isa::Vnni512 => run!(vnni512, $f),
                    _ => run!(avx2, $f),
                }
            };
        }
        match ty {
            GType::Q2_0 => go!(rows_q2_0),
            GType::Ptq1_0 => go!(rows_ptq1_0),
            GType::Q2K => go!(rows_q2_k),
            GType::Iq3Xxs => go!(rows_iq3_xxs),
            GType::Q3K => go!(rows_q3_k),
            GType::Q8_0 => go!(rows_q8_0),
            GType::Iq2Xxs => go!(rows_iq2_xxs),
            GType::Iq1S => go!(rows_iq1_s),
            GType::Iq2Xs => go!(rows_iq2_xs),
            GType::Iq2S => go!(rows_iq2_s),
            _ => panic!("{ty:?}: no CPU kernel"),
        }
    }
}

/// Reference: formats::dequant_row weights x the int8 activations, exact per weight block, f64 across blocks.
pub fn dot_rows_scalar(
    ty: GType,
    rows: &[u8],
    cols: usize,
    act: &QAct,
    toks: &[usize],
    out: &mut [f32],
    ostride: usize,
) {
    let rb = ty.row_bytes(cols);
    let wb = ty.block().0.min(QK); // weight blocks never straddle a 256-block
    let mut w = vec![0f32; cols];
    for (r, row) in rows.chunks_exact(rb).enumerate() {
        dequant_row(ty, row, &mut w);
        for (j, &t) in toks.iter().enumerate() {
            let q = act.qs(t);
            let mut acc = 0f64;
            for b in 0..cols / wb {
                // code * d * q is exact in f32 and so is the block sum (multiples of d, < 2^24 units)
                let s: f32 = (b * wb..(b + 1) * wb).map(|i| w[i] * q[i] as f32).sum();
                acc += s as f64 * act.scale(t, b * wb / QK) as f64;
            }
            out[j * ostride + r] = acc as f32;
        }
    }
}

#[inline(always)]
fn q3k_scales(raw: &[u8]) -> [i8; 16] {
    let a = |i: usize| u32::from_le_bytes(raw[4 * i..4 * i + 4].try_into().unwrap());
    let (a0, a1, tmp) = (a(0), a(1), a(2));
    let (k1, k2) = (0x0303_0303u32, 0x0f0f_0f0fu32);
    let s = [
        (a0 & k2) | ((tmp & k1) << 4),
        (a1 & k2) | (((tmp >> 2) & k1) << 4),
        ((a0 >> 4) & k2) | (((tmp >> 4) & k1) << 4),
        ((a1 >> 4) & k2) | (((tmp >> 6) & k1) << 4),
    ];
    let mut o = [0i8; 16];
    for i in 0..16 {
        o[i] = (s[i / 4] >> (8 * (i % 4))) as u8 as i8 - 32;
    }
    o
}

/// The SIMD kernels, compiled once per instruction set (`isa`). They differ only in `dp`, the u8 x i8 -> i32
/// dot product of four byte pairs added to s: AVX-VNNI's and AVX-512 VNNI's vpdpbusd, or AVX2's
/// vpmaddubsw + vpmaddwd. The AVX2 form is exact here because no byte pair sums past i16: the largest is
/// Q8_0's 2 * 128 * 127 = 32512.
macro_rules! isa_kernels {
    ($m:ident, $feat:literal, $pdep:literal, |$s:ident, $a:ident, $b:ident| $dp:expr) => {
        #[cfg(target_arch = "x86_64")]
        mod $m {
            use super::*;

            #[inline]
            #[target_feature(enable = $feat)]
            unsafe fn dp($s: __m256i, $a: __m256i, $b: __m256i) -> __m256i {
                $dp
            }

            #[target_feature(enable = $feat)]
            pub(super) unsafe fn quantize_block(r: &QActRaw, tok: usize, blk: usize, x: &[f32; QK]) {
                let xp = x.as_ptr();
                let mut m = _mm256_setzero_ps();
                let sign = _mm256_set1_ps(-0.0);
                for i in 0..QK / 8 {
                    m = _mm256_max_ps(m, _mm256_andnot_ps(sign, unsafe { _mm256_loadu_ps(xp.add(8 * i)) }));
                }
                let m4 = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps(m, 1));
                let m2 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
                let amax = _mm_cvtss_f32(_mm_max_ss(m2, _mm_movehdup_ps(m2)));
                let d = amax / 127.0;
                let id = _mm256_set1_ps(if d > 0.0 { 1.0 / d } else { 0.0 });
                let e0 = tok * r.n + blk * QK;
                let ones = _mm256_set1_epi8(1);
                unsafe {
                    *r.d.add(e0 / QK) = d;
                    let q = r.q.add(e0);
                    let fix = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
                    for g in 0..QK / 32 {
                        // cvtps rounds to nearest even (MXCSR default), like round_ties_even
                        let c = |k: usize| _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(xp.add(32 * g + 8 * k)), id));
                        let ab = _mm256_packs_epi32(c(0), c(1));
                        let cd = _mm256_packs_epi32(c(2), c(3));
                        let v = _mm256_permutevar8x32_epi32(_mm256_packs_epi16(ab, cd), fix);
                        _mm256_storeu_si256(q.add(32 * g) as *mut __m256i, v);
                        // per-16 sums: sad of q + 128 over 8 bytes
                        let sad = _mm256_sad_epu8(_mm256_xor_si256(v, _mm256_set1_epi8(-128)), _mm256_setzero_si256());
                        let mut e = [0u64; 4];
                        _mm256_storeu_si256(e.as_mut_ptr() as *mut __m256i, sad);
                        *r.s16.add(e0 / 16 + 2 * g) = (e[0] + e[1]) as i16 - 2048;
                        *r.s16.add(e0 / 16 + 2 * g + 1) = (e[2] + e[3]) as i16 - 2048;
                    }
                    if r.layouts & L_Q2 != 0 {
                        // per 128: 4x4 byte transpose in each 16-byte group, then a 4x4 dword transpose
                        let t4 = _mm256_setr_epi8(
                            0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7,
                            11, 15,
                        );
                        for h in 0..2 {
                            let src = q.add(128 * h);
                            let v: [__m256i; 4] = std::array::from_fn(|g| {
                                let lo = _mm_loadu_si128(src.add(16 * g) as *const __m128i);
                                let hi = _mm_loadu_si128(src.add(64 + 16 * g) as *const __m128i);
                                _mm256_shuffle_epi8(_mm256_set_m128i(hi, lo), t4)
                            });
                            let t0 = _mm256_unpacklo_epi32(v[0], v[1]);
                            let t1 = _mm256_unpacklo_epi32(v[2], v[3]);
                            let t2 = _mm256_unpackhi_epi32(v[0], v[1]);
                            let t3 = _mm256_unpackhi_epi32(v[2], v[3]);
                            let p = [
                                _mm256_unpacklo_epi64(t0, t1),
                                _mm256_unpackhi_epi64(t0, t1),
                                _mm256_unpacklo_epi64(t2, t3),
                                _mm256_unpackhi_epi64(t2, t3),
                            ];
                            let dst = r.q2.add(e0 + 128 * h);
                            let mut s = _mm256_setzero_si256();
                            for k in 0..4 {
                                _mm256_storeu_si256(dst.add(32 * k) as *mut __m256i, p[k]);
                                s = dp(s, ones, p[k]);
                            }
                            let neg = _mm256_sub_epi32(_mm256_setzero_si256(), s);
                            _mm256_storeu_si256(r.q2n.add((e0 + 128 * h) / 16) as *mut __m256i, neg);
                        }
                    }
                    if r.layouts & L_T1 != 0 {
                        for h in 0..2 {
                            let src = q.add(128 * h);
                            let b128 = (e0 + 128 * h) / 128;
                            let dst = r.t1.add(b128 * 160);
                            let mut s = _mm256_setzero_si256();
                            for n in 0..5 {
                                let lo = _mm_loadu_si128(src.add(16 * n) as *const __m128i);
                                let mut hi = _mm_loadl_epi64(src.add(80 + 8 * n) as *const __m128i);
                                if n < 4 {
                                    let w = std::ptr::read_unaligned(src.add(120 + 2 * n) as *const i16);
                                    hi = _mm_insert_epi16(hi, w as i32, 4);
                                }
                                let p = _mm256_set_m128i(hi, lo);
                                _mm256_storeu_si256(dst.add(32 * n) as *mut __m256i, p);
                                s = dp(s, ones, p);
                            }
                            let neg = _mm256_sub_epi32(_mm256_setzero_si256(), s);
                            _mm256_storeu_si256(r.t1n.add(b128 * 8) as *mut __m256i, neg);
                        }
                    }
                }
            }

            #[inline]
            #[target_feature(enable = $feat)]
            unsafe fn hsum(v: __m256) -> f32 {
                let a = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
                let a = _mm_add_ps(a, _mm_movehl_ps(a, a));
                _mm_cvtss_f32(_mm_add_ss(a, _mm_movehdup_ps(a)))
            }

            #[inline]
            #[target_feature(enable = $feat)]
            unsafe fn f16_ld(p: *const u8) -> f32 {
                let bits = unsafe { std::ptr::read_unaligned(p as *const u16) };
                _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(bits as i32)))
            }

            #[inline]
            #[target_feature(enable = $feat)]
            unsafe fn ld(p: *const i8) -> __m256i {
                unsafe { _mm256_loadu_si256(p as *const __m256i) }
            }

            /// Q2_0: two blocks per step; qs of both in one ymm (lane = block), plane k = (v >> 2k) & 3.
            #[target_feature(enable = $feat)]
            pub(super) unsafe fn rows_q2_0<const T: usize>(w: &[u8], nr: usize, cols: usize, tp: &[Tok; MAX_T], out: *mut f32, os: usize) {
                let rb = cols / 64 * 18;
                let m3 = _mm256_set1_epi8(3);
                for r in 0..nr {
                    let row = unsafe { w.as_ptr().add(r * rb) };
                    let mut acc = [_mm256_setzero_ps(); T];
                    for p in 0..cols / 128 {
                        unsafe {
                            let b = row.add(36 * p);
                            let v = _mm256_set_m128i(
                                _mm_loadu_si128(b.add(20) as *const __m128i),
                                _mm_loadu_si128(b.add(2) as *const __m128i),
                            );
                            let c = [
                                _mm256_and_si256(v, m3),
                                _mm256_and_si256(_mm256_srli_epi16(v, 2), m3),
                                _mm256_and_si256(_mm256_srli_epi16(v, 4), m3),
                                _mm256_and_si256(_mm256_srli_epi16(v, 6), m3),
                            ];
                            let (d0, d1) = (
                                std::ptr::read_unaligned(b as *const i16),
                                std::ptr::read_unaligned(b.add(18) as *const i16),
                            );
                            let dw = _mm256_cvtph_ps(_mm_set_epi16(d1, d1, d1, d1, d0, d0, d0, d0));
                            for j in 0..T {
                                let t = &tp[j];
                                let x = t.x.add(128 * p);
                                let mut s = _mm256_loadu_si256(t.o.add(8 * p) as *const __m256i);
                                s = dp(s, c[0], ld(x));
                                s = dp(s, c[1], ld(x.add(32)));
                                s = dp(s, c[2], ld(x.add(64)));
                                s = dp(s, c[3], ld(x.add(96)));
                                let sc = _mm256_mul_ps(dw, _mm256_broadcast_ss(&*t.d.add(p / 2)));
                                acc[j] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s), sc, acc[j]);
                            }
                        }
                    }
                    for j in 0..T {
                        unsafe { *out.add(j * os + r) = hsum(acc[j]) };
                    }
                }
            }

            /// PTQ1_0: one block per step, 32-byte load (26 used); trit planes by u <- 3u (mod 256) on u = q ^ 0x80,
            /// trit = (u > -43) + (u > 42): the decoder's ((q * 3^n) & 255) * 3 >> 8 without byte multiplies.
            #[target_feature(enable = $feat)]
            pub(super) unsafe fn rows_ptq1_0<const T: usize>(w: &[u8], nr: usize, cols: usize, tp: &[Tok; MAX_T], out: *mut f32, os: usize) {
                let rb = cols / 128 * 28;
                let (k80, lo, hi) = (_mm256_set1_epi8(-128), _mm256_set1_epi8(-43), _mm256_set1_epi8(42));
                let zero = _mm256_setzero_si256();
                let mut tail = [0u8; 32];
                for r in 0..nr {
                    let mut acc = [_mm256_setzero_ps(); T];
                    for b in 0..cols / 128 {
                        let off = r * rb + 28 * b;
                        unsafe {
                            let p = if off + 32 <= w.len() {
                                w.as_ptr().add(off)
                            } else {
                                tail[..28].copy_from_slice(&w[off..off + 28]);
                                tail.as_ptr()
                            };
                            let mut u = _mm256_xor_si256(_mm256_loadu_si256(p as *const __m256i), k80);
                            let mut c = [zero; 5];
                            for cn in c.iter_mut() {
                                let (m1, m2) = (_mm256_cmpgt_epi8(u, lo), _mm256_cmpgt_epi8(u, hi));
                                *cn = _mm256_sub_epi8(_mm256_sub_epi8(zero, m1), m2);
                                u = _mm256_add_epi8(u, _mm256_add_epi8(u, u));
                                // keep LLVM from folding the chain into byte multiplies by 3^n (pmullw/pmaddubsw emulation)
                                std::arch::asm!("/* {0} */", inout(ymm_reg) u, options(pure, nomem, nostack, preserves_flags));
                            }
                            let dw = _mm256_set1_ps(f16_ld(p.add(26)));
                            for j in 0..T {
                                let t = &tp[j];
                                let x = t.x.add(160 * b);
                                let mut s = _mm256_loadu_si256(t.o.add(8 * b) as *const __m256i);
                                for n in 0..5 {
                                    s = dp(s, c[n], ld(x.add(32 * n)));
                                }
                                acc[j] = _mm256_fmadd_ps(
                                    _mm256_cvtepi32_ps(s),
                                    _mm256_mul_ps(dw, _mm256_broadcast_ss(&*t.d.add(b / 2))),
                                    acc[j],
                                );
                            }
                        }
                    }
                    for j in 0..T {
                        unsafe { *out.add(j * os + r) = hsum(acc[j]) };
                    }
                }
            }


            /// Q2_K: exact unsigned scaled codes <=45, plus the separate minimum contribution.
            #[target_feature(enable = $feat)]
            pub(super) unsafe fn rows_q2_k<const T: usize>(w: &[u8], nr: usize, cols: usize, tp: &[Tok; MAX_T], out: *mut f32, os: usize) {
                let nb = cols / 256;
                for r in 0..nr {
                    let mut acc = [_mm256_setzero_ps(); T];
                    for b in 0..nb {
                        let p = &w[(r * nb + b) * 84..][..84];
                        let mut plus = [_mm256_setzero_si256(); T];
                        let mut minus = [_mm256_setzero_si256(); T];
                        for g in 0..8 {
                            let mut v = [0u8; 32];
                            let mut m = [0u8; 32];
                            for i in 0..32 {
                                let sc = p[2 * g + i / 16];
                                v[i] = ((p[16 + (g / 4) * 32 + i] >> (2 * (g % 4))) & 3) * (sc & 15);
                                m[i] = sc >> 4;
                            }
                            unsafe {
                                let vv = ld(v.as_ptr() as *const i8);
                                let mm = ld(m.as_ptr() as *const i8);
                                for j in 0..T {
                                    let x = ld(tp[j].x.add(b * 256 + g * 32));
                                    plus[j] = dp(plus[j], vv, x);
                                    minus[j] = dp(minus[j], mm, x);
                                }
                            }
                        }
                        unsafe {
                            let d = _mm256_set1_ps(f16_ld(p.as_ptr().add(80)));
                            let dm = _mm256_set1_ps(f16_ld(p.as_ptr().add(82)));
                            for j in 0..T {
                                let v = _mm256_sub_ps(_mm256_mul_ps(d, _mm256_cvtepi32_ps(plus[j])), _mm256_mul_ps(dm, _mm256_cvtepi32_ps(minus[j])));
                                acc[j] = _mm256_fmadd_ps(v, _mm256_broadcast_ss(&*tp[j].d.add(b)), acc[j]);
                            }
                        }
                    }
                    for j in 0..T { unsafe { *out.add(j * os + r) = hsum(acc[j]); } }
                }
            }

            /// IQ3_XXS: 4-weight grid entries, sign patterns, and odd integer scale per 32.
            #[target_feature(enable = $feat)]
            pub(super) unsafe fn rows_iq3_xxs<const T: usize>(w: &[u8], nr: usize, cols: usize, tp: &[Tok; MAX_T], out: *mut f32, os: usize) {
                use crate::iqgrid::{IQ3XXS_GRID, KSIGNS_IQ2XS};
                let nb = cols / 256;
                for r in 0..nr {
                    let mut acc = [_mm256_setzero_ps(); T];
                    for b in 0..nb {
                        let p = &w[(r * nb + b) * 98..][..98];
                        let mut tot = [_mm256_setzero_si256(); T];
                        for g in 0..8 {
                            let sg = u32::from_le_bytes(p[66 + 4*g..70 + 4*g].try_into().unwrap());
                            let mut v = [0i8; 32];
                            for q in 0..8 {
                                let grid = IQ3XXS_GRID[p[2 + 8*g + q] as usize].to_le_bytes();
                                let sign = KSIGNS_IQ2XS[((sg >> (7 * (q/2))) & 127) as usize];
                                for i in 0..4 { v[q*4+i] = if (sign >> (4*(q%2)+i)) & 1 != 0 { -(grid[i] as i8) } else { grid[i] as i8 }; }
                            }
                            unsafe {
                                let v = ld(v.as_ptr());
                                let av = _mm256_abs_epi8(v);
                                let sc = _mm256_set1_epi32((2 * (sg >> 28) + 1) as i32);
                                for j in 0..T {
                                    let x = _mm256_sign_epi8(ld(tp[j].x.add(b*256+g*32)), v);
                                    let dot = dp(_mm256_setzero_si256(), av, x);
                                    tot[j] = _mm256_add_epi32(tot[j], _mm256_mullo_epi32(dot, sc));
                                }
                            }
                        }
                        unsafe {
                            let dw = _mm256_set1_ps(f16_ld(p.as_ptr()) * 0.25);
                            for j in 0..T {
                                acc[j] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(tot[j]), _mm256_mul_ps(dw, _mm256_broadcast_ss(&*tp[j].d.add(b))), acc[j]);
                            }
                        }
                    }
                    for j in 0..T { unsafe { *out.add(j * os + r) = hsum(acc[j]); } }
                }
            }

            /// Q3_K: chunk (h, j) of 32 weights = 128h + 32j + l, natural order; code u = low2 | hbit << 2 in 0..7 (q = u - 4).
            /// maddubs + madd by the two 16-weight sub-scales, then -4 * sum(scale * s16) per block.
            #[target_feature(enable = $feat)]
            pub(super) unsafe fn rows_q3_k<const T: usize>(w: &[u8], nr: usize, cols: usize, tp: &[Tok; MAX_T], out: *mut f32, os: usize) {
                let rb = cols / 256 * 110;
                let (m3, m4) = (_mm256_set1_epi8(3), _mm256_set1_epi8(4));
                // lane 0 <- i16 element 2j, lane 1 <- 2j + 1
                let shuf: [__m256i; 4] = std::array::from_fn(|j| {
                    let c: [i8; 32] = std::array::from_fn(|i| (4 * j + 2 * (i / 16) + i % 2) as i8);
                    unsafe { _mm256_loadu_si256(c.as_ptr() as *const __m256i) }
                });
                for r in 0..nr {
                    let mut acc = [_mm256_setzero_ps(); T];
                    for blk in 0..cols / 256 {
                        unsafe {
                            let p = w.as_ptr().add(r * rb + 110 * blk);
                            let hm = _mm256_loadu_si256(p as *const __m256i);
                            let qs = [
                                _mm256_loadu_si256(p.add(32) as *const __m256i),
                                _mm256_loadu_si256(p.add(64) as *const __m256i),
                            ];
                            let sc8 = q3k_scales(std::slice::from_raw_parts(p.add(96), 12));
                            let sc16 = _mm256_cvtepi8_epi16(_mm_loadu_si128(sc8.as_ptr() as *const __m128i));
                            let half = [
                                _mm256_permute2x128_si256(sc16, sc16, 0x00),
                                _mm256_permute2x128_si256(sc16, sc16, 0x11),
                            ];
                            let mut u = [_mm256_setzero_si256(); 8];
                            let mut sc = [_mm256_setzero_si256(); 8];
                            for h in 0..2 {
                                for j in 0..4 {
                                    let low = match j {
                                        0 => qs[h],
                                        1 => _mm256_srli_epi16(qs[h], 2),
                                        2 => _mm256_srli_epi16(qs[h], 4),
                                        _ => _mm256_srli_epi16(qs[h], 6),
                                    };
                                    // hmask bit 4h + j -> bit 2
                                    let hb = match 4 * h + j {
                                        0 => _mm256_slli_epi16(hm, 2),
                                        1 => _mm256_slli_epi16(hm, 1),
                                        2 => hm,
                                        3 => _mm256_srli_epi16(hm, 1),
                                        4 => _mm256_srli_epi16(hm, 2),
                                        5 => _mm256_srli_epi16(hm, 3),
                                        6 => _mm256_srli_epi16(hm, 4),
                                        _ => _mm256_srli_epi16(hm, 5),
                                    };
                                    u[4 * h + j] = _mm256_or_si256(_mm256_and_si256(low, m3), _mm256_and_si256(hb, m4));
                                    sc[4 * h + j] = _mm256_shuffle_epi8(half[h], shuf[j]);
                                }
                            }
                            let dw = _mm256_set1_ps(f16_ld(p.add(108)));
                            for jt in 0..T {
                                let t = &tp[jt];
                                let x = t.x.add(256 * blk);
                                let s16 = _mm256_loadu_si256(t.s.add(16 * blk) as *const __m256i);
                                let mut s = _mm256_slli_epi32(_mm256_madd_epi16(sc16, s16), 2);
                                s = _mm256_sub_epi32(_mm256_setzero_si256(), s);
                                for c in 0..8 {
                                    s = _mm256_add_epi32(
                                        s,
                                        _mm256_madd_epi16(_mm256_maddubs_epi16(u[c], ld(x.add(32 * c))), sc[c]),
                                    );
                                }
                                acc[jt] = _mm256_fmadd_ps(
                                    _mm256_cvtepi32_ps(s),
                                    _mm256_mul_ps(dw, _mm256_broadcast_ss(&*t.d.add(blk))),
                                    acc[jt],
                                );
                            }
                        }
                    }
                    for j in 0..T {
                        unsafe { *out.add(j * os + r) = hsum(acc[j]) };
                    }
                }
            }

            /// IQ2_XXS (ggml's format and tables): per 32 weights, four signed codebook rows from `iq2_signed` (one
            /// load per 8 weights instead of a grid and a sign lookup), then |w| x sign(x, w) by maddubs and madd by
            /// the block's 2 ls + 1 (i16 exact: two products of at most 43 * 127). d / 8 per 256-block.
            #[target_feature(enable = $feat)]
            pub(super) unsafe fn rows_iq2_xxs<const T: usize>(w: &[u8], nr: usize, cols: usize, tp: &[Tok; MAX_T], out: *mut f32, os: usize) {
                let sg_t = iq2_signed().as_ptr();
                let rb = cols / 256 * 66;
                for r in 0..nr {
                    let mut acc = [_mm256_setzero_ps(); T];
                    for blk in 0..cols / 256 {
                        unsafe {
                            let p = w.as_ptr().add(r * rb + 66 * blk);
                            let mut tot = [_mm256_setzero_si256(); T];
                            for ib in 0..8 {
                                let g = std::ptr::read_unaligned(p.add(2 + 8 * ib) as *const u32) as u64;
                                let sg = std::ptr::read_unaligned(p.add(6 + 8 * ib) as *const u32) as u64;
                                // entry (grid, signs) = 128 * grid + signs: four 15-bit indices in u16 lanes (two pdeps where
                                // pdep is fast; microcoded on Zen 1/2, which take the AVX2 path)
                                let ix = if $pdep {
                                    _pdep_u64(g, 0x7f80_7f80_7f80_7f80) | _pdep_u64(sg, 0x007f_007f_007f_007f)
                                } else {
                                    (0..4).fold(0u64, |a, l| a | ((g >> (8 * l)) & 255) << (16 * l + 7) | ((sg >> (7 * l)) & 127) << (16 * l))
                                };
                                let e = |l: usize| *sg_t.add((ix >> (16 * l)) as u16 as usize);
                                let lo = _mm_set_epi64x(e(1) as i64, e(0) as i64);
                                let hi = _mm_set_epi64x(e(3) as i64, e(2) as i64);
                                let wv = _mm256_set_m128i(hi, lo);
                                let aw = _mm256_abs_epi8(wv);
                                let ls = _mm256_set1_epi16(2 * (sg >> 28) as i16 + 1);
                                for j in 0..T {
                                    let x = _mm256_sign_epi8(ld(tp[j].x.add(256 * blk + 32 * ib)), wv);
                                    tot[j] = _mm256_add_epi32(tot[j], _mm256_madd_epi16(_mm256_maddubs_epi16(aw, x), ls));
                                }
                            }
                            let dw = _mm256_set1_ps(f16_ld(p) * 0.125);
                            for j in 0..T {
                                acc[j] = _mm256_fmadd_ps(
                                    _mm256_cvtepi32_ps(tot[j]),
                                    _mm256_mul_ps(dw, _mm256_broadcast_ss(&*tp[j].d.add(blk))),
                                    acc[j],
                                );
                            }
                        }
                    }
                    for j in 0..T {
                        unsafe { *out.add(j * os + r) = hsum(acc[j]) };
                    }
                }
            }

            /// IQ1_S (ggml's format and table): per 32 weights four ternary codebook rows (2048 x 8 bytes, L1-resident),
            /// |w| x sign(x, w) by maddubs and madd by the group's 2s + 1; the +-1/8 shift is (2s + 1) x sum(x) from the
            /// per-16 activation sums, kept as a scalar. d per 256-block.
            #[target_feature(enable = $feat)]
            pub(super) unsafe fn rows_iq1_s<const T: usize>(w: &[u8], nr: usize, cols: usize, tp: &[Tok; MAX_T], out: *mut f32, os: usize) {
                let grid = crate::iqgrid::IQ1S_GRID.as_ptr();
                let rb = cols / 256 * 50;
                for r in 0..nr {
                    let mut acc = [_mm256_setzero_ps(); T];
                    let mut accd = [0f32; T];
                    for blk in 0..cols / 256 {
                        unsafe {
                            let p = w.as_ptr().add(r * rb + 50 * blk);
                            let mut tot = [_mm256_setzero_si256(); T];
                            let mut dsum = [0i32; T];
                            for ib in 0..8 {
                                let qs = std::ptr::read_unaligned(p.add(2 + 4 * ib) as *const u32) as u64;
                                let qh = std::ptr::read_unaligned(p.add(34 + 2 * ib) as *const u16) as u64;
                                // four 11-bit indices in u16 lanes: low byte from qs, high 3 bits from qh
                                let ix = if $pdep {
                                    _pdep_u64(qs, 0x00ff_00ff_00ff_00ff) | _pdep_u64(qh, 0x0700_0700_0700_0700)
                                } else {
                                    (0..4).fold(0u64, |a, l| a | ((qs >> (8 * l)) & 255) << (16 * l) | ((qh >> (3 * l)) & 7) << (16 * l + 8))
                                };
                                let e = |l: usize| *grid.add((ix >> (16 * l)) as u16 as usize);
                                let lo = _mm_set_epi64x(e(1) as i64, e(0) as i64);
                                let hi = _mm_set_epi64x(e(3) as i64, e(2) as i64);
                                let wv = _mm256_set_m128i(hi, lo);
                                let aw = _mm256_abs_epi8(wv);
                                let l1 = 2 * ((qh >> 12) & 7) as i32 + 1;
                                let ls = _mm256_set1_epi16(l1 as i16);
                                let neg = qh & 0x8000 != 0;
                                for j in 0..T {
                                    let x = _mm256_sign_epi8(ld(tp[j].x.add(256 * blk + 32 * ib)), wv);
                                    tot[j] = _mm256_add_epi32(tot[j], _mm256_madd_epi16(_mm256_maddubs_epi16(aw, x), ls));
                                    let sx = *tp[j].s.add(16 * blk + 2 * ib) as i32 + *tp[j].s.add(16 * blk + 2 * ib + 1) as i32;
                                    dsum[j] += if neg { -l1 * sx } else { l1 * sx };
                                }
                            }
                            let dw = f16_ld(p);
                            for j in 0..T {
                                let dd = dw * *tp[j].d.add(blk);
                                acc[j] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(tot[j]), _mm256_set1_ps(dd), acc[j]);
                                accd[j] += dd * 0.125 * dsum[j] as f32;
                            }
                        }
                    }
                    for j in 0..T {
                        unsafe { *out.add(j * os + r) = hsum(acc[j]) + accd[j] };
                    }
                }
            }

            /// IQ2_XS: per 8 weights a u16 (9-bit index into the 512-row magnitude grid, 7-bit ksigns index); per 16 weights a
            /// 4-bit scale s (weight = d (2s + 1) / 8 x magnitude x sign). Signs become a byte mask (broadcast, and, cmpeq) for
            /// sign(x, mask | 1); |w| x x by maddubs, madd by the two 16-weight scales.
            #[target_feature(enable = $feat)]
            pub(super) unsafe fn rows_iq2_xs<const T: usize>(w: &[u8], nr: usize, cols: usize, tp: &[Tok; MAX_T], out: *mut f32, os: usize) {
                let grid = crate::iqgrid::IQ2XS_GRID.as_ptr();
                let ks = crate::iqgrid::KSIGNS_IQ2XS.as_ptr();
                let rb = cols / 256 * 74;
                let rep = _mm256_set_epi64x(0x0303030303030303, 0x0202020202020202, 0x0101010101010101, 0);
                let bits = _mm256_set1_epi64x(0x8040201008040201u64 as i64);
                let one = _mm256_set1_epi8(1);
                // i16 lane l of the scale vector: byte 2 ib (low 8 lanes = first 16 weights) or 2 ib + 1, high byte zero
                let lsi = _mm256_set_epi64x(0x8001_8001_8001_8001u64 as i64, 0x8001_8001_8001_8001u64 as i64, 0x8000_8000_8000_8000u64 as i64, 0x8000_8000_8000_8000u64 as i64);
                for r in 0..nr {
                    let mut acc = [_mm256_setzero_ps(); T];
                    for blk in 0..cols / 256 {
                        unsafe {
                            let p = w.as_ptr().add(r * rb + 74 * blk);
                            let mut tot = [_mm256_setzero_si256(); T];
                            let lsb = {
                                // 2s + 1 per 16 weights as bytes (lo, hi of each 32), in both lanes
                                let sc8 = _mm_loadl_epi64(p.add(66) as *const __m128i);
                                let m = _mm_set1_epi8(15);
                                let v = _mm_unpacklo_epi8(_mm_and_si128(sc8, m), _mm_and_si128(_mm_srli_epi16(sc8, 4), m));
                                _mm256_broadcastsi128_si256(_mm_add_epi8(_mm_add_epi8(v, v), _mm_set1_epi8(1)))
                            };
                            for ib in 0..8 {
                                let q = std::ptr::read_unaligned(p.add(2 + 8 * ib) as *const u64);
                                let e = |l: usize| *grid.add(((q >> (16 * l)) & 511) as usize);
                                let sg = |l: usize| *ks.add(((q >> (16 * l + 9)) & 127) as usize) as u32;
                                let wv = _mm256_set_epi64x(e(3) as i64, e(2) as i64, e(1) as i64, e(0) as i64);
                                let sb = sg(0) | sg(1) << 8 | sg(2) << 16 | sg(3) << 24;
                                let m = _mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32(sb as i32), rep), bits), bits);
                                let sm = _mm256_or_si256(m, one);
                                let ls = _mm256_shuffle_epi8(lsb, _mm256_add_epi8(lsi, _mm256_set1_epi8(2 * ib as i8)));
                                for j in 0..T {
                                    let x = _mm256_sign_epi8(ld(tp[j].x.add(256 * blk + 32 * ib)), sm);
                                    tot[j] = _mm256_add_epi32(tot[j], _mm256_madd_epi16(_mm256_maddubs_epi16(wv, x), ls));
                                }
                            }
                            let dw = _mm256_set1_ps(f16_ld(p) * 0.125);
                            for j in 0..T {
                                acc[j] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(tot[j]), _mm256_mul_ps(dw, _mm256_broadcast_ss(&*tp[j].d.add(blk))), acc[j]);
                            }
                        }
                    }
                    for j in 0..T {
                        unsafe { *out.add(j * os + r) = hsum(acc[j]) };
                    }
                }
            }

            /// IQ2_S: per 8 weights a 10-bit index into the 1024-row magnitude grid (low byte in qs, 2 bits in qh) and an
            /// explicit sign byte; scales as IQ2_XS.
            #[target_feature(enable = $feat)]
            pub(super) unsafe fn rows_iq2_s<const T: usize>(w: &[u8], nr: usize, cols: usize, tp: &[Tok; MAX_T], out: *mut f32, os: usize) {
                let grid = crate::iqgrid::IQ2S_GRID.as_ptr();
                let rb = cols / 256 * 82;
                let rep = _mm256_set_epi64x(0x0303030303030303, 0x0202020202020202, 0x0101010101010101, 0);
                let bits = _mm256_set1_epi64x(0x8040201008040201u64 as i64);
                let one = _mm256_set1_epi8(1);
                // i16 lane l of the scale vector: byte 2 ib (low 8 lanes = first 16 weights) or 2 ib + 1, high byte zero
                let lsi = _mm256_set_epi64x(0x8001_8001_8001_8001u64 as i64, 0x8001_8001_8001_8001u64 as i64, 0x8000_8000_8000_8000u64 as i64, 0x8000_8000_8000_8000u64 as i64);
                for r in 0..nr {
                    let mut acc = [_mm256_setzero_ps(); T];
                    for blk in 0..cols / 256 {
                        unsafe {
                            let p = w.as_ptr().add(r * rb + 82 * blk);
                            let mut tot = [_mm256_setzero_si256(); T];
                            let lsb = {
                                // 2s + 1 per 16 weights as bytes (lo, hi of each 32), in both lanes
                                let sc8 = _mm_loadl_epi64(p.add(74) as *const __m128i);
                                let m = _mm_set1_epi8(15);
                                let v = _mm_unpacklo_epi8(_mm_and_si128(sc8, m), _mm_and_si128(_mm_srli_epi16(sc8, 4), m));
                                _mm256_broadcastsi128_si256(_mm_add_epi8(_mm_add_epi8(v, v), _mm_set1_epi8(1)))
                            };
                            for ib in 0..8 {
                                let qs = std::ptr::read_unaligned(p.add(2 + 4 * ib) as *const u32) as usize;
                                let qh = *p.add(66 + ib) as usize;
                                let e = |l: usize| *grid.add((qs >> (8 * l)) & 255 | (qh << (8 - 2 * l)) & 0x300);
                                let wv = _mm256_set_epi64x(e(3) as i64, e(2) as i64, e(1) as i64, e(0) as i64);
                                let sb = std::ptr::read_unaligned(p.add(34 + 4 * ib) as *const i32);
                                let m = _mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32(sb), rep), bits), bits);
                                let sm = _mm256_or_si256(m, one);
                                let ls = _mm256_shuffle_epi8(lsb, _mm256_add_epi8(lsi, _mm256_set1_epi8(2 * ib as i8)));
                                for j in 0..T {
                                    let x = _mm256_sign_epi8(ld(tp[j].x.add(256 * blk + 32 * ib)), sm);
                                    tot[j] = _mm256_add_epi32(tot[j], _mm256_madd_epi16(_mm256_maddubs_epi16(wv, x), ls));
                                }
                            }
                            let dw = _mm256_set1_ps(f16_ld(p) * 0.125);
                            for j in 0..T {
                                acc[j] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(tot[j]), _mm256_mul_ps(dw, _mm256_broadcast_ss(&*tp[j].d.add(blk))), acc[j]);
                            }
                        }
                    }
                    for j in 0..T {
                        unsafe { *out.add(j * os + r) = hsum(acc[j]) };
                    }
                }
            }

            /// Q8_0: |w| x sign(x, w).
            #[target_feature(enable = $feat)]
            pub(super) unsafe fn rows_q8_0<const T: usize>(w: &[u8], nr: usize, cols: usize, tp: &[Tok; MAX_T], out: *mut f32, os: usize) {
                let rb = cols / 32 * 34;
                for r in 0..nr {
                    let mut acc = [_mm256_setzero_ps(); T];
                    for b in 0..cols / 32 {
                        unsafe {
                            let p = w.as_ptr().add(r * rb + 34 * b);
                            let q = _mm256_loadu_si256(p.add(2) as *const __m256i);
                            let aq = _mm256_sign_epi8(q, q);
                            let dw = _mm256_set1_ps(f16_ld(p));
                            for j in 0..T {
                                let t = &tp[j];
                                let xs = _mm256_sign_epi8(ld(t.x.add(32 * b)), q);
                                let s = dp(_mm256_setzero_si256(), aq, xs);
                                acc[j] = _mm256_fmadd_ps(
                                    _mm256_cvtepi32_ps(s),
                                    _mm256_mul_ps(dw, _mm256_broadcast_ss(&*t.d.add(b / 8))),
                                    acc[j],
                                );
                            }
                        }
                    }
                    for j in 0..T {
                        unsafe { *out.add(j * os + r) = hsum(acc[j]) };
                    }
                }
            }
        }
    };
}

isa_kernels!(vnni, "avx2,avxvnni,fma,f16c,bmi2", true, |s, a, b| _mm256_dpbusd_avx_epi32(s, a, b));
isa_kernels!(vnni512, "avx2,avx512vnni,avx512vl,fma,f16c,bmi2", true, |s, a, b| _mm256_dpbusd_epi32(s, a, b));
isa_kernels!(avx2, "avx2,fma,f16c", false, |s, a, b| _mm256_add_epi32(s, _mm256_madd_epi16(_mm256_maddubs_epi16(a, b), _mm256_set1_epi16(1))));

#[cfg(test)]
pub mod tests {
    use super::*;
    use half::f16;

    pub struct Rng(pub u64);
    impl Rng {
        pub fn u64(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        pub fn f32(&mut self) -> f32 {
            (self.u64() >> 40) as f32 / (1u64 << 24) as f32
        }
        pub fn normal(&mut self) -> f32 {
            let (a, b) = (self.f32().max(1e-7), self.f32());
            (-2.0 * a.ln()).sqrt() * (std::f32::consts::TAU * b).cos()
        }
    }

    /// Random rows: random codes, fp16 scales in a sane range.
    pub fn rand_rows(ty: GType, nr: usize, cols: usize, rng: &mut Rng) -> Vec<u8> {
        let mut v: Vec<u8> = (0..ty.row_bytes(cols) * nr).map(|_| rng.u64() as u8).collect();
        fix_scales(ty, &mut v, rng);
        v
    }

    pub fn fix_scales(ty: GType, v: &mut [u8], rng: &mut Rng) {
        let (_, bs) = ty.block();
        let at = match ty {
            GType::Q2_0 | GType::Q8_0 | GType::Iq2Xxs | GType::Iq2Xs | GType::Iq2S | GType::Iq1S => 0,
            GType::Ptq1_0 => 26,
            GType::Q3K => 108,
            GType::Q2K => 80,
            GType::Iq3Xxs => 0,
            _ => unreachable!(),
        };
        let range = if ty == GType::Q3K { 0.002 } else { 0.05 };
        for blk in v.chunks_exact_mut(bs) {
            let d = f16::from_f32(range * (0.5 + rng.f32()));
            blk[at..at + 2].copy_from_slice(&d.to_le_bytes());
            if ty == GType::Q2K { blk[82..84].copy_from_slice(&f16::from_f32(0.015 * rng.f32()).to_le_bytes()); }
        }
    }

    fn rel(a: &[f32], b: &[f32]) -> f64 {
        let num: f64 = a.iter().zip(b).map(|(&x, &y)| ((x - y) as f64).powi(2)).sum();
        let den: f64 = b.iter().map(|&y| (y as f64).powi(2)).sum();
        (num / den).sqrt()
    }

    pub fn f32_ref(ty: GType, rows: &[u8], cols: usize, x: &[f32]) -> Vec<f32> {
        let rb = ty.row_bytes(cols);
        let mut w = vec![0f32; cols];
        rows.chunks_exact(rb)
            .map(|row| {
                dequant_row(ty, row, &mut w);
                w.iter().zip(x).map(|(&a, &b)| a as f64 * b as f64).sum::<f64>() as f32
            })
            .collect()
    }

    /// the SIMD instruction sets this CPU runs
    fn simd_isas() -> Vec<Isa> {
        let v: Vec<Isa> = [Isa::Vnni, Isa::Vnni512, Isa::Avx2].into_iter().filter(|i| i.supported()).collect();
        assert!(!v.is_empty(), "test box lacks AVX2");
        v
    }

    #[test]
    fn kernels_match_scalar() {
        for isa in simd_isas() {
            kernels_match_scalar_on(isa);
        }
    }

    fn kernels_match_scalar_on(isa: Isa) {
        let mut rng = Rng(0x9e3779b97f4a7c15);
        for ty in [GType::Q2K, GType::Iq3Xxs, GType::Q2_0, GType::Ptq1_0, GType::Q3K, GType::Q8_0, GType::Iq2Xxs, GType::Iq2Xs, GType::Iq2S, GType::Iq1S] {
            for &cols in &[5120usize, 2304] {
                for t in [1usize, 3, 6, 8, 11] {
                    let nr = 37;
                    let rows = rand_rows(ty, nr, cols, &mut rng);
                    let x: Vec<f32> = (0..t * cols).map(|_| rng.normal()).collect();
                    let act = QAct::new(&x, t, cols, &[ty]);
                    let toks: Vec<usize> = (0..t).collect();
                    let (mut simd, mut sc) = (vec![0f32; t * nr], vec![0f32; t * nr]);
                    dot_rows_isa(isa, ty, &rows, cols, &act, &toks, &mut simd, nr);
                    dot_rows_scalar(ty, &rows, cols, &act, &toks, &mut sc, nr);
                    let e_sc = rel(&simd, &sc);
                    let reff: Vec<f32> = (0..t)
                        .flat_map(|j| f32_ref(ty, &rows, cols, &x[j * cols..(j + 1) * cols]))
                        .collect();
                    let e_f32 = rel(&simd, &reff);
                    if t == 6 {
                        println!("{isa:?} {ty:?} cols {cols} T {t}: vs int8 scalar {e_sc:.2e}, vs f32 {e_f32:.2e}");
                    }
                    assert!(e_sc < 1e-6, "{isa:?} {ty:?} cols {cols} T {t}: {e_sc}");
                    assert!(e_f32 < 2e-2, "{isa:?} {ty:?} cols {cols} T {t}: {e_f32}");
                }
            }
        }
    }

    #[test]
    fn quantize_simd_matches_scalar() {
        let mut rng = Rng(3);
        let (t, n) = (3, 2304);
        let tys = [GType::Q2_0, GType::Ptq1_0, GType::Q3K];
        let mut x: Vec<f32> = (0..t * n).map(|_| rng.normal() * 3.0).collect();
        x[300..556].fill(0.0); // an all-zero block
        x[5] = 1e4; // an outlier
        for isa in simd_isas() {
            let mut a = QAct::default();
            a.reset(t, n, &tys);
            let r = a.raw();
            for tok in 0..t {
                for blk in 0..n / QK {
                    let xb: &[f32; QK] = x[tok * n + blk * QK..][..QK].try_into().unwrap();
                    unsafe {
                        match isa {
                            Isa::Vnni => vnni::quantize_block(&r, tok, blk, xb),
                            Isa::Vnni512 => vnni512::quantize_block(&r, tok, blk, xb),
                            _ => avx2::quantize_block(&r, tok, blk, xb),
                        }
                    }
                }
            }
            quantize_matches_scalar(&a, &x, t, n, &tys);
        }
    }

    fn quantize_matches_scalar(a: &QAct, x: &[f32], t: usize, n: usize, tys: &[GType]) {
        let mut b = QAct::default();
        b.reset(t, n, tys);
        let r = b.raw();
        for tok in 0..t {
            for blk in 0..n / QK {
                let xb: &[f32; QK] = x[tok * n + blk * QK..][..QK].try_into().unwrap();
                unsafe { r.quantize_block_scalar(tok, blk, xb) };
            }
        }
        assert_eq!(a.d, b.d);
        assert_eq!(a.q, b.q);
        assert_eq!(a.s16, b.s16);
        assert_eq!(a.q2, b.q2);
        assert_eq!(a.q2n, b.q2n);
        assert_eq!(a.t1, b.t1);
        assert_eq!(a.t1n, b.t1n);
    }

    #[test]
    fn selected_tokens_and_stride() {
        let mut rng = Rng(7);
        let (ty, cols, nr) = (GType::Q2_0, 2304, 5);
        let rows = rand_rows(ty, nr, cols, &mut rng);
        let x: Vec<f32> = (0..4 * cols).map(|_| rng.normal()).collect();
        let act = QAct::new(&x, 4, cols, &[ty, GType::Ptq1_0]);
        let mut all = vec![0f32; 4 * nr];
        dot_rows(ty, &rows, cols, &act, &mut all);
        let mut sel = vec![0f32; 2 * 9];
        dot_rows_sel(ty, &rows, cols, &act, &[3, 1], &mut sel, 9);
        assert_eq!(&sel[..nr], &all[3 * nr..4 * nr]);
        assert_eq!(&sel[9..9 + nr], &all[nr..2 * nr]);
    }

    #[test]
    fn ptq1_tail_block() {
        // last block of a slice is read through the 32-byte tail copy
        let mut rng = Rng(11);
        let (ty, cols) = (GType::Ptq1_0, 2304);
        let rows = rand_rows(ty, 1, cols, &mut rng);
        let x: Vec<f32> = (0..cols).map(|_| rng.normal()).collect();
        let act = QAct::new(&x, 1, cols, &[ty]);
        let (mut a, mut b) = ([0f32], [0f32]);
        dot_rows(ty, &rows, cols, &act, &mut a);
        dot_rows_scalar(ty, &rows, cols, &act, &[0], &mut b, 1);
        assert!((a[0] - b[0]).abs() <= 1e-5 * b[0].abs().max(1.0));
    }
}
