// Routed experts, fast decode path. Weights are repacked at upload (repack_expert) into layouts built around the int8
// tensor-core fragment (mma.sync m16n8k32 / m16n8k16): a warp owns 16 rows, the 8 mma columns are the T <= 8 tokens,
// so a weight byte is decoded once for every token and the instruction count does not grow with T. Activations are
// quantized once per call to int8 (quant_act: per-256 scale, per-64 sums).
//
// k order: inside each 64-weight chunk, lane quad position q (= lane & 3) owns the 4 groups s = 0..3 of weights
// 16s + 4q + i (i = 0..3), so the activations are stored permuted (x byte 16q + 4s + i of a chunk = weight 16s + 4q + i)
// and a lane reads its B fragment of a chunk as one 16-byte load. Group pairs (0, 1), (2, 3) are the two k32 halves.
//
// Layouts (ExpSlot type id = ggml id + 2000; one matrix of `rows` x k; "q" = lane & 3; same bytes as the GGUF blocks):
//   2042 Q2_0   (2.25 bpw)  codes [rows][k/256][q][4 blocks] u32: byte i, bits 2s = code of weight 16s + 4q + i of the
//                           block; then d f16 [rows][k/64]. value = (code - 1) d
//   2143 PTQ1_0 (1.75 bpw)  per 128-block, lane q owns 32 weights (chunk h, group s -> group m = 4h + s): 6 base-3 bytes
//                           (GGUF trit encoding) and 2 trits of a shared byte. W [rows][k/256][q][2 blocks] u32:
//                           byte i trit n = weight i of group n (n < 5); H [rows][k/256][q][2 blocks] u16: byte j trits
//                           (n = 0..4) = group 5 weights j, 2 + j, group 6 weights j, 2 + j, group 7 weight j; C [rows][k/128]
//                           4 B: tail byte for q = 0, 1, tail byte for q = 2, 3 (trits 0, 1 / 2, 3 = group 7 weights
//                           2, 3), d f16. value = (trit - 1) d
//   2011 Q3_K   (3.44 bpw)  low [rows][k/256][q][4 chunks] u32: byte i bits 2s = low 2 bits; nh [rows][k/256][q][2] u32:
//                           byte i bit 4 (h & 1) + s = NOT hmask bit (chunk h); then per 256-block the 12 ggml scale bytes,
//                           then d f16. q = low | (nh ? 0xfc : 0) as int8 = the signed 3-bit value
//   2008 Q8_0   (8.5 bpw)   qs [rows][k/64][q][16 B]: byte 4s + i = weight 16s + 4q + i; then d f16 [rows][k/32]
//   2016 IQ2_XXS (2.06 bpw) ggml's 8 bytes per 32-block unchanged, [rows][k/32] (u32 grid indices, u32 signs + scale);
//                           then d f16 [rows][k/256]. Format and tables: ggml (iq2grid.cuh)
//   2019 IQ1_S  (1.56 bpw)  [rows][k/256][48 B]: qs[32] (low index bytes), qh u16[8]; then d f16 [rows][k/256]
//   2017 IQ2_XS (2.31 bpw)  [rows][k/256][64 B]: qs u16[32] (9-bit grid + 7-bit sign index); then scales[8]
//                           [rows][k/256][8 B]; then d f16 [rows][k/256]
//   2022 IQ2_S  (2.56 bpw)  [rows][k/256][64 B]: qs[32] (low index bytes), signs[32]; then [rows][k/256][16 B]:
//                           qh[8] (2 high index bits per group), scales[8]; then d f16 [rows][k/256]
//   (the GGUF block bytes, d split out so each block's codes start 16-byte aligned)
#pragma once
#include "iq2grid.cuh"
#include "native_quant.cuh"
#define RP_Q2K 2010
#define RP_IQ3XXS 2018

#define RP_Q8 2008
#define RP_Q2 2042
#define RP_T1 2143
#define RP_Q3 2011
#define RP_IQ2 2016
#define RP_IQ2XS 2017
#define RP_IQ1S 2019
#define RP_IQ2S 2022
#define PLAN_W 10
// codebook tables of the IQ tiles: dynamic shared memory, sized at launch by the store's types (experts::tile_smem), so
// kernels over other formats keep their occupancy
extern __shared__ unsigned iq_tab[];
// ABL (build with DSX_KDEFS=-DABL=n, timing experiments only; results are wrong): bit 1 drops the IQ codebook lookups
// (the index stands in for the grid entry), 2 the sign lookups, 4 the weight loads (constants instead), 8 the mma (cheap
// integer ops instead), 16 the shared-table fill
#ifndef ABL
#define ABL 0
#endif  // plan entry: expert, 0, pick index per token (-1 = not picked)

struct XQ {
  const int8_t *q;  // [v][k] permuted
  const float *d;   // [v][k / 256]
  const short *s;   // [v][k / 64] sums of q
};

// copy n words (n % 4 == 0, 16-byte aligned) of a codebook table into shared memory, 16 bytes per thread per step
__device__ __forceinline__ void tab_copy(unsigned *dst, const void *src, int n) {
  if (ABL & 16) return;
  for (int i = threadIdx.x; i < n / 4; i += blockDim.x) ((uint4 *)dst)[i] = __ldg((const uint4 *)src + i);
}

__device__ __forceinline__ int comp4(const int4 &v, int i) { return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w; }
__device__ __forceinline__ float h2f(unsigned v, int hi) { return __half2float(__ushort_as_half((unsigned short)(hi ? v >> 16 : v))); }

__device__ __forceinline__ void mma32(int (&c)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3, int b0, int b1) {
  if (ABL & 8) {
    c[0] += a0 ^ b0;
    c[1] += a1 ^ b1;
    c[2] += a2 ^ b0;
    c[3] += a3 ^ b1;
    return;
  }
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma16(int (&c)[4], unsigned a0, unsigned a1, int b0) {
  c[0] = c[1] = c[2] = c[3] = 0;
  if (ABL & 8) {
    c[0] = a0 ^ b0;
    c[1] = a1 ^ b0;
    c[2] = a0 + b0;
    c[3] = a1 + b0;
    return;
  }
  asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
               : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
               : "r"(a0), "r"(a1), "r"(b0));
}

// A warp's view of one call: B fragment token g (lane >> 2) as activation vector xv (-1 = none), and the C columns'
// tokens 2q, 2q + 1 as vectors c0, c1 (their scales and sums). acc[m][i]: C element i (rows g, g, g+8, g+8 x tokens
// 2q, 2q+1, 2q, 2q+1) of matrix m.
struct Lane {
  int q, xv, c0, c1, k0;  // k0: first x-block (warps of a block split the x-blocks, stride KS)
};
// 16-row tiles per block of the fast kernels (build with DSX_ROWT=n; experts::ROWT follows): the IQ tiles fill their
// shared codebook tables once per block, so more tiles per block amortize the fill
#ifndef ROWT
#define ROWT 1
#endif
#ifndef KS
#define KS 2  // warps per block = per 16-row tile (build with DSX_KS=n to change; experts::KS follows)
#endif

// activations of x-block xb (256 weights): X[h] = this lane's 16 bytes of chunk h; dx of the C tokens
__device__ __forceinline__ void load_x(const XQ &x, int k, int xb, const Lane &L, int4 (&X)[4], float &dx0, float &dx1) {
#pragma unroll
  for (int h = 0; h < 4; h++)
    X[h] = L.xv < 0 ? make_int4(0, 0, 0, 0) : __ldg((const int4 *)(x.q + (size_t)L.xv * k + xb * 256 + h * 64 + L.q * 16));
  dx0 = L.c0 < 0 ? 0.f : __ldg(x.d + (size_t)L.c0 * (k >> 8) + xb);
  dx1 = L.c1 < 0 ? 0.f : __ldg(x.d + (size_t)L.c1 * (k >> 8) + xb);
}
// the four per-64 sums of x-block xb for C token vector v
__device__ __forceinline__ int2 load_s(const XQ &x, int k, int xb, int v) {
  return v < 0 ? make_int2(0, 0) : __ldg((const int2 *)(x.s + (size_t)v * (k >> 6) + xb * 4));
}
__device__ __forceinline__ int s64(int2 s, int h) { int v = h < 2 ? s.x : s.y; return (h & 1) ? (v >> 16) : (int)(short)v; }

// Q2_0: x-block = 4 blocks
template <int NM>
__device__ __forceinline__ void tile_q2(const uint8_t *const (&m)[NM], int rows, int k, int ra, int rb, const XQ &x,
                                        const Lane &L, float (&acc)[NM][4]) {
  const int nb = k >> 6, ng = k >> 8;
  const size_t pd = (size_t)rows * nb * 16;
  for (int xb = L.k0; xb < ng; xb += KS) {
    int4 X[4], wa[NM], wb[NM];
    int2 da[NM], db[NM];
#pragma unroll
    for (int j = 0; j < NM; j++) {
      wa[j] = __ldg((const int4 *)(m[j] + (((size_t)ra * ng + xb) * 4 + L.q) * 16));
      wb[j] = __ldg((const int4 *)(m[j] + (((size_t)rb * ng + xb) * 4 + L.q) * 16));
      da[j] = __ldg((const int2 *)(m[j] + pd + ((size_t)ra * nb + 4 * xb) * 2));
      db[j] = __ldg((const int2 *)(m[j] + pd + ((size_t)rb * nb + 4 * xb) * 2));
    }
    float dx0, dx1;
    load_x(x, k, xb, L, X, dx0, dx1);
    int2 s0 = load_s(x, k, xb, L.c0), s1 = load_s(x, k, xb, L.c1);
#pragma unroll
    for (int j = 0; j < NM; j++) {
      float p[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int b = 0; b < 4; b++) {
        unsigned A = (unsigned)comp4(wa[j], b), B = (unsigned)comp4(wb[j], b);
        int c[4] = {0, 0, 0, 0};
        mma32(c, A & 0x03030303u, B & 0x03030303u, (A >> 2) & 0x03030303u, (B >> 2) & 0x03030303u, X[b].x, X[b].y);
        mma32(c, (A >> 4) & 0x03030303u, (B >> 4) & 0x03030303u, (A >> 6) & 0x03030303u, (B >> 6) & 0x03030303u, X[b].z, X[b].w);
        float fa = h2f(b < 2 ? da[j].x : da[j].y, b & 1), fb = h2f(b < 2 ? db[j].x : db[j].y, b & 1);
        int t0 = s64(s0, b), t1 = s64(s1, b);
        p[0] += fa * (float)(c[0] - t0);
        p[1] += fa * (float)(c[1] - t1);
        p[2] += fb * (float)(c[2] - t0);
        p[3] += fb * (float)(c[3] - t1);
      }
      acc[j][0] += p[0] * dx0;
      acc[j][1] += p[1] * dx1;
      acc[j][2] += p[2] * dx0;
      acc[j][3] += p[3] * dx1;
    }
  }
}

// next trit of the 4 bytes held as e = bytes 0, 2 and o = bytes 1, 3 in 16-bit lanes: 3 * lane = trit << 8 | rest,
// so byte 1 / 3 of the products is the trit (the decoder's ((q * 3^n) & 255) * 3 >> 8 without byte multiplies)
__device__ __forceinline__ unsigned t1_step(unsigned &e, unsigned &o) {
  unsigned E = e * 3, O = o * 3;
  e = E & 0x00ff00ffu;
  o = O & 0x00ff00ffu;
  return __byte_perm(E, O, 0x7351);
}
// a lane's 8 trit groups of one PTQ1_0 block (bytes of the groups = trits 0..2)
__device__ __forceinline__ void t1_groups(unsigned w, unsigned hw, unsigned tb, bool odd, unsigned (&gq)[8]) {
  unsigned e = w & 0x00ff00ffu, o = __byte_perm(w, 0, 0x4341);
#pragma unroll
  for (int n = 0; n < 5; n++) gq[n] = t1_step(e, o);
  unsigned v = __byte_perm(hw, 0, 0x4140), P[5];
#pragma unroll
  for (int n = 0; n < 5; n++) {
    P[n] = v * 3;
    v = P[n] & 0x00ff00ffu;
  }
  gq[5] = __byte_perm(P[0], P[1], 0x7531);
  gq[6] = __byte_perm(P[2], P[3], 0x7531);
  unsigned u = tb & 0xff, Q[4];
#pragma unroll
  for (int n = 0; n < 4; n++) {
    Q[n] = u * 3;
    u = Q[n] & 0xff;
  }
  unsigned tp = odd ? __byte_perm(Q[2], Q[3], 0x3351) : __byte_perm(Q[0], Q[1], 0x3351);
  gq[7] = __byte_perm(P[4], tp, 0x5431);
}

// PTQ1_0: x-block = 2 blocks
template <int NM>
__device__ __forceinline__ void tile_t1(const uint8_t *const (&m)[NM], int rows, int k, int ra, int rb, const XQ &x,
                                        const Lane &L, float (&acc)[NM][4]) {
  const int nb = k >> 7, ng = k >> 8;
  const size_t ph = (size_t)rows * nb * 16, pc = (size_t)rows * nb * 24;
  const bool odd = L.q & 1;
  for (int xb = L.k0; xb < ng; xb += KS) {
    int4 X[4];
    int2 wa[NM], wb[NM], ca[NM], cb[NM];
    unsigned ha[NM], hb[NM];
#pragma unroll
    for (int j = 0; j < NM; j++) {
      size_t oa = ((size_t)ra * ng + xb) * 4 + L.q, ob = ((size_t)rb * ng + xb) * 4 + L.q;
      wa[j] = __ldg((const int2 *)(m[j] + oa * 8));
      wb[j] = __ldg((const int2 *)(m[j] + ob * 8));
      ha[j] = __ldg((const unsigned *)(m[j] + ph + oa * 4));
      hb[j] = __ldg((const unsigned *)(m[j] + ph + ob * 4));
      ca[j] = __ldg((const int2 *)(m[j] + pc + ((size_t)ra * nb + 2 * xb) * 4));
      cb[j] = __ldg((const int2 *)(m[j] + pc + ((size_t)rb * nb + 2 * xb) * 4));
    }
    float dx0, dx1;
    load_x(x, k, xb, L, X, dx0, dx1);
    int2 s0 = load_s(x, k, xb, L.c0), s1 = load_s(x, k, xb, L.c1);
#pragma unroll
    for (int j = 0; j < NM; j++) {
      float p[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int b = 0; b < 2; b++) {
        unsigned ga[8], gb[8];
        unsigned cwa = (unsigned)(b ? ca[j].y : ca[j].x), cwb = (unsigned)(b ? cb[j].y : cb[j].x);
        t1_groups((unsigned)(b ? wa[j].y : wa[j].x), (b ? ha[j] >> 16 : ha[j]) & 0xffff, cwa >> (8 * (L.q >> 1)), odd, ga);
        t1_groups((unsigned)(b ? wb[j].y : wb[j].x), (b ? hb[j] >> 16 : hb[j]) & 0xffff, cwb >> (8 * (L.q >> 1)), odd, gb);
        int c[4] = {0, 0, 0, 0};
#pragma unroll
        for (int h = 0; h < 2; h++) {
          int4 xx = X[2 * b + h];
          mma32(c, ga[4 * h], gb[4 * h], ga[4 * h + 1], gb[4 * h + 1], xx.x, xx.y);
          mma32(c, ga[4 * h + 2], gb[4 * h + 2], ga[4 * h + 3], gb[4 * h + 3], xx.z, xx.w);
        }
        float fa = h2f(cwa, 1), fb = h2f(cwb, 1);
        int t0 = s64(s0, 2 * b) + s64(s0, 2 * b + 1), t1 = s64(s1, 2 * b) + s64(s1, 2 * b + 1);
        p[0] += fa * (float)(c[0] - t0);
        p[1] += fa * (float)(c[1] - t1);
        p[2] += fb * (float)(c[2] - t0);
        p[3] += fb * (float)(c[3] - t1);
      }
      acc[j][0] += p[0] * dx0;
      acc[j][1] += p[1] * dx1;
      acc[j][2] += p[2] * dx0;
      acc[j][3] += p[3] * dx1;
    }
  }
}

// the 16 6-bit scales of a Q3_K block (ggml packing): byte j of word w = scale 4w + j
__device__ __forceinline__ void q3_scales(const uint8_t *p, unsigned (&s)[4]) {
  unsigned a = __ldg((const unsigned *)p), c = __ldg((const unsigned *)p + 1), t = __ldg((const unsigned *)p + 2);
  s[0] = (a & 0x0f0f0f0fu) | ((t & 0x03030303u) << 4);
  s[1] = (c & 0x0f0f0f0fu) | (((t >> 2) & 0x03030303u) << 4);
  s[2] = ((a >> 4) & 0x0f0f0f0fu) | (((t >> 4) & 0x03030303u) << 4);
  s[3] = ((c >> 4) & 0x0f0f0f0fu) | (((t >> 6) & 0x03030303u) << 4);
}

// Q3_K: x-block = 1 block, k16 mma per 16-weight sub-block (sub-block 4h + s), integer scale per sub-block
template <int NM>
__device__ __forceinline__ void tile_q3(const uint8_t *const (&m)[NM], int rows, int k, int ra, int rb, const XQ &x,
                                        const Lane &L, float (&acc)[NM][4]) {
  const int nb = k >> 8;
  const size_t ph = (size_t)rows * nb * 64, ps = (size_t)rows * nb * 96, pd = (size_t)rows * nb * 108;
  for (int xb = L.k0; xb < nb; xb += KS) {
    int4 X[4], la[NM], lb[NM];
    int2 na[NM], nbm[NM];
    float fa[NM], fb[NM];
#pragma unroll
    for (int j = 0; j < NM; j++) {
      size_t oa = ((size_t)ra * nb + xb) * 4 + L.q, ob = ((size_t)rb * nb + xb) * 4 + L.q;
      la[j] = __ldg((const int4 *)(m[j] + oa * 16));
      lb[j] = __ldg((const int4 *)(m[j] + ob * 16));
      na[j] = __ldg((const int2 *)(m[j] + ph + oa * 8));
      nbm[j] = __ldg((const int2 *)(m[j] + ph + ob * 8));
      fa[j] = __half2float(__ldg((const __half *)(m[j] + pd) + (size_t)ra * nb + xb));
      fb[j] = __half2float(__ldg((const __half *)(m[j] + pd) + (size_t)rb * nb + xb));
    }
    float dx0, dx1;
    load_x(x, k, xb, L, X, dx0, dx1);
#pragma unroll
    for (int j = 0; j < NM; j++) {
      unsigned sa[4], sb[4];
      q3_scales(m[j] + ps + ((size_t)ra * nb + xb) * 12, sa);
      q3_scales(m[j] + ps + ((size_t)rb * nb + xb) * 12, sb);
      int is[4] = {0, 0, 0, 0};
#pragma unroll
      for (int h = 0; h < 4; h++) {
        unsigned A = (unsigned)comp4(la[j], h), B = (unsigned)comp4(lb[j], h);
        unsigned NA = (unsigned)(h < 2 ? na[j].x : na[j].y) >> (4 * (h & 1)), NB = (unsigned)(h < 2 ? nbm[j].x : nbm[j].y) >> (4 * (h & 1));
#pragma unroll
        for (int s = 0; s < 4; s++) {
          unsigned qa = ((A >> (2 * s)) & 0x03030303u) | ((NA >> s) & 0x01010101u) * 0xfcu;
          unsigned qb = ((B >> (2 * s)) & 0x03030303u) | ((NB >> s) & 0x01010101u) * 0xfcu;
          int c[4];
          mma16(c, qa, qb, comp4(X[h], s));
          int sub = 4 * h + s;
          int va = (int)((sa[sub >> 2] >> (8 * (sub & 3))) & 0xff) - 32, vb = (int)((sb[sub >> 2] >> (8 * (sub & 3))) & 0xff) - 32;
          is[0] += va * c[0];
          is[1] += va * c[1];
          is[2] += vb * c[2];
          is[3] += vb * c[3];
        }
      }
      acc[j][0] += fa[j] * dx0 * (float)is[0];
      acc[j][1] += fa[j] * dx1 * (float)is[1];
      acc[j][2] += fb[j] * dx0 * (float)is[2];
      acc[j][3] += fb[j] * dx1 * (float)is[3];
    }
  }
}

// Q8_0: x-block = 4 chunks of 2 blocks
template <int NM>
__device__ __forceinline__ void tile_q8(const uint8_t *const (&m)[NM], int rows, int k, int ra, int rb, const XQ &x,
                                        const Lane &L, float (&acc)[NM][4]) {
  const int nu = k >> 6, ng = k >> 8;
  const size_t pd = (size_t)rows * k;
  for (int xb = L.k0; xb < ng; xb += KS) {
    int4 X[4];
    float dx0, dx1;
    load_x(x, k, xb, L, X, dx0, dx1);
#pragma unroll
    for (int j = 0; j < NM; j++) {
      int4 da = __ldg((const int4 *)(m[j] + pd + ((size_t)ra * (k >> 5) + 8 * xb) * 2));
      int4 db = __ldg((const int4 *)(m[j] + pd + ((size_t)rb * (k >> 5) + 8 * xb) * 2));
      float p[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int h = 0; h < 4; h++) {
        int4 A = __ldg((const int4 *)(m[j] + (((size_t)ra * nu + 4 * xb + h) * 4 + L.q) * 16));
        int4 B = __ldg((const int4 *)(m[j] + (((size_t)rb * nu + 4 * xb + h) * 4 + L.q) * 16));
#pragma unroll
        for (int hb = 0; hb < 2; hb++) {
          int c[4] = {0, 0, 0, 0};
          if (hb == 0) mma32(c, A.x, B.x, A.y, B.y, X[h].x, X[h].y);
          else mma32(c, A.z, B.z, A.w, B.w, X[h].z, X[h].w);
          float fa = h2f((unsigned)comp4(da, h), hb), fb = h2f((unsigned)comp4(db, h), hb);
          p[0] += fa * (float)c[0];
          p[1] += fa * (float)c[1];
          p[2] += fb * (float)c[2];
          p[3] += fb * (float)c[3];
        }
      }
      acc[j][0] += p[0] * dx0;
      acc[j][1] += p[1] * dx1;
      acc[j][2] += p[2] * dx0;
      acc[j][3] += p[3] * dx1;
    }
  }
}

// IQ2_XXS: x-block = 1 block = 8 k32 halves (one 32-block each). Lane q's bytes of a half are weights 4q..4q+3 and
// 16 + 4q..+3: half q & 1 of codebook groups q >> 1 and 2 + (q >> 1). Grid halves and sign masks come from shared
// tables; signs applied by (g ^ m) + (m & 1) per byte (one 32-bit add: magnitudes are <= 43, so no carries); integer scale 2 ls + 1 per half, d / 8 per block.
// Every thread of the block must call this (the tables are filled with a __syncthreads).
template <int NM>
__device__ __forceinline__ void tile_iq2(const uint8_t *const (&m)[NM], int rows, int k, int ra, int rb, const XQ &x,
                                         const Lane &L, float (&acc)[NM][4], bool fill) {
  unsigned *G32 = iq_tab, *S32 = iq_tab + 512;
  // the u64 grid is its 32-bit halves in order (little endian): plain copies
  if (fill) {
    tab_copy(G32, iq2xxs_grid, 512);
    tab_copy(S32, ksigns_mask32, 256);
    __syncthreads();
  }
  const int nb = k >> 8;
  const size_t pd = (size_t)rows * nb * 64;
  const int hf = L.q & 1, l0 = L.q >> 1, l1 = 2 + (L.q >> 1);
  for (int xb = L.k0; xb < nb; xb += KS) {
    int4 X[4];
    float dx0, dx1;
    load_x(x, k, xb, L, X, dx0, dx1);
#pragma unroll
    for (int j = 0; j < NM; j++) {
      const uint8_t *pa = m[j] + ((size_t)ra * nb + xb) * 64, *pb = m[j] + ((size_t)rb * nb + xb) * 64;
      float fa = __half2float(__ldg((const __half *)(m[j] + pd) + (size_t)ra * nb + xb));
      float fb = __half2float(__ldg((const __half *)(m[j] + pd) + (size_t)rb * nb + xb));
      int is[4] = {0, 0, 0, 0};
#pragma unroll
      for (int h = 0; h < 4; h++) {
        int4 A = (ABL & 4) ? make_int4(xb + h, xb * 3 + j, xb ^ h, xb * 7) : __ldg((const int4 *)pa + h);
        int4 B = (ABL & 4) ? make_int4(xb + 2 * h, xb * 5 + j, xb ^ 3 * h, xb * 9) : __ldg((const int4 *)pb + h);
#pragma unroll
        for (int e = 0; e < 2; e++) {
          unsigned ga = (unsigned)(e ? A.z : A.x), sa = (unsigned)(e ? A.w : A.y);
          unsigned gb = (unsigned)(e ? B.z : B.x), sb = (unsigned)(e ? B.w : B.y);
          auto dec = [&](unsigned g, unsigned sg, int l) {
            unsigned gi = (g >> (8 * l)) & 255, si = (sg >> (7 * l)) & 127;
            unsigned w = (ABL & 1) ? gi * 0x01010101u : G32[2 * gi + hf], mk = (ABL & 2) ? si * 0x01010101u : S32[2 * si + hf];
            return (w ^ mk) + (mk & 0x01010101u);  // -g where mk = 0xff: ~g + 1, no carries (g <= 43)
          };
          int c[4] = {0, 0, 0, 0};
          mma32(c, dec(ga, sa, l0), dec(gb, sb, l0), dec(ga, sa, l1), dec(gb, sb, l1), e ? X[h].z : X[h].x, e ? X[h].w : X[h].y);
          int la = 2 * (int)(sa >> 28) + 1, lb = 2 * (int)(sb >> 28) + 1;
          is[0] += la * c[0];
          is[1] += la * c[1];
          is[2] += lb * c[2];
          is[3] += lb * c[3];
        }
      }
      acc[j][0] += 0.125f * fa * dx0 * (float)is[0];
      acc[j][1] += 0.125f * fa * dx1 * (float)is[1];
      acc[j][2] += 0.125f * fb * dx0 * (float)is[2];
      acc[j][3] += 0.125f * fb * dx1 * (float)is[3];
    }
  }
}

// IQ1_S: x-block = 1 block, one k32 mma per 32-block. A weight is d (2s + 1) (g + delta), g in {-1, 0, 1}, delta =
// +-1/8: the A bytes are 8 g + 8 delta (exact int8, -9..9): 8 g (bytes -8 / 0 / 8) from the grid, plus or minus 1
// per byte (the -1 case biased by 0x80 so no byte borrows); integer scale 2 s + 1 per 32-block, d / 8 per block.
template <int NM>
__device__ __forceinline__ void tile_iq1s(const uint8_t *const (&m)[NM], int rows, int k, int ra, int rb, const XQ &x,
                                          const Lane &L, float (&acc)[NM][4]) {
  const int nb = k >> 8;
  const size_t pd = (size_t)rows * nb * 48;
  const int hf = L.q & 1, l0 = L.q >> 1, l1 = 2 + (L.q >> 1);
  for (int xb = L.k0; xb < nb; xb += KS) {
    int4 X[4];
    float dx0, dx1;
    load_x(x, k, xb, L, X, dx0, dx1);
#pragma unroll
    for (int j = 0; j < NM; j++) {
      const int4 *pa = (const int4 *)(m[j] + ((size_t)ra * nb + xb) * 48), *pb = (const int4 *)(m[j] + ((size_t)rb * nb + xb) * 48);
      int4 A0, A1, A2, B0, B1, B2;
      if (ABL & 4) {
        A0 = make_int4(xb, xb * 3 + j, xb ^ 5, xb * 7); A1 = make_int4(xb + 1, xb * 5, xb ^ 9, xb * 11); A2 = make_int4(xb * 13, xb + j, xb ^ 17, xb * 19);
        B0 = make_int4(xb + 2, xb * 23, xb ^ 29, xb * 31); B1 = make_int4(xb + 3, xb * 37, xb ^ 41, xb * 43); B2 = make_int4(xb * 47, xb + 4, xb ^ 53, xb * 59);
      } else {
        A0 = __ldg(pa), A1 = __ldg(pa + 1), A2 = __ldg(pa + 2), B0 = __ldg(pb), B1 = __ldg(pb + 1), B2 = __ldg(pb + 2);
      }
      float fa = __half2float(__ldg((const __half *)(m[j] + pd) + (size_t)ra * nb + xb));
      float fb = __half2float(__ldg((const __half *)(m[j] + pd) + (size_t)rb * nb + xb));
      int is[4] = {0, 0, 0, 0};
#pragma unroll
      for (int ib = 0; ib < 8; ib++) {
        unsigned qa = (unsigned)comp4(ib < 4 ? A0 : A1, ib & 3), qb = (unsigned)comp4(ib < 4 ? B0 : B1, ib & 3);
        unsigned ha = ((unsigned)comp4(A2, ib >> 1) >> (16 * (ib & 1))) & 0xffff, hb = ((unsigned)comp4(B2, ib >> 1) >> (16 * (ib & 1))) & 0xffff;
        auto dec = [&](unsigned qs, unsigned qh, int l) {
          // grid half as g + 1 nibbles through L1 (8 KB, no shared copy: filling one per block cost as much as the
          // weights); 8 (g + 1) + 0x80 - 7 (or - 9) stays in its byte, ^ 0x80 makes it the signed 8 g + 1 (or - 1)
          unsigned gi = ((qs >> (8 * l)) & 255) | ((qh >> (3 * l)) & 7) << 8;
          unsigned n = ((ABL & 1) ? gi * 0x01010101u : __ldg(iq1s_grid4 + gi) >> (4 * hf)) & 0x0f0f0f0fu;
          return ((n << 3) + ((qh & 0x8000) ? 0x77777777u : 0x79797979u)) ^ 0x80808080u;
        };
        int c[4] = {0, 0, 0, 0};
        mma32(c, dec(qa, ha, l0), dec(qb, hb, l0), dec(qa, ha, l1), dec(qb, hb, l1), (ib & 1) ? X[ib >> 1].z : X[ib >> 1].x,
              (ib & 1) ? X[ib >> 1].w : X[ib >> 1].y);
        int la = 2 * (int)((ha >> 12) & 7) + 1, lb = 2 * (int)((hb >> 12) & 7) + 1;
        is[0] += la * c[0];
        is[1] += la * c[1];
        is[2] += lb * c[2];
        is[3] += lb * c[3];
      }
      acc[j][0] += 0.125f * fa * dx0 * (float)is[0];
      acc[j][1] += 0.125f * fa * dx1 * (float)is[1];
      acc[j][2] += 0.125f * fb * dx0 * (float)is[2];
      acc[j][3] += 0.125f * fb * dx1 * (float)is[3];
    }
  }
}

// IQ2_XS / IQ2_S: x-block = 1 block; a scale per 16 weights, so one k16 mma per 16 (sub-block s of 32-block ib = groups
// 2s, 2s + 1; lane q's bytes are group 2s + (q >> 1), half q & 1). Magnitudes 8 / 25 / 43 from a shared grid table,
// signs by (g ^ m) + (m & 1); integer scale 2 sc + 1 per 16, d / 8 per block. IQ2_XS: 9-bit grid index and 7-bit
// parity sign index per group (u16); IQ2_S: 10-bit grid index (qs byte + 2 qh bits) and a full sign byte.
template <int NM, bool S>
__device__ __forceinline__ void tile_iq2xs(const uint8_t *const (&m)[NM], int rows, int k, int ra, int rb, const XQ &x,
                                           const Lane &L, float (&acc)[NM][4], bool fill) {
  constexpr int NG = S ? 1024 : 512;
  unsigned *G32 = iq_tab, *S32 = iq_tab + 2 * NG;
  if (fill) {
    tab_copy(G32, S ? (const void *)iq2s_grid : (const void *)iq2xs_grid, 2 * NG);
    if (!S) tab_copy(S32, ksigns_mask32, 256);  // IQ2_S signs are a plain byte, expanded inline
    __syncthreads();
  }
  const int nb = k >> 8;
  const size_t p2 = (size_t)rows * nb * 64, pd = p2 + (size_t)rows * nb * (S ? 16 : 8);
  const int hf = L.q & 1;
  for (int xb = L.k0; xb < nb; xb += KS) {
    int4 X[4];
    float dx0, dx1;
    load_x(x, k, xb, L, X, dx0, dx1);
#pragma unroll
    for (int j = 0; j < NM; j++) {
      const int4 *pa = (const int4 *)(m[j] + ((size_t)ra * nb + xb) * 64), *pb = (const int4 *)(m[j] + ((size_t)rb * nb + xb) * 64);
      int4 A[4], B[4];
#pragma unroll
      for (int h = 0; h < 4; h++) {
        A[h] = (ABL & 4) ? make_int4(xb + h, xb * 3 + j, xb ^ h, xb * 7 + h) : __ldg(pa + h);
        B[h] = (ABL & 4) ? make_int4(xb + 2 * h, xb * 5 + j, xb ^ 3 * h, xb * 9 + h) : __ldg(pb + h);
      }
      int4 EA, EB;  // IQ2_S: qh[8], scales[8]; IQ2_XS: scales[8] (x, y)
      if (S) {
        EA = __ldg((const int4 *)(m[j] + p2) + (size_t)ra * nb + xb);
        EB = __ldg((const int4 *)(m[j] + p2) + (size_t)rb * nb + xb);
      } else {
        int2 a2 = __ldg((const int2 *)(m[j] + p2) + (size_t)ra * nb + xb), b2 = __ldg((const int2 *)(m[j] + p2) + (size_t)rb * nb + xb);
        EA = make_int4(a2.x, a2.y, 0, 0);
        EB = make_int4(b2.x, b2.y, 0, 0);
      }
      float fa = __half2float(__ldg((const __half *)(m[j] + pd) + (size_t)ra * nb + xb));
      float fb = __half2float(__ldg((const __half *)(m[j] + pd) + (size_t)rb * nb + xb));
      int is[4] = {0, 0, 0, 0};
#pragma unroll
      for (int ib = 0; ib < 8; ib++) {
        // scale byte of the 32-block (low nibble: sub 0, high: sub 1)
        unsigned sca = ((unsigned)comp4(EA, (S ? 2 : 0) + (ib >> 2)) >> (8 * (ib & 3))) & 255;
        unsigned scb = ((unsigned)comp4(EB, (S ? 2 : 0) + (ib >> 2)) >> (8 * (ib & 3))) & 255;
#pragma unroll
        for (int sub = 0; sub < 2; sub++) {
          const int l = 2 * sub + (L.q >> 1);  // this lane's group
          auto dec = [&](const int4 (&W)[4], const int4 &E) {
            unsigned w, mk;
            if (S) {
              unsigned qs = ((unsigned)comp4(W[ib >> 2], ib & 3) >> (8 * l)) & 255;
              unsigned qh = ((unsigned)comp4(E, ib >> 2) >> (8 * (ib & 3))) & 255;
              unsigned sg = ((unsigned)comp4(W[2 + (ib >> 2)], ib & 3) >> (8 * l)) & 255;
              unsigned gi = qs | ((qh << (8 - 2 * l)) & 0x300);
              w = (ABL & 1) ? gi * 0x01010101u : G32[2 * gi + hf];
              mk = (ABL & 2) ? sg * 0x01010101u : ((((sg >> (4 * hf)) & 15) * 0x00204081u) & 0x01010101u) * 0xffu;
            } else {
              unsigned q = ((unsigned)comp4(W[ib >> 1], 2 * (ib & 1) + (l >> 1)) >> (16 * (l & 1))) & 0xffff;
              w = (ABL & 1) ? (q & 511) * 0x01010101u : G32[2 * (q & 511) + hf];
              mk = (ABL & 2) ? (q >> 9) * 0x01010101u : S32[2 * (q >> 9) + hf];
            }
            return (w ^ mk) + (mk & 0x01010101u);
          };
          int c[4];
          mma16(c, dec(A, EA), dec(B, EB), comp4(X[ib >> 1], 2 * (ib & 1) + sub));
          int la = 2 * (int)((sca >> (4 * sub)) & 15) + 1, lb = 2 * (int)((scb >> (4 * sub)) & 15) + 1;
          is[0] += la * c[0];
          is[1] += la * c[1];
          is[2] += lb * c[2];
          is[3] += lb * c[3];
        }
      }
      acc[j][0] += 0.125f * fa * dx0 * (float)is[0];
      acc[j][1] += 0.125f * fa * dx1 * (float)is[1];
      acc[j][2] += 0.125f * fb * dx0 * (float)is[2];
      acc[j][3] += 0.125f * fb * dx1 * (float)is[3];
    }
  }
}

// Q2_K / IQ3_XXS retain their original 84 / 98-byte blocks. The initial implementation favors
// auditable layouts: integer fragments are expanded in registers, not staged as float matrices.
template <bool IQ3, int NM>
__device__ __forceinline__ void tile_native(const uint8_t *const (&m)[NM], int rows, int k, int ra, int rb,
                                           const XQ &x, const Lane &L, float (&acc)[NM][4]) {
  const int nb = k / 256, bs = IQ3 ? 98 : 84;
  for (int xb = L.k0; xb < nb; xb += KS) {
    int4 X[4]; float dx0, dx1;
    load_x(x, k, xb, L, X, dx0, dx1);
    for (int j = 0; j < NM; j++) {
      const uint8_t *pa = m[j] + ((size_t)ra * nb + xb) * bs;
      const uint8_t *pb = m[j] + ((size_t)rb * nb + xb) * bs;
      int sum[4] = {0,0,0,0}, sub[4] = {0,0,0,0};
      for (int g = 0; g < 8; g++) {
        unsigned sa = IQ3 ? native_u32(pa + 66 + 4*g) : 0;
        unsigned sb = IQ3 ? native_u32(pb + 66 + 4*g) : 0;
        auto quad = [&](const uint8_t *p, unsigned sg, int q, bool minimum) -> unsigned {
          if (IQ3) {
            unsigned v = iq3xxs_grid[p[2 + g*8 + q]];
            unsigned si = ksigns_iq2xs[(sg >> (7*(q/2))) & 127] >> (4*(q%2));
            unsigned out = 0;
            for (int i=0; i<4; i++) {
              int z = (v >> (8*i)) & 255;
              if ((si >> i) & 1) z = -z;
              out |= (unsigned)(uint8_t)z << (8*i);
            }
            return out;
          } else {
            unsigned sc = p[2*g + q/4], out = 0;
            for (int i=0; i<4; i++) {
              unsigned v = minimum ? sc >> 4 : ((p[16 + (g/4)*32 + 4*q+i] >> (2*(g%4))) & 3) * (sc & 15);
              out |= v << (8*i);
            }
            return out;
          }
        };
        int c[4] = {0,0,0,0};
        int b0 = (g&1) ? X[g/2].z : X[g/2].x, b1 = (g&1) ? X[g/2].w : X[g/2].y;
        mma32(c, quad(pa,sa,L.q,false), quad(pb,sb,L.q,false), quad(pa,sa,L.q+4,false), quad(pb,sb,L.q+4,false), b0,b1);
        int la = IQ3 ? 2*(int)(sa>>28)+1 : 1, lb = IQ3 ? 2*(int)(sb>>28)+1 : 1;
        sum[0] += la*c[0]; sum[1] += la*c[1]; sum[2] += lb*c[2]; sum[3] += lb*c[3];
        if (!IQ3) mma32(sub, quad(pa,0,L.q,true), quad(pb,0,L.q,true), quad(pa,0,L.q+4,true), quad(pb,0,L.q+4,true), b0,b1);
      }
      float da = ld_h(pa + (IQ3 ? 0 : 80)) * (IQ3 ? 0.25f : 1.f);
      float db = ld_h(pb + (IQ3 ? 0 : 80)) * (IQ3 ? 0.25f : 1.f);
      float ma = IQ3 ? 0.f : ld_h(pa+82), mb = IQ3 ? 0.f : ld_h(pb+82);
      acc[j][0] += dx0 * (da*(float)sum[0] - ma*(float)sub[0]);
      acc[j][1] += dx1 * (da*(float)sum[1] - ma*(float)sub[1]);
      acc[j][2] += dx0 * (db*(float)sum[2] - mb*(float)sub[2]);
      acc[j][3] += dx1 * (db*(float)sum[3] - mb*(float)sub[3]);
    }
  }
}

template <int NM>
__device__ __forceinline__ void tile_dot(int ty, const uint8_t *const (&m)[NM], int rows, int k, int ra, int rb,
                                         const XQ &x, const Lane &L, float (&acc)[NM][4], bool fill = true) {
  if (ty == RP_Q2K) tile_native<false, NM>(m, rows, k, ra, rb, x, L, acc);
  else if (ty == RP_IQ3XXS) tile_native<true, NM>(m, rows, k, ra, rb, x, L, acc);
  else if (ty == RP_Q2) tile_q2<NM>(m, rows, k, ra, rb, x, L, acc);
  else if (ty == RP_T1) tile_t1<NM>(m, rows, k, ra, rb, x, L, acc);
  else if (ty == RP_Q3) tile_q3<NM>(m, rows, k, ra, rb, x, L, acc);
  else if (ty == RP_Q8) tile_q8<NM>(m, rows, k, ra, rb, x, L, acc);
  else if (ty == RP_IQ2) tile_iq2<NM>(m, rows, k, ra, rb, x, L, acc, fill);
  else if (ty == RP_IQ1S) tile_iq1s<NM>(m, rows, k, ra, rb, x, L, acc);
  else if (ty == RP_IQ2XS) tile_iq2xs<NM, false>(m, rows, k, ra, rb, x, L, acc, fill);
  else if (ty == RP_IQ2S) tile_iq2xs<NM, true>(m, rows, k, ra, rb, x, L, acc, fill);
}

// ---------------------------------------------------------------- activations
// x [nvec][k] -> q (permuted, see top), d, s (sums per 64). grid (k / 256, nvec), block 256.
// With sel != 0, block (0, 0) also writes the call's plan: the distinct resident experts among sel [T][K] in first-
// occurrence order, each with every token's pick index. plan[0] = count, entry u at plan + 1 + u * PLAN_W.
// With rot != 0, each 128-wide half of the block is first rotated by an orthonormal Sylvester Hadamard (experts
// stored as W R see R x; numerics::fwht_blocks on the CPU).
extern "C" __global__ void quant_act(const float *x, int k, int8_t *q, float *d, short *s, const int *sel,
                                     const ExpSlot *tab, int layer, int E, int K, int T, int *plan, int rot) {
  __shared__ float sh[8];
  __shared__ int ss[8], first[64];
  __shared__ float fb[256];
  int tid = threadIdx.x;
  size_t i = (size_t)blockIdx.y * k + blockIdx.x * 256 + tid;
  float v = x[i];
  if (rot) {
    fb[tid] = v;
    __syncthreads();
#pragma unroll
    for (int h = 1; h < 128; h <<= 1) {
      float own = fb[tid], other = fb[tid ^ h];
      __syncthreads();
      fb[tid] = (tid & h) ? other - own : own + other;
      __syncthreads();
    }
    v = fb[tid] * 0.08838834764831845f;  // 1 / sqrt(128)
  }
  float a = warp_max(fabsf(v));
  if ((tid & 31) == 0) sh[tid >> 5] = a;
  __syncthreads();
  a = sh[0];
#pragma unroll
  for (int j = 1; j < 8; j++) a = fmaxf(a, sh[j]);
  float dd = a / 127.f, id = dd > 0.f ? 1.f / dd : 0.f;
  int qv = __float2int_rn(v * id);
  int o = tid & 63;  // weight 16s + 4q + i -> byte 16q + 4s + i of its chunk
  q[i - o + 16 * ((o >> 2) & 3) + 4 * (o >> 4) + (o & 3)] = (int8_t)qv;
#pragma unroll
  for (int b = 1; b < 32; b <<= 1) qv += __shfl_xor_sync(0xffffffff, qv, b);
  if ((tid & 31) == 0) ss[tid >> 5] = qv;
  __syncthreads();
  if ((tid & 63) == 0) s[i >> 6] = (short)(ss[tid >> 5] + ss[(tid >> 5) + 1]);
  if (tid == 0) d[i >> 8] = dd;
  if (sel == nullptr || blockIdx.x != 0 || blockIdx.y != 0) return;
  int n = T * K, e = -1, f = 0;
  if (tid < n) {
    e = sel[tid];
    f = tab[layer * E + e].gate != ~0ull;
    for (int j = 0; j < tid; j++) f &= sel[j] != e;
  }
  if (tid < 64) first[tid] = f;
  __syncthreads();
  if (f) {
    int u = 0;
    for (int j = 0; j < tid; j++) u += first[j];
    int *ent = plan + 1 + u * PLAN_W;
    ent[0] = e;
    ent[1] = 0;
    for (int t = 0; t < 8; t++) {
      int jt = -1;
      if (t < T)
        for (int j = 0; j < K; j++)
          if (sel[t * K + j] == e) jt = j;
      ent[2 + t] = jt;
    }
  }
  if (tid == 0) {
    int c = 0;
    for (int j = 0; j < n; j++) c += first[j];
    plan[0] = c;
  }
}

// ---------------------------------------------------------------- single matrix (tests, benchmarks)
// sum acc over the block's KS warps into warp 0 (all threads call); returns true in warp 0
template <int NM>
__device__ __forceinline__ bool ks_sum(float (&acc)[NM][4]) {
  __shared__ float sh[KS - 1][NM * 4][32];
  int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (w)
#pragma unroll
    for (int j = 0; j < NM * 4; j++) sh[w - 1][j][lane] = acc[j >> 2][j & 3];
  __syncthreads();
  if (w) return false;
#pragma unroll
  for (int v = 0; v < KS - 1; v++)
#pragma unroll
    for (int j = 0; j < NM * 4; j++) acc[j >> 2][j & 3] += sh[v][j][lane];
  return true;
}

// y [T][rows] = W x [T][k] for a repacked matrix (type ty), x from quant_act. grid ceil(rows / 16), block 32 * KS
extern "C" __global__ void __launch_bounds__(32 * KS) xgemv(int ty, const uint8_t *w, int rows, int k, int T,
                                                            const int8_t *xq, const float *xd, const short *xs, float *y) {
  int lane = threadIdx.x & 31, g = lane >> 2;
  Lane L{lane & 3, g < T ? g : -1, 2 * (lane & 3) < T ? 2 * (lane & 3) : -1, 2 * (lane & 3) + 1 < T ? 2 * (lane & 3) + 1 : -1, (int)threadIdx.x >> 5};
  const uint8_t *m[1] = {w};
  for (int tt = 0; tt < ROWT; tt++) {
    int r0 = (blockIdx.x * ROWT + tt) * 16;
    if (r0 >= rows) break;
    int ra = min(r0 + g, rows - 1), rb = min(r0 + g + 8, rows - 1);
    float acc[1][4] = {{0.f, 0.f, 0.f, 0.f}};
    tile_dot<1>(ty, m, rows, k, ra, rb, XQ{xq, xd, xs}, L, acc, tt == 0);
    if (ks_sum<1>(acc))
#pragma unroll
      for (int i = 0; i < 4; i++) {
        int r = r0 + g + (i >> 1) * 8, t = (i & 1) ? L.c1 : L.c0;
        if (t >= 0 && r < rows) y[(size_t)t * rows + r] = acc[0][i];
      }
    if (ROWT > 1) __syncthreads();  // ks_sum's shared sums are rewritten by the next tile
  }
}

// ---------------------------------------------------------------- device-driven experts, T <= 8 tokens
// h [T][K][inter] = bf16(silu(min(gate x_t, lim)) * clamp(up x_t, -lim, lim) * w) for every (token, pick) whose expert
// is resident; one y-block per plan entry, so an expert picked by several tokens is read once.
// x: quant_act of x [T][dim] (which also wrote plan). grid (ceil(inter / 16), T * K), block 32 * KS.
extern "C" __global__ void __launch_bounds__(32 * KS) moe_gu_fast(const ExpSlot *tab, int layer, int E, const int *plan,
                                                             const float *wts, int K, const uint8_t *store,
                                                             const int8_t *xq, const float *xd, const short *xs,
                                                             float *h, int inter, int dim, float lim) {
  if ((int)blockIdx.y >= plan[0]) return;
  const int *ent = plan + 1 + blockIdx.y * PLAN_W;
  int lane = threadIdx.x & 31, g = lane >> 2, q = lane & 3;
  int j0 = ent[2 + 2 * q], j1 = ent[3 + 2 * q];
  Lane L{q, ent[2 + g] >= 0 ? g : -1, j0 >= 0 ? 2 * q : -1, j1 >= 0 ? 2 * q + 1 : -1, (int)threadIdx.x >> 5};
  ExpSlot s = tab[layer * E + ent[0]];
  const uint8_t *m[2] = {store + s.gate, store + s.up};
  for (int tt = 0; tt < ROWT; tt++) {
    int r0 = (blockIdx.x * ROWT + tt) * 16;
    if (r0 >= inter) break;
    float acc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
    tile_dot<2>(s.gu, m, inter, dim, min(r0 + g, inter - 1), min(r0 + g + 8, inter - 1), XQ{xq, xd, xs}, L, acc, tt == 0);
    if (ks_sum<2>(acc))
#pragma unroll
      for (int i = 0; i < 4; i++) {
        int r = r0 + g + (i >> 1) * 8, t = 2 * q + (i & 1), jt = (i & 1) ? j1 : j0;
        if (jt >= 0 && r < inter) {
          float gv = fminf(acc[0][i], lim), uv = fminf(fmaxf(acc[1][i], -lim), lim);
          h[((size_t)t * K + jt) * inter + r] = bf16r(gv / (1.f + expf(-gv)) * uv * wts[t * K + jt]);
        }
      }
    if (ROWT > 1) __syncthreads();
  }
}

// out [T][dim] = sum over each token's resident picks of down . h. hx: quant_act of h [T * K][inter].
// One y-block per plan entry writes its partial sums (part [64][8][dim]); the last block of a row tile to finish adds
// them in entry order (deterministic) into out. cnt [ceil(dim / 16)] zeroed once, reset by the last block.
// grid (ceil(dim / 16), T * K), block 32 * KS.
extern "C" __global__ void __launch_bounds__(32 * KS) moe_down_fast(const ExpSlot *tab, int layer, int E, const int *plan,
                                                               int K, const uint8_t *store, const int8_t *hq,
                                                               const float *hd, const short *hs, float *out, int inter,
                                                               int dim, int T, float *part, int *cnt) {
  __shared__ int last;
  int n = plan[0], u = blockIdx.y;
  int lane = threadIdx.x & 31, g = lane >> 2, q = lane & 3;
  if (n > 0 && u >= n) return;
  for (int tt = 0; tt < ROWT; tt++) {
    int ti = blockIdx.x * ROWT + tt, rt = ti * 16;
    if (rt >= dim) break;
    if (n == 0) {  // nothing resident: zeros
      if (u == 0)
        for (int i = threadIdx.x; i < 16 * T; i += 32 * KS)
          if (rt + (i & 15) < dim) out[(size_t)(i >> 4) * dim + rt + (i & 15)] = 0.f;
      continue;
    }
    const int *ent = plan + 1 + u * PLAN_W;
    {
      int j0 = ent[2 + 2 * q], j1 = ent[3 + 2 * q], jg = ent[2 + g];
      Lane L{q, jg >= 0 ? g * K + jg : -1, j0 >= 0 ? 2 * q * K + j0 : -1, j1 >= 0 ? (2 * q + 1) * K + j1 : -1, (int)threadIdx.x >> 5};
      ExpSlot s = tab[layer * E + ent[0]];
      const uint8_t *m[1] = {store + s.down};
      float acc[1][4] = {{0.f, 0.f, 0.f, 0.f}};
      tile_dot<1>(s.dn, m, dim, inter, min(rt + g, dim - 1), min(rt + g + 8, dim - 1), XQ{hq, hd, hs}, L, acc, tt == 0);
      if (ks_sum<1>(acc))
#pragma unroll
        for (int i = 0; i < 4; i++) {
          int r = rt + g + (i >> 1) * 8, t = 2 * q + (i & 1);
          if (r < dim) part[((size_t)u * 8 + t) * dim + r] = acc[0][i];  // inactive tokens: 0
        }
    }
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) last = atomicAdd(cnt + ti, 1) == n - 1;
    __syncthreads();
    if (last) {
      __threadfence();
      for (int i = threadIdx.x; i < 16 * T; i += 32 * KS) {
        int r = rt + (i & 15), t = i >> 4;
        if (r >= dim) continue;
        float v = 0.f;
        for (int e = 0; e < n; e++) v += __ldcg(part + ((size_t)e * 8 + t) * dim + r);
        out[(size_t)t * dim + r] = v;
      }
      if (threadIdx.x == 0) cnt[ti] = 0;
    }
    if (ROWT > 1) __syncthreads();  // `last` and ks_sum's shared sums are rewritten by the next tile
  }
}

// ---------------------------------------------------------------- upload-time repacking
__device__ __forceinline__ int q3_code(const uint8_t *b, int e) {  // signed value of weight e of a GGUF Q3_K block
  int h = e >> 7, j = (e >> 5) & 3, l = e & 31;
  return ((b[32 + 32 * h + l] >> (2 * j)) & 3) - (((b[l] >> (4 * h + j)) & 1) ? 0 : 4);
}
__device__ __forceinline__ uint8_t b3(const int *d, int n) {  // GGUF base-3 byte of n (4 or 5) trits 0..2
  unsigned v = 0;
  for (int i = 0; i < n; i++) v = v * 3 + d[i];
  if (n == 4) v *= 3;
  return (uint8_t)((v * 256 + 242) / 243);
}

// raw GGUF matrix (ty = ggml id) -> repacked layout; one thread per source block. grid ceil(blocks / 256), block 256
extern "C" __global__ void repack_expert(int ty, const uint8_t *src, uint8_t *dst, int rows, int k) {
  size_t bid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (ty == 10 || ty == 18) {
    size_t nb = k / 256;
    if (bid >= rows * nb) return;
    const int bs = ty == 10 ? 84 : 98;
    for (int i=0; i<bs; i++) dst[bid*bs+i] = src[bid*bs+i];
  } else if (ty == 42) {  // Q2_0
    size_t nb = k / 64, ng = k / 256;
    if (bid >= rows * nb) return;
    size_t r = bid / nb, b = bid % nb;
    const uint8_t *s = src + bid * 18;
    for (int q = 0; q < 4; q++) {
      unsigned v = 0;
      for (int g = 0; g < 4; g++)
        for (int i = 0; i < 4; i++) {
          int o = 16 * g + 4 * q + i;
          v |= (unsigned)((s[2 + o / 4] >> (2 * (o % 4))) & 3) << (8 * i + 2 * g);
        }
      *(unsigned *)(dst + ((r * ng + b / 4) * 4 + q) * 16 + (b % 4) * 4) = v;
    }
    dst[rows * nb * 16 + bid * 2] = s[0];
    dst[rows * nb * 16 + bid * 2 + 1] = s[1];
  } else if (ty == 143) {  // PTQ1_0
    size_t nb = k / 128, ng = k / 256;
    if (bid >= rows * nb) return;
    size_t r = bid / nb, b = bid % nb;
    const uint8_t *s = src + bid * 28;
    const uint32_t p3[5] = {1, 3, 9, 27, 81};
    int t[128];  // trits 0..2 in weight order
    for (int m = 0; m < 16; m++)
      for (int n = 0; n < 5; n++) t[m + 16 * n] = (((s[m] * p3[n]) & 0xFF) * 3) >> 8;
    for (int m = 0; m < 8; m++)
      for (int n = 0; n < 5; n++) t[80 + m + 8 * n] = (((s[16 + m] * p3[n]) & 0xFF) * 3) >> 8;
    for (int h = 0; h < 2; h++)
      for (int n = 0; n < 4; n++) t[120 + h + 2 * n] = (((s[24 + h] * p3[n]) & 0xFF) * 3) >> 8;
    uint8_t tail[2];
    for (int q = 0; q < 4; q++) {
      // group m = 4h + s: weights 64h + 16s + 4q + i
      auto G = [&](int m, int i) { return t[64 * (m >> 2) + 16 * (m & 3) + 4 * q + i]; };
      unsigned w = 0;
      for (int i = 0; i < 4; i++) {
        int d[5] = {G(0, i), G(1, i), G(2, i), G(3, i), G(4, i)};
        w |= (unsigned)b3(d, 5) << (8 * i);
      }
      unsigned short hw = 0;
      for (int j = 0; j < 2; j++) {
        int d[5] = {G(5, j), G(5, 2 + j), G(6, j), G(6, 2 + j), G(7, j)};
        hw |= (unsigned short)(b3(d, 5) << (8 * j));
      }
      size_t o = (r * ng + b / 2) * 4 + q;
      *(unsigned *)(dst + o * 8 + (b % 2) * 4) = w;
      *(unsigned short *)(dst + rows * nb * 16 + o * 4 + (b % 2) * 2) = hw;
      if (q & 1) {  // tail byte of the pair: trits 0, 1 = group 7 weights 2, 3 of q - 1; trits 2, 3 = those of q
        int d[4] = {t[112 + 4 * (q - 1) + 2], t[112 + 4 * (q - 1) + 3], G(7, 2), G(7, 3)};
        tail[q >> 1] = b3(d, 4);
      }
    }
    uint8_t *c = dst + rows * nb * 24 + bid * 4;
    c[0] = tail[0];
    c[1] = tail[1];
    c[2] = s[26];
    c[3] = s[27];
  } else if (ty == 11) {  // Q3_K
    size_t nb = k / 256;
    if (bid >= rows * nb) return;
    const uint8_t *s = src + bid * 110;
    for (int q = 0; q < 4; q++) {
      unsigned lo[4] = {0, 0, 0, 0}, nh[2] = {0, 0};
      for (int h = 0; h < 4; h++)
        for (int g = 0; g < 4; g++)
          for (int i = 0; i < 4; i++) {
            int v = q3_code(s, 64 * h + 16 * g + 4 * q + i);  // -4..3
            lo[h] |= (unsigned)(v & 3) << (8 * i + 2 * g);
            nh[h >> 1] |= (unsigned)(v < 0) << (8 * i + 4 * (h & 1) + g);
          }
      for (int h = 0; h < 4; h++) *(unsigned *)(dst + (bid * 4 + q) * 16 + 4 * h) = lo[h];
      for (int h = 0; h < 2; h++) *(unsigned *)(dst + rows * nb * 64 + (bid * 4 + q) * 8 + 4 * h) = nh[h];
    }
    for (int j = 0; j < 12; j++) dst[rows * nb * 96 + bid * 12 + j] = s[96 + j];
    dst[rows * nb * 108 + bid * 2] = s[108];
    dst[rows * nb * 108 + bid * 2 + 1] = s[109];
  } else if (ty == 19 || ty == 17 || ty == 22) {  // IQ1_S / IQ2_XS / IQ2_S: d moves out, the rest stays in order
    const int bs = ty == 19 ? 50 : ty == 17 ? 74 : 82, body = ty == 19 ? 48 : 64, tail = bs - 2 - body;
    size_t nb = k / 256;
    if (bid >= rows * nb) return;
    const uint8_t *s = src + bid * bs;
    for (int j = 0; j < body; j++) dst[bid * body + j] = s[2 + j];
    for (int j = 0; j < tail; j++) dst[rows * nb * body + bid * tail + j] = s[2 + body + j];
    dst[rows * nb * (body + tail) + bid * 2] = s[0];
    dst[rows * nb * (body + tail) + bid * 2 + 1] = s[1];
  } else if (ty == 16) {  // IQ2_XXS: d moves out, the 64 code bytes stay in order
    size_t nb = k / 256;
    if (bid >= rows * nb) return;
    const uint8_t *s = src + bid * 66;
    for (int j = 0; j < 64; j++) dst[bid * 64 + j] = s[2 + j];
    dst[rows * nb * 64 + bid * 2] = s[0];
    dst[rows * nb * 64 + bid * 2 + 1] = s[1];
  } else if (ty == 8) {  // Q8_0: thread per 32-block
    size_t nb = k / 32;
    if (bid >= rows * nb) return;
    size_t r = bid / nb, b = bid % nb, u = b / 2;
    const uint8_t *s = src + bid * 34;
    for (int o = 0; o < 32; o++) {
      int w = 32 * (b % 2) + o, g = w / 16, q = (w % 16) / 4, i = w % 4;
      dst[((r * (k / 64) + u) * 4 + q) * 16 + 4 * g + i] = s[2 + o];
    }
    dst[(size_t)rows * k + bid * 2] = s[0];
    dst[(size_t)rows * k + bid * 2 + 1] = s[1];
  }
}

// weight (r, c) of a repacked matrix (prefill's dequant_f16)
__device__ float rp_weight(int t, const uint8_t *w, int rows, int k, int r, int c) {
  int o = c & 63, g = o >> 4, q = (o >> 2) & 3, i = o & 3;  // group g, quad position q, byte i inside the chunk
  if (t == RP_Q2K || t == RP_IQ3XXS) {
    const uint8_t *p = w + ((size_t)r*(k/256)+c/256)*(t == RP_Q2K ? 84 : 98);
    return t == RP_Q2K ? q2k_value(p,c&255) : iq3xxs_value(p,c&255);
  } else if (t == RP_Q2) {
    size_t nb = k / 64, ng = k / 256, b = c / 64;
    unsigned v = *(const unsigned *)(w + (((size_t)r * ng + b / 4) * 4 + q) * 16 + (b % 4) * 4);
    int code = (v >> (8 * i + 2 * g)) & 3;
    return ld_h(w + rows * nb * 16 + ((size_t)r * nb + b) * 2) * (float)(code - 1);
  } else if (t == RP_T1) {
    size_t nb = k / 128, ng = k / 256, b = c / 128;
    int m = 4 * ((c & 127) >> 6) + g;  // group of this lane
    size_t ow = ((size_t)r * ng + b / 2) * 4 + q;
    const uint8_t *cc = w + rows * nb * 24 + ((size_t)r * nb + b) * 4;
    const uint32_t p3[5] = {1, 3, 9, 27, 81};
    int byte, n;
    if (m < 5) { byte = w[ow * 8 + (b % 2) * 4 + i]; n = m; }
    else if (m < 7) { byte = w[rows * nb * 16 + ow * 4 + (b % 2) * 2 + (i & 1)]; n = 2 * (m - 5) + (i >> 1); }
    else if (i < 2) { byte = w[rows * nb * 16 + ow * 4 + (b % 2) * 2 + i]; n = 4; }
    else { byte = cc[q >> 1]; n = 2 * (q & 1) + (i - 2); }
    return ld_h(cc + 2) * trit(byte, p3[n]);
  } else if (t == RP_Q3) {
    size_t nb = k / 256, b = c / 256;
    int e = c & 255, h = e >> 6;
    size_t ol = ((size_t)r * nb + b) * 4 + q;
    int low = (w[ol * 16 + 4 * h + i] >> (2 * g)) & 3;
    int neg = (w[rows * nb * 64 + ol * 8 + 4 * (h >> 1) + i] >> (4 * (h & 1) + g)) & 1;
    const uint8_t *raw = w + rows * nb * 96 + ((size_t)r * nb + b) * 12;
    int s16 = e / 16, i4 = s16 & 3, grp = s16 >> 2, a = raw[i4], cc = raw[4 + i4], tmp = raw[8 + i4], sc;
    if (grp == 0) sc = (a & 15) | ((tmp & 3) << 4);
    else if (grp == 1) sc = (cc & 15) | (((tmp >> 2) & 3) << 4);
    else if (grp == 2) sc = (a >> 4) | (((tmp >> 4) & 3) << 4);
    else sc = (cc >> 4) | (((tmp >> 6) & 3) << 4);
    return ld_h(w + rows * nb * 108 + ((size_t)r * nb + b) * 2) * (float)(sc - 32) * (float)(low - 4 * neg);
  } else if (t == RP_IQ1S || t == RP_IQ2XS || t == RP_IQ2S) {  // rebuild the GGUF block, then the reference decoder
    const int bs = t == RP_IQ1S ? 50 : t == RP_IQ2XS ? 74 : 82, body = t == RP_IQ1S ? 48 : 64, tail = bs - 2 - body;
    size_t nb = k / 256, b = (size_t)r * nb + c / 256;
    uint8_t blk[82];
    blk[0] = w[rows * nb * (body + tail) + b * 2];
    blk[1] = w[rows * nb * (body + tail) + b * 2 + 1];
    for (int j = 0; j < body; j++) blk[2 + j] = w[b * body + j];
    for (int j = 0; j < tail; j++) blk[2 + body + j] = w[rows * nb * body + b * tail + j];
    return t == RP_IQ1S ? iq1s_value(blk, c & 255) : t == RP_IQ2XS ? iq2xs_value(blk, c & 255) : iq2s_value(blk, c & 255);
  } else if (t == RP_IQ2) {
    size_t nb = k / 256, b = c / 256;
    const uint8_t *p = w + ((size_t)r * nb + b) * 64 + ((c & 255) >> 5) * 8;
    return iq2_value(p, ld_h(w + rows * nb * 64 + ((size_t)r * nb + b) * 2), c & 31);
  } else {  // RP_Q8
    size_t u = c / 64;
    int8_t v = (int8_t)w[(((size_t)r * (k / 64) + u) * 4 + q) * 16 + 4 * g + i];
    return (float)v * ld_h(w + (size_t)rows * k + ((size_t)r * (k / 32) + c / 32) * 2);
  }
}
