// CPU expert matvec throughput for llama.cpp's codebook formats (IQ1_S, IQ1_M, IQ2_XXS), AVX2 + AVX-VNNI, next to
// a Q8_0 bandwidth reference. Same harness as qdot.c: one "expert" = w1,w3 (2304x5120) + w2 (5120x2304) = 35.4M
// weights, 64 experts per run (far beyond L3), int8 activations, one token; reports weight bytes streamed per second.
// Decode keeps up with RAM (~40 GB/s on .102) only if a format decodes at least that fast; dsx's own kernels reach
// 40-42 GB/s for Q2_0 / PTQ1_0 / Q3_K at 1 token (bench/RESULTS.md, cpubench).
//
// Block layouts are ggml's (ggml-common.h). The codebooks (iq1s_grid: 2048 x 8 bytes, iq2xxs_grid: 256 x 8 bytes)
// and the 128 even-parity sign masks are filled with values of the right range, not ggml's tables: the contents do
// not change the speed (same table sizes, same lookups), and the weights are synthetic anyway.
//   iq2_xxs: 256 w = fp16 d + 8 x (4 grid bytes, 4 x 7-bit sign index + 4-bit scale)     66 B -> 2.06 bpw
//   iq1_s  : 256 w = fp16 d + 32 x 8-bit idx + 8 x u16 (4 x 3 high idx bits, 3-bit scale, delta sign)  50 B -> 1.56 bpw
//   iq1_m  : 256 w = 32 x 8-bit idx + 16 x (2 x 3 high bits + 2 delta signs) + 4 x u16 (3-bit scales, d in top nibbles)
//                                                                                        56 B -> 1.75 bpw
// The IQ1 grids hold -1/0/+1; stored here as 0/1/2 so VNNI's u8 x s8 applies, and the +1 offset folds into the
// delta term with per-4 activation sums (ggml's AVX2 kernel uses maddubs with a sign trick instead).
//
// gcc -O3 -march=alderlake -fopenmp iqdot.c -o iqdot && ./iqdot
#include <immintrin.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define K 5120
#define N 2304
#define NEXP_DEFAULT 64

static uint64_t iq1s_grid[2048], iq2xxs_grid[256], ksigns[128];

static void init_tables(void) {
  uint64_t s = 0x9E3779B97F4A7C15ull;
  for (int i = 0; i < 2048; i++) {
    uint64_t v = 0;
    for (int j = 0; j < 8; j++) {
      s ^= s << 13, s ^= s >> 7, s ^= s << 17;
      v |= (uint64_t)(s % 3) << (8 * j);  // 0/1/2 = -1/0/+1
    }
    iq1s_grid[i] = v;
  }
  const uint8_t lv[3] = {0x08, 0x19, 0x2b};
  for (int i = 0; i < 256; i++) {
    uint64_t v = 0;
    for (int j = 0; j < 8; j++) {
      s ^= s << 13, s ^= s >> 7, s ^= s << 17;
      v |= (uint64_t)lv[s % 3] << (8 * j);
    }
    iq2xxs_grid[i] = v;
  }
  for (int i = 0; i < 128; i++) {  // 7 sign bits + an even-parity 8th, as bytes 0x01 / 0xff for _mm256_sign_epi8
    int b = i | (__builtin_popcount(i) & 1) << 7;
    uint64_t v = 0;
    for (int j = 0; j < 8; j++) v |= (uint64_t)((b >> j) & 1 ? 0xff : 0x01) << (8 * j);
    ksigns[i] = v;
  }
}

static inline __m256i dot_u8s8(__m256i acc, __m256i u, __m256i s) { return _mm256_dpbusd_avx_epi32(acc, u, s); }
static inline float hsum(__m256 v) {
  __m128 a = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
  a = _mm_hadd_ps(a, a);
  a = _mm_hadd_ps(a, a);
  return _mm_cvtss_f32(a);
}

// ---------- iq2_xxs ----------
typedef struct { uint16_t d; uint16_t qs[32]; } blk_iq2xxs;  // 66 B / 256 w
static float row_iq2xxs(const blk_iq2xxs *w, const int8_t *x, const int32_t *unused, int k) {
  (void)unused;
  __m256 acc = _mm256_setzero_ps();
  for (int b = 0; b < k / 256; b++) {
    const uint32_t *q = (const uint32_t *)w[b].qs;
    const int8_t *xb = x + b * 256;
    __m256i tot = _mm256_setzero_si256();
    for (int j = 0; j < 8; j++) {
      uint32_t g = q[2 * j], sg = q[2 * j + 1];
      __m256i gv = _mm256_set_epi64x(iq2xxs_grid[g >> 24], iq2xxs_grid[(g >> 16) & 255], iq2xxs_grid[(g >> 8) & 255], iq2xxs_grid[g & 255]);
      __m256i sv = _mm256_set_epi64x(ksigns[(sg >> 21) & 127], ksigns[(sg >> 14) & 127], ksigns[(sg >> 7) & 127], ksigns[sg & 127]);
      __m256i xs = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *)(xb + 32 * j)), sv);
      __m256i s = dot_u8s8(_mm256_setzero_si256(), gv, xs);
      tot = _mm256_add_epi32(tot, _mm256_mullo_epi32(s, _mm256_set1_epi32(2 * (sg >> 28) + 1)));
    }
    // dl = d * (0.5 + ls) * 0.25 = d * (2 ls + 1) / 8
    acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(tot), _mm256_set1_ps(_cvtsh_ss(w[b].d) * 0.125f), acc);
  }
  return hsum(acc);
}

// ---------- iq1_s ----------
// xs4: per-4 activation sums (int32, k/4 of them), built once per token like dsx's per-dword sums
typedef struct { uint16_t d; uint8_t qs[32]; uint16_t qh[8]; } blk_iq1s;  // 50 B / 256 w
static float row_iq1s(const blk_iq1s *w, const int8_t *x, const int32_t *xs4, int k) {
  __m256 acc = _mm256_setzero_ps();
  for (int b = 0; b < k / 256; b++) {
    const int8_t *xb = x + b * 256;
    const uint8_t *qs = w[b].qs;
    __m256i tot = _mm256_setzero_si256();
    for (int j = 0; j < 8; j++) {
      uint32_t h = w[b].qh[j];
      __m256i gv = _mm256_set_epi64x(iq1s_grid[qs[4 * j + 3] | ((h >> 1) & 0x700)], iq1s_grid[qs[4 * j + 2] | ((h << 2) & 0x700)],
                                     iq1s_grid[qs[4 * j + 1] | ((h << 5) & 0x700)], iq1s_grid[qs[4 * j] | ((h << 8) & 0x700)]);
      __m256i s = dot_u8s8(_mm256_setzero_si256(), gv, _mm256_loadu_si256((const __m256i *)(xb + 32 * j)));
      // sum((g+1) x) * 8 + (8 delta - 8) * sum(x), delta = +-1/8
      __m256i x4 = _mm256_loadu_si256((const __m256i *)(xs4 + b * 64 + 8 * j));
      __m256i t = _mm256_add_epi32(_mm256_slli_epi32(s, 3), _mm256_mullo_epi32(x4, _mm256_set1_epi32(h & 0x8000 ? -9 : -7)));
      tot = _mm256_add_epi32(tot, _mm256_mullo_epi32(t, _mm256_set1_epi32(2 * ((h >> 12) & 7) + 1)));
    }
    acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(tot), _mm256_set1_ps(_cvtsh_ss(w[b].d) * 0.125f), acc);
  }
  return hsum(acc);
}

// ---------- iq1_m ----------
typedef struct { uint8_t qs[32]; uint8_t qh[16]; uint16_t sc[4]; } blk_iq1m;  // 56 B / 256 w
static float row_iq1m(const blk_iq1m *w, const int8_t *x, const int32_t *xs4, int k) {
  __m256 acc = _mm256_setzero_ps();
  for (int b = 0; b < k / 256; b++) {
    const int8_t *xb = x + b * 256;
    const uint8_t *qs = w[b].qs, *qh = w[b].qh;
    const uint16_t *sc = w[b].sc;
    uint16_t dh = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    __m256i tot = _mm256_setzero_si256();
    for (int j = 0; j < 8; j++) {  // 32 weights: 4 groups of 8 (own index + delta), 2 halves of 16 (own scale)
      uint32_t h0 = qh[2 * j], h1 = qh[2 * j + 1];
      __m256i gv = _mm256_set_epi64x(iq1s_grid[qs[4 * j + 3] | ((h1 << 4) & 0x700)], iq1s_grid[qs[4 * j + 2] | ((h1 << 8) & 0x700)],
                                     iq1s_grid[qs[4 * j + 1] | ((h0 << 4) & 0x700)], iq1s_grid[qs[4 * j] | ((h0 << 8) & 0x700)]);
      __m256i s = dot_u8s8(_mm256_setzero_si256(), gv, _mm256_loadu_si256((const __m256i *)(xb + 32 * j)));
      __m256i dl = _mm256_setr_epi32(h0 & 8 ? -9 : -7, h0 & 8 ? -9 : -7, h0 & 128 ? -9 : -7, h0 & 128 ? -9 : -7,
                                     h1 & 8 ? -9 : -7, h1 & 8 ? -9 : -7, h1 & 128 ? -9 : -7, h1 & 128 ? -9 : -7);
      __m256i x4 = _mm256_loadu_si256((const __m256i *)(xs4 + b * 64 + 8 * j));
      __m256i t = _mm256_add_epi32(_mm256_slli_epi32(s, 3), _mm256_mullo_epi32(x4, dl));
      uint32_t s6 = sc[j / 2] >> (6 * (j & 1));  // two 3-bit scales per 32 weights
      int l0 = 2 * (s6 & 7) + 1, l1 = 2 * ((s6 >> 3) & 7) + 1;
      tot = _mm256_add_epi32(tot, _mm256_mullo_epi32(t, _mm256_setr_epi32(l0, l0, l0, l0, l1, l1, l1, l1)));
    }
    acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(tot), _mm256_set1_ps(_cvtsh_ss(dh) * 0.125f), acc);
  }
  return hsum(acc);
}

// ---------- q8 reference ----------
typedef struct { int8_t qs[32]; uint16_t d; } blk_q8;  // 34 B / 32 w
static float row_q8(const blk_q8 *w, const int8_t *x, const int32_t *unused, int k) {
  (void)unused;
  __m256 acc = _mm256_setzero_ps();
  for (int b = 0; b < k / 32; b++) {
    __m256i q = _mm256_loadu_si256((const __m256i *)w[b].qs);
    __m256i xx = _mm256_loadu_si256((const __m256i *)(x + b * 32));
    __m256i s = dot_u8s8(_mm256_setzero_si256(), _mm256_sign_epi8(q, q), _mm256_sign_epi8(xx, q));
    acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s), _mm256_set1_ps(_cvtsh_ss(w[b].d)), acc);
  }
  return hsum(acc);
}

typedef float (*rowfn)(const void *, const int8_t *, const int32_t *, int);

static void run(const char *name, rowfn f, size_t bsz, int bw, int nexp, int threads, const char *cpus) {
  size_t up_rows = 2 * N, dn_rows = K;  // w1+w3: 4608 rows of K; w2: 5120 rows of N
  size_t up_bytes = up_rows * (K / bw) * bsz, dn_bytes = dn_rows * (N / bw) * bsz;
  size_t per_exp = up_bytes + dn_bytes;
  uint8_t *w = aligned_alloc(64, per_exp * nexp);
  for (size_t i = 0; i < per_exp * nexp; i++) w[i] = (uint8_t)(i * 2654435761u >> 13);
  int8_t *x = aligned_alloc(64, 8192), *h = aligned_alloc(64, 8192);
  int32_t *xs = aligned_alloc(64, 8192 * 4);
  memset(x, 1, 8192);
  memset(h, 1, 8192);
  for (int i = 0; i < 8192; i++) xs[i] = 4;
  float *out = aligned_alloc(64, 8192 * 4 * threads);
  double best = 1e9;
  for (int rep = 0; rep < 5; rep++) {
    double t0 = omp_get_wtime();
#pragma omp parallel num_threads(threads)
    {
      for (int e = 0; e < nexp; e++) {
        const uint8_t *we = w + e * per_exp;
#pragma omp for schedule(static) nowait
        for (size_t r = 0; r < up_rows; r++) out[r % 8192] = f(we + r * (K / bw) * bsz, x, xs, K);
#pragma omp for schedule(static)
        for (size_t r = 0; r < dn_rows; r++) out[r % 8192] = f(we + up_bytes + r * (N / bw) * bsz, h, xs, N);
      }
    }
    double dt = omp_get_wtime() - t0;
    if (dt < best) best = dt;
  }
  double gbps = per_exp * nexp / best / 1e9;
  printf("%-7s %4.2f bpw  threads %2d (%s): %6.1f GB/s weights, %5.2f ms per expert, %5.2f ms per expert at 40 GB/s\n",
         name, 8.0 * bsz / bw, threads, cpus, gbps, best / nexp * 1e3, per_exp / 40e9 * 1e3);
  free(w); free(x); free(h); free(xs); free(out);
}

int main(int argc, char **argv) {
  int nexp = argc > 1 ? atoi(argv[1]) : NEXP_DEFAULT;
  init_tables();
  int tl[] = {1, 8, 16, 20};
  for (int i = 0; i < 4; i++) {
    int t = tl[i];
    const char *c = t == 1 ? "1 P" : t == 8 ? "8 P, 1/core" : t == 16 ? "8 P + HT" : "all";
    run("q8", (rowfn)row_q8, sizeof(blk_q8), 32, nexp, t, c);
    run("iq2_xxs", (rowfn)row_iq2xxs, sizeof(blk_iq2xxs), 256, nexp, t, c);
    run("iq1_s", (rowfn)row_iq1s, sizeof(blk_iq1s), 256, nexp, t, c);
    run("iq1_m", (rowfn)row_iq1m, sizeof(blk_iq1m), 256, nexp, t, c);
  }
  return 0;
}
