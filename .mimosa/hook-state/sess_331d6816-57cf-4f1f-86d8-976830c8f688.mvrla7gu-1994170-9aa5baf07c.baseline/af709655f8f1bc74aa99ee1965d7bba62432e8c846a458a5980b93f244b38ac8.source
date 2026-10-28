// CPU expert matvec throughput for candidate low-bit formats, AVX2 + AVX-VNNI.
// One "expert" = w1,w3 (2304x5120) + w2 (5120x2304) = 35.4M weights. Activations are int8 per-32 blocks.
// Reports weight bytes streamed per second, which is what decode needs to keep up with RAM (~40 GB/s).
//
// Formats:
//   q2   : 64 weights = 16 B of 2-bit codes (u8 0..3) + fp16 scale + fp16 min  -> 2.5 bpw
//   q2s  : 256 weights = 64 B codes + 8 x u8 sub-scales + fp16 d (symmetric, code-2) -> 2.31 bpw
//   tq1  : 5 trits per byte (base-3, ggml TQ1_0 style), 160 weights = 32 B + fp16 d -> 1.70 bpw
//   q8   : 32 x int8 + fp16 scale -> 8.5 bpw (bandwidth reference)
//
// gcc -O3 -march=alderlake -fopenmp qdot.c -o qdot && ./qdot
#include <immintrin.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define K 5120
#define N 2304
#define NEXP_DEFAULT 64  // experts per run: 64 x ~10 MB = well beyond L3

static inline __m256i dot_u8s8(__m256i acc, __m256i u, __m256i s) {
  return _mm256_dpbusd_avx_epi32(acc, u, s);  // AVX-VNNI: acc += sum4(u8*s8)
}
static inline float hsum(__m256 v) {
  __m128 a = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
  a = _mm_hadd_ps(a, a);
  a = _mm_hadd_ps(a, a);
  return _mm_cvtss_f32(a);
}

// ---------- q2: 64-weight blocks, asymmetric ----------
typedef struct { uint8_t qs[16]; uint16_t d, m; } blk_q2;  // 20 B / 64 w
// activation: int8 x[K], per-64 sums precomputed for the min term, one global act scale
static float row_q2(const blk_q2 *w, const int8_t *x, const int32_t *xsum64, int k) {
  __m256 acc = _mm256_setzero_ps();
  float macc = 0;
  const __m256i m3 = _mm256_set1_epi8(3);
  for (int b = 0; b < k / 64; b++) {
    // 16 bytes -> 64 codes: byte j holds codes for positions j, j+16, j+32, j+48
    __m128i q = _mm_loadu_si128((const __m128i *)w[b].qs);
    __m256i qq = _mm256_set_m128i(_mm_srli_epi16(q, 2), q);
    __m256i lo = _mm256_and_si256(qq, m3);                       // pos 0..15, 16..31
    __m256i hi = _mm256_and_si256(_mm256_srli_epi16(qq, 4), m3); // pos 32..47, 48..63
    __m256i x0 = _mm256_loadu_si256((const __m256i *)(x + b * 64));
    __m256i x1 = _mm256_loadu_si256((const __m256i *)(x + b * 64 + 32));
    __m256i s = dot_u8s8(_mm256_setzero_si256(), lo, x0);
    s = dot_u8s8(s, hi, x1);
    acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s), _mm256_set1_ps(_cvtsh_ss(w[b].d)), acc);
    macc += _cvtsh_ss(w[b].m) * xsum64[b];
  }
  return hsum(acc) + macc;
}

// ---------- q2s: 256-weight super blocks, symmetric (code - 2), 8 sub-scales ----------
typedef struct { uint8_t qs[64]; uint8_t sc[8]; uint16_t d; } blk_q2s;  // 74 B / 256 w
static float row_q2s(const blk_q2s *w, const int8_t *x, const int32_t *xsum32, int k) {
  __m256 acc = _mm256_setzero_ps();
  const __m256i m3 = _mm256_set1_epi8(3);
  for (int b = 0; b < k / 256; b++) {
    const int8_t *xb = x + b * 256;
    __m256i q = _mm256_loadu_si256((const __m256i *)w[b].qs);
    __m256i q2 = _mm256_loadu_si256((const __m256i *)(w[b].qs + 32));
    __m256i c[8];
    c[0] = _mm256_and_si256(q, m3);
    c[1] = _mm256_and_si256(_mm256_srli_epi16(q, 2), m3);
    c[2] = _mm256_and_si256(_mm256_srli_epi16(q, 4), m3);
    c[3] = _mm256_and_si256(_mm256_srli_epi16(q, 6), m3);
    c[4] = _mm256_and_si256(q2, m3);
    c[5] = _mm256_and_si256(_mm256_srli_epi16(q2, 2), m3);
    c[6] = _mm256_and_si256(_mm256_srli_epi16(q2, 4), m3);
    c[7] = _mm256_and_si256(_mm256_srli_epi16(q2, 6), m3);
    int32_t isum = 0;
    __m256i tot = _mm256_setzero_si256();
    for (int j = 0; j < 8; j++) {
      __m256i s = dot_u8s8(_mm256_setzero_si256(), c[j], _mm256_loadu_si256((const __m256i *)(xb + 32 * j)));
      // subtract 2*sum(x) for the symmetric offset, scale by sub-scale
      s = _mm256_mullo_epi32(s, _mm256_set1_epi32(w[b].sc[j]));
      tot = _mm256_add_epi32(tot, s);
      isum += 2 * w[b].sc[j] * xsum32[b * 8 + j];
    }
    float d = _cvtsh_ss(w[b].d);
    acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(tot), _mm256_set1_ps(d), acc);
    acc = _mm256_add_ps(acc, _mm256_setr_ps(-d * isum, 0, 0, 0, 0, 0, 0, 0));
  }
  return hsum(acc);
}

// ---------- tq1: base-3, 5 trits per byte (ggml TQ1_0 trick) ----------
typedef struct { uint8_t qs[32]; uint16_t d; } blk_tq1;  // 34 B / 160 w
static float row_tq1(const blk_tq1 *w, const int8_t *x, const int32_t *xsum160, int k) {
  __m256 acc = _mm256_setzero_ps();
  const __m256i pow3[5] = {_mm256_set1_epi16(1), _mm256_set1_epi16(3), _mm256_set1_epi16(9),
                           _mm256_set1_epi16(27), _mm256_set1_epi16(81)};
  const __m256i lomask = _mm256_set1_epi16(0xFF);
  int nb = k / 160;
  for (int b = 0; b < nb; b++) {
    __m256i q = _mm256_loadu_si256((const __m256i *)w[b].qs);
    __m256i qlo = _mm256_and_si256(q, lomask);       // even bytes in u16 lanes
    __m256i qhi = _mm256_srli_epi16(q, 8);           // odd bytes
    __m256i s = _mm256_setzero_si256();
    for (int t = 0; t < 5; t++) {
      // trit = ((q * 3^t) & 0xFF) * 3 >> 8, done on 16-bit lanes for both halves, repacked to bytes
      __m256i a = _mm256_and_si256(_mm256_mullo_epi16(qlo, pow3[t]), lomask);
      __m256i c = _mm256_and_si256(_mm256_mullo_epi16(qhi, pow3[t]), lomask);
      a = _mm256_srli_epi16(_mm256_mullo_epi16(a, pow3[1]), 8);
      c = _mm256_srli_epi16(_mm256_mullo_epi16(c, pow3[1]), 8);
      __m256i tr = _mm256_or_si256(a, _mm256_slli_epi16(c, 8));  // 32 codes in 0..2, original byte order
      s = dot_u8s8(s, tr, _mm256_loadu_si256((const __m256i *)(x + b * 160 + 32 * t)));
    }
    acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s), _mm256_set1_ps(_cvtsh_ss(w[b].d)), acc);
    acc = _mm256_add_ps(acc, _mm256_setr_ps(-_cvtsh_ss(w[b].d) * xsum160[b], 0, 0, 0, 0, 0, 0, 0));
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
    // signed x signed via sign trick
    __m256i s = dot_u8s8(_mm256_setzero_si256(), _mm256_sign_epi8(q, q), _mm256_sign_epi8(xx, q));
    acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s), _mm256_set1_ps(_cvtsh_ss(w[b].d)), acc);
  }
  return hsum(acc);
}

typedef float (*rowfn)(const void *, const int8_t *, const int32_t *, int);

// k must be a multiple of the block size; 5120 and 2304 are multiples of 64/256 (2304 = 9*256) and 160 (5120 = 32*160; 2304 is not -> pad to 2400)
static void run(const char *name, rowfn f, size_t bsz, int bw, int nexp, int threads, const char *cpus) {
  int k1 = K, k2 = (N % bw) ? (N / bw + 1) * bw : N;  // padded inner dims
  size_t up_rows = 2 * N, dn_rows = K;                // w1+w3: 4608 rows of K; w2: 5120 rows of N
  size_t up_bytes = up_rows * (k1 / bw) * bsz, dn_bytes = dn_rows * (k2 / bw) * bsz;
  size_t per_exp = up_bytes + dn_bytes;
  uint8_t *w = aligned_alloc(64, per_exp * nexp);
  for (size_t i = 0; i < per_exp * nexp; i++) w[i] = (uint8_t)(i * 2654435761u >> 13);
  // keep scales sane (fp16 1.0 = 0x3C00) by overwriting is unnecessary for timing
  int8_t *x = aligned_alloc(64, 8192), *h = aligned_alloc(64, 8192);
  int32_t *xs = calloc(8192, 4);
  memset(x, 1, 8192);
  memset(h, 1, 8192);
  float *out = aligned_alloc(64, 8192 * 4 * threads);
  double best = 1e9;
  for (int rep = 0; rep < 5; rep++) {
    double t0 = omp_get_wtime();
    // one token's worth of experts: each expert's up rows split across threads, then down rows
#pragma omp parallel num_threads(threads)
    {
      for (int e = 0; e < nexp; e++) {
        const uint8_t *we = w + e * per_exp;
#pragma omp for schedule(static) nowait
        for (size_t r = 0; r < up_rows; r++) out[r % 8192] = f(we + r * (k1 / bw) * bsz, x, xs, k1);
#pragma omp for schedule(static)
        for (size_t r = 0; r < dn_rows; r++) out[r % 8192] = f(we + up_bytes + r * (k2 / bw) * bsz, h, xs, k2);
      }
    }
    double dt = omp_get_wtime() - t0;
    if (dt < best) best = dt;
  }
  double gbps = per_exp * nexp / best / 1e9;
  printf("%-5s %4.2f bpw  threads %2d (%s): %6.1f GB/s weights, %5.2f ms per expert, %5.1f ms per token (180 experts from RAM)\n",
         name, 8.0 * bsz / bw, threads, cpus, gbps, best / nexp * 1e3, best / nexp * 1e3 * 180);
  free(w); free(x); free(h); free(xs); free(out);
}

int main(int argc, char **argv) {
  int nexp = argc > 1 ? atoi(argv[1]) : NEXP_DEFAULT;
  int tl[] = {1, 8, 16, 20};
  for (int i = 0; i < 4; i++) {
    int t = tl[i];
    const char *c = t == 1 ? "1 P" : t == 8 ? "8 P, 1/core" : t == 16 ? "8 P + HT" : "all";
    run("q8", (rowfn)row_q8, sizeof(blk_q8), 32, nexp, t, c);
    run("q2", (rowfn)row_q2, sizeof(blk_q2), 64, nexp, t, c);
    run("q2s", (rowfn)row_q2s, sizeof(blk_q2s), 256, nexp, t, c);
    run("tq1", (rowfn)row_tq1, sizeof(blk_tq1), 160, nexp, t, c);
  }
  return 0;
}
