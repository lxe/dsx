// trellis_floor.c - lower-bound speed of a mul1 bitshift-trellis expert GEMV on this CPU (single token).
// Per 8 weights: broadcast-load one u32 of the packed stream, vpsrlvd + vpand to cut the eight 16-bit
// windows (K bits apart), vpmulld by the mul1 constant, vpdpbusd (AVX-VNNI) of the product's 4 bytes with
// the weight's activation replicated 4x. Synthetic data, real sizes: experts of gate/up [2304 x 5120] and
// down [5120 x 2304], K = 2 or 3 bits/weight, a pool of experts larger than the caches, calls of 6 experts.
// It computes the decode work exactly; the bit layout is simplified (one u32 per 8 weights at K = 2, a
// second u32 when the windows span past 32 bits at K = 3), so it measures cost, not EXL3 compatibility.
//
// build: gcc -O3 -mavx2 -mavxvnni -mfma -fopenmp trellis_floor.c -o trellis_floor
// run:   OMP_PROC_BIND=... ./trellis_floor K experts calls threads [dynamic]
#include <immintrin.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define HID 5120
#define INT 2304
#define MUL1 0x83DCD12Bu   // any odd 32-bit constant: cost is what matters

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

// one matrix: rows x k weights, K bits each; stream words per row = k*K/32 (+ pad)
typedef struct { uint32_t *w; int rows, k, K, wpr; } Mat;

static Mat mk(int rows, int k, int K) {
  Mat m = {0, rows, k, K, k * K / 32 + 2};
  size_t n = (size_t)rows * m.wpr;
  m.w = aligned_alloc(64, n * 4);
  for (size_t i = 0; i < n; i++) m.w[i] = (uint32_t)(i * 2654435761u) ^ 0x5bd1e995u;
  return m;
}

// out[r] = sum_c dec(r, c) * x[c]; act4 = activations as int8 replicated 4x per weight (k*4 bytes)
static void gemv(const Mat *m, const int8_t *act4, int32_t *out, int r0, int r1) {
  const __m256i shK = m->K == 2 ? _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14) : _mm256_setr_epi32(0, 3, 6, 9, 12, 15, 0, 0);
  const __m256i lo16 = _mm256_set1_epi32(0xFFFF), mul = _mm256_set1_epi32((int)MUL1);
  for (int r = r0; r < r1; r++) {
    const uint32_t *s = m->w + (size_t)r * m->wpr;
    __m256i acc0 = _mm256_setzero_si256(), acc1 = _mm256_setzero_si256();
    if (m->K == 2) {
      // 8 weights per u32 (16 new bits + overlap): 4 groups per loop for ILP
      for (int c = 0, wi = 0; c < m->k; c += 32, wi += 4) {
        for (int g = 0; g < 4; g++) {
          __m256i st = _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32((int)s[wi + g]), shK), lo16);
          __m256i pr = _mm256_mullo_epi32(st, mul);
          __m256i a = _mm256_loadu_si256((const __m256i *)(act4 + 4 * (c + 8 * g)));
          if (g & 1) acc1 = _mm256_dpbusd_avx_epi32(acc1, pr, a); else acc0 = _mm256_dpbusd_avx_epi32(acc0, pr, a);
        }
      }
    } else {
      // K = 3: 8 weights span 24 new bits; two u32s give the two halves of the window set
      for (int c = 0, bit = 0; c < m->k; c += 8, bit += 24) {
        uint64_t v; memcpy(&v, (const uint8_t *)s + (bit >> 3), 8);
        __m256i b = _mm256_setr_epi32((int)v, (int)v, (int)v, (int)v, (int)v, (int)v, (int)(v >> 18), (int)(v >> 21));
        __m256i st = _mm256_and_si256(_mm256_srlv_epi32(b, shK), lo16);
        __m256i pr = _mm256_mullo_epi32(st, mul);
        __m256i a = _mm256_loadu_si256((const __m256i *)(act4 + 4 * c));
        if ((c >> 3) & 1) acc1 = _mm256_dpbusd_avx_epi32(acc1, pr, a); else acc0 = _mm256_dpbusd_avx_epi32(acc0, pr, a);
      }
    }
    __m256i t = _mm256_add_epi32(acc0, acc1);
    __m128i h = _mm_add_epi32(_mm256_castsi256_si128(t), _mm256_extracti128_si256(t, 1));
    h = _mm_hadd_epi32(h, h); h = _mm_hadd_epi32(h, h);
    out[r] = _mm_cvtsi128_si32(h);
  }
}


// K = 2, four rows at once: each activation load feeds four rows' decode (more independent multiplies in flight)
static void gemv4(const Mat *m, const int8_t *act4, int32_t *out, int r0, int r1) {
  const __m256i shK = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
  const __m256i lo16 = _mm256_set1_epi32(0xFFFF), mul = _mm256_set1_epi32((int)MUL1);
  for (int r = r0; r + 4 <= r1; r += 4) {
    const uint32_t *s0 = m->w + (size_t)r * m->wpr, *s1 = s0 + m->wpr, *s2 = s1 + m->wpr, *s3 = s2 + m->wpr;
    __m256i a0 = _mm256_setzero_si256(), a1 = a0, a2 = a0, a3 = a0;
    for (int c = 0, wi = 0; c < m->k; c += 8, wi++) {
      __m256i a = _mm256_loadu_si256((const __m256i *)(act4 + 4 * c));
#define ROW(S, A) A = _mm256_dpbusd_avx_epi32(A, _mm256_mullo_epi32(_mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32((int)S[wi]), shK), lo16), mul), a)
      ROW(s0, a0); ROW(s1, a1); ROW(s2, a2); ROW(s3, a3);
#undef ROW
    }
    __m256i acc[4] = {a0, a1, a2, a3};
    for (int q = 0; q < 4; q++) {
      __m128i h = _mm_add_epi32(_mm256_castsi256_si128(acc[q]), _mm256_extracti128_si256(acc[q], 1));
      h = _mm_hadd_epi32(h, h); h = _mm_hadd_epi32(h, h); out[r + q] = _mm_cvtsi128_si32(h);
    }
  }
}

// "mul16" trellis, K = 2, 16-bit lanes: one 32-byte load + one pshufb build 16 windows of 16 bits per half-pair;
// four right-shifts by 0/2/4/6 give the four weight classes (64 weights per load); vpmullw by the 16-bit
// constant; vpdpbusd of the product's 2 bytes with the weight's activation replicated 2x (two weights per dword).
// Stream: 16 new bytes per 64 weights (2 bits/weight); activations: 2 bytes per weight in matching order.
#define MUL16 0x3A2F
static void gemv16(const Mat *m, const int8_t *act2, int32_t *out, int r0, int r1) {
  // lo lane j = bytes (j, j+1), hi lane j = bytes (j+1, j+2): the 16-bit window at bit offset 8j + 2r is
  // (lo >> 2r) | (hi << (16 - 2r)), so every window holds real stream bits
  const __m256i ilo = _mm256_setr_epi8(0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8, 0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8);
  const __m256i ihi = _mm256_setr_epi8(1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9, 1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9);
  const __m256i mul = _mm256_set1_epi16((short)MUL16);
  for (int r = r0; r < r1; r++) {
    const uint8_t *s = (const uint8_t *)(m->w + (size_t)r * m->wpr);
    __m256i a0 = _mm256_setzero_si256(), a1 = a0, a2 = a0, a3 = a0;
    for (int c = 0, b = 0; c < m->k; c += 64, b += 16) {
      __m256i src = _mm256_loadu_si256((const __m256i *)(s + b));
      __m256i lo = _mm256_shuffle_epi8(src, ilo), hi = _mm256_shuffle_epi8(src, ihi);
      const int8_t *a = act2 + 2 * c;
      a0 = _mm256_dpbusd_avx_epi32(a0, _mm256_mullo_epi16(lo, mul), _mm256_loadu_si256((const __m256i *)(a)));
      a1 = _mm256_dpbusd_avx_epi32(a1, _mm256_mullo_epi16(_mm256_or_si256(_mm256_srli_epi16(lo, 2), _mm256_slli_epi16(hi, 14)), mul), _mm256_loadu_si256((const __m256i *)(a + 32)));
      a2 = _mm256_dpbusd_avx_epi32(a2, _mm256_mullo_epi16(_mm256_or_si256(_mm256_srli_epi16(lo, 4), _mm256_slli_epi16(hi, 12)), mul), _mm256_loadu_si256((const __m256i *)(a + 64)));
      a3 = _mm256_dpbusd_avx_epi32(a3, _mm256_mullo_epi16(_mm256_or_si256(_mm256_srli_epi16(lo, 6), _mm256_slli_epi16(hi, 10)), mul), _mm256_loadu_si256((const __m256i *)(a + 96)));
    }
    __m256i t = _mm256_add_epi32(_mm256_add_epi32(a0, a1), _mm256_add_epi32(a2, a3));
    __m128i h = _mm_add_epi32(_mm256_castsi256_si128(t), _mm256_extracti128_si256(t, 1));
    h = _mm_hadd_epi32(h, h); h = _mm_hadd_epi32(h, h); out[r] = _mm_cvtsi128_si32(h);
  }
}

int main(int argc, char **argv) {
  int K = argc > 1 ? atoi(argv[1]) : 2, E = argc > 2 ? atoi(argv[2]) : 64, calls = argc > 3 ? atoi(argv[3]) : 200;
  int th = argc > 4 ? atoi(argv[4]) : 8, dyn = argc > 5 ? atoi(argv[5]) : 0, rows4 = argc > 6 ? atoi(argv[6]) : 0;
  void (*G)(const Mat *, const int8_t *, int32_t *, int, int) = rows4 == 2 ? gemv16 : (rows4 && K == 2) ? gemv4 : gemv;
  omp_set_num_threads(th);
  Mat *g = malloc(E * sizeof(Mat)), *u = malloc(E * sizeof(Mat)), *d = malloc(E * sizeof(Mat));
  for (int e = 0; e < E; e++) { g[e] = mk(INT, HID, K); u[e] = mk(INT, HID, K); d[e] = mk(HID, INT, K); }
  int8_t *ax = aligned_alloc(64, HID * 4), *ah = aligned_alloc(64, INT * 4);
  for (int i = 0; i < HID * 4; i++) ax[i] = (int8_t)(i * 37);
  for (int i = 0; i < INT * 4; i++) ah[i] = (int8_t)(i * 53);
  int32_t *og = aligned_alloc(64, INT * 4 * 6), *ou = aligned_alloc(64, INT * 4 * 6), *od = aligned_alloc(64, HID * 4 * 6);
  size_t bytes_e = ((size_t)g[0].rows * g[0].wpr + (size_t)u[0].rows * u[0].wpr + (size_t)d[0].rows * d[0].wpr) * 4;
  unsigned rs = 12345;
  double best = 1e9, sum = 0; int n = 0;
  for (int it = 0; it < calls + 20; it++) {
    int sel[6];
    for (int j = 0; j < 6; j++) { rs = rs * 1103515245u + 12345u; sel[j] = (rs >> 8) % E; }
    double t0 = now();
    // gate+up rows of the 6 experts, then down rows (the activation between them is negligible here)
    #pragma omp parallel
    {
      if (dyn) {
        #pragma omp for schedule(dynamic, 1) collapse(2)
        for (int j = 0; j < 6; j++) for (int r = 0; r < INT; r += 64) { G(&g[sel[j]], ax, og + j * INT, r, r + 64); G(&u[sel[j]], ax, ou + j * INT, r, r + 64); }
        #pragma omp for schedule(dynamic, 1) collapse(2)
        for (int j = 0; j < 6; j++) for (int r = 0; r < HID; r += 128) G(&d[sel[j]], ah, od + j * HID, r, r + 128);
      } else {
        #pragma omp for schedule(static) collapse(2)
        for (int j = 0; j < 6; j++) for (int r = 0; r < INT; r += 64) { G(&g[sel[j]], ax, og + j * INT, r, r + 64); G(&u[sel[j]], ax, ou + j * INT, r, r + 64); }
        #pragma omp for schedule(static) collapse(2)
        for (int j = 0; j < 6; j++) for (int r = 0; r < HID; r += 128) G(&d[sel[j]], ah, od + j * HID, r, r + 128);
      }
    }
    double dt = now() - t0;
    if (it >= 20) { sum += dt; n++; if (dt < best) best = dt; }
  }
  double avg = sum / n;
  printf("K=%d threads=%d %s%s: %.2f ms per 6-expert call (best %.2f), %.1f GB/s, %.1f G weights/s  [%.1f MB/expert]  (chk %d)\n",
         K, th, dyn ? "dynamic" : "static ", rows4 == 2 ? " mul16" : rows4 ? " 4-row" : "", avg * 1e3, best * 1e3, 6 * bytes_e / avg / 1e9,
         6.0 * 3 * HID * INT / avg / 1e9, bytes_e / 1e6, og[0] ^ ou[1] ^ od[2]);
  return 0;
}
