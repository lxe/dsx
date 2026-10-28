// RAM read bandwidth: each thread streams its slice of a big buffer with AVX2 loads.
// gcc -O3 -mavx2 -fopenmp membw.c -o membw && ./membw [GB]
#include <immintrin.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static double now(void) { return omp_get_wtime(); }

int main(int argc, char **argv) {
  size_t gb = argc > 1 ? atol(argv[1]) : 8;
  size_t n = gb << 30;
  char *buf = aligned_alloc(64, n);
  memset(buf, 1, n);
  int counts[] = {1, 2, 4, 8, 12, 16, 20};
  for (int ci = 0; ci < 7; ci++) {
    int t = counts[ci];
    double best = 0;
    for (int rep = 0; rep < 3; rep++) {
      volatile long sink = 0;
      double t0 = now();
#pragma omp parallel num_threads(t)
      {
        int id = omp_get_thread_num();
        size_t chunk = (n / t) & ~(size_t)127, lo = id * chunk;
        __m256i acc = _mm256_setzero_si256(), acc2 = _mm256_setzero_si256();
        for (size_t i = lo; i < lo + chunk; i += 128) {
          acc = _mm256_xor_si256(acc, _mm256_load_si256((__m256i *)(buf + i)));
          acc2 = _mm256_xor_si256(acc2, _mm256_load_si256((__m256i *)(buf + i + 32)));
          acc = _mm256_xor_si256(acc, _mm256_load_si256((__m256i *)(buf + i + 64)));
          acc2 = _mm256_xor_si256(acc2, _mm256_load_si256((__m256i *)(buf + i + 96)));
        }
        acc = _mm256_xor_si256(acc, acc2);
        sink += _mm256_extract_epi64(acc, 0);
      }
      double dt = now() - t0;
      double bw = (double)((n / t) & ~(size_t)127) * t / dt / 1e9;
      if (bw > best) best = bw;
    }
    printf("threads %2d: %.1f GB/s\n", t, best);
  }
  return 0;
}
