// cpu_isa.c - instruction throughput on this CPU (instructions per core clock, one thread pinned) for the operations a
// weight decoder can use: the CPU inputs of the decode cost model (docs/t16.md). Each test runs 8 independent chains;
// the core clock is calibrated with a dependent scalar add chain (1 cycle per add). Run on a P-core and an E-core.
//
// build: gcc -O2 -mavx2 -mavxvnni -mfma -o cpu_isa cpu_isa.c
// run:   ./cpu_isa CPU
#define _GNU_SOURCE
#include <immintrin.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
#define ITER 20000000L

// cycles per second from a dependent add chain (4 adds per iteration, 1 cycle each)
static double hz(void) {
  uint64_t x = 1;
  double t0 = now();
  for (long i = 0; i < ITER; i++) __asm__ volatile("add %0, %0\n add %0, %0\n add %0, %0\n add %0, %0" : "+r"(x));
  return 4.0 * ITER / (now() - t0);
}

// 8 independent ymm chains of instruction INS (AT&T: ins src, dst-in-place forms)
#define VTEST(name, body)                                                                     \
  {                                                                                           \
    __m256i a0 = _mm256_set1_epi32(1), a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0; \
    const __m256i k = _mm256_set1_epi32(0x01020304), sh = _mm256_set1_epi32(3);               \
    (void)k; (void)sh;                                                                        \
    double t0 = now();                                                                        \
    for (long i = 0; i < ITER / 4; i++) {                                                     \
      __asm__ volatile(body body body body                                                    \
                       : "+x"(a0), "+x"(a1), "+x"(a2), "+x"(a3), "+x"(a4), "+x"(a5), "+x"(a6), "+x"(a7) \
                       : "x"(k), "x"(sh));                                                    \
    }                                                                                         \
    const double dt = now() - t0;                                                             \
    printf("  %-34s %5.2f per clock\n", name, 8.0 * ITER / (dt * f));                         \
    sink ^= _mm256_extract_epi32(_mm256_xor_si256(_mm256_xor_si256(a0, a4), _mm256_xor_si256(a3, a7)), 0); \
  }
#define R8(ins) ins " %8, %0, %0\n" ins " %8, %1, %1\n" ins " %8, %2, %2\n" ins " %8, %3, %3\n" ins " %8, %4, %4\n" ins " %8, %5, %5\n" ins " %8, %6, %6\n" ins " %8, %7, %7\n"
#define R8S(ins) ins " %9, %0, %0\n" ins " %9, %1, %1\n" ins " %9, %2, %2\n" ins " %9, %3, %3\n" ins " %9, %4, %4\n" ins " %9, %5, %5\n" ins " %9, %6, %6\n" ins " %9, %7, %7\n"
#define R8I(ins) ins " $3, %0, %0\n" ins " $3, %1, %1\n" ins " $3, %2, %2\n" ins " $3, %3, %3\n" ins " $3, %4, %4\n" ins " $3, %5, %5\n" ins " $3, %6, %6\n" ins " $3, %7, %7\n"
#define R8V(ins) ins " %8, %0, %0\n" ins " %8, %1, %1\n" ins " %8, %2, %2\n" ins " %8, %3, %3\n" ins " %8, %4, %4\n" ins " %8, %5, %5\n" ins " %8, %6, %6\n" ins " %8, %7, %7\n"

int main(int argc, char **argv) {
  const int cpu = argc > 1 ? atoi(argv[1]) : 0;
  cpu_set_t s;
  CPU_ZERO(&s);
  CPU_SET(cpu, &s);
  sched_setaffinity(0, sizeof s, &s);
  volatile uint32_t sink = 0;
  hz();
  const double f = hz();
  printf("CPU %d: %.2f GHz (dependent add chain); throughput per core clock, one thread\n", cpu, f / 1e9);
  VTEST("vpshufb ymm", R8("vpshufb"));
  VTEST("vpsrlvd ymm (variable shift)", R8S("vpsrlvd"));
  VTEST("vpsrld ymm, imm", R8I("vpsrld"));
  VTEST("vpmullw ymm", R8("vpmullw"));
  VTEST("vpmulld ymm", R8("vpmulld"));
  VTEST("vpdpbusd ymm (AVX-VNNI)", R8V("%{vex%} vpdpbusd"));
  VTEST("vpaddd ymm", R8("vpaddd"));
  VTEST("vpand ymm", R8("vpand"));
  VTEST("vpermd ymm", R8("vpermd"));
  VTEST("vpmaddubsw ymm", R8("vpmaddubsw"));
  // loads: 8 independent 32-byte loads per step from an L1-resident buffer
  {
    static __attribute__((aligned(64))) uint8_t buf[4096];
    double t0 = now();
    __m256i acc = _mm256_setzero_si256();
    for (long i = 0; i < ITER / 4; i++) {
      const uint8_t *p = buf + ((i * 64) & 2047);
      __m256i a = _mm256_loadu_si256((const __m256i *)p), b = _mm256_loadu_si256((const __m256i *)(p + 32)), c = _mm256_loadu_si256((const __m256i *)(p + 64)),
              d = _mm256_loadu_si256((const __m256i *)(p + 96)), e = _mm256_loadu_si256((const __m256i *)(p + 128)), g = _mm256_loadu_si256((const __m256i *)(p + 160)),
              h = _mm256_loadu_si256((const __m256i *)(p + 192)), j = _mm256_loadu_si256((const __m256i *)(p + 224));
      acc = _mm256_xor_si256(acc, _mm256_xor_si256(_mm256_xor_si256(_mm256_xor_si256(a, b), _mm256_xor_si256(c, d)), _mm256_xor_si256(_mm256_xor_si256(e, g), _mm256_xor_si256(h, j))));
    }
    const double dt = now() - t0;
    printf("  %-34s %5.2f per clock (incl. 8 vpxor per 8 loads)\n", "vmovdqu ymm load (L1)", 8.0 * ITER / 4 / (dt * f));
    sink ^= _mm256_extract_epi32(acc, 0);
  }
  // gather: vpgatherdd of 8 dwords from an L1 table, 4 independent chains (the result is the next index)
  {
    static __attribute__((aligned(64))) int32_t tab[1024];
    for (int i = 0; i < 1024; i++) tab[i] = (int32_t)((i * 37u + 11u) & 1023u);
    __m256i i0 = _mm256_setr_epi32(1, 77, 300, 512, 9, 1000, 33, 640), i1 = _mm256_add_epi32(i0, _mm256_set1_epi32(1)),
            i2 = _mm256_add_epi32(i0, _mm256_set1_epi32(2)), i3 = _mm256_add_epi32(i0, _mm256_set1_epi32(3));
    double t0 = now();
    for (long i = 0; i < ITER / 8; i++) {
      __m256i m0 = _mm256_set1_epi32(-1), m1 = m0, m2 = m0, m3 = m0, d0, d1, d2, d3;
      __asm__ volatile("vpgatherdd %4, (%8,%0,4), %5\n vpgatherdd %6, (%8,%1,4), %7\n"
                       : "+x"(i0), "+x"(i1), "=&x"(d0), "=&x"(d1), "+x"(m0), "=x"(d0), "+x"(m1), "=x"(d1) : "r"(tab) : "memory");
      __asm__ volatile("vpgatherdd %4, (%8,%0,4), %5\n vpgatherdd %6, (%8,%1,4), %7\n"
                       : "+x"(i2), "+x"(i3), "=&x"(d2), "=&x"(d3), "+x"(m2), "=x"(d2), "+x"(m3), "=x"(d3) : "r"(tab) : "memory");
      i0 = d0, i1 = d1, i2 = d2, i3 = d3;
    }
    const double dt = now() - t0;
    printf("  %-34s %5.2f per clock (8 dwords each)\n", "vpgatherdd ymm (L1 table)", 4.0 * ITER / 8 / (dt * f));
    sink ^= _mm256_extract_epi32(_mm256_xor_si256(_mm256_xor_si256(i0, i1), _mm256_xor_si256(i2, i3)), 0);
  }
  // scalar: shrx and imul throughput, 8 independent chains
  {
    uint64_t r0 = 1, r1 = 2, r2 = 3, r3 = 4, r4 = 5, r5 = 6, r6 = 7, r7 = 8, c = 3;
    double t0 = now();
    for (long i = 0; i < ITER / 4; i++)
      for (int u = 0; u < 4; u++)
        __asm__ volatile("shrx %8, %0, %0\n shrx %8, %1, %1\n shrx %8, %2, %2\n shrx %8, %3, %3\n shrx %8, %4, %4\n shrx %8, %5, %5\n shrx %8, %6, %6\n shrx %8, %7, %7\n"
                         : "+r"(r0), "+r"(r1), "+r"(r2), "+r"(r3), "+r"(r4), "+r"(r5), "+r"(r6), "+r"(r7) : "r"(c));
    printf("  %-34s %5.2f per clock\n", "shrx r64", 8.0 * ITER / ((now() - t0) * f));
    sink ^= (uint32_t)(r0 ^ r7);
  }
  return (int)(sink & 0);
}
