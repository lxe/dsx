// engram_q6k.c - one Engram layer of the official checkpoint (FP8 E4M3 rows of 256 with an E8M0 scale per 32) into
// Q6_K rows (one 210-byte block per row) at an offset of an output file. Each value is decoded as dsx's FP8 path
// does (fp8 * 2^e, rounded to bf16), then quantized by ggml's own Q6_K quantizer (no imatrix).
//
// build: cc -O3 -fopenmp engram_q6k.c -I LLAMA/ggml/include -L LLAMA/build/bin -lggml-base -Wl,-rpath,LLAMA/build/bin -o engram_q6k
// run:   engram_q6k SHARD W_OFF S_OFF ROWS OUT OUT_OFF   (offsets in bytes; engram_gguf.py computes them)
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "ggml.h"

#define DIM 256
#define CHUNK 8192  // rows per work item
#define Q6_ROW 210

static float lut[256];

static float fp8_e4m3(uint8_t b) {
  const float s = b & 0x80 ? -1.f : 1.f;
  const int e = (b >> 3) & 15, m = b & 7;
  if (e == 0) return s * m * ldexpf(1.f, -9);
  if (e == 15 && m == 7) return NAN;
  return s * (1.f + m / 8.f) * ldexpf(1.f, e - 7);
}

// round to nearest even bf16, as half::bf16::from_f32 (dsx's round_bf16)
static float bf16_round(float x) {
  uint32_t u;
  memcpy(&u, &x, 4);
  u += 0x7FFF + ((u >> 16) & 1);
  u &= 0xFFFF0000u;
  memcpy(&x, &u, 4);
  return x;
}

int main(int argc, char **argv) {
  if (argc != 7) {
    fprintf(stderr, "usage: %s SHARD W_OFF S_OFF ROWS OUT OUT_OFF\n", argv[0]);
    return 2;
  }
  const char *shard = argv[1], *out = argv[5];
  const uint64_t w_off = strtoull(argv[2], 0, 10), s_off = strtoull(argv[3], 0, 10), rows = strtoull(argv[4], 0, 10),
                 o_off = strtoull(argv[6], 0, 10);
  for (int i = 0; i < 256; i++) lut[i] = fp8_e4m3((uint8_t)i);
  const int fd = open(shard, O_RDONLY);
  struct stat st;
  if (fd < 0 || fstat(fd, &st)) { perror(shard); return 1; }
  if (w_off + rows * DIM > (uint64_t)st.st_size || s_off + rows * (DIM / 32) > (uint64_t)st.st_size) {
    fprintf(stderr, "offsets past the end of %s\n", shard);
    return 1;
  }
  const uint8_t *base = mmap(0, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
  if (base == MAP_FAILED) { perror("mmap"); return 1; }
  madvise((void *)base, st.st_size, MADV_SEQUENTIAL);
  const int ofd = open(out, O_WRONLY);
  if (ofd < 0) { perror(out); return 1; }
  if (ggml_row_size(GGML_TYPE_Q6_K, DIM) != Q6_ROW) { fprintf(stderr, "unexpected Q6_K row size\n"); return 1; }
  const uint8_t *w = base + w_off, *sc = base + s_off;
  const int64_t nchunks = (rows + CHUNK - 1) / CHUNK;
  int64_t done = 0, bad = 0;
#pragma omp parallel
  {
    float *f = malloc((size_t)CHUNK * DIM * sizeof(float));
    uint8_t *q = malloc((size_t)CHUNK * Q6_ROW);
#pragma omp for schedule(dynamic, 1)
    for (int64_t c = 0; c < nchunks; c++) {
      const uint64_t r0 = (uint64_t)c * CHUNK, n = rows - r0 < CHUNK ? rows - r0 : CHUNK;
      int64_t nan = 0;
      for (uint64_t r = 0; r < n; r++) {
        const uint8_t *wr = w + (r0 + r) * DIM, *sr = sc + (r0 + r) * (DIM / 32);
        float *fr = f + r * DIM;
        for (int g = 0; g < DIM / 32; g++) {
          const float s = ldexpf(1.f, (int)sr[g] - 127);
          for (int j = 0; j < 32; j++) {
            float v = bf16_round(lut[wr[32 * g + j]] * s);
            if (isnan(v)) v = 0.f, nan++;
            fr[32 * g + j] = v;
          }
        }
      }
      ggml_quantize_chunk(GGML_TYPE_Q6_K, f, q, 0, n, DIM, NULL);
      if (pwrite(ofd, q, n * Q6_ROW, o_off + r0 * Q6_ROW) != (ssize_t)(n * Q6_ROW)) { perror("pwrite"); exit(1); }
      // the source pages are not needed again
      madvise((void *)((uintptr_t)(w + r0 * DIM) & ~(uintptr_t)4095), n * DIM, MADV_DONTNEED);
#pragma omp atomic
      done += n;
#pragma omp atomic
      bad += nan;
      if (c % 2048 == 0) fprintf(stderr, "  %.1f%%\n", 100.0 * done / rows);
    }
    free(f);
    free(q);
  }
  close(ofd);
  printf("%llu rows -> %s @ %llu (%lld NaN values zeroed)\n", (unsigned long long)rows, out, (unsigned long long)o_off, (long long)bad);
  return 0;
}
