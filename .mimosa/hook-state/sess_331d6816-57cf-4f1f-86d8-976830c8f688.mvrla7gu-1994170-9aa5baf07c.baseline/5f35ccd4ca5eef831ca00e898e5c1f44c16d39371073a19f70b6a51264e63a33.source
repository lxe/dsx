// cbq.c - quantization error of bitshift-trellis codebooks on Gaussian data (what weights look like after a
// Hadamard rotation). State = the last 16 bits of the stream; each weight shifts in K bits; the weight's value
// is f(state). Viterbi finds the best bit sequence for a 256-weight tile; error is MSE / variance at the best
// global scale. References at K = 2: rate-distortion bound 0.0625, 2-bit Lloyd-Max scalar 0.1175.
//
// build: gcc -O3 -march=native -fopenmp cbq.c -o cbq -lm      run: ./cbq [tiles] [K]
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define L 16
#define NS (1 << L)
#define T 256

static uint64_t rs = 88172645463325252ull;
static double urand(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return ((rs >> 11) + 0.5) / 9007199254740992.0; }
static double nrand(void) { return sqrt(-2 * log(urand())) * cos(6.283185307179586 * urand()); }

static int bytesum32(uint32_t x) { return (x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + (x >> 24); }

static void normalize(float *v) {
  double m = 0, q = 0;
  for (int s = 0; s < NS; s++) m += v[s];
  m /= NS;
  for (int s = 0; s < NS; s++) q += (v[s] - m) * (v[s] - m);
  q = sqrt(q / NS);
  for (int s = 0; s < NS; s++) v[s] = (float)((v[s] - m) / q);
}

// Viterbi for one tile at scale c: returns sum of squared error
static double tile_err(const float *v, const float *x, int K, float c, float *cost, float *nc, float *mp) {
  const int sh = L - K, nq = 1 << sh, mask = NS - 1;
  for (int s = 0; s < NS; s++) { float d = x[0] - c * v[s]; cost[s] = d * d; }
  for (int t = 1; t < T; t++) {
    for (int q = 0; q < nq; q++) {          // min over the K dropped high bits of the predecessor
      float m = cost[q];
      for (int h = 1; h < (1 << K); h++) { float z = cost[q | (h << sh)]; if (z < m) m = z; }
      mp[q] = m;
    }
    for (int s = 0; s < NS; s++) { float d = x[t] - c * v[s]; nc[s] = d * d + mp[(s >> K) & (nq - 1)]; }
    float *tmp = cost; cost = nc; nc = tmp;
  }
  float best = cost[0];
  for (int s = 1; s < NS; s++) if (cost[s] < best) best = cost[s];
  (void)mask;
  return best;
}

static double eval(const char *name, const float *v, const float *X, int tiles, int K) {
  double bestm = 1e9, bestc = 0;
  for (float c = 0.80f; c <= 1.301f; c += 0.05f) {
    double tot = 0;
    #pragma omp parallel reduction(+:tot)
    {
      float *a = malloc(NS * 4), *b = malloc(NS * 4), *mp = malloc(NS * 4);
      #pragma omp for schedule(dynamic)
      for (int i = 0; i < tiles; i++) tot += tile_err(v, X + (size_t)i * T, K, c, a, b, mp);
      free(a); free(b); free(mp);
    }
    double m = tot / ((double)tiles * T);
    if (m < bestm) { bestm = m; bestc = c; }
  }
  printf("%-34s K=%d  MSE %.4f  (scale %.2f)\n", name, K, bestm, bestc);
  fflush(stdout);
  return bestm;
}

int main(int argc, char **argv) {
  int tiles = argc > 1 ? atoi(argv[1]) : 200, K = argc > 2 ? atoi(argv[2]) : 2;
  float *X = malloc((size_t)tiles * T * 4);
  for (size_t i = 0; i < (size_t)tiles * T; i++) X[i] = (float)nrand();
  float *v = malloc(NS * 4);

  // EXL3 mul1 again as the yardstick
  for (int s = 0; s < NS; s++) v[s] = (float)(bytesum32((uint32_t)s * 0x83DCD12Du) - 510);
  normalize(v); eval("mul1 (EXL3)", v, X, tiles, K);
  // 16-bit multiplier, full 32-bit product, 4-byte sum (pmullw + pmulhuw per 16 lanes)
  uint32_t m16[] = {0xD12Du, 0x83DDu, 0xB5ADu, 0x9E37u};
  for (int i = 0; i < 4; i++) {
    char nm[64]; snprintf(nm, sizeof nm, "mul16 0x%04X, 4-byte sum", m16[i]);
    for (int s = 0; s < NS; s++) v[s] = (float)bytesum32((uint32_t)s * m16[i]);
    normalize(v); eval(nm, v, X, tiles, K);
  }
  // 16-bit multiplier, low 16 bits only, 2-byte sum, several constants
  for (int i = 0; i < 4; i++) {
    char nm[64]; snprintf(nm, sizeof nm, "mul16 0x%04X, 2-byte sum", m16[i]);
    for (int s = 0; s < NS; s++) { uint32_t q = ((uint32_t)s * m16[i]) & 0xFFFF; v[s] = (float)((q & 255) + (q >> 8)); }
    normalize(v); eval(nm, v, X, tiles, K);
  }
  // 16-bit multiply, 3 bytes: low 16 + the low byte of the high half
  for (int s = 0; s < NS; s++) { uint32_t q = (uint32_t)s * 0xD12Du; v[s] = (float)((q & 255) + ((q >> 8) & 255) + ((q >> 16) & 255)); }
  normalize(v); eval("mul16 0xD12D, 3-byte sum", v, X, tiles, K);
  return 0;
}
