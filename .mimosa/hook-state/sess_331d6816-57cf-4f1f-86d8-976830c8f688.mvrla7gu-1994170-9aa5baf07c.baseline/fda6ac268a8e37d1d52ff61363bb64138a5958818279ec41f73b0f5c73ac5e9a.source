// cbsearch.c - screen 16-bit odd multipliers for the "mul16, 2-byte sum" trellis codebook at K bits/weight
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define L 16
#define NS (1 << L)
#define T 256
static uint64_t rs = 88172645463325252ull;
static double urand(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return ((rs >> 11) + 0.5) / 9007199254740992.0; }
static double nrand(void) { return sqrt(-2 * log(urand())) * cos(6.283185307179586 * urand()); }
static void normalize(float *v) { double m = 0, q = 0; for (int s = 0; s < NS; s++) m += v[s]; m /= NS;
  for (int s = 0; s < NS; s++) q += (v[s] - m) * (v[s] - m); q = sqrt(q / NS); for (int s = 0; s < NS; s++) v[s] = (float)((v[s] - m) / q); }
static double tile_err(const float *v, const float *x, int K, float c, float *cost, float *nc, float *mp) {
  const int sh = L - K, nq = 1 << sh;
  for (int s = 0; s < NS; s++) { float d = x[0] - c * v[s]; cost[s] = d * d; }
  for (int t = 1; t < T; t++) {
    for (int q = 0; q < nq; q++) { float m = cost[q]; for (int h = 1; h < (1 << K); h++) { float z = cost[q | (h << sh)]; if (z < m) m = z; } mp[q] = m; }
    for (int s = 0; s < NS; s++) { float d = x[t] - c * v[s]; nc[s] = d * d + mp[(s >> K) & (nq - 1)]; }
    float *tmp = cost; cost = nc; nc = tmp;
  }
  float b = cost[0]; for (int s = 1; s < NS; s++) if (cost[s] < b) b = cost[s]; return b;
}
int main(int argc, char **argv) {
  int tiles = atoi(argv[1]), K = atoi(argv[2]), nconst = atoi(argv[3]), mode = argc > 4 ? atoi(argv[4]) : 2;
  float *X = malloc((size_t)tiles * T * 4); for (size_t i = 0; i < (size_t)tiles * T; i++) X[i] = (float)nrand();
  uint64_t cs = 1234567;
  for (int ci = 0; ci < nconst; ci++) {
    uint32_t M;
    if (argc > 5 + ci && ci < argc - 5) M = (uint32_t)strtoul(argv[5 + ci], 0, 16);
    else { cs = cs * 6364136223846793005ull + 1442695040888963407ull; M = (uint32_t)((cs >> 40) & 0xFFFF) | 1; }
    float *v = malloc(NS * 4);
    for (int s = 0; s < NS; s++) { uint32_t q = (uint32_t)s * M; if (mode == 2) q &= 0xFFFF;
      v[s] = (float)((q & 255) + ((q >> 8) & 255) + (mode == 4 ? ((q >> 16) & 255) + (q >> 24) : 0)); }
    normalize(v);
    double best = 1e9;
    for (float c = 0.95f; c <= 1.101f; c += 0.05f) {
      double tot = 0;
      #pragma omp parallel reduction(+:tot)
      { float *a = malloc(NS * 4), *b = malloc(NS * 4), *mp = malloc(NS * 4);
        #pragma omp for schedule(dynamic)
        for (int i = 0; i < tiles; i++) tot += tile_err(v, X + (size_t)i * T, K, c, a, b, mp);
        free(a); free(b); free(mp); }
      double m = tot / ((double)tiles * T); if (m < best) best = m;
    }
    printf("0x%04X %.4f\n", M, best); fflush(stdout); free(v);
  }
}
