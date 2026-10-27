// Native GGUF Q2_K / IQ3_XXS scalar access. Codes stay packed across PCIe.
#pragma once
#include "iq2grid.cuh"
__device__ __forceinline__ unsigned native_u32(const uint8_t *p) {
  return (unsigned)p[0] | ((unsigned)p[1] << 8) | ((unsigned)p[2] << 16) | ((unsigned)p[3] << 24);
}
__device__ __forceinline__ float q2k_value(const uint8_t *p, int e) {
  unsigned sc = p[e / 16];
  unsigned q = (p[16 + (e / 128) * 32 + e % 32] >> (2 * ((e / 32) % 4))) & 3;
  return (ld_h(p + 80) * (float)(sc & 15)) * (float)q - ld_h(p + 82) * (float)(sc >> 4);
}
__device__ __forceinline__ float iq3xxs_value(const uint8_t *p, int e) {
  int ib = e / 32, q = (e % 32) / 4;
  unsigned sg = native_u32(p + 66 + 4 * ib);
  unsigned grid = iq3xxs_grid[p[2 + 8 * ib + q]];
  unsigned signs = ksigns_iq2xs[(sg >> (7 * (q / 2))) & 127];
  float v = (ld_h(p) * (0.5f + (float)(sg >> 28)) * 0.5f) * (float)((grid >> (8 * (e % 4))) & 255);
  return (signs >> (e % 8)) & 1 ? -v : v;
}
