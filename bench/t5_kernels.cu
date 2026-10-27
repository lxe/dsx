// T5 bench kernel TU (bench/t5.sh builds it with the same nvcc flags nvidia/build.rs uses, sm_70): the expert_fast
// path exactly as dsx.cu compiles it, plus two helpers for the host reference in t5_bench.cu (which loads the fatbin
// through the driver API, as gpu.rs does). Correctness/perf results: bench/T5-RESULTS.md.
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>

__device__ __forceinline__ float bf16r(float x) { return __bfloat162float(__float2bfloat16(x)); }
__device__ __forceinline__ float ld_h(const uint8_t *p) {
  return __half2float(__ushort_as_half((uint16_t)p[0] | ((uint16_t)p[1] << 8)));
}
// dsx.cu helpers that moe.cuh / expert_fast.cuh expect in scope
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
  return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
  return v;
}
__device__ __forceinline__ float trit(uint32_t q, uint32_t p3) { return (float)((((q * p3) & 0xFF) * 3) >> 8) - 1.f; }

#include "kernels/moe.cuh"
#include "kernels/expert_fast.cuh"






// dequantize a whole repacked matrix through rp_weight (the prefill path's decoder). The host reference decoders are
// checked against this first; the fast kernels are then checked against the host reference.
extern "C" __global__ void t5_rp_weight(int t, const uint8_t *w, int rows, int k, float *out) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < (size_t)rows * k; i += (size_t)gridDim.x * blockDim.x)
    out[i] = rp_weight(t, w, rows, k, i / k, i % k);
}

// the ggml codebook tables of iq2grid.cuh, copied out for the host decoders.
// g64: [0,256) iq2xxs_grid, [256,768) iq2xs_grid, [768,1792) iq2s_grid, [1792,3840) iq1s_grid
// g8: [0,128) ksigns_iq2xs
extern "C" __global__ void t5_tables(unsigned long long *g64, uint8_t *g8) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < 256; i += gridDim.x * blockDim.x) {
    g64[i] = iq2xxs_grid[i];
    g64[256 + i] = iq2xs_grid[i];
    g64[768 + i] = iq2s_grid[i];
    if (i < 128) g8[i] = ksigns_iq2xs[i];  // both the table and g8 are 128 entries: stay in bounds
  }
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < 512; i += gridDim.x * blockDim.x) g64[256 + i] = iq2xs_grid[i];
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < 1024; i += gridDim.x * blockDim.x) g64[768 + i] = iq2s_grid[i];
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < 2048; i += gridDim.x * blockDim.x) {
    g64[1792 + i] = iq1s_grid[i];
    if (i < 128) g8[i] = ksigns_iq2xs[i];
  }
}
