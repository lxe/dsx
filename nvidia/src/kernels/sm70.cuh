// sm70.cuh — Volta (SM70) compatibility for dsx kernels. Everything here is
// additive: included unconditionally, but every SM70-only definition is used
// from __CUDA_ARCH__-guarded call sites, so SM80+ builds never instantiate it.
#pragma once
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
#define DSX_SM70 1
// Staged 16-byte global->shared copy (cp.async replacement): same destination
// layout, synchronous. `dst` is a shared-memory address, `src` global.
__device__ __forceinline__ void dsx_cp16(void *dst, const void *src) {
  const uint4 v = *(const uint4 *)src;
  *(uint4 *)dst = v;
}
#endif
