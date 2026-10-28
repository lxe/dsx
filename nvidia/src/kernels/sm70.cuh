// sm70.cuh — Volta (SM70) helpers for dsx kernels. Included first from dsx.cu;
// the whole body is __CUDA_ARCH__-guarded so SM80+ builds never instantiate it.
// NOTE: no callers yet — the current SM70 port routes decode to the in-tree
// gemv0 kernel and attention to the serial kernel, neither of which needs
// these. dsx_cp16 is kept for the planned wmma attention upgrade (T4-NOTES).
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
