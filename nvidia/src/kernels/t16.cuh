// T16 routed experts (core/src/t16.rs, docs/t16.md): exllamav3's bitshift trellis with the T16 codebook.
// A matrix (k inputs, n outputs) is trellis | suh | svh: a circular stream of 128 k2 bits per 16 x 16 tile (k2 = 2K
// = 3, 4, 5); position t holds tile row (input) t % 16 and tile column (output) t / 16, and its state is the 16-bit
// window ending after its own bits; suh f16 [k], svh f16 [n]. Two trellis layouts:
//   stored (GGUF, RAM, CPU kernels; type ids 1010-1012): tile (kb, nb) at nb * k/16 + kb, bytes in stream order;
//   repacked (GPU store; 2000 + id = 3010-3012): exllamav3's own, tile (kb, nb) at kb * n/16 + nb, little-endian u32 words read
//   MSB first (repack_t16).
// y = had128(had128(x * suh) @ W_inner) * svh. Prefill computes it on the activations: t16_pre (x * suh, Hadamard),
// a GEMM with W_inner^T from dequant_t16_f16, then t16_post (Hadamard, * svh).

__device__ __forceinline__ int t16_bit_end(int k2, int t) {
  return (k2 & 1) ? (t + 1) * (k2 >> 1) + ((t + 1) >> 1) : (t + 1) * (k2 >> 1);
}

// the 16-bit state of position t of a tile (w: its u32 words, 4 k2 of them)
__device__ __forceinline__ uint32_t t16_state(const uint32_t *w, int k2, int t) {
  const int nb = 128 * k2, nw = 4 * k2;
  const int b0 = (t16_bit_end(k2, t) - 16 + nb) % nb;
  const int i0 = b0 >> 5, i1 = (i0 + 1) % nw, s = b0 & 31;
  const uint64_t v = ((uint64_t)w[i0] << 32) | w[i1];
  return (uint32_t)(v >> (48 - s)) & 0xFFFFu;
}

// exllamav3's T16 decode: fp16(fma(fp16(1024 + b), 0x20e6, 0xca1e)), b = byte sum of low16(state * 0x3A2F)
__device__ __forceinline__ __half t16_value(uint32_t st) {
  const uint32_t p = (st * 0x3A2Fu) & 0xFFFFu;
  const int b = (int)(p & 0xFF) + (int)(p >> 8);
  return __hfma(__int2half_rn(1024 + b), __ushort_as_half((unsigned short)0x20e6), __ushort_as_half((unsigned short)0xca1e));
}

__device__ __forceinline__ size_t t16_trellis_bytes(int k2, int k, int n) { return (size_t)(k / 16) * (n / 16) * 16 * k2; }

// state of position t from a stored-layout tile (bytes in stream order)
__device__ __forceinline__ uint32_t t16_state_stored(const unsigned char *tb, int k2, int t) {
  const int nb = 128 * k2, nby = 16 * k2;
  const int b0 = (t16_bit_end(k2, t) - 16 + nb) % nb, y = b0 >> 3;
  const uint32_t v = (uint32_t)tb[y] << 16 | (uint32_t)tb[(y + 1) % nby] << 8 | tb[(y + 2) % nby];
  return (v >> (8 - (b0 & 7))) & 0xFFFFu;
}

// W_inner^T as f16 [n][k] (row = output) for cuBLAS; repacked = 0 stored layout, 1 GPU layout. One thread per tile
// row: 16 consecutive stream positions t = c * 16 + r (output c of the tile, inputs r = 0..15), whose 16-bit windows
// all lie in 64 bits of the stream (16 + 16 * 2.5 bits at most), so three u32 loads give all 16 states and the thread
// writes its 16 halves (32 bytes, contiguous in the output row) as two 16-byte stores; a warp covers 32 consecutive
// tiles of one output row. grid (ceil(k / 512), n / 16), block 512 (tile ti = lane, output c = warp). (One thread per weight with a runtime-modulo
// window each wrote ~130 GB/s on an RTX 3090.)
extern "C" __global__ void dequant_t16_f16(int k2, const unsigned char *w, __half *out, int n, int k, int repacked) {
  const int kt = k >> 4, nt = n >> 4;
  const int ti = blockIdx.x * 32 + (threadIdx.x & 31), c = threadIdx.x >> 5, tj = blockIdx.y;
  if (ti >= kt) return;
  const int nw = 4 * k2, nb = 32 * nw;
  // the tile as u32 words (every layout's tiles are 16 * k2 bytes, and matrices start 4-byte aligned)
  const uint32_t *tw = (const uint32_t *)(w + (repacked ? (size_t)ti * nt + tj : (size_t)tj * kt + ti) * 4 * nw);
  const int t0 = c * 16;
  const int s0 = t16_bit_end(k2, t0) - 16 + nb;  // unwrapped start of position t0's window (t0 = 0 wraps)
  int i0 = (s0 >> 5) % nw;
  const int i1 = i0 + 1 == nw ? 0 : i0 + 1, i2 = i1 + 1 == nw ? 0 : i1 + 1;
  uint32_t w0 = tw[i0], w1 = tw[i1], w2 = tw[i2];
  if (!repacked) {  // stored: stream-order bytes; GPU layout: little-endian words already MSB-first
    w0 = __byte_perm(w0, 0, 0x0123), w1 = __byte_perm(w1, 0, 0x0123), w2 = __byte_perm(w2, 0, 0x0123);
  }
  const int sh = s0 & 31;
  const uint64_t hi = (uint64_t)w0 << 32 | w1;
  const uint64_t win = sh ? (hi << sh) | (w2 >> (32 - sh)) : hi;  // stream bits s0 .. s0 + 63, MSB first
  const int base = nb - s0;
  uint32_t h[8];
#pragma unroll
  for (int r = 0; r < 16; r += 2) {
    const int e0 = t16_bit_end(k2, t0 + r) + base, e1 = t16_bit_end(k2, t0 + r + 1) + base;
    const __half v0 = t16_value((uint32_t)(win >> (64 - e0)) & 0xFFFFu);
    const __half v1 = t16_value((uint32_t)(win >> (64 - e1)) & 0xFFFFu);
    h[r >> 1] = (uint32_t)__half_as_ushort(v0) | (uint32_t)__half_as_ushort(v1) << 16;
  }
  uint4 *o = (uint4 *)(out + (size_t)(tj * 16 + c) * k + ti * 16);
  o[0] = make_uint4(h[0], h[1], h[2], h[3]);
  o[1] = make_uint4(h[4], h[5], h[6], h[7]);
}

// Fixed-bit specialization preserves decoded fp16 weights while removing runtime divisions.
template<int K2, bool REPACKED>
__device__ __forceinline__ void t16_dequant_specialized(const unsigned char *w, __half *out, int n, int k) {
  constexpr int k2 = K2;
  constexpr bool repacked = REPACKED;

  const int kt = k >> 4, nt = n >> 4;
  const int ti = blockIdx.x * 32 + (threadIdx.x & 31), c = threadIdx.x >> 5, tj = blockIdx.y;
  if (ti >= kt) return;
  const int nw = 4 * k2, nb = 32 * nw;
  // the tile as u32 words (every layout's tiles are 16 * k2 bytes, and matrices start 4-byte aligned)
  const uint32_t *tw = (const uint32_t *)(w + (repacked ? (size_t)ti * nt + tj : (size_t)tj * kt + ti) * 4 * nw);
  const int t0 = c * 16;
  const int s0 = t16_bit_end(k2, t0) - 16 + nb;  // unwrapped start of position t0's window (t0 = 0 wraps)
  int i0 = (s0 >> 5) % nw;
  const int i1 = i0 + 1 == nw ? 0 : i0 + 1, i2 = i1 + 1 == nw ? 0 : i1 + 1;
  uint32_t w0 = tw[i0], w1 = tw[i1], w2 = tw[i2];
  if (!repacked) {  // stored: stream-order bytes; GPU layout: little-endian words already MSB-first
    w0 = __byte_perm(w0, 0, 0x0123), w1 = __byte_perm(w1, 0, 0x0123), w2 = __byte_perm(w2, 0, 0x0123);
  }
  const int sh = s0 & 31;
  const uint64_t hi = (uint64_t)w0 << 32 | w1;
  const uint64_t win = sh ? (hi << sh) | (w2 >> (32 - sh)) : hi;  // stream bits s0 .. s0 + 63, MSB first
  const int base = nb - s0;
  uint32_t h[8];
#pragma unroll
  for (int r = 0; r < 16; r += 2) {
    const int e0 = t16_bit_end(k2, t0 + r) + base, e1 = t16_bit_end(k2, t0 + r + 1) + base;
    const __half v0 = t16_value((uint32_t)(win >> (64 - e0)) & 0xFFFFu);
    const __half v1 = t16_value((uint32_t)(win >> (64 - e1)) & 0xFFFFu);
    h[r >> 1] = (uint32_t)__half_as_ushort(v0) | (uint32_t)__half_as_ushort(v1) << 16;
  }
  uint4 *o = (uint4 *)(out + (size_t)(tj * 16 + c) * k + ti * 16);
  o[0] = make_uint4(h[0], h[1], h[2], h[3]);
  o[1] = make_uint4(h[4], h[5], h[6], h[7]);
}
extern "C" __global__ void dequant_t16_f16_specialized(int k2, const unsigned char *w, __half *out, int n, int k, int repacked) {
  if (repacked) {
    if (k2 == 3) t16_dequant_specialized<3, true>(w, out, n, k);
    else if (k2 == 4) t16_dequant_specialized<4, true>(w, out, n, k);
    else t16_dequant_specialized<5, true>(w, out, n, k);
  } else {
    if (k2 == 3) t16_dequant_specialized<3, false>(w, out, n, k);
    else if (k2 == 4) t16_dequant_specialized<4, false>(w, out, n, k);
    else t16_dequant_specialized<5, false>(w, out, n, k);
  }
}

// One warp owns 128 values. Butterfly order matches the block-wide reference exactly.
__device__ __forceinline__ void t16_had128_strided(float (&v)[4], int lane) {
#pragma unroll
  for (int h = 1; h < 32; h <<= 1) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float other = __shfl_xor_sync(0xffffffffu, v[j], h);
      v[j] = (lane & h) ? other - v[j] : v[j] + other;
    }
  }
  const float a0 = v[0] + v[1], a1 = v[0] - v[1];
  const float a2 = v[2] + v[3], a3 = v[2] - v[3];
  v[0] = a0 + a2; v[1] = a1 + a3;
  v[2] = a0 - a2; v[3] = a1 - a3;
}
extern "C" __global__ void t16_pre_warp(const __half *src, __half *dst, const __half *suh, int cols, int rows) {
  const int group = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int groups = cols / 128, lane = threadIdx.x & 31;
  if (group >= rows * groups) return;
  const int col = (group % groups) * 128 + lane;
  const size_t base = (size_t)(group / groups) * cols + col;
  float v[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) v[j] = __half2float(src[base + j * 32]) * __half2float(suh[col + j * 32]);
  t16_had128_strided(v, lane);
#pragma unroll
  for (int j = 0; j < 4; ++j) dst[base + j * 32] = __float2half(v[j] * 0.08838834764831845f);
}
extern "C" __global__ void t16_post_warp(float *y, const __half *svh, int cols, float gain, int rows) {
  const int group = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int groups = cols / 128, lane = threadIdx.x & 31;
  if (group >= rows * groups) return;
  const int col = (group % groups) * 128 + lane;
  const size_t base = (size_t)(group / groups) * cols + col;
  float v[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) v[j] = y[base + j * 32];
  t16_had128_strided(v, lane);
#pragma unroll
  for (int j = 0; j < 4; ++j) y[base + j * 32] = v[j] * 0.08838834764831845f * __half2float(svh[col + j * 32]) * gain;
}

// Gate/up Hadamard post transforms followed by the unchanged bf16-rounded SwiGLU.
// The intermediate post-transformed gate/up arrays are dead after SwiGLU, so keep them in registers.
extern "C" __global__ void t16_post_gu_warp(const float *gate, const float *up,
    const __half *sg, const __half *su, const float *weights, __half *out,
    int cols, int rows, float gain, float lim) {
  const int group = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int groups = cols / 128, lane = threadIdx.x & 31;
  if (group >= rows * groups) return;
  const int row = group / groups, col = (group % groups) * 128 + lane;
  const size_t base = (size_t)row * cols + col;
  float g[4], u[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) { g[j] = gate[base + j * 32]; u[j] = up[base + j * 32]; }
  t16_had128_strided(g, lane); t16_had128_strided(u, lane);
  const float wt = weights ? weights[row] : 1.f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float gv = fminf(g[j] * 0.08838834764831845f * __half2float(sg[col + j * 32]) * gain, lim);
    const float uv = fminf(fmaxf(u[j] * 0.08838834764831845f * __half2float(su[col + j * 32]) * gain, -lim), lim);
    out[base + j * 32] = __float2half(bf16r(gv / (1.f + expf(-gv)) * uv * wt));
  }
}

// stored layout -> GPU layout (exllamav3's tile order, little-endian u32 words); suh / svh copied. One thread per
// 4 bytes of the matrix (trellis | suh | svh, the same size).
extern "C" __global__ void repack_t16(int k2, const unsigned char *src, unsigned char *dst, int k, int n) {
  const size_t tbytes = t16_trellis_bytes(k2, k, n), total = tbytes + 2 * (size_t)k + 2 * (size_t)n;
  const size_t o = 4 * ((size_t)blockIdx.x * blockDim.x + threadIdx.x);
  if (o >= total) return;
  if (o >= tbytes) {
    *(uint32_t *)(dst + o) = *(const uint32_t *)(src + o);
    return;
  }
  const int tb = 16 * k2, kt = k / 16, nt = n / 16;
  const size_t tile = o / tb, in = o % tb;  // destination tile index (kb * nt + nb), byte within it
  const int kb = (int)(tile / nt), nb = (int)(tile % nt);
  const unsigned char *s = src + ((size_t)nb * kt + kb) * tb + in;
  *(uint32_t *)(dst + o) = (uint32_t)s[0] << 24 | (uint32_t)s[1] << 16 | (uint32_t)s[2] << 8 | s[3];
}

// dst = had128(src * suh) per row, f16; grid (cols / 128, rows), block 128 (dst may be src)
extern "C" __global__ void t16_pre(const __half *src, __half *dst, const __half *suh, int cols) {
  __shared__ float fb[128];
  const int t = threadIdx.x, c = blockIdx.x * 128 + t;
  const size_t i = (size_t)blockIdx.y * cols + c;
  fb[t] = __half2float(src[i]) * __half2float(suh[c]);
  __syncthreads();
#pragma unroll
  for (int h = 1; h < 128; h <<= 1) {
    float own = fb[t], other = fb[t ^ h];
    __syncthreads();
    fb[t] = (t & h) ? other - own : own + other;
    __syncthreads();
  }
  dst[i] = __float2half(fb[t] * 0.08838834764831845f);
}

// y = had128(y) * svh * gain per row, f32; grid (cols / 128, rows), block 128
extern "C" __global__ void t16_post(float *y, const __half *svh, int cols, float gain) {
  __shared__ float fb[128];
  const int t = threadIdx.x, c = blockIdx.x * 128 + t;
  const size_t i = (size_t)blockIdx.y * cols + c;
  fb[t] = y[i];
  __syncthreads();
#pragma unroll
  for (int h = 1; h < 128; h <<= 1) {
    float own = fb[t], other = fb[t ^ h];
    __syncthreads();
    fb[t] = (t & h) ? other - own : own + other;
    __syncthreads();
  }
  y[i] = fb[t] * 0.08838834764831845f * __half2float(svh[c]) * gain;
}

// ---------------------------------------------------------------- decode (docs/t16.md "v4")
// Device-driven like moe_*_fast: pick j of T x K (token j / K) runs if its expert is resident here (tab entry). Per
// pick and matrix: t16d_prep (x * suh, had128, int8 per 256 written as pair words [x_k x_k x_k+1 x_k+1], group scale
// and sum), t16d_gemv (column-major tiles, one window + one IMAD per weight, two rows' products packed and dp4a'd
// with a pair word, exact affine value 0.009567260742 b - 2.4375), t16d_post_gu (had128, svh, swiglu, pick weight),
// t16d_post_down (had128, svh, sum of the token's picks in pick order).

__device__ __forceinline__ void t16_had128_warp(float (&v)[4], int lane) {
  float a0 = v[0] + v[1], a1 = v[0] - v[1], a2 = v[2] + v[3], a3 = v[2] - v[3];
  v[0] = a0 + a2, v[2] = a0 - a2, v[1] = a1 + a3, v[3] = a1 - a3;
#pragma unroll
  for (int m = 1; m < 32; m <<= 1) {
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const float o = __shfl_xor_sync(0xffffffffu, v[j], m);
      v[j] = (lane & m) ? o - v[j] : v[j] + o;
    }
  }
#pragma unroll
  for (int j = 0; j < 4; j++) v[j] *= 0.08838834764831845f;
}

__device__ __forceinline__ const unsigned char *t16_mat(const ExpSlot &s, const unsigned char *base, int role) {
  return base + (role == 0 ? s.gate : role == 1 ? s.up : s.down);
}

// x: [T][k] (per token, xpp = 0) or [T K][k] (per pick, xpp = 1). Out: xp [T K][k / 2] u32, xd / xs [T K][k / 256].
// One warp per (pick, 256-group); block 128.
extern "C" __global__ void t16d_prep(const float *__restrict__ x, int xpp, const int *__restrict__ sel, const ExpSlot *__restrict__ tab,
                                     const unsigned char *__restrict__ base, int layer, int E, int K, int T, int role, int k, int n, int k2,
                                     uint32_t *__restrict__ xp, float *__restrict__ xd, int *__restrict__ xs) {
  const int lane = threadIdx.x & 31, wg = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, groups = k / 256;
  if (wg >= T * K * groups) return;
  const int j = wg / groups, gr = wg % groups;
  const ExpSlot s = tab[layer * E + sel[j]];
  if (s.gate == ~0ull) return;
  const __half *suh = (const __half *)(t16_mat(s, base, role) + t16_trellis_bytes(k2, k, n));
  const float *xr = x + (size_t)(xpp ? j : j / K) * k;
  float v[2][4];
#pragma unroll
  for (int h = 0; h < 2; h++) {
    const int i = 256 * gr + 128 * h + 4 * lane;
#pragma unroll
    for (int q = 0; q < 4; q++) v[h][q] = xr[i + q] * __half2float(suh[i + q]);
    t16_had128_warp(v[h], lane);
  }
  float m = 0.f;
#pragma unroll
  for (int h = 0; h < 2; h++)
#pragma unroll
    for (int q = 0; q < 4; q++) m = fmaxf(m, fabsf(v[h][q]));
  for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
  const float id = m > 0.f ? 127.f / m : 0.f;
  int sum = 0;
#pragma unroll
  for (int h = 0; h < 2; h++) {
    int q[4];
#pragma unroll
    for (int qq = 0; qq < 4; qq++) q[qq] = __float2int_rn(v[h][qq] * id), sum += q[qq];
    const size_t r = (size_t)j * k + 256 * gr + 128 * h + 4 * lane;
    const uint32_t b0 = q[0] & 255, b1 = q[1] & 255, b2 = q[2] & 255, b3 = q[3] & 255;
    xp[r / 2] = b0 | b0 << 8 | b1 << 16 | b1 << 24;
    xp[r / 2 + 1] = b2 | b2 << 8 | b3 << 16 | b3 << 24;
  }
  for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
  if (lane == 0) xd[(size_t)j * groups + gr] = m / 127.f, xs[(size_t)j * groups + gr] = sum;
}

__device__ __forceinline__ uint32_t t16_fsr(uint32_t lo, uint32_t hi, int s) { return (uint32_t)((((uint64_t)hi << 32) | lo) >> s); }
__host__ __device__ constexpr int t16_hend(int K2, int t) { return K2 % 2 ? (t + 1) * (K2 / 2) + (t + 1) / 2 : (t + 1) * (K2 / 2); }

// the 16 windows of tile column c (positions 16c .. 16c + 15) from three consecutive words; products * 0x3A2F
template <int K2>
__device__ __forceinline__ void t16_prod16(const uint32_t *ptr, int i0, int i1, int i2, int sh, uint32_t (&p)[16]) {
  const uint32_t a = ptr[i0], b = ptr[i1], c = ptr[i2];
  const uint32_t lo = t16_fsr(c, b, sh), hi = t16_fsr(b, a, sh);
#pragma unroll
  for (int j = 0; j < 16; j++) {
    const int o = t16_hend(K2, 15) - t16_hend(K2, j);
    const uint32_t wj = o < 32 ? __funnelshift_r(lo, hi, o) : hi >> (o - 32);
    p[j] = wj * 0x3A2Fu;
  }
}

// t16_prod16 from the three words already loaded (a = ptr[i0], b = ptr[i1], c = ptr[i2])
template <int K2>
__device__ __forceinline__ void t16_prod16w(uint32_t a, uint32_t b, uint32_t c, int sh, uint32_t (&p)[16]) {
  const uint32_t lo = t16_fsr(c, b, sh), hi = t16_fsr(b, a, sh);
#pragma unroll
  for (int j = 0; j < 16; j++) {
    const int o = t16_hend(K2, 15) - t16_hend(K2, j);
    const uint32_t wj = o < 32 ? __funnelshift_r(lo, hi, o) : hi >> (o - 32);
    p[j] = wj * 0x3A2Fu;
  }
}

__device__ __forceinline__ int t16_dp4a(uint32_t a, int b, int c) {
  int d;
  asm("dp4a.u32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}

// part [S][T K][n]: split s of k's 256-groups. One warp per (pick, pair of tile columns, split); block 128.
// SM80 only (cp.async staging); on SM70 the t16d_gemv_##K2 kernels below run the plain t16d_gemv0_k body instead
// (bit-identical output; the staged kernel's SM70 port lands in a later task).
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
template <int K2>
__device__ __forceinline__ void t16d_gemv_k(const uint4 *__restrict__ xp, const float *__restrict__ xd, const int *__restrict__ xs,
                                            const int *__restrict__ sel, const ExpSlot *__restrict__ tab, const unsigned char *__restrict__ base,
                                            int layer, int E, int K, int T, int role, int k, int n, int S, float *__restrict__ part) {
  const int lane = threadIdx.x & 31, warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int nbs = n / 16, groups = k / 256, pairs = nbs / 2, picks = T * K;
  if (warp >= picks * pairs * S) return;
  const int s = warp / (picks * pairs), r = warp % (picks * pairs), j = r / pairs, nb = 2 * (r % pairs) + lane / 16, c = lane % 16;
  const ExpSlot sl = tab[layer * E + sel[j]];
  if (sl.gate == ~0ull) return;
  const int g0 = groups * s / S, g1 = groups * (s + 1) / S;
  constexpr int WT = 4 * K2, NB = 32 * WT;
  const int e15 = t16_hend(K2, 16 * c + 15) + NB, hw = (e15 - 1) / 32;
  const int i2 = hw % WT, i1 = (hw - 1) % WT, i0 = (hw - 2) % WT, sh = (hw + 1) * 32 - e15;
  const uint32_t *tp = (const uint32_t *)t16_mat(sl, base, role) + ((size_t)16 * g0 * nbs + nb) * WT;
  const uint32_t step = (uint32_t)nbs * WT;
  const uint4 *xe = xp + (size_t)j * (k / 8) + 2 * 16 * g0;
  // the warp's two tile columns (nb0, nb0 + 1: TW words a tile row) for a whole 256-group go to shared memory with
  // cp.async, double-buffered: the next group's 16 rows stream in while this group is computed. Bytes in flight per
  // warp, not latency, bound this kernel; the arithmetic and its order are unchanged (bit-identical).
  // The group's 16 x rows (32 B each, the dp4a operand) come along: ncu's stall sampling put 47% of warp stalls on dp4a
  // waiting for its per-iteration x load from global.
  constexpr int TW = 2 * WT, CH = TW / 4;
  __shared__ __align__(16) uint32_t sm[4][2][16 * TW];
  __shared__ __align__(16) uint4 smx[4][2][32];
  uint32_t(*const buf)[16 * TW] = sm[(threadIdx.x >> 5) & 3];
  uint4(*const xbuf)[32] = smx[(threadIdx.x >> 5) & 3];
  const uint32_t *tw = (const uint32_t *)t16_mat(sl, base, role) + ((size_t)16 * g0 * nbs + (nb & ~1)) * WT;
  // each lane's NQ 16-byte chunks of a group: offsets fixed for the whole kernel, computed once
  constexpr int NQ = 16 * CH / 32;
  static_assert(16 * CH % 32 == 0, "chunks per group must split evenly over the warp");
  uint32_t goff[NQ], soff[NQ];
#pragma unroll
  for (int q = 0; q < NQ; q++) {
    const int c16 = lane + 32 * q, row = c16 / CH, ch = c16 % CH;
    goff[q] = row * step + 4 * ch, soff[q] = 4 * (row * TW + 4 * ch);
  }
  const uint32_t sb0 = (uint32_t)__cvta_generic_to_shared(&buf[0][0]), sbs = 4 * 16 * TW;
  auto issue = [&](int gr, int bi) {
    const uint32_t *src = tw + (size_t)(gr - g0) * 16 * step;
    const uint32_t db = sb0 + bi * sbs;
#pragma unroll
    for (int q = 0; q < NQ; q++) asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(db + soff[q]), "l"(src + goff[q]));
    const uint32_t dx = (uint32_t)__cvta_generic_to_shared(&xbuf[bi][lane]);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" ::"r"(dx), "l"(xe + 2 * 16 * (gr - g0) + lane));
    asm volatile("cp.async.commit_group;");
  };
  issue(g0, 0);
  const int lo = (lane / 16) * WT;
  float f = 0.f;
  for (int gr = g0, bi = 0; gr < g1; gr++, bi ^= 1) {
    if (gr + 1 < g1) issue(gr + 1, bi ^ 1);
    else asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 1;" ::: "memory");
    __syncwarp();
    // four dp4a chains (an exact integer sum in any order), so one warp's dependent dp4a latency is not the limit
    int ac[4] = {0, 0, 0, 0};
#pragma unroll 4
    for (int i = 0; i < 16; i++) {
      uint32_t p[16];
      const uint32_t *w = &buf[bi][i * TW + lo];
      t16_prod16w<K2>(w[i0], w[i1], w[i2], sh, p);
      const uint4 xa = xbuf[bi][2 * i], xb = xbuf[bi][2 * i + 1];
      const uint32_t xw[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
#pragma unroll
      for (int q = 0; q < 8; q++) ac[q & 3] = t16_dp4a(__byte_perm(p[2 * q], p[2 * q + 1], 0x5410), (int)xw[q], ac[q & 3]);
    }
    const int acc = (ac[0] + ac[1]) + (ac[2] + ac[3]);
    f += __ldg(xd + (size_t)j * groups + gr) * (0.009567260742f * (float)acc - 2.4375f * (float)__ldg(xs + (size_t)j * groups + gr));
    __syncwarp();
  }
  part[((size_t)s * picks + j) * n + 16 * nb + c] = f;
}
#endif  // __CUDA_ARCH__ >= 800
// DSX_T16_GEMV=0: the kernel before the cp.async staging (for A/Bs; same launch, same output bits).
// part [S][T K][n]: split s of k's 256-groups. One warp per (pick, pair of tile columns, split); block 128.
template <int K2>
__device__ __forceinline__ void t16d_gemv0_k(const uint4 *__restrict__ xp, const float *__restrict__ xd, const int *__restrict__ xs,
                                            const int *__restrict__ sel, const ExpSlot *__restrict__ tab, const unsigned char *__restrict__ base,
                                            int layer, int E, int K, int T, int role, int k, int n, int S, float *__restrict__ part) {
  const int lane = threadIdx.x & 31, warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int nbs = n / 16, groups = k / 256, pairs = nbs / 2, picks = T * K;
  if (warp >= picks * pairs * S) return;
  const int s = warp / (picks * pairs), r = warp % (picks * pairs), j = r / pairs, nb = 2 * (r % pairs) + lane / 16, c = lane % 16;
  const ExpSlot sl = tab[layer * E + sel[j]];
  if (sl.gate == ~0ull) return;
  const int g0 = groups * s / S, g1 = groups * (s + 1) / S;
  constexpr int WT = 4 * K2, NB = 32 * WT;
  const int e15 = t16_hend(K2, 16 * c + 15) + NB, hw = (e15 - 1) / 32;
  const int i2 = hw % WT, i1 = (hw - 1) % WT, i0 = (hw - 2) % WT, sh = (hw + 1) * 32 - e15;
  const uint32_t *tp = (const uint32_t *)t16_mat(sl, base, role) + ((size_t)16 * g0 * nbs + nb) * WT;
  const uint32_t step = (uint32_t)nbs * WT;
  const uint4 *xe = xp + (size_t)j * (k / 8) + 2 * 16 * g0;
  float f = 0.f;
  for (int gr = g0; gr < g1; gr++) {
    int acc = 0;
#pragma unroll 4
    for (int i = 0; i < 16; i++, tp += step, xe += 2) {
      uint32_t p[16];
      t16_prod16<K2>(tp, i0, i1, i2, sh, p);
      const uint4 xa = __ldg(xe), xb = __ldg(xe + 1);
      const uint32_t xw[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
#pragma unroll
      for (int q = 0; q < 8; q++) acc = t16_dp4a(__byte_perm(p[2 * q], p[2 * q + 1], 0x5410), (int)xw[q], acc);
    }
    f += __ldg(xd + (size_t)j * groups + gr) * (0.009567260742f * (float)acc - 2.4375f * (float)__ldg(xs + (size_t)j * groups + gr));
  }
  part[((size_t)s * picks + j) * n + 16 * nb + c] = f;
}
#define T16D_GEMV0(K2)                                                                                                                  \
  extern "C" __global__ void __launch_bounds__(128) t16d_gemv0_##K2(const uint4 *xp, const float *xd, const int *xs, const int *sel,  \
                                                                    const ExpSlot *tab, const unsigned char *base, int layer, int E,    \
                                                                    int K, int T, int role, int k, int n, int S, float *part) {         \
    t16d_gemv0_k<K2>(xp, xd, xs, sel, tab, base, layer, E, K, T, role, k, n, S, part);                                                 \
  }
T16D_GEMV0(3)
T16D_GEMV0(4)
T16D_GEMV0(5)

// SM80+: the cp.async-staged kernel. SM70: the same exported kernel name over the plain, bit-identical t16d_gemv0_k
// body (a preprocessor directive cannot go inside a #define body, so the macro is duplicated per arch).
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
#define T16D_GEMV(K2)                                                                                                                   \
  extern "C" __global__ void __launch_bounds__(128) t16d_gemv_##K2(const uint4 *xp, const float *xd, const int *xs, const int *sel,   \
                                                                   const ExpSlot *tab, const unsigned char *base, int layer, int E,     \
                                                                   int K, int T, int role, int k, int n, int S, float *part) {          \
    t16d_gemv_k<K2>(xp, xd, xs, sel, tab, base, layer, E, K, T, role, k, n, S, part);                                                  \
  }
#else
#define T16D_GEMV(K2)                                                                                                                   \
  extern "C" __global__ void __launch_bounds__(128) t16d_gemv_##K2(const uint4 *xp, const float *xd, const int *xs, const int *sel,   \
                                                                   const ExpSlot *tab, const unsigned char *base, int layer, int E,     \
                                                                   int K, int T, int role, int k, int n, int S, float *part) {          \
    t16d_gemv0_k<K2>(xp, xd, xs, sel, tab, base, layer, E, K, T, role, k, n, S, part);                                                 \
  }
#endif
T16D_GEMV(3)
T16D_GEMV(4)
T16D_GEMV(5)

// h [T K][n] = bf16(silu(min(g, lim)) * clamp(u, +-lim) * w_j), g / u = had128(sum_s part) * svh. One warp per
// (pick, 128-block); block 128.
extern "C" __global__ void t16d_post_gu(const float *__restrict__ pg, const float *__restrict__ pu, int S, const int *__restrict__ sel,
                                        const float *__restrict__ wts, const ExpSlot *__restrict__ tab, const unsigned char *__restrict__ base,
                                        int layer, int E, int K, int T, int k, int n, int k2, float lim, float *__restrict__ h) {
  const int lane = threadIdx.x & 31, wb = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, nblk = n / 128, picks = T * K;
  if (wb >= picks * nblk) return;
  const int j = wb / nblk, b = wb % nblk, i = 128 * b + 4 * lane;
  const ExpSlot s = tab[layer * E + sel[j]];
  if (s.gate == ~0ull) return;
  const size_t svo = t16_trellis_bytes(k2, k, n) + 2 * (size_t)k;
  const __half *svg = (const __half *)(base + s.gate + svo), *svu = (const __half *)(base + s.up + svo);
  float g[4] = {0, 0, 0, 0}, u[4] = {0, 0, 0, 0};
  for (int q = 0; q < S; q++)
#pragma unroll
    for (int t = 0; t < 4; t++) g[t] += pg[((size_t)q * picks + j) * n + i + t], u[t] += pu[((size_t)q * picks + j) * n + i + t];
  t16_had128_warp(g, lane);
  t16_had128_warp(u, lane);
  const float w = wts[j];
#pragma unroll
  for (int t = 0; t < 4; t++) {
    const float gv = fminf(g[t] * __half2float(svg[i + t]), lim), uv = fminf(fmaxf(u[t] * __half2float(svu[i + t]), -lim), lim);
    h[(size_t)j * n + i + t] = bf16r(gv / (1.f + expf(-gv)) * uv * w);
  }
}

// out [T][n] = sum over the token's resident picks (pick order) of had128(sum_s part) * svh; 0 if none. One warp per
// (token, 128-block); block 128.
extern "C" __global__ void t16d_post_down(const float *__restrict__ pd, int S, const int *__restrict__ sel, const ExpSlot *__restrict__ tab,
                                          const unsigned char *__restrict__ base, int layer, int E, int K, int T, int k, int n, int k2,
                                          float *__restrict__ out) {
  const int lane = threadIdx.x & 31, wb = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, nblk = n / 128, picks = T * K;
  if (wb >= T * nblk) return;
  const int t = wb / nblk, b = wb % nblk, i = 128 * b + 4 * lane;
  const size_t svo = t16_trellis_bytes(k2, k, n) + 2 * (size_t)k;
  float o[4] = {0, 0, 0, 0};
  for (int kk = 0; kk < K; kk++) {
    const int j = t * K + kk;
    const ExpSlot s = tab[layer * E + sel[j]];
    if (s.gate == ~0ull) continue;
    float v[4] = {0, 0, 0, 0};
    for (int q = 0; q < S; q++)
#pragma unroll
      for (int r = 0; r < 4; r++) v[r] += pd[((size_t)q * picks + j) * n + i + r];
    t16_had128_warp(v, lane);
    const __half *sv = (const __half *)(base + s.down + svo);
#pragma unroll
    for (int r = 0; r < 4; r++) o[r] += v[r] * __half2float(sv[i + r]);
  }
#pragma unroll
  for (int r = 0; r < 4; r++) out[(size_t)t * n + i + r] = o[r];
}
