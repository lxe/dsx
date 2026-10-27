// DeepSeek-V4.1-Flash kernels for sm_86 (RTX 3090), also built for sm_120 (dev box).
// Activations f32 [T][n] row-major; caches bf16; weights stay in their GGUF block layout.
// Every kernel mirrors a step of model.rs's CPU reference, which follows DeepSeek's model.py.
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>

#define MAXT 8

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
__device__ __forceinline__ float bf16r(float x) { return __bfloat162float(__float2bfloat16(x)); }
// hc_post for element i (output xv of the sublayer): the one expression hc_post_mb and the fused kernels share
__device__ __forceinline__ void hc_post_elem(float *h, int i, int d, const float *m, float xv) {
  float r0 = h[i], r1 = h[d + i], r2 = h[2 * d + i], r3 = h[3 * d + i];
#pragma unroll
  for (int k = 0; k < 4; k++) h[k * d + i] = bf16r(m[4 + k] * xv + m[8 + k] * r0 + m[12 + k] * r1 + m[16 + k] * r2 + m[20 + k] * r3);
}
__device__ __forceinline__ float ld_h(const uint8_t *p) {
  return __half2float(__ushort_as_half((uint16_t)p[0] | ((uint16_t)p[1] << 8)));
}

// block-wide sum over blockDim.x threads (<= 1024); `sh` holds 32 floats
__device__ float block_sum(float v, float *sh) {
  v = warp_sum(v);
  int w = threadIdx.x >> 5, l = threadIdx.x & 31;
  __syncthreads();
  if (l == 0) sh[w] = v;
  __syncthreads();
  int nw = (blockDim.x + 31) >> 5;
  v = (threadIdx.x < nw) ? sh[threadIdx.x] : 0.f;
  if (w == 0) v = warp_sum(v);
  if (threadIdx.x == 0) sh[0] = v;
  __syncthreads();
  float r = sh[0];
  __syncthreads();
  return r;
}

// ---------------------------------------------------------------- GEMV, T <= 8 tokens
// y[t][r] = sum_k W[r][k] x[t][k]. One warp per row, 4 rows per 128-thread block.
// ldx/ldy: row strides of x and y (floats). Weights: row r at w + r * row_bytes.

extern "C" __global__ void gemv_q8_0(const uint8_t *__restrict__ w, const float *__restrict__ x, float *__restrict__ y,
                                     int rows, int k, int T, int ldx, int ldy) {
  int r = blockIdx.x * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (r >= rows) return;
  int nb = k / 32;
  const uint8_t *wr = w + (size_t)r * nb * 34;
  float acc[MAXT] = {0};
  for (int b = lane; b < nb; b += 32) {
    const uint8_t *blk = wr + b * 34;
    float d = ld_h(blk);
    int8_t q[32];
#pragma unroll
    for (int j = 0; j < 32; j++) q[j] = (int8_t)blk[2 + j];
    for (int t = 0; t < T; t++) {
      const float4 *xp = (const float4 *)(x + (size_t)t * ldx + b * 32);
      float s = 0.f;
#pragma unroll
      for (int j = 0; j < 8; j++) {
        float4 v = xp[j];
        s += q[4 * j] * v.x + q[4 * j + 1] * v.y + q[4 * j + 2] * v.z + q[4 * j + 3] * v.w;
      }
      acc[t] += d * s;
    }
  }
  for (int t = 0; t < T; t++) {
    float s = warp_sum(acc[t]);
    if (lane == 0) y[(size_t)t * ldy + r] = s;
  }
}

extern "C" __global__ void gemv_f16(const __half *__restrict__ w, const float *__restrict__ x, float *__restrict__ y,
                                    int rows, int k, int T, int ldx, int ldy) {
  int r = blockIdx.x * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (r >= rows) return;
  const __half *wr = w + (size_t)r * k;
  float acc[MAXT] = {0};
  for (int i = lane * 8; i < k; i += 256) {
    uint4 raw = *(const uint4 *)(wr + i);
    const __half *h = (const __half *)&raw;
    float wf[8];
#pragma unroll
    for (int j = 0; j < 8; j++) wf[j] = __half2float(h[j]);
    for (int t = 0; t < T; t++) {
      const float4 *xp = (const float4 *)(x + (size_t)t * ldx + i);
      float4 a = xp[0], b = xp[1];
      acc[t] += wf[0] * a.x + wf[1] * a.y + wf[2] * a.z + wf[3] * a.w + wf[4] * b.x + wf[5] * b.y + wf[6] * b.z + wf[7] * b.w;
    }
  }
  for (int t = 0; t < T; t++) {
    float s = warp_sum(acc[t]);
    if (lane == 0) y[(size_t)t * ldy + r] = s;
  }
}

extern "C" __global__ void gemv_f32(const float *__restrict__ w, const float *__restrict__ x, float *__restrict__ y,
                                    int rows, int k, int T, int ldx, int ldy) {
  int r = blockIdx.x * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (r >= rows) return;
  const float *wr = w + (size_t)r * k;
  float acc[MAXT] = {0};
  for (int i = lane * 4; i < k; i += 128) {
    float4 wv = *(const float4 *)(wr + i);
    for (int t = 0; t < T; t++) {
      float4 a = *(const float4 *)(x + (size_t)t * ldx + i);
      acc[t] += wv.x * a.x + wv.y * a.y + wv.z * a.z + wv.w * a.w;
    }
  }
  for (int t = 0; t < T; t++) {
    float s = warp_sum(acc[t]);
    if (lane == 0) y[(size_t)t * ldy + r] = s;
  }
}

// ---------------------------------------------------------------- elementwise / norms

// one block per row: out = w * x * rsqrt(mean(x^2) + eps), optionally bf16-rounded
extern "C" __global__ void rmsnorm(const float *x, const float *w, float *out, int n, int ld_in, int ld_out, float eps,
                                   int round) {
  __shared__ float sh[32];
  const float *xr = x + (size_t)blockIdx.x * ld_in;
  float *o = out + (size_t)blockIdx.x * ld_out;
  float s = 0.f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) s += xr[i] * xr[i];
  s = block_sum(s, sh);
  float r = rsqrtf(s / n + eps);
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    float v = w[i] * (xr[i] * r);
    o[i] = round ? bf16r(v) : v;
  }
}

extern "C" __global__ void round_bf16(float *x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = bf16r(x[i]);
}

// RoPE on the last `rd` dims of each of `heads` vectors of length `hd`, token t at position pos0 + t*pos_step.
// freqs: [rd/2] f32. Grid (T, any: splits the heads), block 256.
extern "C" __global__ void rope(float *x, const float *freqs, int heads, int hd, int rd, int ld, int pos0, int inverse) {
  int t = blockIdx.x;
  float pos = (float)(pos0 + t);
  int half = rd / 2;
  for (int i = blockIdx.y * blockDim.x + threadIdx.x; i < heads * half; i += gridDim.y * blockDim.x) {
    int h = i / half, p = i % half;
    float ang = pos * freqs[p];
    float s, c;
    sincosf(ang, &s, &c);
    if (inverse) s = -s;
    float *v = x + (size_t)t * ld + h * hd + hd - rd + 2 * p;
    float a = v[0], b = v[1];
    v[0] = a * c - b * s;
    v[1] = a * s + b * c;
  }
}

// ---- fake quantizers (training-time FP8/FP4), see numerics.rs
__device__ float e4m3_round(float v) {
  if (v == 0.f) return 0.f;
  float a = fabsf(v);
  float q;
  if (a < 0.015625f) {  // 2^-6
    q = rintf(a * 512.f) / 512.f;
  } else {
    int e;
    frexpf(a, &e);  // a = m * 2^e, m in [0.5, 1)
    float step = ldexpf(1.f, e - 1 - 3);
    q = rintf(a / step) * step;
  }
  return copysignf(fminf(q, 448.f), v);
}
__device__ float pow2_ceil(float x) {
  uint32_t b = __float_as_uint(x);
  int e = (int)((b >> 23) & 0xFF) - 127 + ((b & 0x7FFFFF) ? 1 : 0);
  return __uint_as_float((uint32_t)(e + 127) << 23);
}
__device__ float e2m1_round(float v) {
  float a = fminf(fabsf(v), 6.f);
  // grid 0, .5, 1, 1.5, 2, 3, 4, 6 with ties to even mantissa
  float q;
  if (a <= 0.25f) q = 0.f;
  else if (a < 0.75f) q = 0.5f;
  else if (a <= 1.25f) q = 1.f;
  else if (a < 1.75f) q = 1.5f;
  else if (a <= 2.5f) q = 2.f;
  else if (a < 3.5f) q = 3.f;
  else if (a <= 5.f) q = 4.f;
  else q = 6.f;
  return copysignf(q, v);
}

// one warp per block of 32: FP8 E4M3 with a power-of-two scale. n multiple of 32.
extern "C" __global__ void fake_fp8(float *x, int n) {
  int b = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (b * 32 >= n) return;
  float v = x[b * 32 + lane];
  float amax = fmaxf(warp_max(fabsf(v)), 1e-4f);
  float s = pow2_ceil(amax / 448.f);
  x[b * 32 + lane] = e4m3_round(fminf(fmaxf(v / s, -448.f), 448.f)) * s;
}

// decode KV head in one launch: rmsnorm in place (no rounding), RoPE on its last rd dims at `pos`, fake FP8 per 32, then
// bf16 into the window slot `dst`. Same arithmetic as rmsnorm (block 1024) + rope + fake_fp8 + store_bf16. Grid 1,
// block 1024 (n <= 1024).
extern "C" __global__ void kv_norm_rope_store(float *x, const float *w, int n, float eps, const float *freqs, int rd, int pos,
                                              __nv_bfloat16 *dst) {
  __shared__ float sh[32];
  float s = 0.f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) s += x[i] * x[i];
  s = block_sum(s, sh);
  float r = rsqrtf(s / n + eps);
  for (int i = threadIdx.x; i < n; i += blockDim.x) x[i] = w[i] * (x[i] * r);
  __syncthreads();
  int half = rd / 2;
  for (int p = threadIdx.x; p < half; p += blockDim.x) {
    float ang = (float)pos * freqs[p];
    float sn, cs;
    sincosf(ang, &sn, &cs);
    float *v = x + n - rd + 2 * p;
    float a = v[0], b = v[1];
    v[0] = a * cs - b * sn;
    v[1] = a * sn + b * cs;
  }
  __syncthreads();
  int i = threadIdx.x;
  if (i < n) {
    float v = x[i];
    float amax = fmaxf(warp_max(fabsf(v)), 1e-4f);
    float sc = pow2_ceil(amax / 448.f);
    v = e4m3_round(fminf(fmaxf(v / sc, -448.f), 448.f)) * sc;
    x[i] = v;
    dst[i] = __float2bfloat16(v);
  }
}

// FP4 E2M1 per block (16 or 32, one warp handles 32 values = 1 or 2 blocks); e4m3s: scale kind
extern "C" __global__ void fake_fp4(float *x, int n, int block, int e4m3s) {
  int b = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (b * 32 >= n) return;
  float v = x[b * 32 + lane];
  float a = fabsf(v);
  // reduce within groups of `block` lanes
  for (int o = block / 2; o; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, o));
  float s = e4m3s ? e4m3_round(fmaxf(a, 6.f * 0.001953125f) / 6.f) : pow2_ceil(fmaxf(a, 6.f * 1.1754944e-38f) / 6.f);
  x[b * 32 + lane] = e2m1_round(fminf(fmaxf(v / s, -6.f), 6.f)) * s;
}

// ---------------------------------------------------------------- hyper-connections

// hc_mixes + Sinkhorn: per token, mixes[i] = fn[i] . h * rsqrt(mean(h^2)+eps), then pre/post/comb.
// fn: f16 [mix][hcd]. h: [T][hcd]. out: [T][24] = pre[4], post[4], comb[16]. Block 768 = 24 warps, grid T.
extern "C" __global__ void hc_mixes(const float *fn, const float *h, const float *scale, const float *base, float *out,
                                    int hcd, float eps, float hc_eps, int iters) {
  __shared__ float sh[32];
  __shared__ float mix[24];
  int t = blockIdx.x, wid = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const float *x = h + (size_t)t * hcd;
  float ss = 0.f;
  for (int i = threadIdx.x; i < hcd; i += blockDim.x) ss += x[i] * x[i];
  ss = block_sum(ss, sh);
  float r = rsqrtf(ss / hcd + eps);
  {
    const float *row = fn + (size_t)wid * hcd;
    float acc = 0.f;
    for (int i = lane * 4; i < hcd; i += 128) {
      float4 wv = *(const float4 *)(row + i);
      acc += wv.x * x[i] + wv.y * x[i + 1] + wv.z * x[i + 2] + wv.w * x[i + 3];
    }
    acc = warp_sum(acc);
    if (lane == 0) mix[wid] = acc * r;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    const int hc = 4;
    float *o = out + t * 24;
    for (int j = 0; j < hc; j++) o[j] = 1.f / (1.f + expf(-(mix[j] * scale[0] + base[j]))) + hc_eps;
    for (int j = 0; j < hc; j++) o[4 + j] = 2.f / (1.f + expf(-(mix[j + hc] * scale[1] + base[j + hc])));
    float c[16];
    for (int j = 0; j < 16; j++) c[j] = mix[j + 2 * hc] * scale[2] + base[j + 2 * hc];
    for (int j = 0; j < hc; j++) {
      float mx = -INFINITY;
      for (int k = 0; k < hc; k++) mx = fmaxf(mx, c[j * hc + k]);
      float s = 0.f;
      for (int k = 0; k < hc; k++) { c[j * hc + k] = expf(c[j * hc + k] - mx); s += c[j * hc + k]; }
      for (int k = 0; k < hc; k++) c[j * hc + k] = c[j * hc + k] / s + hc_eps;
    }
    for (int k = 0; k < hc; k++) {
      float s = 0.f;
      for (int j = 0; j < hc; j++) s += c[j * hc + k];
      for (int j = 0; j < hc; j++) c[j * hc + k] /= s + hc_eps;
    }
    for (int it = 0; it < iters - 1; it++) {
      for (int j = 0; j < hc; j++) {
        float s = 0.f;
        for (int k = 0; k < hc; k++) s += c[j * hc + k];
        for (int k = 0; k < hc; k++) c[j * hc + k] /= s + hc_eps;
      }
      for (int k = 0; k < hc; k++) {
        float s = 0.f;
        for (int j = 0; j < hc; j++) s += c[j * hc + k];
        for (int j = 0; j < hc; j++) c[j * hc + k] /= s + hc_eps;
      }
    }
    for (int j = 0; j < 16; j++) o[8 + j] = c[j];
  }
}

// y[t] = bf16(sum_k mix[t][moff + k] * h[t][k]); grid T, block 256
extern "C" __global__ void hc_pre(const float *h, const float *mixes, int moff, float *y, int d) {
  int t = blockIdx.x;
  const float *m = mixes + t * 24 + moff;
  const float *ht = h + (size_t)t * 4 * d;
  for (int i = threadIdx.x; i < d; i += blockDim.x)
    y[(size_t)t * d + i] = bf16r(m[0] * ht[i] + m[1] * ht[d + i] + m[2] * ht[2 * d + i] + m[3] * ht[3 * d + i]);
}

// h[t][k] = bf16(post[k] x[t] + sum_j comb[j][k] res[t][j]); in place on h (h holds res). grid T, block 256
extern "C" __global__ void hc_post(float *h, const float *x, const float *mixes, int d) {
  int t = blockIdx.x;
  const float *m = mixes + t * 24;
  float *ht = h + (size_t)t * 4 * d;
  for (int i = threadIdx.x; i < d; i += blockDim.x) {
    float r0 = ht[i], r1 = ht[d + i], r2 = ht[2 * d + i], r3 = ht[3 * d + i], xv = x[(size_t)t * d + i];
#pragma unroll
    for (int k = 0; k < 4; k++)
      ht[k * d + i] = bf16r(m[4 + k] * xv + m[8 + k] * r0 + m[8 + 4 + k] * r1 + m[8 + 8 + k] * r2 + m[8 + 12 + k] * r3);
  }
}

// h[t] = 4 copies of embedding row; emb f16 [vocab][d]; grid T
extern "C" __global__ void embed_hc(const __half *emb, const int *tok, float *h, int d) {
  int t = blockIdx.x;
  const __half *e = emb + (size_t)tok[t] * d;
  for (int i = threadIdx.x; i < d; i += blockDim.x) {
    float v = __half2float(e[i]);
    for (int k = 0; k < 4; k++) h[(size_t)t * 4 * d + k * d + i] = v;
  }
}

// ---------------------------------------------------------------- attention (decode, MQA, sparse)
// For token t: rows = window slots (bf16 [W][hd], ring) listed in win_idx[t][W] (-1 = empty) plus compressed rows
// comp (bf16 [n][hd]) listed in comp_idx[t][K] (-1 = none). q: [T][H][hd] f32, out o: [T][H][hd] f32 (bf16-rounded).
// Grid (T, H/8), block 256: 8 warps = 8 heads, one row at a time, online softmax.
extern "C" __global__ void sparse_attn(const float *q, const __nv_bfloat16 *win, const int *win_idx, int W,
                                       const __nv_bfloat16 *comp, const int *comp_idx, int K, const float *sinks, float *o,
                                       int H, int hd, float scale) {
  int t = blockIdx.x, h = blockIdx.y * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  const float *qh = q + ((size_t)t * H + h) * hd;
  // hd = 512: each lane owns 16 dims
  float qv[16], acc[16];
#pragma unroll
  for (int j = 0; j < 16; j++) { qv[j] = qh[lane * 16 + j]; acc[j] = 0.f; }
  float m = -1e30f, l = 0.f;
  int nrows = W + K;
  for (int i = 0; i < nrows; i++) {
    int idx = i < W ? win_idx[t * W + i] : comp_idx[t * K + (i - W)];
    if (idx < 0) continue;
    const __nv_bfloat16 *row = (i < W ? win : comp) + (size_t)idx * hd + lane * 16;
    float rv[16];
    uint4 a = *(const uint4 *)row, b = *(const uint4 *)(row + 8);
    const __nv_bfloat16 *pa = (const __nv_bfloat16 *)&a, *pb = (const __nv_bfloat16 *)&b;
#pragma unroll
    for (int j = 0; j < 8; j++) { rv[j] = __bfloat162float(pa[j]); rv[8 + j] = __bfloat162float(pb[j]); }
    float s = 0.f;
#pragma unroll
    for (int j = 0; j < 16; j++) s += qv[j] * rv[j];
    s = warp_sum(s) * scale;
    float mn = fmaxf(m, s);
    float corr = expf(m - mn), p = expf(s - mn);
    l = l * corr + p;
#pragma unroll
    for (int j = 0; j < 16; j++) acc[j] = acc[j] * corr + p * rv[j];
    m = mn;
  }
  l += expf(sinks[h] - m);
  float *oh = o + ((size_t)t * H + h) * hd;
#pragma unroll
  for (int j = 0; j < 16; j++) oh[lane * 16 + j] = bf16r(acc[j] / l);
}

// Split-K attention for T <= 8 queries (decode, verify), same contract as sparse_attn, whose serial row loop is
// latency-bound (8 blocks, a warp reduction + exp per row: 276 us for 640 rows on a 3090; this: 21 us). MQA: all heads
// share the rows. Grid (T, H/16, ceil((W+K)/32)), block 256: block z stages
// rows [32z, 32z+32) in shared memory once and serves 16 heads. Scores: warp w = heads 2w, 2w+1, lane = row (q read
// as a broadcast); softmax over the 32 rows per head; PV: thread = 2 output dims for all 16 heads. Same part layout
// part[t][z][h][hd + 2] = unnormalized acc, m, l; sparse_attn_merge adds the splits and the sink. hd = 512.
#define AP_R 32
#define AP_H 16
#define AP_LD 520
extern "C" __global__ void __launch_bounds__(256) sparse_attn_tile(const float *q, const __nv_bfloat16 *win, const int *win_idx, int W,
                                                                   const __nv_bfloat16 *comp, const int *comp_idx, int K, float *part,
                                                                   int H, int hd, float scale) {
  __shared__ __align__(16) __nv_bfloat16 R[AP_R][AP_LD];
  __shared__ float P[AP_H][AP_R];
  __shared__ float Mh[AP_H], Lh[AP_H];
  __shared__ int ok[AP_R];
  int t = blockIdx.x, h0 = blockIdx.y * AP_H, z = blockIdx.z, S = gridDim.z, tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
  int nrows = W + K;
  // stage 32 rows: 32 x 64 chunks of 16 bytes, 8 per thread
  for (int c = tid; c < AP_R * 64; c += 256) {
    int r = c >> 6, part16 = c & 63, i = z * AP_R + r;
    int idx = i >= nrows ? -1 : (i < W ? win_idx[t * W + i] : comp_idx[t * K + (i - W)]);
    uint4 v = make_uint4(0, 0, 0, 0);
    if (idx >= 0) v = *(const uint4 *)((i < W ? win : comp) + (size_t)idx * hd + part16 * 8);
    *(uint4 *)&R[r][part16 * 8] = v;
    if (part16 == 0) ok[r] = idx >= 0;
  }
  __syncthreads();
  // scores: lane = row, warp = 2 heads
  {
    const float *qa = q + ((size_t)t * H + h0 + 2 * w) * hd, *qb = qa + hd;
    float sa = 0.f, sb = 0.f;
    for (int d = 0; d < 512; d += 8) {
      uint4 rv = *(const uint4 *)&R[lane][d];
      const __nv_bfloat162 *r2 = (const __nv_bfloat162 *)&rv;
      float4 a0 = __ldg((const float4 *)(qa + d)), a1 = __ldg((const float4 *)(qa + d + 4));
      float4 b0 = __ldg((const float4 *)(qb + d)), b1 = __ldg((const float4 *)(qb + d + 4));
      float2 x0 = __bfloat1622float2(r2[0]), x1 = __bfloat1622float2(r2[1]), x2 = __bfloat1622float2(r2[2]), x3 = __bfloat1622float2(r2[3]);
      sa += a0.x * x0.x + a0.y * x0.y + a0.z * x1.x + a0.w * x1.y + a1.x * x2.x + a1.y * x2.y + a1.z * x3.x + a1.w * x3.y;
      sb += b0.x * x0.x + b0.y * x0.y + b0.z * x1.x + b0.w * x1.y + b1.x * x2.x + b1.y * x2.y + b1.z * x3.x + b1.w * x3.y;
    }
    bool v = ok[lane];
    sa = v ? sa * scale : -1e30f;
    sb = v ? sb * scale : -1e30f;
    float ma = warp_max(sa), mb = warp_max(sb);
    float pa = v ? expf(sa - ma) : 0.f, pb = v ? expf(sb - mb) : 0.f;
    float la = warp_sum(pa), lb = warp_sum(pb);
    P[2 * w][lane] = pa;
    P[2 * w + 1][lane] = pb;
    if (lane == 0) { Mh[2 * w] = ma; Lh[2 * w] = la; Mh[2 * w + 1] = mb; Lh[2 * w + 1] = lb; }
  }
  __syncthreads();
  // PV: thread = dims 2*tid, 2*tid+1
  float acc[AP_H][2];
#pragma unroll
  for (int h = 0; h < AP_H; h++) acc[h][0] = acc[h][1] = 0.f;
  for (int r = 0; r < AP_R; r++) {
    float2 x = __bfloat1622float2(*(const __nv_bfloat162 *)&R[r][2 * tid]);
#pragma unroll
    for (int h = 0; h < AP_H; h++) {
      float p = P[h][r];
      acc[h][0] += p * x.x;
      acc[h][1] += p * x.y;
    }
  }
#pragma unroll
  for (int h = 0; h < AP_H; h++) {
    float *ph = part + (((size_t)t * S + z) * H + h0 + h) * (hd + 2);
    *(float2 *)(ph + 2 * tid) = make_float2(acc[h][0], acc[h][1]);
  }
  if (tid < AP_H) {
    float *ph = part + (((size_t)t * S + z) * H + h0 + tid) * (hd + 2);
    ph[hd] = Mh[tid];
    ph[hd + 1] = Lh[tid];
  }
}

// merge for sparse_attn_tile: grid (T, H), block 256 (thread = 2 dims)
extern "C" __global__ void sparse_attn_merge(const float *part, int S, const float *sinks, float *o, int H, int hd) {
  int t = blockIdx.x, h = blockIdx.y, tid = threadIdx.x;
  const float *p0 = part + ((size_t)t * S * H + h) * (hd + 2);
  size_t zs = (size_t)H * (hd + 2);
  float M = -1e30f;
  for (int z = 0; z < S; z++) M = fmaxf(M, p0[z * zs + hd]);
  float L = 0.f, a0 = 0.f, a1 = 0.f;
  for (int z = 0; z < S; z++) {
    const float *pz = p0 + z * zs;
    float f = expf(pz[hd] - M);
    L += pz[hd + 1] * f;
    float2 v = *(const float2 *)(pz + 2 * tid);
    a0 += v.x * f;
    a1 += v.y * f;
  }
  L += expf(sinks[h] - M);
  float *oh = o + ((size_t)t * H + h) * hd;
  oh[2 * tid] = bf16r(a0 / L);
  oh[2 * tid + 1] = bf16r(a1 / L);
}

// decode (T = 1): sparse_attn_merge, then the inverse RoPE on each head's last rd dims at `pos` (rope with inverse = 1)
// and fake FP8 per 32 values (fake_fp8), in one launch, bit-identical to the three: thread tid holds dims 2 tid, 2 tid + 1
// (a RoPE pair when 2 tid >= hd - rd), and 16 threads hold a fake-FP8 block of 32. Grid (1, H), block hd / 2 (= 256).
extern "C" __global__ void sparse_attn_merge_rf(const float *part, int S, const float *sinks, float *o, int H, int hd,
                                                const float *freqs, int rd, int pos) {
  int h = blockIdx.y, tid = threadIdx.x;
  const float *p0 = part + (size_t)h * (hd + 2);
  size_t zs = (size_t)H * (hd + 2);
  float M = -1e30f;
  for (int z = 0; z < S; z++) M = fmaxf(M, p0[z * zs + hd]);
  float L = 0.f, a0 = 0.f, a1 = 0.f;
  for (int z = 0; z < S; z++) {
    const float *pz = p0 + z * zs;
    float f = expf(pz[hd] - M);
    L += pz[hd + 1] * f;
    float2 v = *(const float2 *)(pz + 2 * tid);
    a0 += v.x * f;
    a1 += v.y * f;
  }
  L += expf(sinks[h] - M);
  float x0 = bf16r(a0 / L), x1 = bf16r(a1 / L);
  int p = tid - (hd - rd) / 2;
  if (p >= 0) {
    float ang = (float)pos * freqs[p];
    float sn, cs;
    sincosf(ang, &sn, &cs);
    sn = -sn;
    float a = x0, b = x1;
    // as rope compiles it: fma(a, c, -(b * s)) and fma(b, c, a * s), products rounded first
    x0 = __fmaf_rn(a, cs, -__fmul_rn(b, sn));
    x1 = __fmaf_rn(b, cs, __fmul_rn(a, sn));
  }
  float amax = fmaxf(fabsf(x0), fabsf(x1));
#pragma unroll
  for (int off = 1; off < 16; off <<= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));
  amax = fmaxf(amax, 1e-4f);
  float sc = pow2_ceil(amax / 448.f);
  float *oh = o + (size_t)h * hd;
  oh[2 * tid] = e4m3_round(fminf(fmaxf(x0 / sc, -448.f), 448.f)) * sc;
  oh[2 * tid + 1] = e4m3_round(fminf(fmaxf(x1 / sc, -448.f), 448.f)) * sc;
}

// ---------------------------------------------------------------- attention (prefill, tensor cores)
// Same contract as sparse_attn; hd = 512, H % 32 == 0. Grid (T, H/32), block 256, dynamic smem attn::smem_bytes(W + K).
// Per token the 32 heads of a block share the rows: S = Q . R^T, O += P . R (mma m16n8k16 bf16, f32 accumulate),
// online softmax over tiles of TC_NT gathered rows (cp.async, double-buffered, 16-byte chunks XOR-swizzled by row).
// The valid row indices are compacted first, so -1 entries cost nothing. sparse_attn_tc splits q and p into bf16 hi + lo
// (2 mma each; matches sparse_attn to output rounding); sparse_attn_tc_fast rounds them to one bf16 (1.6x faster).
// Warps: S phase warp w = head half (w & 1) x 128-dim slice (w >> 1), partial sums reduced through smem;
// softmax: 8 threads per head; PV phase warp w = 64-dim output slice for all 32 heads (64 f32 accumulators).
#define TC_NT 32
__device__ __forceinline__ uint32_t smem_u32(const void *p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], const void *p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldsm4t(uint32_t (&r)[4], const void *p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void mma_bf16(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// 16-byte async copy; sz = 0 zero-fills
__device__ __forceinline__ void cp_async16(void *dst, const void *src, int sz) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(smem_u32(dst)), "l"(src), "r"(sz));
}
__device__ __forceinline__ uint32_t pack_bf16(float a, float b) {
  __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
  return *(uint32_t *)&v;
}

template <bool SPLIT>
__device__ __forceinline__ void sparse_attn_tc_body(const float *q, const __nv_bfloat16 *win, const int *win_idx, int W,
                                                    const __nv_bfloat16 *comp, const int *comp_idx, int K, const float *sinks,
                                                    float *o, int H, float scale) {
  constexpr int NT = TC_NT, PS = NT + 4, PP = NT + 8, RPT = NT / 8, NQ = SPLIT ? 2 : 1;
  extern __shared__ uint4 smem[];
  __nv_bfloat16 *rb = (__nv_bfloat16 *)smem;              // [2][NT][512] row tiles
  float *part = (float *)(rb + 2 * NT * 512);              // [4 slices][32 heads][PS] partial scores
  __nv_bfloat16 *pb = (__nv_bfloat16 *)(part + 4 * 32 * PS);  // [NQ][32][PP] probabilities (hi, lo)
  float *corr = (float *)(pb + NQ * 32 * PP), *lsum = corr + 32;
  int *wsum = (int *)(lsum + 32), *list = wsum + 8;        // list: valid rows, row * 2 + (from comp)
  int t = blockIdx.x, hb = blockIdx.y * 32, tid = threadIdx.x, w = tid >> 5, lane = tid & 31, g = lane >> 2, c = lane & 3;
  const int *wi = win_idx + (size_t)t * W, *ci = comp_idx + (size_t)t * K;

  // compact valid entries (in order): per-thread contiguous runs + block scan
  int n = W + K, per = (n + 255) / 256, i0 = min(tid * per, n), i1 = min(i0 + per, n), cnt = 0;
  for (int i = i0; i < i1; i++) cnt += (i < W ? wi[i] : ci[i - W]) >= 0;
  int inc = cnt;
#pragma unroll
  for (int d = 1; d < 32; d <<= 1) {
    int v = __shfl_up_sync(0xffffffff, inc, d);
    if (lane >= d) inc += v;
  }
  if (lane == 31) wsum[w] = inc;

  // Q fragments for this warp's S-phase slice: heads hb + mt*16 + {g, g+8}, dims ks*128 + kk*16 + {2c, 2c+8}
  int mt = w & 1, ks = w >> 1;
  uint32_t qa[NQ][8][4];  // SPLIT: q = hi + lo, both bf16
  {
    const float *q0 = q + ((size_t)t * H + hb + mt * 16 + g) * 512 + ks * 128 + 2 * c, *q1 = q0 + 8 * 512;
#pragma unroll
    for (int kk = 0; kk < 8; kk++) {
      float2 x[4] = {*(const float2 *)(q0 + kk * 16), *(const float2 *)(q1 + kk * 16), *(const float2 *)(q0 + kk * 16 + 8),
                     *(const float2 *)(q1 + kk * 16 + 8)};
#pragma unroll
      for (int j = 0; j < 4; j++) {
        qa[0][kk][j] = pack_bf16(x[j].x, x[j].y);
        if (SPLIT) {
          float hx = __bfloat162float(__float2bfloat16(x[j].x)), hy = __bfloat162float(__float2bfloat16(x[j].y));
          qa[NQ - 1][kk][j] = pack_bf16(x[j].x - hx, x[j].y - hy);
        }
      }
    }
  }
  __syncthreads();
  int pos = inc - cnt, nv = 0;
#pragma unroll
  for (int k = 0; k < 8; k++) {
    int s = wsum[k];
    pos += k < w ? s : 0;
    nv += s;
  }
  for (int i = i0; i < i1; i++) {
    int idx = i < W ? wi[i] : ci[i - W];
    if (idx >= 0) list[pos++] = idx * 2 + (i >= W);
  }
  __syncthreads();

  auto load = [&](int tile) {
    __nv_bfloat16 *dst = rb + (tile & 1) * NT * 512;
#pragma unroll
    for (int k = tid; k < NT * 64; k += 256) {
      int r = k >> 6, ch = k & 63, e = tile * NT + r, sz = 0;
      const __nv_bfloat16 *src = win;
      if (e < nv) {
        int v = list[e];
        src = ((v & 1) ? comp : win) + (size_t)(v >> 1) * 512 + ch * 8, sz = 16;
      }
      cp_async16(dst + r * 512 + ((ch ^ (r & 7)) << 3), src, sz);
    }
    asm volatile("cp.async.commit_group;");
  };

  int sh = tid >> 3, sj = tid & 7;  // softmax role: head sh, rows sj*RPT ..
  float m_run = -1e30f, l_run = 0.f;
  float oacc[2][8][4];
#pragma unroll
  for (int a = 0; a < 2; a++)
#pragma unroll
    for (int b = 0; b < 8; b++) oacc[a][b][0] = oacc[a][b][1] = oacc[a][b][2] = oacc[a][b][3] = 0.f;
  int ntile = (nv + NT - 1) / NT, mi = lane >> 3, lr = lane & 7;
  if (ntile) load(0);
  for (int it = 0; it < ntile; it++) {
    if (it + 1 < ntile) {
      load(it + 1);
      asm volatile("cp.async.wait_group 1;");
    } else {
      asm volatile("cp.async.wait_group 0;");
    }
    __syncthreads();
    const __nv_bfloat16 *R = rb + (it & 1) * NT * 512;

    // S partials: 16 heads x NT rows over this warp's 128 dims
    float sacc[NT / 8][4];
#pragma unroll
    for (int b = 0; b < NT / 8; b++) sacc[b][0] = sacc[b][1] = sacc[b][2] = sacc[b][3] = 0.f;
#pragma unroll
    for (int kk = 0; kk < 8; kk++)
#pragma unroll
      for (int np = 0; np < NT / 16; np++) {
        uint32_t b[4];
        int r = np * 16 + (mi >> 1) * 8 + lr, ch = ks * 16 + kk * 2 + (mi & 1);
        ldsm4(b, R + r * 512 + ((ch ^ (r & 7)) << 3));
#pragma unroll
        for (int z = 0; z < NQ; z++) {
          mma_bf16(sacc[2 * np], qa[z][kk], b[0], b[1]);
          mma_bf16(sacc[2 * np + 1], qa[z][kk], b[2], b[3]);
        }
      }
    float *pw = part + (ks * 32 + mt * 16 + g) * PS + 2 * c;
#pragma unroll
    for (int b = 0; b < NT / 8; b++) {
      *(float2 *)(pw + b * 8) = make_float2(sacc[b][0], sacc[b][1]);
      *(float2 *)(pw + 8 * PS + b * 8) = make_float2(sacc[b][2], sacc[b][3]);
    }
    __syncthreads();

    // online softmax: reduce the 4 slices, mask the tile tail, p rounded to bf16 (hi + lo) and l sums the rounded p
    {
      float sv[RPT], mx = -INFINITY;
#pragma unroll
      for (int k = 0; k < RPT; k++) {
        float s = 0.f;
#pragma unroll
        for (int z = 0; z < 4; z++) s += part[(z * 32 + sh) * PS + sj * RPT + k];
        sv[k] = it * NT + sj * RPT + k < nv ? s * scale : -INFINITY;
        mx = fmaxf(mx, sv[k]);
      }
#pragma unroll
      for (int d = 1; d < 8; d <<= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, d));
      float mn = fmaxf(m_run, mx), cr = expf(m_run - mn), ps = 0.f;
#pragma unroll
      for (int k = 0; k < RPT; k++) {
        float pf = expf(sv[k] - mn);
        __nv_bfloat16 p = __float2bfloat16(pf);
        ps += __bfloat162float(p);
        pb[sh * PP + sj * RPT + k] = p;
        if (SPLIT) {
          __nv_bfloat16 pl = __float2bfloat16(pf - __bfloat162float(p));
          ps += __bfloat162float(pl);
          pb[(32 + sh) * PP + sj * RPT + k] = pl;
        }
      }
#pragma unroll
      for (int d = 1; d < 8; d <<= 1) ps += __shfl_xor_sync(0xffffffff, ps, d);
      l_run = l_run * cr + ps, m_run = mn;
      if (sj == 0) corr[sh] = cr;
    }
    __syncthreads();

    // O[32 heads][w*64 ..+64] = O * corr + P . R
#pragma unroll
    for (int a = 0; a < 2; a++) {
      float c0 = corr[a * 16 + g], c1 = corr[a * 16 + g + 8];
#pragma unroll
      for (int b = 0; b < 8; b++) oacc[a][b][0] *= c0, oacc[a][b][1] *= c0, oacc[a][b][2] *= c1, oacc[a][b][3] *= c1;
    }
#pragma unroll
    for (int kk = 0; kk < NT / 16; kk++) {
      uint32_t pa[NQ][2][4];
#pragma unroll
      for (int z = 0; z < NQ; z++)
#pragma unroll
        for (int a = 0; a < 2; a++) ldsm4(pa[z][a], pb + (z * 32 + a * 16 + (mi & 1) * 8 + lr) * PP + kk * 16 + (mi >> 1) * 8);
#pragma unroll
      for (int np = 0; np < 4; np++) {
        uint32_t b[4];
        int r = kk * 16 + (mi & 1) * 8 + lr, ch = w * 8 + np * 2 + (mi >> 1);
        ldsm4t(b, R + r * 512 + ((ch ^ (r & 7)) << 3));
#pragma unroll
        for (int z = 0; z < NQ; z++)
#pragma unroll
          for (int a = 0; a < 2; a++) {
            mma_bf16(oacc[a][2 * np], pa[z][a], b[0], b[1]);
            mma_bf16(oacc[a][2 * np + 1], pa[z][a], b[2], b[3]);
          }
      }
    }
    __syncthreads();
  }
  if (sj == 0) lsum[sh] = l_run + expf(sinks[hb + sh] - m_run);
  __syncthreads();
#pragma unroll
  for (int a = 0; a < 2; a++) {
    int h0 = a * 16 + g;
    float l0 = lsum[h0], l1 = lsum[h0 + 8];
    float *o0 = o + ((size_t)t * H + hb + h0) * 512 + w * 64 + 2 * c, *o1 = o0 + 8 * 512;
#pragma unroll
    for (int b = 0; b < 8; b++) {
      *(float2 *)(o0 + b * 8) = make_float2(bf16r(oacc[a][b][0] / l0), bf16r(oacc[a][b][1] / l0));
      *(float2 *)(o1 + b * 8) = make_float2(bf16r(oacc[a][b][2] / l1), bf16r(oacc[a][b][3] / l1));
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) sparse_attn_tc(const float *q, const __nv_bfloat16 *win, const int *win_idx, int W,
                                                                  const __nv_bfloat16 *comp, const int *comp_idx, int K,
                                                                  const float *sinks, float *o, int H, int hd, float scale) {
  sparse_attn_tc_body<true>(q, win, win_idx, W, comp, comp_idx, K, sinks, o, H, scale);
}
extern "C" __global__ void __launch_bounds__(256) sparse_attn_tc_fast(const float *q, const __nv_bfloat16 *win, const int *win_idx, int W,
                                                                       const __nv_bfloat16 *comp, const int *comp_idx, int K,
                                                                       const float *sinks, float *o, int H, int hd, float scale) {
  sparse_attn_tc_body<false>(q, win, win_idx, W, comp, comp_idx, K, sinks, o, H, scale);
}

// store a [hd] f32 vector into a bf16 cache row
extern "C" __global__ void store_bf16(const float *x, __nv_bfloat16 *dst, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = __float2bfloat16(x[i]);
}

// ---------------------------------------------------------------- indexer score (decode)
// score[t][s] = sum_h relu(q[t][h] . k[s]) * w[t][h], s < n_t (= n_rows visible to token t, given per token).
// q: [T][IH][ID] f32 (fake-fp4), k: bf16 [n][ID], w: [T][IH] (already scaled + bf16-rounded). Grid (ceil(n/8), T),
// block 256: one warp per position. ID = 128 (4 dims per lane), IH = 32.
extern "C" __global__ void indexer_score(const float *q, const __nv_bfloat16 *k, const float *w, float *score,
                                         const int *n_vis, int n, int IH, int ID) {
  int t = blockIdx.y, s = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (s >= n) return;
  float *sc = score + (size_t)t * n;
  if (s >= n_vis[t]) {
    if (lane == 0) sc[s] = -INFINITY;
    return;
  }
  const __nv_bfloat16 *kr = k + (size_t)s * ID + lane * 4;
  float kv[4];
  for (int j = 0; j < 4; j++) kv[j] = __bfloat162float(kr[j]);
  float tot = 0.f;
  for (int h = 0; h < IH; h++) {
    const float *qh = q + ((size_t)t * IH + h) * ID + lane * 4;
    float d = qh[0] * kv[0] + qh[1] * kv[1] + qh[2] * kv[2] + qh[3] * kv[3];
    d = warp_sum(d);
    tot += fmaxf(d, 0.f) * w[t * IH + h];
  }
  if (lane == 0) sc[s] = tot;
}

// ---------------------------------------------------------------- compressor (ratio 2 pooling)
// kv_state/score_state: [ratio][hd] f32; writes pooled latent (bf16-rounded) into out. grid 1, block 512
extern "C" __global__ void compress_pool(const float *kv_state, const float *score_state, float *out, int ratio, int hd) {
  for (int j = threadIdx.x; j < hd; j += blockDim.x) {
    float mx = -INFINITY;
    for (int s = 0; s < ratio; s++) mx = fmaxf(mx, score_state[s * hd + j]);
    float den = 0.f, num = 0.f;
    for (int s = 0; s < ratio; s++) {
      float e = expf(score_state[s * hd + j] - mx);
      den += e;
      num += kv_state[s * hd + j] * e;
    }
    out[j] = bf16r(num / den);
  }
}

// ---------------------------------------------------------------- router
// scores = sqrt(softplus(logit)); pick top-k of scores + bias; weights = score / sum * route_scale.
// logits: [T][E]. out_idx [T][k], out_w [T][k]. grid T, block 32 (E = 384, k = 6).
extern "C" __global__ void router_topk(const float *logits, const float *bias, int *out_idx, float *out_w, int E, int k,
                                       float route_scale, const float *sel_bonus) {
  // one warp per token; lane holds the keys of experts lane, lane + 32, ... in registers (E <= 512)
  int t = blockIdx.x, lane = threadIdx.x;
  const float *lg = logits + (size_t)t * E;
  __shared__ float sc[512];
  float key[16];
#pragma unroll
  for (int j = 0; j < 16; j++) {
    int e = lane + 32 * j;
    key[j] = -INFINITY;
    if (e < E) {
      float x = lg[e];
      float sp = x > 20.f ? x : log1pf(expf(x));
      float s = sqrtf(sp);
      // sel_bonus (optional, off by default): a selection-only nudge toward GPU-resident experts; weights stay exact
      float kk = s + bias[e] + (sel_bonus ? sel_bonus[e] : 0.f);
      // a NaN hidden state must not become an out-of-range expert index downstream
      key[j] = isnan(kk) ? -INFINITY : kk;
      sc[e] = isnan(s) ? 0.f : s;
    }
  }
  __syncwarp();
  float wsum = 0.f;
  int sel[8];
  for (int r = 0; r < k; r++) {
    // lane-local best: ascending expert index, ties keep the lower index
    float best = -INFINITY;
    int bi = 0x7fffffff;
#pragma unroll
    for (int j = 0; j < 16; j++)
      if (key[j] > best) { best = key[j]; bi = lane + 32 * j; }
    for (int o = 16; o; o >>= 1) {
      float ob = __shfl_xor_sync(0xffffffff, best, o);
      int oi = __shfl_xor_sync(0xffffffff, bi, o);
      if (ob > best || (ob == best && oi < bi)) { best = ob; bi = oi; }
    }
    if (bi >= E) bi = r;  // every key -inf (NaN input): any valid expert, zero weight below
    sel[r] = bi;
    wsum += sc[bi];
#pragma unroll
    for (int j = 0; j < 16; j++)
      if (lane + 32 * j == bi) key[j] = -INFINITY;
  }
  if (lane < k) {
    out_idx[t * k + lane] = sel[lane];
    out_w[t * k + lane] = sc[sel[lane]] / (wsum + 1e-20f) * route_scale;
  }
}

// ---------------------------------------------------------------- SwiGLU
// gu: [T][2*inter] (gate | up), out h [T][inter] = bf16(silu(min(g,lim)) * clamp(u,-lim,lim) * w[t])
extern "C" __global__ void swiglu(const float *g, const float *u, int ldgu, const float *w, float *h, int inter, float lim) {
  int t = blockIdx.y;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= inter) return;
  float gv = fminf(g[(size_t)t * ldgu + i], lim);
  float uv = fminf(fmaxf(u[(size_t)t * ldgu + i], -lim), lim);
  float wt = w ? w[t] : 1.f;
  h[(size_t)t * inter + i] = bf16r(gv / (1.f + expf(-gv)) * uv * wt);
}

// y += x (n floats)
extern "C" __global__ void add_inplace(float *y, const float *x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] += x[i];
}

// ---------------------------------------------------------------- engram gate (one token)
// h [hc][d] updated in place: h[k] = bf16(h[k] + sigmoid(signed_sqrt(dot_k)) * value), dot_k from key[k] (kv[0..hc*d]),
// value = kv[hc*d..]. qk: [hc][d]. grid hc, block 1024
extern "C" __global__ void engram_gate(float *h, const float *kv, const float *qk, int d, float eps) {
  __shared__ float sh[32];
  int k = blockIdx.x;
  float *hk = h + (size_t)k * d;
  const float *key = kv + (size_t)k * d, *val = kv + (size_t)4 * d;
  float sh2 = 0.f, sk2 = 0.f, sd = 0.f;
  for (int i = threadIdx.x; i < d; i += blockDim.x) {
    sh2 += hk[i] * hk[i];
    sk2 += key[i] * key[i];
    sd += hk[i] * qk[k * d + i] * key[i];
  }
  sh2 = block_sum(sh2, sh);
  sk2 = block_sum(sk2, sh);
  sd = block_sum(sd, sh);
  float dt = sd * rsqrtf(sh2 / d + eps) * rsqrtf(sk2 / d + eps) * rsqrtf((float)d);
  float g = 1.f / (1.f + expf(-copysignf(sqrtf(fmaxf(fabsf(dt), 1e-6f)), dt)));
  for (int i = threadIdx.x; i < d; i += blockDim.x) hk[i] = bf16r(hk[i] + g * val[i]);
}

// y[i] = x[i] * s (indexer weights scale + bf16 rounding)
extern "C" __global__ void scale_bf16(float *x, int n, float s) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = bf16r(x[i] * s);
}

// ---------------------------------------------------------------- routed-expert GEMV (VRAM-resident experts)
// Same calling convention as gemv_q8_0: one warp per row, T <= 8 tokens, x f32 [T][ldx].

// Q2_0: 64 weights / 18 B: fp16 d, qs[16]; byte m holds weights 4m..4m+3, value (code - 1) * d
extern "C" __global__ void gemv_q2_0(const uint8_t *__restrict__ w, const float *__restrict__ x, float *__restrict__ y,
                                     int rows, int k, int T, int ldx, int ldy) {
  int r = blockIdx.x * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (r >= rows) return;
  int nb = k / 64;
  const uint8_t *wr = w + (size_t)r * nb * 18;
  float acc[MAXT] = {0};
  for (int b = lane; b < nb; b += 32) {
    const uint8_t *blk = wr + b * 18;
    float d = ld_h(blk);
    float wv[64];
#pragma unroll
    for (int m = 0; m < 16; m++) {
      uint8_t q = blk[2 + m];
#pragma unroll
      for (int j = 0; j < 4; j++) wv[4 * m + j] = (float)(((q >> (2 * j)) & 3) - 1);
    }
    for (int t = 0; t < T; t++) {
      const float4 *xp = (const float4 *)(x + (size_t)t * ldx + b * 64);
      float s = 0.f;
#pragma unroll
      for (int j = 0; j < 16; j++) {
        float4 v = xp[j];
        s += wv[4 * j] * v.x + wv[4 * j + 1] * v.y + wv[4 * j + 2] * v.z + wv[4 * j + 3] * v.w;
      }
      acc[t] += d * s;
    }
  }
  for (int t = 0; t < T; t++) {
    float s = warp_sum(acc[t]);
    if (lane == 0) y[(size_t)t * ldy + r] = s;
  }
}

__device__ __forceinline__ float trit(uint32_t q, uint32_t p3) { return (float)((((q * p3) & 0xFF) * 3) >> 8) - 1.f; }

// PTQ1_0: 128 weights / 28 B (see formats.rs)
extern "C" __global__ void gemv_ptq1_0(const uint8_t *__restrict__ w, const float *__restrict__ x, float *__restrict__ y,
                                       int rows, int k, int T, int ldx, int ldy) {
  int r = blockIdx.x * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (r >= rows) return;
  int nb = k / 128;
  const uint8_t *wr = w + (size_t)r * nb * 28;
  const uint32_t p3[5] = {1, 3, 9, 27, 81};
  float acc[MAXT] = {0};
  for (int b = lane; b < nb; b += 32) {
    const uint8_t *blk = wr + b * 28;
    float d = ld_h(blk + 26);
    float wv[128];
#pragma unroll
    for (int m = 0; m < 16; m++)
#pragma unroll
      for (int n = 0; n < 5; n++) wv[m + 16 * n] = trit(blk[m], p3[n]);
#pragma unroll
    for (int m = 0; m < 8; m++)
#pragma unroll
      for (int n = 0; n < 5; n++) wv[80 + m + 8 * n] = trit(blk[16 + m], p3[n]);
#pragma unroll
    for (int h = 0; h < 2; h++)
#pragma unroll
      for (int n = 0; n < 4; n++) wv[120 + h + 2 * n] = trit(blk[24 + h], p3[n]);
    for (int t = 0; t < T; t++) {
      const float4 *xp = (const float4 *)(x + (size_t)t * ldx + b * 128);
      float s = 0.f;
#pragma unroll
      for (int j = 0; j < 32; j++) {
        float4 v = xp[j];
        s += wv[4 * j] * v.x + wv[4 * j + 1] * v.y + wv[4 * j + 2] * v.z + wv[4 * j + 3] * v.w;
      }
      acc[t] += d * s;
    }
  }
  for (int t = 0; t < T; t++) {
    float s = warp_sum(acc[t]);
    if (lane == 0) y[(size_t)t * ldy + r] = s;
  }
}

// Q3_K: ggml block_q3_K, 256 weights / 110 B: hmask[32] qs[64] scales[12] d(fp16). Each lane takes a 16-weight
// sub-block (16 per block), so a warp covers 2 blocks per step.
extern "C" __global__ void gemv_q3_k(const uint8_t *__restrict__ w, const float *__restrict__ x, float *__restrict__ y,
                                     int rows, int k, int T, int ldx, int ldy) {
  int r = blockIdx.x * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (r >= rows) return;
  int nb = k / 256;
  const uint8_t *wr = w + (size_t)r * nb * 110;
  float acc[MAXT] = {0};
  for (int sb = lane; sb < nb * 16; sb += 32) {
    int b = sb >> 4, s16 = sb & 15;
    const uint8_t *blk = wr + b * 110;
    const uint8_t *hm = blk, *qs = blk + 32, *raw = blk + 96;
    float d = ld_h(blk + 108);
    int i4 = s16 & 3, grp = s16 >> 2;  // scale index s16: s[0..4], s[4..8], s[8..12], s[12..16]
    int a = raw[i4], c = raw[4 + i4], tmp = raw[8 + i4];
    int sc;
    if (grp == 0) sc = (a & 15) | ((tmp & 3) << 4);
    else if (grp == 1) sc = (c & 15) | (((tmp >> 2) & 3) << 4);
    else if (grp == 2) sc = (a >> 4) | (((tmp >> 4) & 3) << 4);
    else sc = (c >> 4) | (((tmp >> 6) & 3) << 4);
    float dl = d * (float)(sc - 32);
    // element e = 16*s16 + i, e = 128h + 32j + l
    float wv[16];
#pragma unroll
    for (int i = 0; i < 16; i++) {
      int e = 16 * s16 + i;
      int h = e >> 7, j = (e >> 5) & 3, l = e & 31;
      int low = (qs[32 * h + l] >> (2 * j)) & 3;
      int hb = (hm[l] >> (4 * h + j)) & 1;
      wv[i] = (float)(low - (hb ? 0 : 4));
    }
    for (int t = 0; t < T; t++) {
      const float4 *xp = (const float4 *)(x + (size_t)t * ldx + b * 256 + 16 * s16);
      float s = 0.f;
#pragma unroll
      for (int j = 0; j < 4; j++) {
        float4 v = xp[j];
        s += wv[4 * j] * v.x + wv[4 * j + 1] * v.y + wv[4 * j + 2] * v.z + wv[4 * j + 3] * v.w;
      }
      acc[t] += dl * s;
    }
  }
  for (int t = 0; t < T; t++) {
    float s = warp_sum(acc[t]);
    if (lane == 0) y[(size_t)t * ldy + r] = s;
  }
}

// FP8 E4M3 weights with an E8M0 scale per 32x32 block (DeepSeek checkpoint). scale: [rows/32][k/32]
__device__ __forceinline__ float e4m3_to_f32(uint8_t b) {
  int e = (b >> 3) & 15, m = b & 7;
  float v = e ? ldexpf(1.f + m * 0.125f, e - 7) : ldexpf((float)m, -9);
  return (b & 0x80) ? -v : v;
}
// gemv_f16's exact arithmetic on FP8 + 32x32 E8M0 weights: each e4m3 x 2^s is exactly the f16 value it came from
// (numerics::f16_to_fp8b32), and lane / column order, the 8-product expression and the warp sum are gemv_f16's, so
// the output is bit-identical to gemv_f16 on the F16 copy at half the bytes. Two iterations' codes are loaded ahead.
extern "C" __global__ void gemv_fp8b32_o(const uint8_t *__restrict__ w, const float *__restrict__ x, float *__restrict__ y,
                                         int rows, int k, int T, int ldx, int ldy, const uint8_t *__restrict__ scale) {
  int r = blockIdx.x * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (r >= rows) return;
  const uint8_t *wr = w + (size_t)r * k;
  const uint8_t *sr = scale + (size_t)(r / 32) * (k / 32);
  float acc[MAXT] = {0};
  for (int i = lane * 8; i < k; i += 256) {
    uint2 raw = *(const uint2 *)(wr + i);
    // e4m3 bits shifted into f16 are the value times 2^-8; times (2^s * 2^8) it is exact
    const float d = __uint_as_float(((uint32_t)sr[i / 32] + 8) << 23);
    const uint32_t u[2] = {raw.x, raw.y};
    float wf[8];
#pragma unroll
    for (int j = 0; j < 2; j++) {
#pragma unroll
      for (int h = 0; h < 2; h++) {
        uint32_t xx = __byte_perm(u[j], 0, h ? 0x4342 : 0x4140);
        uint32_t f = ((xx & 0x00800080u) << 8) | ((xx & 0x007F007Fu) << 7);
        float2 v = __half22float2(*(const __half2 *)&f);
        wf[4 * j + 2 * h] = v.x * d;
        wf[4 * j + 2 * h + 1] = v.y * d;
      }
    }
    for (int t = 0; t < T; t++) {
      const float4 *xp = (const float4 *)(x + (size_t)t * ldx + i);
      float4 a = xp[0], b = xp[1];
      acc[t] += wf[0] * a.x + wf[1] * a.y + wf[2] * a.z + wf[3] * a.w + wf[4] * b.x + wf[5] * b.y + wf[6] * b.z + wf[7] * b.w;
    }
  }
  for (int t = 0; t < T; t++) {
    float s = warp_sum(acc[t]);
    if (lane == 0) y[(size_t)t * ldy + r] = s;
  }
}

// 4 rows per warp (16 per block of 128) sharing each x load; lane l takes 16 columns at l*16 + 512*i (k % 32 == 0).
// e4m3 bits shifted into f16 (sign to bit 15, exponent + mantissa to bits 7..14) are the value times 2^-8,
// subnormals included; the 2^8 goes into the block scale, applied once per 16-column partial sum. The token loops run
// to MAXT with a t < T guard so acc stays in registers (a runtime-indexed acc went to local memory on sm_86).
#define FP8_ROWS 4
extern "C" __global__ void gemv_fp8b32(const uint8_t *__restrict__ w, const float *__restrict__ x, float *__restrict__ y,
                                       int rows, int k, int T, int ldx, int ldy, const uint8_t *__restrict__ scale) {
  int r0 = (blockIdx.x * 4 + (threadIdx.x >> 5)) * FP8_ROWS, lane = threadIdx.x & 31;
  if (r0 >= rows) return;
  int nb = k / 32;
  float acc[FP8_ROWS][MAXT];
#pragma unroll
  for (int q = 0; q < FP8_ROWS; q++)
#pragma unroll
    for (int t = 0; t < MAXT; t++) acc[q][t] = 0.f;
  for (int i = lane * 16; i < k; i += 512) {
    float wv[FP8_ROWS][16], d[FP8_ROWS];
#pragma unroll
    for (int q = 0; q < FP8_ROWS; q++) {
      int r = min(r0 + q, rows - 1);
      uint4 a = *(const uint4 *)(w + (size_t)r * k + i);
      d[q] = __uint_as_float(((uint32_t)scale[(size_t)(r / 32) * nb + i / 32] + 8) << 23);
      const uint32_t *u = (const uint32_t *)&a;
#pragma unroll
      for (int j = 0; j < 4; j++) {
#pragma unroll
        for (int h = 0; h < 2; h++) {
          uint32_t xx = __byte_perm(u[j], 0, h ? 0x4342 : 0x4140);
          uint32_t f = ((xx & 0x00800080u) << 8) | ((xx & 0x007F007Fu) << 7);
          float2 v = __half22float2(*(const __half2 *)&f);
          wv[q][4 * j + 2 * h] = v.x;
          wv[q][4 * j + 2 * h + 1] = v.y;
        }
      }
    }
#pragma unroll
    for (int t = 0; t < MAXT; t++) {
      if (t < T) {
        const float4 *xp = (const float4 *)(x + (size_t)t * ldx + i);
        float4 xv[4] = {xp[0], xp[1], xp[2], xp[3]};
        const float *xf = (const float *)xv;
#pragma unroll
        for (int q = 0; q < FP8_ROWS; q++) {
          float s = 0.f;
#pragma unroll
          for (int j = 0; j < 16; j++) s += wv[q][j] * xf[j];
          acc[q][t] += d[q] * s;
        }
      }
    }
  }
#pragma unroll
  for (int q = 0; q < FP8_ROWS; q++)
#pragma unroll
    for (int t = 0; t < MAXT; t++) {
      if (t < T) {
        float s = warp_sum(acc[q][t]);
        if (lane == 0 && r0 + q < rows) y[(size_t)t * ldy + r0 + q] = s;
      }
    }
}

extern "C" __global__ void gemv_bf16(const __nv_bfloat16 *__restrict__ w, const float *__restrict__ x, float *__restrict__ y,
                                     int rows, int k, int T, int ldx, int ldy) {
  int r = blockIdx.x * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (r >= rows) return;
  const __nv_bfloat16 *wr = w + (size_t)r * k;
  float acc[MAXT] = {0};
  for (int i = lane * 8; i < k; i += 256) {
    uint4 raw = *(const uint4 *)(wr + i);
    const __nv_bfloat16 *h = (const __nv_bfloat16 *)&raw;
    float wf[8];
#pragma unroll
    for (int j = 0; j < 8; j++) wf[j] = __bfloat162float(h[j]);
    for (int t = 0; t < T; t++) {
      const float4 *xp = (const float4 *)(x + (size_t)t * ldx + i);
      float4 a = xp[0], b = xp[1];
      acc[t] += wf[0] * a.x + wf[1] * a.y + wf[2] * a.z + wf[3] * a.w + wf[4] * b.x + wf[5] * b.y + wf[6] * b.z + wf[7] * b.w;
    }
  }
  for (int t = 0; t < T; t++) {
    float s = warp_sum(acc[t]);
    if (lane == 0) y[(size_t)t * ldy + r] = s;
  }
}

// dst = fake_fp8(src) (n multiple of 32), for FP8-checkpoint linears whose input DeepSeek quantizes first
extern "C" __global__ void fake_fp8_copy(const float *src, float *dst, int n) {
  int b = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (b * 32 >= n) return;
  float v = src[b * 32 + lane];
  float amax = fmaxf(warp_max(fabsf(v)), 1e-4f);
  float s = pow2_ceil(amax / 448.f);
  dst[b * 32 + lane] = e4m3_round(fminf(fmaxf(v / s, -448.f), 448.f)) * s;
}

// zero-copy handshake: publish `v` at `flag` (mapped host memory) after this kernel's prior writes are visible
extern "C" __global__ void signal_host(volatile uint32_t *flag, uint32_t v) {
  __threadfence_system();
  *flag = v;
}

#include "topk.cuh"

// h = bf16(silu(min(g,lim)) * clamp(u,-lim,lim) * w), one token, scalar routing weight
extern "C" __global__ void swiglu_s(const float *g, const float *u, float *h, int inter, float lim, float w) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= inter) return;
  float gv = fminf(g[i], lim), uv = fminf(fmaxf(u[i], -lim), lim);
  h[i] = bf16r(gv / (1.f + expf(-gv)) * uv * w);
}
#include "moe.cuh"
#include "expert_fast.cuh"
#include "dequant_iq8.cuh"
#include "t16.cuh"

// ---------------------------------------------------------------- cross-device expert handoff (mapped host memory)
// xch layout (bytes): route_flag @0, cpu_flag @64, gpu1_flag @128, ntok @192, sel[8][8] i32 @256, wts[8][8] f32 @512,
// x[T][dim] f32 @1024, y_cpu[T][dim] @1024 + 32*dim, y_aux[i][T][dim] @1024 + (64 + 32i)*dim (room for T <= 8);
// aux flags: 128 for aux 0, 768 + 64 (i - 1) for the rest
#define XCH_T 192
#define XCH_SEL 256
#define XCH_WTS 512
#define XCH_X 1024

// copy this layer's picks and expert inputs (T tokens, k picks each) to mapped host memory, then route_flag = seq.
// grid 1, block 1024
extern "C" __global__ void publish_route(const int *sel, const float *wts, int k, const float *x, int dim, uint8_t *xch,
                                         uint32_t seq, int T) {
  int *hs = (int *)(xch + XCH_SEL);
  float *hw = (float *)(xch + XCH_WTS), *hx = (float *)(xch + XCH_X);
  for (int i = threadIdx.x; i < T * dim; i += blockDim.x) hx[i] = x[i];
  if (threadIdx.x < T * k) {
    hs[threadIdx.x] = sel[threadIdx.x];
    hw[threadIdx.x] = wts[threadIdx.x];
  }
  if (threadIdx.x == 0) *(volatile int *)(xch + XCH_T) = T;
  __syncthreads();
  if (threadIdx.x == 0) {
    unsigned long long now;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
    *(volatile unsigned long long *)(xch + 8) = now;  // handoff timing (DSX_HANDOFF_TS readers)
    __threadfence_system();
    *(volatile uint32_t *)xch = seq;
  }
}

// decode (one token): router_topk + publish_route in one launch, bit-identical to the two. Block 1024: thread e scores
// expert e (in parallel instead of 12 per lane), warp 0 then runs router_topk's selection on the same keys, and every
// thread publishes as publish_route does. E <= 512.
extern "C" __global__ void __launch_bounds__(1024) router_publish(const float *logits, const float *bias, int *out_idx,
                                                                  float *out_w, int E, int k, float route_scale,
                                                                  const float *sel_bonus, const float *x, int dim,
                                                                  uint8_t *xch, uint32_t seq) {
  __shared__ float sc[512], ks[512];
  __shared__ int ssel[8];
  __shared__ float sw[8];
  int tid = threadIdx.x, lane = tid & 31;
  if (tid < E) {
    float v = logits[tid];
    float sp = v > 20.f ? v : log1pf(expf(v));
    float s = sqrtf(sp);
    float kk = s + bias[tid] + (sel_bonus ? sel_bonus[tid] : 0.f);
    ks[tid] = isnan(kk) ? -INFINITY : kk;
    sc[tid] = isnan(s) ? 0.f : s;
  }
  __syncthreads();
  if (tid < 32) {
    float key[16];
#pragma unroll
    for (int j = 0; j < 16; j++) {
      int e = lane + 32 * j;
      key[j] = e < E ? ks[e] : -INFINITY;
    }
    float wsum = 0.f;
    int sel[8];
    for (int r = 0; r < k; r++) {
      float best = -INFINITY;
      int bi = 0x7fffffff;
#pragma unroll
      for (int j = 0; j < 16; j++)
        if (key[j] > best) { best = key[j]; bi = lane + 32 * j; }
      for (int o = 16; o; o >>= 1) {
        float ob = __shfl_xor_sync(0xffffffff, best, o);
        int oi = __shfl_xor_sync(0xffffffff, bi, o);
        if (ob > best || (ob == best && oi < bi)) { best = ob; bi = oi; }
      }
      if (bi >= E) bi = r;
      sel[r] = bi;
      wsum += sc[bi];
#pragma unroll
      for (int j = 0; j < 16; j++)
        if (lane + 32 * j == bi) key[j] = -INFINITY;
    }
    if (lane < k) {
      float w = sc[sel[lane]] / (wsum + 1e-20f) * route_scale;
      out_idx[lane] = sel[lane];
      out_w[lane] = w;
      ssel[lane] = sel[lane];
      sw[lane] = w;
    }
  }
  // publish_route for T = 1
  int *hs = (int *)(xch + XCH_SEL);
  float *hw = (float *)(xch + XCH_WTS), *hx = (float *)(xch + XCH_X);
  for (int i = tid; i < dim; i += blockDim.x) hx[i] = x[i];
  __syncthreads();
  if (tid < k) {
    hs[tid] = ssel[tid];
    hw[tid] = sw[tid];
  }
  if (tid == 0) *(volatile int *)(xch + XCH_T) = 1;
  __syncthreads();
  if (tid == 0) {
    unsigned long long now;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
    *(volatile unsigned long long *)(xch + 8) = now;  // handoff timing (DSX_HANDOFF_TS readers)
    __threadfence_system();
    *(volatile uint32_t *)xch = seq;
  }
}

// test helper: keep one thread busy for `ns` nanoseconds of GPU time
extern "C" __global__ void gpu_sleep(uint32_t ns) {
  unsigned long long t0, t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
  do { asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); } while (t - t0 < ns);
}

// one thread spins on a u32 in mapped host memory until it is >= seq (DSX_SPIN_WAIT: instead of cuStreamWaitValue32)
// diag (optional): [2] u64 += (globaltimer when seen - the CPU's store timestamp at flag + 8), count
extern "C" __global__ void spin_wait(const volatile uint32_t *flag, uint32_t seq, unsigned long long *diag) {
  while (*flag < seq) __nanosleep(200);
  __threadfence_system();
  if (diag) {
    unsigned long long now;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
    diag[0] += now - *(const volatile unsigned long long *)((const volatile uint8_t *)flag + 8);
    diag[1] += 1;
  }
}

// a[t] = bf16(a[t] + out0[t] + the CPU's slots in cpu_mask + sum over aux GPUs of y_aux[i][t]) for T tokens;
// grid ceil(T*dim/256). CPU slots: bit 0 = XCH_X + 32 dim (y_cpu), bit 1 = cpu2_off, bit 2 = cpu3_off (DSX_DEFER's
// deferred slots for even / odd layers); aux i's slot: XCH_X + (64 + 32 i) dim (handoff::aux_y)
extern "C" __global__ void combine_experts(float *a, const float *out0, const uint8_t *xch, int dim, int cpu_mask, int n_aux, int T, int cpu2_off,
                                           int cpu3_off) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= T * dim) return;
  float v = a[i] + out0[i];
  if (cpu_mask & 1) v += ((const volatile float *)(xch + XCH_X + 32 * dim))[i];
  if (cpu_mask & 2) v += ((const volatile float *)(xch + cpu2_off))[i];
  if (cpu_mask & 4) v += ((const volatile float *)(xch + cpu3_off))[i];
  for (int j = 0; j < n_aux; j++) v += ((const volatile float *)(xch + XCH_X + (64 + 32 * j) * dim))[i];
  a[i] = bf16r(v);
}

// decode with DSX_DEFER (T = 1): combine_experts, plus the previous layer's deferred CPU experts entering the residual
// (hc_add_post: h[k] = bf16(h[k] + post_prev[k] * y)) when inj_off >= 0, plus this layer's output weights saved for its
// own deferred experts (post_cur[0..4] = mixes[4..8], the 16-byte copy) when post_cur is set, plus the layer's closing
// hc_post with `mixes` when do_post. One launch instead of four, bit-identical: the same per-element expressions.
// grid ceil(dim / 256), block 256.
extern "C" __global__ void combine_post(float *a, const float *out0, const uint8_t *xch, int dim, int cpu_mask, int n_aux,
                                        int cpu2_off, int cpu3_off, float *h, int inj_off, const float *post_prev,
                                        float *post_cur, const float *mixes, int do_post) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (post_cur && i < 4) post_cur[i] = mixes[4 + i];
  if (i >= dim) return;
  float v = a[i] + out0[i];
  if (cpu_mask & 1) v += ((const volatile float *)(xch + XCH_X + 32 * dim))[i];
  if (cpu_mask & 2) v += ((const volatile float *)(xch + cpu2_off))[i];
  if (cpu_mask & 4) v += ((const volatile float *)(xch + cpu3_off))[i];
  for (int j = 0; j < n_aux; j++) v += ((const volatile float *)(xch + XCH_X + (64 + 32 * j) * dim))[i];
  if (inj_off >= 0) {
    float y = ((const float *)(xch + inj_off))[i];
#pragma unroll
    for (int k = 0; k < 4; k++) h[k * dim + i] = bf16r(h[k * dim + i] + post_prev[k] * y);
  }
  float av = bf16r(v);
  a[i] = av;
  // the end-of-layer hc_post (hc_post_mb with mixes) for this element
  if (do_post) hc_post_elem(h, i, dim, mixes, av);
}

// ---------------------------------------------------------------- prefill: weights -> f16 for cuBLAS
// out[r][c] (f16) for rows [0, rows) of a weight in type t (ggml id, or 1000 = FP8 + 32x32 E8M0 scales, scale[rows/32][k/32]).
// grid (ceil(k/256), rows), block 256: each thread one element.
extern "C" __global__ void dequant_f16(int t, const uint8_t *w, const uint8_t *scale, __half *out, int rows, int k) {
  int r = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= k) return;
  float v;
  if (t == 8) {  // Q8_0
    const uint8_t *blk = w + (size_t)r * (k / 32) * 34 + (c / 32) * 34;
    v = ld_h(blk) * (float)(int8_t)blk[2 + c % 32];
  } else if (t == 42) {  // Q2_0
    const uint8_t *blk = w + (size_t)r * (k / 64) * 18 + (c / 64) * 18;
    int i = c % 64;
    v = ld_h(blk) * (float)(((blk[2 + i / 4] >> (2 * (i % 4))) & 3) - 1);
  } else if (t == 10 || t == 18) {
    const uint8_t *p = w + ((size_t)r*(k/256)+c/256)*(t == 10 ? 84 : 98);
    v = t == 10 ? q2k_value(p,c&255) : iq3xxs_value(p,c&255);
  } else if (t == 16) {  // IQ2_XXS
    const uint8_t *blk = w + (size_t)r * (k / 256) * 66 + (c / 256) * 66;
    v = iq2_value(blk + 2 + ((c & 255) >> 5) * 8, ld_h(blk), c & 31);
  } else if (t == 19) {  // IQ1_S
    v = iq1s_value(w + (size_t)r * (k / 256) * 50 + (c / 256) * 50, c & 255);
  } else if (t == 17) {  // IQ2_XS
    v = iq2xs_value(w + (size_t)r * (k / 256) * 74 + (c / 256) * 74, c & 255);
  } else if (t == 22) {  // IQ2_S
    v = iq2s_value(w + (size_t)r * (k / 256) * 82 + (c / 256) * 82, c & 255);
  } else if (t == 143) {  // PTQ1_0
    const uint8_t *blk = w + (size_t)r * (k / 128) * 28 + (c / 128) * 28;
    int i = c % 128, byte, n;
    if (i < 80) { byte = i % 16; n = i / 16; }
    else if (i < 120) { byte = 16 + (i - 80) % 8; n = (i - 80) / 8; }
    else { byte = 24 + (i - 120) % 2; n = (i - 120) / 2; }
    const uint32_t p3[5] = {1, 3, 9, 27, 81};
    v = ld_h(blk + 26) * trit(blk[byte], p3[n]);
  } else if (t == 11) {  // Q3_K
    const uint8_t *blk = w + (size_t)r * (k / 256) * 110 + (c / 256) * 110;
    int e = c % 256, h = e >> 7, j = (e >> 5) & 3, l = e & 31, s16 = e / 16;
    const uint8_t *hm = blk, *qs = blk + 32, *raw = blk + 96;
    int i4 = s16 & 3, grp = s16 >> 2, a = raw[i4], cc = raw[4 + i4], tmp = raw[8 + i4], sc;
    if (grp == 0) sc = (a & 15) | ((tmp & 3) << 4);
    else if (grp == 1) sc = (cc & 15) | (((tmp >> 2) & 3) << 4);
    else if (grp == 2) sc = (a >> 4) | (((tmp >> 4) & 3) << 4);
    else sc = (cc >> 4) | (((tmp >> 6) & 3) << 4);
    int low = (qs[32 * h + l] >> (2 * j)) & 3, hb = (hm[l] >> (4 * h + j)) & 1;
    v = ld_h(blk + 108) * (float)(sc - 32) * (float)(low - (hb ? 0 : 4));
  } else if (t >= 2000) {  // routed experts repacked for the fast decode path (expert_fast.cuh)
    v = rp_weight(t, w, rows, k, r, c);
  } else if (t == 1008) {  // Q8_0 repacked: w = qs int8 [rows][k], scale = d f16 [rows][k/32]
    v = (float)((const int8_t *)w)[(size_t)r * k + c] * __half2float(((const __half *)scale)[(size_t)r * (k / 32) + c / 32]);
  } else if (t == 1000) {  // FP8 checkpoint
    v = e4m3_to_f32(w[(size_t)r * k + c]) * __uint_as_float((uint32_t)scale[(size_t)(r / 32) * (k / 32) + c / 32] << 23);
  } else if (t == 1) {
    v = __half2float(((const __half *)w)[(size_t)r * k + c]);
  } else if (t == 30) {
    v = __bfloat162float(((const __nv_bfloat16 *)w)[(size_t)r * k + c]);
  } else {
    v = ((const float *)w)[(size_t)r * k + c];
  }
  out[(size_t)r * k + c] = __float2half(v);
}

// f32 -> f16 (activations into a GEMM), n elements
extern "C" __global__ void to_f16(const float *x, __half *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = __float2half(x[i]);
}

// ---------------------------------------------------------------- prefill helpers
// RoPE with an explicit position per row
extern "C" __global__ void rope_pos(float *x, const float *freqs, int heads, int hd, int rd, int ld, const int *pos, int inverse) {
  int t = blockIdx.x;
  float p = (float)pos[t];
  int half = rd / 2;
  for (int i = threadIdx.x; i < heads * half; i += blockDim.x) {
    int h = i / half, q = i % half;
    float s, c;
    sincosf(p * freqs[q], &s, &c);
    if (inverse) s = -s;
    float *v = x + (size_t)t * ld + h * hd + hd - rd + 2 * q;
    float a = v[0], b = v[1];
    v[0] = a * c - b * s;
    v[1] = a * s + b * c;
  }
}

// ratio-2 compressor over a chunk: group g pools rows (a, b) = (2g - off, 2g + 1 - off) of kv/sc [T][hd];
// row -1 (off = 1, first group) comes from the carried state slot 0. out[g][hd] = bf16(softmax-pooled kv). grid G, block 512
extern "C" __global__ void compress_pairs(const float *kv, const float *sc, const float *st_kv, const float *st_sc, int off,
                                          float *out, int hd) {
  int g = blockIdx.x;
  int ra = 2 * g - off, rb = ra + 1;
  for (int j = threadIdx.x; j < hd; j += blockDim.x) {
    float ka = ra < 0 ? st_kv[j] : kv[(size_t)ra * hd + j], sa = ra < 0 ? st_sc[j] : sc[(size_t)ra * hd + j];
    float kb = kv[(size_t)rb * hd + j], sb = sc[(size_t)rb * hd + j];
    float mx = fmaxf(sa, sb), ea = expf(sa - mx), eb = expf(sb - mx);
    out[(size_t)g * hd + j] = bf16r((ka * ea + kb * eb) / (ea + eb));
  }
}

// gather rows (f32 [*, k]) by index into f16 [n][k]; grid (ceil(k/256), n)
extern "C" __global__ void gather_f16(const float *x, const int *rows, __half *y, int k) {
  int r = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < k) y[(size_t)r * k + c] = __float2half(x[(size_t)rows[r] * k + c]);
}

// h (f16) = bf16(silu(min(g,lim)) * clamp(u,-lim,lim) * w[row]); g/u f32 [n][inter]; grid (ceil(inter/256), n)
extern "C" __global__ void swiglu_rows_f16(const float *g, const float *u, const float *w, __half *h, int inter, float lim) {
  int r = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= inter) return;
  float gv = fminf(g[(size_t)r * inter + i], lim), uv = fminf(fmaxf(u[(size_t)r * inter + i], -lim), lim);
  float wt = w ? w[r] : 1.f;
  h[(size_t)r * inter + i] = __float2half(bf16r(gv / (1.f + expf(-gv)) * uv * wt));
}

// out[rows[r]] += o[r] (rows distinct within one call); grid (ceil(k/256), n)
// DSX_NOISE sensitivity probe: o[r] += rel * rms(o[r]) * N(0, 1) per row (hash-seeded Box-Muller); grid rows, block 256
extern "C" __global__ void add_noise_rows(float *o, int k, float rel, unsigned seed) {
  __shared__ float sh[32];
  float *x = o + (size_t)blockIdx.x * k;
  float s = 0.f;
  for (int i = threadIdx.x; i < k; i += blockDim.x) s += x[i] * x[i];
  s = block_sum(s, sh);
  float a = rel * sqrtf(s / k);
  for (int i = threadIdx.x; i < k; i += blockDim.x) {
    unsigned h = seed * 0x9E3779B9u ^ (blockIdx.x * 0x85EBCA6Bu) ^ (i * 0xC2B2AE35u);
    h ^= h >> 16; h *= 0x7FEB352Du; h ^= h >> 15; h *= 0x846CA68Bu; h ^= h >> 16;
    unsigned h2 = h * 0x9E3779B9u + 0x6A09E667u;
    h2 ^= h2 >> 16; h2 *= 0x7FEB352Du; h2 ^= h2 >> 15;
    float u1 = ((h >> 8) + 1) * (1.f / 16777217.f), u2 = (h2 >> 8) * (1.f / 16777216.f);
    x[i] += a * sqrtf(-2.f * logf(u1)) * cospif(2.f * u2);
  }
}

extern "C" __global__ void scatter_add(const float *o, const int *rows, float *out, int k) {
  int r = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < k) out[(size_t)rows[r] * k + c] += o[(size_t)r * k + c];
}

// engram gate for T tokens: h [T][hc][d], kv [T][(hc+1)d]; grid (hc, T), block 1024
extern "C" __global__ void engram_gate_t(float *h, const float *kv, const float *qk, int d, float eps) {
  __shared__ float sh[32];
  int k = blockIdx.x, t = blockIdx.y;
  float *hk = h + ((size_t)t * 4 + k) * d;
  const float *kt = kv + (size_t)t * 5 * d;
  const float *key = kt + (size_t)k * d, *val = kt + (size_t)4 * d;
  float sh2 = 0.f, sk2 = 0.f, sd = 0.f;
  for (int i = threadIdx.x; i < d; i += blockDim.x) {
    sh2 += hk[i] * hk[i];
    sk2 += key[i] * key[i];
    sd += hk[i] * qk[k * d + i] * key[i];
  }
  sh2 = block_sum(sh2, sh);
  sk2 = block_sum(sk2, sh);
  sd = block_sum(sd, sh);
  float dt = sd * rsqrtf(sh2 / d + eps) * rsqrtf(sk2 / d + eps) * rsqrtf((float)d);
  float gt = 1.f / (1.f + expf(-copysignf(sqrtf(fmaxf(fabsf(dt), 1e-6f)), dt)));
  for (int i = threadIdx.x; i < d; i += blockDim.x) hk[i] = bf16r(hk[i] + gt * val[i]);
}

// y = bf16(y) elementwise over n, plus add (y += x) variants are add_inplace / round_bf16

// decode ring <- the last W rows of a chunk: ring[(p0 + t) % W] = src[t] for t >= T - W; grid T, block 128 (hd bf16)
extern "C" __global__ void ring_scatter(const __nv_bfloat16 *src, __nv_bfloat16 *ring, int p0, int T, int W, int hd) {
  int t = blockIdx.x;
  if (t < T - W) return;
  const __nv_bfloat16 *s = src + (size_t)t * hd;
  __nv_bfloat16 *d = ring + (size_t)((p0 + t) % W) * hd;
  for (int i = threadIdx.x; i < hd; i += blockDim.x) d[i] = s[i];
}

// strided f32 columns -> packed f16: y[r][c] = x[r * ldx + c], c < k; grid (ceil(k/256), n)
extern "C" __global__ void pack_f16(const float *x, int ldx, __half *y, int k) {
  int r = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < k) y[(size_t)r * k + c] = __float2half(x[(size_t)r * ldx + c]);
}

// ---------------------------------------------------------------- Q8_0 repacked for coalesced loads
// Layout (built at upload): qs int8 [rows][k] contiguous, then d f16 [rows][k/32]. A lane loads 16 int8 at a time.
// One warp per row, 4 rows per block, T <= 8 tokens.
extern "C" __global__ void gemv_q8r(const int8_t *__restrict__ qs, const float *__restrict__ x, float *__restrict__ y,
                                    int rows, int k, int T, int ldx, int ldy, const __half *__restrict__ d) {
  int r = blockIdx.x * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (r >= rows) return;
  const int8_t *qr = qs + (size_t)r * k;
  const __half *dr = d + (size_t)r * (k / 32);
  float acc[MAXT] = {0};
  for (int i = lane * 16; i < k; i += 512) {
    int4 raw = *(const int4 *)(qr + i);
    const int8_t *q = (const int8_t *)&raw;
    float sc = __half2float(dr[i / 32]);
    float w[16];
#pragma unroll
    for (int j = 0; j < 16; j++) w[j] = (float)q[j] * sc;
    for (int t = 0; t < T; t++) {
      const float4 *xp = (const float4 *)(x + (size_t)t * ldx + i);
      float4 a = xp[0], b = xp[1], c = xp[2], e = xp[3];
      acc[t] += w[0] * a.x + w[1] * a.y + w[2] * a.z + w[3] * a.w + w[4] * b.x + w[5] * b.y + w[6] * b.z + w[7] * b.w +
                w[8] * c.x + w[9] * c.y + w[10] * c.z + w[11] * c.w + w[12] * e.x + w[13] * e.y + w[14] * e.z + w[15] * e.w;
    }
  }
  for (int t = 0; t < T; t++) {
    float s = warp_sum(acc[t]);
    if (lane == 0) y[(size_t)t * ldy + r] = s;
  }
}

// ---------------------------------------------------------------- hc mixes split across blocks
// dots[t][i] = fn[i] . h[t] * rsqrt(mean(h[t]^2) + eps); grid (24, T), block 256
extern "C" __global__ void hc_dots(const float *fn, const float *h, float *dots, int hcd, float eps) {
  __shared__ float sh[32];
  int i = blockIdx.x, t = blockIdx.y;
  const float *x = h + (size_t)t * hcd, *row = fn + (size_t)i * hcd;
  float ss = 0.f, acc = 0.f;
  for (int j = threadIdx.x * 4; j < hcd; j += blockDim.x * 4) {
    float4 xv = *(const float4 *)(x + j), wv = *(const float4 *)(row + j);
    ss += xv.x * xv.x + xv.y * xv.y + xv.z * xv.z + xv.w * xv.w;
    acc += wv.x * xv.x + wv.y * xv.y + wv.z * xv.z + wv.w * xv.w;
  }
  ss = block_sum(ss, sh);
  acc = block_sum(acc, sh);
  if (threadIdx.x == 0) dots[t * 24 + i] = acc * rsqrtf(ss / hcd + eps);
}

// hc_dots over a tile of 4 rows x 4 tokens per block: each h[t] / fn[i] slice is loaded once for the tile (one block
// per (row, token) re-read each token's 80 KB h 24 times: 3.5 ms per 3000-token call on an RTX 3090). Every
// (row, token) keeps hc_dots' per-thread accumulation and block_sum, so the dots are bit-identical.
// grid (24 / 4, ceil(T / 4)), block 256
extern "C" __global__ void hc_dots_t(const float *fn, const float *h, float *dots, int hcd, float eps, int T) {
  __shared__ float sh[32];
  const int i0 = blockIdx.x * 4, t0 = blockIdx.y * 4;
  const int nt = min(4, T - t0);
  float ss[4] = {0.f, 0.f, 0.f, 0.f}, acc[4][4];
#pragma unroll
  for (int r = 0; r < 4; r++)
#pragma unroll
    for (int q = 0; q < 4; q++) acc[r][q] = 0.f;
  for (int j = threadIdx.x * 4; j < hcd; j += blockDim.x * 4) {
    float4 xv[4], wv[4];
#pragma unroll
    for (int q = 0; q < 4; q++) xv[q] = q < nt ? *(const float4 *)(h + (size_t)(t0 + q) * hcd + j) : make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int r = 0; r < 4; r++) wv[r] = *(const float4 *)(fn + (size_t)(i0 + r) * hcd + j);
#pragma unroll
    for (int q = 0; q < 4; q++) {
      ss[q] += xv[q].x * xv[q].x + xv[q].y * xv[q].y + xv[q].z * xv[q].z + xv[q].w * xv[q].w;
#pragma unroll
      for (int r = 0; r < 4; r++) acc[r][q] += wv[r].x * xv[q].x + wv[r].y * xv[q].y + wv[r].z * xv[q].z + wv[r].w * xv[q].w;
    }
  }
  for (int q = 0; q < nt; q++) {
    const float sq = block_sum(ss[q], sh);
    for (int r = 0; r < 4; r++) {
      const float a = block_sum(acc[r][q], sh);
      if (threadIdx.x == 0) dots[(t0 + q) * 24 + i0 + r] = a * rsqrtf(sq / hcd + eps);
    }
  }
}

// pre/post/comb from the dots (Sinkhorn), one thread per token; in place on mixes [T][24]
extern "C" __global__ void hc_finish(float *mixes, const float *scale, const float *base, float hc_eps, int iters, int T) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= T) return;
  const int hc = 4;
  float *o = mixes + t * 24;
  float mix[24];
  for (int j = 0; j < 24; j++) mix[j] = o[j];
  for (int j = 0; j < hc; j++) o[j] = 1.f / (1.f + expf(-(mix[j] * scale[0] + base[j]))) + hc_eps;
  for (int j = 0; j < hc; j++) o[4 + j] = 2.f / (1.f + expf(-(mix[j + hc] * scale[1] + base[j + hc])));
  float c[16];
  for (int j = 0; j < 16; j++) c[j] = mix[j + 2 * hc] * scale[2] + base[j + 2 * hc];
  for (int j = 0; j < hc; j++) {
    float mx = -INFINITY;
    for (int k = 0; k < hc; k++) mx = fmaxf(mx, c[j * hc + k]);
    float s = 0.f;
    for (int k = 0; k < hc; k++) { c[j * hc + k] = expf(c[j * hc + k] - mx); s += c[j * hc + k]; }
    for (int k = 0; k < hc; k++) c[j * hc + k] = c[j * hc + k] / s + hc_eps;
  }
  for (int k = 0; k < hc; k++) {
    float s = 0.f;
    for (int j = 0; j < hc; j++) s += c[j * hc + k];
    for (int j = 0; j < hc; j++) c[j * hc + k] /= s + hc_eps;
  }
  for (int it = 0; it < iters - 1; it++) {
    for (int j = 0; j < hc; j++) {
      float s = 0.f;
      for (int k = 0; k < hc; k++) s += c[j * hc + k];
      for (int k = 0; k < hc; k++) c[j * hc + k] /= s + hc_eps;
    }
    for (int k = 0; k < hc; k++) {
      float s = 0.f;
      for (int j = 0; j < hc; j++) s += c[j * hc + k];
      for (int j = 0; j < hc; j++) c[j * hc + k] /= s + hc_eps;
    }
  }
  for (int j = 0; j < 16; j++) o[8 + j] = c[j];
}

// Q8_0 repacked, 4 rows per warp: one load of x serves 4 rows (x traffic was 4x the weight traffic at 1 row/warp).
// block 128 = 4 warps = 16 rows. T <= 8.
extern "C" __global__ void gemv_q8r4(const int8_t *__restrict__ qs, const float *__restrict__ x, float *__restrict__ y,
                                     int rows, int k, int T, int ldx, int ldy, const __half *__restrict__ d) {
  int r0 = (blockIdx.x * 4 + (threadIdx.x >> 5)) * 4, lane = threadIdx.x & 31;
  if (r0 >= rows) return;
  int nr = min(4, rows - r0);
  float acc[MAXT][4];
#pragma unroll
  for (int t = 0; t < MAXT; t++)
#pragma unroll
    for (int j = 0; j < 4; j++) acc[t][j] = 0.f;
  int nb = k / 32;
  for (int i = lane * 16; i < k; i += 512) {
    float w[4][16];
#pragma unroll
    for (int j = 0; j < 4; j++) {
      if (j < nr) {
        int4 raw = __ldg((const int4 *)(qs + (size_t)(r0 + j) * k + i));
        const int8_t *q = (const int8_t *)&raw;
        float sc = __half2float(d[(size_t)(r0 + j) * nb + i / 32]);
#pragma unroll
        for (int e = 0; e < 16; e++) w[j][e] = (float)q[e] * sc;
      } else {
#pragma unroll
        for (int e = 0; e < 16; e++) w[j][e] = 0.f;
      }
    }
    for (int t = 0; t < T; t++) {
      const float4 *xp = (const float4 *)(x + (size_t)t * ldx + i);
      float xv[16];
      *(float4 *)&xv[0] = xp[0];
      *(float4 *)&xv[4] = xp[1];
      *(float4 *)&xv[8] = xp[2];
      *(float4 *)&xv[12] = xp[3];
#pragma unroll
      for (int j = 0; j < 4; j++) {
        float s = 0.f;
#pragma unroll
        for (int e = 0; e < 16; e++) s += w[j][e] * xv[e];
        acc[t][j] += s;
      }
    }
  }
  for (int t = 0; t < T; t++)
#pragma unroll
    for (int j = 0; j < 4; j++) {
      float s = warp_sum(acc[t][j]);
      if (lane == 0 && j < nr) y[(size_t)t * ldy + r0 + j] = s;
    }
}

// hc_finish for one token in one warp: lane (j*4+k) owns comb[j][k]; row/col sums by shuffles. m = this lane's raw mix
// (lanes < 24), o = the token's 24 outputs.
__device__ __forceinline__ void hc_finish_warp(float *o, float m, int lane, const float *scale, const float *base, float hc_eps, int iters) {
  if (lane < 4) o[lane] = 1.f / (1.f + expf(-(m * scale[0] + base[lane]))) + hc_eps;
  else if (lane < 8) o[lane] = 2.f / (1.f + expf(-(m * scale[1] + base[lane])));
  // comb lanes 8..23 -> index c = lane - 8 = j*4 + k; work in lanes 0..15 via a shuffle
  float c = __shfl_sync(0xffffffff, m, (lane & 15) + 8) * scale[2] + base[(lane & 15) + 8];
  // row softmax over k (groups of 4 lanes)
  float mx = c;
  mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 1));
  mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 2));
  c = expf(c - mx);
  float s = c + __shfl_xor_sync(0xffffffff, c, 1);
  s += __shfl_xor_sync(0xffffffff, s, 2);
  c = c / s + hc_eps;
  // column normalize (lanes with the same k: xor 4, 8)
  s = c + __shfl_xor_sync(0xffffffff, c, 4);
  s += __shfl_xor_sync(0xffffffff, s, 8);
  c = c / (s + hc_eps);
  for (int it = 0; it < iters - 1; it++) {
    s = c + __shfl_xor_sync(0xffffffff, c, 1);
    s += __shfl_xor_sync(0xffffffff, s, 2);
    c = c / (s + hc_eps);
    s = c + __shfl_xor_sync(0xffffffff, c, 4);
    s += __shfl_xor_sync(0xffffffff, s, 8);
    c = c / (s + hc_eps);
  }
  __syncwarp();
  if (lane < 16) o[8 + lane] = c;
}

// hc_finish with one warp per token, raw mixes read from (and results written to) mixes
extern "C" __global__ void hc_finish_w(float *mixes, const float *scale, const float *base, float hc_eps, int iters, int T) {
  int t = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (t >= T) return;
  float *o = mixes + t * 24;
  float m = lane < 24 ? o[lane] : 0.f;
  __syncwarp();
  hc_finish_warp(o, m, lane, scale, base, hc_eps, iters);
}

// decode (one token): hc_reduce + hc_finish_w in one launch; grid 1, block 32
extern "C" __global__ void hc_reduce_finish(const float *part, float *mixes, int hcd, int S, float eps, const float *scale,
                                            const float *base, float hc_eps, int iters) {
  int lane = threadIdx.x;
  float ss = 0.f;
  for (int s = 0; s < S; s++) ss += part[24 * S + s];
  float r = rsqrtf(ss / hcd + eps);
  float acc = 0.f;
  if (lane < 24)
    for (int s = 0; s < S; s++) acc += part[lane * S + s];
  hc_finish_warp(mixes, lane < 24 ? acc * r : 0.f, lane, scale, base, hc_eps, iters);
}

// gemv_q8r for at most TT tokens, with up to U of a lane's 16-byte loads issued before any is used: the same per-chunk
// arithmetic in the same order, so outputs are bit-identical to gemv_q8r (bench/q8h_gemv.cu has the U > 1 variants).
template <int TT, int U>
__device__ __forceinline__ void gemv_q8h_body(const int8_t *__restrict__ qs, const float *__restrict__ x, float *__restrict__ y,
                                              int rows, int k, int T, int ldx, int ldy, const __half *__restrict__ d) {
  int r = blockIdx.x * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (r >= rows) return;
  const int8_t *qr = qs + (size_t)r * k;
  const __half *dr = d + (size_t)r * (k / 32);
  float acc[TT] = {0};
  for (int i0 = lane * 16; i0 < k; i0 += U * 512) {
    int4 raw[U];
    float scs[U];
#pragma unroll
    for (int u = 0; u < U; u++) {
      int i = i0 + u * 512;
      if (i < k) {
        raw[u] = __ldg((const int4 *)(qr + i));
        scs[u] = __half2float(dr[i / 32]);
      }
    }
#pragma unroll
    for (int u = 0; u < U; u++) {
      int i = i0 + u * 512;
      if (i < k) {
        const int8_t *q = (const int8_t *)&raw[u];
        float sc = scs[u];
        float w[16];
#pragma unroll
        for (int j = 0; j < 16; j++) w[j] = (float)q[j] * sc;
#pragma unroll
        for (int t = 0; t < TT; t++) {
          if (t < T) {
            const float4 *xp = (const float4 *)(x + (size_t)t * ldx + i);
            float4 a = xp[0], b = xp[1], c = xp[2], e = xp[3];
            acc[t] += w[0] * a.x + w[1] * a.y + w[2] * a.z + w[3] * a.w + w[4] * b.x + w[5] * b.y + w[6] * b.z + w[7] * b.w +
                      w[8] * c.x + w[9] * c.y + w[10] * c.z + w[11] * c.w + w[12] * e.x + w[13] * e.y + w[14] * e.z + w[15] * e.w;
          }
        }
      }
    }
  }
#pragma unroll
  for (int t = 0; t < TT; t++) {
    if (t < T) {
      float s = warp_sum(acc[t]);
      if (lane == 0) y[(size_t)t * ldy + r] = s;
    }
  }
}
// decode (T = 1): gemv_q8r in 40 registers instead of 62 (no MAXT accumulators): more warps per SM, bit-identical.
// 3090 (bench/q8h_gemv.cu, interleaved): o_b -21..-30%, q_b -10..-15%, shared gate/up -12%, shared down equal.
extern "C" __global__ void gemv_q8r1(const int8_t *__restrict__ qs, const float *__restrict__ x, float *__restrict__ y,
                                     int rows, int k, int T, int ldx, int ldy, const __half *__restrict__ d) {
  gemv_q8h_body<1, 1>(qs, x, y, rows, k, 1, ldx, ldy, d);
}

// decode o_b: gemv_q8r1, then hc_post (hc_post_mb with mixes) for each output element in the epilogue (rows = d).
// Bit-identical to gemv_q8r1 + hc_post_mb. grid ceil(rows / 4), block 128.
extern "C" __global__ void gemv_q8r1_post(const int8_t *__restrict__ qs, const float *__restrict__ x, float *__restrict__ y,
                                          int rows, int k, const __half *__restrict__ d, float *h, const float *mixes) {
  int r = blockIdx.x * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (r >= rows) return;
  const int8_t *qr = qs + (size_t)r * k;
  const __half *dr = d + (size_t)r * (k / 32);
  float acc = 0.f;
  for (int i = lane * 16; i < k; i += 512) {
    int4 raw = __ldg((const int4 *)(qr + i));
    const int8_t *q = (const int8_t *)&raw;
    float sc = __half2float(dr[i / 32]);
    float wv[16];
#pragma unroll
    for (int j = 0; j < 16; j++) wv[j] = (float)q[j] * sc;
    const float4 *xp = (const float4 *)(x + i);
    float4 a = xp[0], b = xp[1], c = xp[2], e = xp[3];
    acc += wv[0] * a.x + wv[1] * a.y + wv[2] * a.z + wv[3] * a.w + wv[4] * b.x + wv[5] * b.y + wv[6] * b.z + wv[7] * b.w +
           wv[8] * c.x + wv[9] * c.y + wv[10] * c.z + wv[11] * c.w + wv[12] * e.x + wv[13] * e.y + wv[14] * e.z + wv[15] * e.w;
  }
  float s = warp_sum(acc);
  if (lane == 0) {
    y[r] = s;
    hc_post_elem(h, r, rows, mixes, s);
  }
}

// decode q_b: gemv_q8r1 with RoPE on each head's last rd rows at `pos` (rope with inverse = 0) in the epilogue: a pair is
// rows (2j, 2j + 1) of the same 4-row block, exchanged through shared memory. Bit-identical to gemv_q8r1 + rope.
// rows % 4 == 0, hd % 4 == 0, rd even. grid rows / 4, block 128.
extern "C" __global__ void gemv_q8r1_rope(const int8_t *__restrict__ qs, const float *__restrict__ x, float *__restrict__ y,
                                          int rows, int k, const __half *__restrict__ d, const float *freqs, int hd, int rd,
                                          int pos) {
  __shared__ float sv[4];
  int w = threadIdx.x >> 5, r = blockIdx.x * 4 + w, lane = threadIdx.x & 31;
  const int8_t *qr = qs + (size_t)r * k;
  const __half *dr = d + (size_t)r * (k / 32);
  float acc = 0.f;
  for (int i = lane * 16; i < k; i += 512) {
    int4 raw = __ldg((const int4 *)(qr + i));
    const int8_t *q = (const int8_t *)&raw;
    float sc = __half2float(dr[i / 32]);
    float wv[16];
#pragma unroll
    for (int j = 0; j < 16; j++) wv[j] = (float)q[j] * sc;
    const float4 *xp = (const float4 *)(x + i);
    float4 a = xp[0], b = xp[1], c = xp[2], e = xp[3];
    acc += wv[0] * a.x + wv[1] * a.y + wv[2] * a.z + wv[3] * a.w + wv[4] * b.x + wv[5] * b.y + wv[6] * b.z + wv[7] * b.w +
           wv[8] * c.x + wv[9] * c.y + wv[10] * c.z + wv[11] * c.w + wv[12] * e.x + wv[13] * e.y + wv[14] * e.z + wv[15] * e.w;
  }
  float v = warp_sum(acc);
  if (lane == 0) sv[w] = v;
  __syncthreads();
  if (threadIdx.x < 4) {
    int rr = blockIdx.x * 4 + threadIdx.x, c0 = rr % hd;
    float out = sv[threadIdx.x];
    if (c0 >= hd - rd) {
      int p = (c0 - (hd - rd)) >> 1;
      float ang = (float)pos * freqs[p];
      float sn, cs;
      sincosf(ang, &sn, &cs);
      float a = sv[threadIdx.x & ~1], b = sv[threadIdx.x | 1];
      // as rope compiles it: fma(a, c, -(b * s)) and fma(b, c, a * s), products rounded first
      out = (threadIdx.x & 1) ? __fmaf_rn(b, cs, __fmul_rn(a, sn)) : __fmaf_rn(a, cs, -__fmul_rn(b, sn));
    }
    y[rr] = out;
  }
}

// Q8_0 repacked, one row per warp, 4 independent 16-byte loads in flight per lane (latency hiding).
extern "C" __global__ void __launch_bounds__(128) gemv_q8u(const int8_t *__restrict__ qs, const float *__restrict__ x,
                                                           float *__restrict__ y, int rows, int k, int T, int ldx, int ldy,
                                                           const __half *__restrict__ d) {
  int r = blockIdx.x * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (r >= rows) return;
  const int8_t *qr = qs + (size_t)r * k;
  const __half *dr = d + (size_t)r * (k / 32);
  float acc[MAXT] = {0};
  int i = lane * 16;
  for (; i + 3 * 512 < k; i += 4 * 512) {
    int4 raw[4];
    float sc[4];
#pragma unroll
    for (int u = 0; u < 4; u++) {
      raw[u] = __ldg((const int4 *)(qr + i + u * 512));
      sc[u] = __half2float(dr[(i + u * 512) / 32]);
    }
#pragma unroll
    for (int u = 0; u < 4; u++) {
      const int8_t *q = (const int8_t *)&raw[u];
      int ii = i + u * 512;
      for (int t = 0; t < T; t++) {
        const float4 *xp = (const float4 *)(x + (size_t)t * ldx + ii);
        float4 a = xp[0], b = xp[1], c = xp[2], e = xp[3];
        acc[t] += sc[u] * (q[0] * a.x + q[1] * a.y + q[2] * a.z + q[3] * a.w + q[4] * b.x + q[5] * b.y + q[6] * b.z +
                           q[7] * b.w + q[8] * c.x + q[9] * c.y + q[10] * c.z + q[11] * c.w + q[12] * e.x + q[13] * e.y +
                           q[14] * e.z + q[15] * e.w);
      }
    }
  }
  for (; i < k; i += 512) {
    int4 raw = __ldg((const int4 *)(qr + i));
    const int8_t *q = (const int8_t *)&raw;
    float sc = __half2float(dr[i / 32]);
    for (int t = 0; t < T; t++) {
      const float4 *xp = (const float4 *)(x + (size_t)t * ldx + i);
      float4 a = xp[0], b = xp[1], c = xp[2], e = xp[3];
      acc[t] += sc * (q[0] * a.x + q[1] * a.y + q[2] * a.z + q[3] * a.w + q[4] * b.x + q[5] * b.y + q[6] * b.z +
                      q[7] * b.w + q[8] * c.x + q[9] * c.y + q[10] * c.z + q[11] * c.w + q[12] * e.x + q[13] * e.y +
                      q[14] * e.z + q[15] * e.w);
    }
  }
  for (int t = 0; t < T; t++) {
    float s = warp_sum(acc[t]);
    if (lane == 0) y[(size_t)t * ldy + r] = s;
  }
}

// Q8_0 repacked with grouped input: row r reads x + (r / rpg) * xgs (the block-diagonal wo_a in one launch)
template <bool RND>
__device__ __forceinline__ void gemv_q8g_body(const int8_t *__restrict__ qs, const float *__restrict__ x, float *__restrict__ y,
                                              int rows, int k, int rpg, int xgs, const __half *__restrict__ d) {
  int r = blockIdx.x * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (r >= rows) return;
  const int8_t *qr = qs + (size_t)r * k;
  const __half *dr = d + (size_t)r * (k / 32);
  const float *xr = x + (size_t)(r / rpg) * xgs;
  float acc = 0.f;
  for (int i = lane * 16; i < k; i += 512) {
    int4 raw = __ldg((const int4 *)(qr + i));
    const int8_t *q = (const int8_t *)&raw;
    const float4 *xp = (const float4 *)(xr + i);
    float4 a = xp[0], b = xp[1], c = xp[2], e = xp[3];
    acc += __half2float(dr[i / 32]) * (q[0] * a.x + q[1] * a.y + q[2] * a.z + q[3] * a.w + q[4] * b.x + q[5] * b.y +
                                       q[6] * b.z + q[7] * b.w + q[8] * c.x + q[9] * c.y + q[10] * c.z + q[11] * c.w +
                                       q[12] * e.x + q[13] * e.y + q[14] * e.z + q[15] * e.w);
  }
  acc = warp_sum(acc);
  if (lane == 0) y[r] = RND ? bf16r(acc) : acc;
}
extern "C" __global__ void gemv_q8g(const int8_t *__restrict__ qs, const float *__restrict__ x, float *__restrict__ y,
                                    int rows, int k, int rpg, int xgs, const __half *__restrict__ d) {
  gemv_q8g_body<false>(qs, x, y, rows, k, rpg, xgs, d);
}
// gemv_q8g with the outputs rounded to bf16 (wo_a followed by round_bf16, in one launch)
extern "C" __global__ void gemv_q8g_bf(const int8_t *__restrict__ qs, const float *__restrict__ x, float *__restrict__ y,
                                       int rows, int k, int rpg, int xgs, const __half *__restrict__ d) {
  gemv_q8g_body<true>(qs, x, y, rows, k, rpg, xgs, d);
}

// shared expert, one token: gate and up rows r (Q8_0 repacked) and swiglu into h[r], in one launch; each dot is
// gemv_q8r1's, the swiglu is swiglu's (w = 1): bit-identical to the three launches. grid ceil(inter / 4), block 128.
extern "C" __global__ void gemv_q8r_gu(const int8_t *__restrict__ gq, const __half *__restrict__ gd, const int8_t *__restrict__ uq,
                                       const __half *__restrict__ ud, const float *__restrict__ x, float *__restrict__ h,
                                       int inter, int k, float lim) {
  int r = blockIdx.x * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (r >= inter) return;
  float acc[2] = {0.f, 0.f};
#pragma unroll
  for (int m = 0; m < 2; m++) {
    const int8_t *qr = (m ? uq : gq) + (size_t)r * k;
    const __half *dr = (m ? ud : gd) + (size_t)r * (k / 32);
    for (int i = lane * 16; i < k; i += 512) {
      int4 raw = __ldg((const int4 *)(qr + i));
      const int8_t *q = (const int8_t *)&raw;
      float sc = __half2float(dr[i / 32]);
      float w[16];
#pragma unroll
      for (int j = 0; j < 16; j++) w[j] = (float)q[j] * sc;
      const float4 *xp = (const float4 *)(x + i);
      float4 a = xp[0], b = xp[1], c = xp[2], e = xp[3];
      acc[m] += w[0] * a.x + w[1] * a.y + w[2] * a.z + w[3] * a.w + w[4] * b.x + w[5] * b.y + w[6] * b.z + w[7] * b.w +
                w[8] * c.x + w[9] * c.y + w[10] * c.z + w[11] * c.w + w[12] * e.x + w[13] * e.y + w[14] * e.z + w[15] * e.w;
    }
  }
  float g = warp_sum(acc[0]), u = warp_sum(acc[1]);
  if (lane == 0) {
    float gv = fminf(g, lim);
    float uv = fminf(fmaxf(u, -lim), lim);
    h[r] = bf16r(gv / (1.f + expf(-gv)) * uv * 1.f);
  }
}

// multi-block hc_pre / hc_post for one token: grid ceil(d/256)
extern "C" __global__ void hc_pre_mb(const float *h, const float *mixes, int moff, float *y, int d) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= d) return;
  const float *m = mixes + moff;
  y[i] = bf16r(m[0] * h[i] + m[1] * h[d + i] + m[2] * h[2 * d + i] + m[3] * h[3 * d + i]);
}
extern "C" __global__ void hc_post_mb(float *h, const float *x, const float *mixes, int d) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= d) return;
  hc_post_elem(h, i, d, mixes, x[i]);
}

// One token: hc_pre_mb + rmsnorm(x, w) in one launch, bit-identical to the two: the same per-element expressions, and
// the sum of squares in rmsnorm's thread order (i += 1024). mixes: pre weights at 0..4. d <= 8192. Grid ceil(d / 1024),
// block 1024: every block computes the whole vector (one SM alone is slower) and writes its own 1024 elements of y.
extern "C" __global__ void __launch_bounds__(1024) hc_pre_norm(const float *h, const float *mixes, float *y, const float *w,
                                                               int d, float eps, int round) {
  __shared__ float sh[32];
  const float *m = mixes;
  float yv[8];
  float s = 0.f;
#pragma unroll
  for (int j = 0; j < 8; j++) {
    int i = threadIdx.x + j * 1024;
    if (i < d) {
      float v = bf16r(m[0] * h[i] + m[1] * h[d + i] + m[2] * h[2 * d + i] + m[3] * h[3 * d + i]);
      yv[j] = v;
      s += v * v;
    }
  }
  s = block_sum(s, sh);
  float r = rsqrtf(s / d + eps);
#pragma unroll
  for (int j = 0; j < 8; j++) {
    int i = threadIdx.x + j * 1024;
    if (i < d && j == (int)blockIdx.x) {
      float v = w[i] * (yv[j] * r);
      y[i] = round ? bf16r(v) : v;
    }
  }
}

// DSX_DEFER, residual injection: h[k][i] = bf16(h[k][i] + post[k] * y[i]) for the 4 streams (a deferred expert sum entering
// the residual with the output weights of the layer it belongs to); grid ceil(d/256)
extern "C" __global__ void hc_add_post(float *h, const float *y, const float *post, int d) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= d) return;
  float v = y[i];
#pragma unroll
  for (int k = 0; k < 4; k++) h[k * d + i] = bf16r(h[k * d + i] + post[k] * v);
}

extern "C" __global__ void to_bf16(const float *x, __nv_bfloat16 *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = __float2bfloat16(x[i]);
}

// gather f16 rows by index: y[r] = x[rows[r]] (x already f16); grid (ceil(k/256), n)
extern "C" __global__ void gather_h(const __half *x, const int *rows, __half *y, int k) {
  int r = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < k) y[(size_t)r * k + c] = x[(size_t)rows[r] * k + c];
}

// ---------------------------------------------------------------- split-K GEMV for matrices with few rows
// One block (4 warps) per row; the warps split k and reduce through shared memory. T <= 8.
template <int TY>
__device__ __forceinline__ float load_w(const void *w, size_t i);
template <> __device__ __forceinline__ float load_w<1>(const void *w, size_t i) { return __half2float(((const __half *)w)[i]); }
template <> __device__ __forceinline__ float load_w<30>(const void *w, size_t i) { return __bfloat162float(((const __nv_bfloat16 *)w)[i]); }
template <> __device__ __forceinline__ float load_w<0>(const void *w, size_t i) { return ((const float *)w)[i]; }

// 4 consecutive weights as floats with one vector load (8 bytes f16 / bf16, 16 bytes f32); i is a multiple of 4
template <int TY>
__device__ __forceinline__ void load_w4(const void *w, size_t i, float &w0, float &w1, float &w2, float &w3);
template <> __device__ __forceinline__ void load_w4<1>(const void *w, size_t i, float &w0, float &w1, float &w2, float &w3) {
  uint2 v = __ldg((const uint2 *)((const __half *)w + i));
  __half2 a = *(__half2 *)&v.x, b = *(__half2 *)&v.y;
  w0 = __low2float(a), w1 = __high2float(a), w2 = __low2float(b), w3 = __high2float(b);
}
template <> __device__ __forceinline__ void load_w4<30>(const void *w, size_t i, float &w0, float &w1, float &w2, float &w3) {
  uint2 v = __ldg((const uint2 *)((const __nv_bfloat16 *)w + i));
  __nv_bfloat162 a = *(__nv_bfloat162 *)&v.x, b = *(__nv_bfloat162 *)&v.y;
  w0 = __low2float(a), w1 = __high2float(a), w2 = __low2float(b), w3 = __high2float(b);
}
template <> __device__ __forceinline__ void load_w4<0>(const void *w, size_t i, float &w0, float &w1, float &w2, float &w3) {
  float4 v = __ldg((const float4 *)((const float *)w + i));
  w0 = v.x, w1 = v.y, w2 = v.z, w3 = v.w;
}

// one row per 128-thread block, the 4 warps splitting k; TT: most tokens (1 for the decode variants, fewer registers)
template <int TY, int TT = MAXT>
__device__ void gemv_sk_plain(const void *w, const float *x, float *y, int rows, int k, int T, int ldx, int ldy) {
  __shared__ float red[4][TT];
  int r = blockIdx.x, wid = threadIdx.x >> 5, lane = threadIdx.x & 31;
  float acc[TT] = {0};
  size_t base = (size_t)r * k;
  for (int i = threadIdx.x * 4; i < k; i += blockDim.x * 4) {
    float w0, w1, w2, w3;
    load_w4<TY>(w, base + i, w0, w1, w2, w3);
#pragma unroll
    for (int t = 0; t < TT; t++) {
      if (t < T) {
        float4 a = *(const float4 *)(x + (size_t)t * ldx + i);
        acc[t] += w0 * a.x + w1 * a.y + w2 * a.z + w3 * a.w;
      }
    }
  }
#pragma unroll
  for (int t = 0; t < TT; t++) {
    if (t < T) {
      float s = warp_sum(acc[t]);
      if (lane == 0) red[wid][t] = s;
    }
  }
  __syncthreads();
  if (threadIdx.x < T) y[(size_t)threadIdx.x * ldy + r] = red[0][threadIdx.x] + red[1][threadIdx.x] + red[2][threadIdx.x] + red[3][threadIdx.x];
}
extern "C" __global__ void gemv_f16_sk(const void *w, const float *x, float *y, int rows, int k, int T, int ldx, int ldy) { gemv_sk_plain<1>(w, x, y, rows, k, T, ldx, ldy); }
extern "C" __global__ void gemv_bf16_sk(const void *w, const float *x, float *y, int rows, int k, int T, int ldx, int ldy) { gemv_sk_plain<30>(w, x, y, rows, k, T, ldx, ldy); }
extern "C" __global__ void gemv_f32_sk(const void *w, const float *x, float *y, int rows, int k, int T, int ldx, int ldy) { gemv_sk_plain<0>(w, x, y, rows, k, T, ldx, ldy); }
extern "C" __global__ void gemv_f16_sk1(const void *w, const float *x, float *y, int rows, int k, int T, int ldx, int ldy) { gemv_sk_plain<1, 1>(w, x, y, rows, k, 1, ldx, ldy); }
extern "C" __global__ void gemv_bf16_sk1(const void *w, const float *x, float *y, int rows, int k, int T, int ldx, int ldy) { gemv_sk_plain<30, 1>(w, x, y, rows, k, 1, ldx, ldy); }

template <int TT>
__device__ void gemv_q8r_sk_body(const int8_t *__restrict__ qs, const float *__restrict__ x, float *__restrict__ y, int rows,
                                 int k, int T, int ldx, int ldy, const __half *__restrict__ d) {
  __shared__ float red[4][TT];
  int r = blockIdx.x, wid = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int8_t *qr = qs + (size_t)r * k;
  const __half *dr = d + (size_t)r * (k / 32);
  float acc[TT] = {0};
  for (int i = threadIdx.x * 16; i < k; i += blockDim.x * 16) {
    int4 raw = __ldg((const int4 *)(qr + i));
    const int8_t *q = (const int8_t *)&raw;
    float sc = __half2float(dr[i / 32]);
#pragma unroll
    for (int t = 0; t < TT; t++) {
      if (t < T) {
        const float4 *xp = (const float4 *)(x + (size_t)t * ldx + i);
        float4 a = xp[0], b = xp[1], c = xp[2], e = xp[3];
        acc[t] += sc * (q[0] * a.x + q[1] * a.y + q[2] * a.z + q[3] * a.w + q[4] * b.x + q[5] * b.y + q[6] * b.z + q[7] * b.w +
                        q[8] * c.x + q[9] * c.y + q[10] * c.z + q[11] * c.w + q[12] * e.x + q[13] * e.y + q[14] * e.z + q[15] * e.w);
      }
    }
  }
#pragma unroll
  for (int t = 0; t < TT; t++) {
    if (t < T) {
      float s = warp_sum(acc[t]);
      if (lane == 0) red[wid][t] = s;
    }
  }
  __syncthreads();
  if (threadIdx.x < T) y[(size_t)threadIdx.x * ldy + r] = red[0][threadIdx.x] + red[1][threadIdx.x] + red[2][threadIdx.x] + red[3][threadIdx.x];
}
extern "C" __global__ void gemv_q8r_sk(const int8_t *__restrict__ qs, const float *__restrict__ x, float *__restrict__ y, int rows,
                                       int k, int T, int ldx, int ldy, const __half *__restrict__ d) {
  gemv_q8r_sk_body<MAXT>(qs, x, y, rows, k, T, ldx, ldy, d);
}
extern "C" __global__ void gemv_q8r_sk1(const int8_t *__restrict__ qs, const float *__restrict__ x, float *__restrict__ y, int rows,
                                        int k, int T, int ldx, int ldy, const __half *__restrict__ d) {
  gemv_q8r_sk_body<1>(qs, x, y, rows, k, 1, ldx, ldy, d);
}

// hc dots in more blocks: part[t][i][s] for rows i < 24 (dot) and i = 24 (sum of squares), S splits of hcd.
// grid (25, T, S), block 256
extern "C" __global__ void hc_part(const float *fn, const float *h, float *part, int hcd, int S) {
  __shared__ float sh[32];
  int i = blockIdx.x, t = blockIdx.y, sp = blockIdx.z;
  int chunk = hcd / S, j0 = sp * chunk;
  const float *x = h + (size_t)t * hcd + j0;
  float acc = 0.f;
  if (i < 24) {
    const float *row = fn + (size_t)i * hcd + j0;
    for (int j = threadIdx.x * 4; j < chunk; j += blockDim.x * 4) {
      float4 xv = *(const float4 *)(x + j), wv = *(const float4 *)(row + j);
      acc += wv.x * xv.x + wv.y * xv.y + wv.z * xv.z + wv.w * xv.w;
    }
  } else {
    for (int j = threadIdx.x * 4; j < chunk; j += blockDim.x * 4) {
      float4 xv = *(const float4 *)(x + j);
      acc += xv.x * xv.x + xv.y * xv.y + xv.z * xv.z + xv.w * xv.w;
    }
  }
  acc = block_sum(acc, sh);
  if (threadIdx.x == 0) part[((size_t)t * 25 + i) * S + sp] = acc;
}

// mixes[t][i] = sum_s part[t][i] * rsqrt(sum_s part[t][24] / hcd + eps); grid T, block 32
extern "C" __global__ void hc_reduce(const float *part, float *mixes, int hcd, int S, float eps) {
  int t = blockIdx.x, i = threadIdx.x;
  float ss = 0.f;
  for (int s = 0; s < S; s++) ss += part[((size_t)t * 25 + 24) * S + s];
  float r = rsqrtf(ss / hcd + eps);
  if (i < 24) {
    float acc = 0.f;
    for (int s = 0; s < S; s++) acc += part[((size_t)t * 25 + i) * S + s];
    mixes[t * 24 + i] = acc * r;
  }
}

// ---------------------------------------------------------------- 6-bit dense weights (q6.rs planes)
// Per row r: lo at lo + r*k/2 (u32 word = 8 weights, byte j = weights j | j+4 << 4), hi at hi + r*k/4 (u32 word = 16
// weights, byte j = high bits of j, j+4, j+8, j+12), int8 sub-scales per 16 at sc + r*k/16, f16 per 256 at d + r*k/256.
// A lane takes 32 consecutive weights per step: 4 codes come from one lo word and one hi word in 3 ops, and a code
// becomes a float by a byte permute into 2^23 + c (no int conversion, which runs at quarter rate on the 3090).
// LPR lanes per row (32, or 8 when k / 32 chunks don't split evenly over 32 lanes, e.g. q_b's k = 1280: 40 chunks
// = 5 per lane of 8), PRW rows per lane group, 4 warps per block. Rows grouped: row r reads x + (r / rpg) * xgs (o_a's
// groups each read their slice of o, one launch); rpg = rows, xgs = 0 for a plain GEMV. rpg % PRW == 0. T <= 8.
__device__ __forceinline__ float q6f(unsigned q, int b) {  // byte b of q as a float code - 32
  return __int_as_float(__byte_perm(q, 0x4B000000u, 0x7540 | b)) - 8388640.f;  // 2^23 + c - (2^23 + 32)
}
template <int PRW, int LPR>
__device__ __forceinline__ void gemv_q6(const uint8_t *__restrict__ lo, const float *__restrict__ x, float *__restrict__ y, int rows,
                                        int k, int T, int ldx, int ldy, const uint8_t *__restrict__ hi, const int8_t *__restrict__ sc,
                                        const __half *__restrict__ d, int rpg, int xgs) {
  const int lane = (threadIdx.x & 31) % LPR;
  const int r0 = ((blockIdx.x * 4 + (threadIdx.x >> 5)) * (32 / LPR) + (threadIdx.x & 31) / LPR) * PRW;
  const bool live = r0 < rows;  // no early return: the lane group's shuffles need every lane
  x += (size_t)(min(r0, rows - 1) / rpg) * xgs;
  float acc[PRW][MAXT];
#pragma unroll
  for (int rr = 0; rr < PRW; rr++)
    for (int t = 0; t < MAXT; t++) acc[rr][t] = 0.f;
  for (int i = lane * 32; live && i < k; i += 32 * LPR) {
    uint4 L[PRW];
    uint2 H[PRW];
    float s0[PRW], s1[PRW];
#pragma unroll
    for (int rr = 0; rr < PRW; rr++) {
      int r = min(r0 + rr, rows - 1);
      L[rr] = __ldg((const uint4 *)(lo + (size_t)r * (k / 2) + i / 2));
      H[rr] = __ldg((const uint2 *)(hi + (size_t)r * (k / 4) + i / 4));
      float dd = __half2float(d[(size_t)r * (k / 256) + i / 256]);
      unsigned short s2 = __ldg((const unsigned short *)(sc + (size_t)r * (k / 16) + i / 16));
      s0[rr] = dd * (float)(int8_t)(s2 & 255);
      s1[rr] = dd * (float)(int8_t)(s2 >> 8);
    }
    for (int t = 0; t < T; t++) {
      const float4 *xp = (const float4 *)(x + (size_t)t * ldx + i);
      float4 xv[8];
#pragma unroll
      for (int q = 0; q < 8; q++) xv[q] = xp[q];
#pragma unroll
      for (int rr = 0; rr < PRW; rr++) {
        float p[2] = {0.f, 0.f};
#pragma unroll
        for (int w = 0; w < 4; w++) {  // weights 8w .. 8w + 7
          unsigned lw = w == 0 ? L[rr].x : w == 1 ? L[rr].y : w == 2 ? L[rr].z : L[rr].w;
          unsigned hw = (w < 2 ? H[rr].x : H[rr].y) >> (4 * (w & 1));
          unsigned q0 = (lw & 0x0f0f0f0fu) | ((hw << 4) & 0x30303030u), q1 = ((lw >> 4) & 0x0f0f0f0fu) | ((hw << 2) & 0x30303030u);
          float4 a = xv[2 * w], b = xv[2 * w + 1];
          p[w >> 1] += q6f(q0, 0) * a.x + q6f(q0, 1) * a.y + q6f(q0, 2) * a.z + q6f(q0, 3) * a.w + q6f(q1, 0) * b.x + q6f(q1, 1) * b.y +
                       q6f(q1, 2) * b.z + q6f(q1, 3) * b.w;
        }
        acc[rr][t] += s0[rr] * p[0] + s1[rr] * p[1];
      }
    }
  }
#pragma unroll
  for (int rr = 0; rr < PRW; rr++)
    for (int t = 0; t < T; t++) {
      float s = acc[rr][t];
#pragma unroll
      for (int o = LPR / 2; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
      if (lane == 0 && r0 + rr < rows) y[(size_t)t * ldy + r0 + rr] = s;
    }
}
#define Q6_ARGS                                                                                                        \
  const uint8_t *__restrict__ lo, const float *__restrict__ x, float *__restrict__ y, int rows, int k, int T, int ldx, \
      int ldy, const uint8_t *__restrict__ hi, const int8_t *__restrict__ sc, const __half *__restrict__ d
// 1 row per warp: grid ceil(rows / 4), block 128
extern "C" __global__ void gemv_q6p(Q6_ARGS) { gemv_q6<1, 32>(lo, x, y, rows, k, T, ldx, ldy, hi, sc, d, rows, 0); }
// 4 / 8 rows per warp, grouped: grid ceil(rows / (4 * PRW)), block 128
extern "C" __global__ void gemv_q6g4(Q6_ARGS, int rpg, int xgs) { gemv_q6<4, 32>(lo, x, y, rows, k, T, ldx, ldy, hi, sc, d, rpg, xgs); }
extern "C" __global__ void gemv_q6g8(Q6_ARGS, int rpg, int xgs) { gemv_q6<8, 32>(lo, x, y, rows, k, T, ldx, ldy, hi, sc, d, rpg, xgs); }
// 8 lanes per row, 4 rows per lane group (16 rows per warp): grid ceil(rows / 64), block 128; for k / 32 % 8 == 0
extern "C" __global__ void gemv_q6s8(Q6_ARGS, int rpg, int xgs) { gemv_q6<4, 8>(lo, x, y, rows, k, T, ldx, ldy, hi, sc, d, rpg, xgs); }

// q6 planes -> f16 for prefill GEMMs: one thread per weight
extern "C" __global__ void dequant_q6p_f16(const uint8_t *lo, const uint8_t *hi, const int8_t *sc, const __half *d, __half *out,
                                           int rows, int k) {
  int r = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= k) return;
  size_t g = (size_t)r * k + c;
  int q = ((lo[(g >> 3) * 4 + (g & 3)] >> (4 * ((g >> 2) & 1))) & 15) | (((hi[(g >> 4) * 4 + (g & 3)] >> (2 * ((g >> 2) & 3))) & 3) << 4);
  out[g] = __float2half(__half2float(d[g / 256]) * (float)sc[g / 16] * (float)(q - 32));
}

// ---------------------------------------------------------------- llama.cpp Q5_K as 5-bit planes (q6.rs from_q5k_row)
// Per row r: lo nibbles at lo + r*k/2, 5th bits at hb + r*k/8, (scale, min) bytes per 32 at sm + r*k/16, f16 (d, dmin) at
// dd + r*k/128. weight = d*scale*code - dmin*min. One warp per row, a lane takes one 32-weight sub-block per step. T <= 8.
extern "C" __global__ void gemv_q5p(const uint8_t *__restrict__ lo, const float *__restrict__ x, float *__restrict__ y, int rows,
                                    int k, int T, int ldx, int ldy, const uint8_t *__restrict__ hb, const uint8_t *__restrict__ sm,
                                    const __half *__restrict__ dd) {
  int r = blockIdx.x * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (r >= rows) return;
  const uint8_t *lr = lo + (size_t)r * (k / 2), *hr = hb + (size_t)r * (k / 8), *sr = sm + (size_t)r * (k / 16);
  const __half *dr = dd + (size_t)r * (k / 128);
  float acc[MAXT] = {0};
  for (int i = lane * 32; i < k; i += 1024) {
    uint4 l16 = __ldg((const uint4 *)(lr + i / 2));
    unsigned h = __ldg((const unsigned *)(hr + i / 8));
    uchar2 s2 = __ldg((const uchar2 *)(sr + i / 16));
    half2 d2 = __ldg((const half2 *)(dr + i / 256 * 2));
    float a = __low2float(d2) * (float)s2.x, b = __high2float(d2) * (float)s2.y;
    const uint8_t *lb = (const uint8_t *)&l16;
    float w[32];
#pragma unroll
    for (int j = 0; j < 32; j++) w[j] = (float)(((lb[j / 2] >> (4 * (j & 1))) & 15) | (((h >> j) & 1) << 4));
    for (int t = 0; t < T; t++) {
      const float4 *xp = (const float4 *)(x + (size_t)t * ldx + i);
      float s = 0.f, sx = 0.f;
#pragma unroll
      for (int q = 0; q < 8; q++) {
        float4 v = xp[q];
        s += w[4 * q] * v.x + w[4 * q + 1] * v.y + w[4 * q + 2] * v.z + w[4 * q + 3] * v.w;
        sx += v.x + v.y + v.z + v.w;
      }
      acc[t] += a * s - b * sx;
    }
  }
  for (int t = 0; t < T; t++) {
    float s = warp_sum(acc[t]);
    if (lane == 0) y[(size_t)t * ldy + r] = s;
  }
}

// Q5_K planes with 4 rows per warp (16 per block of 128): each x load serves 4 rows
#define PR 4
extern "C" __global__ void gemv_q5p4(const uint8_t *__restrict__ lo, const float *__restrict__ x, float *__restrict__ y, int rows,
                                     int k, int T, int ldx, int ldy, const uint8_t *__restrict__ hb, const uint8_t *__restrict__ sm,
                                     const __half *__restrict__ dd) {
  int r0 = (blockIdx.x * 4 + (threadIdx.x >> 5)) * PR, lane = threadIdx.x & 31;
  if (r0 >= rows) return;
  float acc[PR][MAXT];
#pragma unroll
  for (int rr = 0; rr < PR; rr++)
    for (int t = 0; t < MAXT; t++) acc[rr][t] = 0.f;
  for (int i = lane * 32; i < k; i += 1024) {
    uint4 l16[PR];
    unsigned h[PR];
    float a[PR], b[PR];
#pragma unroll
    for (int rr = 0; rr < PR; rr++) {
      int r = min(r0 + rr, rows - 1);
      l16[rr] = __ldg((const uint4 *)(lo + (size_t)r * (k / 2) + i / 2));
      h[rr] = __ldg((const unsigned *)(hb + (size_t)r * (k / 8) + i / 8));
      uchar2 s2 = __ldg((const uchar2 *)(sm + (size_t)r * (k / 16) + i / 16));
      half2 d2 = __ldg((const half2 *)(dd + (size_t)r * (k / 128) + i / 256 * 2));
      a[rr] = __low2float(d2) * (float)s2.x;
      b[rr] = __high2float(d2) * (float)s2.y;
    }
    for (int t = 0; t < T; t++) {
      const float4 *xp = (const float4 *)(x + (size_t)t * ldx + i);
      float sx = 0.f, sw[PR];
#pragma unroll
      for (int rr = 0; rr < PR; rr++) sw[rr] = 0.f;
#pragma unroll
      for (int q = 0; q < 8; q++) {
        float4 v = xp[q];
        sx += v.x + v.y + v.z + v.w;
#pragma unroll
        for (int rr = 0; rr < PR; rr++) {
          unsigned lw = q < 4 ? (q < 2 ? l16[rr].x : l16[rr].y) : (q < 6 ? l16[rr].z : l16[rr].w);
          unsigned l2 = lw >> (16 * (q & 1)), h4 = h[rr] >> (4 * q);
          float c0 = (float)((l2 & 15) | ((h4 & 1) << 4)), c1 = (float)(((l2 >> 4) & 15) | ((h4 & 2) << 3));
          float c2 = (float)(((l2 >> 8) & 15) | ((h4 & 4) << 2)), c3 = (float)(((l2 >> 12) & 15) | ((h4 & 8) << 1));
          sw[rr] += c0 * v.x + c1 * v.y + c2 * v.z + c3 * v.w;
        }
      }
#pragma unroll
      for (int rr = 0; rr < PR; rr++) acc[rr][t] += a[rr] * sw[rr] - b[rr] * sx;
    }
  }
#pragma unroll
  for (int rr = 0; rr < PR; rr++)
    for (int t = 0; t < T; t++) {
      float s = warp_sum(acc[rr][t]);
      if (lane == 0 && r0 + rr < rows) y[(size_t)t * ldy + r0 + rr] = s;
    }
}

// Q5_K planes -> f16 for prefill GEMMs: one thread per weight
extern "C" __global__ void dequant_q5p_f16(const uint8_t *lo, const uint8_t *hb, const uint8_t *sm, const __half *dd, __half *out,
                                           int rows, int k) {
  int r = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= k) return;
  size_t g = (size_t)r * k + c;
  int q = ((lo[g / 2] >> (4 * (g & 1))) & 15) | (((hb[g / 8] >> (g & 7)) & 1) << 4);
  out[g] = __float2half(__half2float(dd[g / 256 * 2]) * (float)sm[g / 32 * 2] * (float)q - __half2float(dd[g / 256 * 2 + 1]) * (float)sm[g / 32 * 2 + 1]);
}


// ---------------------------------------------------------------------------------------------------------------
// Vision tower (DeepSeek ViT + aligner), prefill-time only. Activations f32, GEMM inputs bf16.

// out[n][D] (bf16) = rmsnorm(x[n][D]) * w   (eps 1e-6, as vision.py); one block per row
extern "C" __global__ void vit_rmsnorm(const float *x, const float *w, __nv_bfloat16 *out, int D, float eps) {
  const float *r = x + (size_t)blockIdx.x * D;
  __shared__ float red[32];
  float s = 0.f;
  for (int i = threadIdx.x; i < D; i += blockDim.x) s += r[i] * r[i];
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = s;
  __syncthreads();
  if (threadIdx.x < 32) {
    s = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : 0.f;
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
    if (threadIdx.x == 0) red[0] = rsqrtf(s / D + eps);
  }
  __syncthreads();
  float k = red[0];
  for (int i = threadIdx.x; i < D; i += blockDim.x) out[(size_t)blockIdx.x * D + i] = __float2bfloat16(r[i] * k * w[i]);
}

// x[n][D] += y[n][D] + b[D]  (b may be null); or with acc == 0: x = y + b
extern "C" __global__ void vit_bias_add(float *x, const float *y, const float *b, int n, int D, int acc) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n * D) return;
  float v = y[i] + (b ? b[i % D] : 0.f);
  x[i] = acc ? x[i] + v : v;
}

// qkv[n][3*H*64] (f32, bias not yet added) -> q, k, v as bf16 [H][n][64], 2D RoPE on q and k:
// element i < 32 pairs with i + 32; angle uses the row index for i % 32 < 16, the column index otherwise
extern "C" __global__ void vit_rope_split(const float *qkv, const float *bqkv, __nv_bfloat16 *q, __nv_bfloat16 *k,
                                          __nv_bfloat16 *v, int n, int H, int nw, float theta) {
  int t = blockIdx.x, h = blockIdx.y, i = threadIdx.x;  // 32 threads: i in 0..32
  int D = H * 64;
  const float *row = qkv + (size_t)t * 3 * D;
  int hp = t / nw, wp = t % nw;
  int f = i & 15;
  float inv = powf(theta, -(float)(2 * f) / 32.f);
  float ang = (i < 16 ? hp : wp) * inv;
  float c = cosf(ang), s = sinf(ang);
  size_t o = ((size_t)h * n + t) * 64;
  for (int part = 0; part < 2; part++) {
    const float *src = row + part * D + h * 64;
    const float *bb = bqkv + part * D + h * 64;
    float x1 = src[i] + bb[i], x2 = src[i + 32] + bb[i + 32];
    __nv_bfloat16 *dst = part == 0 ? q : k;
    dst[o + i] = __float2bfloat16(x1 * c - x2 * s);
    dst[o + i + 32] = __float2bfloat16(x2 * c + x1 * s);
  }
  const float *src = row + 2 * D + h * 64;
  const float *bb = bqkv + 2 * D + h * 64;
  v[o + i] = __float2bfloat16(src[i] + bb[i]);
  v[o + i + 32] = __float2bfloat16(src[i + 32] + bb[i + 32]);
}

// row softmax of s[rows][n] * scale -> p (bf16); one block per row
extern "C" __global__ void vit_softmax(const float *s, __nv_bfloat16 *p, int n, float scale) {
  const float *r = s + (size_t)blockIdx.x * n;
  __nv_bfloat16 *o = p + (size_t)blockIdx.x * n;
  __shared__ float red[32];
  float m = -INFINITY;
  for (int i = threadIdx.x; i < n; i += blockDim.x) m = fmaxf(m, r[i]);
  for (int k = 16; k; k >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, k));
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = m;
  __syncthreads();
  if (threadIdx.x < 32) {
    m = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : -INFINITY;
    for (int k = 16; k; k >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, k));
    if (threadIdx.x == 0) red[0] = m;
  }
  __syncthreads();
  m = red[0];
  __syncthreads();
  float sum = 0.f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) sum += expf((r[i] - m) * scale);
  for (int k = 16; k; k >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, k);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = sum;
  __syncthreads();
  if (threadIdx.x < 32) {
    sum = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : 0.f;
    for (int k = 16; k; k >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, k);
    if (threadIdx.x == 0) red[0] = 1.f / sum;
  }
  __syncthreads();
  float inv = red[0];
  for (int i = threadIdx.x; i < n; i += blockDim.x) o[i] = __float2bfloat16(expf((r[i] - m) * scale) * inv);
}

// o[H][qc][64] (f32) -> cat[q0 + t][h * 64 + j] (bf16)
extern "C" __global__ void vit_heads_merge(const float *o, __nv_bfloat16 *cat, int H, int qc, int q0) {
  int t = blockIdx.x, h = blockIdx.y, j = threadIdx.x;
  cat[(size_t)(q0 + t) * H * 64 + h * 64 + j] = __float2bfloat16(o[((size_t)h * qc + t) * 64 + j]);
}

// g[n][2I] (gate | up, f32) -> a[n][I] = silu(gate) * up (bf16)
extern "C" __global__ void vit_swiglu(const float *g, __nv_bfloat16 *a, int n, int I) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n * I) return;
  long t = i / I;
  int j = i % I;
  float gt = g[t * 2 * I + j], up = g[t * 2 * I + I + j];
  a[i] = __float2bfloat16(gt / (1.f + expf(-gt)) * up);
}

// F.unfold(pad(x as [C][nh][nw]), 3, stride 3)^T: out[(i * lw + j)][c * 9 + ky * 3 + kx] = x[(3i+ky) * nw + 3j+kx][c]
extern "C" __global__ void vit_unfold3(const float *x, __nv_bfloat16 *out, int C, int nh, int nw, int lw) {
  int L = blockIdx.x, i = L / lw, j = L % lw;
  for (int e = threadIdx.x; e < C * 9; e += blockDim.x) {
    int c = e / 9, kk = e % 9, ky = kk / 3, kx = kk % 3;
    int y = 3 * i + ky, xx = 3 * j + kx;
    float v = (y < nh && xx < nw) ? x[((size_t)y * nw + xx) * C + c] : 0.f;
    out[(size_t)L * C * 9 + e] = __float2bfloat16(v);
  }
}

// x = gelu(y + b) (exact, erf) -> bf16
extern "C" __global__ void vit_gelu_bias(const float *y, const float *b, __nv_bfloat16 *out, int n, int D) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n * D) return;
  float v = y[i] + b[i % D];
  out[i] = __float2bfloat16(0.5f * v * (1.f + erff(v * 0.70710678f)));
}

// f32 -> bf16
extern "C" __global__ void vit_to_bf16(const float *x, __nv_bfloat16 *out, int n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = __float2bfloat16(x[i]);
}

// rotated experts in prefill: w [rows][cols] f16 dequantized from W R -> W (each 128-wide block of every row through
// the same orthonormal Hadamard, its own inverse). grid (cols / 128, rows), block 128.
extern "C" __global__ void fwht_rows_f16(__half *w, int cols) {
  __shared__ float fb[128];
  int t = threadIdx.x;
  size_t i = (size_t)blockIdx.y * cols + blockIdx.x * 128 + t;
  fb[t] = __half2float(w[i]);
  __syncthreads();
#pragma unroll
  for (int h = 1; h < 128; h <<= 1) {
    float own = fb[t], other = fb[t ^ h];
    __syncthreads();
    fb[t] = (t & h) ? other - own : own + other;
    __syncthreads();
  }
  w[i] = __float2half(fb[t] * 0.08838834764831845f);
}

// pointer chase (one thread): steps dependent loads through `next`, for DRAM load latency (tests/gpu_latency.rs)
extern "C" __global__ void chase(const uint32_t *__restrict__ next, int steps, uint32_t *out) {
  uint32_t i = 0;
  for (int s = 0; s < steps; s++) i = __ldcg(next + i);
  *out = i;
}
