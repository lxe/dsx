// T5: correctness + perf bench for the SM70 dp4a port of expert_fast.cuh (xgemv, moe_gu_fast, moe_down_fast).
// Loads bench/t5.fatbin (built by bench/t5.sh from bench/t5_kernels.cu, the same expert_fast path dsx.cu compiles)
// through the driver API, exactly as nvidia/src/gpu.rs loads dsx.fatbin.
//
// Correctness protocol (all scales dyadic, so every fp32 intermediate is exact and the comparison is bit-exact):
//   (i)   host GGUF decoders vs rp_weight (the prefill decoder, t5_rp_weight) over a whole repacked matrix;
//   (ii)  xgemv vs the host reference y[t][r] = sum_k W[r][k] x[t][k] in double — max-abs must be 0 (the integer
//         path: dp4a and IMMA sum the same products, only in another order);
//   (iii) moe_gu_fast vs the host swiglu (bf16 tolerance, expf is not dyadic) and moe_down_fast vs the host sum
//         (max-abs 0: pure adds of dyadics), including non-resident picks and the plan[0] == 0 path.
// Perf: expert shapes k = 7168, n = 2048, T = 8 (xgemv over a pool so L2 never serves a repeat; moe over 48
// distinct resident experts), 3 warmup + 20 timed launches, cudaEvent timing. Results: bench/T5-RESULTS.md.
#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>

#define PLAN_W 10  // as expert_fast.cuh

#define CK(x)                                                                                                                          \
  do {                                                                                                                                 \
    CUresult r_ = (x);                                                                                                                 \
    if (r_ != CUDA_SUCCESS) {                                                                                                          \
      const char *s_ = nullptr;                                                                                                        \
      cuGetErrorString(r_, &s_);                                                                                                       \
      fprintf(stderr, "%s:%d cu error %d (%s): %s\n", __FILE__, __LINE__, (int)r_, s_ ? s_ : "?", #x);                                 \
      exit(1);                                                                                                                         \
    }                                                                                                                                  \
  } while (0)

static uint64_t rng_s = 0x9e3779b97f4a7c15ull;
static inline uint64_t rnd() {
  rng_s ^= rng_s << 13;
  rng_s ^= rng_s >> 7;
  rng_s ^= rng_s << 17;
  return rng_s;
}
static inline int rndi(int n) { return (int)(rnd() % n); }

// ---------------------------------------------------------------- host format decoders (GGUF bytes -> value)
struct Tables {
  std::vector<unsigned long long> g64;  // iq2xxs [0,256), iq2xs [256,768), iq2s [768,1792), iq1s [1792,3840)
  std::vector<uint8_t> ksigns;          // ksigns_iq2xs [128]
};
static Tables TAB;

static inline uint16_t rd16(const uint8_t *p) { return (uint16_t)(p[0] | (p[1] << 8)); }
static inline uint32_t rd32(const uint8_t *p) { return p[0] | (p[1] << 8) | (p[2] << 16) | ((uint32_t)p[3] << 24); }
static float h2f(uint16_t h) {
  unsigned sign = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 0x3ff;
  uint32_t bits;
  if (e == 0) {
    bits = 0;
    if (m != 0) {  // subnormal (not generated here; kept so the decoder is honest)
      int ee = -1;
      unsigned mm = m;
      do { mm <<= 1; ee++; } while (!(mm & 0x400));
      bits = (sign << 31) | ((uint32_t)(127 - 15 - ee - 1 + 127) << 23) | ((mm & 0x3ff) << 13);
    }
  } else if (e == 31) {
    bits = (sign << 31) | 0x7f800000u | (m << 13);
  } else {
    bits = (sign << 31) | ((e - 15 + 127) << 23) | (m << 13);
  }
  if (!bits) bits = sign << 31;
  float f;
  memcpy(&f, &bits, 4);
  return f;
}

// weight e of one GGUF block of type ty (the rp_weight / iq*_value semantics, as host code)
static double w_value(int ty, const uint8_t *blk, int e) {
  if (ty == 42) {  // Q2_0: 16 code bytes + d
    int code = (blk[2 + e / 4] >> (2 * (e % 4))) & 3;
    return (double)(code - 1) * h2f(rd16(blk));
  }
  if (ty == 8) {  // Q8_0: 32 s8 + d
    return (double)(int8_t)blk[2 + e] * h2f(rd16(blk));
  }
  if (ty == 11) {  // Q3_K: hmask 32, qs 64, scales 12, d (repack's q3_code semantics)
    int h = e >> 7, j = (e >> 5) & 3, l = e & 31;
    int low = (blk[32 + 32 * h + l] >> (2 * j)) & 3;
    int v = low - (((blk[l] >> (4 * h + j)) & 1) ? 0 : 4);
    int s16 = e >> 4, i4 = s16 & 3, grp = s16 >> 2;
    const uint8_t *raw = blk + 96;
    int a = raw[i4], cc = raw[4 + i4], tmp = raw[8 + i4], sc;
    if (grp == 0) sc = (a & 15) | ((tmp & 3) << 4);
    else if (grp == 1) sc = (cc & 15) | (((tmp >> 2) & 3) << 4);
    else if (grp == 2) sc = (a >> 4) | (((tmp >> 4) & 3) << 4);
    else sc = (cc >> 4) | (((tmp >> 6) & 3) << 4);
    return (double)h2f(rd16(blk + 108)) * (double)(sc - 32) * v;
  }
  if (ty == 143) {  // PTQ1_0: base-3 bytes (repack's t[] extraction), d at +26
    static const int p3[5] = {1, 3, 9, 27, 81};
    int t;
    if (e < 80) t = ((blk[e % 16] * p3[e / 16]) & 255) * 3 >> 8;
    else if (e < 120) t = ((blk[16 + (e & 7)] * p3[(e - 80) / 8]) & 255) * 3 >> 8;
    else t = ((blk[24 + (e & 1)] * p3[(e - 120) / 2]) & 255) * 3 >> 8;
    return (double)(t - 1) * h2f(rd16(blk + 26));
  }
  if (ty == 16) {  // IQ2_XXS: per 32: u32 grid indices, u32 signs + scale bit; d
    const uint8_t *p = blk + 2 + 8 * (e >> 5);
    unsigned g = rd32(p), sg = rd32(p + 4);
    int l = (e >> 3) & 3, j = e & 7;
    double v = (double)h2f(rd16(blk)) * (0.5 + (double)(sg >> 28)) * 0.25 * (double)((TAB.g64[(g >> (8 * l)) & 255] >> (8 * j)) & 255);
    return ((TAB.ksigns[(sg >> (7 * l)) & 127] >> j) & 1) ? -v : v;
  }
  if (ty == 19) {  // IQ1_S: qs[32], qh u16[8], d
    int ib = e >> 5, l = (e >> 3) & 3, j = e & 7;
    unsigned qh = rd16(blk + 34 + 2 * ib);
    double dl = (double)h2f(rd16(blk)) * (double)(2 * ((qh >> 12) & 7) + 1);
    int g = (int)(int8_t)((TAB.g64[1792 + (blk[2 + 4 * ib + l] | ((qh >> (3 * l)) & 7) << 8)] >> (8 * j)) & 255);
    return dl * ((double)g + ((qh & 0x8000) ? -0.125 : 0.125));
  }
  if (ty == 17) {  // IQ2_XS: qs u16[32] (9-bit grid + 7-bit sign index), scales[8], d
    int ib = e >> 5, l = (e >> 3) & 3, j = e & 7;
    unsigned q = rd16(blk + 2 + 2 * (4 * ib + l));
    int sc = blk[66 + ib];
    double db = (double)h2f(rd16(blk)) * (0.5 + (double)((l < 2 ? sc : sc >> 4) & 15)) * 0.25;
    double v = db * (double)((TAB.g64[256 + (q & 511)] >> (8 * j)) & 255);
    return ((TAB.ksigns[q >> 9] >> j) & 1) ? -v : v;
  }
  if (ty == 22) {  // IQ2_S: qs[32], signs[32], qh[8], scales[8], d
    int ib = e >> 5, l = (e >> 3) & 3, j = e & 7;
    int sc = blk[74 + ib];
    double db = (double)h2f(rd16(blk)) * (0.5 + (double)((l < 2 ? sc : sc >> 4) & 15)) * 0.25;
    unsigned idx = blk[2 + 4 * ib + l] | ((blk[66 + ib] << (8 - 2 * l)) & 0x300);
    double v = db * (double)((TAB.g64[768 + idx] >> (8 * j)) & 255);
    return (blk[34 + 4 * ib + l] >> j) & 1 ? -v : v;
  }
  fprintf(stderr, "bad ty %d\n", ty);
  exit(1);
}

// ggml id -> repacked type id / GGUF block size / repacked bytes of a rows x k matrix
static int rp_ty(int ty) { return ty == 42 ? 2042 : ty == 8 ? 2008 : ty == 11 ? 2011 : ty == 143 ? 2143 : ty == 16 ? 2016 : ty == 19 ? 2019 : ty == 17 ? 2017 : 2022; }
static int gguf_bs(int ty) { return ty == 42 ? 18 : ty == 8 ? 34 : ty == 11 ? 110 : ty == 143 ? 28 : ty == 16 ? 66 : ty == 19 ? 50 : ty == 17 ? 74 : 82; }
static int gguf_per(int ty) { return ty == 42 ? 64 : ty == 8 ? 32 : ty == 143 ? 128 : 256; }
static size_t gguf_bytes(int ty, int rows, int k) { return (size_t)rows * (k / gguf_per(ty)) * gguf_bs(ty); }
static size_t rp_bytes(int ty, int rows, int k) {
  size_t nb256 = (size_t)rows * (k >> 8);
  switch (ty) {
    case 42: return (size_t)rows * (k >> 6) * 18;
    case 143: return (size_t)rows * (k >> 7) * 28;
    case 11: return nb256 * 110;
    case 8: return (size_t)rows * k + (size_t)rows * (k >> 5) * 2;
    case 16: return nb256 * 66;
    case 19: return nb256 * 50;
    case 17: return nb256 * 74;
    default: return nb256 * 82;
  }
}
static int tile_smem(int ty) {
  int t = rp_ty(ty);
  return 4 * (t == 2022 ? 2048 + 256 : t == 2017 ? 1024 + 256 : t == 2016 ? 512 + 256 : 0);
}
static const char *ty_name(int ty) {
  return ty == 42 ? "Q2_0" : ty == 8 ? "Q8_0" : ty == 11 ? "Q3_K" : ty == 143 ? "PTQ1_0" : ty == 16 ? "IQ2_XXS" : ty == 19 ? "IQ1_S" : ty == 17 ? "IQ2_XS" : "IQ2_S";
}
static inline uint16_t f16_pow2(int a) { return (uint16_t)((15 - a) << 10); }  // dyadic 2^-a
static inline float f32_pow2(int a) { return ldexpf(1.f, -a); }

// ---------------------------------------------------------------- device helpers
struct Dev {
  CUdeviceptr p = 0;
  size_t n = 0;
  Dev() = default;
  Dev(size_t bytes) { n = bytes; CK(cuMemAlloc(&p, bytes)); }
  ~Dev() { if (p) cuMemFree(p); }
  void h2d(const void *h, size_t bytes) { CK(cuMemcpyHtoD(p, h, bytes)); }
  void d2h(void *h, size_t bytes) const { CK(cuMemcpyDtoH(h, p, bytes)); }
};

static CUfunction F_REPACK, F_RPW, F_TAB, F_XGEMV, F_GU, F_DOWN;
static CUmodule F_MOD;

static void launch(CUfunction f, unsigned gx, unsigned gy, unsigned bx, unsigned smem, std::vector<void *> &args) {
  CK(cuLaunchKernel(f, gx, gy, 1, bx, 1, 1, smem, nullptr, args.data(), nullptr));
}

static void repack_to(CUdeviceptr dst, const std::vector<uint8_t> &gg, int ty, int rows, int k) {
  Dev src(gg.size());
  src.h2d(gg.data(), gg.size());
  size_t nb = (size_t)rows * (k / gguf_per(ty));
  int a = ty, r = rows, kk = k;
  void *s = (void *)src.p, *d = (void *)dst;
  std::vector<void *> args{&a, &s, &d, &r, &kk};
  launch(F_REPACK, (unsigned)((nb + 255) / 256), 1, 256, 0, args);
}

static void gen_gguf(std::vector<uint8_t> &gg, int ty, int rows, int k) {
  size_t nb = (size_t)rows * (k / gguf_per(ty));
  gg.assign(nb * gguf_bs(ty), 0);
  for (size_t b = 0; b < nb; b++) {
    uint8_t *blk = &gg[b * gguf_bs(ty)];
    for (int i = 0; i < gguf_bs(ty); i++) blk[i] = (uint8_t)rnd();
    uint16_t d = f16_pow2(1 + rndi(5));  // dyadic 2^-1..2^-5
    blk[0] = d & 0xff;
    blk[1] = d >> 8;
    if (ty == 11) {
      blk[108] = d & 0xff;
      blk[109] = d >> 8;
    } else if (ty == 143) {
      blk[26] = d & 0xff;
      blk[27] = d >> 8;
    }
  }
}

static std::vector<double> dequant_host(const std::vector<uint8_t> &gg, int ty, int rows, int k) {
  std::vector<double> w((size_t)rows * k);
  int per = gguf_per(ty);
  for (int r = 0; r < rows; r++)
    for (int c = 0; c < k; c++) w[(size_t)r * k + c] = w_value(ty, &gg[((size_t)r * (k / per) + c / per) * gguf_bs(ty)], c % per);
  return w;
}

// activations: v vectors of k, int8 in [-16, 15] (bounds keep every fp32 intermediate exact), dyadic per-256
// scales, q in quant_act's permuted layout, exact per-64 sums
struct Act {
  int v, k;
  std::vector<int8_t> q;
  std::vector<float> d;
  std::vector<short> s;
  std::vector<double> val;
  Act(int v_, int k_) : v(v_), k(k_), q((size_t)v_ * k_), d((size_t)v_ * (k_ >> 8)), s((size_t)v_ * (k_ >> 6)), val((size_t)v_ * k_) {
    for (int i = 0; i < v; i++) {
      float dd = f32_pow2(rndi(4));
      for (int j = 0; j < (k_ >> 8); j++) d[i * (k_ >> 8) + j] = dd;  // one scale per 256-block, all of them
      std::vector<int8_t> nat(k);
      for (int c = 0; c < k; c++) nat[c] = (int8_t)(rndi(32) - 16);
      for (int o = 0; o < k; o++) {  // weight 16s + 4q + i -> byte 16q + 4s + i of its 64-chunk
        int ch = o >> 6, sG = (o >> 4) & 3, qL = (o >> 2) & 3, i2 = o & 3;
        q[(size_t)i * k + ch * 64 + 16 * qL + 4 * sG + i2] = nat[o];
      }
      for (int c = 0; c < k; c++) val[(size_t)i * k + c] = (double)nat[c] * dd;
      for (int g = 0; g < k >> 6; g++) {
        int sum = 0;
        for (int c = 0; c < 64; c++) sum += nat[g * 64 + c];
        s[i * (k >> 6) + g] = (short)sum;
      }
    }
  }
};

static inline float bf16r_h(double x) {  // RNE, as __float2bfloat16
  float f = (float)x;
  uint32_t u;
  memcpy(&u, &f, 4);
  uint32_t b = ((u + 0x7fff + ((u >> 16) & 1)) >> 16) << 16;
  memcpy(&f, &b, 4);
  return f;
}

struct SlotH {  // host image of moe.cuh's ExpSlot
  unsigned long long gate, up, down;
  int gu, dn;
};

// ---------------------------------------------------------------- correctness
static int g_fail = 0;

static void check_xgemv(int ty, int rows, int k, int T) {
  std::vector<uint8_t> gg;
  gen_gguf(gg, ty, rows, k);
  Dev w(rp_bytes(ty, rows, k));
  repack_to(w.p, gg, ty, rows, k);
  Dev wd((size_t)rows * k * 4);
  {
    int a = rp_ty(ty), r = rows, kk = k;
    void *wp = (void *)w.p, *o = (void *)wd.p;
    std::vector<void *> args{&a, &wp, &r, &kk, &o};
    launch(F_RPW, 320, 1, 256, 0, args);
  }
  std::vector<float> wdev((size_t)rows * k);
  wd.d2h(wdev.data(), wdev.size() * 4);
  std::vector<double> wref = dequant_host(gg, ty, rows, k);
  double dw = 0;
  for (size_t i = 0; i < wref.size(); i++) dw = std::max(dw, fabs((double)wdev[i] - wref[i]));
  Act x(T, k);
  Dev xq(x.q.size()), xd(x.d.size() * 4), xs(x.s.size() * 2), y((size_t)T * rows * 4);
  xq.h2d(x.q.data(), x.q.size());
  xd.h2d(x.d.data(), x.d.size() * 4);
  xs.h2d(x.s.data(), x.s.size() * 2);
  std::vector<float> yh((size_t)T * rows, NAN);
  y.h2d(yh.data(), yh.size() * 4);
  {
    int a = rp_ty(ty), r = rows, kk = k, t = T;
    void *wp = (void *)w.p, *qp = (void *)xq.p, *dp = (void *)xd.p, *sp = (void *)xs.p, *yp = (void *)y.p;
    std::vector<void *> args{&a, &wp, &r, &kk, &t, &qp, &dp, &sp, &yp};
    launch(F_XGEMV, (unsigned)((rows + 15) / 16), 1, 64, tile_smem(ty), args);
  }
  y.d2h(yh.data(), yh.size() * 4);
  double dy = 0;
  int bad = 0;
  for (int t = 0; t < T; t++)
    for (int r = 0; r < rows; r++) {
      double ref = 0;
      for (int c = 0; c < k; c++) ref += wref[(size_t)r * k + c] * x.val[(size_t)t * k + c];
      double d = fabs((double)yh[(size_t)t * rows + r] - ref);
      if (d != 0) bad++;
      dy = std::max(dy, d);
    }
  printf("  xgemv %-7s rows=%-5d k=%-5d T=%d  rp_weight maxabs %.3g  y maxabs %.3g  bad %d\n", ty_name(ty), rows, k, T, dw, dy, bad);
  if (dw != 0 || dy != 0) g_fail++;
}

static void check_moe(int ty, int dim, int inter, int T, int K, int E, int nres) {
  std::vector<uint8_t> gg_g, gg_u, gg_d;
  gen_gguf(gg_g, ty, inter, dim);  // gate/up: [inter][dim]
  gen_gguf(gg_u, ty, inter, dim);
  gen_gguf(gg_d, ty, dim, inter);  // down: [dim][inter]
  size_t mg = rp_bytes(ty, inter, dim), md = rp_bytes(ty, dim, inter);
  size_t pg = (mg + 15) & ~(size_t)15, pd = (md + 15) & ~(size_t)15, one = 2 * pg + pd;
  Dev store((size_t)E * one), t0(one);
  repack_to(t0.p, gg_g, ty, inter, dim);
  repack_to(t0.p + pg, gg_u, ty, inter, dim);
  repack_to(t0.p + 2 * pg, gg_d, ty, dim, inter);
  std::vector<double> wg = dequant_host(gg_g, ty, inter, dim), wu = dequant_host(gg_u, ty, inter, dim),
                      wd = dequant_host(gg_d, ty, dim, inter);
  std::vector<SlotH> tab((size_t)E * 2);
  for (auto &s : tab) {
    s.gate = ~0ull;
    s.gu = s.dn = rp_ty(ty);
  }
  for (int e = 0; e < nres; e++) {  // resident experts at layer 1: odd slots
    int ee = (e * 2 + 1) % E;
    auto &s = tab[E + ee];
    s.gate = (size_t)ee * one;
    s.up = s.gate + pg;
    s.down = s.gate + 2 * pg;
    CK(cuMemcpyDtoD(store.p + s.gate, t0.p, one));
  }
  Dev tabd(tab.size() * sizeof(SlotH));
  tabd.h2d(tab.data(), tab.size() * sizeof(SlotH));
  // sel: distinct experts per token (as a real top-k router produces), over all E (some not resident)
  std::vector<int> sel(T * K);
  for (int t = 0; t < T; t++)
    for (int j = 0; j < K; j++) {
      int e;
      do {
        e = rndi(E);
        for (int j2 = 0; j2 < j; j2++) e = sel[t * K + j2] == e ? -1 : e;
      } while (e < 0);
      sel[t * K + j] = e;
    }
  std::vector<float> wts(T * K);
  for (float &w : wts) w = f32_pow2(1 + rndi(3));
  const float lim = 4.f;
  // plan as quant_act builds it: distinct resident experts of sel in first-occurrence order, each token's last pick
  std::vector<int> plan(1 + 64 * PLAN_W);
  int n = 0;
  for (int i = 0; i < T * K && n < 64; i++)
    if (tab[E + sel[i]].gate != ~0ull) {
      bool first = true;
      for (int j = 0; j < i; j++) first &= sel[j] != sel[i];
      if (!first) continue;
      int *ent = &plan[1 + n * PLAN_W];
      ent[0] = sel[i];
      ent[1] = 0;
      for (int t = 0; t < 8; t++) {
        int jt = -1;
        if (t < T)
          for (int j = 0; j < K; j++)
            if (sel[t * K + j] == sel[i]) jt = j;
        ent[2 + t] = jt;
      }
      n++;
    }
  plan[0] = n;

  Act x(T, dim), acth(T * K, inter);
  Dev xq(x.q.size()), xd(x.d.size() * 4), xs(x.s.size() * 2), hq(acth.q.size()), hd(acth.d.size() * 4), hs(acth.s.size() * 2);
  xq.h2d(x.q.data(), x.q.size());
  xd.h2d(x.d.data(), x.d.size() * 4);
  xs.h2d(x.s.data(), x.s.size() * 2);
  hq.h2d(acth.q.data(), acth.q.size());
  hd.h2d(acth.d.data(), acth.d.size() * 4);
  hs.h2d(acth.s.data(), acth.s.size() * 2);
  Dev pland(plan.size() * 4), wtsd(wts.size() * 4), hh((size_t)T * K * inter * 4), out((size_t)T * dim * 4),
      part((size_t)64 * 8 * dim * 4), cnt(((dim + 15) / 16) * 4);
  pland.h2d(plan.data(), plan.size() * 4);
  wtsd.h2d(wts.data(), wts.size() * 4);
  CK(cuMemsetD8(cnt.p, 0, ((dim + 15) / 16) * 4));
  std::vector<float> hsent((size_t)T * K * inter, NAN), osent((size_t)T * dim, NAN);
  hh.h2d(hsent.data(), hsent.size() * 4);
  out.h2d(osent.data(), osent.size() * 4);
  {
    int layer = 1, Ei = E, Ki = K, in = inter, dm = dim;
    float lm = lim;
    void *tp = (void *)tabd.p, *pl = (void *)pland.p, *wp = (void *)wtsd.p, *st = (void *)store.p;
    void *a = (void *)xq.p, *b = (void *)xd.p, *c = (void *)xs.p, *hp = (void *)hh.p;
    std::vector<void *> args{&tp, &layer, &Ei, &pl, &wp, &Ki, &st, &a, &b, &c, &hp, &in, &dm, &lm};
    launch(F_GU, (unsigned)((inter + 15) / 16), T * K, 64, tile_smem(ty), args);
  }
  std::vector<float> hv((size_t)T * K * inter);
  hh.d2h(hv.data(), hv.size() * 4);
  double dh = 0;
  int exact = 0, tot = 0, untouched_ok = 1;
  for (int t = 0; t < T; t++)
    for (int j = 0; j < K; j++) {
      bool picked_resident = tab[E + sel[t * K + j]].gate != ~0ull;
      int u = -1;  // the entry that serves this (token, pick): ent[2 + t] == j
      for (int v = 0; v < n; v++)
        if (plan[1 + v * PLAN_W] == sel[t * K + j] && plan[1 + v * PLAN_W + 2 + t] == j) u = v;
      if (!picked_resident || u < 0) {  // never written: sentinel must survive
        for (int r = 0; r < inter; r++)
          if (!std::isnan(hv[(size_t)(t * K + j) * inter + r])) untouched_ok = 0;
        continue;
      }
      for (int r = 0; r < inter; r++) {
        double g = 0, up = 0;
        for (int c = 0; c < dim; c++) {
          g += wg[(size_t)r * dim + c] * x.val[(size_t)t * dim + c];
          up += wu[(size_t)r * dim + c] * x.val[(size_t)t * dim + c];
        }
        double gv = std::min(g, (double)lim), uv = std::min(std::max(up, -(double)lim), (double)lim);
        float ref = bf16r_h(gv / (1.0 + exp(-gv)) * uv * wts[t * K + j]);
        float got = hv[(size_t)(t * K + j) * inter + r];
        if (got == ref) exact++;
        tot++;
        dh = std::max(dh, fabs((double)got - ref) - std::max(fabs((double)ref) * 0.00390625, 1e-30));  // one bf16 ulp
      }
    }
  {
    int layer = 1, Ei = E, Ki = K, in = inter, dm = dim, Ti = T;
    void *tp = (void *)tabd.p, *pl = (void *)pland.p, *st = (void *)store.p;
    void *a = (void *)hq.p, *b = (void *)hd.p, *c = (void *)hs.p, *o = (void *)out.p, *pp = (void *)part.p, *cp = (void *)cnt.p;
    std::vector<void *> args{&tp, &layer, &Ei, &pl, &Ki, &st, &a, &b, &c, &o, &in, &dm, &Ti, &pp, &cp};
    launch(F_DOWN, (unsigned)((dim + 15) / 16), T * K, 64, tile_smem(ty), args);
  }
  std::vector<float> ov((size_t)T * dim);
  out.d2h(ov.data(), ov.size() * 4);
  double dou = 0;
  for (int t = 0; t < T; t++)
    for (int r = 0; r < dim; r++) {
      double ref = 0;
      for (int j = 0; j < K; j++)
        if (tab[E + sel[t * K + j]].gate != ~0ull)
          for (int c = 0; c < inter; c++) ref += wd[(size_t)r * inter + c] * acth.val[(size_t)(t * K + j) * inter + c];
      dou = std::max(dou, fabs((double)ov[(size_t)t * dim + r] - ref));
    }
  int p0 = 0;  // plan[0] == 0: everything zeroed
  pland.h2d(&p0, 4);
  CK(cuMemsetD8(cnt.p, 0, ((dim + 15) / 16) * 4));
  {
    int layer = 1, Ei = E, Ki = K, in = inter, dm = dim, Ti = T;
    void *tp = (void *)tabd.p, *pl = (void *)pland.p, *st = (void *)store.p;
    void *a = (void *)hq.p, *b = (void *)hd.p, *c = (void *)hs.p, *o = (void *)out.p, *pp = (void *)part.p, *cp = (void *)cnt.p;
    std::vector<void *> args{&tp, &layer, &Ei, &pl, &Ki, &st, &a, &b, &c, &o, &in, &dm, &Ti, &pp, &cp};
    launch(F_DOWN, (unsigned)((dim + 15) / 16), T * K, 64, tile_smem(ty), args);
  }
  out.d2h(ov.data(), ov.size() * 4);
  double dz = 0;
  for (float v : ov) dz = std::max(dz, fabs((double)v));
  printf("  moe    %-7s dim=%-5d inter=%-3d T=%d K=%d E=%d res=%d plan=%d  h max(err-1ulp) %.3g (%d/%d bit-exact, untouched %d)"
         "  down maxabs %.3g  empty-plan maxabs %.3g\n",
         ty_name(ty), dim, inter, T, K, E, nres, n, dh, exact, tot, untouched_ok, dou, dz);
  if (dh > 0 || dou != 0 || dz != 0 || !untouched_ok) g_fail++;
}

// ---------------------------------------------------------------- perf
static double bench_xgemv(int ty, int rows, int k, int T) {
  size_t mb = rp_bytes(ty, rows, k);
  int npool = (int)std::max<size_t>(2, (280ull << 20) / mb + 1);
  Dev pool((size_t)npool * mb);
  std::vector<uint8_t> gg;
  gen_gguf(gg, ty, rows, k);
  repack_to(pool.p, gg, ty, rows, k);
  for (int i = 1; i < npool; i++) CK(cuMemcpyDtoD(pool.p + (size_t)i * mb, pool.p, mb));
  Act x(T, k);
  Dev xq(x.q.size()), xd(x.d.size() * 4), xs(x.s.size() * 2), y((size_t)T * rows * 4);
  xq.h2d(x.q.data(), x.q.size());
  xd.h2d(x.d.data(), x.d.size() * 4);
  xs.h2d(x.s.data(), x.s.size() * 2);
  int a = rp_ty(ty), r = rows, kk = k;
  void *qp = (void *)xq.p, *dp = (void *)xd.p, *sp = (void *)xs.p, *yp = (void *)y.p;
  CUdeviceptr wcur = pool.p;  // cuLaunchKernel takes the ADDRESS of each parameter value
  std::vector<void *> args{&a, &wcur, &r, &kk, &T, &qp, &dp, &sp, &yp};
  CUevent e0, e1;
  CK(cuEventCreate(&e0, 0));
  CK(cuEventCreate(&e1, 0));
  int iters = 20;
  for (int rep = 0; rep < 3 + iters; rep++) {
    if (rep == 3) CK(cuEventRecord(e0, 0));
    wcur = pool.p + (size_t)(rep % npool) * mb;
    launch(F_XGEMV, (unsigned)((rows + 15) / 16), 1, 64, tile_smem(ty), args);
  }
  CK(cuEventRecord(e1, 0));
  CK(cuEventSynchronize(e1));
  float ms;
  CK(cuEventElapsedTime(&ms, e0, e1));
  double us = ms * 1000.0 / iters;
  printf("  xgemv %-7s k=%d rows=%d T=%d  %8.2f us/pass  %7.1f GB/s weight-stream (%.2f MB, pool %d)\n", ty_name(ty), k, rows, T, us,
         mb / us / 1e3, mb / 1e6, npool);
  return mb / us / 1e3;
}

static double bench_moe(int ty, int dim, int inter, int T, int K, int nent) {
  size_t mg = (rp_bytes(ty, inter, dim) + 15) & ~(size_t)15, md = (rp_bytes(ty, dim, inter) + 15) & ~(size_t)15, one = 2 * mg + md;
  Dev store((size_t)nent * one);
  CK(cuMemsetD8(store.p, 0, store.n));  // perf only: any bytes decode the same way
  {
    std::vector<uint8_t> z(gguf_bytes(ty, inter, dim), 0);
    repack_to(store.p, z, ty, inter, dim);
    CK(cuMemcpyDtoD(store.p + mg, store.p, mg));
    std::vector<uint8_t> z2(gguf_bytes(ty, dim, inter), 0);
    repack_to(store.p + 2 * mg, z2, ty, dim, inter);
  }
  for (int e = 1; e < nent; e++) CK(cuMemcpyDtoD(store.p + (size_t)e * one, store.p, one));
  std::vector<SlotH> tab(nent);
  for (int e = 0; e < nent; e++)
    tab[e] = {(unsigned long long)((size_t)e * one), (unsigned long long)((size_t)e * one + mg), (unsigned long long)((size_t)e * one + 2 * mg),
              rp_ty(ty), rp_ty(ty)};
  Dev tabd(tab.size() * sizeof(SlotH));
  tabd.h2d(tab.data(), tab.size() * sizeof(SlotH));
  std::vector<int> plan(1 + 64 * PLAN_W);
  plan[0] = nent;  // one entry per expert, each picked once: token u/K picks expert u at pick u%K
  for (int u = 0; u < nent; u++) {
    int *ent = &plan[1 + u * PLAN_W];
    ent[0] = u;
    ent[1] = 0;
    for (int t = 0; t < 8; t++) ent[2 + t] = -1;
    ent[2 + u / K] = u % K;
  }
  Act x(T, dim), acth(T * K, inter);
  Dev xq(x.q.size()), xd(x.d.size() * 4), xs(x.s.size() * 2), hq(acth.q.size()), hd(acth.d.size() * 4), hs(acth.s.size() * 2);
  xq.h2d(x.q.data(), x.q.size());
  xd.h2d(x.d.data(), x.d.size() * 4);
  xs.h2d(x.s.data(), x.s.size() * 2);
  hq.h2d(acth.q.data(), acth.q.size());
  hd.h2d(acth.d.data(), acth.d.size() * 4);
  hs.h2d(acth.s.data(), acth.s.size() * 2);
  Dev pland(plan.size() * 4), wtsd((size_t)T * K * 4), hh((size_t)T * K * inter * 4), out((size_t)T * dim * 4),
      part((size_t)64 * 8 * dim * 4), cnt(((dim + 15) / 16) * 4);
  pland.h2d(plan.data(), plan.size() * 4);
  CK(cuMemsetD8(wtsd.p, 0, (size_t)T * K * 4));
  CUevent e0, e1;
  CK(cuEventCreate(&e0, 0));
  CK(cuEventCreate(&e1, 0));
  int layer = 0, E = nent, Ki = K, in = inter, dm = dim, Ti = T;
  float lm = 1e30f;
  auto runGu = [&] {
    void *tp = (void *)tabd.p, *pl = (void *)pland.p, *wp = (void *)wtsd.p, *st = (void *)store.p;
    void *a = (void *)xq.p, *b = (void *)xd.p, *c = (void *)xs.p, *hp = (void *)hh.p;
    std::vector<void *> args{&tp, &layer, &E, &pl, &wp, &Ki, &st, &a, &b, &c, &hp, &in, &dm, &lm};
    launch(F_GU, (unsigned)((inter + 15) / 16), T * K, 64, tile_smem(ty), args);
  };
  auto runDown = [&] {
    CK(cuMemsetD8(cnt.p, 0, ((dm + 15) / 16) * 4));
    void *tp = (void *)tabd.p, *pl = (void *)pland.p, *st = (void *)store.p;
    void *a = (void *)hq.p, *b = (void *)hd.p, *c = (void *)hs.p, *o = (void *)out.p, *pp = (void *)part.p, *cp = (void *)cnt.p;
    std::vector<void *> args{&tp, &layer, &E, &pl, &Ki, &st, &a, &b, &c, &o, &in, &dm, &Ti, &pp, &cp};
    launch(F_DOWN, (unsigned)((dm + 15) / 16), T * K, 64, tile_smem(ty), args);
  };
  auto timeit = [&](auto &&fn) {
    int iters = 20;
    for (int rep = 0; rep < 3 + iters; rep++) {
      if (rep == 3) CK(cuEventRecord(e0, 0));
      fn();
    }
    CK(cuEventRecord(e1, 0));
    CK(cuEventSynchronize(e1));
    float ms;
    CK(cuEventElapsedTime(&ms, e0, e1));
    return ms * 1000.0 / iters;
  };
  double ug = timeit(runGu), ud = timeit(runDown);
  double gbytes = (double)nent * 2 * rp_bytes(ty, inter, dim), dbytes = (double)nent * rp_bytes(ty, dim, inter);
  printf("  moe    %-7s dim=%d inter=%d T=%d K=%d entries=%d  gu %8.1f us (%7.1f GB/s)  down %8.1f us (%7.1f GB/s)\n", ty_name(ty),
         dim, inter, T, K, nent, ug, gbytes / ug / 1e3, ud, dbytes / ud / 1e3);
  return std::min(gbytes / ug / 1e3, dbytes / ud / 1e3);
}

int main(int argc, char **argv) {
  std::string mode = argc > 1 ? argv[1] : "all";
  setvbuf(stdout, nullptr, _IONBF, 0);
  CK(cuInit(0));
  CUdevice dev;
  CK(cuDeviceGet(&dev, 0));
  CUcontext ctx;
  CK(cuCtxCreate(&ctx, 0, dev));
  char name[128];
  cuDeviceGetName(name, sizeof name, dev);
  int maj, min_;
  cuDeviceGetAttribute(&maj, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
  cuDeviceGetAttribute(&min_, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
  printf("%s (sm_%d%d), CUDA context up\n", name, maj, min_);
  CUmodule mod;
  {
    FILE *f = fopen(getenv("T5_FATBIN") ? getenv("T5_FATBIN") : (argc > 2 ? argv[2] : "bench/t5.fatbin"), "rb");
    if (!f) {
      fprintf(stderr, "fatbin not found (run from the repo root)\n");
      return 1;
    }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<char> img(n);
    if (fread(img.data(), 1, n, f) != (size_t)n) return 1;
    fclose(f);
    CK(cuModuleLoadData(&mod, img.data()));
    F_MOD = mod;
  }
  CK(cuModuleGetFunction(&F_REPACK, mod, "repack_expert"));
  CK(cuModuleGetFunction(&F_RPW, mod, "t5_rp_weight"));
  CK(cuModuleGetFunction(&F_TAB, mod, "t5_tables"));
  CK(cuModuleGetFunction(&F_XGEMV, mod, "xgemv"));
  CK(cuModuleGetFunction(&F_GU, mod, "moe_gu_fast"));
  CK(cuModuleGetFunction(&F_DOWN, mod, "moe_down_fast"));
  {
    Dev g64(3840 * 8), g8(128);
    void *a = (void *)g64.p, *b = (void *)g8.p;
    std::vector<void *> args{&a, &b};
    launch(F_TAB, 8, 1, 256, 0, args);
    TAB.g64.resize(3840);
    TAB.ksigns.resize(128);
    g64.d2h(TAB.g64.data(), 3840 * 8);
    g8.d2h(TAB.ksigns.data(), 128);
  }

  static const int tys[8] = {42, 143, 11, 8, 16, 19, 17, 22};
  if (mode == "moe") {  // t5_bench moe <ty> <dim> <inter> <T> <K> <E> <nres>
    check_moe(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), atoi(argv[6]), atoi(argv[7]), atoi(argv[8]));
    printf("%s\n", g_fail ? "FAIL" : "PASS");
    return g_fail != 0;
  }
  if (mode == "correct" || mode == "all") {
    printf("== correctness (dyadic scales: the fp32 path is exact, maxabs must be 0)\n");
    for (int ty : tys) {
      check_xgemv(ty, 16, 256, 8);   // smallest: one tile, one x-block
      check_xgemv(ty, 24, 512, 3);   // partial tiles, odd rows, T < 8, KS = 2 warps split 2 x-blocks
      check_xgemv(ty, 2048, 7168, 8);  // expert shape
    }
    for (int ty : tys) check_moe(ty, 512, 256, 8, 6, 16, 6);  // inter must be a multiple of 256 (it is the down k)
  }
  if (mode == "perf" || mode == "all") {
    printf("== perf (expert shapes k=7168 n=2048 T=8; gate >= 104 GB/s weight-stream = 0.5 x Phase-0 K1)\n");
    double worst = 1e9;
    for (int ty : tys) worst = std::min(worst, bench_xgemv(ty, 2048, 7168, 8));
    printf("  (xgemv at the expert shape runs 128 blocks of 64 threads: occupancy-limited, as on the SM80 target;\n"
           "   the weight-streaming role is moe_gu_fast / moe_down_fast below. Same kernel, more blocks:)\n");
    for (int ty : tys) bench_xgemv(ty, 16384, 7168, 8);
    double wg = 1e9;
    for (int ty : tys) wg = std::min(wg, bench_moe(ty, 7168, 2048, 8, 6, 48));
    printf("GATE: moe gu/down worst %.1f GB/s weight-stream (need >= 104 = 0.5 x Phase-0 K1); xgemv n=2048 worst %.1f GB/s\n", wg,
           worst);
    if (wg < 104) g_fail++;
  }
  printf("%s\n", g_fail ? "FAIL" : "PASS");
  return g_fail != 0;
}
