// prefill.h: many-token matrix products for the prompt path, on the decode weight tiles (FP8 dense with E8M0
// 32x32 block scales, MXFP4 experts, 3/2-bit codebook experts) x FP8 activations (codes [tok][K] + scales
// [tok][K/32]). A wave owns one 16-row tile; each weight fragment is decoded once per k-block and multiplied
// against up to MMG groups of 16 tokens (MFMA columns = tokens).
#pragma once
#include "gemv.h"

#define MMG 8  // token groups of 16 per block (128 tokens)
// batched-decode slot window rings: 16 spare rows, so a step may feed up to 16 consecutive positions of one slot
// (draft verification) without overwriting rows its earlier tokens still read
#define SRING (WIN + 16)

// Graph-replayed passes (verify, draft) pass p0 = -1: the position comes from the decode state on the device
__device__ const int* g_posp;
#define P0_DEV(p0) \
    if (p0 < 0) p0 = ld_dev_i(g_posp)

struct MmJob {
    const void* w;       // tiles (FP8: [rt][kpair][lane][16 B]; experts: arena)
    const uint8_t* sc;   // FP8: E8M0 [N/32][K/32]
    long long woff, soff; // experts: offsets in the arena; fmt below
    float mul;
    int fmt;             // 0 FP8 dense, 1 MXFP4, 2 CB3, 3 CB2
    int n, k;            // rows, cols
    const uint8_t* xq;   // activation codes [*][k]
    const float* xs;     // activation scales [*][k/32]
    const int* toks;     // token rows for the activation / output (null: identity from tok0)
    int tok0, ntok;      // this job's tokens: toks[tok0 .. tok0+ntok) (or rows tok0..)
    float* y;            // output [*][ldy]
    int ldy, yatomic;    // yatomic: += (scatter-add), else store
    const int* yrow;     // output row per token (null: same as activation row)
    int ldx;             // activation row stride (codes; scales use ldx/32)
    int pad;
    const uint16_t* xb;  // fmt 4 (BF16 weights): bf16 activation rows [*][ldx]
};

// FP8 activation fragment for MFMA column `c` of token group g: token row r (or -1 = zeros)
DEV uint64_t mm_xfrag(const uint8_t* xq, int ld, int row, int kb, int g8) {
    return row >= 0 ? *reinterpret_cast<const uint64_t*>(xq + (size_t)row * ld + kb * 32 + g8) : 0ull;
}

// One block: 4 waves = 4 row tiles (64 rows), up to 128 tokens; blocks enumerate (job, row-block, token-block).
// Block table: blk[b] = job index << 20 | rowblock << 8 | tokblock (host-built).
extern "C" __global__ void __launch_bounds__(256) mm(const MmJob* __restrict__ jobs, const int* __restrict__ blk,
                                                     uint32_t cb3lo, uint32_t cb3hi, uint32_t cb2) {
    const int code = blk[blockIdx.x];
    const MmJob J = jobs[code >> 20];  // job (10 bits) << 20 | row block (12) << 8 | token block (8)
    const int rb = (code >> 8) & 0xFFF, tb = code & 0xFF;
    const int lane = threadIdx.x & 63, wv = threadIdx.x >> 6;
    const int rt = rb * 4 + wv;
    if (rt * 16 >= J.n) return;
    const int t0 = tb * 16 * MMG;
    const int nt = min(16 * MMG, J.ntok - t0);
    if (nt <= 0) return;
    const int ng = (nt + 15) / 16;
    const int g8 = 8 * (lane >> 4), col = lane & 15;
    // this lane's token (MFMA column) per group
    int xrow[MMG];
#pragma unroll
    for (int g = 0; g < MMG; g++) {
        const int t = t0 + g * 16 + col;
        xrow[g] = (g < ng && t < J.ntok) ? (J.toks ? J.toks[J.tok0 + t] : J.tok0 + t) : -1;
    }
    f4 acc[MMG];
#pragma unroll
    for (int g = 0; g < MMG; g++) acc[g] = f4{0, 0, 0, 0};
    const int KB = J.k / 32;
    const int ldxs = J.ldx / 32;
    // rows used for loads (invalid columns read row 0 and are masked by a zero scale)
    int lrow[MMG];
    float msk[MMG];
#pragma unroll
    for (int g = 0; g < MMG; g++) { lrow[g] = xrow[g] >= 0 ? xrow[g] : 0; msk[g] = xrow[g] >= 0 ? 1.f : 0.f; }
    if (J.fmt == 0) {
        const u4v* wp = (const u4v*)J.w + (size_t)rt * (J.k / 64) * 64 + lane;
        const uint8_t* sp = J.sc + (size_t)(rt >> 1) * KB;
        const int KP = J.k / 64;
        u4v wc = __builtin_nontemporal_load(wp), wn;
        uint64_t xc[2][MMG];
        float sc_[2][MMG];
        auto ldx = [&](int kb, uint64_t* x, float* sv) {
#pragma unroll
            for (int g = 0; g < MMG; g++) {
                x[g] = *reinterpret_cast<const uint64_t*>(J.xq + (size_t)lrow[g] * J.ldx + kb * 32 + g8);
                sv[g] = J.xs[(size_t)lrow[g] * ldxs + kb] * msk[g];
            }
        };
        ldx(0, xc[0], sc_[0]);
        ldx(1, xc[1], sc_[1]);
        for (int kp = 0; kp < KP; kp++) {
            if (kp + 1 < KP) wn = __builtin_nontemporal_load(wp + (size_t)(kp + 1) * 64);
            uint64_t xn[2][MMG];
            float sn[2][MMG];
            if (kp + 1 < KP) { ldx(2 * kp + 2, xn[0], sn[0]); ldx(2 * kp + 3, xn[1], sn[1]); }
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const int kb = 2 * kp + h;
                const uint64_t a = h ? ((uint64_t)wc.w << 32 | wc.z) : ((uint64_t)wc.y << 32 | wc.x);
                const float ws = __uint_as_float((uint32_t)sp[kb] << 23) * 2.f;
#pragma unroll
                for (int g = 0; g < MMG; g++) {
                    if (g < ng) {
                        const f4 d = mfma_fp8(a, xc[h][g], f4{0, 0, 0, 0});
                        acc[g] += d * (ws * sc_[h][g]);
                    }
                }
            }
            wc = wn;
#pragma unroll
            for (int h = 0; h < 2; h++)
#pragma unroll
                for (int g = 0; g < MMG; g++) { xc[h][g] = xn[h][g]; sc_[h][g] = sn[h][g]; }
        }
    } else if (J.fmt == 4) {
        // BF16 weights (tile 16 rows x 32 k) x bf16 activations
        const u4v* wp = (const u4v*)J.w + (size_t)rt * (J.k / 32) * 64 + lane;
        const int g4 = 4 * (lane >> 4);
        for (int kq = 0; kq < J.k / 32; kq++) {
            const u4v w = __builtin_nontemporal_load(wp + (size_t)kq * 64);
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const uint32_t w0 = h ? w.z : w.x, w1 = h ? w.w : w.y;
                s4v a;
                a[0] = (short)(w0 & 0xFFFF); a[1] = (short)(w0 >> 16); a[2] = (short)(w1 & 0xFFFF); a[3] = (short)(w1 >> 16);
#pragma unroll
                for (int g = 0; g < MMG; g++) {
                    if (g < ng) {
                        uint64_t bv = 0;
                        if (xrow[g] >= 0) bv = *reinterpret_cast<const uint64_t*>(J.xb + (size_t)xrow[g] * J.ldx + kq * 32 + 16 * h + g4);
                        acc[g] = mfma_bf16(a, *reinterpret_cast<const s4v*>(&bv), acc[g]);
                    }
                }
            }
        }
    } else {
        // experts: per-row scales, rows 4*(lane/16)+j; lane needs its 4 rows' scale bytes per k-block
        const char* base = (const char*)J.w;
        const u4v* sp = (const u4v*)(base + J.soff) + (size_t)rt * (J.k / 128) * 4 + (lane >> 4);
        const int NDW = J.fmt == 1 ? 4 : (J.fmt == 2 ? 3 : 2);
        const uint32_t* wp = (const uint32_t*)(base + J.woff) + (size_t)rt * (J.k / 128) * 64 * NDW + lane * NDW;
        const int KU = J.k / 128;
        uint32_t cw[4] = {0, 0, 0, 0}, nw[4] = {0, 0, 0, 0};
        u4v cs, ns = {0, 0, 0, 0};
        for (int q = 0; q < NDW; q++) cw[q] = __builtin_nontemporal_load(wp + q);
        cs = sp[0];
        for (int u = 0; u < KU; u++) {
            if (u + 1 < KU) {
                for (int q = 0; q < NDW; q++) nw[q] = __builtin_nontemporal_load(wp + (size_t)(u + 1) * 64 * NDW + q);
                ns = sp[(size_t)(u + 1) * 4];
            }
            const uint32_t ss[4] = {cs.x, cs.y, cs.z, cs.w};
#pragma unroll
            for (int m = 0; m < 4; m++) {
                uint64_t a;
                if (J.fmt == 1) a = fp4x8_to_fp8(cw[m]);
                else {
                    const uint32_t tlo = J.fmt == 2 ? cb3lo : cb2, thi = J.fmt == 2 ? cb3hi : 0u;
                    const uint32_t dl = cw[m >> 1];
                    const int sh = 4 * (m & 1);
                    uint32_t s0 = (dl >> sh) & 0x03030303u, s1 = (dl >> (sh + 2)) & 0x03030303u;
                    if (J.fmt == 2) {
                        s0 |= ((cw[2] >> (2 * m)) & 0x01010101u) << 2;
                        s1 |= ((cw[2] >> (2 * m + 1)) & 0x01010101u) << 2;
                    }
                    a = (uint64_t)perm(thi, tlo, s1) << 32 | perm(thi, tlo, s0);
                }
                const int kb = 4 * u + m;
                uint64_t xf[MMG];
                float xsv[MMG];
#pragma unroll
                for (int g = 0; g < MMG; g++) {
                    xf[g] = *reinterpret_cast<const uint64_t*>(J.xq + (size_t)lrow[g] * J.ldx + kb * 32 + g8);
                    xsv[g] = J.xs[(size_t)lrow[g] * ldxs + kb] * msk[g];
                }
                float wsj[4];
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    const uint32_t b = (ss[j] >> (8 * m)) & 0xFFu;
                    wsj[j] = (J.fmt == 1 ? __uint_as_float(b << 23) : fp8_to_f32(b, 0)) * J.mul;
                }
#pragma unroll
                for (int g = 0; g < MMG; g++) {
                    if (g < ng) {
                        const f4 d = mfma_fp8(a, xf[g], f4{0, 0, 0, 0});
#pragma unroll
                        for (int j = 0; j < 4; j++) acc[g][j] += d[j] * (wsj[j] * xsv[g]);
                    }
                }
            }
            for (int q = 0; q < 4; q++) cw[q] = nw[q];
            cs = ns;
        }
    }
    // lane holds rows 16*rt + 4*(lane/16) + j for token column `col` of each group
    const int r0 = rt * 16 + 4 * (lane >> 4);
#pragma unroll
    for (int g = 0; g < MMG; g++) {
        if (xrow[g] < 0) continue;
        const int t = t0 + g * 16 + col;
        const int orow = J.yrow ? J.yrow[J.tok0 + t] : xrow[g];
        float* yp = J.y + (size_t)orow * J.ldy + r0;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            if (J.yatomic) atomicAdd(yp + j, acc[g][j]);
            else yp[j] = acc[g][j];
        }
    }
}

// ------------------------------------------------------------------ per-token kernels of the prompt path

// Activation rows for the many-token GEMMs, one block per token row r:
//   mode 0: rmsnorm(x_raw[r], nw) with sum of squares from NPART partials -> bf16
//   mode 1: rmsnorm over the row itself (qr) -> bf16
//   mode 2: bf16(src)                                     (wo_b input)
//   mode 3: bf16(silu(min(g,lim)) * clamp(u) * wt[r])     (expert down input; wt null = 1)
//   mode 4: src as is (already final)
// Writes FP8 codes/scales [r][n] and (ob) bf16 [r][n].
struct ZeroList {
    float* p[6];
    int n[6];
};
extern "C" __global__ void __launch_bounds__(256) prep_t(int mode, const float* __restrict__ src, const float* __restrict__ src2,
                                                         int lds, const float* nw, const float* part, const float* wt,
                                                         int n, float eps, float lim, uint8_t* oq, float* os, uint16_t* ob,
                                                         uint16_t* obq, ZeroList zl) {
    __shared__ float red[16];
    {
        // clear the next products' k-split outputs (all blocks, grid-stride), saving their own zeroing launches
        const int gb = blockIdx.y * gridDim.x + blockIdx.x, nb = gridDim.x * gridDim.y;
        for (int z = 0; z < 6; z++)
            for (int i = gb * 256 + threadIdx.x; i < zl.n[z]; i += nb * 256) zl.p[z][i] = 0.f;
    }
    const int r = blockIdx.x;
    const float* x = src + (size_t)r * lds;
    const float* x2 = src2 ? src2 + (size_t)r * lds : nullptr;
    float sc = 1.f;
    if (mode == 0) {
        // the NPART per-block partial sums of squares: one load per lane and a wave sum (each wave on its own; a
        // serial loop of 80 loads per thread was the latency chain of this small kernel)
        const int ln = threadIdx.x & 63;
        float ss = 0;
        for (int b = ln; b < NPART; b += 64) ss += part[(size_t)r * NPART + b];
        ss = wave_sum(ss);
        sc = 1.f / sqrtf(ss / n + eps);
    } else if (mode == 1) {
        float ss = 0;
        for (int i = threadIdx.x; i < n; i += 256) ss += x[i] * x[i];
        ss = block_sum(ss, red);
        sc = 1.f / sqrtf(ss / n + eps);
    }
    const float w = (mode == 3 && wt) ? wt[r] : 1.f;
    // blockIdx.y: 1024-wide slice of the row (n is a multiple of 32, so FP8 scale groups never straddle slices)
    const int c0 = blockIdx.y * 1024, nc = min(1024, n - c0);
    if (nc <= 0) return;
    write_act(nc, [&](int i) -> float {
        i += c0;
        const float v = x[i];
        switch (mode) {
            case 0:
            case 1: return round_bf16(nw[i] * (v * sc));
            case 2: return round_bf16(v);
            case 3: {
                const float g = fminf(v, lim), u = fminf(fmaxf(x2[i], -lim), lim);
                return round_bf16(g / (1.f + expf(-g)) * u * w);
            }
            default: return v;
        }
    }, oq + (size_t)r * n + c0, os + (size_t)r * (n / 32) + c0 / 32, ob ? ob + (size_t)r * n + c0 : nullptr, obq ? obq + (size_t)r * n + c0 : nullptr);
}

// Routed and shared SwiGLU rows share one launch. Each block retains prep_t's
// exact row/slice arithmetic and write_act quantization.
extern "C" __global__ void __launch_bounds__(256) prep_swiglu_pair_t(
    int routed, const float* gate, const float* up, const float* weights,
    uint8_t* codes, float* scales, uint16_t* bf, uint16_t* bq,
    const float* shared_gate, const float* shared_up,
    uint8_t* shared_codes, float* shared_scales, uint16_t* shared_bf, uint16_t* shared_bq,
    float lim) {
    int r = blockIdx.x;
    const bool shared = r >= routed;
    if (shared) {
        r -= routed; gate = shared_gate; up = shared_up;
        codes = shared_codes; scales = shared_scales; bf = shared_bf; bq = shared_bq;
    }
    const float w = shared ? 1.f : weights[r];
    const int c0 = blockIdx.y * 1024, nc = min(1024, INTER - c0);
    const float* x = gate + (size_t)r * INTER;
    const float* x2 = up + (size_t)r * INTER;
    write_act(nc, [&](int i) -> float {
        i += c0;
        const float g = fminf(x[i], lim), u = fminf(fmaxf(x2[i], -lim), lim);
        return round_bf16(g / (1.f + expf(-g)) * u * w);
    }, codes + (size_t)r * INTER + c0, scales + (size_t)r * (INTER / 32) + c0 / 32,
       bf ? bf + (size_t)r * INTER + c0 : nullptr,
       bq ? bq + (size_t)r * INTER + c0 : nullptr);
}

// RoPE on the last 64 dims of each head for a batch: x [T][heads][hd], position p0 + t (inverse: conjugate)
extern "C" __global__ void rope_t(float* x, int heads, int hd, const float* freqs, int p0, int inverse) {
    P0_DEV(p0);
    const int t = blockIdx.x, h = blockIdx.y, i = threadIdx.x;  // 32 pairs
    float* v = x + ((size_t)t * heads + h) * hd + hd - RD;
    const float ang = (float)(p0 + t) * freqs[i];
    float sn = sinf(ang), cs = cosf(ang);
    if (inverse) sn = -sn;
    const float a = v[2 * i], b = v[2 * i + 1];
    v[2 * i] = a * cs - b * sn;
    v[2 * i + 1] = a * sn + b * cs;
}

// Window kv rows of a batch: kv_raw [T][512] -> rmsnorm, rope(p0+t), fake_fp8 -> lin[base + t] (bf16)
extern "C" __global__ void __launch_bounds__(512) kv_t(const float* kv_raw, const float* kv_norm, const float* rf, int p0, uint16_t* lin, int base) {
    P0_DEV(p0);
    __shared__ float red[16];
    const int t = blockIdx.x, i = threadIdx.x;
    float v = kv_raw[(size_t)t * HD + i];
    const float ss = block_sum(v * v, red);
    v = kv_norm[i] * (v * (1.f / sqrtf(ss / HD + 1e-20f)));
    const float other = __shfl_xor(v, 1);
    if (i >= HD - RD) {
        const int p = (i - (HD - RD)) / 2;
        const float ang = (float)(p0 + t) * rf[p];
        const float sn = sinf(ang), cs = cosf(ang);
        v = (i & 1) ? other * sn + v * cs : v * cs - other * sn;
    }
    v = fake_fp8_lane(v);
    lin[(size_t)(base + t) * HD + i] = (uint16_t)bf16_bits(v);
}

// Compressor + indexer keys for a batch (one block per token, 512 threads): latents at positions p with
// (p+1) % ratio == 0, pooled with the previous position (from the batch or from cstate slot 0); the batch's last
// even position is left in cstate slot 0 for the next batch. Appends idx keys and compressed rows at (p+1)/ratio-1.
extern "C" __global__ void __launch_bounds__(512) comp_t(int p0, int T, int ratio, const float* ckv, const float* cgate,
                                                         float* cstate, const float* comp_norm, const uint16_t* wk,
                                                         const float* knorm, uint16_t* comp_cache, uint16_t* idx_cache,
                                                         const float* rope_fc, float eps) {
    P0_DEV(p0);
    __shared__ float red[16];
    __shared__ float lat[HD];
    __shared__ float kk[ID];
    const int t = blockIdx.x, i = threadIdx.x, p = p0 + t;
    if (ratio == 2 && t == T - 1 && (p + 1) % 2 != 0) {
        cstate[i] = ckv[(size_t)t * HD + i];
        cstate[HD + i] = cgate[(size_t)t * HD + i];
    }
    if ((p + 1) % ratio != 0) return;
    float v;
    if (ratio == 1) {
        v = ckv[(size_t)t * HD + i];
    } else {
        const float k0 = t > 0 ? ckv[(size_t)(t - 1) * HD + i] : cstate[i];
        const float s0 = t > 0 ? cgate[(size_t)(t - 1) * HD + i] : cstate[HD + i];
        const float k1 = ckv[(size_t)t * HD + i], s1 = cgate[(size_t)t * HD + i];
        const float mx = fmaxf(s0, s1);
        const float e0 = expf(s0 - mx), e1 = expf(s1 - mx);
        const float den = e0 + e1;
        v = round_bf16(k0 * e0 / den + k1 * e1 / den);
    }
    const float ss = block_sum(v * v, red);
    lat[i] = comp_norm[i] * (v * (1.f / sqrtf(ss / HD + eps)));
    __syncthreads();
    const int n = (p + 1) / ratio - 1, cpos = p + 1 - ratio;
    {
        const int o = i >> 2, q = i & 3;
        float sacc = 0;
        for (int j = q; j < HD; j += 4) sacc += bf16_to_f32(wk[o * HD + j]) * lat[j];
        sacc += __shfl_xor(sacc, 1);
        sacc += __shfl_xor(sacc, 2);
        if (q == 0) kk[o] = sacc;
    }
    __syncthreads();
    if (i < ID) {
        const float kv = kk[i];
        const float ssum = wave_sum(kv * kv);
        if ((i & 63) == 0) red[8 + (i >> 6)] = ssum;
    }
    __syncthreads();
    if (i < ID) {
        const float sk = red[8] + red[9];
        float x = knorm[i] * (kk[i] * (1.f / sqrtf(sk / ID + eps)));
        const float other = __shfl_xor(x, 1);
        if (i >= ID - RD) {
            const int pp = (i - (ID - RD)) / 2;
            const float ang = (float)cpos * rope_fc[pp];
            const float sn = sinf(ang), cs = cosf(ang);
            x = (i & 1) ? other * sn + x * cs : x * cs - other * sn;
        }
        x = fake_fp4_lane<32>(x, false);
        idx_cache[(size_t)n * ID + i] = (uint16_t)bf16_bits(x);
    }
    {
        float x = lat[i];
        const float other = __shfl_xor(x, 1);
        if (i >= HD - RD) {
            const int pp = (i - (HD - RD)) / 2;
            const float ang = (float)cpos * rope_fc[pp];
            const float sn = sinf(ang), cs = cosf(ang);
            x = (i & 1) ? other * sn + x * cs : x * cs - other * sn;
        }
        x = fake_fp4_lane<16>(x, true);
        comp_cache[(size_t)n * HD + i] = (uint16_t)bf16_bits(x);
    }
}

// Indexer scores for a batch of queries: block (key chunk, query q): score[q][s] for s < end_q (or the query's
// candidate rows rows[q][..nrows[q]]). end_q = (p0 + q0 + q + 1) / ratio.
extern "C" __global__ void __launch_bounds__(256) iscore_t(int p0, int q0, int ratio, const float* iq, const float* iw, float wscale,
                                                           const uint16_t* __restrict__ kc, const int* rows, const int* nrows,
                                                           int ldrows, float* score, int lds) {
    P0_DEV(p0);
    __shared__ uint16_t qs[IH][ID + 8];
    __shared__ float ws[IH];
    const int qi = blockIdx.y, tq = q0 + qi;
    const int end = rows ? nrows[tq] : (p0 + tq + 1) / ratio;
    const int tid = threadIdx.x;
    const int chunk0 = blockIdx.x * 1024;
    if (chunk0 >= end) return;
    for (int i = tid; i < IH * ID; i += 256) {
        const float v = fake_fp4_lane<32>(iq[(size_t)tq * IH * ID + i], false);
        qs[i / ID][i % ID] = (uint16_t)bf16_bits(v);
    }
    if (tid < IH) ws[tid] = round_bf16(iw[(size_t)tq * IH + tid] * wscale);
    __syncthreads();
    const int lane = tid & 63, g = lane >> 4, c = lane & 15;
    const int* rw = rows ? rows + (size_t)tq * ldrows : nullptr;
    float* out = score + (size_t)qi * lds;
    for (int base = chunk0 + (tid >> 6) * 16; base < min(end, chunk0 + 1024); base += 64) {
        const int sidx = base + c;
        const int row = sidx < end ? (rw ? rw[sidx] : sidx) : -1;
        f4 acc0 = {0, 0, 0, 0}, acc1 = {0, 0, 0, 0};
#pragma unroll
        for (int kq = 0; kq < ID / 16; kq++) {
            s4v a = {0, 0, 0, 0};
            if (row >= 0) a = *reinterpret_cast<const s4v*>(kc + (size_t)row * ID + kq * 16 + 4 * g);
            const s4v b0 = *reinterpret_cast<const s4v*>(&qs[c][kq * 16 + 4 * g]);
            const s4v b1 = *reinterpret_cast<const s4v*>(&qs[16 + c][kq * 16 + 4 * g]);
            acc0 = mfma_bf16(a, b0, acc0);
            acc1 = mfma_bf16(a, b1, acc1);
        }
        float pj[4];
        for (int j = 0; j < 4; j++) pj[j] = group_sum<16>(fmaxf(acc0[j], 0.f) * ws[c] + fmaxf(acc1[j], 0.f) * ws[16 + c]);
        if (c == 0)
            for (int j = 0; j < 4; j++) {
                const int sj = base + 4 * g + j;
                if (sj < end) out[sj] = (rw && rw[sj] < 0) ? -INFINITY : pj[j];
            }
    }
}

// In-block top-k (k <= 2048) of n scores (score -inf = invalid), 1024 threads: radix on the top 12 bits, then a
// bitonic sort of the threshold bin's candidates (largest score, then lowest index). out[i] = map ? map[idx] : idx,
// plus offset; -1 padding to kmax; sorted ascending if `sorted`. Returns count in *nout.
DEV void block_topk(const float* sc, int n, int k, const int* map, int offset, int* out, int kmax, int* nout, bool sorted) {
    __shared__ uint32_t hist[4096];
    __shared__ int sel[2048];
    __shared__ uint64_t cand[2048];
    __shared__ int nsel, ncand, bin, need, total;
    const int tid = threadIdx.x;
    for (int i = tid; i < 4096; i += 1024) hist[i] = 0;
    if (tid == 0) { nsel = 0; ncand = 0; }
    __syncthreads();
    for (int i = tid; i < n; i += 1024) {
        const float v = sc[i];
        if (v != -INFINITY) atomicAdd(&hist[fkey(v) >> 20], 1u);
    }
    __syncthreads();
    {
        // the bin where the count from the top reaches k: thread tid owns bins 4*tid..4*tid+3, suffix scan over threads
        // (cand doubles as scan scratch; it is filled only after this)
        uint32_t* tsum = reinterpret_cast<uint32_t*>(cand);
        const uint32_t own = hist[4 * tid] + hist[4 * tid + 1] + hist[4 * tid + 2] + hist[4 * tid + 3];
        tsum[tid] = own;
        __syncthreads();
        for (int d = 1; d < 1024; d <<= 1) {
            const uint32_t v = tid + d < 1024 ? tsum[tid + d] : 0u;
            __syncthreads();
            tsum[tid] += v;
            __syncthreads();
        }
        const uint32_t incl = tsum[tid], above = incl - own;  // bins above this thread's range
        if (tid == 0 && incl < (uint32_t)k) {
            bin = -1;  // fewer than k valid
            need = 0;
            total = (int)incl;
        }
        if (above < (uint32_t)k && incl >= (uint32_t)k) {
            uint32_t acc = above;
            int b = 4 * tid + 3;
            for (; b > 4 * tid; b--) {
                if (acc + hist[b] >= (uint32_t)k) break;
                acc += hist[b];
            }
            bin = b;
            need = k - (int)acc;
            total = k;
        }
        __syncthreads();
    }
    for (int i = tid; i < n; i += 1024) {
        const float v = sc[i];
        if (v == -INFINITY) continue;
        const uint32_t key = fkey(v);
        const int b = key >> 20;
        if (b > bin) sel[atomicAdd(&nsel, 1)] = i;
        else if (b == bin) {
            const int pidx = atomicAdd(&ncand, 1);
            if (pidx < 2048) cand[pidx] = (uint64_t)key << 32 | (uint32_t)(~i);
        }
    }
    __syncthreads();
    const int m = min(ncand, 2048);
    int P = 1;
    while (P < m) P <<= 1;
    for (int i = m + tid; i < P; i += 1024) cand[i] = 0;
    __syncthreads();
    for (int sz = 2; sz <= P; sz <<= 1)
        for (int st2 = sz >> 1; st2 > 0; st2 >>= 1) {
            for (int i = tid; i < P; i += 1024) {
                const int j = i ^ st2;
                if (j > i) {
                    const bool desc = (i & sz) == 0;
                    const uint64_t a = cand[i], b = cand[j];
                    if (desc ? a < b : a > b) { cand[i] = b; cand[j] = a; }
                }
            }
            __syncthreads();
        }
    const int ns = nsel;
    for (int i = tid; i < need && i < m; i += 1024) sel[ns + i] = (int)(~(uint32_t)(cand[i] & 0xFFFFFFFFu));
    __syncthreads();
    const int kk = total;
    if (sorted) {
        int P2 = 1;
        while (P2 < kk) P2 <<= 1;
        for (int i = kk + tid; i < P2; i += 1024) sel[i] = 0x7FFFFFFF;
        __syncthreads();
        for (int sz = 2; sz <= P2; sz <<= 1)
            for (int st2 = sz >> 1; st2 > 0; st2 >>= 1) {
                for (int i = tid; i < P2; i += 1024) {
                    const int j = i ^ st2;
                    if (j > i) {
                        const bool asc = (i & sz) == 0;
                        const int a = sel[i], b = sel[j];
                        if (asc ? a > b : a < b) { sel[i] = b; sel[j] = a; }
                    }
                }
                __syncthreads();
            }
    }
    for (int i = tid; i < kmax; i += 1024) out[i] = i < kk ? (map ? map[sel[i]] : sel[i]) + offset : -1;
    if (tid == 0) *nout = kk;
    __syncthreads();
}

// top-k per query of a batch (block = query): scores [q][lds]; n = end_q, or nrows[q] with map rows[q][..]
extern "C" __global__ void __launch_bounds__(1024) itopk_t(int p0, int q0, int ratio, const float* score, int lds, const int* rows,
                                                           const int* nrows, int ldrows, int k, int offset, int* out, int ldo,
                                                           int* nout, int sorted) {
    P0_DEV(p0);
    const int qi = blockIdx.x, tq = q0 + qi;
    const int n = rows ? nrows[tq] : (p0 + tq + 1) / ratio;
    block_topk(score + (size_t)qi * lds, n, k, rows ? rows + (size_t)tq * ldrows : nullptr, offset, out + (size_t)tq * ldo, k, nout + tq, sorted != 0);
}

// candidate layer, per query: block maxima over 8 positions (newest block pinned), top-2048 blocks (ascending),
// expanded to positions -> keep[q][16384], nkeep[q]; scores [q][lds] in, block scores scratch bsc [q][lds/8]
extern "C" __global__ void __launch_bounds__(1024) icand_t(int p0, int q0, int ratio, int bs, int nblk, const float* score, int lds,
                                                           float* bsc, int* kb, int* keep, int* nkeep) {
    P0_DEV(p0);
    __shared__ int nb_s;
    const int qi = blockIdx.x, tq = q0 + qi;
    const int end = (p0 + tq + 1) / ratio;
    const int nb = (end + bs - 1) / bs;
    const float* sc = score + (size_t)qi * lds;
    float* b = bsc + (size_t)qi * (lds / bs + 1);
    for (int i = threadIdx.x; i < nb; i += 1024) {
        float m = -INFINITY;
        for (int j = i * bs; j < min((i + 1) * bs, end); j++) m = fmaxf(m, sc[j]);
        b[i] = (i == (end - 1) / bs) ? INFINITY : m;
    }
    __syncthreads();
    int* kbq = kb + (size_t)qi * nblk;
    block_topk(b, nb, nblk, nullptr, 0, kbq, nblk, &nb_s, true);
    const int n = nb_s;
    int* kq = keep + (size_t)tq * nblk * bs;
    for (int i = threadIdx.x; i < n * bs; i += 1024) {
        const int blk = kbq[i / bs], pp = blk * bs + i % bs;
        kq[i] = pp < end ? pp : -1;
    }
    if (threadIdx.x == 0) nkeep[tq] = n * bs;
}

// Sparse attention for a batch: block (query t, head group of 16), 256 threads; rows = window positions
// p-127..p (lin rows: position - (p0 - WIN)) then the query's compressed rows (cidx[t][..ncomp[t]], -1 skipped,
// minus woff). Online softmax over chunks of 16 rows. Writes FP8 codes/scales of round_bf16(o) after inverse rope.
extern "C" __global__ void __launch_bounds__(256) attn_t(int p0, const float* __restrict__ q, const uint16_t* __restrict__ lin,
                                                         const uint16_t* __restrict__ comp, const int* __restrict__ cidx,
                                                         const int* ncomp, int ldc, int woff, float scale, const float* sinks,
                                                         const float* rf, uint8_t* oq, float* os, uint16_t* obq) {
    P0_DEV(p0);
    __shared__ uint16_t rows[16][HD + 8];
    __shared__ float p[16][16];
    __shared__ int rowsrc[16];
    __shared__ float ob[16][HD + 4];
    const int t = blockIdx.x, g = blockIdx.y, tid = threadIdx.x;
    const int h = tid >> 4, qs = tid & 15;
    const int pos = p0 + t;
    float4 qv[8];
    {
        const float* qp = q + ((size_t)t * NH + g * 16 + h) * HD + 4 * qs;
#pragma unroll
        for (int u = 0; u < 8; u++) qv[u] = *reinterpret_cast<const float4*>(qp + 64 * u);
    }
    const int lo = max(0, pos - (WIN - 1));
    const int nwin = pos - lo + 1;
    const int nc = comp ? ncomp[t] : 0;
    const int total = nwin + nc;
    float M = -INFINITY, L = 0.f;
    float acc[32];
#pragma unroll
    for (int kk = 0; kk < 32; kk++) acc[kk] = 0.f;
    for (int r0 = 0; r0 < total; r0 += 16) {
        __syncthreads();
        if (tid < 16) {
            const int rr = r0 + tid;
            int src = -1;
            if (rr < nwin) src = (lo + rr) - (p0 - WIN);
            else if (rr < total) {
                const int ci = cidx[(size_t)t * ldc + rr - nwin];
                src = ci >= 0 ? 0x40000000 | (ci - woff) : -1;
            }
            rowsrc[tid] = src;
        }
        __syncthreads();
        for (int i = tid; i < 16 * HD / 8; i += 256) {
            const int rw = i / (HD / 8), j = (i % (HD / 8)) * 8;
            const int src = rowsrc[rw];
            uint4 v = {0, 0, 0, 0};
            if (src >= 0) {
                const uint16_t* bp = (src & 0x40000000) ? comp + (size_t)(src & 0x3FFFFFFF) * HD : lin + (size_t)src * HD;
                v = *reinterpret_cast<const uint4*>(bp + j);
            }
            *reinterpret_cast<uint4*>(&rows[rw][j]) = v;
        }
        __syncthreads();
        float mysc = 0;
#pragma unroll
        for (int r = 0; r < 16; r++) {
            const uint16_t* rp = &rows[r][4 * qs];
            float sacc = 0;
#pragma unroll
            for (int u = 0; u < 8; u++) {
                const uint2 b = *reinterpret_cast<const uint2*>(rp + 64 * u);
                sacc += qv[u].x * __uint_as_float(b.x << 16) + qv[u].y * __uint_as_float(b.x & 0xFFFF0000u) +
                        qv[u].z * __uint_as_float(b.y << 16) + qv[u].w * __uint_as_float(b.y & 0xFFFF0000u);
            }
            sacc = group_sum<16>(sacc);
            if (qs == r) mysc = sacc;
        }
        const float v = (rowsrc[qs] >= 0 && r0 + qs < total) ? mysc * scale : -INFINITY;
        const float mc = group_max<16>(v);
        const float Mn = fmaxf(M, mc);
        const float a = M == -INFINITY ? 0.f : expf(M - Mn);
        const float e = v == -INFINITY ? 0.f : expf(v - Mn);
        L = L * a + group_sum<16>(e);
        M = Mn;
        p[h][qs] = e;
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < 32; kk++) {
            const int j = qs + 16 * kk;
            float sacc = 0;
#pragma unroll
            for (int r = 0; r < 16; r++) sacc += p[h][r] * __uint_as_float((uint32_t)rows[r][j] << 16);
            acc[kk] = acc[kk] * a + sacc;
        }
    }
    const float Lt = L + expf(sinks[g * 16 + h] - M);
#pragma unroll
    for (int kk = 0; kk < 32; kk++) ob[h][qs + 16 * kk] = round_bf16(acc[kk] / Lt);
    __syncthreads();
    // inverse rope on the last 64 dims of each head, then FP8 per 32 along the head
    if (qs < 2) {
        for (int i = qs * 16; i < qs * 16 + 16; i++) {
            const float ang = (float)pos * rf[i];
            const float sn = -sinf(ang), cs = cosf(ang);
            const float x = ob[h][HD - RD + 2 * i], y2 = ob[h][HD - RD + 2 * i + 1];
            ob[h][HD - RD + 2 * i] = x * cs - y2 * sn;
            ob[h][HD - RD + 2 * i + 1] = x * sn + y2 * cs;
        }
    }
    __syncthreads();
    // 16 heads x 512 = 8192 values; 4 per thread-step, 8 threads per 32-block (quant32)
    uint8_t* oqt = oq + ((size_t)t * NH + g * 16) * HD;
    float* ost = os + ((size_t)t * NH + g * 16) * HD / 32;
    uint16_t* obt = obq + ((size_t)t * NH + g * 16) * HD;
    write_act(16 * HD, [&](int i) -> float { return ob[i / HD][i % HD]; }, oqt, ost, nullptr, obt);
}

// Router for a batch: one wave per token: top-6 -> sel[t][6], wts[t][6]; counts[e] += 1
extern "C" __global__ void route_t(const float* logits, const float* bias, float scale, int* sel, float* wts, int* counts) {
    __shared__ Route r;
    const int t = blockIdx.x, lane = threadIdx.x;
    route_top6(logits + (size_t)t * NE, bias, scale, &r, lane);
    __builtin_amdgcn_wave_barrier();
    if (lane < 6) {
        sel[t * 6 + lane] = r.ids[lane];
        wts[t * 6 + lane] = r.wts[lane];
        atomicAdd(&counts[r.ids[lane]], 1);
    }
}

// Expert lists: for (t, slot): pos = off[e] + atomicAdd(fill[e]); tok[pos] = t, wrow[pos] = wts[t][slot]
extern "C" __global__ void route_fill_t(int T, const int* sel, const float* wts, const int* off, int* fill, int* tok, float* wrow) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= T * 6) return;
    const int e = sel[i];
    const int pidx = off[e] + atomicAdd(&fill[e], 1);
    tok[pidx] = i / 6;
    wrow[pidx] = wts[i];
}

extern "C" __global__ void embed_t(const uint16_t* __restrict__ emb, const int* toks, float* h) {
    const int t = blockIdx.y;
    const int tok = toks[t];
    for (int d = blockIdx.x * blockDim.x + threadIdx.x; d < D; d += gridDim.x * blockDim.x) {
        const float v = bf16_to_f32(emb[(size_t)tok * D + d]);
        for (int k = 0; k < HC; k++) h[((size_t)t * HC + k) * D + d] = v;
    }
}

// window rows of positions p0-128 .. p0-1 from the ring into lin[0..128) (positions < 0 left as is)
extern "C" __global__ void ring_to_lin(const uint16_t* ring, int p0, uint16_t* lin) {
    P0_DEV(p0);
    const int r = blockIdx.x;  // lin row: position p0 - 128 + r
    const int p = p0 - WIN + r;
    if (p < 0) return;
    for (int i = threadIdx.x; i < HD; i += blockDim.x) lin[(size_t)r * HD + i] = ring[(size_t)(p % WIN) * HD + i];
}
// the batch's newest min(T, 128) positions from lin back into the ring
extern "C" __global__ void lin_to_ring(const uint16_t* lin, int p0, int T, uint16_t* ring) {
    P0_DEV(p0);
    const int j = blockIdx.x;  // 0 .. min(T,128)-1
    const int n = min(T, WIN);
    const int t = T - n + j, p = p0 + t;
    for (int i = threadIdx.x; i < HD; i += blockDim.x) ring[(size_t)(p % WIN) * HD + i] = lin[(size_t)(WIN + t) * HD + i];
}

// Engram rows of a batch: rows [T][ENG_ROW_BYTES] (this layer's) -> FP8 activation [T][6144]
extern "C" __global__ void __launch_bounds__(256) engram_act_t(const uint8_t* rows, int rstride, uint8_t* oq, float* os, uint16_t* obq) {
    const int t = blockIdx.x;
    const uint8_t* r = rows + (size_t)t * rstride;
    write_act(24 * 256, [&](int i) -> float {
        const int row = i >> 8, j = i & 255;
        const uint8_t b8 = r[row * 256 + j];
        const uint8_t sc = r[24 * 256 + row * 8 + (j >> 5)];
        const int s = b8 >> 7, e = (b8 >> 3) & 15, m = b8 & 7;
        float v = e == 0 ? ldexpf((float)m, -9) : ldexpf(1.f + m / 8.f, e - 7);
        v = s ? -v : v;
        return round_bf16(v * __uint_as_float((uint32_t)sc << 23));
    }, oq + (size_t)t * 6144, os + (size_t)t * 192, nullptr, obq + (size_t)t * 6144);
}


// ------------------------------------------------------------------ mm2: tiled bf16 GEMM for the prompt path
// Block = 64 weight rows x 128 tokens, 256 threads (wave w: rows 16w..16w+15, all 128 tokens = 8 MFMA columns of 16).
// K advances 32 at a time through LDS: weights decoded to bf16 with their scales (FP8 E8M0 blocks, MXFP4 E8M0 rows,
// codebook e4m3 rows x mul), activations = bf16 rows of the FP8-quantized values (job.xb, row stride ldx), so the
// products are exact and accumulate in f32 with no per-k-block rescaling. fmt 4: bf16 weights.
DEV uint32_t pack_bf16x2(float a, float b) { return bf16_bits(a) | bf16_bits(b) << 16; }

extern "C" __global__ void __launch_bounds__(256) mm2(const MmJob* __restrict__ jobs, const int* __restrict__ blk,
                                                      uint32_t cb3lo, uint32_t cb3hi, uint32_t cb2) {
    __shared__ uint16_t Ws[2][64][32 + 8];
    __shared__ uint16_t Xs[2][128][32 + 8];
    const int code = blk[blockIdx.x];
    const MmJob J = jobs[code >> 20];
    const int rb = (code >> 8) & 0xFFF, tb = code & 0xFF;
    const int tid = threadIdx.x, lane = tid & 63, wv = tid >> 6;
    const int rt = rb * 4 + wv;
    const bool rowok = rt * 16 < J.n;
    const int t0 = tb * 128;
    const int nt = min(128, J.ntok - t0);
    if (nt <= 0) return;
    const int ng = (nt + 15) / 16;
    const int KC = J.k / 32;
    // ---- this thread's activation slice: token tl = tid / 2, k half (tid & 1) * 16 (16 bf16 = 32 B)
    const int tl = tid >> 1, kh = (tid & 1) * 16;
    const int xr = tl < nt ? (J.toks ? J.toks[J.tok0 + t0 + tl] : J.tok0 + t0 + tl) : -1;
    const uint16_t* xp = J.xb + (size_t)(xr >= 0 ? xr : 0) * J.ldx + kh;
    // ---- this lane's weight slice per k32 chunk: row l%16 of tile rt, k 8*(l/16)..+8 (8 weights)
    const int wrow = lane & 15, wk = 8 * (lane >> 4);
    const int r = rt * 16 + wrow;
    const char* eb = (const char*)J.w;
    const int NDW = J.fmt == 1 ? 4 : (J.fmt == 2 ? 3 : 2);
    auto ldw = [&](int kc, u4v& raw) {
        if (!rowok) { raw = u4v{0, 0, 0, 0}; return; }
        if (J.fmt == 0) {
            const u4v w = __builtin_nontemporal_load((const u4v*)J.w + ((size_t)rt * (J.k / 64) + kc / 2) * 64 + lane);
            raw = (kc & 1) ? u4v{w.z, w.w, 0, 0} : u4v{w.x, w.y, 0, 0};
        } else if (J.fmt == 4) {
            // bf16 tile [rt][k32][lane][8]: lane holds row l%16, k 4*(l/16)..+4 and +16: re-gather to 8 consecutive k
            raw = __builtin_nontemporal_load((const u4v*)J.w + ((size_t)rt * KC + kc) * 64 + lane);
        } else {
            const uint32_t* wp = (const uint32_t*)(eb + J.woff) + ((size_t)rt * (J.k / 128) + kc / 4) * 64 * NDW + lane * NDW;
            raw.x = __builtin_nontemporal_load(wp + (J.fmt == 1 ? (kc & 3) : ((kc & 3) >> 1)));
            raw.y = J.fmt == 2 ? __builtin_nontemporal_load(wp + 2) : 0u;
            const uint8_t* sb = (const uint8_t*)(eb + J.soff) + ((size_t)rt * (J.k / 128) + kc / 4) * 64 + wrow * 4 + (kc & 3);
            raw.z = *sb;
        }
    };
    // decode the slice to 8 bf16 (k = wk .. wk+7 of chunk kc) and store into Ws[buf]
    auto stw = [&](int kc, const u4v& raw, int buf) {
        uint32_t o[4];
        if (J.fmt == 4) {
            // lane holds k {4g..4g+3} and {16+4g..} of row: write those two 4-groups directly
            uint16_t* d = &Ws[buf][wv * 16 + wrow][0];
            const int g4 = 4 * (lane >> 4);
            *reinterpret_cast<uint2*>(d + g4) = uint2{raw.x, raw.y};
            *reinterpret_cast<uint2*>(d + 16 + g4) = uint2{raw.z, raw.w};
            return;
        }
        float v[8];
        if (J.fmt == 0) {
            const float sc = __uint_as_float((uint32_t)J.sc[(size_t)(rt >> 1) * KC + kc] << 23) * 2.f;
            for (int j = 0; j < 8; j++) v[j] = fp8_to_f32(j < 4 ? raw.x : raw.y, j & 3) * sc;
        } else {
            uint64_t a8;
            if (J.fmt == 1) a8 = fp4x8_to_fp8(raw.x);
            else {
                const uint32_t tlo = J.fmt == 2 ? cb3lo : cb2, thi = J.fmt == 2 ? cb3hi : 0u;
                const int m = kc & 3, sh = 4 * (m & 1);
                uint32_t s0 = (raw.x >> sh) & 0x03030303u, s1 = (raw.x >> (sh + 2)) & 0x03030303u;
                if (J.fmt == 2) {
                    s0 |= ((raw.y >> (2 * m)) & 0x01010101u) << 2;
                    s1 |= ((raw.y >> (2 * m + 1)) & 0x01010101u) << 2;
                }
                a8 = (uint64_t)perm(thi, tlo, s1) << 32 | perm(thi, tlo, s0);
            }
            const float sc = J.fmt == 1 ? __uint_as_float(raw.z << 23) : fp8_to_f32(raw.z, 0) * J.mul;
            for (int j = 0; j < 8; j++) v[j] = fp8_to_f32((uint32_t)(a8 >> (32 * (j >> 2))), j & 3) * sc;
        }
        for (int j = 0; j < 4; j++) o[j] = pack_bf16x2(v[2 * j], v[2 * j + 1]);
        *reinterpret_cast<u4v*>(&Ws[buf][wv * 16 + wrow][wk]) = u4v{o[0], o[1], o[2], o[3]};
    };
    auto ldx = [&](int kc, u4v* x2) {
        if (xr >= 0) {
            x2[0] = *reinterpret_cast<const u4v*>(xp + kc * 32);
            x2[1] = *reinterpret_cast<const u4v*>(xp + kc * 32 + 8);
        } else {
            x2[0] = x2[1] = u4v{0, 0, 0, 0};
        }
    };
    auto stx = [&](const u4v* x2, int buf) {
        *reinterpret_cast<u4v*>(&Xs[buf][tl][kh]) = x2[0];
        *reinterpret_cast<u4v*>(&Xs[buf][tl][kh + 8]) = x2[1];
    };
    f4 acc[8];
#pragma unroll
    for (int g = 0; g < 8; g++) acc[g] = f4{0, 0, 0, 0};
    u4v wraw, xr2[2];
    ldw(0, wraw);
    ldx(0, xr2);
    stw(0, wraw, 0);
    stx(xr2, 0);
    __syncthreads();
    const int g4 = 4 * (lane >> 4), col = lane & 15;
    for (int kc = 0; kc < KC; kc++) {
        const int buf = kc & 1;
        if (kc + 1 < KC) { ldw(kc + 1, wraw); ldx(kc + 1, xr2); }
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const s4v a = *reinterpret_cast<const s4v*>(&Ws[buf][wv * 16 + col][16 * h + g4]);
#pragma unroll
            for (int g = 0; g < 8; g++) {
                if (g < ng) {
                    const s4v b = *reinterpret_cast<const s4v*>(&Xs[buf][16 * g + col][16 * h + g4]);
                    acc[g] = mfma_bf16(a, b, acc[g]);
                }
            }
        }
        if (kc + 1 < KC) { stw(kc + 1, wraw, buf ^ 1); stx(xr2, buf ^ 1); }
        __syncthreads();
    }
    if (!rowok) return;
    const int r0 = rt * 16 + 4 * (lane >> 4);
#pragma unroll
    for (int g = 0; g < 8; g++) {
        if (g >= ng) continue;
        const int t = t0 + g * 16 + col;
        if (t >= J.ntok) continue;
        const int arow = J.toks ? J.toks[J.tok0 + t] : J.tok0 + t;
        const int orow = J.yrow ? J.yrow[J.tok0 + t] : arow;
        float* yp = J.y + (size_t)orow * J.ldy + r0;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            if (J.yatomic) atomicAdd(yp + j, acc[g][j]);
            else yp[j] = acc[g][j];
        }
    }
}

// ------------------------------------------------------------------ attn_m: MFMA sparse attention for the prompt path
// One block per query (1024 threads = 16 waves), all 64 heads; rows in chunks of 32 through LDS.
// Scores S[64][32] = Q . R^T on bf16 MFMA with Q split into hi + lo bf16 (f32-accurate); waves: head tile w%4,
// row tile (w/4)%2, k half w/8 (summed through LDS). Online softmax per head; O[64][512] += P . R with P split
// hi + lo; waves: head tile w%4, dim tiles (w/4)*8 .. +8. Writes O / L (after the sink) over the query's q row.
// attn_core: query t (its q row), window rows lo .. lo+nwin-1 read from lin at row p - wsub or, ring (a batch slot's),
// p % SRING, then nc compressed rows comp[cidx[t][..] - woff]
__device__ __forceinline__ void attn_core(int t, int lo, int nwin, int wsub, bool ring, float* __restrict__ q, const uint16_t* __restrict__ lin,
                                          const uint16_t* __restrict__ comp, const int* __restrict__ cidx, int nc, int ldc,
                                          int woff, float scale, const float* sinks, bool cpk = false, int rbeg = 0,
                                          int rend = 1 << 30, float* part = nullptr) {
    __shared__ uint16_t rows[32][HD + 8];
    __shared__ float sp[2][64][33];
    __shared__ uint16_t ph[64][32 + 8], pl[64][32 + 8];
    __shared__ float hm[64], hl[64], ha[64];
    __shared__ int rowsrc[32];
    const int tid = threadIdx.x, lane = tid & 63, w = tid >> 6;
    const int col = lane & 15, g4 = 4 * (lane >> 4);
    // ---- Q fragments for the score role: head tile ht, k half kh (16 k-steps of 16)
    const int ht = w & 3, rtile = (w >> 2) & 1, kh = w >> 3;
    s4v qhi[16], qlo[16];
    {
        const float* qp = q + ((size_t)t * NH + 16 * ht + col) * HD + 256 * kh + g4;
#pragma unroll
        for (int ks = 0; ks < 16; ks++) {
            const float4 v = *reinterpret_cast<const float4*>(qp + 16 * ks);
            const float vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const uint32_t hb = bf16_bits(vv[j]);
                qhi[ks][j] = (short)hb;
                qlo[ks][j] = (short)bf16_bits(vv[j] - __uint_as_float(hb << 16));
            }
        }
    }
    // ---- output tiles for the PV role
    const int oht = w & 3, odt0 = (w >> 2) * 8;
    f4 o[8];
#pragma unroll
    for (int i = 0; i < 8; i++) o[i] = f4{0, 0, 0, 0};
    if (tid < 64) { hm[tid] = -INFINITY; hl[tid] = 0.f; }
    // rows [rbeg, rend) of the window + compressed rows (split attention: `part` gets the unnormalized partial)
    const int total = min(nwin + nc, rend);
    for (int r0 = rbeg; r0 < total; r0 += 32) {
        __syncthreads();
        if (tid < 32) {
            const int rr = r0 + tid;
            int src = -1;
            if (rr < nwin) src = ring ? (lo + rr) % SRING : (lo + rr) - wsub;
            else if (rr < total) {
                const int ci = cidx[(size_t)t * ldc + rr - nwin];
                src = ci >= 0 ? 0x40000000 | (ci - woff) : -1;
            }
            rowsrc[tid] = src;
        }
        __syncthreads();
        {
            // 32 rows x 512 bf16 = 32 KB: 32 B per thread
            const int rw = tid >> 5, j = (tid & 31) * 16;
            const int src = rowsrc[rw];
            u4v v0 = {0, 0, 0, 0}, v1 = {0, 0, 0, 0};
            if (src >= 0) {
                if (cpk && (src & 0x40000000)) {
                    // packed FP4 compressed row (batched decode slots)
                    const uint8_t* rp = (const uint8_t*)comp + (size_t)(src & 0x3FFFFFFF) * CROW_B;
                    v0 = fp4_row_bf16x8(rp, HD, 16, j, true);
                    v1 = fp4_row_bf16x8(rp, HD, 16, j + 8, true);
                } else {
                    const uint16_t* bp = (src & 0x40000000) ? comp + (size_t)(src & 0x3FFFFFFF) * HD : lin + (size_t)src * HD;
                    v0 = *reinterpret_cast<const u4v*>(bp + j);
                    v1 = *reinterpret_cast<const u4v*>(bp + j + 8);
                }
            }
            *reinterpret_cast<u4v*>(&rows[rw][j]) = v0;
            *reinterpret_cast<u4v*>(&rows[rw][j + 8]) = v1;
        }
        __syncthreads();
        // ---- partial scores: head tile ht x row tile rtile over k half kh
        {
            f4 s = {0, 0, 0, 0};
#pragma unroll
            for (int ks = 0; ks < 16; ks++) {
                const s4v b = *reinterpret_cast<const s4v*>(&rows[16 * rtile + col][256 * kh + 16 * ks + g4]);
                s = mfma_bf16(qhi[ks], b, s);
                s = mfma_bf16(qlo[ks], b, s);
            }
            // lane: heads 16ht + g4 + j, row 16rtile + col
#pragma unroll
            for (int j = 0; j < 4; j++) sp[kh][16 * ht + g4 + j][16 * rtile + col] = s[j];
        }
        __syncthreads();
        // ---- online softmax per head: 2 values per thread (head h = tid / 16, rows (tid % 16) and +16)
        {
            const int h = tid >> 4, rr = tid & 15;
            float v[2];
#pragma unroll
            for (int u = 0; u < 2; u++) {
                const int ri = rr + 16 * u;
                const bool ok = r0 + ri < total && rowsrc[ri] >= 0;
                v[u] = ok ? (sp[0][h][ri] + sp[1][h][ri]) * scale : -INFINITY;
            }
            const float mc = group_max<16>(fmaxf(v[0], v[1]));
            const float Mo = hm[h];
            const float Mn = fmaxf(Mo, mc);
            float e[2], es = 0;
#pragma unroll
            for (int u = 0; u < 2; u++) {
                e[u] = v[u] == -INFINITY ? 0.f : expf(v[u] - Mn);
                es += e[u];
                const uint32_t hb = bf16_bits(e[u]);
                ph[h][rr + 16 * u] = (uint16_t)hb;
                pl[h][rr + 16 * u] = (uint16_t)bf16_bits(e[u] - __uint_as_float(hb << 16));
            }
            es = group_sum<16>(es);
            __syncthreads();
            if (rr == 0) {
                const float a = Mo == -INFINITY ? 0.f : expf(Mo - Mn);
                ha[h] = a;
                hl[h] = hl[h] * a + es;
                hm[h] = Mn;
            }
        }
        __syncthreads();
        // ---- PV: o[head tile oht][dim tiles] = o * alpha + P . R
        {
            float al[4];
#pragma unroll
            for (int j = 0; j < 4; j++) al[j] = ha[16 * oht + g4 + j];
#pragma unroll
            for (int i = 0; i < 8; i++)
#pragma unroll
                for (int j = 0; j < 4; j++) o[i][j] *= al[j];
#pragma unroll
            for (int ks = 0; ks < 2; ks++) {
                const s4v ah = *reinterpret_cast<const s4v*>(&ph[16 * oht + col][16 * ks + g4]);
                const s4v alo = *reinterpret_cast<const s4v*>(&pl[16 * oht + col][16 * ks + g4]);
#pragma unroll
                for (int i = 0; i < 8; i++) {
                    const int d = 16 * (odt0 + i) + col;
                    s4v b;
#pragma unroll
                    for (int j = 0; j < 4; j++) b[j] = (short)rows[16 * ks + g4 + j][d];
                    o[i] = mfma_bf16(ah, b, o[i]);
                    o[i] = mfma_bf16(alo, b, o[i]);
                }
            }
        }
    }
    __syncthreads();
    if (part) {
        // partial: o [64][512] unnormalized, then m [64], l [64]
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int h = 16 * oht + g4 + j;
#pragma unroll
            for (int i = 0; i < 8; i++) part[(size_t)h * HD + 16 * (odt0 + i) + col] = o[i][j];
        }
        if (tid < 64) { part[NH * HD + tid] = hm[tid]; part[NH * HD + 64 + tid] = hl[tid]; }
        return;
    }
    // ---- normalize with the sink, write over the query's q row (heads 16*oht + g4 + j, dims 16*(odt0+i) + col)
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const int h = 16 * oht + g4 + j;
        const float L = hl[h] + (hm[h] == -INFINITY ? 1.f : expf(sinks[h] - hm[h]));
#pragma unroll
        for (int i = 0; i < 8; i++) q[((size_t)t * NH + h) * HD + 16 * (odt0 + i) + col] = round_bf16(o[i][j] / L);
    }
}
extern "C" __global__ void __launch_bounds__(1024) attn_m(int p0, float* __restrict__ q, const uint16_t* __restrict__ lin,
                                                          const uint16_t* __restrict__ comp, const int* __restrict__ cidx,
                                                          const int* ncomp, int ldc, int woff, float scale, const float* sinks,
                                                          int blk) {
    P0_DEV(p0);
    const int t = blockIdx.x, pos = p0 + t;
    // blk > 0 (DSpark draft block): every query sees the window before the block and the whole block
    const int lo = blk > 0 ? max(0, p0 - WIN) : max(0, pos - (WIN - 1));
    const int nwin = (blk > 0 ? p0 + blk - 1 : pos) - lo + 1;
    attn_core(t, lo, nwin, p0 - WIN, false, q, lin, comp, cidx, comp ? ncomp[t] : 0, ldc, woff, scale, sinks);
}

// after attn_m: inverse RoPE on the last 64 dims of each head, then FP8 per 32 (+ bf16 copy) for wo_a. Block per query.
// blockIdx.y: group of 8 heads (the FP8 scale groups of 32 never straddle groups)
extern "C" __global__ void __launch_bounds__(256) attn_m_out(int p0, float* __restrict__ o, const float* rf, uint8_t* oq, float* os, uint16_t* obq) {
    P0_DEV(p0);
    const int t = blockIdx.x, pos = p0 + t, h0 = blockIdx.y * 8;
    float* ot = o + (size_t)t * NH * HD + (size_t)h0 * HD;
    __shared__ float cs[32], sn[32];
    if (threadIdx.x < 32) {
        const float ang = (float)pos * rf[threadIdx.x];
        cs[threadIdx.x] = cosf(ang);
        sn[threadIdx.x] = -sinf(ang);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < 8 * 32; i += 256) {
        const int h = i / 32, pi = i % 32;
        float* v = ot + h * HD + HD - RD + 2 * pi;
        const float a = v[0], b = v[1];
        v[0] = a * cs[pi] - b * sn[pi];
        v[1] = a * sn[pi] + b * cs[pi];
    }
    __syncthreads();
    const size_t off = (size_t)t * NH * HD + (size_t)h0 * HD;
    write_act(8 * HD, [&](int i) -> float { return ot[i]; }, oq + off, os + off / 32, nullptr, obq + off);
}

// DSpark router: one wave per token over ne <= 128 experts; sqrt(softplus) scores, top-k by score + bias,
// weights = score / sum * scale (as the main router). sel/wts [t][k].
extern "C" __global__ void route_k_t(const float* logits, int ne, int k, const float* bias, float scale, int* sel, float* wts) {
    const int t = blockIdx.x, lane = threadIdx.x;
    float sc[2], key[2];
    for (int j = 0; j < 2; j++) {
        const int e = lane + 64 * j;
        if (e < ne) {
            const float l = logits[(size_t)t * ne + e];
            const float ex = __expf(l);
            const float sp = l > 20.f ? l : (ex < 1e-3f ? ex * (1.f - 0.5f * ex) : __logf(1.f + ex));
            sc[j] = sqrtf(sp);
            key[j] = sc[j] + bias[e];
        } else {
            sc[j] = 0.f;
            key[j] = -INFINITY;
        }
    }
    float sum = 0, w[8];
    int id[8];
    for (int s = 0; s < k; s++) {
        float bk = key[0] >= key[1] ? key[0] : key[1];
        int bi = key[0] >= key[1] ? lane : lane + 64;
        for (int o = 1; o < 64; o <<= 1) {
            const float ok = __shfl_xor(bk, o);
            const int oi = __shfl_xor(bi, o);
            if (ok > bk || (ok == bk && oi < bi)) { bk = ok; bi = oi; }
        }
        const float v = __shfl(sc[bi >> 6], bi & 63);
        if (lane == (bi & 63)) key[bi >> 6] = -INFINITY;
        id[s] = bi;
        w[s] = v;
        sum += v;
    }
    if (lane == 0)
        for (int s = 0; s < k; s++) {
            sel[t * k + s] = id[s];
            wts[t * k + s] = w[s] / (sum + 1e-20f) * scale;
        }
}

// DSpark main_hidden: the mean of the 4 residual streams of h [t][4][D], bf16-rounded, into out[t][ld] + off
extern "C" __global__ void mh_t(const float* h, float* out, int ld, int off) {
    const int t = blockIdx.y;
    for (int d = blockIdx.x * blockDim.x + threadIdx.x; d < D; d += gridDim.x * blockDim.x) {
        float v = 0;
        for (int k = 0; k < HC; k++) v += h[((size_t)t * HC + k) * D + d];
        out[(size_t)t * ld + off + d] = round_bf16(v / HC);
    }
}
