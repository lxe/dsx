// verify.h: decode-style matrix-vector products for 2..16 tokens at once (speculative verify, draft passes).
// Weights stream exactly as in decode (one wave per 16-row tile and k-slice); MFMA column c = token c, whose FP8
// activation row and scales the lane reads itself, so every per-k-block scale is per lane (token) and per row.
#pragma once
#include "gemv.h"
#ifndef VDU
#define VDU 2  // dense FP8 k-pairs in flight per wave (verify)
#endif
#ifndef VXU
#define VXU 1  // expert k-units in flight per wave (verify)
#endif
typedef float f2v __attribute__((ext_vector_type(2)));
#ifndef XG_NOX1
#define XG_NOX1 1  // 0: one-token batched expert jobs column-rotated (faster at B=1 only; more VGPRs)
#endif

struct VJob {
    const void* w;      // tiles
    const void* sc;     // FP8: E8M0 [N/32][K/32]; experts: arena base (w) + offsets below
    long long woff, soff;
    float mul;
    int fmt;            // 0 FP8 dense, 1 MXFP4, 2 CB3, 3 CB2, 4 BF16
    int n, k, split;
    int zero;           // vzero: clear this job's output rows first (k-split partial sums into a stored output)
    const uint8_t* xq;  // FP8 codes [row][ldx]
    const float* xs;    // scales [row][ldx/32]
    const uint16_t* xb; // bf16 rows [row][ldx] (fmt 4)
    int ldx;
    int ntok;           // tokens (<= 16)
    const int* rows;    // activation row per token (null: token index)
    float* y;           // output [orow][ldy]
    const int* orows;   // output row per token (null: token index)
    int ldy, atomic;    // atomic: += (split > 1 or scatter-add)
    int blocks, epi;    // epi 1/2: RoPE (q: 512-dim heads, idx: 128) at position pos0 + token
    int ropec, ndyn;    // ndyn: dyn points at ndyn consecutive device-built jobs (40 ints each)
    const int* dyn;     // device-built expert job (null: static): [fmt, ntok, woff_lo, woff_hi, soff_lo, soff_hi, mul_bits, ...]
    const uint16_t* xl; // fmt 4: optional bf16 low parts (x = xb + xl, as decode's head)
};

struct VJobs {
    VJob j[12];
    int n;
    uint32_t cb3lo, cb3hi, cb2;
};

// One wave: one 16-row tile over k-units [u0, u1) for up to 16 tokens.
template <int KIND>  // 0: any format, 1: dense only (FP8, BF16), 2: experts only
DEV f4 vwave(const VJob& J, int rt, int u0, int u1, int lane, uint32_t cb3lo, uint32_t cb3hi, uint32_t cb2) {
    const int col = lane & 15, g8 = 8 * (lane >> 4), g4 = 4 * (lane >> 4);
    const bool live = col < J.ntok;
    const int xrow = live ? (J.rows ? J.rows[col] : col) : 0;
    const uint8_t* xq = J.xq + (size_t)xrow * J.ldx;
    const float* xs = J.xs + (size_t)xrow * (J.ldx / 32);
    const float lm = live ? 1.f : 0.f;
    f4 acc = {0, 0, 0, 0};
    if (KIND != 2 && J.fmt == 0) {
        const int KP = J.k / 64, KB = J.k / 32;
        const u4v* wp = (const u4v*)J.w + (size_t)rt * KP * 64 + lane;
        const uint8_t* sp = (const uint8_t*)J.sc + (size_t)(rt >> 1) * KB;
        constexpr int U = VDU;
        u4v cw[U], nw[U];
        uint64_t cx[U][2], nx[U][2];
        float cs[U][2], ns[U][2];
        auto ld = [&](int kp, u4v& w, uint64_t* x, float* s) {
            w = __builtin_nontemporal_load(wp + (size_t)kp * 64);
            x[0] = *reinterpret_cast<const uint64_t*>(xq + kp * 64 + g8);
            x[1] = *reinterpret_cast<const uint64_t*>(xq + kp * 64 + 32 + g8);
            s[0] = xs[2 * kp] * __uint_as_float((uint32_t)sp[2 * kp] << 23) * lm;
            s[1] = xs[2 * kp + 1] * __uint_as_float((uint32_t)sp[2 * kp + 1] << 23) * lm;
        };
        for (int u = 0; u < U; u++)
            if (u0 + u < u1) ld(u0 + u, cw[u], cx[u], cs[u]);
        for (int k = u0; k < u1; k += U) {
            for (int u = 0; u < U; u++)
                if (k + U + u < u1) ld(k + U + u, nw[u], nx[u], ns[u]);
#pragma unroll
            for (int u = 0; u < U; u++) {
                if (k + u < u1) {
                    const f4 d0 = mfma_fp8((uint64_t)cw[u].y << 32 | cw[u].x, cx[u][0], f4{0, 0, 0, 0});
                    const f4 d1 = mfma_fp8((uint64_t)cw[u].w << 32 | cw[u].z, cx[u][1], f4{0, 0, 0, 0});
                    acc += d0 * cs[u][0] + d1 * cs[u][1];
                }
            }
#pragma unroll
            for (int u = 0; u < U; u++) {
                cw[u] = nw[u];
                cx[u][0] = nx[u][0]; cx[u][1] = nx[u][1];
                cs[u][0] = ns[u][0]; cs[u][1] = ns[u][1];
            }
        }
        return acc * 2.f;
    }
    if (KIND != 2 && J.fmt == 4) {
        const int KQ = J.k / 32;
        const u4v* wp = (const u4v*)J.w + (size_t)rt * KQ * 64 + lane;
        const uint16_t* xb = J.xb + (size_t)xrow * J.ldx;
        for (int kq = u0; kq < u1; kq++) {
            const u4v w = __builtin_nontemporal_load(wp + (size_t)kq * 64);
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const uint32_t w0 = h ? w.z : w.x, w1 = h ? w.w : w.y;
                s4v a;
                a[0] = (short)(w0 & 0xFFFF); a[1] = (short)(w0 >> 16); a[2] = (short)(w1 & 0xFFFF); a[3] = (short)(w1 >> 16);
                uint64_t bv = live ? *reinterpret_cast<const uint64_t*>(xb + kq * 32 + 16 * h + g4) : 0;
                acc = mfma_bf16(a, *reinterpret_cast<const s4v*>(&bv), acc);
                if (J.xl) {
                    uint64_t lv = live ? *reinterpret_cast<const uint64_t*>(J.xl + (size_t)xrow * J.ldx + kq * 32 + 16 * h + g4) : 0;
                    acc = mfma_bf16(a, *reinterpret_cast<const s4v*>(&lv), acc);
                }
            }
        }
        return acc;
    }
    if (KIND == 1) return acc;
    // experts: per-row scales (lane rows 4*(lane/16)+j), per-lane token scales. Weights, scales and the token's
    // activation fragments are loaded U units ahead (latency, not bandwidth, bounds a 16-row tile).
    const char* base = (const char*)J.w;
    const int KU = J.k / 128;
    const int NDW = J.fmt == 1 ? 4 : (J.fmt == 2 ? 3 : 2);
    const u4v* sp = (const u4v*)(base + J.soff) + (size_t)rt * KU * 4 + (lane >> 4);
    const uint32_t* wp = (const uint32_t*)(base + J.woff) + (size_t)rt * KU * 64 * NDW + lane * NDW;
    constexpr int U = VXU;
    struct Unit { uint32_t w[4]; u4v s; uint64_t x[4]; float xs[4]; };
    Unit cu[U], nu[U];
    auto ld = [&](int u, Unit& o) {
#pragma unroll
        for (int q = 0; q < 4; q++) o.w[q] = q < NDW ? __builtin_nontemporal_load(wp + (size_t)u * 64 * NDW + q) : 0u;
        o.s = sp[(size_t)u * 4];
#pragma unroll
        for (int m = 0; m < 4; m++) {
            o.x[m] = live ? *reinterpret_cast<const uint64_t*>(xq + (4 * u + m) * 32 + g8) : 0ull;
            o.xs[m] = xs[4 * u + m] * lm;
        }
    };
#pragma unroll
    for (int i = 0; i < U; i++)
        if (u0 + i < u1) ld(u0 + i, cu[i]);
    for (int u = u0; u < u1; u += U) {
#pragma unroll
        for (int i = 0; i < U; i++)
            if (u + U + i < u1) ld(u + U + i, nu[i]);
#pragma unroll
        for (int i = 0; i < U; i++) {
            if (u + i >= u1) break;
            const Unit& c = cu[i];
            const uint32_t ss[4] = {c.s.x, c.s.y, c.s.z, c.s.w};
#pragma unroll
            for (int m = 0; m < 4; m++) {
                uint64_t a;
                if (J.fmt == 1) a = fp4x8_to_fp8(c.w[m]);
                else {
                    const uint32_t tlo = J.fmt == 2 ? cb3lo : cb2, thi = J.fmt == 2 ? cb3hi : 0u;
                    const uint32_t dl = c.w[m >> 1];
                    const int sh = 4 * (m & 1);
                    uint32_t s0 = (dl >> sh) & 0x03030303u, s1 = (dl >> (sh + 2)) & 0x03030303u;
                    if (J.fmt == 2) {
                        s0 |= ((c.w[2] >> (2 * m)) & 0x01010101u) << 2;
                        s1 |= ((c.w[2] >> (2 * m + 1)) & 0x01010101u) << 2;
                    }
                    a = (uint64_t)perm(thi, tlo, s1) << 32 | perm(thi, tlo, s0);
                }
                const f4 d = mfma_fp8(a, c.x[m], f4{0, 0, 0, 0});
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    const uint32_t bb = (ss[j] >> (8 * m)) & 0xFFu;
                    const float ws = J.fmt == 1 ? __uint_as_float(bb << 23) : fp8_to_f32(bb, 0);
                    acc[j] += d[j] * (ws * c.xs[m]);
                }
            }
        }
#pragma unroll
        for (int i = 0; i < U; i++) cu[i] = nu[i];
    }
    return acc * J.mul;
}

// Blocks enumerate (job, row-block of 4 tiles, k-slice). Dynamic expert jobs read fmt/offsets/tokens from `dyn`.
template <int KIND>
DEV void vgemv_impl(VJob J, int b, uint32_t cb3lo, uint32_t cb3hi, uint32_t cb2, const int* pos0p, const float* rope_f, const float* rope_fc) {
    const int lane = threadIdx.x & 63, wv = threadIdx.x >> 6;
    if (J.dyn) {
        // device-built expert job: [fmt, ntok, woff lo/hi, soff lo/hi, mul bits, rows[16], orows[16]]
        const int per = J.blocks / J.ndyn;
        const int* d = J.dyn + (size_t)(b / per) * 40;
        b %= per;
        J.fmt = d[0];
        J.ntok = d[1];
        if (J.ntok == 0) return;
        J.woff = (long long)(uint32_t)d[2] | (long long)d[3] << 32;
        J.soff = (long long)(uint32_t)d[4] | (long long)d[5] << 32;
        J.mul = __int_as_float(d[6]);
        J.rows = d + 8;
        J.orows = d + 24;
    }
    const int RB = (J.n + 63) / 64;
    const int rb = b % RB, sl = b / RB;
    const int rt = rb * 4 + wv;
    if (rt * 16 >= J.n) return;
    const int KU = J.fmt == 0 ? 64 : (J.fmt == 4 ? 32 : 128);
    const int KUs = J.k / KU;
    const int u0 = KUs * sl / J.split, u1 = KUs * (sl + 1) / J.split;
    f4 acc = vwave<KIND>(J, rt, u0, u1, lane, cb3lo, cb3hi, cb2);
    const int col = lane & 15;
    if (col >= J.ntok) return;
    const int r0 = rt * 16 + 4 * (lane >> 4);
    float v[4] = {acc[0], acc[1], acc[2], acc[3]};
    if (J.epi == 1 || J.epi == 2) {
        const int hd = J.epi == 1 ? HD : ID;
        const int dim = r0 % hd;
        if (dim >= hd - RD) {
            const float* f = J.ropec ? rope_fc : rope_f;
            const int pos = *pos0p + col;
            for (int p = 0; p < 2; p++) rope_pair(v[2 * p], v[2 * p + 1], (float)pos * f[(dim - (hd - RD)) / 2 + p]);
        }
    }
    const int orow = J.orows ? J.orows[col] : col;
    float* yp = J.y + (size_t)orow * J.ldy + r0;
    for (int j = 0; j < 4; j++) {
        if (J.atomic) atomicAdd(yp + j, v[j]);
        else yp[j] = v[j];
    }
}

extern "C" __global__ void __launch_bounds__(256) vgemv(VJobs js, const int* pos0p, const float* rope_f, const float* rope_fc) {
    int b = blockIdx.x, ji = 0;
    while (ji < js.n - 1 && b >= js.j[ji].blocks) b -= js.j[ji++].blocks;
    vgemv_impl<0>(js.j[ji], b, js.cb3lo, js.cb3hi, js.cb2, pos0p, rope_f, rope_fc);
}
extern "C" __global__ void __launch_bounds__(256) vgemv_d(VJobs js, const int* pos0p, const float* rope_f, const float* rope_fc) {
    int b = blockIdx.x, ji = 0;
    while (ji < js.n - 1 && b >= js.j[ji].blocks) b -= js.j[ji++].blocks;
    vgemv_impl<1>(js.j[ji], b, js.cb3lo, js.cb3hi, js.cb2, pos0p, rope_f, rope_fc);
}
extern "C" __global__ void __launch_bounds__(256) vgemv_x(VJobs js, const int* pos0p, const float* rope_f, const float* rope_fc) {
    int b = blockIdx.x, ji = 0;
    while (ji < js.n - 1 && b >= js.j[ji].blocks) b -= js.j[ji++].blocks;
    vgemv_impl<2>(js.j[ji], b, js.cb3lo, js.cb3hi, js.cb2, pos0p, rope_f, rope_fc);
}

// Batched decode: device-built expert jobs (all js.j dyn), only the first *njp of each job list used (bgroup's job
// count): a fixed grid of blocks loops over the real jobs' blocks (a full-size grid of mostly empty jobs left the
// launch bound by block dispatch, ~1 TB/s)
extern "C" __global__ void __launch_bounds__(256) vgemv_xp(VJobs js, const int* njp, const int* pos0p, const float* rope_f, const float* rope_fc) {
    const int nj = *njp;
    int tot = 0;
    for (int ji = 0; ji < js.n; ji++) tot += nj * (js.j[ji].blocks / js.j[ji].ndyn);
    for (int w = blockIdx.x; w < tot; w += gridDim.x) {
        int b = w, ji = 0;
        while (ji < js.n - 1 && b >= nj * (js.j[ji].blocks / js.j[ji].ndyn)) {
            b -= nj * (js.j[ji].blocks / js.j[ji].ndyn);
            ji++;
        }
        vgemv_impl<2>(js.j[ji], b, js.cb3lo, js.cb3hi, js.cb2, pos0p, rope_f, rope_fc);
    }
}

// Batched decode experts (device-built jobs only): one wave owns R adjacent 16-row tiles and loads each token's
// activation fragments once for all of them (the activation gathers - 16 token rows per k-block - cost more memory
// instructions than the 3-bit weights themselves). Block = 4 waves = 64R rows; blocks per job: ceil(n/64R) * split.
template <int R>
DEV void vgemv_xr_impl(VJob J, int b, uint32_t cb3lo, uint32_t cb3hi, uint32_t cb2) {
    const int lane = threadIdx.x & 63, wv = threadIdx.x >> 6;
    {
        const int per = J.blocks / J.ndyn;
        const int* d = J.dyn + (size_t)(b / per) * 40;
        b %= per;
        J.fmt = d[0];
        J.ntok = d[1];
        if (J.ntok == 0) return;
        J.woff = (long long)(uint32_t)d[2] | (long long)d[3] << 32;
        J.soff = (long long)(uint32_t)d[4] | (long long)d[5] << 32;
        J.mul = __int_as_float(d[6]);
        J.rows = d + 8;
        J.orows = d + 24;
    }
    const int RB = (J.n + 64 * R - 1) / (64 * R);
    const int rb = b % RB, sl = b / RB;
    const int rt0 = (rb * 4 + wv) * R;
    if (rt0 * 16 >= J.n) return;
    const int KU = J.k / 128;
    const int u0 = KU * sl / J.split, u1 = KU * (sl + 1) / J.split;
    const int col = lane & 15, g8 = 8 * (lane >> 4);
    const bool live = col < J.ntok;
    const int xrow = live ? J.rows[col] : 0;
    const uint8_t* xq = J.xq + (size_t)xrow * J.ldx;
    const float* xs = J.xs + (size_t)xrow * (J.ldx / 32);
    const float lm = live ? 1.f : 0.f;
    const char* base = (const char*)J.w;
    const int NDW = J.fmt == 1 ? 4 : (J.fmt == 2 ? 3 : 2);
    const uint32_t tlo = J.fmt == 2 ? cb3lo : cb2, thi = J.fmt == 2 ? cb3hi : 0u;
    f4 acc[R];
#pragma unroll
    for (int r = 0; r < R; r++) acc[r] = f4{0, 0, 0, 0};
    struct Unit { uint32_t w[R][4]; u4v s[R]; uint64_t x[4]; float xs[4]; };
    Unit cu, nu;
    auto ld = [&](int u, Unit& o) {
#pragma unroll
        for (int r = 0; r < R; r++) {
            const int rt = min(rt0 + r, J.n / 16 - 1);
            const uint32_t* wp = (const uint32_t*)(base + J.woff) + ((size_t)rt * KU + u) * 64 * NDW + lane * NDW;
#pragma unroll
            for (int q = 0; q < 4; q++) o.w[r][q] = q < NDW ? __builtin_nontemporal_load(wp + q) : 0u;
            o.s[r] = ((const u4v*)(base + J.soff))[((size_t)rt * KU + u) * 4 + (lane >> 4)];
        }
#pragma unroll
        for (int m = 0; m < 4; m++) {
            o.x[m] = live ? *reinterpret_cast<const uint64_t*>(xq + (4 * u + m) * 32 + g8) : 0ull;
            o.xs[m] = xs[4 * u + m] * lm;
        }
    };
    if (u0 < u1) ld(u0, cu);
    for (int u = u0; u < u1; u++) {
        if (u + 1 < u1) ld(u + 1, nu);
#pragma unroll
        for (int r = 0; r < R; r++) {
            const uint32_t ss[4] = {cu.s[r].x, cu.s[r].y, cu.s[r].z, cu.s[r].w};
#pragma unroll
            for (int m = 0; m < 4; m++) {
                uint64_t a;
                if (J.fmt == 1) a = fp4x8_to_fp8(cu.w[r][m]);
                else {
                    const uint32_t dl = cu.w[r][m >> 1];
                    const int sh = 4 * (m & 1);
                    uint32_t s0 = (dl >> sh) & 0x03030303u, s1 = (dl >> (sh + 2)) & 0x03030303u;
                    if (J.fmt == 2) {
                        s0 |= ((cu.w[r][2] >> (2 * m)) & 0x01010101u) << 2;
                        s1 |= ((cu.w[r][2] >> (2 * m + 1)) & 0x01010101u) << 2;
                    }
                    a = (uint64_t)perm(thi, tlo, s1) << 32 | perm(thi, tlo, s0);
                }
                const f4 dd = mfma_fp8(a, cu.x[m], f4{0, 0, 0, 0});
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    const uint32_t bb = (ss[j] >> (8 * m)) & 0xFFu;
                    const float ws = J.fmt == 1 ? __uint_as_float(bb << 23) : fp8_to_f32(bb, 0);
                    acc[r][j] += dd[j] * (ws * cu.xs[m]);
                }
            }
        }
        cu = nu;
    }
    if (!live) return;
    const int orow = J.orows[col];
#pragma unroll
    for (int r = 0; r < R; r++) {
        const int rt = rt0 + r;
        if (rt * 16 >= J.n) break;
        float* yp = J.y + (size_t)orow * J.ldy + rt * 16 + 4 * (lane >> 4);
        for (int j = 0; j < 4; j++) atomicAdd(yp + j, acc[r][j] * J.mul);
    }
}
// As vgemv_xr_impl, with the block's activation slice (16 token rows x the k-slice, FP8 + scales) staged in LDS
// once: waves read their token fragments from LDS. Dynamic LDS: 16 * (kn + 8) + 16 * (kn / 32 + 1) * 4 bytes.
template <int R>
DEV void vgemv_xs_impl(VJob J, int b, uint32_t cb3lo, uint32_t cb3hi, uint32_t cb2) {
    extern __shared__ char xsm[];
    const int lane = threadIdx.x & 63, wv = threadIdx.x >> 6;
    {
        const int per = J.blocks / J.ndyn;
        const int* d = J.dyn + (size_t)(b / per) * 40;
        b %= per;
        J.fmt = d[0];
        J.ntok = d[1];
        if (J.ntok == 0) return;
        J.woff = (long long)(uint32_t)d[2] | (long long)d[3] << 32;
        J.soff = (long long)(uint32_t)d[4] | (long long)d[5] << 32;
        J.mul = __int_as_float(d[6]);
        J.rows = d + 8;
        J.orows = d + 24;
    }
    const int RB = (J.n + 64 * R - 1) / (64 * R);
    const int rb = b % RB, sl = b / RB;
    const int rt0 = (rb * 4 + wv) * R;
    const int KU = J.k / 128;
    const int u0 = KU * sl / J.split, u1 = KU * (sl + 1) / J.split;
    const int kn = (u1 - u0) * 128, k0 = u0 * 128;
    // rows padded (+8 bytes, +1 scale) so the 16 token rows of a fragment read fall in different banks
    const int xst = kn + 8, sst = kn / 32 + 1;
    uint8_t* xl = (uint8_t*)xsm;                   // [16][xst]
    float* xsl = (float*)(xsm + 16 * xst);          // [16][sst]
    for (int i = threadIdx.x; i < 16 * kn / 8; i += 256) {
        const int t = i / (kn / 8), c = i % (kn / 8);
        uint64_t v = 0;
        if (t < J.ntok) v = *reinterpret_cast<const uint64_t*>(J.xq + (size_t)J.rows[t] * J.ldx + k0 + 8 * c);
        *reinterpret_cast<uint64_t*>(xl + t * xst + 8 * c) = v;
    }
    for (int i = threadIdx.x; i < 16 * (kn / 32); i += 256) {
        const int t = i / (kn / 32), c = i % (kn / 32);
        xsl[t * sst + c] = t < J.ntok ? J.xs[(size_t)J.rows[t] * (J.ldx / 32) + k0 / 32 + c] : 0.f;
    }
    __syncthreads();
    if (rt0 * 16 >= J.n) return;
    const int col = lane & 15, g8 = 8 * (lane >> 4);
    const bool live = col < J.ntok;
    const char* base = (const char*)J.w;
    const int NDW = J.fmt == 1 ? 4 : (J.fmt == 2 ? 3 : 2);
    const uint32_t tlo = J.fmt == 2 ? cb3lo : cb2, thi = J.fmt == 2 ? cb3hi : 0u;
    f4 acc[R];
#pragma unroll
    for (int r = 0; r < R; r++) acc[r] = f4{0, 0, 0, 0};
    struct Unit { uint32_t w[R][4]; u4v s[R]; };
    Unit cu, nu;
    auto ld = [&](int u, Unit& o) {
#pragma unroll
        for (int r = 0; r < R; r++) {
            const int rt = min(rt0 + r, J.n / 16 - 1);
            const uint32_t* wp = (const uint32_t*)(base + J.woff) + ((size_t)rt * KU + u) * 64 * NDW + lane * NDW;
#pragma unroll
            for (int q = 0; q < 4; q++) o.w[r][q] = q < NDW ? __builtin_nontemporal_load(wp + q) : 0u;
            o.s[r] = ((const u4v*)(base + J.soff))[((size_t)rt * KU + u) * 4 + (lane >> 4)];
        }
    };
    if (u0 < u1) ld(u0, cu);
    for (int u = u0; u < u1; u++) {
        if (u + 1 < u1) ld(u + 1, nu);
        const int ku = (u - u0) * 128;
        uint64_t xv[4];
        float xsv[4];
#pragma unroll
        for (int m = 0; m < 4; m++) {
            xv[m] = *reinterpret_cast<const uint64_t*>(xl + col * xst + ku + m * 32 + g8);
            xsv[m] = xsl[col * sst + ku / 32 + m];
        }
#pragma unroll
        for (int r = 0; r < R; r++) {
            const uint32_t ss[4] = {cu.s[r].x, cu.s[r].y, cu.s[r].z, cu.s[r].w};
#pragma unroll
            for (int m = 0; m < 4; m++) {
                uint64_t a;
                if (J.fmt == 1) a = fp4x8_to_fp8(cu.w[r][m]);
                else {
                    const uint32_t dl = cu.w[r][m >> 1];
                    const int sh = 4 * (m & 1);
                    uint32_t s0 = (dl >> sh) & 0x03030303u, s1 = (dl >> (sh + 2)) & 0x03030303u;
                    if (J.fmt == 2) {
                        s0 |= ((cu.w[r][2] >> (2 * m)) & 0x01010101u) << 2;
                        s1 |= ((cu.w[r][2] >> (2 * m + 1)) & 0x01010101u) << 2;
                    }
                    a = (uint64_t)perm(thi, tlo, s1) << 32 | perm(thi, tlo, s0);
                }
                const f4 dd = mfma_fp8(a, xv[m], f4{0, 0, 0, 0});
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    const uint32_t bb = (ss[j] >> (8 * m)) & 0xFFu;
                    const float ws = J.fmt == 1 ? __uint_as_float(bb << 23) : fp8_to_f32(bb, 0);
                    acc[r][j] += dd[j] * (ws * xsv[m]);
                }
            }
        }
        cu = nu;
    }
    if (!live) return;
    const int orow = J.orows[col];
#pragma unroll
    for (int r = 0; r < R; r++) {
        const int rt = rt0 + r;
        if (rt * 16 >= J.n) break;
        float* yp = J.y + (size_t)orow * J.ldy + rt * 16 + 4 * (lane >> 4);
        for (int j = 0; j < 4; j++) atomicAdd(yp + j, acc[r][j] * J.mul);
    }
}
#define VGEMV_XS(R)                                                                                           \
    extern "C" __global__ void __launch_bounds__(256) vgemv_xs##R(VJobs js) {                                 \
        int b = blockIdx.x, ji = 0;                                                                           \
        while (ji < js.n - 1 && b >= js.j[ji].blocks) b -= js.j[ji++].blocks;                                \
        vgemv_xs_impl<R>(js.j[ji], b, js.cb3lo, js.cb3hi, js.cb2);                                            \
    }
VGEMV_XS(1)
VGEMV_XS(2)

// Expert jobs of up to 16*G tokens (bgroup jt = 16*G): R 16-row tiles per wave, each weight fragment decoded once and
// multiplied into G token groups. Prompt chunks (~32 tokens per expert per 2048 tokens) read each expert once.
// FMT: the job's weight format as a constant (1 MXFP4, 2 CB3, 3 CB2): a straight-line inner loop without format
// branches (the runtime format made the loop dozens of small blocks with masked loads)
template <int R, int G, int FMT>
DEV void vgemv_xg_body(VJob J, int b, uint32_t cb3lo, uint32_t cb3hi, uint32_t cb2) {
    const int lane = threadIdx.x & 63, wv = threadIdx.x >> 6;
    constexpr int JT = 16 * G;
    {
        const int per = J.blocks / J.ndyn;
        const int* d = J.dyn + (size_t)(b / per) * (8 + 2 * JT);
        b %= per;
        J.fmt = FMT;
        J.ntok = d[1];
        if (J.ntok == 0) return;
        J.woff = (long long)(uint32_t)d[2] | (long long)d[3] << 32;
        J.soff = (long long)(uint32_t)d[4] | (long long)d[5] << 32;
        J.mul = __int_as_float(d[6]);
        J.rows = d + 8;
        J.orows = d + 8 + JT;
    }
    const int RB = (J.n + 64 * R - 1) / (64 * R);
    const int rb = b % RB, sl = b / RB;
    const int rt0 = (rb * 4 + wv) * R;
    if (rt0 * 16 >= J.n) return;
    const int KU = J.k / 128;
    const int u0 = KU * sl / J.split, u1 = KU * (sl + 1) / J.split;
    const int col = lane & 15, g8 = 8 * (lane >> 4);
    if (J.ntok == 1 && !XG_NOX1) {
        // one token: column-rotated tiles (16 k-blocks per MFMA, scales applied once per 16), as single-token decode
        const char* base = (const char*)J.w;
        const int xrow = J.rows[0];
        const uint8_t* xq1 = J.xq + (size_t)xrow * J.ldx + u0 * 128;
        const float* xs1 = J.xs + (size_t)xrow * (J.ldx / 32) + u0 * 4;
        const int S = u1 - u0;
#pragma unroll
        for (int r = 0; r < R; r++) {
            const int rt = rt0 + r;
            if (rt * 16 >= J.n) break;
            const u4v* sp = (const u4v*)(base + J.soff) + ((size_t)rt * KU + u0) * 4 + (lane >> 4);
            f4 acc;
            if (FMT == 1) acc = wave_x1g<0, X1U, true>((const uint32_t*)((const u4v*)(base + J.woff) + ((size_t)rt * KU + u0) * 64 + lane), sp, S, xq1, xs1, 0u, 0u, lane);
            else if (FMT == 2) acc = wave_x1g<3, X1U, true>((const uint32_t*)(base + J.woff) + ((size_t)rt * KU + u0) * 64 * 3 + lane * 3, sp, S, xq1, xs1, cb3lo, cb3hi, lane);
            else acc = wave_x1g<2, X1U, true>((const uint32_t*)(base + J.woff) + ((size_t)rt * KU + u0) * 64 * 2 + lane * 2, sp, S, xq1, xs1, cb2, 0u, lane);
#pragma unroll
            for (int j = 0; j < 4; j++) acc[j] = group_sum<16>(acc[j]);
            if (col == 0) {
                float* yp = J.y + (size_t)J.orows[0] * J.ldy + rt * 16 + 4 * (lane >> 4);
                for (int j = 0; j < 4; j++) atomicAdd(yp + j, acc[j] * J.mul);
            }
        }
        return;
    }
    const uint8_t* xq[G];
    const float* xs[G];
    float lm[G];
#pragma unroll
    for (int gi = 0; gi < G; gi++) {
        const int tok = gi * 16 + col;
        const bool live = tok < J.ntok;
        const int xrow = live ? J.rows[tok] : 0;
        xq[gi] = J.xq + (size_t)xrow * J.ldx;
        xs[gi] = J.xs + (size_t)xrow * (J.ldx / 32);
        lm[gi] = live ? 1.f : 0.f;
    }
    const char* base = (const char*)J.w;
    constexpr int NDW = FMT == 1 ? 4 : (FMT == 2 ? 3 : 2);
    const uint32_t tlo = FMT == 2 ? cb3lo : cb2, thi = FMT == 2 ? cb3hi : 0u;
    f4 acc[R][G];
#pragma unroll
    for (int r = 0; r < R; r++)
#pragma unroll
        for (int gi = 0; gi < G; gi++) acc[r][gi] = f4{0, 0, 0, 0};
    struct Unit { uint32_t w[R][4]; u4v s[R]; };
    Unit cu, nu;
    auto ld = [&](int u, Unit& o) {
#pragma unroll
        for (int r = 0; r < R; r++) {
            const int rt = min(rt0 + r, J.n / 16 - 1);
            const uint32_t* wp = (const uint32_t*)(base + J.woff) + ((size_t)rt * KU + u) * 64 * NDW + lane * NDW;
#pragma unroll
            for (int q = 0; q < 4; q++) o.w[r][q] = q < NDW ? __builtin_nontemporal_load(wp + q) : 0u;
            o.s[r] = ((const u4v*)(base + J.soff))[((size_t)rt * KU + u) * 4 + (lane >> 4)];
        }
    };
    if (u0 < u1) ld(u0, cu);
    for (int u = u0; u < u1; u++) {
        if (u + 1 < u1) ld(u + 1, nu);
        uint64_t xv[G][4];
        float xsv[G][4];
#pragma unroll
        for (int gi = 0; gi < G; gi++)
#pragma unroll
            for (int m = 0; m < 4; m++) {
                xv[gi][m] = lm[gi] != 0.f ? *reinterpret_cast<const uint64_t*>(xq[gi] + (4 * u + m) * 32 + g8) : 0ull;
                xsv[gi][m] = xs[gi][4 * u + m] * lm[gi];
            }
#pragma unroll
        for (int r = 0; r < R; r++) {
            const uint32_t ss[4] = {cu.s[r].x, cu.s[r].y, cu.s[r].z, cu.s[r].w};
            // the unit's 16 row scales (rows j, k-blocks m) as wsc[m][j]: packed FP8 -> f32 conversions (2 per
            // instruction) for the codebooks, exponent shifts for MXFP4; the products below use packed f32 math
            f4 wsc[4];
            if (FMT == 1) {
#pragma unroll
                for (int m = 0; m < 4; m++)
#pragma unroll
                    for (int j = 0; j < 4; j++) wsc[m][j] = __uint_as_float(((ss[j] >> (8 * m)) & 0xFFu) << 23);
            } else {
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    const f2v lo = __builtin_amdgcn_cvt_pk_f32_fp8(ss[j], false), hi = __builtin_amdgcn_cvt_pk_f32_fp8(ss[j], true);
                    wsc[0][j] = lo[0]; wsc[1][j] = lo[1]; wsc[2][j] = hi[0]; wsc[3][j] = hi[1];
                }
            }
#pragma unroll
            for (int m = 0; m < 4; m++) {
                uint64_t a;
                if (FMT == 1) a = fp4x8_to_fp8(cu.w[r][m]);
                else {
                    const uint32_t dl = cu.w[r][m >> 1];
                    const int sh = 4 * (m & 1);
                    uint32_t s0 = (dl >> sh) & 0x03030303u, s1 = (dl >> (sh + 2)) & 0x03030303u;
                    if (FMT == 2) {
                        s0 |= ((cu.w[r][2] >> (2 * m)) & 0x01010101u) << 2;
                        s1 |= ((cu.w[r][2] >> (2 * m + 1)) & 0x01010101u) << 2;
                    }
                    a = (uint64_t)perm(thi, tlo, s1) << 32 | perm(thi, tlo, s0);
                }
#pragma unroll
                for (int gi = 0; gi < G; gi++) {
                    const f4 dd = mfma_fp8(a, xv[gi][m], f4{0, 0, 0, 0});
                    acc[r][gi] += dd * (wsc[m] * xsv[gi][m]);
                }
            }
        }
        cu = nu;
    }
#pragma unroll
    for (int gi = 0; gi < G; gi++) {
        const int tok = gi * 16 + col;
        if (tok >= J.ntok) break;
        const int orow = J.orows[tok];
#pragma unroll
        for (int r = 0; r < R; r++) {
            const int rt = rt0 + r;
            if (rt * 16 >= J.n) break;
            float* yp = J.y + (size_t)orow * J.ldy + rt * 16 + 4 * (lane >> 4);
            for (int j = 0; j < 4; j++) atomicAdd(yp + j, acc[r][gi][j] * J.mul);
        }
    }
}
template <int R, int G>
DEV void vgemv_xg_impl(VJob J, int b, uint32_t cb3lo, uint32_t cb3hi, uint32_t cb2) {
    // the job's format, then the specialized body (uniform per wave: one job per block)
    const int per = J.blocks / J.ndyn;
    const int f = J.dyn[(size_t)(b / per) * (8 + 2 * 16 * G)];
    if (f == 1) vgemv_xg_body<R, G, 1>(J, b, cb3lo, cb3hi, cb2);
    else if (f == 2) vgemv_xg_body<R, G, 2>(J, b, cb3lo, cb3hi, cb2);
    else vgemv_xg_body<R, G, 3>(J, b, cb3lo, cb3hi, cb2);
}
#define VGEMV_XG(R, G)                                                                                        \
    extern "C" __global__ void __launch_bounds__(256) vgemv_xg##R##G(VJobs js) {                              \
        int b = blockIdx.x, ji = 0;                                                                           \
        while (ji < js.n - 1 && b >= js.j[ji].blocks) b -= js.j[ji++].blocks;                                \
        vgemv_xg_impl<R, G>(js.j[ji], b, js.cb3lo, js.cb3hi, js.cb2);                                         \
    }
VGEMV_XG(2, 1)
VGEMV_XG(2, 2)
VGEMV_XG(1, 4)
VGEMV_XG(2, 4)

#define VGEMV_XR(R)                                                                                           \
    extern "C" __global__ void __launch_bounds__(256) vgemv_xr##R(VJobs js) {                                 \
        int b = blockIdx.x, ji = 0;                                                                           \
        while (ji < js.n - 1 && b >= js.j[ji].blocks) b -= js.j[ji++].blocks;                                \
        vgemv_xr_impl<R>(js.j[ji], b, js.cb3lo, js.cb3hi, js.cb2);                                            \
    }
VGEMV_XR(2)
VGEMV_XR(4)

// Zero the outputs [ntok][n] (stride ldy) of the jobs flagged `zero`, before their vgemv launch
extern "C" __global__ void __launch_bounds__(256) vzero(VJobs js) {
    for (int ji = 0; ji < js.n; ji++) {
        if (!js.j[ji].zero) continue;
        float* y = js.j[ji].y;
        const int n = js.j[ji].n, ldy = js.j[ji].ldy, tot = js.j[ji].ntok * n;
        for (int i = blockIdx.x * 256 + threadIdx.x; i < tot; i += gridDim.x * 256) y[(size_t)(i / n) * ldy + i % n] = 0.f;
    }
}

// Commit of a verify pass: window rows of the kept positions p0 .. p0+keep into each layer's ring
extern "C" __global__ void commit_rings(const unsigned long long* rings, const uint16_t* vlin, int p0) {
    const int li = blockIdx.x, j = blockIdx.y;
    uint16_t* ring = reinterpret_cast<uint16_t*>(rings[li]);
    const uint32_t* src = reinterpret_cast<const uint32_t*>(vlin + ((size_t)li * 16 + j) * HD);
    uint32_t* dst = reinterpret_cast<uint32_t*>(ring + (size_t)((p0 + j) % WIN) * HD);
    for (int i = threadIdx.x; i < HD / 2; i += blockDim.x) dst[i] = src[i];
}

// Expert grouping for a verify batch (one block): picks sel[t][6] -> one job per distinct expert with its tokens.
// dyn layout per job: [fmt, ntok, woff lo, hi, soff lo, hi, mul bits, pad, rows[16] (activation rows = token),
// orows[16] (output rows = compact pair index t*6+slot)], for gate/up mats 0/1 and down mat 2 (3 job arrays of 36).
// Also writes the pair weights wrow[t*6+slot].
extern "C" __global__ void vgroup(int T, int K, const int* sel, const float* wts, const ExpertMat* tab, int* dyn, float* wrow, int* npairs,
                                  float* gg, float* uu, float* y) {
    __shared__ int ex[96], cnt[96], nex;
    __shared__ int members[96][16];
    __shared__ int ss[96], lead[96], gidx[96];
    const int N = T * K, i = threadIdx.x;
    if (i < N) {
        ss[i] = sel[i];
        wrow[i] = wts[i];
    }
    // the k-split expert GEMVs add into gate/up rows [N][INTER] and the down output y [T][D]: all blocks zero them,
    // block 0 also groups
    {
        const int gi = blockIdx.x * blockDim.x + i, gs = gridDim.x * blockDim.x;
        for (int q = gi; q < N * INTER; q += gs) { gg[q] = 0.f; uu[q] = 0.f; }
        for (int q = gi; q < T * D; q += gs) y[q] = 0.f;
    }
    if (blockIdx.x != 0) return;
    if (i == 0) *npairs = N;
    __syncthreads();
    if (i < N) {
        int L = 0;
        while (ss[L] != ss[i]) L++;
        lead[i] = L;
    }
    __syncthreads();
    if (i < N && lead[i] == i) {
        int g = 0, c = 0;
        for (int j = 0; j < i; j++) g += lead[j] == j;
        for (int j = i; j < N; j++) c += lead[j] == i;
        gidx[i] = g;
        ex[g] = ss[i];
        cnt[g] = c < 16 ? c : 16;
    }
    if (i == 0) {
        int n = 0;
        for (int j = 0; j < N; j++) n += lead[j] == j;
        nex = n;
    }
    __syncthreads();
    if (i < N) {
        const int L = lead[i];
        int r = 0;
        for (int j = L; j < i; j++) r += lead[j] == L;
        if (r < 16) members[gidx[L]][r] = i;
    }
    __syncthreads();
    for (int q = threadIdx.x; q < 3 * 48; q += blockDim.x) {
        const int m = q / 48, j = q % 48;
        int* d = dyn + (size_t)(m * 48 + j) * 40;
        if (j >= nex) { d[1] = 0; continue; }
        const ExpertMat em = tab[ex[j] * 3 + m];
        d[0] = em.fmt == E_FP4 ? 1 : (em.fmt == E_CB3 ? 2 : 3);
        d[1] = cnt[j];
        d[2] = (int)(em.woff & 0xFFFFFFFF); d[3] = (int)(em.woff >> 32);
        d[4] = (int)(em.soff & 0xFFFFFFFF); d[5] = (int)(em.soff >> 32);
        d[6] = __float_as_int(em.mul);
        for (int i = 0; i < 16; i++) {
            const int pr = i < cnt[j] ? members[j][i] : 0;
            // gate/up: activation row = token, output row = pair; down: activation row = pair, output row = token
            d[8 + i] = m < 2 ? pr / K : pr;
            d[24 + i] = m < 2 ? pr : pr / K;
        }
    }
}

// Device grouping of N = T*K (token, pick) pairs (up to BG_PAIRS) by expert, for batched decode and prompt chunks.
// One job per `jt` of an expert's pairs (16, 32 or 64; the hottest experts get several). Job record (JS = 8 + 2*jt ints):
// [fmt, ntok, woff lo/hi, soff lo/hi, mul bits, pad, rows[jt], orows[jt]]; jobs [3][maxj] (gate, up, down), unused
// ones with ntok 0. With jt = 16 this is vgroup's layout (vgemv_x reads it). All blocks zero the k-split outputs
// (gate/up rows [N][INTER], y [T][D]); block 0 groups. npairs[0] = N, npairs[1] = job count. Launch: block 1024.
#ifndef BG_PAIRS
#define BG_PAIRS 12288
#endif
extern "C" __global__ void __launch_bounds__(1024) bgroup(int T, int K, const int* sel, const float* wts, const ExpertMat* tab, int* dyn,
                                                          float* wrow, int* npairs, float* gg, float* uu, float* y, int jt, int maxj) {
    __shared__ int cnt[NE], base[NE + 1];
    __shared__ int rank[BG_PAIRS];
    const int N = T * K, i = threadIdx.x, JS = 8 + 2 * jt;
    {
        const int gi = blockIdx.x * blockDim.x + i, gs = gridDim.x * blockDim.x;
        for (int q = gi; q < N * INTER; q += gs) { gg[q] = 0.f; uu[q] = 0.f; }
        for (int q = gi; q < T * D; q += gs) y[q] = 0.f;
    }
    if (blockIdx.x != 0) return;
    if (i == 0) *npairs = N;
    for (int e = i; e < NE; e += blockDim.x) cnt[e] = 0;
    __syncthreads();
    for (int p = i; p < N; p += blockDim.x) {
        wrow[p] = wts[p];
        rank[p] = atomicAdd(&cnt[sel[p]], 1);
    }
    __syncthreads();
    // job base per expert (one job per started jt pairs): exclusive scan over NE <= 1024 threads
    int v = i < NE ? (cnt[i] + jt - 1) / jt : 0;
    {
        __shared__ int sc[1024];
        sc[i] = v;
        __syncthreads();
        for (int o = 1; o < 1024; o <<= 1) {
            const int a = i >= o ? sc[i - o] : 0;
            __syncthreads();
            sc[i] += a;
            __syncthreads();
        }
        if (i < NE) base[i] = sc[i] - v;
        if (i == NE - 1) { base[NE] = sc[i]; npairs[1] = sc[i]; }
    }
    __syncthreads();
    const int nj = base[NE];
    for (int q = i; q < 3 * maxj; q += blockDim.x) {
        const int m = q / maxj, j = q % maxj;
        int* d = dyn + (size_t)q * JS;
        if (j >= nj) { d[1] = 0; continue; }
        int lo = 0, hi = NE - 1;   // expert of job j: the last e with base[e] <= j
        while (lo < hi) { const int mid = (lo + hi + 1) >> 1; if (base[mid] <= j) lo = mid; else hi = mid - 1; }
        const int e = lo, ch = j - base[e];
        const ExpertMat em = tab[e * 3 + m];
        d[0] = em.fmt == E_FP4 ? 1 : (em.fmt == E_CB3 ? 2 : 3);
        d[1] = min(jt, cnt[e] - jt * ch);
        d[2] = (int)(em.woff & 0xFFFFFFFF); d[3] = (int)(em.woff >> 32);
        d[4] = (int)(em.soff & 0xFFFFFFFF); d[5] = (int)(em.soff >> 32);
        d[6] = __float_as_int(em.mul);
    }
    // members: pair p goes to job base[e] + rank/jt, column rank%jt
    for (int p = i; p < N; p += blockDim.x) {
        const int e = sel[p], j = base[e] + rank[p] / jt, c = rank[p] % jt;
        for (int m = 0; m < 3; m++) {
            int* d = dyn + (size_t)(m * maxj + j) * JS;
            // gate/up: activation row = token, output row = pair; down: activation row = pair, output row = token
            d[8 + c] = m < 2 ? p / K : p;
            d[8 + jt + c] = m < 2 ? p : p / K;
        }
    }
}

// Batched decode, dense matrices: vgemv's layout (one wave per 16-row tile and k-slice) over up to 16 * G tokens in
// G groups of 16; each weight fragment is loaded once and multiplied into every group (accumulators acc[G]), so the
// dense weights stream once per step whatever the batch. Static jobs only (fmt 0 FP8, fmt 4 BF16; no epilogue).
template <int G>
DEV void bwave_dense(const VJob& J, int rt, int u0, int u1, int lane, f4 (&acc)[G]) {
    const int col = lane & 15, g8 = 8 * (lane >> 4), g4 = 4 * (lane >> 4);
#pragma unroll
    for (int gi = 0; gi < G; gi++) acc[gi] = f4{0, 0, 0, 0};
    if (J.fmt == 0) {
        const int KP = J.k / 64, KB = J.k / 32;
        const u4v* wp = (const u4v*)J.w + (size_t)rt * KP * 64 + lane;
        const uint8_t* sp = (const uint8_t*)J.sc + (size_t)(rt >> 1) * KB;
        for (int kp = u0; kp < u1; kp++) {
            const u4v w = __builtin_nontemporal_load(wp + (size_t)kp * 64);
            const float ws0 = __uint_as_float((uint32_t)sp[2 * kp] << 23), ws1 = __uint_as_float((uint32_t)sp[2 * kp + 1] << 23);
            const uint64_t a0 = (uint64_t)w.y << 32 | w.x, a1 = (uint64_t)w.w << 32 | w.z;
#pragma unroll
            for (int gi = 0; gi < G; gi++) {
                const int tok = gi * 16 + col;
                const bool live = tok < J.ntok;
                const int xrow = live ? (J.rows ? J.rows[tok] : tok) : 0;
                const uint8_t* xq = J.xq + (size_t)xrow * J.ldx;
                const float* xs = J.xs + (size_t)xrow * (J.ldx / 32);
                const uint64_t x0 = live ? *reinterpret_cast<const uint64_t*>(xq + kp * 64 + g8) : 0ull;
                const uint64_t x1 = live ? *reinterpret_cast<const uint64_t*>(xq + kp * 64 + 32 + g8) : 0ull;
                const float s0 = live ? xs[2 * kp] * ws0 : 0.f, s1 = live ? xs[2 * kp + 1] * ws1 : 0.f;
                const f4 d0 = mfma_fp8(a0, x0, f4{0, 0, 0, 0});
                const f4 d1 = mfma_fp8(a1, x1, f4{0, 0, 0, 0});
                acc[gi] += d0 * s0 + d1 * s1;
            }
        }
#pragma unroll
        for (int gi = 0; gi < G; gi++) acc[gi] *= 2.f;
        return;
    }
    // fmt 4: BF16 weights, bf16 activations
    const int KQ = J.k / 32;
    const u4v* wp = (const u4v*)J.w + (size_t)rt * KQ * 64 + lane;
    for (int kq = u0; kq < u1; kq++) {
        const u4v w = __builtin_nontemporal_load(wp + (size_t)kq * 64);
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const uint32_t w0 = h ? w.z : w.x, w1 = h ? w.w : w.y;
            s4v a;
            a[0] = (short)(w0 & 0xFFFF); a[1] = (short)(w0 >> 16); a[2] = (short)(w1 & 0xFFFF); a[3] = (short)(w1 >> 16);
#pragma unroll
            for (int gi = 0; gi < G; gi++) {
                const int tok = gi * 16 + col;
                const bool live = tok < J.ntok;
                const int xrow = live ? (J.rows ? J.rows[tok] : tok) : 0;
                const uint64_t bv = live ? *reinterpret_cast<const uint64_t*>(J.xb + (size_t)xrow * J.ldx + kq * 32 + 16 * h + g4) : 0ull;
                acc[gi] = mfma_bf16(a, *reinterpret_cast<const s4v*>(&bv), acc[gi]);
                if (J.xl) {
                    const uint64_t lv = live ? *reinterpret_cast<const uint64_t*>(J.xl + (size_t)xrow * J.ldx + kq * 32 + 16 * h + g4) : 0ull;
                    acc[gi] = mfma_bf16(a, *reinterpret_cast<const s4v*>(&lv), acc[gi]);
                }
            }
        }
    }
}

template <int G>
DEV void bgemv_impl(const VJob& J, int b) {
    const int lane = threadIdx.x & 63, wv = threadIdx.x >> 6;
    const int RB = (J.n + 63) / 64;
    const int rb = b % RB, sl = b / RB;
    const int rt = rb * 4 + wv;
    if (rt * 16 >= J.n) return;
    const int KU = J.fmt == 0 ? 64 : 32;
    const int KUs = J.k / KU;
    const int u0 = KUs * sl / J.split, u1 = KUs * (sl + 1) / J.split;
    f4 acc[G];
    bwave_dense<G>(J, rt, u0, u1, lane, acc);
    const int col = lane & 15, r0 = rt * 16 + 4 * (lane >> 4);
#pragma unroll
    for (int gi = 0; gi < G; gi++) {
        const int tok = gi * 16 + col;
        if (tok >= J.ntok) break;
        const int orow = J.orows ? J.orows[tok] : tok;
        float* yp = J.y + (size_t)orow * J.ldy + r0;
        for (int j = 0; j < 4; j++) {
            if (J.atomic) atomicAdd(yp + j, acc[gi][j]);
            else yp[j] = acc[gi][j];
        }
    }
}

#define BGEMV(G)                                                                                              \
    extern "C" __global__ void __launch_bounds__(256) bgemv_d##G(VJobs js) {                                  \
        int b = blockIdx.x, ji = 0;                                                                           \
        while (ji < js.n - 1 && b >= js.j[ji].blocks) b -= js.j[ji++].blocks;                                 \
        bgemv_impl<G>(js.j[ji], b);                                                                           \
    }
BGEMV(2)
BGEMV(4)
BGEMV(8)
BGEMV(16)
