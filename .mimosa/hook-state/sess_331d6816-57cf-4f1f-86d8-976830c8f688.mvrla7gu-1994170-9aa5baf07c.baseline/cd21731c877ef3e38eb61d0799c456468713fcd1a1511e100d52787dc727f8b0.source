# Derived from Xolotl tools/v41 (github.com/Indras-Mirror/Xolotl, commit 7375e305), MIT License,
# Copyright (c) 2026 Niko1221 and the Strata contributors; see LICENSE-Xolotl. Modified for dsx.
"""v41_gptq.py - GPTQ over the three expert tiers of the V4.1 mixed-tier plan (V41_PLAN.md), exact-format grids.

Tiers (grid integers x one scale per group; the scale may be negative, which flips an asymmetric grid):
  ptq1  {-1,0,1}        x fp16 per 128           PTQ1_0 (id 143), 1.75 bpw
  q2_0  {-1,0,1,2}      x fp16 per 64            Q2_0 (GGUF block: 00=-1 .. 11=+2), 2.25 bpw
  q3_k  {-4..3}         x (d * (sc-32)) per 16   Q3_K: fp16 d per 256, 6-bit sub-scales sc, 3.4375 bpw
Scales are chosen BEFORE GPTQ ("static groups") from the original weights by a diag(H)-weighted search over clip
ratios and both signs, then rounded to what the format stores (fp16 / Q3_K d*sc), so every GPTQ result is exactly
representable in its GGUF type.  GPTQ then only decides the integers (error feedback through H^-1), so this is the
init GSQ refines later.

Rows of different tiers can be stacked into one GPTQ run (same columns / same Hessian): per-row grid bounds and a
per-(row, column) scale.  gptq() is the plain blocked algorithm (Frantar et al.), no act-order.
"""
import torch

TIERS = {
    #        lo, hi, group, bits/weight
    "ptq1": (-1, 1, 128, 1.75),
    "q2_0": (-1, 2, 64, 2.25),
    "q3_k": (-4, 3, 16, 3.4375),
}
CLIP = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]


def _rtn(w, s, lo, hi):
    return torch.clamp(torch.round(w / s), lo, hi) * s


def search_scales(W, d, tier):
    """W [R, C] fp32, d [C] importance (diag H).  Returns scales [R, C/group] fp32 already in the stored format."""
    lo, hi, G, _ = TIERS[tier]
    R, C = W.shape
    g = W.view(R, C // G, G)
    dg = d.view(1, C // G, G)
    amax = g.abs().amax(-1, keepdim=True).clamp_min(1e-12)
    best_e = None
    best_s = None
    denoms = sorted({hi, -lo})
    for sign in (1.0, -1.0) if lo != -hi else (1.0,):
        for den in denoms:
            for c in CLIP:
                s = sign * amax * c / den
                e = ((g - _rtn(g, s, lo, hi)).square() * dg).sum(-1, keepdim=True)
                if best_e is None:
                    best_e, best_s = e, s
                else:
                    m = e < best_e
                    best_e = torch.where(m, e, best_e)
                    best_s = torch.where(m, s, best_s)
    s = best_s.squeeze(-1)                                  # [R, C/G]
    if tier == "q3_k":                                      # 16 sub-scales per 256 -> fp16 d * int in [-32, 31]
        sb = s.view(R, C // 256, 16)
        dsup = (sb.abs().amax(-1, keepdim=True) / 31.0).half().float().clamp_min(1e-30)
        sc = torch.clamp(torch.round(sb / dsup), -32, 31)
        s = (dsup * sc).view(R, C // G)
    else:
        s = s.half().float()
    return torch.where(s == 0, torch.full_like(s, 1e-30), s)


def expand_scales(s, tier, C):
    return s.repeat_interleave(TIERS[tier][2], dim=1)[:, :C]


def rtn(W, S, lo, hi):
    """W, S [R, C]; lo/hi [R, 1]."""
    return torch.minimum(torch.maximum(torch.round(W / S), lo), hi) * S


def hinv_chol(H, damp=0.01):
    H = H.clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0
    H += damp * torch.mean(torch.diag(H)) * torch.eye(H.shape[0], device=H.device, dtype=H.dtype)
    L = torch.linalg.cholesky(H)
    Hi = torch.cholesky_inverse(L)
    return torch.linalg.cholesky(Hi, upper=True), dead


def gptq(W, S, lo, hi, Hinv, dead, block=128):
    """W [R, C] fp32 (not modified), S [R, C] scales, lo/hi [R, 1], Hinv upper Cholesky of H^-1.  Returns Q [R, C]."""
    W = W.clone()
    W[:, dead] = 0
    R, C = W.shape
    Q = torch.zeros_like(W)
    for i1 in range(0, C, block):
        i2 = min(i1 + block, C)
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        E1 = torch.zeros_like(W1)
        Hi1 = Hinv[i1:i2, i1:i2]
        S1 = S[:, i1:i2]
        for i in range(i2 - i1):
            w = W1[:, i]
            q = torch.minimum(torch.maximum(torch.round(w / S1[:, i]), lo[:, 0]), hi[:, 0]) * S1[:, i]
            Q1[:, i] = q
            err = (w - q) / Hi1[i, i]
            W1[:, i:] -= err.unsqueeze(1) * Hi1[i, i:].unsqueeze(0)
            E1[:, i] = err
        Q[:, i1:i2] = Q1
        W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    return Q


def quantize_stack(Ws, H, tiers, damp=0.01, do_gptq=True):
    """Quantize each W in Ws (all [r_i, C], sharing H [C, C]) at every tier in `tiers`, in ONE stacked GPTQ run.
    Returns {tier: [Q_i ...]} for GPTQ and the same for RTN (static scales, no error feedback)."""
    C = H.shape[0]
    d = torch.diag(H).clamp_min(0) + 1e-8
    rows, S, lo, hi, spans = [], [], [], [], []
    r0 = 0
    for t in tiers:
        l, h_, _, _ = TIERS[t]
        for W in Ws:
            s = expand_scales(search_scales(W, d, t), t, C)
            rows.append(W); S.append(s)
            lo.append(torch.full((W.shape[0], 1), float(l), device=W.device))
            hi.append(torch.full((W.shape[0], 1), float(h_), device=W.device))
            spans.append((t, r0, r0 + W.shape[0])); r0 += W.shape[0]
    Wst, Sst, lost, hist = torch.cat(rows), torch.cat(S), torch.cat(lo), torch.cat(hi)
    Qr = rtn(Wst, Sst, lost, hist)
    Qg = None
    if do_gptq:
        Hinv, dead = hinv_chol(H, damp)
        Qg = gptq(Wst, Sst, lost, hist, Hinv, dead)
    out_g, out_r = {t: [] for t in tiers}, {t: [] for t in tiers}
    for t, a, b in spans:
        out_r[t].append(Qr[a:b])
        if Qg is not None:
            out_g[t].append(Qg[a:b])
    return (out_g if do_gptq else None), out_r
