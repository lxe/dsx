"""v41_iq2.py - IQ2_XXS experts for the V4.1 quantizer: ggml's format and codebook (ggml / llama.cpp, MIT;
v41_iqgrid.py), quantized by codebook GPTQ as in GPTVQ (van Baalen et al., arXiv 2402.15319) and QuIP# (Tseng et al.,
arXiv 2402.04396): each group of 8 columns rounds jointly to the 256-point grid with even-parity signs, then its error
is fed back column by column as in scalar GPTQ. Kernels: native/qkern.cu (iq2_scales_k, gptq_iq2_block_k).

Block (256 weights, 66 bytes): fp16 d; per 32 weights a u32 of four grid indices (bytes) and a u32 of four 7-bit sign
patterns (bits 7l..7l+7) with the 4-bit scale ls in bits 28..32. Weight = grid magnitude * d * (0.5 + ls) / 4, negated
where the pattern (plus its parity bit) has the weight's bit set.

Scales: per 32 columns the best db = amax / c over CAND (d-weighted error of the rounded block, first best), then
stored as d = fp16(max db of the 256-block * 4 / 15.5) and ls = clamp(round(4 db / d - 0.5), 0, 15).
Same rules as quant/experiments/quant_variants.py IQ2XXS (the torch reference)."""
import ctypes
import os

import numpy as np
import torch

import v41_iqgrid as T

IQ2_XXS = 16
CAND = (30.0, 34.0, 38.0, 43.0, 48.0, 55.0, 64.0)
BLOCK = 128

_lib = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "native", "libqkern.so"))
_lib.ds41q_iq2_scales.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                                  ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
_lib.ds41q_gptq_iq2_block.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int] + [ctypes.c_void_p] * 8 + [ctypes.c_int, ctypes.c_int, ctypes.c_size_t]
_cache = {}


def _p(t):
    assert t.is_cuda and t.is_contiguous()
    return ctypes.c_void_p(t.data_ptr())


def _stream():
    return torch.cuda.current_stream().cuda_stream


def grid(dev):
    """[256, 8] float magnitudes on dev."""
    k = ("grid", str(dev))
    if k not in _cache:
        g = torch.tensor([[(v >> (8 * j)) & 0xFF for j in range(8)] for v in T.IQ2XXS_GRID], dtype=torch.float32)
        _cache[k] = g.to(dev).contiguous()
    return _cache[k]


def search_scales(W, d):
    """W [R, C], d [C] (diag H) -> (dsup [R, C/256], ls [R, C/32] int, db [R, C/32] effective scales)."""
    R, C = W.shape
    dev = W.device
    ck = ("cand", str(dev))
    if ck not in _cache:
        _cache[ck] = torch.tensor(CAND, device=dev, dtype=torch.float32)
    out = torch.empty((R, C // 32), device=dev, dtype=torch.float32)
    rc = _lib.ds41q_iq2_scales(_p(W.float().contiguous()), _p(d.float().contiguous()), R, C, _p(grid(dev)), _p(_cache[ck]),
                               len(CAND), _p(out), _stream())
    if rc:
        raise RuntimeError(f"iq2_scales: CUDA error {rc}")
    blk = out.view(R, C // 256, 8)
    dsup = (blk.amax(-1, keepdim=True) * 4 / 15.5).half().float().clamp_min(1e-30)
    ls = torch.clamp(torch.round(blk * 4 / dsup - 0.5), 0, 15)
    db = (dsup * (0.5 + ls) / 4).view(R, C // 32)
    return dsup.view(R, C // 256), ls.view(R, C // 32).to(torch.int32), db


def gptq_batched(W, DB, d, Hinv, dead=None):
    """W [B, R, C]; DB [B, R, C/32] effective scales; d [B, C] rounding weights (diag H); Hinv [B, C, C] upper
    Cholesky of H^-1. Returns Q [B, R, C] (values), IDX, SGN [B, R, C/8] uint8 (grid index, 7-bit sign pattern)."""
    W = W.float().clone().contiguous()
    if dead is not None and dead.any():
        W.masked_fill_(dead[:, None, :], 0.0)
    B, R, C = W.shape
    assert C % BLOCK == 0
    DB, d, Hinv = DB.float().contiguous(), d.float().contiguous(), Hinv.float().contiguous()
    Q = torch.empty_like(W)
    E = torch.empty((B, R, BLOCK), device=W.device, dtype=torch.float32)
    IDX = torch.empty((B, R, C // 8), device=W.device, dtype=torch.uint8)
    SGN = torch.empty_like(IDX)
    g, st = grid(W.device), _stream()
    for i1 in range(0, C, BLOCK):
        i2 = i1 + BLOCK
        rc = _lib.ds41q_gptq_iq2_block(_p(Hinv), C, i1, _p(W), _p(DB), _p(d), _p(g), _p(Q), _p(E), _p(IDX), _p(SGN), R, B, st)
        if rc:
            raise RuntimeError(f"gptq_iq2_block: CUDA error {rc}")
        if i2 < C:
            W[:, :, i2:].baddbmm_(E, Hinv[:, i1:i2, i2:], alpha=-1.0)
    return Q, IDX, SGN


def encode(IDX, SGN, ls, dsup):
    """IDX, SGN [R, C/8]; ls [R, C/32]; dsup [R, C/256] (fp16-exact) -> GGUF rows, uint8 [R, C/256 * 66]."""
    IDX, SGN = np.asarray(IDX, np.uint32), np.asarray(SGN, np.uint32)
    ls, dsup = np.asarray(ls, np.uint32), np.asarray(dsup, np.float32)
    R, n8 = IDX.shape
    nb = n8 // 32
    i4, s4 = IDX.reshape(R, nb * 8, 4), SGN.reshape(R, nb * 8, 4)
    g = i4[..., 0] | i4[..., 1] << 8 | i4[..., 2] << 16 | i4[..., 3] << 24
    s = s4[..., 0] | s4[..., 1] << 7 | s4[..., 2] << 14 | s4[..., 3] << 21 | ls.reshape(R, nb * 8) << 28
    qs = np.stack([g, s], -1).astype("<u4").reshape(R, nb, 64 // 4).view(np.uint8).reshape(R, nb, 64)
    d = dsup.astype(np.float16).reshape(R, nb, 1).view(np.uint8)
    return np.concatenate([d, qs], -1).reshape(R, nb * 66)


def decode(raw, C):
    """GGUF rows -> float [R, C] (reference, as formats::dequant_row)."""
    raw = np.asarray(raw, np.uint8)
    R = raw.shape[0]
    nb = C // 256
    b = raw.reshape(R, nb, 66)
    d = b[..., :2].copy().view(np.float16).astype(np.float32)[..., 0]
    u = b[..., 2:].copy().view("<u4").reshape(R, nb, 8, 2)
    g, s = u[..., 0].astype(np.int64), u[..., 1].astype(np.int64)
    gt = np.array(T.IQ2XXS_GRID, dtype=np.uint64)
    ks = np.array(T.KSIGNS_IQ2XS, dtype=np.int64)
    out = np.zeros((R, nb, 8, 4, 8), np.float32)
    db = d[..., None] * (0.5 + (s >> 28)) * 0.25
    for l in range(4):
        e = gt[(g >> (8 * l)) & 255]
        p = ks[(s >> (7 * l)) & 127]
        for j in range(8):
            m = ((e >> np.uint64(8 * j)) & np.uint64(255)).astype(np.float32)
            out[..., l, j] = np.where((p >> j) & 1, -1.0, 1.0) * db * m
    return out.reshape(R, C)


def store_scales(db):
    """Ideal scales [R, C/32] -> the stored form: (dsup [R, C/256], ls [R, C/32] int, effective db [R, C/32])."""
    R, nb = db.shape
    blk = db.clamp_min(0).view(R, nb // 8, 8)
    dsup = (blk.amax(-1, keepdim=True) * 4 / 15.5).half().float().clamp_min(1e-30)
    ls = torch.clamp(torch.round(blk * 4 / dsup - 0.5), 0, 15)
    return dsup.view(R, nb // 8), ls.view(R, nb).to(torch.int32), (dsup * (0.5 + ls) / 4).view(R, nb)


def refit(W, Q, db, H, old=None, chunk=256):
    """Least-squares scale per 32 columns for GPTQ's fixed codes (Q / db), minimizing tr((W - Q') H (W - Q')^T) per row,
    then stored (store_scales). With old = (dsup, ls) (the stored form of db), a row keeps its old scales where the
    stored refit is not better: the LS optimum is for unconstrained scales, and an ill-conditioned H can give one block a
    wild scale that the shared fp16 d (max of the 256-block) then forces onto the others' 4-bit ls.
    Returns (Q', dsup, ls, db')."""
    R, C = W.shape
    nb = C // 32
    code = Q / db.repeat_interleave(32, 1)            # signed grid magnitudes
    s = torch.empty_like(db)
    for r0 in range(0, R, chunk):
        r1 = min(R, r0 + chunk)
        k = r1 - r0
        Qg = torch.zeros(k, C, nb, device=W.device)
        Qg.view(k, nb, 32, nb).diagonal(dim1=1, dim2=3).copy_(code[r0:r1].view(k, nb, 32).transpose(1, 2))
        HQ = torch.einsum("ij,kjg->kig", H, Qg)
        A = torch.einsum("kig,kih->kgh", Qg, HQ) + 1e-6 * torch.eye(nb, device=W.device)
        b = torch.einsum("kig,ki->kg", HQ, W[r0:r1])
        s[r0:r1] = torch.linalg.solve(A, b.unsqueeze(-1)).squeeze(-1)
    dsup, ls, db2 = store_scales(s)
    Q2 = code * db2.repeat_interleave(32, 1)
    if old is not None:
        rowerr = lambda q: (((W - q) @ H) * (W - q)).sum(1)
        keep = rowerr(Q2) >= rowerr(Q)
        dsup, ls, db2 = torch.where(keep[:, None], old[0], dsup), torch.where(keep[:, None], old[1], ls), torch.where(keep[:, None], db, db2)
        Q2 = torch.where(keep[:, None], Q, Q2)
    return Q2, dsup, ls, db2


def quantize(W, H, damp=1.0, iters=0):
    """One matrix end to end: scales, codebook GPTQ, and `iters` rounds of refit + GPTQ on the refit scales, then a
    final refit when iters > 0. Returns (Q, IDX, SGN, ls, dsup)."""
    d = torch.diag(H).clamp_min(0) + 1e-8
    dsup, ls, db = search_scales(W, d)
    Hinv, dead = __import__("v41_gptq").hinv_chol(H, damp)
    Q, IDX, SGN = (x[0] for x in gptq_batched(W[None], db[None], d[None], Hinv[None], dead[None]))
    for _ in range(iters):
        _, dsup, ls, db = refit(W, Q, db, H, (dsup, ls))
        Q, IDX, SGN = (x[0] for x in gptq_batched(W[None], db[None], d[None], Hinv[None], dead[None]))
    if iters:
        Q, dsup, ls, db = refit(W, Q, db, H, (dsup, ls))
    return Q, IDX, SGN, ls, dsup
