"""v41_native.py - native (CUDA) GPTQ and scale search for v41_quant.py, batched over experts (native/qkern.cu).

gptq_batched: blocked GPTQ for B experts at once; the 128-column block runs in one kernel (a warp per row, the block
in registers, H^-1 block in shared memory), the trailing update is one batched GEMM per block. Same algorithm and
rounding as v41_gptq.gptq (round-half-even, per-row grid bounds, per-column scales); float results match to the
last few ulps (FMA contraction), which can flip an integer at an exact rounding tie.
search_scales: v41_gptq.search_scales in one kernel pass (same candidates, same first-best tie rule), then the same
fp16 / Q3_K sub-scale rounding."""
import ctypes
import os

import torch

import v41_gptq as G

_lib = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "native", "libqkern.so"))
_lib.ds41q_gptq_block.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int] + [ctypes.c_void_p] * 6 + [ctypes.c_int, ctypes.c_int, ctypes.c_size_t]
_lib.ds41q_search_scales.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float,
                                     ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
_clip = {}
BLOCK = 128


def _p(t):
    assert t.is_cuda and t.dtype == torch.float32 and t.is_contiguous()
    return ctypes.c_void_p(t.data_ptr())


def _stream():
    return torch.cuda.current_stream().cuda_stream


def gptq_batched(W, S, lo, hi, Hinv, dead=None):
    """W, S [B, R, C]; lo, hi [B, R]; Hinv [B, C, C] upper Cholesky of H^-1; dead [B, C] bool. Returns Q [B, R, C]."""
    # model.py makes bf16 the default dtype: everything here is explicitly fp32
    W = W.float().clone().contiguous()
    if dead is not None and dead.any():
        W.masked_fill_(dead[:, None, :], 0.0)
    B, R, C = W.shape
    assert C % BLOCK == 0
    S, lo, hi, Hinv = S.float().contiguous(), lo.float().contiguous(), hi.float().contiguous(), Hinv.float().contiguous()
    Q = torch.empty_like(W)
    E = torch.empty((B, R, BLOCK), device=W.device, dtype=torch.float32)
    st = _stream()
    for i1 in range(0, C, BLOCK):
        i2 = i1 + BLOCK
        rc = _lib.ds41q_gptq_block(_p(Hinv), C, i1, _p(W), _p(S), _p(lo), _p(hi), _p(Q), _p(E), R, B, st)
        if rc:
            raise RuntimeError(f"gptq_block: CUDA error {rc}")
        if i2 < C:
            W[:, :, i2:].baddbmm_(E, Hinv[:, i1:i2, i2:], alpha=-1.0)
    return Q


def search_scales(W, d, tier):
    """Drop-in for v41_gptq.search_scales: W [R, C], d [C] -> scales [R, C/group] in the stored format."""
    lo, hi, Gs, _ = G.TIERS[tier]
    R, C = W.shape
    dev = W.device
    clip = _clip.get(dev)
    if clip is None:
        clip = _clip[dev] = torch.tensor(G.CLIP, device=dev, dtype=torch.float32)
    out = torch.empty((R, C // Gs), device=dev, dtype=torch.float32)
    rc = _lib.ds41q_search_scales(_p(W.float().contiguous()), _p(d.float().contiguous()), R, C, Gs, float(lo), float(hi), _p(clip), len(G.CLIP), _p(out), _stream())
    if rc:
        raise RuntimeError(f"search_scales: CUDA error {rc}")
    s = out
    if tier == "q3_k":
        sb = s.view(R, C // 256, 16)
        dsup = (sb.abs().amax(-1, keepdim=True) / 31.0).half().float().clamp_min(1e-30)
        sc = torch.clamp(torch.round(sb / dsup), -32, 31)
        s = (dsup * sc).view(R, C // Gs)
    else:
        s = s.half().float()
    return torch.where(s == 0, torch.full_like(s, 1e-30), s)
