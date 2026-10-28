# From Xolotl tools/v41 (github.com/Indras-Mirror/Xolotl, commit 7375e305), MIT License,
# Copyright (c) 2026 Niko1221 and the Strata contributors; see ../LICENSE-Xolotl. Unchanged apart from this header.
"""Torch-only stand-in for DeepSeek-V4.1-Flash `inference/kernel.py` (tilelang).

Same math as the tilelang kernels, written as dequantize -> fp32 matmul so it runs on any CUDA GPU (sm_89 here) or the
CPU, with no tilelang/JIT.  Put this directory first on sys.path before importing DeepSeek's model.py.

Rounding rules copied from the kernels:
  act_quant      per-block amax (floor 1e-4), scale = amax/448 or 2^ceil(log2(amax/448)) (ue8m0), x/s clamped to
                 +-448 and cast to E4M3; inplace=True writes the dequantized value back.
  fp4_act_quant  E8M0 scale 2^ceil(log2(amax/6)) (amax floor 6*2^-126) or E4M3 scale e4m3(amax/6) (floor 6*2^-9);
                 x/s clamped to +-6 and rounded to E2M1 (round-to-nearest-even); only inplace=True is used by model.py.
  fp8/fp4_gemm   sum over K blocks of (A_blk . B_blk^T) * s_a * s_b in fp32, output in the default dtype.
  sparse_attn    gather by index (-1 = masked), fp32 scores * scale, softmax with a per-head sink logit in the
                 denominator, probabilities cast to bf16 before P.V (as the kernel does), max floor -1e30.
  hc_split_sinkhorn  pre = sigmoid(.)+eps, post = 2 sigmoid(.), comb = softmax rows + eps, then column/row Sinkhorn.
"""
import torch
from typing import Optional

FP8_MAX = 448.0
FP4_MAX = 6.0
# E2M1 code -> value (low nibble = even index, as in the checkpoint's int8-packed experts and torch float4_e2m1fn_x2)
FP4_LUT = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float32)
_E2M1_POS = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32)


def _pow2_ceil(v: torch.Tensor) -> torch.Tensor:
    """2^ceil(log2(v)) exactly, via the float32 exponent (the kernel's fast_round_scale)."""
    m, e = torch.frexp(v)                     # v = m * 2^e, m in [0.5, 1)
    e = torch.where(m == 0.5, e - 1, e)       # exact powers of two stay put
    return torch.ldexp(torch.ones_like(v), e)


def _round_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Round |x| <= 6 to the nearest E2M1 value, ties to the even code (what a hardware cvt to fp4 does)."""
    a = x.abs()
    grid = _E2M1_POS.to(x.device)
    d = (a.unsqueeze(-1) - grid).abs()
    best = d.min(-1, keepdim=True).values
    cand = (d == best)
    # tie -> even code index (codes 0,2,4,6 have an even mantissa bit)
    idx = torch.arange(8, device=x.device)
    pick = torch.where(cand & (idx % 2 == 0), idx, torch.where(cand, idx + 8, torch.full_like(idx, 99)))
    code = pick.min(-1).values % 8
    return torch.copysign(grid[code], x)


def _blocks(x: torch.Tensor, bs: int):
    return x.unflatten(-1, (-1, bs))


def act_quant(x: torch.Tensor, block_size: int = 128, scale_fmt: Optional[str] = None,
              scale_dtype: torch.dtype = torch.float32, inplace: bool = False):
    xf = _blocks(x.float(), block_size)
    amax = xf.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    s = _pow2_ceil(amax / FP8_MAX) if scale_fmt is not None else amax / FP8_MAX
    q = (xf / s).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    if inplace:
        x.copy_((q.float() * s).flatten(-2).to(x.dtype))
        return x
    return q.flatten(-2), s.squeeze(-1).to(scale_dtype)


def fp4_act_quant(x: torch.Tensor, block_size: int = 32, inplace: bool = False,
                  scale_dtype: torch.dtype = torch.float8_e8m0fnu):
    assert scale_dtype in (torch.float8_e8m0fnu, torch.float8_e4m3fn)
    assert inplace, "model.py only uses the in-place (quantize-dequantize) form"
    xf = _blocks(x.float(), block_size)
    amax = xf.abs().amax(-1, keepdim=True)
    if scale_dtype == torch.float8_e4m3fn:
        s = (amax.clamp_min(6 * 2.0**-9) / FP4_MAX).to(torch.float8_e4m3fn).float()
    else:
        s = _pow2_ceil(amax.clamp_min(6 * 2.0**-126) / FP4_MAX)
    q = _round_e2m1((xf / s).clamp(-FP4_MAX, FP4_MAX))
    x.copy_((q * s).flatten(-2).to(x.dtype))
    return x


def dequant_fp8_weight(b: torch.Tensor, b_s: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    """[N, K] E4M3 weight with [ceil(N/bs), ceil(K/bs)] scales -> fp32."""
    N, K = b.shape
    s = b_s.float().repeat_interleave(block_size, 0)[:N].repeat_interleave(block_size, 1)[:, :K]
    return b.float() * s


def unpack_fp4(b: torch.Tensor) -> torch.Tensor:
    """[N, K/2] packed E2M1 (float4_e2m1fn_x2 or uint8/int8) -> [N, K] fp32 codes' values (no scale)."""
    codes = b.view(torch.uint8)
    lut = FP4_LUT.to(codes.device)
    out = torch.empty(*codes.shape[:-1], codes.shape[-1] * 2, dtype=torch.float32, device=codes.device)
    out[..., 0::2] = lut[(codes & 15).long()]
    out[..., 1::2] = lut[(codes >> 4).long()]
    return out


def dequant_fp4_weight(b: torch.Tensor, b_s: torch.Tensor) -> torch.Tensor:
    """[N, K/2] packed fp4 with [N, K/32] E8M0 scales -> [N, K] fp32."""
    w = unpack_fp4(b)
    return (w.unflatten(-1, (-1, 32)) * b_s.float().unsqueeze(-1)).flatten(-2)


def _dequant_act(a: torch.Tensor, a_s: torch.Tensor, block_size: int) -> torch.Tensor:
    K = a.size(-1)
    return (_blocks(a.float(), block_size) * a_s.float().reshape(*a.shape[:-1], K // block_size, 1)).flatten(-2)


def fp8_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, block_size=128):
    A = _dequant_act(a, a_s, block_size)
    B = dequant_fp8_weight(b, b_s, block_size)
    return (A @ B.t()).to(torch.get_default_dtype())


def fp4_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, act_block_size=128):
    A = _dequant_act(a, a_s, act_block_size)
    B = dequant_fp4_weight(b, b_s)
    return (A @ B.t()).to(torch.get_default_dtype())


def sparse_attn(q: torch.Tensor, kv: torch.Tensor, attn_sink: torch.Tensor, topk_idxs: torch.Tensor,
                softmax_scale: float, chunk: int = 256) -> torch.Tensor:
    """q [b,m,h,d] bf16, kv [b,n,d], topk_idxs [b,m,k] int (-1 masked) -> o [b,m,h,d] bf16.  Chunked over m."""
    b, m, h, d = q.shape
    out = torch.empty_like(q)
    sink = attn_sink.float().view(1, 1, h)
    for i0 in range(0, m, chunk):
        i1 = min(m, i0 + chunk)
        idx = topk_idxs[:, i0:i1].long()                               # [b,c,k]
        valid = idx >= 0
        g = torch.gather(kv.unsqueeze(1).expand(b, i1 - i0, kv.size(1), d), 2,
                         idx.clamp_min(0).unsqueeze(-1).expand(*idx.shape, d))   # [b,c,k,d]
        g = torch.where(valid.unsqueeze(-1), g, torch.zeros((), dtype=g.dtype, device=g.device))
        sc = torch.einsum("bchd,bckd->bchk", q[:, i0:i1].float(), g.float()) * softmax_scale
        sc = sc.masked_fill(~valid.unsqueeze(2), float("-inf"))
        mx = torch.maximum(sc.amax(-1), torch.full((), -1e30, device=sc.device))           # [b,c,h]
        p = torch.exp(sc - mx.unsqueeze(-1))
        den = p.sum(-1) + torch.exp(sink - mx)
        o = torch.einsum("bchk,bckd->bchd", p.to(torch.bfloat16).float(), g.float()) / den.unsqueeze(-1)
        out[:, i0:i1] = o.to(q.dtype)
    return out


def hc_split_sinkhorn(mixes: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor, hc_mult: int = 4,
                      sinkhorn_iters: int = 20, eps: float = 1e-6):
    hc = hc_mult
    m = mixes.float()
    sc, bs = hc_scale.float(), hc_base.float()
    pre = torch.sigmoid(m[..., :hc] * sc[0] + bs[:hc]) + eps
    post = 2 * torch.sigmoid(m[..., hc:2 * hc] * sc[1] + bs[hc:2 * hc])
    comb = (m[..., 2 * hc:] * sc[2] + bs[2 * hc:]).unflatten(-1, (hc, hc))
    comb = torch.softmax(comb, -1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb
