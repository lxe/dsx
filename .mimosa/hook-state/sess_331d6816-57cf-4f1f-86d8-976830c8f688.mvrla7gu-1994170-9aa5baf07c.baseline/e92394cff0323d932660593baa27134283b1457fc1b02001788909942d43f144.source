# Derived from Xolotl tools/v41 (github.com/Indras-Mirror/Xolotl, commit 7375e305), MIT License,
# Copyright (c) 2026 Niko1221 and the Strata contributors; see LICENSE-Xolotl. Modified for dsx.
"""v41_gptq_graph.py - v41_gptq.gptq with the 128-column inner loop captured once as a CUDA graph and replayed.

Same kernels on the same inputs in the same order as v41_gptq.gptq (elementwise ops only inside the block; the
cross-block update stays an eager matmul), so the result is bit-identical - test_gptq_graph.py checks that on every
tier.  Only the per-column Python launch overhead goes (it was ~85% of a gptq call: 4096 columns x ~8 launches).
Falls back to v41_gptq.gptq off-CUDA or for a block narrower than `block`."""
import torch
import v41_gptq as G

_graphs = {}


class _Block:
    def __init__(self, R, B, dev):
        f = dict(device=dev, dtype=torch.float32)   # explicit: model.py sets the default dtype to bf16
        self.W1 = torch.zeros((R, B), **f)
        self.S1 = torch.zeros((R, B), **f)
        self.Hi1 = torch.zeros((B, B), **f)
        self.lo = torch.zeros((R,), **f)
        self.hi = torch.zeros((R,), **f)
        self.Q1 = torch.zeros((R, B), **f)
        self.E1 = torch.zeros((R, B), **f)
        s = torch.cuda.Stream(dev)
        s.wait_stream(torch.cuda.current_stream(dev))
        with torch.cuda.stream(s):
            self._body()                       # warm-up outside capture (allocator, kernel selection)
        torch.cuda.current_stream(dev).wait_stream(s)
        self.g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.g):
            self._body()

    def _body(self):
        W1, S1, Hi1, Q1, E1, lo, hi = self.W1, self.S1, self.Hi1, self.Q1, self.E1, self.lo, self.hi
        for i in range(W1.shape[1]):
            w = W1[:, i]
            q = torch.minimum(torch.maximum(torch.round(w / S1[:, i]), lo), hi) * S1[:, i]
            Q1[:, i] = q
            err = (w - q) / Hi1[i, i]
            W1[:, i:] -= err.unsqueeze(1) * Hi1[i, i:].unsqueeze(0)
            E1[:, i] = err


def gptq(W, S, lo, hi, Hinv, dead, block=128):
    """Drop-in for v41_gptq.gptq (same arguments, bit-identical result)."""
    if not W.is_cuda or W.dtype != torch.float32 or Hinv.dtype != torch.float32 or S.dtype != torch.float32:
        return G.gptq(W, S, lo, hi, Hinv, dead, block)
    W = W.clone()
    W[:, dead] = 0
    R, C = W.shape
    if C % block:
        return G.gptq(W, S, lo, hi, Hinv, dead, block)
    key = (R, block, W.device.index)
    b = _graphs.get(key)
    if b is None:
        b = _graphs[key] = _Block(R, block, W.device)
    Q = torch.zeros_like(W)
    b.lo.copy_(lo[:, 0]); b.hi.copy_(hi[:, 0])
    for i1 in range(0, C, block):
        i2 = i1 + block
        b.W1.copy_(W[:, i1:i2]); b.S1.copy_(S[:, i1:i2]); b.Hi1.copy_(Hinv[i1:i2, i1:i2])
        b.Q1.zero_(); b.E1.zero_()
        b.g.replay()
        Q[:, i1:i2] = b.Q1
        W[:, i2:] -= b.E1 @ Hinv[i1:i2, i2:]
    return Q
