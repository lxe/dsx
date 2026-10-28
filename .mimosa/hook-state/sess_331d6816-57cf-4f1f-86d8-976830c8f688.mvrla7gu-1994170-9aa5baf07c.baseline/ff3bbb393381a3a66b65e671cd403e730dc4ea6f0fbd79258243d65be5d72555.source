"""teacher_check.py LUCEBOX_DIR MINE_DIR: our teacher path (v41_calib chain + head --teacher-out) against Lucebox's
own teacher on the same 8 x 1024 tokens: top-1 agreement and KL(Lucebox || ours) over Lucebox's top-256 (ids missing
from our top-256 get our 256th log-prob, so the KL is an upper bound)."""
import os, sys
import numpy as np
a, b = sys.argv[1], sys.argv[2]
assert (np.fromfile(f"{a}/heldout.i32", "<i4") == np.fromfile(f"{b}/heldout.i32", "<i4")).all(), "different tokens"
K = 256
ia, la = np.fromfile(f"{a}/teacher_idx.i32", "<i4").reshape(-1, K), np.fromfile(f"{a}/teacher_lp.f32", "<f4").reshape(-1, K)
ib, lb = np.fromfile(f"{b}/teacher_idx.i32", "<i4").reshape(-1, K), np.fromfile(f"{b}/teacher_lp.f32", "<f4").reshape(-1, K)
kl, top1 = 0.0, 0
for r in range(len(ia)):
    m = dict(zip(ib[r].tolist(), lb[r].tolist()))
    q = np.array([m.get(i, lb[r, -1]) for i in ia[r]])
    p = np.exp(la[r])
    kl += float((p * (la[r] - q)).sum())
    top1 += int(ia[r, 0] == ib[r, 0])
print(f"rows {len(ia)}: KL(Lucebox || ours) {kl / len(ia):.5f}, top-1 agreement {top1 / len(ia):.4f}")
