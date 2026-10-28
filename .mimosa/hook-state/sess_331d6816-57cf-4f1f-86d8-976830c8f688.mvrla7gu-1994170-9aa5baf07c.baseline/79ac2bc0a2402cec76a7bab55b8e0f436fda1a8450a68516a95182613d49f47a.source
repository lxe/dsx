"""rowstats.py L: per (token, expert) row on layer L's dumped experts: how output energy and quantization error energy
distribute over rows, for our Q2_0 (rotation + refit) and Unsloth UD-IQ1_M's experts."""
import glob, os, sys, json
import numpy as np, torch
sys.path[:0] = ["/root/iq2test/experiments", "/root/iq2test/v41"]
import gguf
from gguf import quants
import quant_variants as QV
L = int(sys.argv[1])
readers = [gguf.GGUFReader(f) for f in sorted(glob.glob("/work/um/UD-IQ1_M/*.gguf"))]
T = {t.name: t for r in readers for t in r.tensors}
grids = json.load(open("/root/dumpjob/iqgrids.json"))
iq = {"iq2xxs": QV.IQ2XXS(grids)}
yy, eq, eu = [], [], []
for f in sorted(glob.glob(f"/work/ud{L}/*.pt")):
    d = torch.load(f); e = d["ex"]
    W1, W3, W2 = (d[k].float().cuda() for k in ("W1", "W3", "W2"))
    xv, gv, lim = d["xv"].cuda(), d["gv"].cuda(), d["lim"]
    Y = QV.expert_f(xv, W1, W3, W2, gv, lim)
    q1, q3, q2 = QV.quant_pair(W1, W3, W2, d["H1"].cuda(), d["H2"].cuda(), "had+refit", ("q2_0", "q2_0"), iq, 1.0)
    U = [torch.from_numpy(np.ascontiguousarray(quants.dequantize(np.asarray(T[f"blk.{L}.ffn_{r}_exps.weight"].data)[e], T[f"blk.{L}.ffn_{r}_exps.weight"].tensor_type))).float().cuda() for r in ("gate", "up", "down")]
    yy.append(Y.square().sum(1).cpu()); eq.append((QV.expert_f(xv, q1, q3, q2, gv, lim) - Y).square().sum(1).cpu())
    eu.append((QV.expert_f(xv, U[0], U[1], U[2], gv, lim) - Y).square().sum(1).cpu())
yy, eq, eu = (torch.cat(v).double() for v in (yy, eq, eu))
o = torch.argsort(yy, descending=True); k = max(1, len(yy) // 100)
top = o[:k]
print(f"layer {L}: {len(yy)} rows; top 1% of rows by ||y||^2 carry {float(yy[top].sum() / yy.sum()):.3f} of output energy, "
      f"{float(eq[top].sum() / eq.sum()):.3f} of Q2_0's error energy, {float(eu[top].sum() / eu.sum()):.3f} of UD-IQ1_M's")
print(f"  E_L: Q2_0 {float(eq.sum() / yy.sum()):.5f}, UD-IQ1_M {float(eu.sum() / yy.sum()):.5f};  per-row mean ||e||^2/||y||^2: "
      f"Q2_0 {float((eq / yy.clamp_min(1e-30)).mean()):.4f}, UD-IQ1_M {float((eu / yy.clamp_min(1e-30)).mean()):.4f}")
print(f"  rows excluding the top 1%: E_L Q2_0 {float(eq[o[k:]].sum() / yy[o[k:]].sum()):.5f}, UD-IQ1_M {float(eu[o[k:]].sum() / yy[o[k:]].sum()):.5f}")
