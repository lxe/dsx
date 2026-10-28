"""unsloth_score.py DUMPDIR UMDIR OUT: Unsloth UD-IQ1_M's routed experts (dequantized by gguf-py) on the dumped experts'
held-out tokens, the same metric as quant_variants (whole SwiGLU expert, rel = sum (f(x; W_hat) - f(x; W))^2 / sum f^2,
abs per token); their weights are in the original basis, so no rotation."""
import glob, json, os, sys
import numpy as np, torch
sys.path[:0] = ["/root/iq2test/experiments", "/root/iq2test/v41"]
import gguf
from gguf import quants
import quant_variants as QV

dump, umdir, out = sys.argv[1:4]
readers = [gguf.GGUFReader(f) for f in sorted(glob.glob(os.path.join(umdir, "*.gguf")))]
T = {t.name: t for r in readers for t in r.tensors}
BPW = {"IQ1_S": 1.5625, "IQ1_M": 1.75, "IQ2_XXS": 2.0625, "IQ2_XS": 2.3125, "IQ2_S": 2.5625, "Q2_K": 2.625, "IQ3_XXS": 3.0625}
rows = []
for f in sorted(glob.glob(os.path.join(dump, "*.pt"))):
    d = torch.load(f)
    L, e = d["L"], d["ex"]
    W = {}
    ty = {}
    for role, key in (("gate", "W1"), ("up", "W3"), ("down", "W2")):
        t = T[f"blk.{L}.ffn_{role}_exps.weight"]
        raw = np.asarray(t.data)[e]                                  # this expert's rows, raw blocks
        w = quants.dequantize(raw, t.tensor_type)                    # [rows, cols] float32
        W[key] = torch.from_numpy(np.ascontiguousarray(w)).float().cuda()
        ty[role] = t.tensor_type.name
        assert W[key].shape == d[key].shape, (L, e, role, W[key].shape, d[key].shape)
    xv, gv, lim = d["xv"].cuda(), d["gv"].cuda(), d["lim"]
    W1, W3, W2 = (d[k].float().cuda() for k in ("W1", "W3", "W2"))
    Y = QV.expert_f(xv, W1, W3, W2, gv, lim)
    err = float((QV.expert_f(xv, W["W1"], W["W3"], W["W2"], gv, lim) - Y).square().sum())
    bpw = (2 * BPW[ty["gate"]] + BPW[ty["down"]]) / 3
    r = {"file": os.path.basename(f), "L": L, "ex": e, "n_eval": len(xv), "variant": "unsloth", "tier": f"{ty['gate']}/{ty['down']}",
         "bpw": bpw, "rel": err / float(Y.square().sum()), "abs": err / len(xv)}
    rows.append(r)
    print(json.dumps(r), flush=True)
json.dump(rows, open(out, "w"))
