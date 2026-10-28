"""eval2.py L: layer L's dumped experts on two token sets: the calib held-out tokens (the calib dump) and the Lucebox eval
tokens (the eval-corpus dump of the same expert). Variants: our tiers with rotation + refit (quantized on the calib
Hessians), IQ2_XXS (native), Unsloth UD-IQ1_M's experts, and our v5 / v7 allocations (each expert at its mapped tier).
Reports token-weighted mean relative error and E_L (sum err / sum y^2) per token set."""
import glob, json, os, sys
import numpy as np, torch
sys.path[:0] = ["/root/iq2test/experiments", "/root/iq2test/v41"]
import gguf
from gguf import quants
import quant_variants as QV
L = int(sys.argv[1])
readers = [gguf.GGUFReader(f) for f in sorted(glob.glob("/work/um/UD-IQ1_M/*.gguf"))]
T = {t.name: t for r in readers for t in r.tensors}
iq = {"iq2xxs": QV.IQ2XXS(json.load(open("/root/dumpjob/iqgrids.json"))), "iq2n": QV.IQ2Native(0)}
v5 = json.load(open("/work/ds41/alloc-v5.json"))["map"][str(L)]; v7 = json.load(open("/work/v7/alloc-v7.json"))["map"][str(L)]
BPW = {"ptq1": 1.75, "q2_0": 2.25, "q3_k": 3.4375, "iq2xxs": 2.0625}
rows = []
for f in sorted(glob.glob(f"/work/ud{L}/*.pt")):
    d = torch.load(f); e = d["ex"]
    fe = f"/work/edump/L{L:02d}_e{e:03d}.pt"
    if not os.path.exists(fe):
        continue
    de = torch.load(fe)
    W1, W3, W2 = (d[k].float().cuda() for k in ("W1", "W3", "W2"))
    H1, H2, lim = d["H1"].cuda(), d["H2"].cuda(), d["lim"]
    sets = {"calib": (d["xv"].cuda(), d["gv"].cuda()), "eval": (de["xv"].cuda(), de["gv"].cuda())}
    Y = {k: QV.expert_f(x, W1, W3, W2, g, lim) for k, (x, g) in sets.items()}
    mats = {}
    for t in ("ptq1", "q2_0", "q3_k"):
        q1, q3, q2 = QV.quant_pair(W1, W3, W2, H1, H2, "had+refit", (t, t), iq, 1.0)
        mats[t] = (q1, q3, q2)
    mats["iq2xxs"] = QV.quant_pair(W1, W3, W2, H1, H2, "iq2n", ("iq2n", "iq2n"), iq, 1.0)
    U = [torch.from_numpy(np.ascontiguousarray(quants.dequantize(np.asarray(T[f"blk.{L}.ffn_{r}_exps.weight"].data)[e], T[f"blk.{L}.ffn_{r}_exps.weight"].tensor_type))).float().cuda() for r in ("gate", "up", "down")]
    mats["unsloth"] = tuple(U)
    for name, (a, b, c) in mats.items():
        r = {"ex": e, "variant": name}
        for k, (x, g) in sets.items():
            Yq = QV.expert_f(x, a, b, c, g, lim); y = Y[k]; E = Yq - y
            err = float(E.square().sum()); ye = float(y.square().sum())
            me, my = E.mean(0), y.mean(0)
            r[k] = {"n": len(x), "err": err, "ye": ye,
                    "bias": float(me.square().sum()) * len(x) / max(err, 1e-30),   # share of error energy that is a constant offset
                    "shrink": float((Yq * y).sum()) / max(ye, 1e-30),             # <y_hat, y> / <y, y>
                    "cos": float(torch.nn.functional.cosine_similarity(me, my, dim=0))}
        rows.append(r)
    for name, m in (("v5", v5), ("v7", v7)):
        t = m[str(e)].get("all") or m[str(e)].get("gu")
        src = next(r for r in rows if r["ex"] == e and r["variant"] == t)
        rows.append({"ex": e, "variant": name, "tier": t, "calib": src["calib"], "eval": src["eval"]})
    print(f"L{L} e{e} done", flush=True)
json.dump(rows, open(f"/root/e2_{L}.json", "w"))
ex = sorted({r["ex"] for r in rows})
print(f"layer {L}: {len(ex)} experts with tokens in both sets")
print(f"  {'variant':9} {'bpw':>6}   calib tok-w rel    E_L  |  eval tok-w rel    E_L   | eval/calib (tok-w, E_L)")
for v in ("ptq1", "iq2xxs", "q2_0", "q3_k", "v5", "v7", "unsloth"):
    rs = [r for r in rows if r["variant"] == v]
    out = []
    for k in ("calib", "eval"):
        n = sum(r[k]["n"] for r in rs)
        tw = sum(r[k]["err"] / max(r[k]["ye"], 1e-30) * r[k]["n"] for r in rs) / n
        el = sum(r[k]["err"] for r in rs) / sum(r[k]["ye"] for r in rs)
        out += [tw, el]
    if v in BPW: b = BPW[v]
    elif v == "unsloth":
        UB = {"IQ1_S": 1.5625, "IQ1_M": 1.75, "IQ2_XXS": 2.0625, "IQ2_XS": 2.3125, "IQ2_S": 2.5625}
        ty = [T[f"blk.{L}.ffn_{r}_exps.weight"].tensor_type.name for r in ("gate", "up", "down")]
        b = sum(UB[t] for t in ty) / 3
    else: b = sum(BPW[r["tier"]] for r in rs) / len(rs)
    bs = f"{b:6.3f}" if b else "  (UD)"
    print(f"  {v:9} {bs}   {out[0]:.4f}  {out[1]:.5f}  |  {out[2]:.4f}  {out[3]:.5f}  | {out[2] / out[0]:.2f} {out[3] / out[1]:.2f}")
    dec = []
    for k in ("calib", "eval"):
        m = lambda f: sum(r[k][f] for r in rs) / len(rs)
        dec.append(f"{k}: bias share {m('bias'):.4f} shrink {m('shrink'):.4f} cos(mean e, mean y) {m('cos'):+.3f}")
    print(f"  {'':9}        " + "  |  ".join(dec))
