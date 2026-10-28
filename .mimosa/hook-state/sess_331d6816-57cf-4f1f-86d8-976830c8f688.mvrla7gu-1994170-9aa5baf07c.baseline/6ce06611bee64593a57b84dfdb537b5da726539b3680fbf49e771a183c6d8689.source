"""eval3.py L: which step of our scalar quantization overfits and shrinks: per tier (ptq1, q2_0), on calib held-out and
Lucebox eval tokens: rtn (H = I: uniform-weight scale search, no GPTQ), diag (H = diag(H): importance-weighted scales,
no error feedback, like llama.cpp's imatrix), gptq (full H, damp 1), gptq-d10 (damp 10), had+refit (production)."""
import glob, json, os, sys
import torch
sys.path[:0] = ["/root/iq2test/experiments", "/root/iq2test/v41"]
import quant_variants as QV
L = int(sys.argv[1])
iq = {}
acc = {}
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
    I1, I2 = torch.eye(H1.shape[0], device="cuda"), torch.eye(H2.shape[0], device="cuda")
    D1, D2 = torch.diag(torch.diag(H1)), torch.diag(torch.diag(H2))
    for t in ("ptq1", "q2_0"):
        for name, (h1, h2, var, damp) in {"rtn": (I1, I2, "base", 1.0), "diag": (D1, D2, "base", 1.0), "gptq": (H1, H2, "base", 1.0),
                                          "gptq-d10": (H1, H2, "base", 10.0), "had+refit": (H1, H2, "had+refit", 1.0)}.items():
            q1, q3, q2 = QV.quant_pair(W1, W3, W2, h1, h2, var, (t, t), iq, damp)
            for k, (x, g) in sets.items():
                Yq = QV.expert_f(x, q1[0] if isinstance(q1, tuple) else q1, q3[0] if isinstance(q3, tuple) else q3, q2[0] if isinstance(q2, tuple) else q2, g, lim)
                a = acc.setdefault((t, name, k), [0.0, 0, 0.0, 0])
                y = Y[k]; err = float((Yq - y).square().sum()); ye = float(y.square().sum())
                a[0] += err / max(ye, 1e-30) * len(x); a[1] += len(x); a[2] += float((Yq * y).sum()) / max(ye, 1e-30); a[3] += 1
    print(f"L{L} e{e}", flush=True)
print(f"layer {L}: token-weighted rel error and mean shrink <y_hat,y>/<y,y>, calib held-out | Lucebox eval")
for t in ("ptq1", "q2_0"):
    for name in ("rtn", "diag", "gptq", "gptq-d10", "had+refit"):
        c, ev = acc[(t, name, "calib")], acc[(t, name, "eval")]
        print(f"  {t:5} {name:10} calib {c[0] / c[1]:.4f} shrink {c[2] / c[3]:.3f} | eval {ev[0] / ev[1]:.4f} shrink {ev[2] / ev[3]:.3f} | ratio {ev[0] / ev[1] / (c[0] / c[1]):.2f}")
