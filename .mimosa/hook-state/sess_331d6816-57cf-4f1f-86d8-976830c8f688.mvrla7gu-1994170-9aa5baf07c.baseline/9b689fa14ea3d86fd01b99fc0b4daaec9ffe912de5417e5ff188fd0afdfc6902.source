"""eval4.py L: Hessians from all calib domains vs from its 20 web sequences only; GPTQ (full H) vs diag-H (importance-
weighted rounding, no error feedback); tiers ptq1, q2_0, iq2xxs. Scored on the Lucebox eval tokens, next to UD-IQ1_M."""
import glob, json, os, sys
import torch
sys.path[:0] = ["/root/iq2test/experiments", "/root/iq2test/v41"]
import quant_variants as QV
L = int(sys.argv[1])
iq = {"iq2n": QV.IQ2Native(0)}
acc = {}
ud = {r["ex"]: r for r in json.load(open(f"/root/e2_{L}.json")) if r["variant"] == "unsloth"}
for f in sorted(glob.glob(f"/work/ud{L}/*.pt")):
    d = torch.load(f); e = d["ex"]
    fw, fe = f"/work/wdump/L{L:02d}_e{e:03d}.pt", f"/work/edump/L{L:02d}_e{e:03d}.pt"
    if not (os.path.exists(fw) and os.path.exists(fe)):
        continue
    dw, de = torch.load(fw), torch.load(fe)
    if dw["n_fit"] < 32 or e not in ud:
        continue
    W1, W3, W2 = (d[k].float().cuda() for k in ("W1", "W3", "W2"))
    lim = d["lim"]
    x, g = de["xv"].cuda(), de["gv"].cuda()
    Y = QV.expert_f(x, W1, W3, W2, g, lim); ye = float(Y.square().sum())
    Hs = {"all": (d["H1"].cuda(), d["H2"].cuda()), "web": (dw["H1"].cuda(), dw["H2"].cuda())}
    for hn, (h1, h2) in Hs.items():
        for mode in ("gptq", "diag"):
            a1, a2 = (h1, h2) if mode == "gptq" else (torch.diag(torch.diag(h1)), torch.diag(torch.diag(h2)))
            for t, var in (("ptq1", "base"), ("q2_0", "base"), ("iq2", "iq2n")):
                q1, q3, q2 = QV.quant_pair(W1, W3, W2, a1, a2, var, (t, t) if t != "iq2" else ("iq2n", "iq2n"), iq, 1.0)
                err = float((QV.expert_f(x, q1, q3, q2, g, lim) - Y).square().sum())
                a = acc.setdefault((t, mode, hn), [0.0, 0]); a[0] += err / ye * len(x); a[1] += len(x)
    a = acc.setdefault(("UD", "", ""), [0.0, 0]); a[0] += ud[e]["eval"]["err"] / ud[e]["eval"]["ye"] * len(x); a[1] += len(x)
    print(f"L{L} e{e}", flush=True)
n = acc[("UD", "", "")][1]
print(f"layer {L}: token-weighted rel error on Lucebox eval tokens ({len([k for k in acc]) and n} tokens), Hessians from all calib domains | web only")
for t in ("ptq1", "q2_0", "iq2"):
    for mode in ("gptq", "diag"):
        al, wb = acc[(t, mode, "all")], acc[(t, mode, "web")]
        print(f"  {t:5} {mode:5}  all {al[0] / al[1]:.4f}  web {wb[0] / wb[1]:.4f}")
u = acc[("UD", "", "")]; print(f"  UD-IQ1_M        {u[0] / u[1]:.4f}")
