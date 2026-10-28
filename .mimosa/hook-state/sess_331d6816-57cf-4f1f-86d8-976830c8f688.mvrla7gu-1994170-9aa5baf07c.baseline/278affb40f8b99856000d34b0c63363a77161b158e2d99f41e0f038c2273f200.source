"""table.py: merge /root/t16/rows_*_L.json and print, per layer, token-weighted rel and E_L on calib and eval, the
eval/calib ratios and mean shrink, with bpw per row (UD from its per-layer types)."""
import glob, json
UB = {"IQ1_S": 1.5625, "IQ1_M": 1.75, "IQ2_XXS": 2.0625, "IQ2_XS": 2.3125, "IQ2_S": 2.5625}
meta = json.load(open("/root/ud/meta.json"))
for f in sorted(glob.glob("/root/t16/gauss_*.json")):
    print(f.split("gauss_")[1][:-5], "gaussian 4096^2, H = I, K2: weight MSE", json.load(open(f))["mse"])
for L in (5, 20):
    rows = [r for f in glob.glob(f"/root/t16/rows_*_{L}.json") + glob.glob(f"/root/t16/rows_*_{L}_*.json") for r in json.load(open(f))]
    if not rows:
        continue
    common = set.intersection(*[{r["ex"] for r in rows if r["variant"] == v} for v in {r["variant"] for r in rows}])
    print(f"\nlayer {L}: {len(common)} experts with tokens in both sets")
    print(f"  {'variant':22} {'bpw':>5}   calib tok-w rel   E_L  |  eval tok-w rel   E_L   | eval/calib (tok-w, E_L) | shrink calib, eval")
    for v in sorted({r["variant"] for r in rows}):
        rs = [r for r in rows if r["variant"] == v and r["ex"] in common]
        out = []
        for k in ("calib", "eval"):
            n = sum(r[k]["n"] for r in rs)
            out += [sum(r[k]["err"] / max(r[k]["ye"], 1e-30) * r[k]["n"] for r in rs) / n, sum(r[k]["err"] for r in rs) / sum(r[k]["ye"] for r in rs)]
        sh = [sum(r[k]["shrink"] for r in rs) / len(rs) for k in ("calib", "eval")]
        spec = v.split(":")[1]
        bpw = sum(UB[meta[f"blk.{L}.ffn_{x}_exps.weight"][0]] for x in ("gate", "up", "down")) / 3 if spec == "ud" else sum(float(x) for x in spec.split("@")[0].split(",")) / 3
        print(f"  {v:22} {bpw:5.3f}   {out[0]:.4f}  {out[1]:.5f}  |  {out[2]:.4f}  {out[3]:.5f}  | {out[2] / out[0]:.2f} {out[3] / out[1]:.2f}  | {sh[0]:.3f} {sh[1]:.3f}")
