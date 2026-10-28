#!/usr/bin/env python3
"""make_layout_map.py RANK OUT [--bpw 2.03125] - a tier map with Unsloth UD-IQ1_M's layout in our formats: layers 0-4
ternary, layers 5-19 ternary gate/up with Q2_0 down, layers 20-39 Q2_0; to land on the bit budget, the coldest decoder
experts (by RANK, hottest first) keep ternary gate/up. Per-projection map for v41_quant.py --map."""
import argparse, json
BPW = {"ptq1": 1.75, "q2_0": 2.25, "q3_k": 3.4375}
GU, DN = 2 * 5120 * 2304, 5120 * 2304
NL, NE = 40, 384

p = argparse.ArgumentParser()
p.add_argument("rank")
p.add_argument("out")
p.add_argument("--bpw", type=float, default=2.03125)
a = p.parse_args()
heat = {(l, e): n for l, e, n in json.load(open(a.rank))}
m = {}
for l in range(NL):
    for e in range(NE):
        gu, dn = ("ptq1", "ptq1") if l < 5 else ("ptq1", "q2_0") if l < 20 else ("q2_0", "q2_0")
        m.setdefault(str(l), {})[str(e)] = {"gu": gu, "dn": dn}
size = lambda: sum(GU * BPW[v["gu"]] + DN * BPW[v["dn"]] for lm in m.values() for v in lm.values()) / 8
budget = NL * NE * (GU + DN) * a.bpw / 8
cold = sorted(((heat.get((l, e), 0), l, e) for l in range(20, NL) for e in range(NE)))
i = 0
while size() > budget and i < len(cold):   # one expert at a time from the coldest (exact enough: ~3 MB steps)
    _, l, e = cold[i]
    m[str(l)][str(e)]["gu"] = "ptq1"
    i += 1
json.dump({"layers": list(range(NL)), "per_proj": True, "budget_bytes": budget, "used_bytes": size(), "map": m}, open(a.out, "w"))
print(f"{i} decoder experts keep ternary gate/up; {size() / 1e9:.2f} GB of experts ({8 * size() / (NL * NE * (GU + DN)):.4f} bpw)")
