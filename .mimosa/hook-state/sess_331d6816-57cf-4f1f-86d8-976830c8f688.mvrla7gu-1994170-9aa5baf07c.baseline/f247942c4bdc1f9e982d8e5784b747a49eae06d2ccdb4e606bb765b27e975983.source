#!/usr/bin/env python3
# Derived from Xolotl tools/v41 (github.com/Indras-Mirror/Xolotl, commit 7375e305), MIT License,
# Copyright (c) 2026 Niko1221 and the Strata contributors; see LICENSE-Xolotl. Modified for dsx.
"""v41_alloc.py - mixed-tier allocation (RCO-lite) from v41_calib.py screening stats.

Each routed expert gets one tier (ptq1 / q2_0 / q3_k), or with --per-proj one tier for gate+up and one for down.
Cost of a choice = held-out output error ||f(x;W) - f(x;Q)||^2 (gate weight included, GPTQ) normalized by the
layer's held-out routed output energy, so every layer counts the same; --edge-weight boosts layers 0-1 and 37-39.
Budget = all-Q2_0 bytes of the experts covered (budget-neutral, V41_PLAN.md).  Solved by Lagrangian relaxation
(per-item argmin err + lam*bytes, bisection on lam), then a greedy pass spends what is left.
Guard rails: --max-ternary-frac per layer (default 0.5), --unseen-floor (tier floor for experts with < --min-eval
held-out tokens: their error is not measured, so they must not drift to ternary on no evidence).
  v41_alloc.py [--work DIR] [--layers 0-7] [--per-proj] [--out alloc.json]
"""
import argparse, glob, json, os, re
import numpy as np

TIERS = ["ptq1", "iq2xxs", "q2_0", "q3_k"]   # ascending bytes (the solver moves between neighbors)
BPW = {"ptq1": 1.75, "iq2xxs": 2.0625, "q2_0": 2.25, "q3_k": 3.4375}
GU, DN = 2 * 5120 * 2304, 5120 * 2304          # params per expert: gate+up, down


def load(work, layers):
    out = {}
    for p in sorted(glob.glob(os.path.join(work, "stats", "L*.json"))):
        L = int(re.search(r"L(\d+)\.json", p).group(1))
        if layers is None or L in layers:
            out[L] = json.load(open(p))
    return out


def items_for(stats, a):
    """-> list of (layer, expert, part, [(tier, err_norm, bytes)...])."""
    items = []
    for L, j in stats.items():
        ex = j["experts"]
        energy = sum(s.get("y_energy", 0.0) for s in ex.values()) or 1.0
        w = a.lw[L] if a.lw is not None else (a.edge_weight if (L <= 1 or L >= 37) else 1.0)
        for e, s in ex.items():
            unseen = s.get("n_eval", 0) < a.min_eval or "e_q2_0" not in s
            floor = TIERS.index(a.unseen_floor) if unseen else 0
            parts = [("gu", GU, "egu_"), ("dn", DN, "edn_")] if a.per_proj else [("all", GU + DN, "e_")]
            for part, n, key in parts:
                ch = []
                for ti, t in enumerate(TIERS):
                    if ti < floor:
                        continue
                    # IQ2_XXS only where screened (e_iq2xxs), and not for the experts --iq2-not-hot keeps in Q2_0
                    if t == "iq2xxs" and (not unseen and key + t not in s or (L, int(e)) in a.hot):
                        continue
                    err = s.get(key + t, 0.0) if not unseen else 0.0
                    ch.append((t, w * err / energy, n * BPW[t] / 8))
                items.append((L, int(e), part, ch, unseen))
    return items


def solve(items, budget, a, layer_n):
    def pick(lam):
        sel = [min(range(len(ch)), key=lambda i: ch[i][1] + lam * ch[i][2]) for (_, _, _, ch, _) in items]
        return sel, sum(items[k][3][i][2] for k, i in enumerate(sel))
    lo, hi = 0.0, 1.0
    while pick(hi)[1] > budget:
        hi *= 4
    for _ in range(60):
        mid = (lo + hi) / 2
        if pick(mid)[1] > budget:
            lo = mid
        else:
            hi = mid
    sel, used = pick(hi)
    # per-layer ternary cap: promote the cheapest-to-promote ternary picks to the next tier
    for L, nl in layer_n.items():
        tern = [k for k, it in enumerate(items) if it[0] == L and it[3][sel[k]][0] == "ptq1"]
        cap = int(a.max_ternary_frac * nl * (2 if a.per_proj else 1))
        if len(tern) > cap:
            tern.sort(key=lambda k: items[k][3][sel[k] + 1][1] - items[k][3][sel[k]][1], reverse=True)
            for k in tern[: len(tern) - cap]:
                used += items[k][3][sel[k] + 1][2] - items[k][3][sel[k]][2]
                sel[k] += 1
    # cap may overshoot the budget: demote q3 picks with the smallest loss; else spend leftover greedily
    def gain(k, d):
        ch, i = items[k][3], sel[k]
        j = i + d
        if not 0 <= j < len(ch) or (d < 0 and ch[j][0] == "ptq1"):
            return None
        return (ch[i][1] - ch[j][1]) / abs(ch[j][2] - ch[i][2])
    while used > budget:
        cand = [(g, k) for k in range(len(items)) if items[k][3][sel[k]][0] == "q3_k" and (g := gain(k, -1)) is not None]
        if not cand:
            break
        _, k = min(cand)
        used += items[k][3][sel[k] - 1][2] - items[k][3][sel[k]][2]; sel[k] -= 1
    while True:
        cand = sorted(((g, k) for k in range(len(items)) if (g := gain(k, +1)) is not None and g > 0), reverse=True)
        moved = False
        for g, k in cand:
            dbytes = items[k][3][sel[k] + 1][2] - items[k][3][sel[k]][2]
            if used + dbytes <= budget:
                used += dbytes; sel[k] += 1; moved = True
                break
        if not moved:
            break
    return sel, used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=os.environ.get("V41_WORK", "/media/Crucial1TB/v41-calib"))
    ap.add_argument("--layers", default=None, help="e.g. 0-7")
    ap.add_argument("--per-proj", action="store_true")
    ap.add_argument("--edge-weight", type=float, default=2.0)
    ap.add_argument("--layer-weights", default=None, help="JSON list: per-layer weight on the normalized error (measured KL sensitivity); replaces --edge-weight")
    ap.add_argument("--max-ternary-frac", type=float, default=0.5)
    ap.add_argument("--min-eval", type=int, default=8)
    ap.add_argument("--unseen-floor", default="q2_0", choices=TIERS)
    ap.add_argument("--out", default=None)
    ap.add_argument("--budget-bpw", type=float, default=None, help="ds41: budget in bits/weight (default: all-Q2_0 = 2.25)")
    ap.add_argument("--rank", default=None, help="expert-rank.json ([[layer, expert, tokens], ...] hottest first)")
    ap.add_argument("--iq2-not-hot", type=int, default=0, metavar="N",
                    help="the N hottest experts of --rank (the ones dsx places in VRAM) may not take iq2xxs: Q2_0's GPU "
                         "kernel is ~1.6x faster than IQ2_XXS's on an RTX 3090")
    a = ap.parse_args()
    a.lw = json.load(open(a.layer_weights)) if a.layer_weights else None
    a.hot = {(int(r[0]), int(r[1])) for r in json.load(open(a.rank))[:a.iq2_not_hot]} if a.iq2_not_hot else set()
    layers = None
    if a.layers:
        x, y = map(int, a.layers.split("-"))
        layers = set(range(x, y + 1))
    stats = load(a.work, layers)
    assert stats, "no stats"
    items = items_for(stats, a)
    layer_n = {L: len(j["experts"]) for L, j in stats.items()}
    budget = sum(it[3][[c[0] for c in it[3]].index("q2_0")][2] for it in items)
    if a.budget_bpw:
        budget = budget * a.budget_bpw / BPW["q2_0"]
    sel, used = solve(items, budget, a, layer_n)
    err = lambda t: sum(next(c[1] for c in it[3] if c[0] == t) for it in items if t in [c[0] for c in it[3]])
    mixed_err = sum(items[k][3][i][1] for k, i in enumerate(sel))
    q2_err = sum(next(c[1] for c in it[3] if c[0] == "q2_0") for it in items)
    print(f"layers {sorted(stats)}  items {len(items)}  budget {budget / 2**30:.2f} GiB (all-Q2_0)  used {used / 2**30:.2f} GiB")
    print(f"proxy error (sum of per-layer relative output error): all-Q2_0 {q2_err:.4f}  mixed {mixed_err:.4f}  "
          f"({100 * (mixed_err / q2_err - 1):+.1f}%)")
    print(f"{'L':>3} {'ptq1':>5} {'iq2':>5} {'q2_0':>5} {'q3_k':>5} {'unseen':>6} {'bpw':>6}   GPTQ/RTN err ratio (ptq1 q2 q3)   rel err @q2")
    amap = {}
    for L in sorted(stats):
        ks = [k for k, it in enumerate(items) if it[0] == L]
        c = {t: sum(items[k][3][sel[k]][0] == t for k in ks) for t in TIERS}
        bits = sum(items[k][3][sel[k]][2] for k in ks) * 8 / sum((GU + DN) if items[k][2] == "all" else (GU if items[k][2] == "gu" else DN) for k in ks)
        ex = stats[L]["experts"].values()
        r = [np.nansum([s.get(f"e_{t}", np.nan) for s in ex]) / max(np.nansum([s.get(f"rtn_{t}", np.nan) for s in ex]), 1e-30)
             for t in ("ptq1", "q2_0", "q3_k")]
        relq2 = np.nansum([s.get("e_q2_0", np.nan) for s in ex]) / max(sum(s.get("y_energy", 0) for s in ex), 1e-30)
        print(f"{L:>3} {c['ptq1']:>5} {c['iq2xxs']:>5} {c['q2_0']:>5} {c['q3_k']:>5} {sum(items[k][4] for k in ks):>6} {bits:>6.3f}   "
              f"{r[0]:.3f} {r[1]:.3f} {r[2]:.3f}                  {relq2:.4f}")
        for k in ks:
            it = items[k]
            amap.setdefault(str(L), {}).setdefault(str(it[1]), {})[it[2]] = it[3][sel[k]][0]
    if a.out:
        json.dump({"layers": sorted(stats), "per_proj": a.per_proj, "budget_bytes": budget, "used_bytes": used,
                   "proxy_err_all_q2": q2_err, "proxy_err_mixed": mixed_err,
                   "args": {k: v for k, v in vars(a).items() if k != "hot"}, "map": amap},
                  open(a.out, "w"))
        print("wrote", a.out)


if __name__ == "__main__":
    main()
