"""merge.py OLD NEW OUT: the iq2xxs keys of an iq2xxs-only screening run (NEW) into the earlier run's stats (OLD).
Both runs use the same corpus and teacher, but bf16 near-ties in routing move a few tokens between experts, so each
expert's held-out set differs slightly. Each new error is scaled by y_energy(old) / y_energy(new) of the expert (the
errors are sums over the held-out tokens, as is y_energy), which puts it on the old run's token set to first order."""
import json, os, sys
old, new, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
diffs, ratios = [], []
for f in sorted(os.listdir(old)):
    if not f.startswith("L") or "_" in f:
        continue
    a, b = json.load(open(os.path.join(old, f))), json.load(open(os.path.join(new, f)))
    n = 0
    for e, s in a["experts"].items():
        t = b["experts"].get(e, {})
        if "e_iq2xxs" not in t:
            continue
        sc = 1.0
        if s.get("y_energy") and t.get("y_energy"):
            sc = s["y_energy"] / t["y_energy"]
            ratios.append(sc)
            diffs.append(abs(s.get("n_eval", 0) - t.get("n_eval", 0)) / max(s.get("n_eval", 1), 1))
        for k in ("e_iq2xxs", "egu_iq2xxs", "edn_iq2xxs"):
            if k in t:
                s[k] = t[k] * sc; n += 1
    json.dump(a, open(os.path.join(out, f), "w"))
    print(f, "iq2xxs keys", n)
diffs.sort(); ratios.sort()
q = lambda v, p: v[min(len(v) - 1, int(p * len(v)))]
print(f"experts {len(diffs)}: n_eval relative difference median {q(diffs, .5):.4f} p99 {q(diffs, .99):.4f} max {diffs[-1]:.4f}; "
      f"y_energy old/new p1 {q(ratios, .01):.3f} median {q(ratios, .5):.4f} p99 {q(ratios, .99):.3f}")
