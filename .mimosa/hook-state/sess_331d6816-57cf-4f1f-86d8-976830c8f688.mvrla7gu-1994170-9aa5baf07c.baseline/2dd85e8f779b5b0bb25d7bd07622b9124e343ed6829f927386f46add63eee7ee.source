"""probe_select.py PASS_DIR OUT_DIR [--min 256] [--cap-tokens 160000] [--layers 0-39]: the coverage top-up of calib5.

PASS_DIR is a v41_calib work dir after a routing pass (init, then `layer L --no-gptq` for every layer, no dumps): its
tokens.npy / hashes.npy / meta.json hold the calibration rows plus candidate rows (meta "pool": true), and
route/LNN.npz the 6 experts every token was routed to. Every row is a fit row (is_eval false), so an expert's fit
tokens are the tokens routed to it.

Counts each (layer, expert)'s tokens over the non-pool rows, then adds pool rows greedily, each time the row that fills
the most remaining deficit (sum over layers and experts of min(row's tokens, MIN - count)), until no expert is under
--min or the added rows reach --cap-tokens. Writes OUT_DIR/tokens.npy, hashes.npy (sliced from PASS_DIR, not
recomputed), meta.json (pool flag dropped) and coverage_report.json (per layer: experts under MIN and the minimum
count, before and after)."""
import argparse, json, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("pass_dir"); ap.add_argument("out_dir")
ap.add_argument("--min", type=int, default=256)
ap.add_argument("--cap-tokens", type=int, default=160_000)
ap.add_argument("--layers", default="0-39")
a = ap.parse_args()
lo, hi = map(int, a.layers.split("-"))
layers = list(range(lo, hi + 1))
NE = 384

meta = json.load(open(os.path.join(a.pass_dir, "meta.json")))
N, S = meta["n"], meta["seq"]
pool = np.array(meta.get("pool", [False] * N), dtype=bool)
assert not any(meta["is_eval"]), "calib5 rows are all fit rows"
base_rows, pool_rows = np.flatnonzero(~pool), np.flatnonzero(pool)
base = np.zeros((len(layers), NE), dtype=np.int64)                  # tokens per (layer, expert) over base rows
cand = np.zeros((len(pool_rows), len(layers), NE), dtype=np.int32)   # the same per pool row
for li, L in enumerate(layers):
    idx = np.load(os.path.join(a.pass_dir, "route", f"L{L:02d}.npz"))["idx"].astype(np.int64).reshape(N, S * 6)
    base[li] = np.bincount(idx[base_rows].ravel(), minlength=NE)
    for j, r in enumerate(pool_rows):
        cand[j, li] = np.bincount(idx[r], minlength=NE)

def summary(cnt):
    return [{"layer": L, "under_min": int((cnt[li] < a.min).sum()), "min": int(cnt[li].min()),
             "min_expert": int(cnt[li].argmin())} for li, L in enumerate(layers)]

before = summary(base)
deficit = np.maximum(0, a.min - base)
chosen, added = [], 0
avail = np.ones(len(pool_rows), dtype=bool)
while deficit.any() and added + S <= a.cap_tokens and avail.any():
    gain = np.minimum(cand, deficit[None]).reshape(len(pool_rows), -1).sum(1)
    gain[~avail] = -1
    j = int(gain.argmax())
    if gain[j] <= 0:
        break
    chosen.append(j); avail[j] = False; added += S
    deficit = np.maximum(0, deficit - cand[j])
after_cnt = base + cand[chosen].sum(0) if chosen else base
after = summary(after_cnt)
keep = np.sort(np.concatenate([base_rows, pool_rows[chosen]]))

os.makedirs(a.out_dir, exist_ok=True)
tok = np.load(os.path.join(a.pass_dir, "tokens.npy"), mmap_mode="r")
np.save(os.path.join(a.out_dir, "tokens.npy"), np.ascontiguousarray(tok[keep]))
hs = np.load(os.path.join(a.pass_dir, "hashes.npy"), mmap_mode="r")
np.save(os.path.join(a.out_dir, "hashes.npy"), np.ascontiguousarray(hs[keep]))
json.dump({"n": len(keep), "seq": S, "domains": [meta["domains"][i] for i in keep], "is_eval": [False] * len(keep),
           "corpus": f"{meta.get('corpus')} + coverage probes ({len(chosen)} rows, min {a.min})"},
          open(os.path.join(a.out_dir, "meta.json"), "w"))
rep = {"min": a.min, "base_rows": len(base_rows), "pool_rows": len(pool_rows), "probe_rows": len(chosen),
       "probe_domains": {d: sum(meta["domains"][pool_rows[j]] == d for j in chosen) for d in sorted({meta["domains"][pool_rows[j]] for j in chosen})},
       "experts_under_min_before": sum(x["under_min"] for x in before), "experts_under_min_after": sum(x["under_min"] for x in after),
       "before": before, "after": after}
json.dump(rep, open(os.path.join(a.out_dir, "coverage_report.json"), "w"), indent=1)
print(f"{len(base_rows)} base + {len(chosen)} probe rows of {len(pool_rows)} candidates -> {len(keep)} rows; experts under "
      f"{a.min}: {rep['experts_under_min_before']} -> {rep['experts_under_min_after']} (over {len(layers)} layers); "
      f"min per layer before {min(x['min'] for x in before)}, after {min(x['min'] for x in after)}; probes {rep['probe_domains']}")
