"""Expert hotness ranking from calibration stats: [[layer, expert, tokens], ...] hottest first -> rank.json"""
import glob, json, os, re, sys
work = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/ai/ds41/calib")
rows = []
for p in sorted(glob.glob(os.path.join(work, "stats", "L*.json"))):
    L = int(re.search(r"L(\d+)\.json", p).group(1))
    for e, s in json.load(open(p))["experts"].items():
        rows.append([L, int(e), int(s.get("n_fit", 0) + s.get("n_eval", 0))])
rows.sort(key=lambda r: -r[2])
out = os.path.join(work, "rank.json")
json.dump(rows, open(out, "w"))
print(f"{len(rows)} experts from {len(set(r[0] for r in rows))} layers -> {out}")
