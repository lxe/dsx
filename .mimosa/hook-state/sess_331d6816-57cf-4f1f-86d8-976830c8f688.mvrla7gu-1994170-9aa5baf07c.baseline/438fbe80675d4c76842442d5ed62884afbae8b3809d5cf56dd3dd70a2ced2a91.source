#!/bin/bash
# assemble_pass.sh (on .102, in ~/src/t16q/calib5): the coverage pass set of calibration v5.
#   base5   calib3 without its web rows, every row fit (is_eval false)
#   v5w     base5 + agent_lxe.jsonl + mix5_public.jsonl, packed by v41_calib.py prep-mix (2048-token rows, engram hashes)
#   passw   v5w + the candidate pool (pool_public.jsonl + the other half of calib4's code as pool_code); meta "pool"
#           marks the candidate rows for probe_select.py
# Prints the row and token counts of each set.
set -eo pipefail
cd ~/src/t16q/calib5
PY=~/ai/ds41/venv/bin/python
Q=~/src/t16q/chain/quant
export V41_MODEL=$HOME/ai/ds41/fp8-1017 V41_REF=$HOME/ai/ds41/fp8-1017/inference
rm -rf base5 v5w passw && mkdir -p base5
$PY - <<'PY'
import json, numpy as np
src = "/home/lxe/ai/ds41/calib3"
m = json.load(open(f"{src}/meta.json")); t = np.load(f"{src}/tokens.npy")
keep = [i for i, d in enumerate(m["domains"]) if d != "web"]
np.save("base5/tokens.npy", t[keep])
json.dump({"n": len(keep), "seq": m["seq"], "domains": [m["domains"][i] for i in keep], "is_eval": [False] * len(keep),
           "corpus": f"{m['corpus']} (calib3 without web, all fit)"}, open("base5/meta.json", "w"))
print(f"base5: {len(keep)} rows (dropped {m['n'] - len(keep)} web rows)")
# the candidate code: calib4's code texts that the v5 mix does not use (calib5_public.py takes every other one)
mix4 = [json.loads(l) for l in open("/home/lxe/src/t16q/t16data/calib_mix.jsonl")]
code = [r["text"] for r in mix4 if r["domain"] == "code"][1::2]
with open("pool_code.jsonl", "w") as fh:
    for t_ in code:
        fh.write(json.dumps({"text": t_, "domain": "pool_code"}, ensure_ascii=False) + "\n")
print(f"pool_code: {len(code)} texts")
PY
cat agent_lxe.jsonl mix5_public.jsonl > mix5.jsonl
cat pool_public.jsonl pool_code.jsonl > pool.jsonl
( cd $Q && V41_WORK=$HOME/src/t16q/calib5/v5w $PY v41_calib.py prep-mix --base $HOME/src/t16q/calib5/base5 --mix $HOME/src/t16q/calib5/mix5.jsonl 2>&1 | grep prep-mix )
( cd $Q && V41_WORK=$HOME/src/t16q/calib5/passw $PY v41_calib.py prep-mix --base $HOME/src/t16q/calib5/v5w --mix $HOME/src/t16q/calib5/pool.jsonl 2>&1 | grep prep-mix )
$PY - <<'PY'
import json, collections
for d in ("v5w", "passw"):
    m = json.load(open(f"{d}/meta.json"))
    if d == "passw":
        m["pool"] = [x.startswith("pool_") for x in m["domains"]]
        json.dump(m, open(f"{d}/meta.json", "w"))
    c = collections.Counter(m["domains"])
    print(f"{d}: {m['n']} rows = {m['n'] * m['seq']} tokens; pool rows {sum(m.get('pool', []))}; {dict(sorted(c.items(), key=lambda kv: -kv[1]))}")
PY
