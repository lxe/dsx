#!/usr/bin/env python3
"""Split decode tokens in an nsys sqlite export (dsx gen, -t cuda) into GPU0 busy time, idle gaps and the host
interval between tokens.

Token boundaries are the logits downloads (device-to-host copies of vocab*4 bytes). For each token after warm-up:
  - token time: logits copy end to the next logits copy end
  - between tokens: logits copy end to GPU0's first kernel of the next token (host sampling, adapt step, embedding,
    uploads, launches), with the CUDA API calls the host thread made in that interval
  - GPU0 busy: union of kernel intervals; idle inside the token = the rest
  - head: the last kernel before the logits copy (the LM head GEMV)
Usage: nsys_token_split.py dec7.sqlite [vocab]
"""
import sqlite3, sys, collections, statistics as st

db = sqlite3.connect(sys.argv[1])
vocab = int(sys.argv[2]) if len(sys.argv) > 2 else 129280
names = dict(db.execute("select id, value from StringIds"))
tables = {r[0] for r in db.execute("select name from sqlite_master where type='table'")}

cp = db.execute(
    "select start, end, deviceId, bytes, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY order by start").fetchall()
logit = [(s, e) for s, e, d, b, k in cp if b == vocab * 4 and k == 2]  # 2 = device to host
dev0 = min(d for s, e, d, b, k in cp if b == vocab * 4)
kern = db.execute(
    "select start, end, deviceId, shortName from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
k0 = [(s, e, names.get(n, str(n))) for s, e, d, n in kern if d == dev0]
api = []
for t in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"):
    if t in tables:
        api += db.execute(f"select start, end, globalTid, nameId from {t}").fetchall()
api.sort()
print(f"{len(logit)} logits copies, {len(k0)} GPU0 kernels, {len(api)} API calls")

import bisect
ks = [k[0] for k in k0]
apis = [a[0] for a in api]
rows = []
gap_api = collections.Counter()
gap_api_n = collections.Counter()
kname_t = collections.Counter()
skip = 8  # warm-up tokens
for i in range(skip, len(logit) - 1):
    a, b = logit[i][1], logit[i + 1][1]
    j0 = bisect.bisect_left(ks, a)
    j1 = bisect.bisect_left(ks, logit[i + 1][0])
    ts = k0[j0:j1]
    if not ts:
        continue
    first = ts[0][0]
    busy, cur_s, cur_e = 0, None, None
    for s, e, n in ts:
        kname_t[n] += e - s
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                busy += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    busy += cur_e - cur_s
    head = ts[-1]
    # host calls between the logits copy and the first kernel
    q0, q1 = bisect.bisect_left(apis, a), bisect.bisect_left(apis, first)
    for s, e, tid, n in api[q0:q1]:
        nm = names.get(n, str(n))
        gap_api[nm] += min(e, first) - s
        gap_api_n[nm] += 1
    rows.append(dict(tok=b - a, gap=first - a, busy=busy, idle=(b - a) - busy, head=head[1] - head[0],
                     tail=b - head[1], headname=head[2]))

n = len(rows)
med = lambda k: st.median(r[k] for r in rows) / 1e6
mean = lambda k: st.mean(r[k] for r in rows) / 1e6
print(f"{n} tokens; ms/token median (mean)")
for k in ("tok", "busy", "idle", "gap", "head", "tail"):
    print(f"  {k:<5} {med(k):7.3f} ({mean(k):.3f})")
print(f"  head kernel: {collections.Counter(r['headname'] for r in rows).most_common(2)}")
print("host API calls between tokens (ms/token, calls/token):")
for nm, t in gap_api.most_common(12):
    print(f"  {nm:<40} {t / n / 1e6:7.3f}  {gap_api_n[nm] / n:6.1f}")
print("GPU0 kernels by time (ms/token):")
for nm, t in kname_t.most_common(25):
    print(f"  {nm[:50]:<50} {t / n / 1e6:7.3f}")
