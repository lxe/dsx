#!/usr/bin/env python3
"""Fit 8- and 4-level codebooks (FNUZ fp8 entries) for re-quantizing the checkpoint's MXFP4 experts with one e4m3
scale per 32 weights. Lloyd iterations with a per-group scale search, on a random sample of experts.
usage: fit_codebook.py CKPT_DIR [n_experts]"""
import json, struct, sys, numpy as np
d = sys.argv[1]; nexp = int(sys.argv[2]) if len(sys.argv) > 2 else 48
ix = json.load(open(f"{d}/model.safetensors.index.json"))["weight_map"]
E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], np.float32)
hdr = {}
def tensor(name):
    f = f"{d}/{ix[name]}"
    if f not in hdr:
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]; hdr[f] = (8 + n, json.loads(fh.read(n)))
    off, h = hdr[f]; t = h[name]; a, b = t["data_offsets"]
    return np.fromfile(f, np.uint8, count=b - a, offset=off + a).reshape(t["shape"])
rng = np.random.default_rng(0)
groups = []
for _ in range(nexp):
    l, e, w = rng.integers(0, 40), rng.integers(0, 384), rng.choice(["w1", "w2", "w3"])
    q = tensor(f"layers.{l}.ffn.experts.{e}.{w}.weight"); s = tensor(f"layers.{l}.ffn.experts.{e}.{w}.scale")
    v = np.stack([E2M1[q & 15], E2M1[q >> 4]], -1).reshape(q.shape[0], -1)  # e2m1 values, scale-free
    groups.append(v.reshape(-1, 32))
g = np.concatenate(groups); print("groups", g.shape, file=sys.stderr)
g = g[rng.choice(len(g), min(len(g), 400000), replace=False)]

def fnuz_grid():
    vals = []
    for b in range(256):
        if b == 0x80: continue
        s, e, m = b >> 7, (b >> 3) & 15, b & 7
        v = (m / 8) * 2.0 ** -7 if e == 0 else (1 + m / 8) * 2.0 ** (e - 8)
        vals.append(-v if s else v)
    return np.unique(np.array(vals, np.float32))
GRID = fnuz_grid()
def rnd(x):  # nearest FNUZ fp8 value
    i = np.clip(np.searchsorted(GRID, x), 1, len(GRID) - 1)
    return np.where(np.abs(GRID[i] - x) < np.abs(GRID[i - 1] - x), GRID[i], GRID[i - 1])

def assign(g, cb, fac=np.linspace(0.55, 1.25, 15)):
    amax = np.abs(g).max(1, keepdims=True) + 1e-12
    best_err = np.full(len(g), np.inf); best_a = np.zeros(len(g)); best_q = np.zeros(g.shape, np.int64)
    mids = (cb[1:] + cb[:-1]) / 2
    for f in fac:
        a = amax[:, 0] * f / np.abs(cb).max()
        a = rnd(a * 64) / 64  # e4m3 scale relative to a per-matrix multiplier (here 1/64)
        a = np.maximum(a, 1e-6)
        q = np.searchsorted(mids, g / a[:, None])
        err = ((g - a[:, None] * cb[q]) ** 2).sum(1)
        m = err < best_err
        best_err[m] = err[m]; best_a[m] = a[m]; best_q[m] = q[m]
    return best_err, best_a, best_q

def lloyd(cb, iters=12):
    for it in range(iters):
        err, a, q = assign(g, cb)
        new = cb.copy()
        for c in range(len(cb)):
            m = q == c
            if m.any():
                # least squares for the entry given the scales: sum(a*g)/sum(a^2)
                aa = np.broadcast_to(a[:, None], g.shape)[m]
                new[c] = (aa * g[m]).sum() / (aa * aa).sum()
        new = np.sort(rnd(new / np.abs(new).max() * 8))  # keep entries on the FNUZ grid, max magnitude 8
        rel = err.sum() / (g ** 2).sum()
        print(f"{len(cb)} levels it {it}: rel err {rel:.5f} cb {new.tolist()}", file=sys.stderr)
        if np.array_equal(new, cb): break
        cb = new
    return cb, rel

cb3, e3 = lloyd(np.sort(rnd(np.array([-2.152, -1.344, -0.756, -0.245, 0.245, 0.756, 1.344, 2.152]) / 2.152 * 8)))
cb2, e2 = lloyd(np.sort(rnd(np.array([-1.51, -0.4528, 0.4528, 1.51]) / 1.51 * 8)))
print(json.dumps({"cb3": cb3.tolist(), "cb3_rel_err": e3, "cb2": cb2.tolist(), "cb2_rel_err": e2}))
