"""quant_variants.py - one-layer A/B of expert quantization variants on dumped experts (V41_DUMP from v41_quant.py).

For each dumped expert (weights, fit Hessians, held-out inputs) and each variant, quantize gate/up (H1) and down (H2),
then measure the held-out output error of the whole SwiGLU expert: rel = sum((f(x; W_hat) - f(x; W))^2) / sum(f^2).

Variants:
  base      today's pipeline: static diag(H)-weighted scale search, then GPTQ (scalar tiers ptq1 / q2_0 / q3_k)
  refit     base, then a least-squares refit of every group scale against the full H (codes fixed), rounded to fp16,
            alternated with GPTQ (ITERS times)
  had       a 128-wide block Hadamard on the input dimension first (W R, R^T H R), then base
  had+refit both
  iq2xxs    llama.cpp IQ2_XXS (2.0625 bpw): 8-weight vectors from its 256-entry grid with sign patterns, 4-bit
            scale per 32 x fp16 per 256; static scales, then GPTQ with block-of-8 error feedback
  had+iq2xxs
  iq2n, iq2nr  IQ2_XXS by the native quantizer (v41_iq2.py), iq2nr with 2 rounds of scale refit

usage: python quant_variants.py DUMPDIR GRIDS.json [--variants base,refit,...] [--limit N]
"""
import argparse, glob, json, math, os, sys, time
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "v41"), os.path.join(HERE, "..")]   # repo layout, or the flat build-box copy
import v41_gptq as G  # noqa: E402

dev = "cuda"
ROWS = 4096


def expert_f(x, W1, W3, W2, g, lim):
    out = []
    for c in range(0, len(x), ROWS):
        xx, gg = x[c:c + ROWS].float(), g[c:c + ROWS].float()
        gate, up = xx @ W1.T, xx @ W3.T
        if lim > 0:
            up = up.clamp(-lim, lim); gate = gate.clamp(max=lim)
        out.append((F.silu(gate) * up * gg) @ W2.T)
    return torch.cat(out)


def hadamard(n):
    H = torch.ones(1, 1)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(n)


def block_rot(C, b=128, signs=False):
    """Block Hadamard of size b (b must divide C); with random +-1 column signs first (QuIP#-style) if signs."""
    R = torch.block_diag(*[hadamard(b)] * (C // b))
    if signs:
        g = torch.Generator().manual_seed(1017 + C)
        R = torch.diag(torch.randint(0, 2, (C,), generator=g).float() * 2 - 1) @ R
    return R.to(dev)


# ---------------------------------------------------------------- scalar tiers

def scalar_gptq(W, H, tier, damp, S=None):
    lo, hi, _, _ = G.TIERS[tier]
    C = W.shape[1]
    if S is None:
        d = torch.diag(H).clamp_min(0) + 1e-8
        S = G.expand_scales(G.search_scales(W, d, tier), tier, C)
    lov = torch.full((W.shape[0], 1), float(lo), device=dev)
    hiv = torch.full((W.shape[0], 1), float(hi), device=dev)
    Hinv, dead = G.hinv_chol(H, damp)
    Q = G.gptq(W, S, lov, hiv, Hinv, dead)
    return Q, S


def refit_scales(W, Q, S, H, tier, chunk=256):
    """Least-squares group scales for fixed integer codes, minimizing tr((W - s*q) H (W - s*q)^T) per row."""
    _, _, Gs, _ = G.TIERS[tier]
    R, C = W.shape
    ng = C // Gs
    qi = torch.round(Q / S)                      # integer codes
    Snew = S.clone()
    for r0 in range(0, R, chunk):
        r1 = min(R, r0 + chunk)
        q = qi[r0:r1]                            # [k, C]
        k = r1 - r0
        Qg = torch.zeros(k, C, ng, device=dev)
        Qg.view(k, ng, Gs, ng).diagonal(dim1=1, dim2=3).copy_(q.view(k, ng, Gs).transpose(1, 2))
        HQ = torch.einsum("ij,kjg->kig", H, Qg)  # [k, C, ng]
        A = torch.einsum("kig,kih->kgh", Qg, HQ) + 1e-6 * torch.eye(ng, device=dev)
        b = torch.einsum("kig,ki->kg", HQ, W[r0:r1])
        s = torch.linalg.solve(A, b.unsqueeze(-1)).squeeze(-1)          # [k, ng]
        s = s.half().float()
        s = torch.where(s == 0, torch.full_like(s, 1e-30), s)
        Snew[r0:r1] = s.repeat_interleave(Gs, dim=1)
    return qi * Snew, Snew


def scalar_variant(W, H, tier, damp, refit, iters=2):
    Q, S = scalar_gptq(W, H, tier, damp)
    if not refit or tier == "q3_k":              # q3_k's 6-bit sub-scales: no fp16 refit here
        return Q
    for _ in range(iters):
        Q, S = refit_scales(W, Q, S, H, tier)
        Q, S = scalar_gptq(W, H, tier, damp, S)
    Q, S = refit_scales(W, Q, S, H, tier)
    return Q


# ---------------------------------------------------------------- IQ2_XXS

class IQ2XXS:
    def __init__(self, grids):
        g = torch.tensor(grids["iq2xxs_grid"], dtype=torch.int64)
        self.grid = torch.stack([((g >> (8 * j)) & 0xFF) for j in range(8)], 1).float().to(dev)      # [256, 8]
        ks = torch.tensor(grids["ksigns_iq2xs"], dtype=torch.int64)
        self.signs = torch.stack([((ks >> j) & 1) for j in range(8)], 1).float().to(dev)            # [128, 8] even parity

    def round_groups(self, v, d, db):
        """v [N, 8] targets, d [N, 8] importance, db [N] scale. Best signed grid point (even sign parity)."""
        neg = (v < 0).float()
        odd = (neg.sum(1) % 2) == 1
        # odd parity: flip the sign bit of the element whose wrong sign costs least (smallest d * |v|)
        cost = d * v.abs()
        flip = torch.argmin(cost, 1)
        sgn = torch.where(neg > 0, -1.0, 1.0)
        sgn[odd, flip[odd]] *= -1
        t = v * sgn                                        # target magnitudes (negative where the sign was forced)
        cand = db[:, None, None] * self.grid[None]          # [N, 256, 8]
        e = ((t[:, None, :] - cand) ** 2 * d[:, None, :]).sum(-1)
        best = torch.argmin(e, 1)
        return sgn * db[:, None] * self.grid[best]

    def static_scales(self, W, dvec):
        """Per row and 32 columns: db = d * (0.5 + s4) / 4, d fp16 per 256, s4 in 0..15. Search ideal db per 32."""
        R, C = W.shape
        sub = W.view(R, C // 32, 32)
        amax = sub.abs().amax(-1).clamp_min(1e-12)                     # [R, C/32]
        best_e, best_db = None, None
        dd = dvec.view(1, C // 32, 32).expand(R, -1, -1)
        for c in [30.0, 34.0, 38.0, 43.0, 48.0, 55.0, 64.0]:
            db = amax / c
            e = torch.zeros_like(db)
            for r0 in range(0, R, 128):
                v = sub[r0:r0 + 128].reshape(-1, 8)
                dg = dd[r0:r0 + 128].reshape(-1, 8)
                q = self.round_groups(v, dg, db[r0:r0 + 128].repeat_interleave(4, 1).reshape(-1))
                e[r0:r0 + 128] = ((v - q) ** 2 * dg).view(-1, C // 32, 32).sum(-1)
            if best_e is None:
                best_e, best_db = e, db
            else:
                m = e < best_e
                best_e = torch.where(m, e, best_e); best_db = torch.where(m, db, best_db)
        # store as fp16 d per 256 and 4-bit s4 per 32
        blk = best_db.view(R, C // 256, 8)
        dsup = (blk.amax(-1, keepdim=True) * 4 / 15.5).half().float().clamp_min(1e-30)
        s4 = torch.clamp(torch.round(blk * 4 / dsup - 0.5), 0, 15)
        return (dsup * (0.5 + s4) / 4).view(R, C // 32)                # [R, C/32] effective db

    def gptq(self, W, H, damp):
        R, C = W.shape
        dvec = torch.diag(H).clamp_min(0) + 1e-8
        db = self.static_scales(W, dvec)
        Hinv, dead = G.hinv_chol(H, damp)            # upper Cholesky U of H^-1
        W = W.clone(); W[:, dead] = 0
        Q = torch.zeros_like(W)
        for c0 in range(0, C, 8):
            c1 = c0 + 8
            wb = W[:, c0:c1]
            qb = self.round_groups(wb, dvec[c0:c1].expand(R, 8), db[:, c0 // 32])
            Q[:, c0:c1] = qb
            if c1 < C:
                E = torch.linalg.solve_triangular(Hinv[c0:c1, c0:c1], (wb - qb).T, upper=True, left=True).T
                W[:, c1:] -= E @ Hinv[c0:c1, c1:]
        return Q


class IQ2Native:
    """IQ2_XXS by the native quantizer (v41_iq2.py): same rules as IQ2XXS; `iters` rounds of least-squares scale refit
    + GPTQ (iq2nr: 2, then a final refit)."""
    def __init__(self, iters):
        self.iters = iters

    def gptq(self, W, H, damp):
        import v41_iq2
        return v41_iq2.quantize(W.float().contiguous(), H.float(), damp, self.iters)[0]


# ---------------------------------------------------------------- IQ1_S / IQ1_M

class IQ1:
    """llama.cpp IQ1_S (1.5625 bpw: 3-bit odd scale + shared shift per 32) / IQ1_M (1.75 bpw: 3-bit odd scale per 16,
    shift per 8, fp16 super-scale per 256). Each 8-weight group: one of 2048 {-1,0,1} grid vectors, y = dl*(g + delta)."""
    def __init__(self, grids, kind):
        g = torch.tensor([x - (1 << 64) if x >= 1 << 63 else x for x in grids["iq1s_grid"]], dtype=torch.int64)
        b = torch.stack([((g >> (8 * j)) & 0xFF) for j in range(8)], 1)
        self.grid = torch.where(b > 127, b - 256, b).float().to(dev)            # [2048, 8] in {-1,0,1}
        self.kind = kind
        self.sub = 16 if kind == "m" else 32                                      # scale granularity

    def round_groups(self, v, d, dl, delta=None):
        """v,d [N,8]; dl [N]; delta [N] fixed (IQ1_S) or None (IQ1_M: best of +-0.125 per group)."""
        out = torch.empty_like(v)
        for c0 in range(0, len(v), 2048):
            vv, dd, ll = v[c0:c0 + 2048], d[c0:c0 + 2048], dl[c0:c0 + 2048]
            if delta is None:
                best_e, best_q = None, None
                for dt in (0.125, -0.125):
                    cand = ll[:, None, None] * (self.grid[None] + dt)
                    e = ((vv[:, None, :] - cand) ** 2 * dd[:, None, :]).sum(-1)
                    i = torch.argmin(e, 1)
                    ee = e.gather(1, i[:, None])[:, 0]
                    q = ll[:, None] * (self.grid[i] + dt)
                    if best_e is None:
                        best_e, best_q = ee, q
                    else:
                        m = ee < best_e
                        best_e = torch.where(m, ee, best_e); best_q = torch.where(m[:, None], q, best_q)
                out[c0:c0 + 2048] = best_q
            else:
                dt = delta[c0:c0 + 2048]
                cand = ll[:, None, None] * (self.grid[None] + dt[:, None, None])
                e = ((vv[:, None, :] - cand) ** 2 * dd[:, None, :]).sum(-1)
                out[c0:c0 + 2048] = ll[:, None] * (self.grid[torch.argmin(e, 1)] + dt[:, None])
        return out

    def static_scales(self, W, dvec):
        R, C = W.shape
        S = self.sub
        nsub = C // S
        sub = W.view(R, nsub, S)
        amax = sub.abs().amax(-1).clamp_min(1e-12)
        dd = dvec.view(1, nsub, S).expand(R, -1, -1)
        best_e, best_dl, best_dt = None, None, None
        deltas = (0.125, -0.125) if self.kind == "s" else (None,)
        for c in [0.7, 0.85, 1.0, 1.15, 1.3, 1.5]:
            dl = amax / c
            for dt in deltas:
                e = torch.zeros_like(dl)
                for r0 in range(0, R, 64):
                    v = sub[r0:r0 + 64].reshape(-1, 8)
                    dg = dd[r0:r0 + 64].reshape(-1, 8)
                    l = dl[r0:r0 + 64].repeat_interleave(S // 8, 1).reshape(-1)
                    q = self.round_groups(v, dg, l, None if dt is None else torch.full_like(l, dt))
                    e[r0:r0 + 64] = ((v - q) ** 2 * dg).view(-1, nsub, S).sum(-1)
                if best_e is None:
                    best_e, best_dl = e, dl
                    best_dt = torch.full_like(dl, 0.125 if dt is None else dt)
                else:
                    m = e < best_e
                    best_e = torch.where(m, e, best_e); best_dl = torch.where(m, dl, best_dl)
                    if dt is not None:
                        best_dt = torch.where(m, torch.full_like(dl, dt), best_dt)
        # store: fp16 d per 256, odd 3-bit multiplier per sub-block: dl = d * (2s + 1), s in 0..7
        per = 256 // S
        blk = best_dl.view(R, C // 256, per)
        dsup = (blk.amax(-1, keepdim=True) / 15.0).half().float().clamp_min(1e-30)
        s3 = torch.clamp(torch.round((blk / dsup - 1) / 2), 0, 7)
        return (dsup * (2 * s3 + 1)).view(R, nsub), best_dt

    def gptq(self, W, H, damp):
        R, C = W.shape
        dvec = torch.diag(H).clamp_min(0) + 1e-8
        dl, dt = self.static_scales(W, dvec)
        Hinv, dead = G.hinv_chol(H, damp)
        W = W.clone(); W[:, dead] = 0
        Q = torch.zeros_like(W)
        for c0 in range(0, C, 8):
            c1 = c0 + 8
            wb = W[:, c0:c1]
            j = c0 // self.sub
            qb = self.round_groups(wb, dvec[c0:c1].expand(R, 8), dl[:, j], dt[:, j] if self.kind == "s" else None)
            Q[:, c0:c1] = qb
            if c1 < C:
                E = torch.linalg.solve_triangular(Hinv[c0:c1, c0:c1], (wb - qb).T, upper=True, left=True).T
                W[:, c1:] -= E @ Hinv[c0:c1, c1:]
        return Q


# ---------------------------------------------------------------- driver

def quant_pair(W1, W3, W2, H1, H2, variant, tiers, iq, damp):
    # variant: [had|hads][N][+refit|+<codebook>]  e.g. had, had512+refit, hads128, iq2xxs
    import re
    m = re.match(r"(hads?)(\d*)", variant)
    rot = bool(m)
    bsz = int(m.group(2)) if m and m.group(2) else 128
    sgn = bool(m) and m.group(1) == "hads"
    base = variant.split("+")[-1] if "+" in variant else ("base" if rot else variant)

    def qmat(Wst, H, tier):
        C = Wst.shape[1]
        b = bsz if C % bsz == 0 else 128
        Rm = block_rot(C, b, sgn) if rot else None
        Wq, Hq = (Wst @ Rm, Rm.T @ H @ Rm) if rot else (Wst, H)
        if base in iq:
            Q = iq[base].gptq(Wq, Hq, damp)
        else:
            Q = scalar_variant(Wq, Hq, tier, damp, refit=(base == "refit"))
        return Q @ Rm.T if rot else Q

    gu = qmat(torch.cat([W1, W3]), H1, tiers[0])
    q1, q3 = gu[:W1.shape[0]], gu[W1.shape[0]:]
    q2 = qmat(W2, H2, tiers[1])
    return q1, q3, q2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump"); ap.add_argument("grids")
    ap.add_argument("--variants", default="base,refit,had,had+refit,iq2xxs,had+iq2xxs")
    ap.add_argument("--tiers", default="ptq1,q2_0,q3_k")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--damp", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    grids = json.load(open(a.grids))
    iq = {"iq2xxs": IQ2XXS(grids), "iq2n": IQ2Native(0), "iq2nr": IQ2Native(2)}
    if "iq1s_grid" in grids:   # IQ1 grids only in the IQ1 grids file
        iq.update({"iq1s": IQ1(grids, "s"), "iq1m": IQ1(grids, "m")})
    files = sorted(glob.glob(os.path.join(a.dump, "*.pt")))
    if a.limit:
        files = files[:a.limit]
    rows = []
    for f in files:
        dmp = torch.load(f)
        W1, W3, W2 = (dmp[k].float().to(dev) for k in ("W1", "W3", "W2"))
        H1, H2 = dmp["H1"].to(dev), dmp["H2"].to(dev)
        xv, gv, lim = dmp["xv"].to(dev), dmp["gv"].to(dev), dmp["lim"]
        if len(xv) == 0:
            continue
        Y = expert_f(xv, W1, W3, W2, gv, lim)
        ye = float(Y.square().sum())
        for v in a.variants.split(","):
            cb = [k for k in iq if v.endswith(k)]
            tier_list = cb if cb else a.tiers.split(",")
            for t in tier_list:
                t0 = time.time()
                q1, q3, q2 = quant_pair(W1, W3, W2, H1, H2, v, (t, t), iq, a.damp)
                err = float((expert_f(xv, q1, q3, q2, gv, lim) - Y).square().sum())
                rel = err / ye
                # abs: squared output error per held-out token (decoder layers' relative errors are tiny; compare both)
                r = {"file": os.path.basename(f), "ex": dmp["ex"], "n_eval": len(xv), "variant": v, "tier": t, "rel": rel, "abs": err / len(xv),
                     "s": round(time.time() - t0, 1)}
                rows.append(r)
                print(json.dumps(r), flush=True)
    if a.out:
        json.dump(rows, open(a.out, "w"))
    # summary: mean rel per (variant, tier), and token-weighted
    from collections import defaultdict
    agg = defaultdict(list)
    for r in rows:
        agg[(r["variant"], r["tier"])].append(r)
    print("\nvariant         tier     mean rel   token-weighted   abs per token (token-weighted)   n")
    for (v, t), rs in sorted(agg.items()):
        m = sum(r["rel"] for r in rs) / len(rs)
        nt = sum(r["n_eval"] for r in rs)
        w = sum(r["rel"] * r["n_eval"] for r in rs) / nt
        ab = sum(r["abs"] * r["n_eval"] for r in rs) / nt
        print(f"{v:<15} {t:<7} {m:9.4f} {w:14.4f} {ab:20.6g} {len(rs):14d}")


if __name__ == "__main__":
    main()
