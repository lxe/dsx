# Derived from Xolotl tools/v41 (github.com/Indras-Mirror/Xolotl, commit 7375e305), MIT License,
# Copyright (c) 2026 Niko1221 and the Strata contributors; see LICENSE-Xolotl. Modified for dsx.
"""v41_quant.py - the quantize pass: every routed expert of every layer at its chosen tier, written as the GGUF block
bytes v41_gguf.py --experts reads.  A second layer-by-layer chain through DeepSeek's model.py (v41_calib's
cmd_layer, with its expert hook), in its own work dir.

    v41_quant.py init  --qwork DIR                       links tokens/meta/hashes from the calibration work dir,
                                                         writes h/L00 + pre/L00 (CPU)
    v41_quant.py layer L --qwork DIR (--map alloc.json | --tier q2_0) [--propagate quant|teacher] [--gc]
    v41_quant.py fill  --gguf SHARD1 --qdir DIR/q            QDIR -> an --experts-in-place file
    v41_quant.py head  --qwork DIR [--layer 40]          per-domain ppl / top-1 of the QUANTIZED chain (v41_calib head)

Per expert: GPTQ (v41_gptq, static diag(H)-weighted scales, damp 1.0) at its tier - gate+up stacked on the input
Hessian, down on the Hessian of the clamped SwiGLU x routing weight - fitted on the FIT sequences only, so the held-out
sequences stay held out: the head gate's eval split is then an honest comparison of a mixed map against the all-Q2_0
control (run the same chain with --tier q2_0 in another qwork).  Experts with fewer than --min-fit fit tokens use
an identity Hessian (static scales, no error feedback); experts no token reached are written the same way.

--propagate quant (default): the quantized experts' outputs feed layer L+1 (sequential GPTQ: later layers are fitted
on the inputs the quantized model will really see); teacher: the FP4 outputs, as the screening run.
Dense weights (attention, shared expert) stay at the source's FP8 here; the GGUF stores them as Q8_0 (rel 5e-3).

Output (Sink): --gguf SHARD1 writes each expert in place into the final file (v41_gguf.py --experts-in-place made it,
with the same map); --no-write writes none (the control); default DIR/q/LNN.bin (each expert's gate, up, down rows
in expert order; v41_gguf --experts reads it).  Always DIR/q/LNN.json ({"gu_type": [...], "d_type": [...], "experts": {e: stats}}), written after the layer's MoE - so
a layer is done when its json exists (the chain resumes like the calibration one).  GSQ refinement: TODO (plugs in
where GPTQ is called; same output).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import v41_calib as C  # noqa: E402  (sets up model.py / kernel.py)
import v41_formats as F  # noqa: E402
import v41_gptq as G  # noqa: E402

NATIVE = os.environ.get("V41_NATIVE", "1") != "0"   # native/qkern.cu GPTQ + scale search, batched over experts
if NATIVE:
    import v41_native as NV  # noqa: E402
    import v41_iq2 as IQ2  # noqa: E402  (IQ2_XXS tier: codebook GPTQ, native/qkern.cu)
import v41_gptq_graph as GG  # noqa: E402  (bit-identical CUDA-graph gptq, 8x faster)
from concurrent.futures import ThreadPoolExecutor  # noqa: E402

K = C.K
CALIB = C.WORK
WRITE_THREADS = int(os.environ.get("V41_WRITE_THREADS", "3"))
FF = 2304   # moe_intermediate_size (checked against each expert's w2)


def set_work(qwork):
    C.WORK = qwork
    os.makedirs(os.path.join(qwork, "q"), exist_ok=True)


def tiers_for(a, L):
    """(gu tiers [384], down tiers [384]) for layer L from --map (v41_alloc --out) or the uniform --tier."""
    gu, dn = [a.tier] * C.NE, [a.tier] * C.NE
    if a.map:
        m = json.load(open(a.map))["map"].get(str(L))
        if m is None:
            raise SystemExit(f"layer {L} is not in {a.map}")
        for e, pr in m.items():
            for k, t in pr.items():
                if k in ("all", "gu"):
                    gu[int(e)] = t
                if k in ("all", "dn"):
                    dn[int(e)] = t
    return gu, dn


def quant(Ws, H, tier, damp, do_gptq):
    """GPTQ of the stacked Ws (sharing H) at `tier`: [(Q, S)] per W, S in the stored format (v41_formats input)."""
    Cc = H.shape[0]
    d = torch.diag(H).clamp_min(0) + 1e-8
    lo, hi, _, _ = G.TIERS[tier]
    S = [G.search_scales(W, d, tier) for W in Ws]
    Wst = torch.cat(Ws)
    Sst = torch.cat([G.expand_scales(s, tier, Cc) for s in S])
    lov = torch.full((Wst.shape[0], 1), float(lo), device=Wst.device)
    hiv = torch.full((Wst.shape[0], 1), float(hi), device=Wst.device)
    if do_gptq:
        Hinv, dead = G.hinv_chol(H, damp)
        Q = GG.gptq(Wst, Sst, lov, hiv, Hinv, dead)
    else:
        Q = G.rtn(Wst, Sst, lov, hiv)
    out, r0 = [], 0
    for W, s in zip(Ws, S):
        out.append((Q[r0:r0 + W.shape[0]], s))
        r0 += W.shape[0]
    return out


class Sink:
    """Where a layer's expert bytes go: `qdir` (DIR/q/LNN.bin, expert order - v41_gguf --experts), `gguf` (in place,
    at the offsets of v41_gguf --experts-in-place's <shard>.experts.json; the types there must be this layer's), or
    `none` (--no-write: the all-Q2_0 control only needs its head-gate numbers)."""

    def __init__(self, a, L, gt, dt, size):
        self.size = size
        self.mode = "none" if a.no_write else ("gguf" if a.gguf else "qdir")
        if self.mode == "qdir":
            self.off = np.cumsum([0] + [2 * g + d for g, d in size]).tolist()
            self.fd = os.open(C.wpath("q", f"L{L:02d}.bin"), os.O_RDWR | os.O_CREAT, 0o644)
            os.ftruncate(self.fd, self.off[-1])
        elif self.mode == "gguf":
            lay = json.load(open(a.gguf + ".experts.json"))["layers"].get(str(L))
            if lay is None:
                raise SystemExit(f"{a.gguf}: layer {L} is not in the file")
            if lay["gu_type"] != list(gt) or lay["d_type"] != list(dt):
                raise SystemExit(f"{a.gguf}: layer {L} tiers differ from this run's map")
            self.goff = {int(e): o for e, o in lay["off"].items()}
            self.fd = os.open(a.gguf, os.O_RDWR)

    def put(self, ex, gate, up, down):
        g, d = self.size[ex]
        assert (len(gate), len(up), len(down)) == (g, g, d), (ex, len(gate), len(up), len(down), g, d)
        if self.mode == "qdir":
            os.pwrite(self.fd, gate + up + down, self.off[ex])
        else:
            o = self.goff[ex]
            for role, b in (("gate", gate), ("up", up), ("down", down)):
                os.pwrite(self.fd, b, o[role])

    def close(self):
        if self.mode != "none":
            os.fsync(self.fd)
            os.close(self.fd)


class Layer:
    """One layer's output file and the per-expert work the hooks do."""

    def __init__(self, a, L):
        self.a, self.L = a, L
        self.gu, self.dn = tiers_for(a, L)
        self.gt = [F.TIER_TYPE[t] for t in self.gu]
        self.dt = [F.TIER_TYPE[t] for t in self.dn]
        self.size = [(F.nbytes(self.gt[e], (C.D, FF)), F.nbytes(self.dt[e], (FF, C.D))) for e in range(C.NE)]
        self.sink = Sink(a, L, self.gt, self.dt, self.size)
        self.done, self.info = set(), {}
        # encode + pwrite on worker threads while the GPU does the next expert (numpy, same bytes; distinct offsets)
        self.pool, self.pending = ThreadPoolExecutor(max_workers=max(WRITE_THREADS, 1)), []

    def _encode_put(self, ex, mats):
        b = [(s if t == "iq2xxs" else _pack_codes(q, s, t) if q.dtype == np.int8 else F.encode_tier(q, s, t)).tobytes() for q, s, t in mats]
        self.sink.put(ex, *b)

    def write(self, ex, q1, q3, q2):
        if self.sink.mode != "none":
            if NATIVE:   # integer codes computed on the GPU: int8 to the host instead of fp32 values (4x fewer bytes)
                mats = [(None, IQ2.encode(*(x.cpu().numpy() for x in s[1:])), t) if t == "iq2xxs" else
                        (_gpu_codes(q, s, t).cpu().numpy(), s.cpu().numpy(), t)
                        for (q, s), t in ((q1, self.gu[ex]), (q3, self.gu[ex]), (q2, self.dn[ex]))]
            else:
                mats = [(q.cpu().numpy(), s.cpu().numpy(), t)
                        for (q, s), t in ((q1, self.gu[ex]), (q3, self.gu[ex]), (q2, self.dn[ex]))]
            if WRITE_THREADS == 0:                 # V41_WRITE_THREADS=0: in-line, as before
                self._encode_put(ex, mats)
                self.done.add(ex)
                return
            while len(self.pending) >= 6:          # bound the host copies in flight (~115 MB each)
                self.pending.pop(0).result()       # re-raises a worker's error here
            self.pending.append(self.pool.submit(self._encode_put, ex, mats))
        self.done.add(ex)

    def drain(self):
        for f in self.pending:
            f.result()
        self.pending = []
        self.pool.shutdown(wait=True)

    def weights(self, exp):
        return (K.dequant_fp4_weight(exp.w1.weight, exp.w1.scale), K.dequant_fp4_weight(exp.w3.weight, exp.w3.scale),
                K.dequant_fp4_weight(exp.w2.weight, exp.w2.scale))

    def expert(self, L, ex, exp, xe, ge, fit, lim, out_teacher):
        a = self.a
        W1, W3, W2 = self.weights(exp)
        assert W2.shape == (C.D, FF), W2.shape
        xf, gf = xe[fit], ge[fit]          # bf16; C.expert_f / C.act_hessians convert per chunk
        n_fit = int(fit.sum())
        do = n_fit >= a.min_fit and ex < a.gptq_experts
        if do:
            H1, H2 = C.act_hessians(xf, W1, W3, gf, lim)
        else:
            H1 = torch.eye(C.D, device=xe.device)
            H2 = torch.eye(W2.shape[1], device=xe.device)
        q1, q3 = quant([W1, W3], H1, self.gu[ex], a.damp, do)
        (q2,) = quant([W2], H2, self.dn[ex], a.damp, do)
        self.write(ex, q1, q3, q2)
        st = {"gu": self.gu[ex], "dn": self.dn[ex], "n_fit": n_fit, "gptq": do}
        xv, gv = xe[~fit], ge[~fit]
        if len(xv):   # held-out output error at the chosen tiers (the fit used none of these tokens)
            Y = C.expert_f(xv, W1, W3, W2, gv, lim)
            st["q_err"] = C.sq_err(xv, q1[0], q3[0], q2[0], gv, lim, Y)
            st["y_energy"] = float(Y.square().sum())
        if a.propagate == "quant":
            # chunk by chunk straight into the output dtype: a hot expert's whole fp32 output (L7: 1.88 GiB) OOM'd
            out = torch.empty((len(xe), C.D), dtype=out_teacher.dtype, device=xe.device)
            for c in range(0, len(xe), C.ROWS):
                out[c:c + C.ROWS] = C.expert_f(xe[c:c + C.ROWS], q1[0], q3[0], q2[0], ge[c:c + C.ROWS], lim).to(out.dtype)
        else:
            out = out_teacher
        return out, st

    def expert_batch(self, L, items):
        """Native path (V41_NATIVE=1): items = [(ex, exp, xe, ge, fit, lim, out_teacher)]. Same per-expert result as
        expert() (GPTQ on the fit tokens at the mapped tiers, static scales), with the scale search, H^-1 and GPTQ
        done by native/qkern.cu for the whole batch: ~10x less GPU time per expert, no per-column launches."""
        a = self.a
        prep = []
        tm = self.tm = getattr(self, "tm", {})
        def mark(k, t0):
            torch.cuda.synchronize()
            tm[k] = tm.get(k, 0.0) + time.time() - t0
            return time.time()
        t0 = time.time()
        orig = []
        for ex, exp, xe, ge, fit, lim, out_t in items:
            W1, W3, W2 = self.weights(exp)
            n_fit = int(fit.sum())
            do = n_fit >= a.min_fit and ex < a.gptq_experts
            if do:
                H1, H2 = C.act_hessians(xe[fit], W1, W3, ge[fit], lim)
            else:
                H1, H2 = torch.eye(C.D, device=xe.device), torch.eye(W2.shape[1], device=xe.device)
            orig.append((W1, W3, W2))
            dump = os.environ.get("V41_DUMP")   # DIR: save every 8th expert's weights, Hessians, held-out inputs (original basis, before --rot)
            # in the layers listed by V41_DUMP_LAYER (comma-separated; default all)
            if dump and do and ex % 8 == 0 and str(L) in os.environ.get("V41_DUMP_LAYER", str(L)).split(","):
                os.makedirs(dump, exist_ok=True)
                torch.save({"L": L, "ex": ex, "W1": W1.bfloat16().cpu(), "W3": W3.bfloat16().cpu(), "W2": W2.bfloat16().cpu(),
                            "H1": H1.cpu(), "H2": H2.cpu(), "xv": xe[~fit].cpu(), "gv": ge[~fit].cpu(), "lim": lim,
                            "gu": self.gu[ex], "dn": self.dn[ex], "n_fit": n_fit}, os.path.join(dump, f"L{L:02d}_e{ex:03d}.pt"))
            if a.rot:   # quantize W R against R H R; evaluate and propagate in the original basis
                Rd, Rf = rot_mat(C.D, xe.device), rot_mat(W2.shape[1], xe.device)
                W1, W3, W2 = W1.float() @ Rd, W3.float() @ Rd, W2.float() @ Rf
                H1, H2 = Rd @ H1.float() @ Rd, Rf @ H2.float() @ Rf
            prep.append((ex, W1, W3, W2, H1, H2, do, n_fit))
        t0 = mark("dequant+hessians", t0)
        q = {}
        for part in ("gu", "dn"):
            tiers = self.gu if part == "gu" else self.dn
            Ws, Ss, los, his, Hs, idx = [], [], [], [], [], []
            iq = []   # IQ2_XXS experts of this part: (i, mats, W stacked, H)
            for i, (ex, W1, W3, W2, H1, H2, do, n_fit) in enumerate(prep):
                H = H1 if part == "gu" else H2
                mats = [W1, W3] if part == "gu" else [W2]
                t = tiers[ex]
                if t == "iq2xxs":   # without GPTQ H is the identity: the same path rounds group by group
                    iq.append((i, mats, torch.cat(mats).float(), H.float()))
                    continue
                lo, hi, _, _ = G.TIERS[t]
                d = torch.diag(H).clamp_min(0) + 1e-8
                sc = [NV.search_scales(W, d, t) for W in mats]
                Wst = torch.cat(mats)
                Sst = torch.cat([G.expand_scales(x, t, H.shape[0]) for x in sc])
                if not do:   # no GPTQ: static scales, round to nearest (as quant(..., do_gptq=False))
                    lov = torch.full((Wst.shape[0], 1), float(lo), device=Wst.device, dtype=torch.float32)
                    hiv = torch.full((Wst.shape[0], 1), float(hi), device=Wst.device, dtype=torch.float32)
                    Q = G.rtn(Wst, Sst, lov, hiv)
                    q[(i, part)] = self._split(Q, mats, sc)
                    continue
                Ws.append(Wst); Ss.append(Sst); Hs.append(H); idx.append((i, mats, sc))
                los.append(torch.full((Wst.shape[0],), float(lo), device=Wst.device, dtype=torch.float32))
                his.append(torch.full((Wst.shape[0],), float(hi), device=Wst.device, dtype=torch.float32))
            t0 = mark("scales", t0)
            if Ws:
                Hinv, dead = _hinv_chol_batched(torch.stack(Hs), a.damp)
                t0 = mark("hinv", t0)
                Qb = NV.gptq_batched(torch.stack(Ws), torch.stack(Ss), torch.stack(los), torch.stack(his), Hinv, dead)
                t0 = mark("gptq", t0)
                if a.refit:   # least-squares scales for GPTQ's codes, GPTQ again on them, a final refit (ptq1 / q2_0)
                    tk = [tiers[prep[i][0]] for (i, _, _) in idx]
                    ok = [t in ("ptq1", "q2_0") for t in tk]
                    if any(ok):
                        def refit_all(Qx, Sx):
                            gs = [refit_scales(Ws[k], Qx[k], Sx[k], Hs[k], G.TIERS[tk[k]][2]) if ok[k] else None for k in range(len(idx))]
                            ex = [G.expand_scales(gs[k], tk[k], Ws[k].shape[1]) if ok[k] else Sx[k] for k in range(len(idx))]
                            return gs, ex
                        _, E1 = refit_all(Qb, Ss)
                        Qb = NV.gptq_batched(torch.stack(Ws), torch.stack(E1), torch.stack(los), torch.stack(his), Hinv, dead)
                        g2, E2 = refit_all(Qb, E1)
                        Qn = []
                        for k, (i, mats, sc) in enumerate(idx):
                            if not ok[k]:
                                Qn.append(Qb[k]); continue
                            codes = torch.where(E1[k].abs() < 1e-20, torch.zeros_like(Qb[k]), torch.round(Qb[k] / E1[k]))
                            Qn.append(codes * E2[k])
                            r0, sc2 = 0, []          # the stored group scales, split back per matrix
                            for W in mats:
                                sc2.append(g2[k][r0:r0 + W.shape[0]]); r0 += W.shape[0]
                            idx[k] = (i, mats, sc2)
                        Qb = torch.stack(Qn)
                    t0 = mark("refit", t0)
                del Hinv
                for k, (i, mats, sc) in enumerate(idx):
                    q[(i, part)] = self._split(Qb[k], mats, sc)
            if iq:
                q.update(self._iq2_part(iq, part, a))
                t0 = mark("iq2xxs", t0)
        res = []
        for i, (ex, W1, W3, W2, H1, H2, do, n_fit) in enumerate(prep):
            _, _, xe, ge, fit, lim, out_t = items[i]
            q1, q3 = q[(i, "gu")]
            (q2,) = q[(i, "dn")]
            t0 = time.time()
            self.write(ex, q1, q3, q2)
            W1, W3, W2 = orig[i]
            if a.rot:   # back to the original basis (R is its own inverse) for the error and the next layer's input
                Rd, Rf = rot_mat(C.D, W1.device), rot_mat(W2.shape[1], W1.device)
                q1, q3, q2 = (q1[0].float() @ Rd, q1[1]), (q3[0].float() @ Rd, q3[1]), (q2[0].float() @ Rf, q2[1])
            t0 = mark("write (to host)", t0)
            st = {"gu": self.gu[ex], "dn": self.dn[ex], "n_fit": n_fit, "gptq": do}
            xv, gv = xe[~fit], ge[~fit]
            if len(xv):
                Y = C.expert_f(xv, W1, W3, W2, gv, lim)
                st["q_err"] = C.sq_err(xv, q1[0], q3[0], q2[0], gv, lim, Y)
                st["y_energy"] = float(Y.square().sum())
            t0 = mark("held-out error", t0)
            if a.propagate == "quant":
                out = torch.empty((len(xe), C.D), dtype=out_t.dtype, device=xe.device)
                for c in range(0, len(xe), C.ROWS):
                    out[c:c + C.ROWS] = C.expert_f(xe[c:c + C.ROWS], q1[0], q3[0], q2[0], ge[c:c + C.ROWS], lim).to(out.dtype)
            else:
                out = out_t
            mark("quantized forward", t0)
            res.append((out, st))
        return res

    @staticmethod
    def _iq2_part(iq, part, a):
        """IQ2_XXS experts of one part, batched: scales, codebook GPTQ, and with --refit and V41_IQ2_REFIT=1 the scalar
        tiers' sequence (refit, GPTQ on the refit scales, a final refit). -> {(i, part): [(Q, ("iq2xxs", IDX, SGN, ls, dsup)) per matrix]}"""
        Ws = torch.stack([w for _, _, w, _ in iq])
        Hs = torch.stack([h for _, _, _, h in iq])
        ds = torch.diagonal(Hs, dim1=-2, dim2=-1).clamp_min(0) + 1e-8
        sc = [IQ2.search_scales(Ws[k], ds[k]) for k in range(len(iq))]
        DB = torch.stack([s[2] for s in sc])
        Hinv, dead = _hinv_chol_batched(Hs, a.damp)
        Qb, IDX, SGN = IQ2.gptq_batched(Ws, DB, ds, Hinv, dead)
        res = [(Qb[k], IDX[k], SGN[k], sc[k][1], sc[k][0]) for k in range(len(iq))]
        # refit lowers the fit tokens' error but not held-out error for the codebook (A/B, layer 20: 0.1463 -> 0.1467):
        # opt-in with V41_IQ2_REFIT=1
        if a.refit and os.environ.get("V41_IQ2_REFIT") == "1":
            r1 = [IQ2.refit(Ws[k], Qb[k], DB[k], Hs[k], (sc[k][0], sc[k][1])) for k in range(len(iq))]
            DB1 = torch.stack([r[3] for r in r1])
            Qb, IDX, SGN = IQ2.gptq_batched(Ws, DB1, ds, Hinv, dead)
            res = []
            for k in range(len(iq)):
                Qk, dsup, ls, _ = IQ2.refit(Ws[k], Qb[k], DB1[k], Hs[k], (r1[k][1], r1[k][2]))
                res.append((Qk, IDX[k], SGN[k], ls, dsup))
        del Hinv
        out = {}
        for (i, mats, _, _), (Qk, ik, sk, lk, dk) in zip(iq, res):
            parts, r0 = [], 0
            for W in mats:
                r1 = r0 + W.shape[0]
                parts.append((Qk[r0:r1], ("iq2xxs", ik[r0:r1], sk[r0:r1], lk[r0:r1], dk[r0:r1])))
                r0 = r1
            out[(i, part)] = parts
        return out

    @staticmethod
    def _split(Q, mats, sc):
        out, r0 = [], 0
        for W, s in zip(mats, sc):
            out.append((Q[r0:r0 + W.shape[0]], s))
            r0 += W.shape[0]
        return out

    def finish(self, L, blk, stats):
        for ex in range(C.NE):   # experts no token reached: static scales, identity importance
            if ex not in self.done:
                W1, W3, W2 = self.weights(blk.ffn.experts[ex])
                if self.a.rot:
                    Rd, Rf = rot_mat(C.D, W1.device), rot_mat(W2.shape[1], W1.device)
                    W1, W3, W2 = W1.float() @ Rd, W3.float() @ Rd, W2.float() @ Rf
                q1, q3 = quant([W1, W3], torch.eye(C.D, device=W1.device), self.gu[ex], self.a.damp, False)
                (q2,) = quant([W2], torch.eye(W2.shape[1], device=W1.device), self.dn[ex], self.a.damp, False)
                self.write(ex, q1, q3, q2)
                stats[ex].update({"gu": self.gu[ex], "dn": self.dn[ex], "gptq": False, "unrouted": True})
        self.drain()                              # every expert's bytes written before fsync + the done json
        self.sink.close()
        j = {"layer": L, "sink": self.sink.mode, "gu_type": self.gt, "d_type": self.dt, "propagate": self.a.propagate, "damp": self.a.damp,
             "map": self.a.map, "tier": self.a.tier, "rot": 128 if self.a.rot else 0, "experts": {str(e): stats[e] for e in range(C.NE)}}
        tmp = C.wpath("q", f"L{L:02d}.json.tmp")
        json.dump(j, open(tmp, "w"))
        os.replace(tmp, C.wpath("q", f"L{L:02d}.json"))
        if getattr(self, "tm", None):
            C.log(f"quant L{L} time: " + ", ".join(f"{k} {v:.1f} s" for k, v in self.tm.items()))
        e_q = sum(s.get("q_err", 0) for s in stats.values())
        e_y = sum(s.get("y_energy", 0) for s in stats.values())
        C.log(f"quant L{L}: {sum(2 * g + d for g, d in self.size) / 2**30:.2f} GiB -> {self.sink.mode}, held-out rel err {e_q / max(e_y, 1e-30):.4f}, "
              f"tiers gu {dict((t, self.gu.count(t)) for t in G.TIERS)} dn {dict((t, self.dn.count(t)) for t in G.TIERS)}")


def refit_scales(W, Q, Sx, H, G, chunk=256):
    """Least-squares group scales for fixed codes: per row, minimize (w - sum_g s_g q_g)^T H (w - sum_g s_g q_g).
    W, Q, Sx [R, C] (Sx: current scales expanded per weight), H [C, C]. Returns stored-format (fp16) group scales
    [R, C/G]. Batched on the GPU: T = q-blocks x H (one matmul per chunk), A = <q_g, T_h>, then a batched ng x ng solve."""
    W, Q, Sx, H = W.float(), Q.float(), Sx.float(), H.float()
    R, C = W.shape
    ng = C // G
    qi = torch.where(Sx.abs() < 1e-20, torch.zeros_like(Q), torch.round(Q / Sx))
    WH = W @ H
    Hb = H.reshape(ng, G, C)
    out = torch.empty(R, ng, device=W.device)
    old = Sx[:, ::G]
    for r0 in range(0, R, chunk):
        r1 = min(R, r0 + chunk)
        k = r1 - r0
        qb = qi[r0:r1].reshape(k, ng, G)
        T = torch.einsum("rhk,hkc->rhc", qb, Hb)                                  # [k, ng, C]
        A = torch.einsum("rgk,rhgk->rgh", qb, T.reshape(k, ng, ng, G))            # [k, ng, ng]
        b = (qb * WH[r0:r1].reshape(k, ng, G)).sum(-1)                            # [k, ng]
        live = qb.abs().sum(-1) > 0                                               # groups with any nonzero code
        diag = torch.diagonal(A, dim1=1, dim2=2)
        diag += torch.where(live, 1e-6 * diag.abs().mean(-1, keepdim=True), torch.ones_like(diag))
        s = torch.linalg.solve(A, b.unsqueeze(-1)).squeeze(-1)
        out[r0:r1] = torch.where(live, s, old[r0:r1])
    out = out.half().float()
    return torch.where(out == 0, torch.full_like(out, 1e-30), out)


_ROT = {}


def rot_mat(C, device):
    """Block-diagonal orthonormal Sylvester Hadamard (128-wide blocks), symmetric and its own inverse: experts are
    stored as W R and the engine feeds them R x (numerics::fwht_blocks / quant_act rot)."""
    k = (C, str(device))
    if k not in _ROT:
        H = torch.ones(1, 1, dtype=torch.float32)   # v41_calib makes bf16 the default dtype
        while H.shape[0] < 128:
            H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
        _ROT[k] = torch.block_diag(*[H / 128 ** 0.5] * (C // 128)).to(device=device, dtype=torch.float32)
    return _ROT[k]


def _gpu_codes(Q, S, tier):
    """GPTQ values Q [R, C] / stored-format scales S [R, C/group] -> int8 codes [R, C] on the GPU (v41_formats._codes
    without the host round trip: Q is exactly codes x scales)."""
    lo, hi, Gs, _ = G.TIERS[tier]
    Se = S.float().repeat_interleave(Gs, dim=1)
    c = torch.where(Se.abs() < 1e-20, torch.zeros_like(Q), Q.float() / Se)
    return torch.clamp(torch.round(c), lo, hi).to(torch.int8)


def _pack_codes(codes, S, tier):
    """v41_formats.encode_tier from codes already computed (same bytes)."""
    if tier == "q2_0":
        return F.pack_q2_0(codes, S)
    if tier == "ptq1":
        return F.pack_ptq1_0(codes, S)
    if tier == "q3_k":
        dsup, sc = F.q3k_split(S)
        return F.pack_q3_k(codes, dsup, sc)
    raise ValueError(tier)


def _hinv_chol_batched(H, damp):
    """v41_gptq.hinv_chol over a batch [B, C, C]: (upper Cholesky of H^-1 [B, C, C], dead columns [B, C])."""
    H = H.clone()
    dg = torch.diagonal(H, dim1=-2, dim2=-1)
    dead = dg == 0
    dg.masked_fill_(dead, 1.0)
    dg += damp * dg.mean(-1, keepdim=True)
    # the upper Cholesky factor of H^-1 without forming H^-1: factor the reversed matrix, J H J = M M^T, so
    # H = N N^T with N = J M J upper-triangular and H^-1 = (N^-1)^T N^-1: the factor is N^-1 (one Cholesky + one
    # triangular inverse instead of three O(n^3) factorizations; 2.2x faster, rel. difference 6e-7)
    N = torch.linalg.cholesky(H.flip(-1, -2)).flip(-1, -2)
    del H
    I = torch.eye(N.shape[-1], device=N.device, dtype=N.dtype).expand_as(N)
    return torch.linalg.solve_triangular(N, I, upper=True), dead


def cmd_init(a):
    for f in ("tokens.npy", "meta.json", "hashes.npy"):
        dst = C.wpath(f)
        if not os.path.exists(dst):
            os.symlink(os.path.join(a.calib, f), dst)
    C.cmd_init(a)


def cmd_layer(a):
    assert NATIVE or not a.rot, "--rot needs the native batched path (V41_NATIVE=1)"
    assert NATIVE or "iq2xxs" not in sum(tiers_for(a, a.layer), []), "the iq2 tier needs the native batched path (V41_NATIVE=1)"
    if os.path.exists(C.wpath("q", f"L{a.layer:02d}.json")) and os.path.exists(C.wpath("stats", f"L{a.layer:02d}.json")):
        C.log(f"quant L{a.layer}: done already")
        return
    if a.fast_scales:   # smoke tests only: one clip ratio
        G.CLIP = [1.0]
    lay = Layer(a, a.layer)
    a.expert_hook, a.moe_done_hook = lay.expert, lay.finish
    if NATIVE:
        a.expert_batch_hook = lay.expert_batch
    a.no_gptq, a.screen_only, a.damps, a.expert_stride = True, None, [a.damp], 1
    C.cmd_layer(a)


def cmd_fill(a):
    """Copy a QDIR's layers into an --experts-in-place GGUF (tests the in-place path; or a QDIR run -> final file)."""
    meta = json.load(open(a.gguf + ".experts.json"))["layers"]
    for Ls in sorted(meta, key=int):
        L = int(Ls)
        j = json.load(open(os.path.join(a.qdir, f"L{L:02d}.json")))
        size = [(F.nbytes(j["gu_type"][e], (C.D, FF)), F.nbytes(j["d_type"][e], (FF, C.D))) for e in range(C.NE)]
        a.no_write = False
        sink = Sink(a, L, j["gu_type"], j["d_type"], size)
        with open(os.path.join(a.qdir, f"L{L:02d}.bin"), "rb") as f:
            for e, (g, d) in enumerate(size):
                b = f.read(2 * g + d)
                sink.put(e, b[:g], b[g:2 * g], b[2 * g:])
        sink.close()
        C.log(f"fill: layer {L} from {a.qdir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fill")
    p.add_argument("--gguf", required=True)
    p.add_argument("--qdir", required=True)
    for name in ("init", "layer", "head"):
        p = sub.add_parser(name)
        p.add_argument("--qwork", required=True)
        p.add_argument("--calib", default=CALIB, help="calibration work dir (tokens/meta/hashes)")
        if name == "layer":
            p.add_argument("layer", type=int)
            g = p.add_mutually_exclusive_group(required=True)
            g.add_argument("--map")
            g.add_argument("--tier", choices=list(G.TIERS))
            p.add_argument("--propagate", choices=("quant", "teacher"), default="quant")
            p.add_argument("--damp", type=float, default=1.0)
            p.add_argument("--refit", action="store_true", help="least-squares group scales against the full H after GPTQ (ptq1 / q2_0)")
            p.add_argument("--rot", action="store_true", help="store experts in the 128-block Hadamard-rotated input basis (engine: dsx.expert_rot)")
            p.add_argument("--min-fit", type=int, default=32)
            p.add_argument("--batch", type=int, default=0)
            p.add_argument("--device", default="cuda")
            p.add_argument("--gc", action="store_true")
            w = p.add_mutually_exclusive_group()
            w.add_argument("--gguf", help="write the experts in place into this v41_gguf --experts-in-place shard 1")
            w.add_argument("--no-write", action="store_true", help="no expert bytes (the all-Q2_0 control: head numbers only)")
            p.add_argument("--gptq-experts", type=int, default=C.NE, help="smoke: GPTQ only experts < K")
            p.add_argument("--fast-scales", action="store_true", help="smoke: one clip ratio")
        if name == "head":
            p.add_argument("--layer", type=int, default=40)
            p.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if a.cmd == "fill":
        cmd_fill(a)
        return
    if a.cmd == "layer" and a.map is not None:
        a.tier = "q2_0"   # fallback for nothing; tiers_for reads the map
    set_work(a.qwork)
    {"init": cmd_init, "layer": cmd_layer, "head": C.cmd_head}[a.cmd](a)


if __name__ == "__main__":
    main()
