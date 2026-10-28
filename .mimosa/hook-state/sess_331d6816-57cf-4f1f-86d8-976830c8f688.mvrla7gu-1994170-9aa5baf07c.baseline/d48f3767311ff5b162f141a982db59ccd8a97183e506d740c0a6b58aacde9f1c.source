"""repro.py L: reproduce Unsloth UD-IQ1_M's routed experts with ggml's own quantizers (ggml_quantize_chunk via ctypes on
libggml-base, imatrix as quant_weights; dequantized by gguf-py), next to UD's actual experts and our per-matrix tier
splits. Same experts and token sets as eval2 (calib held-out | Lucebox eval): token-weighted rel, E_L, shrink.
imatrix (a) ours: diag of the dump's fit Hessian per input column (H1 for gate/up, H2 for down); (b) Unsloth's
imatrix_unsloth.gguf_file (in_sum2 / counts per expert)."""
import ctypes, glob, json, os, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np, torch
sys.path[:0] = ["/root/iq2test/experiments", "/root/iq2test/v41"]
import gguf
from gguf import quants
import quant_variants as QV

L = int(sys.argv[1])
lib = ctypes.CDLL("/work/ggml/build/bin/libggml-base.so")
lib.ggml_quantize_chunk.restype = ctypes.c_size_t
lib.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
lib.ggml_row_size.restype = ctypes.c_size_t
lib.ggml_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
GT = gguf.GGMLQuantizationType
BPW = {"IQ1_S": 1.5625, "IQ1_M": 1.75, "IQ2_XXS": 2.0625, "IQ2_XS": 2.3125, "IQ2_S": 2.5625, "IQ3_XXS": 3.0625}
pool = ThreadPoolExecutor(64)


def ggml_q(W, tname, imat):
    """W [R, C] float32 numpy; imat [C] float32 -> dequantized [R, C] via ggml's quantizer."""
    t = int(GT[tname])
    R, C = W.shape
    W = np.ascontiguousarray(W, np.float32)
    im = np.ascontiguousarray(imat, np.float32)
    rb = lib.ggml_row_size(t, C)
    out = np.zeros(R * rb, np.uint8)
    step = 64
    def run(r0):
        r1 = min(R, r0 + step)
        # base pointers + element offset `start`: ggml reads src + start and writes dst + (start / C) * row size
        lib.ggml_quantize_chunk(t, W.ctypes.data, out.ctypes.data, r0 * C, r1 - r0, C, im.ctypes.data)
    list(pool.map(run, range(0, R, step)))
    return quants.dequantize(out.reshape(R, rb), GT[tname])


# Unsloth's imatrix: per layer tensors <name>.in_sum2 [n_expert, C] and <name>.counts [n_expert]
umx = {}
fn = glob.glob("/work/um/imatrix*")
if fn:
    r = gguf.GGUFReader(fn[0])
    T = {t.name: np.asarray(t.data) for t in r.tensors}
    for role in ("gate", "up", "down"):
        base = f"blk.{L}.ffn_{role}_exps.weight"
        if base + ".in_sum2" in T:
            s2, cn = T[base + ".in_sum2"].astype(np.float64), T[base + ".counts"].astype(np.float64).reshape(-1)
            umx[role] = (s2.reshape(len(cn), -1) / np.maximum(cn, 1)[:, None]).astype(np.float32)
print("unsloth imatrix:", {k: v.shape for k, v in umx.items()}, flush=True)

LAYOUT = {5: ("IQ1_S", "IQ2_XXS"), 20: ("IQ2_XXS", "IQ2_S"), 32: ("IQ2_XXS", "IQ2_XS"), 39: ("IQ2_XS", "IQ2_XS")}
PLUS = {5: ("IQ2_XXS", "IQ2_XXS"), 20: ("IQ2_XS", "IQ3_XXS"), 32: ("IQ2_XS", "IQ2_S"), 39: ("IQ2_S", "IQ2_S")}
OURS = [("ptq1", "q2_0"), ("q2_0", "q3_k"), ("q2_0", "q2_0"), ("ptq1", "ptq1"), ("q2_0", "ptq1")] if L == 20 else []   # per-matrix splits: L20 only (whole-expert tiers are in eval2)
OBPW = {"ptq1": 1.75, "q2_0": 2.25, "q3_k": 3.4375, "iq2n": 2.0625}
readers = [gguf.GGUFReader(f) for f in sorted(glob.glob("/work/um/UD-IQ1_M/*.gguf"))]
UT = {t.name: t for r in readers for t in r.tensors}
acc = {}
for f in sorted(glob.glob(f"/work/ud{L}/*.pt")):
    d = torch.load(f); e = d["ex"]
    fe = f"/work/edump/L{L:02d}_e{e:03d}.pt"
    if not os.path.exists(fe):
        continue
    de = torch.load(fe)
    W1, W3, W2 = (d[k].float().cuda() for k in ("W1", "W3", "W2"))
    H1, H2, lim = d["H1"].cuda(), d["H2"].cuda(), d["lim"]
    sets = {"calib": (d["xv"].cuda(), d["gv"].cuda()), "eval": (de["xv"].cuda(), de["gv"].cuda())}
    Y = {k: QV.expert_f(x, W1, W3, W2, g, lim) for k, (x, g) in sets.items()}
    V = {}
    w1, w3, w2 = (w.cpu().numpy() for w in (W1, W3, W2))
    im = {"ours": (torch.diag(H1).clamp_min(0).cpu().numpy(), torch.diag(H2).clamp_min(0).cpu().numpy())}
    if umx:
        im["unsloth"] = (umx["gate"][e], umx["down"][e])
    for imn, (i1, i2) in im.items():
        for lay, (tg, td) in (("UD layout", LAYOUT[L]), ("+bits", PLUS[L])):
            q = [ggml_q(w1, tg, i1), ggml_q(w3, tg, i1), ggml_q(w2, td, i2)]
            V[(f"ggml {lay} imatrix {imn}", f"{tg}/{td}", (2 * BPW[tg] + BPW[td]) / 3)] = tuple(torch.from_numpy(np.ascontiguousarray(x)).float().cuda() for x in q)
    U = [torch.from_numpy(np.ascontiguousarray(quants.dequantize(np.asarray(UT[f"blk.{L}.ffn_{r}_exps.weight"].data)[e], UT[f"blk.{L}.ffn_{r}_exps.weight"].tensor_type))).float().cuda() for r in ("gate", "up", "down")]
    V[("UD-IQ1_M actual", "/".join(UT[f"blk.{L}.ffn_{r}_exps.weight"].tensor_type.name for r in ("gate", "down")), (2 * BPW[UT[f"blk.{L}.ffn_gate_exps.weight"].tensor_type.name] + BPW[UT[f"blk.{L}.ffn_down_exps.weight"].tensor_type.name]) / 3)] = tuple(U)
    for tg, td in OURS:
        V[("ours rot+refit", f"{tg}/{td}", (2 * OBPW[tg] + OBPW[td]) / 3)] = QV.quant_pair(W1, W3, W2, H1, H2, "had+refit", (tg, td), {}, 1.0)
    for key, (a, b, c) in V.items():
        for k, (x, g) in sets.items():
            Yq = QV.expert_f(x, a, b, c, g, lim); y = Y[k]
            r = acc.setdefault(key + (k,), [0.0, 0, 0.0, 0.0, 0.0, 0])
            ye = float(y.square().sum()); err = float((Yq - y).square().sum())
            r[0] += err / ye * len(x); r[1] += len(x); r[2] += err; r[3] += ye; r[4] += float((Yq * y).sum()) / ye; r[5] += 1
    print(f"L{L} e{e}", flush=True)
print(f"layer {L}: token-weighted rel | E_L | shrink, calib held-out -> Lucebox eval")
for key in sorted({k[:3] for k in acc}, key=lambda k: (k[0], k[2])):
    c, v = acc[key + ("calib",)], acc[key + ("eval",)]
    print(f"  {key[0]:34} {key[1]:16} {key[2]:5.3f} bpw  calib {c[0] / c[1]:.4f} {c[2] / c[3]:.5f} {c[4] / c[5]:.3f}"
          f"  eval {v[0] / v[1]:.4f} {v[2] / v[3]:.5f} {v[4] / v[5]:.3f}  ratio {v[0] / v[1] / (c[0] / c[1]):.2f}")
