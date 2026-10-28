"""wspace.py L: weight-space error vs our reference weights (FP4 checkpoint dequantized) for UD-IQ1_M's actual experts and
for stock ggml (same type, Unsloth's imatrix): plain relative ||W_q - W||^2 / ||W||^2, imatrix-weighted, and the
least-squares gain <W_q, W> / <W_q, W_q> per matrix (whether either is scaled relative to W)."""
import ctypes, glob, sys
import numpy as np, torch
import gguf
from gguf import quants
lib = ctypes.CDLL("/work/ggml/build/bin/libggml-base.so")
lib.ggml_quantize_chunk.restype = ctypes.c_size_t
lib.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
lib.ggml_row_size.restype = ctypes.c_size_t
lib.ggml_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
L = int(sys.argv[1])
r = gguf.GGUFReader(glob.glob("/work/um/imatrix*")[0])
T = {t.name: np.asarray(t.data) for t in r.tensors}
def imx(role):
    s2, cn = T[f"blk.{L}.ffn_{role}_exps.weight.in_sum2"].astype(np.float64), T[f"blk.{L}.ffn_{role}_exps.weight.counts"].astype(np.float64).reshape(-1)
    return (s2.reshape(len(cn), -1) / np.maximum(cn, 1)[:, None]).astype(np.float32)
IM = {r_: imx(r_) for r_ in ("gate", "up", "down")}
UT = {t.name: t for f in sorted(glob.glob("/work/um/UD-IQ1_M/*.gguf")) for t in gguf.GGUFReader(f).tensors}
acc = {}
for f in sorted(glob.glob(f"/work/ud{L}/*.pt"))[:12]:
    d = torch.load(f); e = d["ex"]
    for role, key in (("gate", "W1"), ("up", "W3"), ("down", "W2")):
        W = np.ascontiguousarray(d[key].float().numpy()); R, C = W.shape
        ut = UT[f"blk.{L}.ffn_{role}_exps.weight"]; t = int(ut.tensor_type)
        ud = quants.dequantize(np.asarray(ut.data)[e], ut.tensor_type).astype(np.float64)
        out = np.zeros(R * lib.ggml_row_size(t, C), np.uint8); im = np.ascontiguousarray(IM[role][e])
        lib.ggml_quantize_chunk(t, W.ctypes.data, out.ctypes.data, 0, R, C, im.ctypes.data)
        gg = quants.dequantize(out.reshape(R, -1), ut.tensor_type).astype(np.float64)
        W64 = W.astype(np.float64)
        for name, Q in (("UD", ud), ("ggml", gg)):
            a = acc.setdefault((role, name), np.zeros(4))
            a += [np.square(Q - W64).sum() / np.square(W64).sum(), (np.square(Q - W64) * im).sum() / (np.square(W64) * im).sum(),
                  (Q * W64).sum() / (Q * Q).sum(), 1]
for (role, name), a in sorted(acc.items()):
    print(f"L{L} {role:4} {name:4}: rel {a[0] / a[3]:.4f}  imatrix-weighted {a[1] / a[3]:.4f}  LS gain <Wq,W>/<Wq,Wq> {a[2] / a[3]:.3f}")
