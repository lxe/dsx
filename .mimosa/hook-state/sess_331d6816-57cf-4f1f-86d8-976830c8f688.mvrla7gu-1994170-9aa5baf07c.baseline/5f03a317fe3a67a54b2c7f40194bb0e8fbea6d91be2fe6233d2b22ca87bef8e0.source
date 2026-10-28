"""cmpbytes.py: UD-IQ1_M's raw IQ2_XXS gate blocks at layer 32 vs ggml's stock quantize_iq2_xxs with Unsloth's own
imatrix on the same expert weights (our FP4-dequantized reference): fraction of identical fp16 d per 256-block, of
identical 4-bit scales per 32-block, of identical grid indices per 8 weights, and of identical sign patterns."""
import ctypes, glob, sys
import numpy as np, torch
sys.path[:0] = ["/root/iq2test/experiments", "/root/v7"]
import gguf
lib = ctypes.CDLL("/work/ggml/build/bin/libggml-base.so")
lib.ggml_quantize_chunk.restype = ctypes.c_size_t
lib.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
L = 32
r = gguf.GGUFReader(glob.glob("/work/um/imatrix*")[0])
T = {t.name: np.asarray(t.data) for t in r.tensors}
s2, cn = T[f"blk.{L}.ffn_gate_exps.weight.in_sum2"].astype(np.float64), T[f"blk.{L}.ffn_gate_exps.weight.counts"].astype(np.float64).reshape(-1)
imx = (s2.reshape(len(cn), -1) / np.maximum(cn, 1)[:, None]).astype(np.float32)
UT = {t.name: t for f in sorted(glob.glob("/work/um/UD-IQ1_M/*.gguf")) for t in gguf.GGUFReader(f).tensors}
ut = UT[f"blk.{L}.ffn_gate_exps.weight"]
assert ut.tensor_type.name == "IQ2_XXS"
for f in sorted(glob.glob(f"/work/ud{L}/*.pt"))[:4]:
    d = torch.load(f); e = d["ex"]
    W = np.ascontiguousarray(d["W1"].float().numpy())
    R, C = W.shape
    out = np.zeros(R * C // 256 * 66, np.uint8)
    im = np.ascontiguousarray(imx[e])
    lib.ggml_quantize_chunk(16, W.ctypes.data, out.ctypes.data, 0, R, C, im.ctypes.data)
    a = out.reshape(-1, 66); b = np.asarray(ut.data)[e].reshape(-1, 66)
    da, db = a[:, :2].copy().view(np.float16)[:, 0], b[:, :2].copy().view(np.float16)[:, 0]
    qa, qb = a[:, 2:].copy().view("<u4").reshape(-1, 8, 2), b[:, 2:].copy().view("<u4").reshape(-1, 8, 2)
    ga = np.stack([(qa[..., 0] >> (8 * l)) & 255 for l in range(4)], -1); gb = np.stack([(qb[..., 0] >> (8 * l)) & 255 for l in range(4)], -1)
    sa = np.stack([(qa[..., 1] >> (7 * l)) & 127 for l in range(4)], -1); sb = np.stack([(qb[..., 1] >> (7 * l)) & 127 for l in range(4)], -1)
    print(f"L{L} e{e}: identical d {np.mean(da == db):.3f} (median ratio ours/UD {np.median(da.astype(np.float32) / np.maximum(db.astype(np.float32), 1e-30)):.3f}), "
          f"ls {np.mean((qa[..., 1] >> 28) == (qb[..., 1] >> 28)):.3f}, grid index {np.mean(ga == gb):.3f}, signs {np.mean(sa == sb):.3f}", flush=True)
