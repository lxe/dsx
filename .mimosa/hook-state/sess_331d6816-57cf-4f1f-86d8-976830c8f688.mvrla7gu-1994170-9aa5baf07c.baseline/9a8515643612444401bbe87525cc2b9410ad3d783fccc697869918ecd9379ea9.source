"""t16_ref.py: a standalone reference for dsx's T16 expert format (docs/t16-format.md), numpy only.

  python3 t16_ref.py inspect FILE.gguf            metadata and tensor list of a dsx GGUF (gguf-py refuses dsx's types)
  python3 t16_ref.py dequant FILE.gguf TENSOR E   the effective f32 matrix of expert E of a T16 tensor, as .npy
  python3 t16_ref.py selftest GOLDEN_DIR          decode the golden files (core/tests/data/t16) and compare

A T16 matrix W (inputs k, outputs n) computes y = svh * had128(had128(x * suh) @ W_inner), where had128 is the
orthonormal Walsh-Hadamard transform over consecutive blocks of 128 and W_inner [k][n] is a bitshift trellis:
16 x 16 tiles, each one circular bit stream of 256 K bits; position t of a tile is input t % 16 and output t // 16 of
that tile; its 16-bit state is the window of the stream ending after its own bits (MSB first, wrapping); the value is
fp16(fma(1024 + bytesum(low16(state * 0x3A2F)), fp16 0x20e6, fp16 0xca1e)) = 0.009567260742 b - 2.4375, rounded to fp16.
"""
import json, struct, sys
import numpy as np

T16_TYPES = {1010: 3, 1011: 4, 1012: 5}   # dsx GGUF type id -> 2K (twice the bits per weight)
MUL, K_INV, K_BIAS = 0x3A2F, 0x20E6, 0xCA1E

# the 511 possible values by byte sum b: the fma's exact product and sum, rounded once to fp16 (as exllamav3 / dsx)
_kinv = float(np.array([K_INV], dtype=np.uint16).view(np.float16)[0])
_kbias = float(np.array([K_BIAS], dtype=np.uint16).view(np.float16)[0])
VALUES = np.array([(1024.0 + b) * _kinv + _kbias for b in range(511)], dtype=np.float64).astype(np.float16).astype(np.float32)
_s = np.arange(65536, dtype=np.uint32) * MUL & 0xFFFF
STATE_VALUE = VALUES[(_s & 0xFF) + (_s >> 8)]          # value of every 16-bit state


def bit_end(k2, t):
    """end bit (exclusive) of position t; half-integer K: even positions floor(K) bits, odd ones floor(K) + 1"""
    return (t + 1) * (k2 // 2) + ((t + 1) // 2 if k2 % 2 else 0)


def tile_states(tile, k2):
    """the 256 16-bit states of one tile (16 * k2 bytes in stream order, bits MSB first)"""
    nb = 128 * k2
    bits = np.unpackbits(np.frombuffer(tile, dtype=np.uint8))
    t = np.arange(256)
    start = (bit_end(k2, t) + nb - 16) % nb
    idx = (start[:, None] + np.arange(16)[None, :]) % nb
    return (bits[idx].astype(np.uint32) << np.arange(15, -1, -1, dtype=np.uint32)).sum(1)


def decode_inner(trellis, k2, k, n):
    """dsx's stored trellis (tiles by column: tile (kb, nb) at nb * k/16 + kb) -> W_inner f32 [k][n]"""
    tb, kt, nt = 16 * k2, k // 16, n // 16
    assert len(trellis) == kt * nt * tb
    w = np.empty((k, n), dtype=np.float32)
    t = np.arange(256)
    r, c = t % 16, t // 16
    for nb_ in range(nt):
        for kb in range(kt):
            tile = trellis[(nb_ * kt + kb) * tb:][:tb]
            w[16 * kb + r, 16 * nb_ + c] = STATE_VALUE[tile_states(tile, k2)]
    return w


def from_exl3(trellis_i16, k2, k, n):
    """exllamav3's trellis (int16 [k/16][n/16][8 k2], u32 words read MSB first) -> dsx's stored bytes"""
    a = np.ascontiguousarray(trellis_i16).view("<u4").reshape(k // 16, n // 16, 4 * k2)
    return np.ascontiguousarray(a.transpose(1, 0, 2)).byteswap().tobytes()


def had128(x, axis):
    """orthonormal Walsh-Hadamard transform over consecutive blocks of 128 along axis"""
    x = np.moveaxis(np.asarray(x, dtype=np.float64), axis, -1)
    shp = x.shape
    y = x.reshape(-1, 128).copy()
    h = 1
    while h < 128:
        y = y.reshape(-1, 128 // (2 * h), 2, h)
        a, b = y[:, :, 0, :].copy(), y[:, :, 1, :].copy()
        y[:, :, 0, :], y[:, :, 1, :] = a + b, a - b
        h *= 2
    return np.moveaxis((y.reshape(shp) / np.sqrt(128)), -1, axis)


def effective(trellis, suh, svh, k2, k, n):
    """W_eff f32 [n][k] with y = W_eff @ x"""
    m = had128(had128(decode_inner(trellis, k2, k, n), 1), 0)   # H_k W_inner H_n, [k][n]
    return (m.T * svh[:, None] * suh[None, :]).astype(np.float32)


# ------------------------------------------------------------------------------------------------ GGUF (dsx-tolerant)
GT = {0: (1, 4), 1: (1, 2), 8: (32, 34), 14: (256, 210), 20: (32, 18), 24: (1, 1), 26: (1, 4), 30: (1, 2)}   # common types


def read_gguf(path):
    f = open(path, "rb")
    rd = lambda fmt: struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]
    rs = lambda: f.read(rd("Q")).decode("utf-8", "replace")
    assert f.read(4) == b"GGUF", "not a GGUF file"
    ver, nt, nkv = rd("I"), rd("Q"), rd("Q")
    sc = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}

    def val(t):
        if t == 8:
            return rs()
        if t == 9:
            et, cnt = rd("I"), rd("Q")
            return [val(et) for _ in range(cnt)]
        return rd(sc[t])
    meta = {}
    for _ in range(nkv):
        key = rs()
        meta[key] = val(rd("I"))
    tensors = []
    for _ in range(nt):
        name = rs()
        dims = [rd("Q") for _ in range(rd("I"))]
        tensors.append((name, dims, rd("I"), rd("Q")))
    align = meta.get("general.alignment", 32)
    base = (f.tell() + align - 1) // align * align
    return meta, tensors, base


def t16_matrix(path, name, e):
    """(trellis bytes, suh, svh, k2, k, n) of expert e of T16 tensor `name` (stored per expert: trellis | suh | svh)"""
    meta, tensors, base = read_gguf(path)
    (dims, ty, off), = [(d, t, o) for nm, d, t, o in tensors if nm == name]
    k2, (k, n) = T16_TYPES[ty], dims[:2]
    mb = (k // 16) * (n // 16) * 16 * k2 + 2 * k + 2 * n
    with open(path, "rb") as f:
        f.seek(base + off + e * mb)
        b = f.read(mb)
    tb = mb - 2 * k - 2 * n
    f16 = lambda x: np.frombuffer(x, dtype="<f2").astype(np.float32)
    return b[:tb], f16(b[tb:tb + 2 * k]), f16(b[tb + 2 * k:]), k2, k, n


def load_file(path):
    """minimal safetensors reader (numpy only)"""
    dt = {"F32": "<f4", "F16": "<f2", "I16": "<i2", "I32": "<i4", "U8": "u1"}
    raw = open(path, "rb").read()
    hn = struct.unpack("<Q", raw[:8])[0]
    hdr = json.loads(raw[8:8 + hn])
    hdr.pop("__metadata__", None)
    return {k: np.frombuffer(raw[8 + hn + v["data_offsets"][0]:8 + hn + v["data_offsets"][1]], dtype=dt[v["dtype"]]).reshape(v["shape"])
            for k, v in hdr.items()}


def selftest(gdir):
    """decode golden files made by exllamav3 itself: W_inner exact, W_eff within fp16 rounding, y"""
    ok = True
    for K, k2 in ((1.5, 3), (2, 4), (2.5, 5)):
        g = load_file(f"{gdir}/t16_golden_K{K}_cm.safetensors")
        k, n = g["w_inner"].shape                                  # golden weights are (in, out)
        tr = from_exl3(g["trellis"], k2, k, n)
        inner = decode_inner(tr, k2, k, n)
        d_inner = np.abs(inner - g["w_inner"].astype(np.float32)).max()
        w = effective(tr, g["suh"].astype(np.float32), g["svh"].astype(np.float32), k2, k, n)
        ref = g["w_eff"].astype(np.float32).T                     # -> [n][k]
        d_eff = np.abs(w - ref).max() / np.abs(ref).max()
        y = w @ g["x"]
        d_y = np.abs(y - g["y"]).max() / np.abs(g["y"]).max()
        good = d_inner == 0 and d_eff < 2e-3 and d_y < 2e-3
        ok &= good
        print(f"K {K}: W_inner max |diff| {d_inner:.1e} (exact: {d_inner == 0}), W_eff {d_eff:.1e}, y {d_y:.1e}  {'ok' if good else 'FAIL'}")
    return ok


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "inspect":
        meta, tensors, base = read_gguf(sys.argv[2])
        for k_, v in meta.items():
            s = json.dumps(v) if not isinstance(v, list) or len(v) < 8 else f"[{len(v)} values]"
            print(f"{k_} = {s[:120]}")
        for nm, d, t, o in tensors:
            print(f"{nm:48s} {str(d):28s} type {t}{' (T16, K ' + str(T16_TYPES[t] / 2) + ')' if t in T16_TYPES else ''}")
    elif cmd == "dequant":
        tr, suh, svh, k2, k, n = t16_matrix(sys.argv[2], sys.argv[3], int(sys.argv[4]))
        np.save(f"{sys.argv[3]}.e{sys.argv[4]}.npy", effective(tr, suh, svh, k2, k, n))
    elif cmd == "selftest":
        sys.exit(0 if selftest(sys.argv[2]) else 1)
