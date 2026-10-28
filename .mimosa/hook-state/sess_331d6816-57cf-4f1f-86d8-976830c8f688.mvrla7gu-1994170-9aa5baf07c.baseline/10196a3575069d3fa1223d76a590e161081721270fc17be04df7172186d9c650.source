"""maya2llama.py SRC_SHARD1 OUT_PREFIX [--header-only] [--delete]: Project Maya's Maya-S GGUF (6 shards) -> the deepseek41 layout of
llama.cpp PR #30817 (5 shards), without touching any weight.

Maya stores DeepSeek's two Engram tables as blk.1.engram_embd / blk.14.engram_embd (shards 5, 6) with engram.* keys;
the PR reads one table, per_layer_token_embd (layer 1's rows, then layer 14's), and ple.* keys:
  ple.layers = engram.layers, ple.ngram_size = engram.max_ngram_size, ple.heads_per_ngram = engram.head_count,
  ple.layer_multipliers = engram.multipliers, ple.head_vocab_sizes = engram.primes,
  ple.head_offsets = engram.offsets with each later layer's shifted by the rows before it (offsets into one table),
  embedding_length_per_layer_input = engram.head_dim, hash_layer_count = 0.
Shards 1-4 keep their tensors and data bytes (headers rewritten: split.count 5, the new tensor count, ple.* in shard 1);
shard 5 is the joined table (Q6_K rows are independent, so joining is a byte copy).
"""
import os, struct, sys

GGUF_ALIGN = 32
U8, I8, U16, I16, U32, I32, F32, BOOL, STR, ARR, U64, I64, F64 = range(13)
FMT = {U8: "B", I8: "b", U16: "H", I16: "h", U32: "I", I32: "i", F32: "f", BOOL: "?", U64: "Q", I64: "q", F64: "d"}


class R:
    def __init__(self, path):
        self.f = open(path, "rb")

    def take(self, n):
        b = self.f.read(n)
        assert len(b) == n, "short read"
        return b

    def u32(self): return struct.unpack("<I", self.take(4))[0]
    def u64(self): return struct.unpack("<Q", self.take(8))[0]
    def s(self): return self.take(self.u64()).decode()

    def val(self, t):
        if t == STR:
            return self.s()
        if t == ARR:
            et, n = self.u32(), self.u64()
            return (et, [self.val(et) for _ in range(n)])
        return struct.unpack("<" + FMT[t], self.take(struct.calcsize(FMT[t])))[0]


def read_header(path):
    r = R(path)
    assert r.take(4) == b"GGUF" and r.u32() == 3
    nt, nkv = r.u64(), r.u64()
    kv = []  # (key, type, value) in file order
    for _ in range(nkv):
        k = r.s()
        t = r.u32()
        kv.append([k, t, r.val(t)])
    ti = []  # (name, dims, type, offset)
    for _ in range(nt):
        name = r.s()
        nd = r.u32()
        dims = [r.u64() for _ in range(nd)]
        ti.append([name, dims, r.u32(), r.u64()])
    align = next((v for k, t, v in kv if k == "general.alignment"), GGUF_ALIGN)
    pos = r.f.tell()
    data0 = (pos + align - 1) // align * align
    return kv, ti, align, data0, os.path.getsize(path)


def w_s(b, s):
    e = s.encode()
    b += struct.pack("<Q", len(e)) + e


def w_val(b, t, v):
    if t == STR:
        w_s(b, v)
    elif t == ARR:
        et, items = v
        b += struct.pack("<IQ", et, len(items))
        for x in items:
            w_val(b, et, x)
    else:
        b += struct.pack("<" + FMT[t], v)


def header_bytes(kv, ti, align):
    b = bytearray(b"GGUF") + struct.pack("<IQQ", 3, len(ti), len(kv))
    for k, t, v in kv:
        w_s(b, k)
        b += struct.pack("<I", t)
        w_val(b, t, v)
    for name, dims, ty, off in ti:
        w_s(b, name)
        b += struct.pack("<I", len(dims)) + struct.pack("<" + "Q" * len(dims), *dims) + struct.pack("<IQ", ty, off)
    b += b"\0" * ((-len(b)) % align)
    return bytes(b)


def setkv(kv, key, t, v):
    for e in kv:
        if e[0] == key:
            e[1], e[2] = t, v
            return
    kv.append([key, t, v])


def copy_range(src, off, n, dst):
    with open(src, "rb") as f:
        f.seek(off)
        left = n
        while left:
            c = f.read(min(left, 256 << 20))
            assert c, "short source"
            dst.write(c)
            left -= len(c)


def main():
    src1, out = sys.argv[1], sys.argv[2]
    header_only = "--header-only" in sys.argv
    delete = "--delete" in sys.argv  # remove each source shard once copied (disk)
    stem = src1.rsplit("-00001-of-00006.gguf", 1)[0]
    src = [f"{stem}-{i:05d}-of-00006.gguf" for i in range(1, 7)]
    hd = [read_header(p) for p in src]
    kv1 = {k: (t, v) for k, t, v in hd[0][0]}
    layers = kv1["deepseek41.engram.layers"][1][1]
    nemb = kv1["deepseek41.engram.num_embeddings"][1][1]
    offs = kv1["deepseek41.engram.offsets"][1][1]
    heads = len(offs) // len(layers)
    # the joined table: layer li's rows start after every earlier layer's
    base = [sum(nemb[:i]) for i in range(len(layers))]
    gofs = [offs[i] + base[i // heads] for i in range(len(offs))]
    # the engram tables (shards 5, 6) in layer order
    tabs = []
    for si in (4, 5):
        kv, ti, align, data0, size = hd[si]
        assert len(ti) == 1 and ti[0][0].endswith(".engram_embd.weight"), ti
        name, dims, ty, off = ti[0]
        tabs.append((int(name.split(".")[1]), src[si], data0 + off, dims, ty))
    tabs.sort()
    assert [t[0] for t in tabs] == layers, (tabs, layers)
    assert all(t[3][0] == tabs[0][3][0] and t[4] == tabs[0][4] for t in tabs)
    assert [t[3][1] for t in tabs] == nemb, ([t[3][1] for t in tabs], nemb)
    ncols, ty = tabs[0][3][0], tabs[0][4]
    assert ty == 14, f"engram type {ty} (expected Q6_K)"
    row_bytes = ncols // 256 * 210
    n_total = sum(len(h[1]) for h in hd[:4]) + 1
    for si in range(5):
        if si < 4:
            kv, ti, align, data0, size = hd[si]
            kv = [list(e) for e in kv]
        else:
            kv = [list(e) for e in hd[4][0]]
            align = hd[4][2]
            ti = [["per_layer_token_embd.weight", [ncols, sum(nemb)], ty, 0]]
        setkv(kv, "split.no", U16, si)
        setkv(kv, "split.count", U16, 5)
        setkv(kv, "split.tensors.count", I32, n_total)
        if si == 0:
            setkv(kv, "deepseek41.ple.layers", ARR, (U32, list(layers)))
            setkv(kv, "deepseek41.ple.ngram_size", U32, kv1["deepseek41.engram.max_ngram_size"][1])
            setkv(kv, "deepseek41.ple.heads_per_ngram", U32, kv1["deepseek41.engram.head_count"][1])
            setkv(kv, "deepseek41.embedding_length_per_layer_input", U32, kv1["deepseek41.engram.head_dim"][1])
            setkv(kv, "deepseek41.ple.layer_multipliers", ARR, (U64, list(kv1["deepseek41.engram.multipliers"][1][1])))
            setkv(kv, "deepseek41.ple.head_offsets", ARR, (U64, gofs))
            setkv(kv, "deepseek41.ple.head_vocab_sizes", ARR, (U64, list(kv1["deepseek41.engram.primes"][1][1])))
            setkv(kv, "deepseek41.hash_layer_count", U32, 0)
        hb = header_bytes(kv, ti, align)
        path = f"{out}-{si + 1:05d}-of-00005.gguf"
        print(f"{path}: {len(ti)} tensors, header {len(hb)} B", flush=True)
        if header_only:
            continue
        with open(path, "wb") as dst:
            dst.write(hb)
            if si < 4:
                copy_range(src[si], data0, size - data0, dst)
                if delete:
                    os.remove(src[si])
            else:
                for li, p, off, dims, _ in tabs:
                    copy_range(p, off, dims[1] * row_bytes, dst)
                    print(f"  layer {li}: {dims[1]} rows", flush=True)
                    if delete:
                        os.remove(p)


if __name__ == "__main__":
    main()
