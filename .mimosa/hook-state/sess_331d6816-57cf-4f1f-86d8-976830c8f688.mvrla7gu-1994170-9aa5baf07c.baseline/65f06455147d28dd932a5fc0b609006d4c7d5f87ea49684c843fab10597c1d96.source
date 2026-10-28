#!/usr/bin/env python3
"""hybrid_gguf.py OUT BASE_SHARDS... -- EXPERT_SHARDS...
One GGUF: every tensor and key of BASE (dsx build) except its routed experts, plus EXPERT's routed experts
(blk.N.ffn_{gate,up,down}_exps.weight, llama.cpp layout). Drops split.* and dsx.expert_rot (EXPERT's weights are unrotated)."""
import struct, sys, re, os

BLK = {0: (1, 4), 1: (1, 2), 30: (1, 2), 26: (1, 4), 24: (1, 1), 8: (32, 34), 11: (256, 110), 12: (256, 144), 13: (256, 176),
       14: (256, 210), 16: (256, 66), 17: (256, 74), 19: (256, 50), 20: (32, 18), 22: (256, 82), 42: (64, 18), 143: (128, 28)}

class G:
    def __init__(s, path):
        s.path = path; f = s.f = open(path, 'rb')
        assert f.read(4) == b'GGUF'; s.ver, = struct.unpack('<I', f.read(4))
        nt, nkv = struct.unpack('<QQ', f.read(16))
        s.kv = []   # (key, raw bytes of type + value)
        for _ in range(nkv):
            k = s.str(); start = f.tell(); t = s.u32(); s.skip(t); s.kv.append((k, s.blob(start)))
        s.t = []
        for _ in range(nt):
            name = s.str(); nd = s.u32(); ne = list(struct.unpack(f'<{nd}Q', f.read(8 * nd))); ty = s.u32(); off, = struct.unpack('<Q', f.read(8))
            s.t.append((name, ne, ty, off))
        s.align = align = 32
        for k, b in s.kv:
            if k == 'general.alignment': s.align = align = struct.unpack('<I', b[4:8])[0]
        s.data = (f.tell() + align - 1) // align * align
    def u32(s): return struct.unpack('<I', s.f.read(4))[0]
    def str(s): n, = struct.unpack('<Q', s.f.read(8)); return s.f.read(n).decode()
    def blob(s, start):
        end = s.f.tell(); s.f.seek(start); b = s.f.read(end - start); return b
    def skip(s, t):
        f = s.f
        if t in (0, 1, 7): f.read(1)
        elif t in (2, 3): f.read(2)
        elif t in (4, 5, 6): f.read(4)
        elif t in (10, 11, 12): f.read(8)
        elif t == 8: n, = struct.unpack('<Q', f.read(8)); f.read(n)
        elif t == 9:
            et = s.u32(); n, = struct.unpack('<Q', f.read(8))
            for _ in range(n): s.skip(et)
        else: raise ValueError(t)

def nbytes(ne, ty):
    w, b = BLK[ty]; n = 1
    for x in ne: n *= x
    assert ne[0] % w == 0; return n // w * b

def main():
    out = sys.argv[1]; i = sys.argv.index('--'); base = [G(p) for p in sys.argv[2:i]]; exp = [G(p) for p in sys.argv[i + 1:]]
    is_exp = re.compile(r'blk\.\d+\.ffn_(gate|up|down)_exps(\..*)?\.weight$|blk\.\d+\.ffn_exps_(gu|down)_type$')
    tensors = []   # (name, ne, ty, src G, src offset)
    for g in base:
        for name, ne, ty, off in g.t:
            if not is_exp.match(name): tensors.append((name, ne, ty, g, g.data + off))
    nexp = 0
    for g in exp:
        for name, ne, ty, off in g.t:
            if re.match(r'blk\.\d+\.ffn_(gate|up|down)_exps\.weight$', name): tensors.append((name, ne, ty, g, g.data + off)); nexp += 1
    print(f'{len(tensors)} tensors, {nexp} expert tensors from {len(exp)} files', flush=True)
    kv = [(k, b) for k, b in base[0].kv if not k.startswith('split.') and k != 'dsx.expert_rot']
    dropped = [k for k, _ in base[0].kv if k.startswith('dsx.') and k != 'dsx.expert_rot']
    if dropped: print('kept dsx keys:', dropped)
    def s(x): b = x.encode(); return struct.pack('<Q', len(b)) + b
    hdr = bytearray(b'GGUF' + struct.pack('<IQQ', 3, len(tensors), len(kv)))
    for k, b in kv: hdr += s(k) + b
    A = base[0].align; off = 0; offs = []
    for name, ne, ty, g, so in tensors:
        offs.append(off); hdr += s(name) + struct.pack('<I', len(ne)) + struct.pack(f'<{len(ne)}Q', *ne) + struct.pack('<IQ', ty, off)
        off = (off + nbytes(ne, ty) + A - 1) // A * A
    with open(out, 'wb') as f:
        f.write(hdr); f.write(b'\0' * ((-len(hdr)) % A)); d0 = f.tell()
        for (name, ne, ty, g, so), o in zip(tensors, offs):
            n = nbytes(ne, ty); f.seek(d0 + o); g.f.seek(so)
            while n:
                c = g.f.read(min(n, 256 << 20)); f.write(c); n -= len(c)
        f.truncate(d0 + off)
    print('wrote', out, os.path.getsize(out) / 1e9, 'GB')

main()
