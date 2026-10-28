#!/usr/bin/env python3
"""t16_gguf.py [--no-engram] [--gain ROLE:K=G,...] OUT PACKDIR BASE_SHARDS...
One GGUF for dsx with T16 routed experts: every tensor and key of BASE (a dsx build, e.g. dsx-ud's 5 shards: dense
weights, Engram table, tokenizer) except its routed experts, plus the T16 experts that t16_worker.py wrote with
T16_PACKDIR (PACKDIR/LNN/eNNN.safetensors, keys blk.L.ffn_{gate,up,down}_exps.eNNN.{trellis,suh,svh,mul1}), or the
full-run worker (same files, keys L{LL}.e{EEE}.{gate,up,down}.{trellis,suh,svh,mul1}).

Layers that PACKDIR does not have keep BASE's experts, so one T16 layer can be tested inside a working build.
Each T16 role of a layer becomes one tensor blk.L.ffn_{role}_exps.weight of type T16K15 / T16K2 / T16K25 (dsx ids
1010-1012, by the trellis' 16 K u16 per tile), dims [k, n, experts], each expert's matrix stored as
trellis | suh (f16 [k]) | svh (f16 [n]) (core/src/t16.rs). The trellis is converted from exllamav3's order
(u16 [k/16][n/16][16 K], u32 words read MSB first) to dsx's stored layout: tiles by column (tile (kb, nb) at
nb * k/16 + kb), each tile's bytes in bit-stream order (every u32 word byte-swapped); t16::from_exl3 is the same.
Drops split.* and dsx.expert_rot (T16 carries its own Hadamards). Split the result with `dsx split`. --no-engram leaves
out the Engram table (per_layer_token_embd.weight) for test builds; run them with --engram BASE's shard that has it.

--gain multiplies svh (each output's scale) of the matching matrices, e.g. --gain gu:1.5=1.13,down:2=1.02 (ROLE gu =
gate and up, or down; K = bits per weight). exllamav3's Viterbi reconstruction is MSE-optimal and shrinks weights by
about 1 - (relative MSE) (0.865x at K 1.5, 0.929x at K 2), and that bias in the expert outputs costs far more KL than
the error itself; the gain undoes it at no cost (same bytes, same kernels)."""
import json, os, re, struct, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import importlib.util
spec = importlib.util.spec_from_file_location('hyb', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'hybrid_gguf.py'))
src = open(spec.origin).read().replace('\nmain()\n', '\n')   # reuse the reader without running its main
hyb = {}
exec(compile(src, spec.origin, 'exec'), hyb)
G, nbytes = hyb['G'], hyb['nbytes']

T16_TY = {24: 1010, 32: 1011, 40: 1012}   # u16 per tile -> dsx type id


def from_exl3(b, k, n, tw):
    """exllamav3 trellis bytes -> dsx stored layout (core/src/t16.rs from_exl3)"""
    a = np.frombuffer(b, dtype='<u4').reshape(k // 16, n // 16, tw // 2)
    return np.ascontiguousarray(a.transpose(1, 0, 2)).byteswap().tobytes()
MUL = 0x3A2F
ROLES = ('gate', 'up', 'down')


def key(st, L, e, role, part):
    a = f'blk.{L}.ffn_{role}_exps.e{e:03d}.{part}'
    return a if a in st.h else f'L{L:02d}.e{e:03d}.{role}.{part}'


class ST:
    """a safetensors file: header only, tensors read on demand"""
    def __init__(s, path):
        s.path = path
        with open(path, 'rb') as f:
            n, = struct.unpack('<Q', f.read(8))
            s.h = json.loads(f.read(n))
        s.base = 8 + n
    def info(s, k):
        t = s.h[k]
        return t['dtype'], t['shape'], s.base + t['data_offsets'][0], t['data_offsets'][1] - t['data_offsets'][0]
    def read(s, k):
        _, _, o, n = s.info(k)
        with open(s.path, 'rb') as f:
            f.seek(o)
            return f.read(n)


def expert_files(packdir):
    layers = {}
    for d in sorted(os.listdir(packdir)):
        m = re.fullmatch(r'L(\d+)', d)
        if not m: continue
        L = int(m.group(1))
        files = {}
        for f in os.listdir(os.path.join(packdir, d)):
            m2 = re.fullmatch(r'e(\d+)\.safetensors', f)
            if m2: files[int(m2.group(1))] = os.path.join(packdir, d, f)
        layers[L] = files
    return layers


def main():
    args = sys.argv[1:]
    no_engram = '--no-engram' in args
    args = [a for a in args if a != '--no-engram']
    gains = {}   # (role, tw) -> gain
    if '--gain' in args:
        i = args.index('--gain')
        for item in args[i + 1].split(','):
            rk, g = item.split('=')
            role, k = rk.split(':')
            tw = {'1.5': 24, '2': 32, '2.5': 40}[k]
            for r in (('gate', 'up') if role == 'gu' else ('down',)):
                gains[(r, tw)] = float(g)
        del args[i:i + 2]
        print('svh gains:', {f'{r} K{tw / 16}': g for (r, tw), g in gains.items()})
    out, packdir = args[0], args[1]
    base = [G(p) for p in args[2:]]
    layers = expert_files(packdir)
    is_exp = re.compile(r'blk\.(\d+)\.ffn_(gate|up|down)_exps(\..*)?\.weight$|blk\.(\d+)\.ffn_exps_(gu|down)_type$')
    tensors = []   # (name, ne, ty, source): source = (G, offset) or ('t16', L, role, files)
    nexp_base = 0
    for g in base:
        for name, ne, ty, off in g.t:
            if no_engram and name == 'per_layer_token_embd.weight':
                continue
            mm = is_exp.match(name)
            if mm and int(mm.group(1) or mm.group(4)) in layers:
                nexp_base += 1
                continue
            tensors.append((name, ne, ty, (g, g.data + off)))
    n_experts = None
    for L in sorted(layers):
        files = layers[L]
        if n_experts is None: n_experts = len(files)
        assert sorted(files) == list(range(n_experts)), f'layer {L}: experts {len(files)} of {n_experts}'
        st0 = ST(files[0])
        for role in ROLES:
            _, tshape, _, _ = st0.info(key(st0, L, 0, role, 'trellis'))
            k, n, tw = tshape[0] * 16, tshape[1] * 16, tshape[2]
            ty = T16_TY[tw]
            tensors.append((f'blk.{L}.ffn_{role}_exps.weight', [k, n, n_experts], ty, ('t16', L, role, files, k, n, tw)))
    print(f'{len(tensors)} tensors: {nexp_base} base expert tensors dropped, {3 * len(layers)} T16 tensors from '
          f'{len(layers)} layers x {n_experts} experts', flush=True)

    def tbytes(ne, ty):
        if ty in (1010, 1011, 1012):
            k, n, e = ne
            tw = {1010: 24, 1011: 32, 1012: 40}[ty]
            return e * ((k // 16) * (n // 16) * tw * 2 + 2 * k + 2 * n)
        return nbytes(ne, ty)

    kv = [(k, b) for k, b in base[0].kv if not k.startswith('split.') and k != 'dsx.expert_rot']
    def s(x): b = x.encode(); return struct.pack('<Q', len(b)) + b
    hdr = bytearray(b'GGUF' + struct.pack('<IQQ', 3, len(tensors), len(kv)))
    for k, b in kv: hdr += s(k) + b
    A = base[0].align
    off = 0
    offs = []
    for name, ne, ty, _ in tensors:
        offs.append(off)
        hdr += s(name) + struct.pack('<I', len(ne)) + struct.pack(f'<{len(ne)}Q', *ne) + struct.pack('<IQ', ty, off)
        off = (off + tbytes(ne, ty) + A - 1) // A * A
    with open(out, 'wb') as f:
        f.write(hdr)
        f.write(b'\0' * ((-len(hdr)) % A))
        d0 = f.tell()
        for (name, ne, ty, srcd), o in zip(tensors, offs):
            f.seek(d0 + o)
            if srcd[0] == 't16':
                _, L, role, files, k, n, tw = srcd
                for e in range(len(files)):
                    st = ST(files[e])
                    kk = lambda part: key(st, L, e, role, part)
                    dt, sh, _, _ = st.info(kk('trellis'))
                    assert dt in ('I16', 'U16') and sh == [k // 16, n // 16, tw], (kk('trellis'), dt, sh)
                    mul = st.read(kk('mul1'))
                    assert int.from_bytes(mul[:4], 'little') == MUL, f'{kk("mul1")} is not the T16 codebook'
                    f.write(from_exl3(st.read(kk('trellis')), k, n, tw))
                    for part, want in (('suh', [k]), ('svh', [n])):
                        dt, sh, _, nb = st.info(kk(part))
                        assert dt == 'F16' and sh == want, (kk(part), dt, sh)
                        b = st.read(kk(part))
                        g = gains.get((role, tw), 1.0)
                        if part == 'svh' and g != 1.0:
                            b = (np.frombuffer(b, dtype='<f2').astype(np.float32) * g).astype('<f2').tobytes()
                        f.write(b)
            else:
                g, so = srcd
                nb = tbytes(ne, ty)
                g.f.seek(so)
                while nb:
                    c = g.f.read(min(nb, 256 << 20))
                    f.write(c)
                    nb -= len(c)
        f.truncate(d0 + off)
    print('wrote', out, os.path.getsize(out) / 1e9, 'GB')


if __name__ == '__main__':
    main()
