#!/usr/bin/env python3
"""overlay_gguf.py OUT BASE.gguf REPO LAYERS TENSORS [--name NAME] [--check UPSTREAM]
A dsx overlay: a GGUF of the dense matrices a fine-tune or weight edit of DeepSeek-V4.1-Flash changed, quantized the
way BASE stores them (Q8_0), for `dsx-server --overlay OUT` over BASE (core/src/gguf.rs Gguf::overlay).

REPO is a Hugging Face model id in DeepSeek's FP8 layout (FP8 E4M3 weights, E8M0 scale per 128x128 block). Only the
named tensors are read, by HTTP range requests, so an edit of a few matrices costs its own size to fetch, not the
checkpoint's. LAYERS is a range like 10-36; TENSORS is a comma list of HF names below `layers.L.` (attn.wo_b,
ffn.shared_experts.w2, ...). Find the changed ones first by comparing the two repos' tensors, e.g. abliteration
edits only the matrices that write the residual stream.

--check UPSTREAM quantizes the same tensors from UPSTREAM (the checkpoint BASE was made from) and requires them to
match BASE byte for byte: proof that this quantizer is the one BASE used, so the overlay differs from BASE only
where REPO does. HF_TOKEN or ~/.cache/huggingface/token is sent when present (gated repos)."""
import json, os, re, struct, sys, time, urllib.request
import numpy as np

HF_TO_GGUF = {'attn.wq_a': 'attn_q_a', 'attn.wq_b': 'attn_q_b', 'attn.wkv': 'attn_kv', 'attn.wo_a': 'attn_output_a',
              'attn.wo_b': 'attn_output_b', 'ffn.shared_experts.w1': 'ffn_gate_shexp',
              'ffn.shared_experts.w3': 'ffn_up_shexp', 'ffn.shared_experts.w2': 'ffn_down_shexp'}
Q8_0 = 8
TOKEN = os.environ.get('HF_TOKEN') or next((open(p).read().strip() for p in [os.path.expanduser('~/.cache/huggingface/token')] if os.path.exists(p)), None)


def fetch(repo, f, a, b, tries=6):
    h = {'Range': f'bytes={a}-{b}'}
    if TOKEN: h['Authorization'] = 'Bearer ' + TOKEN
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(f'https://huggingface.co/{repo}/resolve/main/{f}', headers=h), timeout=300) as r:
                d = r.read()
            if len(d) == b - a + 1: return d
        except Exception as e:
            if i == tries - 1: raise
        time.sleep(2 + 3 * i)
    raise IOError(f'{repo}/{f}: short read')


class Repo:
    def __init__(s, repo):
        s.repo = repo
        h = {'Authorization': 'Bearer ' + TOKEN} if TOKEN else {}
        with urllib.request.urlopen(urllib.request.Request(f'https://huggingface.co/{repo}/resolve/main/model.safetensors.index.json', headers=h), timeout=120) as r:
            s.map = json.loads(r.read())['weight_map']
        s.hdr = {}

    def tensor(s, name):
        f = s.map[name]
        if f not in s.hdr:
            n = struct.unpack('<Q', fetch(s.repo, f, 0, 7))[0]
            s.hdr[f] = (8 + n, json.loads(fetch(s.repo, f, 8, 8 + n - 1)))
        base, h = s.hdr[f]
        t = h[name]
        a, b = t['data_offsets']
        return t['dtype'], t['shape'], fetch(s.repo, f, base + a, base + b - 1)


def e4m3_table():
    v = np.zeros(256, np.float32)
    for b in range(256):
        sg, e, m = b >> 7, (b >> 3) & 15, b & 7
        x = np.float32(m / 8 * 2.0 ** -6) if e == 0 else (np.float32('nan') if (e == 15 and m == 7) else np.float32((1 + m / 8) * 2.0 ** (e - 7)))
        v[b] = -x if sg else x
    return v


E4M3 = e4m3_table()


def dequant(r, name):
    """FP8 E4M3 [out, in] x E8M0 scale per 128 x 128 block -> f32 [out, in]"""
    dt, sh, w = r.tensor(name + '.weight')
    sdt, ssh, sc = r.tensor(name + '.scale')
    assert dt == 'F8_E4M3' and sdt == 'F8_E8M0', (name, dt, sdt)
    out, inn = sh
    x = E4M3[np.frombuffer(w, np.uint8)].reshape(out, inn)
    s = np.ldexp(np.float32(1), np.frombuffer(sc, np.uint8).astype(np.int32) - 127).astype(np.float32).reshape(ssh)
    bo, bi = -(-out // ssh[0]), -(-inn // ssh[1])
    return x * np.repeat(np.repeat(s, bo, 0), bi, 1)[:out, :inn]


def q8_0(x):
    """ggml quantize_row_q8_0_ref: per 32 weights d = amax / 127 (stored fp16), q = round(x / d), ties away from zero"""
    b = x.reshape(-1, 32).astype(np.float32)
    d = np.abs(b).max(1) / np.float32(127)
    idv = np.where(d > 0, np.float32(1) / np.where(d > 0, d, 1), 0).astype(np.float32)
    v = b * idv[:, None]
    q = (np.sign(v) * np.floor(np.abs(v) + np.float32(0.5))).astype(np.int8)
    blk = np.empty(len(b), dtype=[('d', '<f2'), ('q', 'i1', 32)])
    blk['d'], blk['q'] = d.astype(np.float16), q
    return blk.tobytes()


def base_tensor(path, name):
    """bytes of one tensor of a single-file GGUF"""
    with open(path, 'rb') as f:
        hdr = f.read(64 << 20)
    p = 8
    def u(fmt):
        nonlocal p; v = struct.unpack_from(fmt, hdr, p); p += struct.calcsize(fmt); return v[0]
    def st():
        nonlocal p; n = u('<Q'); v = hdr[p:p + n]; p += n; return v.decode()
    def skip(t):
        nonlocal p
        if t == 8: st()
        elif t == 9:
            et, n = u('<I'), u('<Q')
            for _ in range(n): skip(et)
        else: p += {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}[t]
    nt, nk = u('<Q'), u('<Q')
    kv, align = {}, 32
    for _ in range(nk):
        k, t = st(), u('<I')
        if t == 8: kv[k] = st()
        elif k == 'general.alignment': align = u('<I')
        else: skip(t)
    want = None
    for _ in range(nt):
        n = st(); nd = u('<I'); dims = [u('<Q') for _ in range(nd)]; ty = u('<I'); off = u('<Q')
        if n == name: want = (dims, ty, off)
    data = -(-p // align) * align
    if name is None: return kv
    dims, ty, off = want
    assert ty == Q8_0, (name, ty)
    nb = dims[0] // 32 * 34 * int(np.prod(dims[1:]))
    with open(path, 'rb') as f:
        f.seek(data + off)
        return dims, f.read(nb)


def main():
    a = sys.argv[1:]
    opt = lambda k: (a.pop(a.index(k) + 1), a.remove(k))[0] if k in a else None
    name, check = opt('--name'), opt('--check')
    out, base, repo, layers, tens = a
    lo, hi = map(int, layers.split('-'))
    hf = tens.split(',')
    bname = base_tensor(base, None).get('general.name')
    if check:
        up = Repo(check)
        for t in hf:
            L = lo
            dims, want = base_tensor(base, f'blk.{L}.{HF_TO_GGUF[t]}.weight')
            got = q8_0(dequant(up, f'layers.{L}.{t}'))
            assert got == want, f'{check} layers.{L}.{t}: this quantizer does not reproduce {base}'
            print(f'check: {check} layers.{L}.{t} -> Q8_0 matches the base byte for byte', flush=True)
    r = Repo(repo)
    tensors = []
    for L in range(lo, hi + 1):
        for t in hf:
            x = dequant(r, f'layers.{L}.{t}')
            gname = f'blk.{L}.{HF_TO_GGUF[t]}.weight'
            dims, old = base_tensor(base, gname)
            assert dims == [x.shape[1], x.shape[0]], (gname, dims, x.shape)
            q = q8_0(x)
            tensors.append((gname, dims, q))
            print(f'{gname}: {x.shape[0]} x {x.shape[1]}, {sum(1 for i in range(0, len(q), 34) if q[i:i + 34] != old[i:i + 34])} of {len(q) // 34} blocks differ from the base', flush=True)
    kv = [('general.architecture', 'deepseek41'), ('general.type', 'adapter'), ('general.name', name or f'{repo.split("/")[-1]} overlay'),
          ('dsx.overlay.source', f'https://huggingface.co/{repo}')]
    if bname: kv.append(('dsx.overlay.base', bname))
    s = lambda x: struct.pack('<Q', len(x.encode())) + x.encode()
    hdr = bytearray(b'GGUF' + struct.pack('<IQQ', 3, len(tensors), len(kv)))
    for k, v in kv: hdr += s(k) + struct.pack('<I', 8) + s(v)
    off, offs = 0, []
    for n, dims, q in tensors:
        offs.append(off)
        hdr += s(n) + struct.pack('<I', len(dims)) + struct.pack(f'<{len(dims)}Q', *dims) + struct.pack('<IQ', Q8_0, off)
        off = -(-(off + len(q)) // 32) * 32
    with open(out, 'wb') as f:
        f.write(hdr + b'\0' * (-len(hdr) % 32))
        d0 = f.tell()
        for (n, dims, q), o in zip(tensors, offs):
            f.seek(d0 + o); f.write(q)
        f.truncate(d0 + off)
    print('wrote', out, f'{os.path.getsize(out) / 1e9:.2f} GB, {len(tensors)} tensors over {bname}')


if __name__ == '__main__':
    main()
