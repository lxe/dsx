"""ud_fetch.py OUTDIR: Unsloth UD-IQ1_M's routed-expert tensors of layers 5 and 20 (gate, up, down), by HTTP range from
Hugging Face (6 tensors, a few GB, instead of the 140 GB of shards). Writes NAME.bin (raw ggml blocks, [n_expert][rows]
rows) and meta.json {name: [type, dims]}."""
import json, os, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ggufhdr import read_header
import gguf
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
B = "https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF/resolve/main/UD-IQ1_M/DeepSeek-V4.1-Flash-1017-UD-IQ1_M-{i:05d}-of-00005.gguf"
want = {f"blk.{L}.ffn_{r}_exps.weight" for L in (5, 20) for r in ("gate", "up", "down")}
meta = {}
for i in range(1, 5):
    url = B.format(i=i)
    h = f"{out}/h{i}.bin"
    subprocess.run(["curl", "-sfL", "-r", "0-16777215", "-o", h, url], check=True)
    kv, ti, align, data0, _ = read_header(h)
    for name, dims, ty, off in ti:
        if name not in want:
            continue
        bs, ts = gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType(ty)]
        n = 1
        for d in dims: n *= d
        nbytes = n // bs * ts
        a = data0 + off
        subprocess.run(["curl", "-sfL", "-r", f"{a}-{a + nbytes - 1}", "-o", f"{out}/{name}.bin", url], check=True)
        assert os.path.getsize(f"{out}/{name}.bin") == nbytes, name
        meta[name] = [gguf.GGMLQuantizationType(ty).name, dims]
        print(name, meta[name], nbytes, flush=True)
assert set(meta) == want, set(want) - set(meta)
json.dump(meta, open(f"{out}/meta.json", "w"))
