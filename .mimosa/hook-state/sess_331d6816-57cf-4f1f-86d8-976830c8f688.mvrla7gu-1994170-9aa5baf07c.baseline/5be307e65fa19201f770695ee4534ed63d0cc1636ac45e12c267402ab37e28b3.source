#!/usr/bin/env python3
"""The official checkpoint's Engram tables as one Q6_K GGUF table in llama.cpp's layout: tensor
per_layer_token_embd.weight [256, rows of all Engram layers in layer order], what dsx reads with --engram FILE.gguf.

Usage: engram_gguf.py FP8_DIR OUT.gguf ENGRAM_Q6K_BINARY   (stdlib only; the binary does the quantization)
"""
import json, os, struct, subprocess, sys

ALIGN = 32
Q6_K, Q6_ROW = 14, 210


def gguf_str(s):
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def tensor_loc(fp8, name):
    """(shard path, absolute data offset, shape) of a safetensors tensor"""
    idx = json.load(open(os.path.join(fp8, "model.safetensors.index.json")))["weight_map"]
    path = os.path.join(fp8, idx[name])
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    t = h[name]
    return path, 8 + n + t["data_offsets"][0], t["shape"]


def main():
    fp8, out, tool = sys.argv[1], sys.argv[2], sys.argv[3]
    cfg = json.load(open(os.path.join(fp8, "config.json")))
    cfg = cfg.get("text_config", cfg)
    layers, dim = cfg["engram_layer_ids"], cfg["engram_head_dim"]
    assert dim == 256, dim
    parts = []
    for l in layers:
        wp, wo, ws = tensor_loc(fp8, f"layers.{l}.engram.embed.weight")
        sp, so, ss = tensor_loc(fp8, f"layers.{l}.engram.embed.scale")
        assert wp == sp and ws[1] == dim and ss == [ws[0], dim // 32], (wp, sp, ws, ss)
        parts.append((l, wp, wo, so, ws[0]))
    total = sum(p[4] for p in parts)
    assert total == sum(cfg["engram_num_embeddings"]), (total, cfg["engram_num_embeddings"])
    kv = [("general.name", 8, gguf_str("DeepSeek-V4.1-Flash-1017 Engram tables, Q6_K")),
          ("general.alignment", 4, struct.pack("<I", ALIGN))]
    hdr = b"GGUF" + struct.pack("<IQQ", 3, 1, len(kv))
    for k, ty, v in kv:
        hdr += gguf_str(k) + struct.pack("<I", ty) + v
    hdr += gguf_str("per_layer_token_embd.weight") + struct.pack("<I", 2) + struct.pack("<QQ", dim, total) + struct.pack("<IQ", Q6_K, 0)
    data = (len(hdr) + ALIGN - 1) // ALIGN * ALIGN
    with open(out, "wb") as f:
        f.write(hdr + b"\0" * (data - len(hdr)))
        f.truncate(data + total * Q6_ROW)
    print(f"{out}: {total} rows x {dim}, {(data + total * Q6_ROW) / 1e9:.1f} GB", flush=True)
    row = 0
    for l, wp, wo, so, rows in parts:
        print(f"layer {l}: {rows} rows from {os.path.basename(wp)}", flush=True)
        subprocess.run([tool, wp, str(wo), str(so), str(rows), out, str(data + row * Q6_ROW)], check=True)
        row += rows


if __name__ == "__main__":
    main()
