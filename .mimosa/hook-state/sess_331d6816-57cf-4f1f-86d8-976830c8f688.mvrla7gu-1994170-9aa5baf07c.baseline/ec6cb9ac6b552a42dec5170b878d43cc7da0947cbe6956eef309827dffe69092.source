# From Xolotl tools/v41 (github.com/Indras-Mirror/Xolotl, commit 7375e305), MIT License,
# Copyright (c) 2026 Niko1221 and the Strata contributors; see LICENSE-Xolotl. Unchanged apart from this header.
"""v41_gguf_meta.py - GGUF key/value records for the V4.1 export (`deepseek41`): tokenizer, Engram hash layout and
the config keys, plus the small GGUF encoding helpers v41_gguf.py uses.

The records are DwarfStar's (antirez/ds4 @fc80bd6, gguf-tools/deepseek41_metadata.py + glm53_quantize.py helpers,
MIT, Copyright (c) 2026 The ds4.c authors): the same key names and values, so the dense half of our file reads like
theirs and their validator / runtime can serve as a second oracle.  The Engram normalization, prime allocation and
seeded multipliers follow DeepSeek's MIT-licensed inference/engram.py, as DwarfStar's do.
"""
import json
import os
import struct

GGUF_ALIGNMENT = 16384   # DwarfStar's: the Engram extent can be unmapped without touching a resident tensor's page
GGUF_UINT16, GGUF_INT32, GGUF_UINT32, GGUF_FLOAT32, GGUF_BOOL, GGUF_STRING, GGUF_ARRAY, GGUF_UINT64 = 2, 5, 4, 6, 7, 8, 9, 10


def pack_string(value):
    data = value.encode("utf-8") if isinstance(value, str) else value
    return struct.pack("<Q", len(data)) + data


def kv_string(key, value):
    return pack_string(key) + struct.pack("<I", GGUF_STRING) + pack_string(value)


def kv_u16(key, value):
    return pack_string(key) + struct.pack("<IH", GGUF_UINT16, value)


def kv_i32(key, value):
    return pack_string(key) + struct.pack("<Ii", GGUF_INT32, value)


def kv_u32(key, value):
    return pack_string(key) + struct.pack("<II", GGUF_UINT32, value)


def kv_f32(key, value):
    return pack_string(key) + struct.pack("<If", GGUF_FLOAT32, value)


def kv_bool(key, value):
    return pack_string(key) + struct.pack("<IB", GGUF_BOOL, bool(value))


def array_record(key, kind, values):
    header = pack_string(key) + struct.pack("<IIQ", GGUF_ARRAY, kind, len(values))
    if kind == GGUF_STRING:
        return header + b"".join(pack_string(v) for v in values)
    fmt = {GGUF_UINT32: "I", GGUF_UINT64: "Q", GGUF_INT32: "i", GGUF_FLOAT32: "f"}[kind]
    return header + struct.pack(f"<{len(values)}{fmt}", *values)


def kv_u32_array(key, values):
    return array_record(key, GGUF_UINT32, list(values))


def compressed_token_map(tokenizer):
    from tokenizers import Regex, normalizers

    sentinel = ""
    normalizer = normalizers.Sequence([
        normalizers.NFKC(), normalizers.NFD(), normalizers.StripAccents(),
        normalizers.Lowercase(), normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
        normalizers.Replace(Regex(r"^ $"), sentinel), normalizers.Strip(),
        normalizers.Replace(sentinel, " "),
    ])
    unique, result = {}, []
    for token_id in range(tokenizer.get_vocab_size(with_added_tokens=True)):
        text = tokenizer.decode([token_id], skip_special_tokens=False)
        key = tokenizer.id_to_token(token_id) if "�" in text else (normalizer.normalize_str(text) or text)
        result.append(unique.setdefault(key, len(unique)))
    return result, len(unique)


def engram_layout(config, tokenizer):
    import numpy as np
    from sympy import nextprime

    token_map, vocab_size = compressed_token_map(tokenizer)
    if vocab_size != config["engram_compressed_vocab_size"]:
        raise ValueError(f"Engram vocabulary mismatch: {vocab_size}")
    if len(token_map) != config["vocab_size"]:
        raise ValueError("Engram requires every released tokenizer ID")
    layers = config["engram_layer_ids"]
    if len(layers) != 2 or config["engram_max_ngram_size"] != 4 or \
            config["engram_n_heads"] != 8 or config["engram_head_dim"] != 256:
        raise ValueError("unsupported Engram shape")
    seen, primes, multipliers = set(), [], []
    bound = (np.iinfo(np.int64).max // vocab_size) // 2
    for layer in layers:
        row = []
        for _ in range(config["engram_max_ngram_size"] - 1):
            current = config["engram_vocab_size"] - 1
            for _ in range(config["engram_n_heads"]):
                current = int(nextprime(current))
                while current in seen:
                    current = int(nextprime(current))
                seen.add(current)
                row.append(current)
        primes.append(row)
        rng = np.random.default_rng(10007 * layer)
        multipliers.append((rng.integers(0, bound, size=4, dtype=np.int64) * 2 + 1).tolist())
    if [sum(row) for row in primes] != config["engram_num_embeddings"]:
        raise ValueError("Engram table size differs from hash bucket layout")
    return dict(token_map=token_map, compressed_vocab_size=vocab_size,
                pad_id=token_map[config["engram_pad_token_id"]],
                rows=config["engram_num_embeddings"], layers=layers,
                primes=primes, multipliers=multipliers)


def metadata(hf_dir, name, source_url):
    """(config, [kv record bytes]) for the HF checkpoint in hf_dir."""
    from tokenizers import Tokenizer

    with open(os.path.join(hf_dir, "config.json"), "rb") as fp:
        config = json.load(fp)
    if config["model_type"] != "deepseek_v41":
        raise ValueError("not a DeepSeek V4.1 source checkpoint")
    text = config["text_config"]
    tokenizer = Tokenizer.from_file(os.path.join(hf_dir, "tokenizer.json"))
    layout = engram_layout(text, tokenizer)
    with open(os.path.join(hf_dir, "tokenizer.json"), "rb") as fp:
        tok = json.load(fp)
    if tok["model"]["type"] != "BPE":
        raise ValueError("unsupported tokenizer")
    # special tokens are in both the BPE vocabulary and added_tokens; the parsed ID table resolves the overlap
    tokens = [tokenizer.id_to_token(i) for i in range(text["vocab_size"])]
    if any(t is None for t in tokens):
        raise ValueError("tokenizer has missing IDs")
    special = {item["id"] for item in tok["added_tokens"] if item["special"]}
    added = {item["id"] for item in tok["added_tokens"]}   # not special (<｜System｜>, <｜User｜>, ...): USER_DEFINED
    merges = [" ".join(pair) if isinstance(pair, list) else pair for pair in tok["model"]["merges"]]
    records = [
        kv_string("general.architecture", "deepseek41"),
        kv_string("general.name", name),
        kv_string("general.source.url", source_url),
        kv_u32("general.alignment", GGUF_ALIGNMENT),
        kv_string("deepseek41.config", json.dumps(config, sort_keys=True, separators=(",", ":"))),
        kv_string("deepseek41.engram.encoding", "e4m3_e8m0_32_row264"),
        kv_u32_array("deepseek41.engram.layer_ids", layout["layers"]),
        kv_u32_array("deepseek41.engram.rows", layout["rows"]),
        kv_u32("deepseek41.engram.compressed_vocab_size", layout["compressed_vocab_size"]),
        kv_u32("deepseek41.engram.pad_id", layout["pad_id"]),
        kv_u32_array("deepseek41.engram.token_map", layout["token_map"]),
        kv_u32_array("deepseek41.engram.primes", sum(layout["primes"], [])),
        array_record("deepseek41.engram.multipliers", GGUF_UINT64, sum(layout["multipliers"], [])),
        kv_string("tokenizer.ggml.model", "gpt2"),
        kv_string("tokenizer.ggml.pre", "joyai-llm"),
        array_record("tokenizer.ggml.tokens", GGUF_STRING, tokens),
        array_record("tokenizer.ggml.token_type", GGUF_INT32, [3 if i in special else 4 if i in added else 1
                                                                  for i in range(len(tokens))]),
        array_record("tokenizer.ggml.merges", GGUF_STRING, merges),
        kv_u32("tokenizer.ggml.bos_token_id", config["bos_token_id"]),
        kv_u32("tokenizer.ggml.eos_token_id", config["eos_token_id"]),
        kv_bool("tokenizer.ggml.add_bos_token", True),
        kv_bool("tokenizer.ggml.add_eos_token", False),
    ]
    for key in (
        "vocab_size", "hidden_size", "moe_intermediate_size", "num_hidden_layers",
        "num_attention_heads", "num_key_value_heads", "head_dim", "qk_rope_head_dim",
        "q_lora_rank", "o_lora_rank", "o_groups", "n_routed_experts", "n_shared_experts",
        "num_experts_per_tok", "max_position_embeddings", "sliding_window",
        "index_n_heads", "index_head_dim", "index_topk", "candidate_source_layer_id",
        "candidate_topk_blocks", "candidate_block_size", "hc_mult", "hc_sinkhorn_iters",
        "rope_theta", "compress_rope_theta",
    ):
        records.append(kv_u32(f"deepseek41.{key}", text[key]))
    for key in ("rms_norm_eps", "hc_eps", "swiglu_limit", "routed_scaling_factor"):
        records.append(kv_f32(f"deepseek41.{key}", text[key]))
    for key in ("scoring_func", "hidden_act", "topk_method"):
        records.append(kv_string(f"deepseek41.{key}", text[key]))
    records.append(kv_bool("deepseek41.norm_topk_prob", text["norm_topk_prob"]))
    records.append(kv_u32_array("deepseek41.compress_ratios", text["compress_ratios"][:text["num_hidden_layers"]]))
    for key in ("kv_source_layer_ids", "index_source_layer_ids"):
        records.append(kv_u32_array(f"deepseek41.{key}", text[key]))
    for key in ("factor", "beta_fast", "beta_slow", "original_max_position_embeddings"):
        records.append(kv_f32(f"deepseek41.rope_scaling.{key}", text["rope_scaling"][key]))
    return config, records
