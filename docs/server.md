# NVIDIA server configuration

Normal model inference requires `--model`; the remaining settings have defaults. `--mock` runs the deterministic
fixture engine without model weights; supply `--tokenizer` for that mode.

**Model files**

| | |
|---|---|
| `--model` | the model GGUF; for a model in shards, the first one (the others load from beside it) |
| `--engram` | directory with the official checkpoint's Engram shards (`model-00047/00048-of-00048.safetensors`), or a GGUF holding the table; default: the table in the model GGUF |
| `--tokenizer` | the official `tokenizer.json`; default: the one stored in the model GGUF |
| `--rank` | expert hotness order (`expert-rank.json` in the quant repo) |
| `--vision` / `--no-vision` | image input (default: `vision.safetensors` beside the model); `--no-vision` for text only |

**Server**

| | |
|---|---|
| `--host`, `--port` | listen address (default `127.0.0.1:8090`) |
| `--api-key` | require this key (`Authorization: Bearer <key>` or `x-api-key`; env `DSX_API_KEY`) |
| `--model-name` | model id reported by `/v1/models` (default `deepseek-v4.1-flash`) |
| `--ctx` | context window, prompt + generated tokens (default 32768) |
| `--max-tokens` | generation cap when a request sets none (default 8192) |

**Sampling defaults** (a request's own fields override them)

| | |
|---|---|
| `--temperature`, `--top-p` | default 1.0 and 1.0, the model's recommended sampling |
| `--hesitation-penalty` | logit penalty on hesitation tokens ("Wait", "But", ...) while reasoning; 0 = off (default) |

**Memory** (dsx places experts on the GPUs automatically; [expert placement](placement.md) lists the overrides)

| | |
|---|---|
| `--arena-gb` | RAM for experts the CPU computes (default 0; ~110 on 128 GB) |
| `--chunk` | prefill tokens per pass (default 4096); smaller leaves more VRAM for experts |
| `--cache-sessions N` | conversation lineages kept for prefix reuse, including the active one (default 1). `1` is the existing single-lineage fast path with no host parking; `>1` parks inactive lineages in the host cache so `A -> B -> A` resumes instead of prefilling from token 0 |
| `--session-cache-gb X` | hard budget for the host cache of inactive lineages, decimal GB (default 8). Admission bounds committed arena bytes and the logical host payload (arena extents, saved logits, Engram histories); exceeding it degrades the hit rate, never correctness. Ignored by `--cache-sessions 1` |
| `--prefix-anchor-tokens N` | sparse prefix-anchor interval in prompt tokens for multi-session mode (default 16384; `0` disables anchors). A second agent sharing a long system/tools prefix replays only the divergent tail from an earlier anchor |
| `DSX_GPUS=0,2` | use only these GPUs (all by default) |

Multi-session cache architecture and state audit: [session-cache.md](session-cache.md). Runtime validation was
performed locally; benchmark numbers are not part of this change.

**Speed**

| | |
|---|---|
| `--defer-rank R` | add the CPU's lower-ranked experts (router rank R or lower, 0 = best of 6) one layer later instead of waiting for them. T16 decode eval: KL ~0.208 → 0.237 at R=3; R=2/1 raised it to 0.260/0.306 and were rejected. Off by default; the quick start opts into R=3. Speed depends on build and text; see [measurement status](current-state.md) |

Settings by machine (the flags are explained in [expert placement](placement.md)):

| Machine | Flags |
|---|---|
| 2x 24 GB GPU, 128 GB RAM | `--arena-gb 110` |
| 2x 24 GB GPU, both PCIe x16, 150+ GB RAM | `--arena-gb 135` |
| 1x 32 GB GPU, 128 GB RAM | `--main-gb=-2.0 --arena-gb 80`, `DSX_ADAPT=0` |

## Thinking

Thinking is on by default, at high effort. A request turns it off or sets the effort:

| | OpenAI API (`/v1/chat/completions`) | Anthropic API (`/v1/messages`) |
|---|---|---|
| Thinking off | `"reasoning_effort": "none"` or `"thinking": {"type": "disabled"}` | `"thinking": {"type": "disabled"}` |
| Effort | `"reasoning_effort"`: `low` (or `minimal`), `high` (or `medium`), `xhigh`, `max` | `"output_config": {"effort": ...}`: `low`, `high` (or `medium`), `xhigh`, `max` |

