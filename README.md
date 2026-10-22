# dsx

[![Model: dsx-2bit](https://img.shields.io/badge/%F0%9F%A4%97%20model-dsx--2bit-yellow)](https://huggingface.co/lxe/DeepSeek-V4.1-Flash-1017-dsx-2bit)
[![Model: dsx-ud](https://img.shields.io/badge/%F0%9F%A4%97%20model-dsx--ud-yellow)](https://huggingface.co/lxe/DeepSeek-V4.1-Flash-1017-dsx-ud)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-stable-orange?logo=rust)](https://www.rust-lang.org/)
[![CUDA 13](https://img.shields.io/badge/CUDA-13-76B900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)

Run [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) (552B MoE + 196B Engram)
at full routing on a desktop with two RTX 3090s, 128 GB of RAM and an NVMe drive. dsx is a Rust + CUDA engine built
for this one model. It runs three builds:

- [**lxe/DeepSeek-V4.1-Flash-1017-dsx-ud**](https://huggingface.co/lxe/DeepSeek-V4.1-Flash-1017-dsx-ud): Unsloth's
  UD-IQ1_M routed experts with dsx's Q8_0 dense weights and the official FP8 Engram. The best quality.
- [unsloth/DeepSeek-V4.1-Flash-1017-GGUF](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF) `UD-IQ1_M`,
  as downloaded: the smallest download, Engram included.
- [lxe/DeepSeek-V4.1-Flash-1017-dsx-2bit](https://huggingface.co/lxe/DeepSeek-V4.1-Flash-1017-dsx-2bit): dsx's own
  2-bit quant (GPTQ, Hadamard rotation, per-expert tiers).

| 2x RTX 3090, 128 GB DDR4 | dsx-ud | Unsloth UD-IQ1_M as-is | dsx-2bit |
|---|---:|---:|---:|
| Download | 142 GB + 203 GB official Engram | 250 GB | 148 GB + 203 GB official Engram |
| Decode, 3K-token prompt | 27-28.5 tok/s | 27-29.5 tok/s | 29.5 tok/s |
| Decode with `--defer-rank 3` | 31.5 tok/s | 31-31.7 tok/s | 35.7 tok/s |
| KL vs the FP8 model (Lucebox eval) | 0.270 (0.283 with deferral) | 0.277 | 0.373 |
| Top-1 agreement with the FP8 model | 79.0% | 78.0% | 74.4% |
| Perplexity (FP8 model: 8.56) | 10.13 | 10.11 | 10.68 |

For comparison, Unsloth's UD-IQ1_M in llama.cpp on the same machine: KL 0.275, top-1 78.8%, 12.3 tok/s decode.
Unsloth's file needs no `--engram`: dsx reads its Engram table from the GGUF.

## Quick start

Download the build, plus the Engram tables and tokenizer from the official checkpoint:

```bash
hf download lxe/DeepSeek-V4.1-Flash-1017-dsx-ud --local-dir dsx-ud
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 --local-dir v41-1017 \
  --include model.safetensors.index.json \
            model-00047-of-00048.safetensors \
            model-00048-of-00048.safetensors \
            tokenizer.json
```

Build for your GPU (`CUDA_ARCHS` from the table under [Requirements](#requirements)) and start the server:

```bash
CUDA_ARCHS=86 cargo build --release -p dsx --features cuda

./target/release/dsx-server \
  --model dsx-ud/DeepSeek-V4.1-Flash-1017-dsx-ud-00001-of-00004.gguf \
  --engram v41-1017 \
  --tokenizer v41-1017/tokenizer.json \
  --rank dsx-ud/expert-rank.json \
  --ctx 131072 \
  --arena-gb 110 \
  --defer-rank 3 \
  --host 0.0.0.0 --port 8090
```

`--defer-rank 3` trades a little quality for speed (see the table); leave it out for the best quality. For dsx's own
quant, use `dsx-2bit` and its file names in place of `dsx-ud`.

Open `http://localhost:8090/` for the web UI, or point any OpenAI client at `http://localhost:8090/v1` (Anthropic
clients at `http://localhost:8090`):

```bash
curl http://localhost:8090/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "deepseek-v4.1-flash",
    "stream": true,
    "reasoning_effort": "none",
    "messages": [
      {"role": "user", "content": "Write a haiku about MoE routing."}
    ]
  }'
```

### Agents

For the [pi](https://github.com/badlogic/pi-mono) coding agent, add this to `~/.pi/agent/models.json`:

```json
{
  "providers": {
    "dsx": {
      "baseUrl": "http://localhost:8090/v1",
      "api": "openai-completions",
      "apiKey": "none",
      "authHeader": false,
      "compat": {
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": true,
        "maxTokensField": "max_tokens"
      },
      "models": [
        {
          "id": "deepseek-v4.1-flash",
          "reasoning": true,
          "input": ["text", "image"],
          "contextWindow": 131072,
          "maxTokens": 8192,
          "thinkingLevelMap": {
            "off": "none",
            "minimal": null,
            "low": "low",
            "medium": null,
            "high": "high",
            "xhigh": null,
            "max": "max"
          }
        }
      ]
    }
  }
}
```

```bash
pi --provider dsx --model deepseek-v4.1-flash:off
```

With `--api-key`: drop `"authHeader": false`, set `"apiKey"` to the key and point `baseUrl` at the server's URL.

## Requirements

| | |
|---|---|
| GPU | NVIDIA, 24 GB of VRAM or more |
| CUDA | 13 |
| RAM | 128 GB |
| CPU | x86 with AVX2 (AVX-VNNI or AVX-512 VNNI preferred) |
| Disk | ~350 GB: the quant (148 GB) + the Engram shards (203 GB) |

| GPU | `CUDA_ARCHS` | Status |
|---|---|---|
| RTX 3090 | 86 | runs |
| RTX 4090 | 89 | runs |
| RTX 5090 | 120 | runs |
| A100 | 80 | builds, not run yet |

## Parameters

`dsx-server` needs `--model`, `--engram` and `--tokenizer`; every other option has a default.

**Model files**

| | |
|---|---|
| `--model` | the quant's first GGUF shard (the others load from beside it) |
| `--engram` | directory with the official checkpoint's Engram shards (`model-00047/00048-of-00048.safetensors`) |
| `--tokenizer` | the official `tokenizer.json` |
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

**Memory** (dsx places experts on the GPUs automatically; [docs/placement.md](docs/placement.md) lists the overrides)

| | |
|---|---|
| `--arena-gb` | RAM for experts the CPU computes (default 0; ~110 on 128 GB) |
| `--chunk` | prefill tokens per pass (default 4096); smaller leaves more VRAM for experts |
| `DSX_GPUS=0,2` | use only these GPUs (all by default) |

**Speed**

| | |
|---|---|
| `--defer-rank R` | add the CPU's lower-ranked experts (router rank R or lower, 0 = best of 6) one layer later instead of waiting for them. `--defer-rank 3`: decode ~30 → ~36 tok/s on 2x RTX 3090, KL 0.369 → 0.383 against the FP8 model. Off by default because outputs change slightly |

Settings by machine (the flags are explained in [docs/placement.md](docs/placement.md)):

| Machine | Flags |
|---|---|
| 2x 24 GB GPU, 128 GB RAM | `--arena-gb 110` |
| 1x 32 GB GPU, 128 GB RAM | `--main-gb=-2.0 --arena-gb 80`, `DSX_ADAPT=0` |

## Thinking

Thinking is on by default, at high effort. A request turns it off or sets the effort:

| | OpenAI API (`/v1/chat/completions`) | Anthropic API (`/v1/messages`) |
|---|---|---|
| Thinking off | `"reasoning_effort": "none"` or `"thinking": {"type": "disabled"}` | `"thinking": {"type": "disabled"}` |
| Effort | `"reasoning_effort"`: `low` (or `minimal`), `high` (or `medium`), `xhigh`, `max` | `"output_config": {"effort": ...}`: `low`, `high` (or `medium`), `xhigh`, `max` |

## Docs

| | |
|---|---|
| [docs/placement.md](docs/placement.md) | where the experts go (VRAM, RAM, disk) and how to tune it |
| [DESIGN.md](DESIGN.md) | how a token runs: the per-layer handoff between GPUs and CPU, prefill, the server |
| [nvidia/README.md](nvidia/README.md) | the `dsx` command-line tools (`gen`, `eval`, `cmp`, `ref`, `spec`) and debug switches |
| [quant/README.md](quant/README.md) | how the 2-bit quant is built |
| [docs/techniques.md](docs/techniques.md), [FINDINGS.md](FINDINGS.md) | techniques with sources, and measurements |

## Repo

| Path | |
|---|---|
| [`core/`](core/) | config, GGUF / safetensors readers, block formats, the CPU reference forward, the server and web UI |
| [`nvidia/`](nvidia/) | the desktop runner: CUDA kernels, expert placement over VRAM / RAM / disk, CPU expert kernels ([nvidia/README.md](nvidia/README.md)) |
| [`amd/`](amd/) | a separate runner for one AMD MI325X, with its own near-lossless MXFP4 build ([amd/README.md](amd/README.md)) |
| [`quant/`](quant/) | the 2-bit quantization pipeline, with native CUDA GPTQ |
| [`cards/`](cards/) | the Hugging Face model cards |
| [`docs/`](docs/) | the docs listed above |

`cargo build --release` builds `core` and the NVIDIA runner; without `--features cuda`, the runner is built for the
CPU only. Binaries go to `target/release/`.

## Credits

[DeepSeek](https://huggingface.co/deepseek-ai) for the model (MIT) ·
[Xolotl / Strata](https://github.com/Indras-Mirror/Xolotl) for the GSQ tools and V4.1 GGUF format (MIT; derived files in `quant/v41` carry its license) ·
DeepSeek for the reference `model.py` the calibration runs · GPTQ ([Frantar et al.](https://arxiv.org/abs/2210.17323)) ·
[Lucebox](https://huggingface.co/Lucebox/DeepSeek-V4.1-Flash-1017-ROCMFP-GGUF) for the eval set ·
antirez for the calibration prompts · the web UI is [llama.cpp](https://github.com/ggml-org/llama.cpp)'s, by Georgi
Gerganov and the ggml authors (MIT), vendored in [`core/webui`](core/webui) with its license.

MIT license.
