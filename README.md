<h1 align="center">dsx</h1>

<p align="center"><b>DeepSeek-V4.1-Flash-1017 (552 B + 196 B Engram) on your old gaming PC</b><br>
two RTX 3090s and 128 GB of RAM · 32-36 tokens/s · full routing, no expert pruning · OpenAI- and
Anthropic-compatible API · chat in the browser, pictures, thinking levels</p>

<p align="center">
<a href="https://huggingface.co/lxe/DeepSeek-V4.1-Flash-1017-dsx-ud"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20model-dsx--ud-yellow" alt="Model: dsx-ud"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="License: MIT"></a>
<a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/Rust-stable-orange?logo=rust" alt="Rust"></a>
<a href="https://developer.nvidia.com/cuda-toolkit"><img src="https://img.shields.io/badge/CUDA-13-76B900?logo=nvidia" alt="CUDA 13"></a>
</p>

Models this size usually need a datacenter. dsx runs
**[DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)** (DeepSeek, MIT license;
a 552 B mixture of experts plus a 196 B Engram memory) on a desktop with two RTX 3090s, 128 GB of RAM and an NVMe
drive. It's an engine written in Rust and CUDA for this one model. The experts each token uses most stay on the GPUs,
the next ones in RAM where the CPU computes them, and the GPUs and CPU hand each layer back and forth without stalling
each other.

- **Fast:** 32-36 tokens/s writing the answer and ~370 tokens/s reading a 3,000-token prompt on two RTX 3090s.
- **Close to the original:** the same next token as the FP8 model 78-79% of the time on the Lucebox eval.
- **One download:** one 305 GB GGUF in 5 shards, with the Engram table and tokenizer inside.
- **Plugs into your tools:** OpenAI- and Anthropic-compatible API with streaming, ready for coding agents.
- **Everything in the box:** a chat in the browser, pictures, thinking on or off, effort levels.

## The model

https://huggingface.co/lxe/DeepSeek-V4.1-Flash-1017-dsx-ud

One 305 GB download in 5 GGUF shards. The routed experts are 1-2 bit, from
[Unsloth](https://github.com/unslothai/unsloth)'s [UD-IQ1_M](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF).
The dense weights are Q8_0 and the Engram table is Q6_K, and the tokenizer and vision tower are included.

Against the FP8 model on the [Lucebox](https://huggingface.co/Lucebox/DeepSeek-V4.1-Flash-1017-ROCMFP-GGUF) eval
(8 x 1,024 tokens), it picks the same next token 78.9% of the time, with KL 0.269 and perplexity 10.13 (FP8: 8.56).
With `--defer-rank 3`, which is faster, those are 78.2%, KL 0.284 and perplexity 10.25.

## How fast is it?

On two RTX 3090s (24 GB each) with an i7-12700K and 128 GB of DDR4, dsx writes **32-36 tokens/s** with
`--defer-rank 3` and 27-29.5 without it. It reads a 3,000-token prompt at ~370 tokens/s, and a 340-token prompt in
4.0 s. On the same PC, the same expert weights run at 12 tokens/s in llama.cpp.

## Quick start

Download the build (the Engram table and tokenizer are inside it):

```bash
hf download lxe/DeepSeek-V4.1-Flash-1017-dsx-ud --local-dir dsx-ud
```

Build for your GPU (`CUDA_ARCHS` from the table under [Requirements](#requirements)) and start the server:

```bash
CUDA_ARCHS=86 cargo build --release -p dsx --features cuda

./target/release/dsx-server \
  --model dsx-ud/DeepSeek-V4.1-Flash-1017-dsx-ud-00001-of-00005.gguf \
  --rank dsx-ud/expert-rank.json \
  --ctx 131072 \
  --arena-gb 110 \
  --defer-rank 3 \
  --host 0.0.0.0 --port 8090
```

`--defer-rank 3` trades a little quality for speed (see [The model](#the-model)); leave it out for the best quality.

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
          "input": ["text"],
          "contextWindow": 131072,
          "maxTokens": 8192,
          "thinkingLevelMap": { "off": "none", "high": "high" }
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
| Disk | ~305 GB on an NVMe drive |

| GPU | `CUDA_ARCHS` | Status |
|---|---|---|
| RTX 3090 | 86 | runs |
| RTX 4090 | 89 | runs |
| RTX 5090 | 120 | runs |
| A100 | 80 | builds, not run yet |

## Parameters

`dsx-server` needs `--model`; every other option has a default.

**Model files**

| | |
|---|---|
| `--model` | the quant's first GGUF shard (the others load from beside it) |
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

**Memory** (dsx places experts on the GPUs automatically; [docs/placement.md](docs/placement.md) lists the overrides)

| | |
|---|---|
| `--arena-gb` | RAM for experts the CPU computes (default 0; ~110 on 128 GB) |
| `--chunk` | prefill tokens per pass (default 4096); smaller leaves more VRAM for experts |
| `DSX_GPUS=0,2` | use only these GPUs (all by default) |

**Speed**

| | |
|---|---|
| `--defer-rank R` | add the CPU's lower-ranked experts (router rank R or lower, 0 = best of 6) one layer later instead of waiting for them. `--defer-rank 3`: decode 27-29.5 → 32-36 tok/s on 2x RTX 3090, KL 0.269 → 0.284 against the FP8 model. Off by default because outputs change slightly |

Settings by machine (the flags are explained in [docs/placement.md](docs/placement.md)):

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

## Docs

| | |
|---|---|
| [docs/placement.md](docs/placement.md) | where the experts go (VRAM, RAM, disk) and how to tune it |
| [DESIGN.md](DESIGN.md) | how a token runs: the per-layer handoff between GPUs and CPU, prefill, the server |
| [nvidia/README.md](nvidia/README.md) | the `dsx` command-line tools (`gen`, `eval`, `cmp`, `ref`, `spec`) and debug switches |
| [docs/techniques.md](docs/techniques.md), [FINDINGS.md](FINDINGS.md) | techniques with sources, and measurements |

## Repo

| Path | |
|---|---|
| [`core/`](core/) | config, GGUF / safetensors readers, block formats, the CPU reference forward, the server and web UI |
| [`nvidia/`](nvidia/) | the desktop runner: CUDA kernels, expert placement over VRAM / RAM / disk, CPU expert kernels ([nvidia/README.md](nvidia/README.md)) |
| [`amd/`](amd/) | a separate runner for one AMD MI325X, with its own near-lossless MXFP4 build ([amd/README.md](amd/README.md)) |
| [`quant/`](quant/) | quantization experiments and tools, with native CUDA GPTQ |
| [`cards/`](cards/) | the Hugging Face model cards |
| [`docs/`](docs/) | the docs listed above |

`cargo build --release` builds `core` and the NVIDIA runner; without `--features cuda`, the runner is built for the
CPU only. Binaries go to `target/release/`.

## How it works

The model has 15,360 experts (384 in each of its 40 layers), and each token uses 6 per layer. Together they take far
more memory than two GPUs have, so dsx spreads them over everything the PC has and keeps each one where it's cheapest
to run.

- **Hot experts on the GPUs, the rest in RAM:** a ranking from calibration text says which experts the model picks
  most. The hottest fill the GPUs, the next ~110 GB go into RAM, where the CPU computes them, and the few coldest stay
  in the file on the SSD. On two RTX 3090s, about three of every four expert picks land on a GPU.
- **It follows your conversation:** after every token, dsx counts which experts were used and swaps a few hot ones
  from RAM onto the main GPU in the background, replacing ones that went cold. A coding session and a chat about
  history end up with different experts on the card.
- **It measures your GPUs:** at startup, dsx times how fast each GPU talks to the CPU and checks its memory. The
  best-connected card becomes the main GPU and holds the dense weights (attention and the shared parts), the conversation's memory (the KV cache)
  and its own expert cache. Every other card only holds experts. A card with a link about as fast swaps its experts
  too and helps stream experts during long prompts. A card in a slow slot (a PCIe x4 slot, say) keeps a fixed set of
  experts instead, because copying experts would compete with the work it needs every layer. Mismatched cards are
  fine: up to six GPUs, each used for what it's good at.
- **GPUs and CPU work at the same time:** in every layer, the main GPU picks the experts and hands their input to the
  CPU and the other GPUs through shared memory. Each computes its own experts in parallel, and the main GPU adds the
  results. The handoff takes about 6 µs and never waits on the operating system.
- **Fast CPU kernels:** the experts in RAM run on hand-written AVX2 / AVX-VNNI kernels across the CPU's performance
  cores, from memory locked in large pages so the CPU reads it at full speed.
- **Optional deferral:** `--defer-rank 3` lets the CPU's least important experts finish one layer late instead of
  holding up the GPU: 15-20% faster decode for a small change in the output.
- **Long prompts:** prompts are read in chunks of 4,096 tokens. The experts no GPU holds stream from RAM while the
  GPUs compute, over every fast link at once, and the other GPUs work on the same chunk in parallel.
- **Picks up where you left off:** the last conversation stays on the GPU, so the next message only reads the new
  text.

## Credits

[Unsloth](https://github.com/unslothai/unsloth) for the expert weights in dsx-ud, from their [UD-IQ1_M GGUF](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF) · [DeepSeek](https://huggingface.co/deepseek-ai) for the model (MIT) ·
[Xolotl / Strata](https://github.com/Indras-Mirror/Xolotl) for the GSQ tools and V4.1 GGUF format (MIT; derived files in `quant/v41` carry its license) ·
DeepSeek for the reference `model.py` the calibration runs · GPTQ ([Frantar et al.](https://arxiv.org/abs/2210.17323)) ·
[Lucebox](https://huggingface.co/Lucebox/DeepSeek-V4.1-Flash-1017-ROCMFP-GGUF) for the eval set ·
antirez for the calibration prompts · the web UI is [llama.cpp](https://github.com/ggml-org/llama.cpp)'s, by Georgi
Gerganov and the ggml authors (MIT), vendored in [`core/webui`](core/webui) with its license.

MIT license.
