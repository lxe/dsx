<h1 align="center">dsx</h1>

<p align="center"><b>DeepSeek-V4.1-Flash-1017 (552 B + 196 B Engram) on your old gaming PC</b><br>
two RTX 3090s and 128 GB of RAM · 40+ tokens/s decode · 1,000+ tokens/s prefill measured · full routing, no expert pruning · OpenAI- and
Anthropic-compatible API · chat in the browser, pictures, thinking levels</p>

<p align="center">
<a href="https://huggingface.co/lxe/DeepSeek-V4.1-Flash-1017-dsx-t16"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20model-dsx--t16-yellow" alt="Model: dsx-t16"></a>
<a href="https://x.com/lxe"><img src="docs/assets/twitter-lxe.svg" alt="Follow @lxe on Twitter / X"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="License: MIT"></a>
<a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/Rust-stable-orange?logo=rust" alt="Rust"></a>
<a href="https://developer.nvidia.com/cuda-toolkit"><img src="https://img.shields.io/badge/CUDA-13-76B900?logo=nvidia" alt="CUDA 13"></a>
</p>

dsx runs **[DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)**
locally on two RTX 3090s and 128 GB of RAM. It is a Rust and CUDA inference engine with a T16 quantized model,
a browser chat UI, image input, and OpenAI- and Anthropic-compatible APIs for coding agents.

The [current model](https://huggingface.co/lxe/DeepSeek-V4.1-Flash-1017-dsx-t16) is a single 252 GB GGUF with
the experts, Engram memory and tokenizer included. The download also supplies the expert ranking and a separate
vision tower. [T16](docs/t16-format.md) uses a trellis format designed for both desktop CPU and GPU execution.

## Quick start

The reference machine has two 24 GB RTX 3090s, an i7-12700K, 128 GB of RAM and an NVMe drive with about 255 GB free.
You need Rust, CUDA 13 and an x86 CPU with AVX2; AVX-VNNI or AVX-512 VNNI is preferred.

Download the model and build the NVIDIA runner:

```bash
hf download lxe/DeepSeek-V4.1-Flash-1017-dsx-t16 --local-dir dsx-t16
CUDA_ARCHS=86 cargo build --release -p dsx --features cuda
```

Use `CUDA_ARCHS=89` for an RTX 4090 or `120` for an RTX 5090. **`CUDA_ARCHS=70` runs on Tesla V100 (Volta) — see [Tesla V100](#tesla-v100-sm70).**

### Tesla V100 (SM70)

`CUDA_ARCHS=70` (with CUDA 12.9) builds for Volta: every SM80-only kernel has an SM70 twin, and SM80+ binaries are
unchanged. Measured on 2x V100-PCIE-32GB with 108 GB of RAM: 21-23 tokens/s decode and 467 tokens/s prefill on a
123K-token prompt. Details and checks: [bench/results/2026-11-01-volta](bench/results/2026-11-01-volta/README.md).

```bash
CUDA_ARCHS=70 cargo build --release -p dsx --features cuda
DSX_GPUS=0,1 ./target/release/dsx-server --model dsx-t16/DeepSeek-V4.1-Flash-1017-dsx-t16.gguf \
  --rank dsx-t16/expert-rank.json --main-gb=-2.0 --arena-gb 85 --ctx 131072
```

Start the server:

```bash
./target/release/dsx-server \
  --model dsx-t16/DeepSeek-V4.1-Flash-1017-dsx-t16.gguf \
  --rank dsx-t16/expert-rank.json \
  --ctx 131072 \
  --arena-gb 110 \
  --defer-rank 3 \
  --host 0.0.0.0 --port 8090
```

Open **http://localhost:8090/** for the chat UI. The server loads `vision.safetensors` beside the model for image input.
Use `--no-vision` for text only.

`--defer-rank 3` lets lower-ranked CPU experts finish one layer late to reduce GPU waiting. It changes the model’s
computation; omit the flag to disable that approximation. See [server configuration](docs/server.md) for all options
and [expert placement](docs/placement.md) for other memory and GPU configurations.

Or build and run the same server in Docker (NVIDIA Container Toolkit required). The default image carries sm_86,
sm_89 and sm_120 kernels; `--build-arg CUDA_ARCHS=86` narrows the fatbin to one GPU:

```bash
docker build -t dsx .
docker run --gpus all --ipc=host --ulimit memlock=-1:-1 \
  -v "$PWD/dsx-t16:/models:ro" -p 8090:8090 dsx \
  --model /models/DeepSeek-V4.1-Flash-1017-dsx-t16.gguf \
  --rank /models/expert-rank.json --ctx 131072 --arena-gb 110 \
  --host 0.0.0.0 --port 8090
```

`--target amd` builds the MI325X runner instead.

## Coding agents and API

Point an OpenAI-compatible client at `http://localhost:8090/v1`, or an Anthropic-compatible client at
`http://localhost:8090`. The model ID is `deepseek-v4.1-flash`.

For [Pi](https://github.com/badlogic/pi-mono), follow the [provider setup](docs/agents.md), then run:

```bash
pi --provider dsx --model deepseek-v4.1-flash:off
```

Thinking is on at high effort by default. Requests can turn it off or choose another effort. For example:

```bash
curl http://localhost:8090/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "deepseek-v4.1-flash",
    "stream": true,
    "reasoning_effort": "none",
    "messages": [
      {"role": "user", "content": "Write a Python function that merges sorted iterators."}
    ]
  }'
```

The default sampling is temperature 1 and top-p 1. [Thinking controls](docs/server.md#thinking) cover both APIs.
Use `--api-key` to require authentication.

## Performance

On the reference two-3090 machine, a T16 free-generation run measured **41.0 tokens/s** with `--defer-rank 3` and
optional FP8 Engram projections (`DSX_ENGRAM_FP8=1`). FP8 projections are off by default. This was a 120-token run
after 300 tokens; throughput depends on context length, expert routing and cache warm-up.

The October 25 prefill checkpoint measured **1,008 tokens/s** on an 8,000-token prompt with an 8K chunk,
256K context allocation and the same T16 weights. The [prefill profile and receipts](bench/results/2026-10-25-prefill/README.md)
cover the opt-in kernels, buffer reuse, warm-up and expert placement. This is a separate measurement from the
41 tokens/s generation result above; smaller prompt updates have different throughput.

The [measurement record](docs/current-state.md) gives build identities, quality measurements and open checks.

## How it works

The model selects six of its 384 experts in each layer. dsx keeps frequently used experts on the GPUs and computes
the remaining selected experts on the CPU from RAM. The CPU and GPUs work in parallel, and the main GPU combines
their results. An adaptive cache follows the experts used by the current conversation.

At startup, dsx measures each GPU’s memory and host-link speed to assign its role. During prefill, experts stream
from RAM over the available fast links. Follow-up requests reuse the cached conversation prefix so only new text
needs prefill. [DESIGN.md](DESIGN.md) describes the execution and memory layout.

## Documentation

| Guide | Contents |
|---|---|
| [Server configuration](docs/server.md) | Model files, authentication, sampling, thinking and runtime flags |
| [Coding agents](docs/agents.md) | Pi provider configuration |
| [Expert placement](docs/placement.md) | GPU, RAM and disk placement; tuning for your machine |
| [T16 format](docs/t16-format.md) | Quantization and CPU/GPU decoding |
| [NVIDIA runner](nvidia/README.md) | CLI tools, build details and debugging |
| [AMD runner](amd/README.md) | The separate MI325X implementation and model |
| [Measurements](docs/current-state.md) | Current runtime status, quality evidence and benchmark history |
| [Findings](FINDINGS.md) | Detailed experiments and optimization results |

MIT license. See [credits](CREDITS.md) for upstream projects, model sources and attribution.
