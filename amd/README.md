<h1 align="center">dsx-amd</h1>

<p align="center"><b>DeepSeek-V4.1-Flash-1017 (552 B + 196 B Engram) on one AMD MI325X</b><br>
~100 tokens/s per session with four coding agents · full routing, no expert pruning · Rust + HIP ·
OpenAI- and Anthropic-compatible API</p>

<p align="center">
<a href="https://huggingface.co/lxe/DeepSeek-V4.1-Flash-1017-dsx-mxfp4"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20model-dsx--mxfp4-yellow" alt="Model: dsx-mxfp4"></a>
<a href="../LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="License: MIT"></a>
<a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/Rust-stable-orange?logo=rust" alt="Rust"></a>
<a href="https://rocm.docs.amd.com/"><img src="https://img.shields.io/badge/ROCm-7-ED1C24?logo=amd" alt="ROCm 7"></a>
</p>

**[DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)** on one 256 GiB AMD
Instinct MI325X. dsx-amd keeps the model's experts and dense weights in GPU memory, with the 196 B Engram table on
NVMe and in the host's page cache. It is an engine written in Rust and HIP for this model and this GPU. The model
configuration, file readers, CPU reference and API server come from the same [`dsx-core`](../core) as the
[two-3090 runner](../README.md).

- **Four agents at once:** about 100 decode tokens/s per session in measured four-session coding workloads.
- **Room for the model:** native MXFP4 experts mixed with fitted 3-bit experts, placed to fit the available GPU memory.
- **Close to the original:** the published build picks the same next token as FP8 90.6% of the time on the Lucebox eval.
  The serving profile below makes more room for batching and drafts, with about 88.3% agreement in an earlier evaluation of its expert placement.
- **Plugs into your tools:** OpenAI- and Anthropic-compatible API with streaming and tool calls, plus a chat in the browser.
- **Long conversations:** a 256K context budget and up to eight concurrent sessions in the coding profile.

## The model

https://huggingface.co/lxe/DeepSeek-V4.1-Flash-1017-dsx-mxfp4

Six GGUF shards, 270.5 GB. The published build keeps 8,992 routed experts in the checkpoint's own MXFP4 format and
fits the remaining 6,368 to a 3-bit codebook. All 15,360 experts remain available to the router. Dense weights retain
the checkpoint's FP8 and BF16 formats; the Engram table is read from the official checkpoint's separate shards.

Against the FP8 model on the [Lucebox](https://huggingface.co/Lucebox/DeepSeek-V4.1-Flash-1017-ROCMFP-GGUF) eval
(8 × 1,024 tokens), the published build has **90.6% next-token agreement**, KL 0.049 and perplexity 8.71
(FP8: 8.56).

The coding profile reserves more GPU memory for concurrent sessions and the checkpoint's native draft stack.
Its expert placement is 3,378 MXFP4 and 11,982 3-bit experts, about 235.8 GB. An earlier evaluation with this expert
placement measured top-256 KL 0.076–0.083 and about 88.3% next-token agreement. That is a placement measurement,
not a fresh quality evaluation of the final serving build. [Measurements and quantization details](../FINDINGS-AMD.md).

## How fast is it?

The coding profile measured **102–108 decode tokens/s per session** in two fixed four-session code replays.
Earlier trials reached as low as 88; these are short windows with all four sessions decoding and no prefill,
so they do not describe end-to-end request throughput. A single-session code replay measured 246 tokens/s;
eight concurrent sessions averaged 61 each, or 486 total. The four-session workload is the main tuning target.

The earlier published-build measurements used a different serving configuration:

| Published build, one MI325X | |
|---|---:|
| Decode at 4K / 128K / 256K context | 114 / 112 / 110 tokens/s |
| Prefill at 4K / 16K / 128K | ~1,640 / ~1,640 / ~1,160 tokens/s |
| Greedy DSpark speedup | 1.4–1.7× on code, 1.2–1.4× on math, ~1.0× on prose |

The coding profile uses target-verified five-token native MTP proposals and prompt lookup. At positive temperature,
proposals go through probabilistic acceptance and rejection. It gives ongoing output priority by running extra
decode rounds between prompt chunks. [Profile settings and limits](box/AGENTIC_MI325X.md).

## Quick start

From the repository root, download the build and the official checkpoint files used for configuration, the Engram
table and the native draft stack:

```bash
hf download lxe/DeepSeek-V4.1-Flash-1017-dsx-mxfp4 --local-dir dsx-mxfp4
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 --local-dir v41-1017 \
  --include config.json tokenizer.json model.safetensors.index.json \
    model-00044-of-00048.safetensors model-00045-of-00048.safetensors \
    model-00046-of-00048.safetensors model-00047-of-00048.safetensors \
    model-00048-of-00048.safetensors
cp dsx-mxfp4/dsx-engram.json v41-1017/

cargo build --release -p dsx-amd
```

Start the coding profile:

```bash
DSX_MODEL_DIR="$PWD/v41-1017" \
DSXA_WEIGHTS="$PWD/dsx-mxfp4/DeepSeek-V4.1-Flash-1017-dsx-mxfp4-00001-of-00006.gguf" \
DSX_RANK="$PWD/dsx-mxfp4/expert-rank.json" \
  bash amd/box/serve-agentic-mi325x.sh
```

Open `http://localhost:8090/` for the web UI, or point an OpenAI client at `http://localhost:8090/v1`
(Anthropic clients at `http://localhost:8090`). The profile binds the API and admin service to loopback;
use an SSH tunnel when the GPU is on another machine.

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

The profile defaults to temperature 0.1 and top_p 1.0. A request can override them. The decode measurements above
used thinking off, as in this example.

To run the published build without the draft stack or the coding profile:

```bash
DSXA_WEIGHTS="$PWD/dsx-mxfp4/DeepSeek-V4.1-Flash-1017-dsx-mxfp4-00001-of-00006.gguf" \
  ./target/release/dsx-amd serve --model v41-1017 --ctx 262144 \
    --host 127.0.0.1 --port 8090
```

## Requirements

| | |
|---|---|
| GPU | One AMD Instinct MI325X, 256 GiB HBM3E (`gfx942`), or MI350X, 288 GiB (`gfx950`, see below) |
| ROCm | 7 for `gfx942`, measured on 7.14; 10.1 for `gfx950`, the only version tested there |
| Host | Tested with 20 vCPUs and 157 GiB RAM |
| Disk | NVMe, with room for the 270.5 GB export and the official Engram and MTP shards |
| Build | Rust and `hipcc` (`/opt/rocm/bin/hipcc`, or set `HIPCC`) |

The kernels compile for `gfx942` by default; set `DSX_AMD_ARCH=gfx950` for the MI350X. Other AMD GPUs need their own kernel build and memory configuration.
Without ROCm, the host code can be checked with `cargo check -p dsx-amd`, but the resulting build has no GPU kernels.

### MI350X (`gfx950`)

Build with `DSX_AMD_ARCH=gfx950 cargo build --release -p dsx-amd`; the model files and `serve-agentic-mi325x.sh` are unchanged.
With 288 GiB the coding profile keeps the published placement (8,992 MXFP4, 6,368 3-bit) and leaves 28.8 GB of HBM free.

gfx950 reads the same fp8 byte as OCP e4m3 rather than FNUZ, at twice the value. `kernels/common.h` converts at the hardware
builtins, so the kernels and stored bytes stay as they are on gfx942.

On Ubuntu 26.04 install `amdrocm-core-dev10.1-gfx950`, `amdrocm-blas-dev10.1` and `amdrocm-blas10.1-gfx950`, plus
`linux-firmware-amd-graphics`: without that firmware `amdgpu` fails at boot on this GPU.

Measured on one MI350X VF under KVM (24 vCPUs, 251 GiB RAM, ROCm 10.1, in-box `amdgpu`, Linux 7.0):

| | |
|---|---:|
| Lucebox eval, two runs | KL 0.0505 / 0.0507, top-1 90.65% / 90.38%, perplexity 8.66 / 8.67 |
| Single-stream decode, `bench` at 4K / 256K context | 114.6 / 110.5 tokens/s |
| Coding-profile server, 300-token greedy requests including prefill, six prompts, drafts off | 90 tokens/s |
| Same, native drafts on, prompt lookup off | 181–182 tokens/s (108–263 by prompt), 47–48% acceptance |

The Docker image still builds for `gfx942` only, and `box/setup.sh` still installs the ROCm 7.14 `gfx942` packages.

## Tools

Run `./target/release/dsx-amd --help` for the command list, or add `--help` to a command for its options.

| Command | |
|---|---|
| `serve --model DIR` | API server and web UI |
| `export --model DIR --rank RANK --out FILE` | write the MXFP4 + 3-bit placement to GGUF; split with `dsx split` |
| `eval --model DIR --eval EVALDIR` | KL, next-token agreement and perplexity on the Lucebox eval |
| `check --model DIR --ids ...` | compare GPU decode with the CPU reference |
| `bench`, `selftest` | decode timing and kernel checks against CPU implementations |
| `dev --model DIR` | load once, then run benchmarks or reload kernels through a command file |

## How it works

- **Experts stay in GPU memory.** The loader keeps the hottest experts in native MXFP4 and uses fitted 3-bit
  codebooks for colder experts. The memory budget determines how many fit in each format.
- **The Engram table stays on the host.** Each token looks up a small set of rows from the NVMe-backed table;
  the operating system caches frequently used pages in RAM.
- **Drafts save weight reads.** The checkpoint's MTP stack proposes tokens, and the main model verifies several
  positions together. Prompt lookup can also propose continuations from text already in the conversation.
- **Sessions share GPU work.** Continuous batching combines active sessions while admitting new prompts in chunks.
  The coding profile limits verification rows and spaces prompt chunks to keep decoding responsive.

## Docs

| | |
|---|---|
| [box/AGENTIC_MI325X.md](box/AGENTIC_MI325X.md) | coding profile, memory placement, sampling and live tuning |
| [../FINDINGS-AMD.md](../FINDINGS-AMD.md) | published-build measurements, quantization and kernel design |
| [RUNNER.md](RUNNER.md) | development notes and earlier experiments |
| [../core/](../core/) | shared model code, readers, CPU reference, server and web UI |
| [src/](src/), [kernels/](kernels/) | AMD runner and HIP kernels |
| [box/](box/), [tools/](tools/), [lab/](lab/) | host setup, quantization tools and kernel experiments |

MIT license.
