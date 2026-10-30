# dsx design

The NVIDIA runner targets DeepSeek-V4.1-Flash-1017 on desktop GPUs and RAM. Its default downloadable model uses
T16 routed experts, Q8_0 dense weights and an IQ4_NL Engram lookup table; it also reads the older scalar and Unsloth IQ
formats. The reference machine has two 24 GB GPUs, 128 GB of RAM and an NVMe drive. The performance model behind
these choices is in [FINDINGS.md](FINDINGS.md), section 3.

## Principles

1. **Fit the expert pool in VRAM and RAM.** A read from NVMe costs ~7× the same read from RAM. The measured T16
   reference configuration covers every expert; smaller configurations can fall back to the GGUF.
2. **The CPU computes RAM experts in place** during decode. GPUs take RAM experts only during prefill, where one copy
   serves a whole chunk.
3. **VRAM holds the hottest experts**, ranked by measured routing frequency; the main GPU's share adapts to the
   conversation.
4. **Roles follow the links.** At startup each GPU's host bandwidth is measured: the best-linked GPU (GPU0 here) is
   the main one; other GPUs are expert stores. A store whose link is at least half as fast swaps experts like the main
   GPU's cache and takes a share of prefill's expert stream; a slower one (GPU1 on a PCIe 3.0 x4 slot) keeps a fixed
   set. It receives activations and computes its resident experts, without streaming nonresident expert weights.
5. **Quant formats fit the kernels:** the T16 CPU path uses AVX-VNNI integer dot products, while its GPU decode path uses
   staged trellis loads and `dp4a`. Other supported expert formats use their own CPU and GPU paths, including int8
   `mma.sync` tiles. Throughput depends on bit rate and processor; see [docs/t16.md](docs/t16.md).
6. **Exact routing.** A routing bias exists for experiments, off by default.

## Memory layout

| Device | Contents |
|---|---|
| GPU0 (24 GB, x16) | dense weights (Q8_0, ~8 GB), KV cache and indexer keys, adaptive expert cache (~11 GB), prefill buffers |
| GPU1 (24 GB, x4) | fixed set of hot experts (~23 GB) |
| GPU1 (24 GB, x16) | adaptive expert cache (~23 GB), prefill buffers and four staging buffers for its share of the stream |
| RAM (~110 GB; ~135 GB with an x16 GPU1) | CUDA-registered arena, requesting huge pages with a 4 KiB fallback: CPU experts plus backing copies of adaptive caches' experts |
| NVMe | the GGUF and Engram lookup table; file-only experts are read on demand if the selected arena and GPU stores do not cover them |

The recorded T16 x16+x4 placement has no file-only experts. See [placement counts](docs/placement.md) for the
configuration and how backing copies are counted.

A fast-linked adaptive GPU1 also needs arena backing copies. If that would leave more than 5% of expert bytes in
the file, automatic placement keeps it fixed while still allowing its prefill stream. Forcing it adaptive can
create file-only experts; the historical 110/135 GB arena comparison is in [FINDINGS §8c](FINDINGS.md#8c-two-gpus-on-x16-links-gpu1-as-a-second-cache-and-a-second-stream).

## Decode

Per layer, with no host synchronization inside a token:

1. GPU0 runs attention and the router, then writes the 6 expert ids and the expert input into mapped host memory
   and raises a flag.
2. The CPU worker (a coordinator on its own core, 14 compute threads on the P-cores) sees the flag, computes the
   RAM-resident experts, writes the partial sum and raises its flag.
3. Each aux GPU's stream (GPU1 here) waits on the same flag (`cuStreamWaitValue32`), runs its resident experts and
   writes its partial sum into its own slot.
4. GPU0 runs its own resident experts and the shared expert meanwhile, waits on every flag, adds the sums and
   continues.

A flag round trip costs ~6 µs. After each token, each adaptive expert cache commits finished copies and starts up to
8 new ones: the hottest experts by decayed usage replace the coldest resident expert of the same quant type, copied from the
arena and repacked on a side stream.

## Prefill

- Layers 0-20 run on every prompt token; layers 21-39 run only on the tokens later positions can still attend to.
- Prompts are processed in chunks (4096 tokens by default). Each chunk streams the arena's experts through four
  staging buffers per GPU while the previous expert computes; each aux GPU computes its own experts on the shipped
  activations. An aux GPU with a fast link (at least half the main GPU's) also takes a share of the streamed experts:
  each goes to the GPU whose queued copies finish first, with the main GPU's link counted twice (the aux GPU also
  computes its larger resident set). The host enqueues the GPUs' expert work round-robin, since a CUDA stream accepts
  only ~1K queued operations before the host blocks.
- Attention uses a tensor-core kernel for long chunks, and Engram rows are deduplicated per chunk.
- For chunks up to 512 tokens, a cost model assigns some nonresident experts to the CPU while the GPUs stream the
  rest. Large prefill chunks stream all nonresident experts; the CPU split and its controls are in
  [docs/placement.md](docs/placement.md).

## Server

- OpenAI chat completions and responses, Anthropic messages, tool calls, streaming, through DeepSeek's
  `deepseek-recipe` prompt encoding.
- Prefix reuse: the KV of the current conversation stays on the GPU. The engine keeps a bounded history of
  prompt-boundary snapshots and resumes from the newest checkpoint at or before the longest common token prefix;
  only a prefix shorter than every checkpoint starts a cold prefill. Exact retries reuse checkpoint logits.
- llama.cpp's web UI is embedded and served at `/`, with the llama.cpp endpoints it expects (`/props`, `/slots`) and
  a `timings` chunk at the end of streamed responses.
- Optional `--api-key`: one shared key, checked before every route that runs the model, in `Authorization: Bearer`
  or `x-api-key`. Health checks and the UI's static files stay open; the key arrives as a header, so TLS is the
  fronting proxy's job.

## Quantization

T16 experts use ExLlamaV3's Viterbi/LDLQ quantizer with Hadamard-128 transforms, sigma_reg=2 damping, a 16-bit
computed codebook and column-major positions within 16×16 tiles. Layers 0–19 use 1.5/1.5/2 bits for gate/up/down;
layers 20–39 use 2/2/2.5. Output gains are baked into the stored scales. Those gains were selected on the reported
evaluation set, which limits the independence of its quality figures. The shipped lookup table is IQ4_NL; the
separate Engram projection matrices remain F16 unless the optional runtime FP8 conversion is enabled.

The older dsx-2bit pipeline uses per-expert ternary, Q2_0 and Q3_K tiers with GPTQ, Hadamard rotation and scale refit.
dsx-ud instead uses Unsloth's routed-expert bytes unchanged. Details: [quant/README.md](quant/README.md),
[T16 format](docs/t16-format.md), [measurements and open checks](docs/current-state.md).
