# dsx design

One model (DeepSeek-V4.1-Flash-1017), one quant format, one class of machine: two 24 GB GPUs, 128 GB of RAM, an NVMe
drive. The performance model behind these choices is in [FINDINGS.md](FINDINGS.md), section 3.

## Principles

1. **No experts on disk.** A read from NVMe costs ~7× the same read from RAM; VRAM + RAM hold the whole expert pool.
2. **The CPU computes RAM experts in place** during decode. GPUs take RAM experts only during prefill, where one copy
   serves a whole chunk.
3. **VRAM holds the hottest experts**, ranked by measured routing frequency; the main GPU's share adapts to the
   conversation.
4. **Roles follow the links.** At startup each GPU's host bandwidth is measured: the best-linked GPU (GPU0 here) is
   the main one; other GPUs are expert stores. A store whose link is at least half as fast swaps experts like the main
   GPU's cache and takes a share of prefill's expert stream; a slower one (GPU1 on a PCIe 3.0 x4 slot) keeps a fixed
   set and carries only activations.
5. **Quant formats fit the kernels:** CPU unpacking keeps up with RAM bandwidth (AVX2 / AVX-VNNI), GPU experts run on
   int8 `mma.sync`.
6. **Exact routing.** A routing bias exists for experiments, off by default.

## Memory layout

| Device | Contents |
|---|---|
| GPU0 (24 GB, x16) | dense weights (Q8_0, ~8 GB), KV cache and indexer keys, adaptive expert cache (~11 GB), prefill buffers |
| GPU1 (24 GB, x4) | fixed set of hot experts (~23 GB) |
| GPU1 (24 GB, x16) | adaptive expert cache (~23 GB), prefill buffers and four staging buffers for its share of the stream |
| RAM (~110 GB; ~135 GB with an x16 GPU1) | hugepage-backed, CUDA-registered arena: every expert the CPU computes, plus copies of the adaptive caches' experts, laid out per (layer, expert) |
| NVMe | the GGUF (loaded once; the coldest experts are read from it on demand) and the Engram tables |

With GPU1 on x16 the arena also holds copies of GPU1's experts (an adaptive cache evicts back to RAM), so at
`--arena-gb 110` ~3,500 cold experts stay in the file; 150+ GB of RAM fits `--arena-gb 135`.

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

## Server

- OpenAI chat completions and responses, Anthropic messages, tool calls, streaming, through DeepSeek's
  `deepseek-recipe` prompt encoding.
- Prefix reuse: the KV of the last conversation stays on the GPU; a request continues it or rewinds to the snapshot
  at the end of the previous prompt.
- llama.cpp's web UI is embedded and served at `/`, with the llama.cpp endpoints it expects (`/props`, `/slots`) and
  a `timings` chunk at the end of streamed responses.
- Optional `--api-key`: one shared key, checked before every route that runs the model, in `Authorization: Bearer`
  or `x-api-key`. Health checks and the UI's static files stay open; the key arrives as a header, so TLS is the
  fronting proxy's job.

## Quantization

Per-expert tiers (ternary 1.75, Q2_0 2.25, Q3_K 3.44 bpw) allocated by measured error under one budget shared across
layers; GPTQ with Hessians from a layer-streaming calibration pass through DeepSeek's `model.py`, on calibration text
that includes the full model's own reasoning traces. Experts are quantized in a 128-block Hadamard-rotated basis
(GGUF key `dsx.expert_rot`; the engine rotates their inputs to match) and their group scales refit by least squares
after GPTQ. Native CUDA kernels handle GPTQ and the scale search, batched over experts. Details:
[quant/README.md](quant/README.md); techniques and their measured effects: [docs/techniques.md](docs/techniques.md).
