# DeepSeek-V4.1-Flash-1017 on a two-GPU desktop: findings

How fast can a 552B-parameter MoE run on a desktop with two RTX 3090s and 128 GB of DDR4, with exact routing, and
what limits it? This document collects the analysis, measurements and results behind dsx. Engine architecture:
[DESIGN.md](DESIGN.md). Hardware microbenchmarks: [bench/RESULTS.md](bench/RESULTS.md). Speed model:
[roofline.py](roofline.py).

## Summary

| | |
|---|---|
| Decode | ~29 tok/s on 2x RTX 3090 + 128 GB DDR4, exact routing (v4) |
| Prefill | 301 tok/s (3K tokens), 470 tok/s (8K tokens) (v4) |
| Quality | KL 0.270, top-1 79.0%, perplexity 10.13 vs the FP8 model's 8.56 with Unsloth UD-IQ1_M's experts (142 GB, dsx-ud); dsx's own quant v4b: KL 0.373, top-1 74.4% (148 GB) |
| Limit | the experts the CPU computes from RAM (33-36 GB/s on DDR4-3200), then the dense half on GPU0 |

The levers that mattered, in order: keep every expert in VRAM or RAM (none on disk); rank VRAM residency by measured
expert hotness and let it follow the conversation; one CPU core per spinning thread; a parallel decode-attention kernel;
a quant whose bit budget is shared across layers, calibrated on the model's own reasoning; Hadamard-rotated experts with
refit scales.

## 1. The model

[DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017), MIT.

| | |
|---|---|
| Type | MoE: 384 routed experts + 1 shared per layer, 6 routed per token |
| Size | 552B backbone + 196B Engram tables + vision + 3 draft (DSpark / MTP) layers |
| Active | 16B parameters per decoded token |
| Layers | 40 (20 "encoder" + 20 "decoder"), hidden 5120, expert width 2304 |
| Attention | MQA sparse attention: 128-token window + top-512 compressed rows chosen by an indexer |
| KV cache | FP4, 890 bytes per token (128K tokens = 117 MB) |
| Checkpoint | 510 GB: dense FP8, experts FP4, Engram FP8 |

1017 has the same architecture as the October 16 V4.1 release, but every weight shard changed, the Engram tables
included. A V4.1 quant is not a 1017 quant.

| Part | Parameters | Placement in dsx |
|---|---:|---|
| Routed experts | 544B (15,360 × 35.4M) | quantized; VRAM, RAM, a few on disk |
| Dense (attention, shared expert, routers, embeddings, head) | ~8B | GPU0, Q8_0 |
| Engram tables | 196B, 2 × 384M rows | disk, read per token (~48 rows of 264 bytes per table) |

## 2. Target hardware

| | |
|---|---|
| CPU | i7-12700K (8P + 4E), AVX2 + AVX-VNNI, no AVX-512 |
| RAM | 4 × 32 GB DDR4-3200, dual channel: 51 GB/s peak, **39.5 GB/s measured** |
| GPU0 | RTX 3090, PCIe 4.0 x16: 24.7 GB/s host-to-device from hugepage memory |
| GPU1 | RTX 3090, chipset slot, PCIe 3.0 x4: 3.3 GB/s |
| Disk | NVMe, ~6 GB/s sequential |

Two properties shape everything: RAM bandwidth is the slow tier, and GPU1's link is too slow to stream experts, so it
holds a fixed set. Where both GPUs sit on CPU lanes at the same speed, GPU1 becomes a second cache and a second prefill
stream (section 8c).

## 3. Performance model

**Decode.** Each token passes through 40 layers. The dense half runs on GPU0 every token (~14 ms on a 3090 after
optimization). The routed experts are 6 per layer, 8.5B parameters, ~2.1 GB per token at ~2 bits per weight. Each byte
costs according to where its expert lives:

| Tier | Bandwidth | Computed by |
|---|---:|---|
| VRAM | 936 GB/s | GPU |
| RAM | ~40 GB/s | CPU, in place |
| NVMe | ~6 GB/s | read, then CPU |

One NVMe read costs ~7× the same read from RAM, so the first rule is to keep every expert in VRAM or RAM. With two
24 GB GPUs and 128 GB of RAM that allows ~140 GB of experts: about 2 bits per weight.

**Routing is skewed on real text.** On the agentic calibration set, the hottest 10% of experts take 32-41% of routed
tokens, the hottest 25% take 55-61%, the hottest 50% ~80%, the coldest 10% 1.0-1.7%. Ranking VRAM residency by
hotness lets ~34 GB of VRAM serve over half of the traffic; RAM carries the rest.

**Prefill.** Every chunk of a long prompt uses nearly every expert, so the non-resident experts stream over GPU0's
PCIe link once per chunk (~100 GB at 24.7 GB/s ≈ 4 s). Larger chunks amortize it. Decoder layers only need the last
`(40 − L) × 128` tokens, so they run on that tail only.

**Long agentic contexts.** KV is small (117 MB at 128K), and attention reads a fixed window plus 512 rows, so long
context barely slows decode. A server that keeps each conversation's KV and only prefills new tokens makes follow-up
turns cheap.

## 4. Ecosystem (October 2026)

Runtimes that load V4.1 / 1017:

| Runtime | Status |
|---|---|
| llama.cpp | not merged ([PR #28696](https://github.com/ggml-org/llama.cpp/pull/28696)); forks run it |
| ik_llama.cpp | merged, with DSpark; open long-context issue |
| vLLM / SGLang | supported; no Ampere path |
| ExLlamaV3 (EXL3) | served through a vLLM-based kit, multi-GPU or DGX Spark |
| antirez/ds4 and derivatives | DS4 GGUFs on Mac, ROCm, CUDA |
| [Xolotl](https://github.com/Indras-Mirror/Xolotl) | one GPU + RAM + NVMe tiers, GSQ quant; ~26 tok/s on a 4090 with biased routing |

Published 1017 quants span 2-bit GGUFs (~150-375 GB with Engram), EXL3 2.8-3.5 bpw (196-240 GiB, perplexity within
0.4-1.8% of FP8), MLX, NVFP4 and VQ formats. Few publish quality numbers; few target 24 GB GPUs with 128 GB of RAM.

## 5. Engine results

Correctness: GPU decode matches a CPU reference implementation of DeepSeek's `model.py` to bf16 rounding (differences
only at router near-ties), and prefill matches token-by-token decode.

Decode on 2x RTX 3090 (3000-token agent prompt, exact routing):

| Change | Decode |
|---|---:|
| Baseline | 12.5 tok/s |
| One CPU per spinning thread (the CPU-expert coordinator shared a core with the GPU launch thread and saw routed picks up to 1.5 ms late) | 20.3 tok/s |
| Tiled split-K decode attention (276 → 21 µs per layer) | 31.5 tok/s |

Dense-half optimizations: repacked Q8 GEMV (22.6 → 10.6 ms per token), hyper-connection mixing (6.1 → 1.1 ms), partial
top-token selection instead of a full vocabulary sort, Engram row prefetch on its own threads (6.2 → 0.6 ms).

Prefill: a tensor-core attention kernel for long chunks (0.53 → 0.15 s per 4K chunk), deduplicated Engram reads, no
host stalls before the expert stream, and four staging buffers: 8K-token prefill 478 → 508 tok/s.

## 6. Expert placement follows the conversation

A VRAM set ranked on agentic text fits agentic text. Logging decode-time expert picks for three kinds of output:

| Text | Picks served from VRAM, static ranking | With an adaptive GPU0 cache | Best possible for the turn |
|---|---:|---:|---:|
| Prose essay | 15% | 68-72% | 90% |
| Python module | 43% | 73-74% | 84% |
| Agent / tool-call JSON | 47% | 78-79% | 86% |

GPU0's experts are therefore a cache over the RAM arena. After each token, the hottest non-resident experts (by
decayed usage, seeded from calibration counts) replace the coldest resident expert of the same quant type in place:
up to 8 per token (~72 MB over PCIe, overlapped with compute), committed between tokens. Short prose decode went from
26.0 to 29.9 tok/s on v1 with quality unchanged. The arena keeps copies of GPU0's experts (110 GB); the ~530 coldest
experts stay in the file.

## 7. Quantization

Each routed expert gets one of three tiers, chosen from its measured output error on held-out calibration text:

| Tier | Bits per weight | Grid |
|---|---:|---|
| PTQ1_0 | 1.75 | ternary, fp16 scale per 128 |
| Q2_0 | 2.25 | {-1, 0, 1, 2}, fp16 scale per 64 |
| Q3_K | 3.44 | 8 levels, 6-bit sub-scales per 16 |

Method: layer-by-layer calibration through DeepSeek's reference `model.py` on the FP8/FP4 checkpoint, GPTQ per
expert, then tier allocation by error per byte (Xolotl's GSQ pipeline). Calibration text: 128 × 2048 tokens of
coding-agent sessions and agent/code prompts through v2; from v3a, 160 × 2048 tokens with 53 sequences of the full
model's own reasoning traces. Dense weights are Q8_0; Engram stays unquantized.

What moves quality at a fixed size (Lucebox's 1017 eval, 8 × 1024 tokens against FP8 top-256 log-probs):

| Allocation, 148 GB | KL | Top-1 | Perplexity |
|---|---:|---:|---:|
| 2.0 bpw per layer (v1) | 0.462 | 71.5% | 11.66 |
| 2.0 bpw per layer, no limit on ternary | 0.463 | 71.8% | 11.93 |
| One budget for layers 2-36 (v2) | 0.395 | 74.1% | 10.93 |
| v2's tiers, calibration with reasoning traces (v3a) | 0.397 | 74.1% | 10.81 |
| + 128-block Hadamard rotation of every expert (v4) | 0.382 | 74.9% | 10.69 |
| + least-squares scale refit after GPTQ (v4b) | **0.373** | **74.4%** | **10.68** |
| Reference: 2.5 bpw (Lucebox ROCMFP) | 0.253 | 78.7% | 9.78 |
| Reference: Unsloth UD-IQ1_S (llama.cpp codebooks, ~122 GB without Engram) | 0.361 | 75.1% | 10.76 |
| Reference: Unsloth UD-IQ1_M (codebooks, ~140 GB without Engram) | 0.275 | 78.8% | 10.11 |

v3a changes what the model does more than these numbers show: its reasoning traces are ~1.2x those of a
near-lossless build on two hard math problems, against ~1.8x for v2. Rotation spreads each expert's outlier weights
before rounding (the engine rotates expert inputs to match); the refit re-solves each group's scale for GPTQ's codes
against the full Hessian, then GPTQ runs again. v4b's row was measured with the current prefill code, under which v4
scores KL 0.3816.

Two allocation experiments at v4b's size, measured on one rented RTX PRO 6000 next to v4b (KL 0.3746 / top-1 74.2% /
ppl 10.69 there): weighting each layer's error by its measured KL sensitivity (v5: noise injected per 5-layer group)
gave 0.3759 / 74.2% / 10.65, no gain; Unsloth UD-IQ1_M's layout in our formats (v6: layers 0-19 mostly ternary, more
bits in layers 20-39 and down projections) gave 0.400 / 73.7% / 11.08, worse. With our grids the early layers need the
bits; Unsloth's edge is in its experts themselves (below).

**A codebook tier (v7) and why UD-IQ1_M's experts win.** ggml's IQ2_XXS (8-weight groups from a 256-point codebook,
2.06 bpw) quantized by our GPTQ beats our scalar tiers by 18-22% in held-out output error at equal bits (layers 5, 20,
32, 39). As a fourth allocator tier (v7: 1,968 IQ2_XXS experts, none of the 3,238 VRAM-resident ones) it scored
0.3765 / 74.2% / 10.65 against v5's 0.3759. Its realized layer errors fell 5-18% in layers 3-19 but doubled in layer 32:
the screening compared IQ2_XXS against unrotated, unrefit scalar tiers, which overstates its advantage ~1.35x, and with
that correction the allocator keeps only ~480 IQ2_XXS experts for a 0.6% gain. The tier and its kernels stay available
(CPU 40.6 GB/s from RAM on the 12700K; GPU ~1.6x slower than Q2_0 on a 3090).

Scoring the same 44-48 experts per layer on Lucebox's own text (its 8 x 1024 tokens through the reference model)
showed what the calibration text hides. Token-weighted relative output error, calibration held-out → Lucebox:

| Layer | Our Q2_0, rotated + refit (2.25 bpw) | Our ternary (1.75) | UD-IQ1_M's experts |
|---:|---|---|---|
| 5 | 0.092 → 0.153 | 0.141 → 0.232 | 0.174 → 0.198 (1.73 bpw) |
| 20 | 0.157 → 0.232 | 0.235 → 0.345 | 0.156 → 0.154 (2.23 bpw) |
| 32 | 0.231 → 0.281 | 0.341 → 0.414 | 0.258 → 0.261 (2.15 bpw) |
| 39 | 0.102 → 0.161 | 0.153 → 0.239 | 0.129 → 0.146 (2.31 bpw) |

Our experts lose 1.2-1.7x from calibration text to general text (whole allocations up to 2.9x); UD's lose 0.99-1.14x
and beat ours on Lucebox text at equal bits. Ours also shrink their outputs: <ŷ, y>/<y, y> is 0.65-0.81 at 2 bits
(0.93-0.95 at Q3_K), against 0.92-0.99 for UD's, and the error points against the expert's mean output. Breaking the
pipeline down (layers 5 and 20): rounding without error feedback transfers perfectly but is poor and halves the
outputs (ternary shrink 0.49); GPTQ roughly halves the calibration error, and about a third of that gain survives on
Lucebox text; damping 10x transfers a little better at the same Lucebox error; Hessians from the corpus's web sequences
alone change nothing (< 2%). The held-out sequences share no documents with the fit ones but come from the same pools.
Correcting the shrink at run time helps only partly (a quarter-power gain: KL 0.373 → 0.365; the full 1/s: 0.384).

ggml's stock quantizers do not reproduce UD's experts: with UD's exact layout and either our imatrix or Unsloth's
published one they score 9-33% worse than UD on Lucebox text (about our level), and their IQ2_XXS blocks share only 6%
of the fp16 scales and 71% of the grid indices with UD's. UD's weight-space error is even slightly larger than stock
ggml's (L32 gate 0.131 vs 0.124), so their quantization optimizes something other than weight error. Loading UD's
actual experts into dsx with our dense half gives KL 0.2703 / top-1 79.0% / ppl 10.13, better than UD in llama.cpp.

Sharing the budget across layers gives sensitive layers (3-13, 17, 32) 2.2-2.75 bpw and most decoder layers ~1.76.
Separate tiers for down and gate/up projections did not help. v2 is also faster: its smaller decoder-layer experts let
~200 more experts fit in VRAM (agent 30.0 tok/s; a 37.2 tok/s prose run had only 10.6% of picks on the CPU, against
25-31% on typical prose).

Quantizer speed: GPTQ's column loop and the scale search run as native CUDA kernels batched over experts, the H^-1
factor comes from one Cholesky of the reversed Hessian, and integer codes rather than fp32 values go to the writer.
One layer's experts: 159 → 70 s; a full model: ~2.5 h → ~1.5 h on two 3090s.

## 8. Other hardware: RTX 5090 + 128 GB DDR5

| | Static placement, 80 GB arena | Adaptive, 75 GB arena |
|---|---:|---:|
| Decode, prose | 12.0 tok/s | 7.6 tok/s |
| Decode, 3K-token agent prompt | 15.2 tok/s | 7.8 tok/s |
| Prefill, 3K / 8K tokens | 93 / 172 tok/s | 68 / 114 tok/s |

With one 32 GB GPU and ~90 GB of usable RAM, ~40 GB of experts are read from disk through the page cache, and that
sets the speed. Static placement wins there: the adaptive cache's RAM copies of GPU0's experts cost space the machine
doesn't have. Such a machine needs ~160 GB of RAM, or a disk tier with batched direct reads and prefetch.

## 8b. Overlapping the CPU's experts with the GPU

Each decode layer alternates GPU attention (~0.35 ms) and CPU experts (~0.4 ms); RAM idles in the first, the GPU in
the second. Perfect overlap would cost max(~12.6 ms of RAM reads, ~14 ms of GPU work) per token instead of ~35 ms.

**Expert deferral** (`DSX_DEFER=R`, after KTransformers, SOSP '25): the CPU's experts that the router ranked R or lower
(0 = best of 6) are added one layer later, so the main GPU does not wait for them; the CPU computes the immediate ones
first. Outputs change. 2x RTX 3090, v4b, 3K agent prompt, decode-path eval (2 sequences):

| Deferred | Decode | KL | Top-1 |
|---|---:|---:|---:|
| none | 29.9 tok/s | 0.369 | 76.5% |
| ranks 3-5 (`DSX_DEFER=3`) | 36.3 tok/s | 0.395 | 75.2% |
| ranks 2-5 (`DSX_DEFER=2`) | 39.0 tok/s | 0.422 | 75.0% |
| ranks 1-5 | 37.6 tok/s | 0.466 | 72.7% |
| all | 36.8 tok/s | 0.537 | 70.9% |

Deferring more than ranks 2-5 is slower again: the CPU's deferred work for one layer then delays its immediate work for
the next.

Adding a layer's deferred experts straight into the residual streams with that layer's own output weights (instead of
into the next layer's expert output, which mixes them with the next layer's weights) recovers about half the quality:
ranks 3-5 deferred, KL 0.383 / top-1 75.7% at 35.9 tok/s; ranks 2-5, KL 0.394 / 75.7% at 36.3 tok/s. That is how
`DSX_DEFER` works now. A repeat on the 3K prompt (two alternating rounds) gave 29.8 / 29.9 tok/s without deferral,
36.5 / 36.3 with ranks 3-5 and 37.2 / 35.8 with ranks 2-5: deferring rank 2 as well is no faster within run-to-run
variance, so `--defer-rank 3` is the setting.

**Prefetch into L3** (exact): with the same 2 experts reused, the CPU kernels run 0.69 ms per 6-expert call (~87 GB/s)
against 1.30 ms from RAM (~48 GB/s). Predicting a layer's picks one layer early (its router on the previous layer's
expert input, `DSX_PREGATE=1`) finds 71% of the CPU's picks in a top-8; its router on its own attention input
(`DSX_PREGATE=2`) only 52-55%.
Acting on the one-layer-early prediction did not pay: reading the next layer's 1, 2 or 3 predicted CPU experts into
cache after each layer (one more pass over ~9 MB each, on the CPU worker) left decode at 29.3 tok/s with 1 and slowed it
to 27.4 and 25.1 tok/s with 2 and 3. The prefetch pass costs ~6.5 ms per token per expert and often delays the next
layer's real work; the code was dropped.

## 8c. Two GPUs on x16 links: GPU1 as a second cache and a second stream

On a host with both RTX 3090s on CPU lanes (i9-10900X / X299, PCIe 3.0 x16 each, `nvidia-smi topo` NODE, 156 GB
DDR4), `bench/pcie.cu` measures 12.4 GB/s host-to-device on each GPU, also with both copying at once (the reference
machine: 24.7 GB/s on GPU0, 3.3 on GPU1). The x4 rule "GPU1 holds a fixed set and carries only activations" no
longer applies, and during decode GPU1 idled at 0-44% while GPU0 ran at 94-100%.

Two changes give GPU1 work. Both follow the measured link (at least half the main GPU's bandwidth) and can be forced
with `DSX_AUX_ADAPTIVE` / `DSX_AUX_STREAM`, so an x4 host keeps the old behavior:

- **Adaptive GPU1 cache** (upstream's automatic placement): GPU1's 23 GB swap experts after every token like GPU0's
  cache. Its experts then need RAM copies: the arena grows from 114.5 GB (fixed GPU1, everything fits) to ~138 GB.
  With `--arena-gb 110`, 3,531 experts stay in the file and both prefill and decode lose (table below); with 135,
  346. dsx therefore keeps GPU1's set fixed when the arena would leave more than 5% of the expert bytes in the file.
- **Prefill streams over both links:** each expert no GPU holds goes to the GPU whose queued copies finish first, the
  main GPU's link counting twice. GPU1 also computes its larger resident set (2,181 experts against GPU0's ~1,100),
  and at 12.4 GB/s an expert's compute costs about as much as its copy. Warm 8K prefill with all streamed experts on
  GPU0 15.6 s; split by link 14.4-15.4 s; GPU0 weighted 1.5 / 2 / 2.5 / 3: 13.3-14.0 / 12.9-13.3 / 13.2-13.4 /
  13.5 s. The host enqueues both GPUs' expert work round-robin: queueing GPU1's whole layer first filled its stream's
  ~1K-entry launch queue and blocked the host while GPU0 idled (prefill 385 tok/s, slower than without the split).

Measured with `dsx gen` (one cold process per row: first prefill after load, then greedy decode) and with `dsx eval
--decode` on 8 x 1023 tokens of held-out code (the same tokens for every row, so decode speed compares placements on
identical text):

| | Prefill 64K | Prefill 3K | Decode, identical text | Perplexity (decode) |
|---|---:|---:|---:|---:|
| Before: GPU1 fixed, GPU0 streams alone, arena 110 | 441 tok/s | 273 tok/s | 25.8 tok/s | 2.305 |
| GPU1 fixed, prefill over both links, arena 110 | **505 tok/s** | – | 26.0 tok/s (arena 135) | 2.303 |
| GPU1 adaptive, prefill over both links, arena 110 | 443 tok/s | 169 tok/s | 21.6 tok/s | 2.310 |
| **GPU1 adaptive, prefill over both links, arena 135** | **503 tok/s (+14%)** | **335 tok/s (+23%)** | **28.8 tok/s (+11%)** | 2.318 |

The binary llama-swap ran before (upstream b4e19e0) measured 450 tok/s on 64K and 275 on 3K with the same flags.
Free-running decode after the prompts moved by +-2 tok/s between rows because each placement rounds differently and
generates different text; the teacher-forced column is the comparison. A bigger arena alone does not speed up decode
with a fixed GPU1 (25.8 vs 26.0 tok/s): the gain is GPU1's cache following the text (CPU share of picks 24-26% ->
17-22%).

Quality: prefill-path perplexity on the same 8K tokens is 2.313 without the split, 2.308 with it, 2.313 adaptive
(decode reference 2.307). `dsx cmp` against the CPU reference over 32 tokens: top-1 matches 28/32 (adaptive, arena
135) and 30/32 (before), median max logit difference 1.80 / 1.92. Moving an expert between GPUs changes the order of
the f32 partial sums, which moves an 8K prompt's final logits by up to ~4-5 (routing near-ties flip downstream), so
the split counts link speeds in quarters of the main GPU's: run-to-run noise in the startup measurement then does not
change the output.

Not split: with the expert stream halved, an 8K prefill spends ~4.4 s gathering Engram rows from disk on the host while
both GPUs wait (`DSX_PROF`), next to ~2.8 s of attention.

## 9. Negative results

- **Speculative decoding (DSpark).** A verify pass of 2 tokens costs 55 ms (1.5 decode steps), of 3 tokens ~70 ms,
  after moving verify onto decode kernels (from 101 / 109 ms). Most of what remains is the CPU's expert work, which
  grows with the union of the tokens' experts (20.9 distinct experts per layer for 6 tokens vs 6.0 for one). With the
  draft's expected acceptance on a 2-bit model it would gain ~0-5% at best; it pays where every expert sits in GPU
  memory (1.2-1.7x on the MI325X build, [FINDINGS-AMD.md](FINDINGS-AMD.md)).
- **Routing bias toward VRAM-resident experts.** +4 tok/s at a bias of 0.2, for 2 points of top-1. Off by default.
- **EXL2 / EXL3.** ExLlama keeps all weights in VRAM and has no RAM tier. EXL3's trellis codes would improve quality per
  bit for GPU-resident experts, but decoding them on the CPU is likely compute-bound.
- **Experts on disk.** Each token needs ~2 GB of expert weights; serving 10% of it from a 6 GB/s drive adds ~30 ms.
- **Swapping on the x4 GPU** (`DSX_AUX_ADAPTIVE=1`). Decode on a 3K agent prompt 28.8 → 31.0 tok/s, on prose 29.1 →
  22.0: evictable experts need RAM copies, GPU1's 23 GB of them push ~2,200 cold experts out of the arena onto disk.
  Off by default (slow links keep a fixed set).
- **Cache tuning.** Usage decay 0.90-0.995 (`DSX_DECAY`) and 16 instead of 8 swaps per token: all within ±1 tok/s.
  Hit rate on this machine is set by capacity, not by the eviction rule.
- **CPU expert speed.** The CPU kernels read 30-35 GB/s. The memory controller moves ~42.6 GB/s during a plain read test
  (DDR4-3200 Gear 1, but 4 dual-rank DIMMs run 2N command rate and slow refresh); the v5-era benchmarks also lost a few
  GB/s to a concurrent job. Pinning the expert pool to one thread per physical P-core (`DSX_CPU_LIST`) was 1-2% slower
  than the default 14 threads on 7 P-cores with their hyperthreads.

## 10. Open problems

- Quality per byte: KL 0.373 at 148 GB against Unsloth's 0.361 at ~122 GB (codebook formats, more bits in late layers
  and down projections), and the gap to 2.5 bpw (0.253) at a size that fits 128 GB + 48 GB.
- A disk tier for RAM-limited machines: batched direct reads, one read per expert, next-layer prefetch (as in
  [colibri](https://github.com/JustVugg/colibri)).
- Prefill: the expert stream runs at ~75% of the PCIe floor.
- The dense half (~14 ms per token) as decode's next floor: its weight GEMVs run at 740-860 GB/s (q_b 800, o_a 740, LM
  head 860), and ~1,000 small kernels take ~3 ms; gaps between kernels are only ~1.2 ms per token.
