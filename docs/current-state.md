# NVIDIA runtime and measurement status

The default desktop model is T16 v1: DeepSeek-V4.1-Flash-1017 with T16 routed experts, Q8_0 dense weights and an
IQ4_NL Engram lookup table in one GGUF. The October 23, 2026 NVIDIA kernel integration is af2c154; 76a5d0a records
its serving configuration. The October 25 prefill checkpoint adds opt-in kernels and buffer reuse, with an 8K-chunk serving profile.
This page separates those runtime measurements from model-release quality scores and unfinished proposals.

## Model package

| Component | Shipped representation |
|---|---|
| Routed experts | T16, types 1010/1011/1012; each matrix is `trellis \| suh \| svh` |
| Bit layout | gate/up/down 1.5/1.5/2 in layers 0–19; 2/2/2.5 in layers 20–39 |
| Quantizer | patched ExLlamaV3, Hadamard-128, column-major tile positions, sigma_reg=2 |
| Output gains | gate/up ×1.13 at K1.5 and ×1.04 at K2, baked into `svh`; chosen on the reported eval |
| Dense weights | Q8_0; separate F16 Engram projection matrices remain as shipped |
| Engram lookup table | IQ4_NL; the `q6k-engram` model branch provides Q6_K instead |
| Tokenizer | embedded in GGUF metadata |
| Vision | optional `vision.safetensors` beside the model |
| Placement ranking | `expert-rank.json`, passed through `--rank` |

The default GGUF is 251,592,261,632 bytes (~252 decimal GB). The Q6_K alternative is ~302 GB in four shards.
These file layouts were checked against the Hugging Face model API on October 24. The T16 model revision was
`8bf76fe667734aff5c4d7fcd88ed855a3191b244` at that check; future downloads can change independently of this repo.

The [format specification](t16-format.md) describes the v1 byte layout and a proposed container v2 with separate
scale tensors. The current loader and converter implement v1, not that proposal. T16 model release v2 refers to the
calib5 requant experiment, which is a separate version number. It completed but scored KL ~0.202 against v1's 0.196
with the same gains; it did not replace the default model. No improved agentic quality was established.

## Runtime defaults

`dsx-server` defaults to a 32K context, chunk size 4096, arena budget 0, and deferral off. Its two-3090 quick start
explicitly selects 128K context, a 110 GB arena and `--defer-rank 3`. The recorded reference serving configuration
used those values, `--main-gb=-1.5`, `--aux-gb=-1.0`, and no experimental environment overrides.
This is a recorded benchmark configuration, not an inventory of live services. For example, later coding runs
used a 256K context; changing `--ctx` changes available expert-cache memory and must accompany any timing claim.

| Control | Default | Meaning |
|---|---|---|
| `DSX_T16_GEMV` | staged kernel; `0` selects old kernel | cp.async T16 expert GEMV |
| `DSX_Q8R1`, `DSX_SK1` | enabled; `0` disables | single-token Q8 and split-K variants |
| `DSX_COMBINE_FUSED`, `DSX_FUSE2`, `DSX_FUSE3` | enabled; `0` disables | combine, dense/shared and attention epilogue fusions |
| `DSX_AUX_LATE` | enabled; `0` disables | enqueue aux GPU after main GPU layer-0 attention |
| `DSX_QB_ROPE` | off; `1` enables | experimental fused q_b rotation; inexact on sm_86 |
| `DSX_PAR_ATTN`, `DSX_FUSE4` | off; `1` enables | parallel attention branch and closing hc_post fusion; below the gain threshold |
| `DSX_ADAPT_EARLY` | off; `1` enables | earlier cache update; measured worse residency |
| `DSX_ENGRAM_FP8` | off; `1` enables | convert F16 Engram **projections** to exact FP8 weights at load |
| `DSX_ENGRAM_PERSIST` | off; `1` enables | persistent lookup-row gather threads; marginal measured gain |
| `DSX_ENGRAM_PREFETCH` | enabled; `0` disables | prefill gathers every Engram layer's lookup rows for the chunk on a helper thread at chunk start; [receipts](../bench/results/2026-10-26-engram-prefetch/README.md) |
| `DSX_T16_WARP_HAD`, `DSX_T16_SPECIALIZED_DQ` | off; `1` enables | exact prefill Hadamard and weight-expansion variants |
| `DSX_T16_FUSED_POST_GU` | off; `1` enables | combine T16 gate/up post-processing with SwiGLU |
| `DSX_PREFILL_REUSE_BUFFERS` | off; `1` enables | share prefill query/output/scratch storage; saves ~1.91 GB at 8K |
| `DSX_PREFILL_CPU` | enabled; `0` disables | CPU assistance for short chunks |
| `DSX_PREFILL_CPU_MAX_N` | 512 | maximum chunk size eligible for CPU assistance |
| `DSX_ADAPT` | 8 | maximum adaptive expert replacements per token; `0` fixes residency |

The lookup table and the projection matrices are different weights. Keeping F16 projections does not mean the
252 GB package stores the large lookup table in F16: that table remains IQ4_NL.

The reference hardware has an RTX 3090 on a ~24.6 GB/s x16 host link and another on a ~3.3 GB/s x4 link. The x4
card keeps a fixed expert set. The main GPU adapts its cache, and the CPU computes its assigned experts from RAM
while the GPUs work. [Placement](placement.md) explains backing copies, capacities and multi-GPU controls.

## Measurement provenance

All numbers below are measurements from their named configuration, not promises for a fresh server or other hardware.
The reference machine is two RTX 3090s, an i7-12700K and 128 GB DDR4. Sizes are decimal GB.

| Result | Build and workload | Source |
|---|---|---|
| KL 0.196, top-1 80.9%, perplexity 9.54 | T16 v1, IQ4_NL lookup table, prefill-path Lucebox 8×1024, deferral off | [model card](../cards/dsx-t16/README.md#quality) |
| KL 0.194, top-1 81.4% | same expert build, Q6_K lookup table, prefill-path eval | model card |
| 34.9 tok/s; 36.8 with defer=3 | earlier T16 release build, free generation after 3K prompt, 300 new tokens; GPU limits 250/280 W | [model card speed table](../cards/dsx-t16/README.md#speed) |
| 1,007.84 tok/s prefill | October 25 checkpoint, 8K prompt/chunk, 256K allocation, fixed T16, F16 Engram projections, all four new prefill flags; two warm rounds, zero reused KV tokens | [prefill receipts](../bench/results/2026-10-25-prefill/README.md) |
| ~529 tok/s prefill | 3K prompt after vectorized T16 dequant and hugepage arena fixes | [FINDINGS §8d](../FINDINGS.md#8d-where-the-time-goes-on-102-2026-10-22-t16-iq4_nl-build) |
| 33.86 → 37.03 tok/s (+9.4%) | old base vs integrated kernels **with optional FP8 projections**, teacher-forced 1×1023 decode steps, defer=3, adapt=8; means of two interleaved rounds | [FINDINGS §9b](../FINDINGS.md#9b-decode-small-kernels-launch-structure-and-the-dev-4-merge-2026-10-23), windows 30–34 |
| 36.7 → 41.0 tok/s | free generation, 120 after 300 tokens; candidate also includes optional FP8 projections; different generated text | FINDINGS §9b |
| 31.63 tok/s | final default af2c154, F16 projections, 128K context, 38-token cold serving reply | recorded serving smoke check; not a sustained A/B |

The prefill checkpoint's initial 8K pass measured 928.49 tok/s; subsequent fresh-prefix passes measured
1,008.50 and 1,007.18 tok/s. The model state resets between rounds, so warm-up refers to loaded kernels and weights,
not reused prompt KV. Its 84 kernel cases and 129,280 integration logits match bit for bit. Earlier fixed-placement
checks cover 3K prompts and 8/1-token additions. These checks establish agreement for the tested changes and inputs;
they do not resolve the older decode-quality check below.

The integrated maximum above includes an option the default configuration leaves off. It is not a newly measured
37–41 tok/s guarantee for that default. A 128K context also leaves fewer experts on the main GPU than the 32K gen
benchmarks, and the adaptive cache takes time to follow a new conversation.

Free-running numeric changes alter generated text, routing and CPU pick share, so their tok/s cannot isolate a
kernel's speed. Use fixed teacher tokens, interleaved/order-swapped runs, matching power caps and repeat statistics.
Exactness comparisons additionally require `DSX_ADAPT=0` and equal resident expert counts. Changing GPU placement
can change rounding even when weight bytes are unchanged. Cooler 8-thread quality checks are not production-thread
timing measurements.

## Quality and the open decode check

The published 0.196 quality figure is a prefill-path result with deferral off. Its output gains were selected on the
same eval, so it is not an independent quality estimate. Later decode checks use different sequence counts,
residency and deferral settings and must be reported separately.

A T16 fixed-token decode comparison on 2046 positions measured KL ~0.208 without deferral, 0.237 at rank 3,
0.260 at rank 2 and 0.306 at rank 1. Ranks 2/1 failed the additional +0.01 KL gate and remain rejected. Rank 3 was
retained for the reference serving configuration. Deferral keeps top-6 routing but injects low-ranked CPU outputs
one layer late, changing the residual computation; it is an approximation, even with exact expert selection.

[Window 35](../bench/results/2026-10-22-102/win35.log) fixed adaptation and equalized GPU0 residency at 1568 experts.
Base KL was 0.160876; the integrated arm was 0.170395, with top-1 0.832845 → 0.822092. FP8 and F16 projection arms
at equal residency gave the same result. Thus that window did not attribute the shift to FP8 storage.

The fused q_b rotation differed by one ulp in ~700/4096 rotated values on sm_86 despite a matching sm_120 bench.
It is disabled by default. The planned default-flags comparison and individual T16 GEMV isolation were not completed
before the final serving switch. The final deployment report kept the KL caveat open. Disabling q_b fusion must
not be presented as proof that the entire default integration matches the base numerics.

The original FP8 projection kernel had a separate deterministic KL change from summation order. Its replacement,
`gemv_fp8b32_o`, preserves F16 arithmetic order; window 35 confirmed equal F16/FP8 KL at equal placement. That fix
is merged, but optional conversion remains off. Persistent gather threads removed the join wait but added only
~0.49% above FP8, near the kill threshold, so they remain off too.

## Recorded negative results

- Next-layer expert prefetch (`DSX_PREFETCH`) reduced CPU work but slowed decode: 33.1 → 25.3/22.2 tok/s at one/two
  copies per layer, or 37.4 → 28.3/20.7 with defer=3. DRAM reads, repack work, wasted copies and GPU0 contention
  outweighed the saved CPU work. [Budget analysis](budget-102.md#4-dsx_prefetch-against-this-budget-negative-result).
- Desktop DSpark verify passes of 2/3/4 tokens cost 51/69/77 ms versus ~33 ms for one plain step. The union of CPU
  experts grows with the window; the all-GPU AMD gains do not transfer to this desktop result.
- Early cache update worsened the CPU pick share from ~23% to ~26%. The proposed commit-timing diagnosis was not
  confirmed by its rework. The cause remains open and the option remains off.
- Forcing the x4 GPU adaptive can evict arena coverage onto disk and hurt prose speed. Decay/swap-count sweeps did
  not produce a useful consistent gain. The measured configuration keeps the x4 store fixed.

## Remaining work

The first correctness task is the default-flags, fixed-residency comparison against the old base on sm_86, followed
by individual kernel isolation. It requires a coordinated hardware window; the scripts in the results directory
stop the server and should not be launched as unattended cleanup.

Subsequent proposals include profiling the adaptive-cache scan, combining T16 gate/up launch work and improving
multi-conversation prefix reuse. Within one conversation the server now keeps a bounded history of prompt-boundary
KV snapshots (8) and rewinds to the newest checkpoint at or before the longest common prefix, so tool calls and
retries avoid a full 200K+ token re-prefill. That is intra-conversation recovery, not a multi-conversation cache;
the latter remains a proposal. Lower-bit dense weights are estimates, not implemented gains. Requantization is
not currently scheduled. The gpu-unlock proposal was dropped. Old budget tables retain those proposals to explain
their assumptions, not to designate current jobs.

Before a benchmark, read actual power and thermal limits. The recorded October 23 window used 230 W GPU limits
and a 125/180 W CPU cap. An October 24 read-only check found the CPU long-term limit at ~4095.9 W, not 125 W, with
short-term still 180 W. The watchdog has previously stopped jobs at CPU ≥95°C. Large pinned allocations can also
kill serving through memory pressure. Do not assume old protection settings still apply.

## Documentation map

| Document | Scope |
|---|---|
| [README](../README.md) | download, quick start, supported hardware and API |
| [Server configuration](server.md) | server flags, sampling and thinking controls |
| [Coding agents](agents.md) | Pi provider setup and response limits |
| [DESIGN](../DESIGN.md) | current NVIDIA execution and storage architecture |
| [placement](placement.md) | current residency accounting and tuning flags; dated older sweeps |
| [T16 format](t16-format.md) | shipped v1 bytes; clearly separated v2 proposal |
| [T16 experiments](t16.md) | chronological codebook, layout and real-expert research |
| [FINDINGS](../FINDINGS.md) | measured development history, with current summary and later integration |
| [budget](budget-102.md), [inflight analysis](t16-decode-inflight.md) | dated resource models and intermediate kernel profiles |
| [bench results](../bench/results/2026-10-22-102/README.md) | raw window scripts and receipts; not all large traces are in Git |
| [quantization tools](../quant/README.md) | current T16 entry points and historical scalar pipeline |
| [model release history](model-history.md) | package-specific release measurements; older models retained for reproducibility |
| [model cards](../cards/dsx-t16/README.md) | model-release metrics; local copies require separate publication to update the Hub |
| [AMD runner](../amd/README.md), [batched log](../amd/RUNNER.md) | separate HIP implementation and completed rental measurements |

Historical raw logs, third-party vendored docs and gauntlet task definitions are retained as records. They are not
rewritten to look like current measurements. Public model cards and GitHub files change only when these edits are
published; a local documentation commit does not update either service.
