# Quantization pipeline

The default model uses T16 v1, quantized with the patched ExLlamaV3 pipeline in `t16/`. See
[the format](../docs/t16-format.md), [experiments](../docs/t16.md) and [current results](../docs/current-state.md).
`t16/t16_qworker.py` fits expert matrices, `t16/t16_gguf.py` writes the v1 GGUF blobs, and `t16/t16_ref.py` inspects
and decodes them. Pod scripts in this directory create billable resources and are not part of ordinary inference;
old budgets and completed-run scripts are historical records, not standing authorization.

The sections below document the older dsx-2bit v4b scalar pipeline and its rental tools.

How [DeepSeek-V4.1-Flash-1017-dsx-2bit](https://huggingface.co/lxe/DeepSeek-V4.1-Flash-1017-dsx-2bit) was made. Built on the V4.1 tools of
[Xolotl](https://github.com/Indras-Mirror/Xolotl) (MIT, Niko1221 and the Strata contributors), commit `7375e305`.

The published dsx-2bit build is v4b; it is not the recommended T16 download. Each build changed one thing; its script runs it end to end on one machine (paths in the
scripts are the build machine's):

| Build | What changed | Script |
|---|---|---|
| v1 | per-layer tier allocation at 2.0 bpw | `calib_all.sh` (calibration), `quant_chain.sh` (GPTQ one layer behind it), `finish.sh` (GGUF) |
| v2 | one bit budget across layers 2-36 | `quant_chain_joint.sh`, `finish_v2.sh` |
| v3a | calibration text with the full model's reasoning traces (`traces/collect.py`), v2's tiers | `run_v3a.sh` (`quant_chain_v3.sh`) |
| v4 | experts quantized in a 128-block Hadamard-rotated basis (`v41_quant.py --rot`) | `run_v4.sh` |
| v4b | least-squares scale refit after GPTQ (`--refit`) | `run_v4b.sh` |

| File | |
|---|---|
| `xolotl-v41.patch` | our changes to Xolotl's `tools/v41`: load the 1017 checkpoint, preload the Engram rows a batch needs in one sequential sweep, a `--budget-bpw` allocator, GPU RTN option, corpus from local agent sessions |
| `v41/` | the calibration and quantization scripts (Xolotl-derived, with our changes: native GPTQ, `--rot`, `--refit`) |
| `native/` | CUDA kernels for GPTQ and the scale search (below) |
| `traces/collect.py` | the full model's reasoning traces for the calibration text (via OpenRouter) |
| `make_rank.py` | expert hotness from the calibration routing counts: `expert-rank.json`, used to put the hottest experts in VRAM |
| `experiments/quant_variants.py` | one-layer A/B of quantization variants on dumped experts (`V41_DUMP`) |
| `guard.sh`, `watchdog.sh` | every job in a memory-capped cgroup; a watchdog stops jobs on GPU Xid errors, low RAM or high temperatures |

Calibration text (v3a on): 160 x 2048 tokens of coding-agent sessions, antirez's agent and code prompts, and 53
sequences of the full model's reasoning traces on Omni-MATH and SuperGPQA questions (none from the evaluated rows).
The T16 v1 release's calibration set (calib4) is published in
[lxe/dsx-calibration](https://github.com/lxe/dsx-calibration), without its private agent-session rows.
v4b's GPTQ pass takes ~3 h on one RTX 3090, plus GGUF assembly and the eval.

```bash
git clone https://github.com/Indras-Mirror/Xolotl && cd Xolotl && git checkout 7375e305
git apply /path/to/dsx/quant/xolotl-v41.patch
# then copy quant/v41/*.py over tools/v41 and adapt the paths in run_v4b.sh
```

## Renting a GPU

`rent/` runs a build on a rented RunPod GPU, hydrated from Cloudflare R2 and Hugging Face. Credentials stay on the
machine that drives it: pods get presigned URLs.

1. Stage the inputs in R2 under `dsx/stage/` (quantizer code + `native/qkern.cu` as `quant2.tar.gz`, `calib3/`, the tier
   map, `ds41-engram.json`); `rent/r2.py put KEY` gives a URL to `curl -T` into.
2. `rent/pod.py up STATE` rents a pod (RTX PRO 6000 by default) and replaces one that has no ssh within 2 minutes.
3. `rent/hydrate.sh MANIFEST` on the pod: Python deps, the staged files (presigned URLs from `r2.py get`), the FP8
   checkpoint from Hugging Face (~2 GB/s), the native GPTQ library. About 4 minutes.
4. `rent/quant_job.sh NAME MAP` runs the GPTQ chain and writes the GGUF; `rent/eval_job.sh NAME` splits it and runs the
   Lucebox eval with dsx built on the pod.
5. `rent/mirror.py STATE DIR PREFIX` copies results (or the checkpoint) to R2 as presigned multipart uploads;
   `rent/pod.py down STATE` releases the pod.

On an RTX PRO 6000 a layer takes ~80 s (~3x an RTX 3090); a full build is about an hour.
Configuration (`~/.config/dsx/rent.env` or the environment): `R2_ACCOUNT`, `R2_BUCKET`, `R2_ENV` (file with the R2 keys)
and `RUNPOD_ENV` (file with `RUNPOD_API_KEY`).

## Native quantizer (faster runs)

`native/qkern.cu` (+ `native/build.sh`) replaces the two hot loops with CUDA kernels, batched over experts:
GPTQ (a warp per row, the 128-column block in registers, the H^-1 block in shared memory; the trailing update one
batched GEMM) and the static-group scale search (one pass, every candidate per group in registers).
`v41/v41_native.py` (ours) wraps them; `v41/v41_quant.py` + `v41/v41_calib.py` (Xolotl's files with our changes) queue
experts (8 or 160K routed tokens per batch) and quantize each batch in one call. `V41_NATIVE=0` keeps the old path.

| per expert (RTX 5090) | before | native |
|---|---:|---:|
| GPTQ gate+up / down | 73 / 31 ms | 6.9 / 2.5 ms |
| scale search | 3-49 ms | 0.7-9.7 ms |

Integer codes match the old path to >= 99.999% (rounding ties under FMA contraction), same H-weighted error.

## Credits

- `v41/v41_alloc.py`, `v41_calib.py`, `v41_formats.py`, `v41_gguf.py`, `v41_gptq.py`, `v41_gptq_graph.py`, `v41_quant.py`
  are derived from [Xolotl](https://github.com/Indras-Mirror/Xolotl)'s `tools/v41` (commit `7375e305`), and
  `v41_gguf_meta.py` and `v41/torch_kernels/kernel.py` are copied from it unchanged; MIT License, Copyright (c) 2026 Niko1221 and the Strata contributors
  ([`v41/LICENSE-Xolotl`](v41/LICENSE-Xolotl)); our changes are marked in the files and in `xolotl-v41.patch`.
- IQ2_XXS (the `iq2xxs` tier: `v41/v41_iq2.py`, `v41/v41_iqgrid.py`, the engine's kernels): the format and its codebook
  tables are ggml's ([ggml](https://github.com/ggml-org/ggml) / [llama.cpp](https://github.com/ggml-org/llama.cpp), MIT,
  Copyright (c) 2023-2026 The ggml authors; the notice is in `v41_iqgrid.py`). Codebook GPTQ (each group of 8 columns
  rounded jointly, error fed back column by column) follows GPTVQ (van Baalen et al.,
  [arXiv 2402.15319](https://arxiv.org/abs/2402.15319)), QuIP# (Tseng et al.,
  [arXiv 2402.04396](https://arxiv.org/abs/2402.04396)) and AQLM (Egiazarian et al.,
  [arXiv 2401.06118](https://arxiv.org/abs/2401.06118)).
- GPTQ: Frantar, Ashkboos, Hoefler, Alistarh, *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained
  Transformers* ([arXiv 2210.17323](https://arxiv.org/abs/2210.17323)).
- Hadamard rotation of expert inputs (`--rot`): incoherence processing as in QuIP# (Tseng et al.,
  [arXiv 2402.04396](https://arxiv.org/abs/2402.04396)) and QuaRot (Ashkboos et al., [arXiv 2404.00456](https://arxiv.org/abs/2404.00456)).
- The calibration forward runs DeepSeek's reference `inference/model.py` and kernels from the
  [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) checkpoint (MIT).
