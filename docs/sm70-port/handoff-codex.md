# Handoff: dsx SM70 (Volta/V100) Port — PLAN ONLY

**To:** Codex  **From:** ZCode session (2026-10-27)  **Repo:** `./dsx` (lxe/dsx @ HEAD, depth-1 clone)

## Your mission

Produce `../PLAN.md` — a phased engineering plan to port dsx's NVIDIA CUDA
kernels to **SM70 (Tesla V100)** so `DeepSeek-V4.1-Flash-1017-dsx-t16` can be
served on 2× V100-PCIE-32GB. **You are writing the plan, not the port.**
Where the plan calls for on-GPU measurements, mark the step `EXEC:VM` — the
orchestrator will run it and return numbers; design gates around that
round-trip (batch your measurement requests).

## Context (all verified this session)

- dsx is a Rust+CUDA engine: dense weights + KV + adaptive expert cache on a
  main GPU, routed experts computed on CPU from an RAM arena, OpenAI-API
  server. Reference machine: 2× RTX 3090, 41 tok/s decode, ~1008 tok/s
  prefill. Our target: **2× V100-PCIE-32GB (SM70, no NVLink, PIX to each
  other), 108 GB RAM (~99 usable), Xeon E5-2682 v4 (AVX2, no AVX-VNNI)**,
  CUDA 12.4 (host) / 12.9 (container) toolchains, driver 580.173.02.
- Build with `CUDA_ARCHS=70 cargo build --release -p dsx --features cuda`
  fails in `nvidia/src/kernels` with ptxas errors (full log excerpt below):
  `cp.async*` requires sm_80; `ldmatrix.m8n8` requires sm_75;
  `mma.sync…bf16 m16n8k16` requires sm_80.
- Kernel inventory (`nvidia/src/kernels/`, 5,535 lines total):
  `dsx.cu` 2,566 (hub) · `iq2grid.cuh` 1,040 · `expert_fast.cuh` 932 ·
  `t16.cuh` 525 · `topk.cuh` 276 · `dequant_iq8.cuh` 73 · `moe.cuh` 123.
  Feature sites: `bf16` 91 · `__nv_bfloat16` 49 · `mma` 27 · `cp_async` 2 ·
  `ldmatrix` 2 (grep counts; verify exact call sites yourself — some hits are
  comments/headers). 19 `asm volatile` blocks. No wmma/pipeline-API use;
  kernels are hand-written inline-PTX style. The `amd/` tree is a separate
  ROCm port — useful precedent for "second architecture" structure.
- Known host quirk: this VM hard-reset 4× under *sustained* dual-GPU vLLM
  load (boot transients / long prefill), while a lighter llama.cpp workload
  ran for hours. Unknown root cause (guest logs silent). Plan benchmarks
  with warmup ramps and single-GPU-first stages.

## The three walls (our assessment — verify and refine)

1. **bf16 numerics (biggest).** Volta tensor cores have no bf16. Options:
   fp16-MMA with a range strategy (bf16 exponent range ≈ fp32; fp16 max
   ±65504 — trellis/dequant values may overflow; per-tile/per-tensor scaling,
   or fp16-with-saturation + error analysis), or fp32 CUDA-core fallback
   (loses TC, ~10× math slowdown). The plan must pick a strategy per kernel
   with a stated accuracy acceptance test (e.g., max-abs and cos-sim vs an
   SM80 reference run on identical inputs — define the harness).
2. **MMA geometry.** SM70 has fp16 `wmma` (16×16×16 fragment API) and
   `mma.sync m8n8k4` PTX; no m16n8k8 (sm_75) or m16n8k16. All 27 sites need
   fragment-layout rewrites; register pressure will change (V100 64K
   regs/SM vs 3090 64K — same, but occupancy math differs at 21.1 vs
   35.6 SMs… V100-PCIE has 80 SMs at ~1.38 GHz, ~112 fp16-TC TFLOPS peak).
3. **cp.async (2 sites) / ldmatrix (2 sites).** Mechanical: register-staged
   copies; manual fragment loads. Cheap.

## Hard constraints

- **Do not break SM80+ builds.** All changes behind `#if __CUDA_ARCH__ >=
  800`-style guards; the SM70 path is additive and upstreamable. CI check =
  both `CUDA_ARCHS=70` and `CUDA_ARCHS=86` compile.
- **Benchmark-first**: Phase 0 must be a single-kernel T16 dequant-GEMM
  prototype on real V100 measuring achievable fp16-TC throughput vs the
  SM80 kernel, with a numeric go/no-go gate (state the threshold and the
  decision rule, e.g., "≥X% of SM80 kernel TFLOPS and cos-sim ≥ 0.999 on
  fixed inputs → proceed").
- **No CPU-side rewrites.** The Rust core, T16 file format, expert-ranking
  and arena logic stay untouched. Kernels + their launch configs only.
- Estimate honestly: include a perf model (roofline-ish) projecting decode
  tok/s on 2× V100 at the end state, with the top-3 uncertainty terms.
- Phases must each end in something compilable and testable; list exact
  verification commands (build + test + `EXEC:VM` benchmark steps).
- Read `DESIGN.md`, `FINDINGS.md`, `docs/placement.md`, `bench/` first —
  cite them where they matter. The author's tuning notes may already cover
  fallback shapes.

## Deliverable

`/home/dogda/Documents/dsx-sm70-port/PLAN.md` containing:
1. Refined inventory (exact call sites per file, after your own greps).
2. Chosen strategy per wall (numerics, MMA, async) with alternatives
   considered and rejection reasons.
3. Phased plan (Phase 0 = go/no-go prototype … final = full server boot +
  41→? tok/s acceptance), each phase: changes, files, verification, risk.
4. The accuracy-comparison harness design (how SM70 outputs get validated
   against SM80 references).
5. Perf model + projected tok/s with error bars.
6. Risk register with kill criteria (when to abandon the port).
7. Open questions for the orchestrator (batched, answerable without GPU).

## Build-failure excerpt (for orientation)

```
ptxas error: Feature 'cp.async' requires .target sm_80 or higher   (×14)
ptxas error: Feature 'cp.async.commit_group' requires sm_80        (×2)
ptxas error: Feature 'ldmatrix' requires .target sm_75 or higher
ptxas error: Modifier '.m8n8' requires .target sm_75 or higher
ptxas error: Feature '.bf16' requires .target sm_80 or higher
ptxas error: Feature '.m16n8k16' requires .target sm_80 or higher
```

## What happens after

The orchestrator (ZCode) executes your plan phase-by-phase on the VM
(builds, benchmarks, numeric checks) and returns measurements to unblock
gated steps. Model download (234 GiB) is deferred until after the Phase-0
gate passes. Target artifacts live in `/home/dogda/Documents/dsx-sm70-port/`.
