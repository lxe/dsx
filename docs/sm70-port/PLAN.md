# dsx SM70 (Volta / 2× V100) Port Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve `DeepSeek-V4.1-Flash-1017-dsx-t16` (234 GiB, 552B MoE) on the debianv100s VM's 2× Tesla V100-PCIE-32GB by porting dsx's CUDA kernels from SM80/86 to SM70 — additively, without breaking existing architectures.

**Architecture:** dsx decode (token generation, the throughput-critical path) is integer `dp4a` math that Volta already runs — it needs only a compile-time route away from its `cp.async` staging variant (the plain variant already exists in-tree, bit-identical). The real port work is the prefill side: one attention-tile pattern (cp.async pipeline + `ldmatrix` + bf16 `mma.sync m16n8k16`) and the int8-`mma` fused expert kernels, both replaced by SM70-native paths (staged copies / manual fragment loads / `wmma` fp16 / `dp4a` loops) behind `__CUDA_ARCH__` guards.

**Tech Stack:** CUDA C++ (nvcc 12.4 host / 12.9 container on VM), Rust (cargo, existing toolchain on VM), `nvcuda::wmma`, cuBLAS via cudarc (existing), python for numerics harness.

---

## Ground truth (verified 2026-10-27, repo @ HEAD depth-1)

Workspace: `/home/dogda/Documents/dsx-sm70-port/` (git repo; `dsx/` = upstream clone).
Executes on VM `dogda@192.168.2.216` (2× V100 SM70, CUDA 12.4 host toolchain at
`/usr/local/cuda/bin/nvcc`, Rust at `~/.cargo/bin`; SSH key auth). Build:
`CUDA_ARCHS=70 cargo build --release -p dsx --features cuda` from `dsx/` (needs
`PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH`).

**Kernel inventory** (`dsx/nvidia/src/kernels/`, 5,535 lines):

| file | lines | SM80 dependencies | verdict |
|---|---|---|---|
| `t16.cuh` | 525 | `cp.async` ×7 lines — only in `t16d_gemv_k` (L388-399). `t16d_gemv0_k` (L422-453) is the same math without cp.async, **already in-tree**, selected at runtime by `DSX_T16_GEMV=0` (`nvidia/src/experts.rs:687`) | near-free: guard + route |
| `expert_fast.cuh` | 932 | int8 `mma.sync m16n8k32/m16n8k16 s8.s8.s32` (L76, L89) — IMMA is SM75+ | dp4a loop replacement (same integer math) |
| `dsx.cu` | 2,566 | **one function**: helpers `ldsm4`/`ldsm4t`/`mma_bf16` (L541-561) + `sparse_attn_tc_body` (L563-756) with `cp.async` double-buffer, `ldmatrix` (L665,726,731), bf16 `mma m16n8k16` (L668-669,736-737); wrappers `sparse_attn_tc`/`sparse_attn_tc_fast` (L758-767). **All** cp.async/ldmatrix/mma call sites in dsx.cu live inside this one function. The other ~50 `bf16` grep hits are portable conversions (`bf16r`, `__bfloat162float`) | wmma-fp16 variant of this one body |
| `iq2grid.cuh` | 1,040 | (not yet enumerated — IQ2 quant kernels; **not needed for the T16 model**) | guard-only, port later if ever |
| `dequant_iq8.cuh`, `moe.cuh`, `topk.cuh` | 472 | none found by grep | verify compile, no port |
| `bf16r` (dsx.cu:20) | — | `__float2bfloat16` round-trip = **portable conversion**, works on SM70 | none |

**Machine facts:** V100-PCIE: 80 SMs @1.38 GHz, 112 fp16-TC TFLOPS peak, 64 dp4a
OPS/clock/SM (same dp4a throughput as 3090 per SM), no cp.async, no IMMA, no
bf16 arithmetic, `wmma` fp16 16×16×16 only. Host quirk: this VM hard-reset 4×
under sustained dual-GPU vLLM transients; ramp benchmarks gradually, single GPU
first. Reference perf (author, 2× 3090): 41 tok/s decode, ~1008 tok/s prefill.

## Strategy decisions (locked)

1. **Decode:** compile-time guard keeps `t16d_gemv_k` (cp.async) for ≥SM80; SM70
   builds route to `t16d_gemv0_k`. Bit-identical by construction (author's own
   comment). Runtime default flips by `__CUDA_ARCH__`-chosen kernel, no env var
   needed.
2. **Attention/dense bf16 tile GEMM:** SM70 gets a `wmma` fp16 implementation
   with fp32 accumulators (`wmma::fragment<matrix_a,16,16,16,half,row_major>`).
   Justification: attention values are softmax-bounded (|x| small); fp16 range
   is safe with fp32 accumulation; accuracy gate vs CPU fp32 reference (§
   accuracy harness) enforces it. Alternative rejected: fp32 FMA fallback
   (~10× slower, kills prefill).
3. **cp.async staging elsewhere:** register-staged `__ldg`→`st.shared` copies in
   the same buffer layout (SM70's "old way"); structure-preserving, costs
   registers not correctness.
4. **int8 IMMA (expert_fast):** dp4a loops (mathematically identical integer
   accumulation; V100 dp4a throughput is proven by the decode path).
5. **iq2grid:** excluded — the dsx-t16 model never instantiates IQ2 kernels.
   `#if` guards keep them out of SM70 compilation without porting.
6. **All changes additive:** every new path is `#if __CUDA_ARCH__ < 800` (or
   `>= 70 && < 80` where cleaner); SM86/120 build must stay bit-identical
   (CI gate: both arches compile; SM86 object hash unchanged is impractical —
   gate = compile + existing tests).

## File structure

- `dsx/nvidia/src/kernels/sm70.cuh` — NEW: shared SM70 helpers (staged-copy
  macros, wmma fragment fill from smem, arch dispatch macros).
- `dsx/nvidia/src/kernels/t16.cuh` — MODIFY: guard `t16d_gemv_k` body; arch-peg
  the launcher macros.
- `dsx/nvidia/src/kernels/expert_fast.cuh` — MODIFY: dp4a variant under guard.
- `dsx/nvidia/src/kernels/dsx.cu` — MODIFY: attention/dense tile pattern gains
  an SM70 branch; guards for iq2grid includes.
- `dsx/bench/sm70_micro.cu` — NEW: standalone Phase-0 micro-benchmarks (no
  cargo needed).
- `dsx/bench/sm70_accuracy.py` — NEW: numerics harness driver.
- Work happens on branch `sm70-port` in the `dsx-sm70-port` workspace repo
  (upstream clone stays pristine under `dsx/`; port branch lives in a second
  worktree `dsx-sm70/` to keep `dsx/` clean — `git -C dsx worktree add ../dsx-sm70 -b sm70-port`).

Accuracy reference: **CPU**, not SM80 (no Ampere box available). The repo's CPU
kernels (`core/src/t16.rs`, per `t16.cuh` header comment) compute the same
trellis math; harness compares GPU vs CPU fp32 with thresholds.

---

## Chunk 1: Phase 0 (go/no-go micro-benchmarks) + Phase 1 (dual-arch scaffold)

### Task 1: Phase-0 micro-benchmark file

**Files:**
- Create: `dsx/bench/sm70_micro.cu`

- [ ] **Step 1: Write `sm70_micro.cu`** — three standalone kernels, no dsx
  includes, compiled directly with nvcc:

```cuda
// sm70_micro.cu — Phase-0 go/no-go for the SM70 port. EXEC:VM.
// K1 dp4a-GEMV: the decode inner loop shape (t16d_gemv0 pattern, 8 dp4a/16B row).
// K2 wmma fp16 16x16x16 tile GEMM: the attention/dense replacement shape.
// K3 cuBLAS fp16 SGEMM: prefill dense fallback shape.
// Each: warmup 3, timed 20 iterations, prints GB/s and TFLOPS/op-s.
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>
using namespace nvcuda;
// ... (full ~220-line implementation: kernel bodies modeled on
// t16.cuh L422-453 for K1; the canonical wmma pipelined GEMM for K2;
// cublasHgemm call for K3; timing harness with cudaEvent; shapes:
// K1 k=7168,n=2048 (expert), K2 M=N=4096,K=7168, K3 M=8192,N=4096,K=7168)
```

The full file is written at execution time. K1 is modeled on the in-tree
`t16d_gemv0_k` (t16.cuh L422-453). K2 is written **from scratch** — the repo
contains no wmma code; use the CUDA toolkit sample
`samples/0_Introduction/simpleTensorCoreGEMM` (any CUDA 12.x install or
github.com/NVIDIA/cuda-samples) as the canonical wmma pipelined-GEMM pattern.
K3 is a plain `cublasHgemm` loop. No dsx includes, no external deps beyond
cublas.

- [ ] **Step 2: Compile locally for syntax (workstation has no nvcc — skip if absent)**

Run: `command -v nvcc || echo "compile on VM only"`

- [ ] **Step 3: EXEC:VM — compile and run on V100**

```bash
scp dsx/bench/sm70_micro.cu dogda@192.168.2.216:~/
ssh dogda@192.168.2.216 '/usr/local/cuda/bin/nvcc -O3 -arch=sm_70 \
  -lcublas ~/sm70_micro.cu -o ~/sm70_micro && ~/sm70_micro'
```
Expected: three result lines with throughput numbers, no errors.

- [ ] **Step 4: Record the three measured numbers** in
  `dsx-sm70-port/PHASE0-RESULTS.md` **regardless of outcome**, then evaluate
  the gate:

Gate (all three must hold, measured on GPU 0 only, `CUDA_VISIBLE_DEVICES=<gpu0-uuid>`):
- K1 dp4a-GEMV ≥ **35 GB/s effective weight-stream** per kernel instance
  (decode streams ~66 MB of expert trellis bytes per token step —
  6 experts × 3 matrices × ~3.7 MB at k=7168,n=2048,k2=4; 35 GB/s ⇒ the
  expert math is not the wall).
- K2 wmma GEMM ≥ **25 TFLOPS** fp16 (≥22% of V100 peak ⇒ attention tiles
  stay viable; 3090 reference bf16 path ≈ 45% peak).
- K3 cuBLAS fp16 ≥ **60 TFLOPS** (V100 should do ~90).
If any fails → STOP; append the kill note to PHASE0-RESULTS.md and escalate
(kill criterion K1 in risk register).

- [ ] **Step 5: Commit results**

```bash
cd /home/dogda/Documents/dsx-sm70-port && git add PHASE0-RESULTS.md && \
  git commit -m "docs: phase-0 sm70 micro-bench results"
```

### Task 2: SM70 worktree + guard scaffold

**Files:**
- Create: `dsx-sm70/` (worktree, branch `sm70-port`)
- Create: `dsx/nvidia/src/kernels/sm70.cuh` (in worktree path)
- Modify: `dsx/nvidia/build.rs:24` (arch default unchanged; no edit needed — CUDA_ARCHS env is the input)

- [ ] **Step 1: Create the worktree**

```bash
cd /home/dogda/Documents/dsx-sm70-port/dsx && git worktree add ../dsx-sm70 -b sm70-port
```
Expected: `../dsx-sm70` populated, branch `sm70-port` created.

- [ ] **Step 2: Write `sm70.cuh`**

```cuda
// sm70.cuh — Volta (SM70) compatibility for dsx kernels. Everything here is
// additive: included unconditionally, but every SM70-only definition is used
// from __CUDA_ARCH__-guarded call sites, so SM80+ builds never instantiate it.
#pragma once
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
#define DSX_SM70 1
// Staged 16-byte global->shared copy (cp.async replacement): same destination
// layout, synchronous. `dst` is a shared-memory uint32 address, `src` global.
__device__ __forceinline__ void dsx_cp16(void *dst, const void *src) {
  const uint4 v = *(const uint4 *)src;
  *(uint4 *)dst = v;
}
#endif
```

- [ ] **Step 3: Compile-gate check — SM70 fatbin with kernels stubbed**

Temporarily wrap each SM80-only construct so SM70 compiles. Per file:
`t16.cuh`: preprocessor directives cannot appear inside a `#define` body, so
duplicate the launcher macro pair: wrap the existing `#define T16D_GEMV(K2)`
block in `#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800`, and add an
`#else` copy whose macro body calls `t16d_gemv0_k<K2>(...)` (the bit-identical
pre-cp.async kernel), so SM70 builds instantiate the plain kernel under the
same exported name `t16d_gemv_{k2}`. Guard the body of `t16d_gemv_k`
(L347-418) itself with the same `#if` so its cp.async asm never reaches the
SM70 compiler.
`dsx.cu`: guard the helpers (L541-561) and `sparse_attn_tc_body` (L563-756)
with `#if __CUDA_ARCH__ >= 800`, and give the two `extern "C"` wrappers
(L758-767) an SM70 branch whose kernel body writes nothing and sets a device
error flag (temporary — replaced in Phase 3; goal is only dual-arch compile).
You cannot `#if`-out the middle of a function — the unit of guarding is the
whole body + wrappers, exactly as scoped above.
`iq2grid.cuh` + `expert_fast.cuh` mma sites: same temporary stub treatment.
**Do not delete any SM80 code.**

- [ ] **Step 4: EXEC:VM — dual-arch compile gate**

```bash
rsync -a --delete /home/dogda/Documents/dsx-sm70-port/dsx-sm70/ \
  dogda@192.168.2.216:dsx-sm70/    # VM has no rsync server: use
  # tar czf /tmp/wt.tar.gz -C dsx-sm70 . && scp + tar xzf instead
ssh dogda@192.168.2.216 'cd dsx-sm70 && PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH \
  CUDA_ARCHS=70 cargo build --release -p dsx --features cuda && \
  CUDA_ARCHS=86 cargo build --release -p dsx --features cuda && echo DUAL-ARCH-OK'
```
Expected: both compile clean. (`CUDA_ARCHS=86` compiles fine on the VM —
ptxas for sm_86 works under CUDA 12.4 even without an 86 GPU present.)

- [ ] **Step 5: Commit**

```bash
git -C dsx-sm70 add -A && git -C dsx-sm70 commit -m "feat(sm70): arch guards + stubs, dual-arch compile gate"
```

---

## Chunk 2: Phase 2 (decode live) + Phase 3 (prefill port) + Phase 4 (integration)

### Task 3: Decode path live on SM70 — bit-exactness + throughput

**Files:**
- Modify: `dsx-sm70/nvidia/src/kernels/t16.cuh` (from Task 2's route)
- Modify: `dsx-sm70/nvidia/src/experts.rs:687` (auto-select, no env needed)
- Create: `dsx-sm70/bench/sm70_decode_check.py`

- [ ] **Step 1: Runtime auto-select in `experts.rs`** — `gpu.rs` already
  stores the device arch (`gpu.rs:93`, populated at init); add
  `pub fn is_sm70(&self) -> bool` reading that field (do NOT probe again), and
  in `experts.rs:687` change
  `let old = std::env::var("DSX_T16_GEMV").is_ok_and(|v| v == "0");` to
  `let old = gpu.is_sm70() || env_override;` (env override preserved).

- [ ] **Step 2: Correctness test** — `sm70_decode_check.py` builds a random
  T16 tensor (k2=4, k=512, n=128: random trellis bytes + suh/svh + random x),
  runs the CPU decode GEMV reference (**`nvidia/src/cpu/t16k.rs`**, functions
  `gemv`/`gemv_multi` — NOT core/src/t16.rs, which is the stored-layout
  prefill reference) and the GPU decode chain
  (`t16d_prep → t16d_gemv(0) → t16d_post_*`), and compares the final `part`
  floats with rel-tol **1e-6** (the affine float correction
  `0.009567260742f*acc − 2.4375f*xs` is applied inside the kernel, so there is
  no separable integer output; bit-exactness is not claimed). The python
  script drives both sides through a tiny `cargo` bin modeled on the existing
  `nvidia/src/bin/t16-prefill-audit.rs` (prints CPU and GPU outputs as JSON).
  **Layout warning:** the CPU reference walks the stored byte layout while the
  GPU expert store is the repacked layout (t16.cuh header comment) — the
  harness must either generate trellis bytes directly in repacked order or
  run the repack transform (`repack_t16` kernel / core `from_exl3`), or a
  layout mixup will masquerade as a guard failure.
- [ ] **Step 3: EXEC:VM — run the check**

Expected: PASS exact / PASS relative. Failure → the guard leaked a cp.async
path; fix before proceeding.

- [ ] **Step 4: EXEC:VM — synthetic decode throughput probe** — reuse
  `bench/` (author's `exl3_gpubench.py` pattern) with a synthetic T16 expert
  table sized like the real model (K=6, E=384/layer, k=7168, n=2048);
  measure expert-step ms on ONE V100. Gate: expert-step ≤ 90 ms (⇒ ≥ 11 tok/s
  decode ceiling before attention/dense; see perf model).

- [ ] **Step 5: Commit** — `feat(sm70): decode routed + verified bit-exact`.

### Task 4: Attention/dense tile pattern → wmma fp16 (the real work)

**Files:**
- Modify: `dsx-sm70/nvidia/src/kernels/dsx.cu:541-767` (helpers, body, wrappers)
- Modify: `dsx-sm70/nvidia/src/kernels/sm70.cuh` (grow the wmma helpers)

- [ ] **Step 1: Confirm scope** — `grep -n "cp.async\|ldmatrix\|mma.sync" dsx.cu`
  : every hit must be inside `sparse_attn_tc_body` (L563-756) or its helpers
  (L541-561). If any hit falls OUTSIDE that range, add it to the port list;
  as inventoried there are no other instantiation sites in dsx.cu.
- [ ] **Step 2: Write the SM70 attention body in `sm70.cuh`** — an SM70
  reimplementation of `sparse_attn_tc_body<FASTMATH>` semantics using
  `wmma::fragment<matrix_a/b, 16,16,16, __half, row/col_major>` with fp32
  accumulators. **The SM80 XOR-swizzled smem chunk layout
  (`R + r*512 + ((ch ^ (r&7)) << 3)`, dsx.cu L631/665) is NOT reusable**:
  `wmma::load_matrix_sync(ptr, ldm)` needs a plain row-major layout. The SM70
  path defines its own unswizzled smem staging (accept bank conflicts, or pad
  the stride by +8 halves; measure both, keep the faster). bf16→fp16
  conversion happens once at smem fill:
  `__float2half_rn(__bfloat162float(x))`. The q/p split-into-bf16-hi/lo trick
  (L536 comment) becomes plain fp16 fragments (fp16 range holds attention
  magnitudes; the Step-4 accuracy gate enforces it). cp.async staging →
  `dsx_cp16` synchronous copies in the same double-buffer rhythm (the
  `issue()` lambda structure, t16.cuh L384-392, is the model).
- [ ] **Step 3: Instantiate under `#if DSX_SM70`** at each site from Step 1;
  delete the Task-2 stubs for these kernels only.
- [ ] **Step 4: Accuracy harness** — `sm70_accuracy.py`: random Q/K/V at
  realistic magnitudes; run the SM70 kernel vs a pure-fp32 CPU attention
  reference (same tiling, naive loops); thresholds: cosine ≥ **0.9999**,
  max-rel ≤ **2e-2** at |x|~O(10). Also run the full kernel under
  `compute-sanitizer --tool memcheck` on the VM (EXEC:VM).
- [ ] **Step 5: EXEC:VM — accuracy + memcheck + perf**

Gate: accuracy thresholds pass; memcheck clean; attention prefill tile
throughput ≥ 40% of the K2 micro-benchmark number.

- [ ] **Step 6: Commit** — `feat(sm70): wmma-fp16 attention/dense tiles`.

### Task 5: expert_fast int8-MMA → dp4a

**Files:**
- Modify: `dsx-sm70/nvidia/src/kernels/expert_fast.cuh:70-95` (the two IMMA
  sites) — SM70 branch: replace each `mma.sync...s8` warp tile with an
  unrolled `dp4a` accumulation over the same 16×32 row block the fragment
  held (registers per lane: 16 int8 B-values × k-tile; the file's own ABL
  ablation machinery shows the surrounding loop structure). Integer sums are
  order-free ⇒ bit-identical accumulators.
- Steps mirror Task 4: implement under guard → correctness via
  `dsx-sm70/bench/sm70_expert_check.py` (same cargo-bin driver pattern as
  Task 3; integer accumulators are order-free, so expect max-abs **0** vs a
  CPU int32 reference loop) → EXEC:VM perf:
  `CUDA_VISIBLE_DEVICES=<gpu0> ~/sm70_expert_bench` (built from
  `bench/sm70_expert_bench.cu`, expert shapes k=7168 n=2048 T=8) with gate
  ≥ 60% of the IMMA path's op/s if quoted in FINDINGS.md, else ≥ 0.5 × the
  K1 micro-benchmark number → commit `feat(sm70): dp4a expert_fast`.

### Task 6: Integration + model download + acceptance

- [ ] **Step 1: EXEC:VM — full build, both arches, `cargo test -p dsx`** (the
  repo's tests; MockEngine tests are arch-neutral, must stay green).
- [ ] **Step 2: Download the model** (gated on Phases 0-3 all green):
  `lxe/DeepSeek-V4.1-Flash-1017-dsx-t16` (234.3 GiB + 0.9 vision + ranks) to
  `/home/dogda/models/` (~1.5-2 h; SHA256SUMS verify).
- [ ] **Step 3: EXEC:VM — boot `dsx-server`** (single GPU first:
  `DSX_GPUS=0 ./target/release/dsx-server --model <t16.gguf> --rank
  <expert-rank.json> --main-gb=-2.0 --arena-gb 80 --ctx 65536`; ramp: 8K → 64K
  context prompts, watching for the host-reset signature — if the VM resets,
  drop arena 10 GB and retry once; two resets = kill criterion K4).
- [ ] **Step 4: Acceptance numbers** (two GPUs, `--ctx 131072 --arena-gb 85`):
  decode tok/s (gate ≥ **10**, stretch 20), prefill tok/s (gate ≥ **150**),
  8K-prompt chat TTFT. Record in `dsx-sm70-port/RESULTS.md`.
- [ ] **Step 5: Commit + upstream prep** — branch summary, `git diff main
  --stat`, note for an upstream PR (guards make every hunk additive).

## Perf model (2× V100, decode)

Decode step ≈ attention(non-MoE layers, wmma fp16) + 6 experts×(trellis stream
+ dp4a) + dense. Expert trellis bytes/token ≈ 6 experts × 3 matrices ×
(k/16 × n/16 × 16 × k2 = 3.67 MB at k=7168,n=2048,k2=4) ≈ **66 MB** streamed
from the RAM arena over PCIe per token (~12 GB/s/direction ⇒ ~6 ms); dp4a
math at the K1-measured rate ⇒ budget 15-30 ms; attention+dense ≈ 10-20 ms ⇒
**~15-25 tok/s projected**, uncertainty dominated by (a) PCIe arena streaming
vs the 3090's faster host links, (b) wmma attention efficiency, (c) host-reset
risk under sustained load.

## Risk register / kill criteria

- **K1 Phase-0 gate fails** (dp4a or wmma numbers) → stop; write findings.
- **K2 accuracy** unachievable (cos < 0.999 after two fix iterations) → stop;
  fp32-FMA fallback evaluation is a separate decision.
- **K3 expert-step > 90 ms** after one optimization pass → decode ceiling
  < 11 tok/s; stop and report.
- **K4 two VM host resets** during Phase 4 boots/tests → freeze GPU work,
  deliver plan-state + findings; hardware diagnosis first (PSU suspect).
- **K5 upstream rebase conflict** (repo moves fast) → pin current HEAD; note
  divergence; port continues on the pinned tree.

## Open questions (answerable without GPU)

1. Does `dsx-server` require the vision tower loaded even with `--no-vision`?
   (Affects download size only.)
2. Confirm arena streaming uses pinned host memory (check `core/src` allocator)
   — affects the PCIe term of the perf model.
3. Does the dense/attention path already have a cuBLAS fallback selectable at
   runtime (gpu.rs exposes cublas) that could pre-empt Task 4 for prefill?
