# T5: expert_fast int8-IMMA kernels -> dp4a on SM70 (V100)

Date: 2026-10-27. Branch `t5-expert-dp4a` (on scaffold e88a39d). Worktree `dsx-sm70-t5`.
Host: debianv100s (`ssh dogda@192.168.2.216`), Tesla V100-PCIE-32GB (80 SMs, driver 580.173.02), CUDA 12.9
(`/usr/local/cuda/bin/nvcc`), all GPU runs under `flock /tmp/dsx-gpu.lock`, GPU UUID
`GPU-4e4e3d86-f648-3706-6489-f5e8e17fcab8`.

## What changed (`nvidia/src/kernels/expert_fast.cuh` only)

- The SM70 `mma32`/`mma16` placeholder bodies now `__trap()` (was: junk XOR arithmetic): a hypothetical future
  caller fails loudly on device instead of silently computing garbage. Signatures unchanged.
- New `mma32t<DP>`/`mma16t<DP>` templates: DP = false calls the free IMMA functions (SM80+ path, unchanged
  semantics incl. ABL); DP = true replaces the m16n8k32 / m16n8k16 `.s8.s8.s32` IMMA with unrolled `dp4a` over the
  same logical tile. The fragments already hold the mma layouts — lane 4g+q owns A rows g/g+8 at k = 4q+i and
  16+4q+i and x of token g at the same k — so the emulation is pure operand exchange plus integer dot products:
  per k-slice u, 4 quad-shuffles gather A row g's four slices (lanes `base|u`) and 4 shuffles gather x of tokens
  2q/2q+1 (lanes `8q+u`, `8q+4+u`), then 16 (8 for mma16) `__dp4a` accumulate the same products the mma would sum.
  Integer sums are order-free, so the accumulators are bit-identical to the IMMA's. mma16t keeps mma16's
  reset-the-accumulator contract. Per mma32: 8 SHFL + 32 dp4a per lane; no smem, no extra syncs (tiles only reach
  the mma in warp-uniform control flow).
- The `tile_*` templates and `tile_dot` take a `bool DP` parameter (default false); every `mma32`/`mma16` call site
  routes through `mma32t<DP>`/`mma16t<DP>`. Decode/fixup code is byte-identical between the arches.
- The three kernels (`xgemv`, `moe_gu_fast`, `moe_down_fast`) share one body each via `*_k<DP>` device functions;
  the SM80 `#if` branch instantiates `<false>` (IMMA), the SM70 `#else` branch `<true>` (dp4a). extern "C" names,
  signatures and `__launch_bounds__` unchanged — `gpu.rs`/`cuModuleGetFunction` loading is unaffected.

## Bench harness (new files under bench/)

- `bench/t5_kernels.cu`: the expert_fast path compiled exactly as dsx.cu builds it (`bench/t5.sh` uses the same
  nvcc flags as `nvidia/build.rs`, sm_70), plus a `t5_rp_weight` dump kernel (rp_weight over a whole repacked
  matrix) and a `t5_tables` dump kernel (the ggml codebook tables for the host decoders).
- `bench/t5_bench.cu`: loads the fatbin via the driver API (as gpu.rs does) and carries independent host GGUF
  decoders for all 8 formats (checked against rp_weight first), plus the launch shapes from experts.rs.
- Correctness protocol: all scales dyadic (f16/f32 powers of two, x int8 in [-16,15] so every sum of products
  stays < 2^24) — the entire fp32 pipeline is then exact and comparisons are bit-exact against a double-precision
  host reference. Run: `bash bench/t5.sh [correct|perf|all]` on the VM.

## Correctness (max-abs; expect exactly 0 on the integer path)

xgemv, all 8 repacked formats, three shapes each (single tile/x-block; odd rows + T=3 + KS-split x-blocks; expert
shape rows=2048 k=7168 T=8): **rp_weight-vs-host-decoder maxabs 0 and y maxabs 0, bad cells 0** for every format
and shape. Repacked matrices come from the on-device `repack_expert` kernel, x from the quant_act layout built
host-side (permuted q, per-256 dyadic d, exact per-64 sums).

moe_gu_fast + moe_down_fast (T=8, K=6, E=16 with 6 resident, non-resident picks, plan built as quant_act builds
it, dim=512, inter=256, per format):

- h (gu): max error over all picked (token, pick) within **1 bf16 ulp** (93-100% of cells bit-exact; the rest are
  1-ulp expf rounding differences, expected: silu is not dyadic). Non-picked cells keep their sentinel (untouched).
- out (down): **maxabs 0** vs the host sum (pure adds of dyadics). The `plan[0] == 0` path zeroes out exactly.

Three consecutive full-suite runs: PASS, PASS, PASS (deterministic).

## Perf (expert shapes k=7168, n=2048, T=8; 3 warmup + 20 timed, cudaEvent; xgemv cycles a >= 280 MB weight pool so
L2 never serves a repeat)

Weight-streaming role (moe_gu_fast over 48 distinct resident experts, gate+up; moe_down_fast, down; dim=7168
inter=2048):

| format | gate/up GB/s | down GB/s |
|---|---|---|
| Q2_0    | 359.7 | 300.3 |
| PTQ1_0  | 261.6 | 220.2 |
| Q3_K    | 402.2 | 385.7 |
| Q8_0    | 645.2 | 689.9 |
| IQ2_XXS | 206.6 | 189.0 |
| IQ1_S   | 172.9 | 147.0 |
| IQ2_XS  | 250.0 | 205.3 |
| IQ2_S   | 233.3 | 197.9 |

**GATE: PASS — worst 147.0 GB/s vs the required >= 104 GB/s** (0.5 x Phase-0 K1's 207.8), 1.4x the gate at the
slowest format, 6.6x at Q8 (689.9 GB/s, ~77% of the ~900 GB/s HBM2 peak).

xgemv (single-matrix test kernel; at the expert shape it runs only ceil(2048/16) = 128 blocks of 64 threads, so it
is occupancy-limited exactly as on the 82-SM SM80 target): 49-330 GB/s at rows=2048; the same kernel at
rows=16384 (1024 blocks) streams 117-606 GB/s (Q2 274, PTQ1_0 201, Q3_K 336, Q8 606, IQ2_XXS 155, IQ1_S 118,
IQ2_XS 169, IQ2_S 174). The production role never launches it at this shape.

## Dual-arch gate

`nvcc --fatbin -std=c++17 -O3 --prec-div=true --prec-sqrt=true -gencode=arch=compute_70,code=sm_70` and the same
for `compute_86` on `nvidia/src/kernels/dsx.cu` (what `CUDA_ARCHS=70` / `=86` feed through `nvidia/build.rs`):
**both build clean**. The bench fatbin also builds for 70+86 in one shot. (cargo-level build: see commit message /
report note.)

## Reproduce (on the VM, repo root = ~/dsx-sm70-t5)

```
bash bench/t5.sh correct   # correctness suite (also 'perf' or 'all'; T5_FATBIN overrides the fatbin path)
```

## Debugging notes (harness bugs found and fixed during bring-up, worth keeping in mind)

- The first correctness runs failed at k >= 512 with exact 2x/4x multiples of per-warp partials. That was a bench
  bug, not a kernel bug: the harness only filled the first per-256 x-block scale of each activation vector (the
  reference applied the scale everywhere), and an earlier `t5_tables` variant wrote 128 bytes past the ksigns dump
  buffer (found with compute-sanitizer memcheck; it could smear later allocations). After fixing both, plus a
  missing `& 3` in the host IQ2_XXS decoder and a `cuLaunchKernel` param-array slot (the pool pointer must be
  passed by address), every check is bit-exact. The kernel port itself was correct from the first build; the
  dp4a emulation was additionally validated by a fragment-layout unit test (all 256 accumulator words of a warp,
  repeated calls) before the end-to-end suite.
