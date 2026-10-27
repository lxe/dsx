# T4 — SM70 (Volta) prefill attention: replacing the `sparse_attn_tc*` stubs

Branch `t4-attention` (from e88a39d). Files: `nvidia/src/attn.rs`, `nvidia/src/kernels/dsx.cu`,
`nvidia/tests/gpu_attn_sm70.rs` (new harness). Numbers: [T4-RESULTS.md](T4-RESULTS.md).

## Decision: Option A — wire SM70 to the existing non-TC kernel (no wmma port)

**Evidence** (all verified by reading the code, then by building and running on the V100):

1. The non-TC reference kernel `sparse_attn` (dsx.cu, attention-decode section) is SM70-clean: bf16 loads
   converted to f32, fp32 FMA, `__shfl_xor_sync`, `expf` — grep finds no `cp.async` / `ldmatrix` / `mma` /
   bf16 arithmetic in it, and it sits outside every `__CUDA_ARCH__ >= 800` guard (only the `sparse_attn_tc*`
   machinery is guarded). It already ships in the SM70 fatbin because decode uses it, so it compiles for
   SM70 by construction.
2. `Gpu.arch` is `pub` (gpu.rs), so `attn.rs` can select at runtime without touching gpu.rs. `sparse_attn`
   has an *identical* 12-argument signature to `sparse_attn_tc` — the wiring is one launch-config branch:
   grid `(T, H/8)`, block 256, no dynamic smem (`nh % 32 == 0` already ensured, implying `nh % 8 == 0`).
3. The split-K pair (`sparse_attn_tile`/`sparse_attn_merge`, also SM70-clean) cannot serve prefill sizes:
   its partials buffer is `[T][S][H][hd+2]` f32 ≈ 2.7 GB at T = 4096, W+K = 640. Serial `sparse_attn`
   is the wired kernel.
4. The **only** in-tree launcher of `sparse_attn_tc*` is `SparseAttnTc` (attn.rs) — used by prefill
   (`nvidia/src/gpumodel/prefill.rs`, `n > 8` tokens) and `t16-prefill-audit`. One wiring point covers
   every caller. Decode (T <= 8) never uses this path (it has its own SM70-clean split-K kernels).
5. One trap found while wiring: `SparseAttnTc::new` raised the kernels' dynamic-smem limit to 99 KB, but
   V100's per-block ceiling is 96 KB — `cuFuncSetAttribute` would fail on Volta. The arch branch skips the
   raise (the serial kernel needs no dynamic smem).
6. Cost model said wiring would pass the gate before any GPU time was spent: `sparse_attn` measured 15.5
   TFLOP/s of a ~104.8 TFLOP/s fp32 peak on the 5090 (14.8%); naive peak-scaled expectation on V100
   (~14.1 TFLOP/s fp32 @ 1380 MHz) = 22.2 ms x 7.4 = **165 ms at T = 4096 = 40.3 µs/token**, vs the
   <= 10 ms/token budget gate of the 41 tok/s (24.4 ms/token) reference — three orders of margin.
   Measured (T4-RESULTS): **82.6 ms = 20.2 µs/token = 4.16 TFLOP/s**, i.e. **2.0x faster than the naive
   peak-scaled expectation** (the kernel runs at 29% of V100's fp32 peak, vs 14.8% on the 5090) — both the
   ">= 2x the naive expectation" reading and the "<= 10 ms/token" reading of the Option-A gate PASS.
   Even summed over all ~41 layer calls of a chunk it is 0.83 ms/token.

The wmma port (Option B) remains the documented upgrade path; it is *not* needed for the gate (see bottom).

## What changed

- `nvidia/src/attn.rs` — `SparseAttnTc` gains `serial: Option<Kernel>`: on `arch < 80` it holds
  `sparse_attn`, `new()` skips the smem-limit raise, and `launch()` routes to it (same arguments, grid
  `(T, H/8)`, block 256, smem 0). SM80+ behavior is byte-identical to before.
- `nvidia/src/kernels/dsx.cu`
  - `#include "sm70.cuh"` moved to the top of the include block (scaffold-review hardening; the SM70
    shims are now visible to every later kernel header — moe / expert_fast / t16 / other tasks' code).
  - The SM70 `sparse_attn_tc` / `sparse_attn_tc_fast` stub bodies now `__trap()` instead of silently
    writing nothing: the names must stay in the fatbin (export surface), and after the wiring nothing
    in-tree reaches them — so reaching one is a wiring bug you want to see loudly, not silent garbage.
- `nvidia/tests/gpu_attn_sm70.rs` (new) — accuracy (vs a plain fp32 CPU reference at |x| ~ 10, cosine
  >= 0.9999 / max-rel <= 2e-2 gates, full gathered-row semantics: ring window + compressed cache, -1
  skips, duplicate indices, all-masked token) and perf (`#[ignore]`d, T = 16..4096, the <= 10 ms/token
  gate) for whichever kernel the wiring picks on the host GPU.
- `sm70.cuh` intentionally untouched: the `dsx_cp16` signature stays as the scaffold left it (other
  tasks' in-flight code may call the 2-arg form; the `sz` zero-fill parameter belongs to the Option-B
  wmma port, which was not needed).

## Verification (all on the VM, GPU 0 under `/tmp/dsx-gpu.lock`)

- Dual-arch build gate: `CUDA_ARCHS=70` and `CUDA_ARCHS=86` `cargo build --release -p dsx --features
  cuda` both pass (CUDA 12.9).
- Accuracy: cos(min) = 1.000000, max rel 7.8e-3 (≈ the kernels' bf16 output rounding), all-masked token
  zeros. Existing `gpu_attn` suite (tc/tc_fast/serial vs f64) also passes on SM70.
- `compute-sanitizer --tool memcheck`: 0 errors.
- Perf: 20.2-34.3 µs/token over T = 16..4096 (gate <= 10 ms/token, PASS at 300-500x margin).

## When to upgrade to wmma (Option B sketch, for a later task)

Trigger: prefill throughput becomes attention-bound on V100 (attention share per chunk > ~20%). Today
attention is 82.6 ms of a T = 4096 chunk per layer call — small next to the GEMM-heavy rest at ~90
TFLOP/s cuBLAS fp16 (Phase-0 K3 = 94.5). The upgrade: implement `sparse_attn_tc_body` semantics in
sm70.cuh with `wmma::fragment<_,16,16,16,__half,_>` fp32-accumulate; plain row-major smem staging
(the SM80 XOR swizzle `R + r*512 + ((ch ^ (r&7))<<3)` is not `load_matrix_sync`-consumable), bf16->fp16
once at fill (`__float2half_rn(__bfloat162float(x))`), `dsx_cp16(dst, src, sz)` extended with cp.async's
zero-fill contract. Expected >= 19 TFLOP/s (40% of Phase-0 K2's 47.6) ≈ 4.4 µs/token — ~4.5x this
wiring. Pitfall on record from Phase-0: a shared A tile stored `[m][k]` is `matrix_a::row_major`, NOT
col_major — the wrong tag is memory-safe and silently wrong; CPU spot-check everything.
