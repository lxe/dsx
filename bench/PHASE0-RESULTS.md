# Phase-0 SM70 micro-benchmark results (go/no-go gate)

Date: 2026-10-27. Source: `dsx/bench/sm70_micro.cu` (standalone; CUDA runtime + wmma + cuBLAS only).

## Environment

- Host: a 2x V100-PCIE-32GB host (`ssh <v100-host-user>@<v100-host>`), driver 580.173.02
- GPU: Tesla V100-PCIE-32GB (UUID `<gpu0-uuid>`), 80 SMs, max SM clock 1380 MHz, HBM2 @ 877 MHz (single GPU, exclusive use, run under `/tmp/dsx-gpu.lock`)
- Toolchain: `/usr/local/cuda/bin/nvcc`, CUDA **12.9** (V12.9.86) — note: the plan said CUDA 12.4; the box now runs 12.9. Build: `nvcc -O3 -arch=sm_70 -o sm70_micro sm70_micro.cu -lcublas`
- Protocol: one correctness launch (CPU-checked) + 3 warmup + 20 timed iterations per benchmark, `cudaEvent` timing around the timed loop; K1 cycles a 272 MB pool (74 copies) of weight tables so L2 (6 MB) never serves a repeat, same protocol as `dsx/bench/t16_gemv.cu`.

## Results

| Benchmark | Shape | Result | Gate | Verdict |
|---|---|---|---|---|
| K1 dp4a trellis GEMV (t16d_gemv0_k pattern, K2=4, S=14) | k=7168, n=2048 | **207.8 GB/s** weight-stream (17.66 us/pass, 3.67 MB packed table/pass), check maxrel 1.6e-07 | >= 35 GB/s | **PASS** (5.9x) |
| K2 wmma fp16 GEMM (hand-written, 128x128x32 tiles, double-buffered smem) | M=N=4096, K=7168 | **47.60 TFLOPS** (5.053 ms/call), check maxrel 8.9e-06 | >= 25 TFLOPS | **PASS** (1.9x) |
| K3 cuBLAS fp16 GEMM (cublasHgemm, TENSOR_OP math) | M=8192, N=4096, K=7168 | **94.52 TFLOPS** (5.089 ms/call), check maxrel 2.0e-02 | >= 60 TFLOPS | **PASS** (1.6x) |

```
GATE: PASS
```

All three items pass; the Volta port proceeds.

Raw output of the official run:

```
Tesla V100-PCIE-32GB: 80 SMs, 34.1 GB global, 1380 MHz max clock
K1 dp4a trellis GEMV k=7168 n=2048 K2=4 S=14:    17.66 us/pass    207.8 GB/s weight-stream (3.67 MB/pass, pool 74 copies)  check maxrel 1.6e-07
K2 wmma fp16 GEMM 4096x4096x7168:    5.053 ms/call    47.60 TFLOPS  check maxrel 8.9e-06
K3 cublas fp16 GEMM 8192x4096x7168:  5.089 ms/call    94.52 TFLOPS  check maxrel 2.0e-02
GATE: K1 PASS (207.8 GB/s vs 35) K2 PASS (47.60 TFLOPS vs 25) K3 PASS (94.52 TFLOPS vs 60)
```

## Interpretation

- **K1** measures the decode inner-loop shape at 207.8 GB/s of trellis weight-stream (~23% of the ~900 GB/s HBM2 peak). The kernel is instruction-bound (per 16 weights and lane: 16 SHF + 16 IMAD + 8 PRMT + 8 IDP4A plus loads), not bandwidth-bound; the reference RTX 3090 kernel in `dsx/bench/t16_gemv.cu` reached ~536 GB/s on a 936 GB/s part (57%), so there is headroom on V100 via the four-accumulator-chain variant (`t16d_gemv_k`'s `ac[4]`) and wider stripes, neither of which is needed for the gate. The CPU check validates the exact integer path (funnel-shift windows, `*0x3A2F`, `byte_perm 0x5410`, `dp4a.u32.s32`, per-group scale/sum combine) against `t16d_gemv0_k`'s semantics.
- **K2** is a from-scratch wmma pipeline (the repo has no wmma code): 47.6 TFLOPS = ~42% of the clock-adjusted dense fp16 TC peak (~112.7 TFLOPS at 1380 MHz). Ample for the port's hand kernels; cuBLAS-level scheduling (multi-stage, swizzled smem) is what the remaining 2.4x gap is made of.
- **K3** cuBLAS reaches 94.5 TFLOPS = ~84% of peak at these shapes, confirming fp16 tensor-core GEMM is in healthy shape on this card/driver, and that fp16 prefill math can lean on cuBLAS directly.

## Bugs found while writing the reference (worth knowing for the port tasks)

- `wmma::load_matrix_sync` layout tags describe the *memory* layout: a shared A tile stored `[m][k]` (row stride between m) is `matrix_a::row_major`, not `col_major`. With the wrong tag the loads stay in-bounds (memcheck clean) and the GEMM silently produces wrong numbers (~5x rel error). The CPU spot-check caught it.
- sm_70 has no `cp.async`; the double buffer must stage global loads through registers (as implemented in K2).

## Reproduce

```
scp dsx/bench/sm70_micro.cu <v100-host-user>@<v100-host>:
ssh <v100-host-user>@<v100-host> '/usr/local/cuda/bin/nvcc -O3 -arch=sm_70 -Wno-deprecated-gpu-targets -o sm70_micro sm70_micro.cu -lcublas'
ssh <v100-host-user>@<v100-host> 'flock /tmp/dsx-gpu.lock env CUDA_VISIBLE_DEVICES=<gpu0-uuid> ./sm70_micro'
```

Optionally `./sm70_micro k1|k2|k3` runs a single benchmark.
PHASE0-RESULTS.md: measured 2026-10-27 on a 2x V100-PCIE-32GB host (2x V100-PCIE-32GB); source harness: dsx/bench/sm70_micro.cu in the outer workspace (untracked upstream); numbers anchor T3/T4/T5 gates.
