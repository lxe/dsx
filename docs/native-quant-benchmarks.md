# Native Q2_K / IQ3_XXS CPU kernel A/B measurements

These are **CPU matrix-vector kernel timings on a virtual AMD EPYC test host**,
not end-to-end model tokens/s, not CUDA timings and not measurements of the
user's i9-10900X/CMP 170HX workstation. Lower time is better.

| Format | Matrix (rows × columns) | Matrix pool | Before (ms) | After (ms) | Speedup | Time saved |
|---|---|---:|---:|---:|---:|---:|
| Q2K | 2304 × 5120 | 1 | 0.827 | 0.390 | **2.12×** | 52.8% |
| Q2K | 2304 × 5120 | 32 | 0.817 | 0.385 | **2.12×** | 52.9% |
| Q2K | 5120 × 2304 | 1 | 0.828 | 0.385 | **2.15×** | 53.4% |
| Q2K | 5120 × 2304 | 32 | 0.823 | 0.383 | **2.15×** | 53.4% |
| Iq3Xxs | 2304 × 5120 | 1 | 4.739 | 1.567 | **3.03×** | 66.9% |
| Iq3Xxs | 2304 × 5120 | 32 | 4.739 | 1.577 | **3.00×** | 66.7% |
| Iq3Xxs | 5120 × 2304 | 1 | 4.559 | 1.538 | **2.96×** | 66.3% |
| Iq3Xxs | 5120 × 2304 | 32 | 4.608 | 1.608 | **2.87×** | 65.1% |

## Method and scope

- Baseline: `e9838cc1a614d768884fe518cae5be08d4cce5f5`, the format-support commit
  with the original scalar byte-unpack loops inside the SIMD kernels.
- Optimized: `c6ccf4e49a3395cf366f6f0eb2245d60a4caa267`.
- Both use the **same** `native-qdot-bench.rs` from the optimized commit. Upstream
  main is not the baseline because it has no kernels for these formats.
- Rust 1.99.1 (58cf8f9c3 2026-10-13), release profile, committed Cargo.lock,
  repository `-C target-cpu=native`, no CUDA feature.
- AMD EPYC 9V74 VM; one thread pinned to logical CPU 0 with `taskset -c 0`;
  `DSX_CPU_ISA=avx512vnni`; T=1. No CPU power/frequency settings changed.
- Run order A-B-A-B. Each run warms every matrix, then takes five timing
  samples, each traversing the pool three times. Each run reports its median;
  the table is the median of the two run medians for each variant.
- Timed region excludes allocation and activation quantization. Outputs are
  consumed with black_box. Synthetic finite inputs; pools of 1 and 32 matrices.
- Speedup = before/after; time saved = 1 - after/before. Nominal weight GB/s in
  raw output is not measured DRAM bandwidth. VM scheduling and CPU caches
  affect results; two rounds do not establish statistical confidence bounds.
- These measurements do not predict proportional full-model speedups. GPU,
  routing, activation quantization, synchronization and memory-placement costs
  remain outside the measured region. i9 and GPU/full-model A/B remain open.

## Reproduction

Build separate baseline and optimized worktrees with the identical benchmark
source (copy it from the optimized commit into the baseline worktree). Keep
both executables, then alternate:

```sh
DSX_CPU_ISA=avx512vnni taskset -c 0 /path/to/baseline/native-qdot-bench 3
DSX_CPU_ISA=avx512vnni taskset -c 0 /path/to/optimized/native-qdot-bench 3
# Repeat A then B once more; choose an allowed CPU on your host.
```

## Raw runs

### Run 1: baseline

```text
ISA=Vnni512; one thread, T=1; activation quantization and allocation excluded
format,rows,cols,pool,repeats,median_ms_per_matrix,nominal_weight_GB_s
Q2K,2304,5120,1,3,0.797731,4.8522
Q2K,2304,5120,32,3,0.798741,4.8460
Q2K,5120,2304,1,3,0.806915,4.7969
Q2K,5120,2304,32,3,0.824408,4.6952
Iq3Xxs,2304,5120,1,3,4.656438,0.9698
Iq3Xxs,2304,5120,32,3,4.608181,0.9800
Iq3Xxs,5120,2304,1,3,4.592416,0.9833
Iq3Xxs,5120,2304,32,3,4.600125,0.9817
cpu kernels: Vnni512
```

### Run 2: optimized

```text
ISA=Vnni512; one thread, T=1; activation quantization and allocation excluded
format,rows,cols,pool,repeats,median_ms_per_matrix,nominal_weight_GB_s
Q2K,2304,5120,1,3,0.375272,10.3144
Q2K,2304,5120,32,3,0.365712,10.5841
Q2K,5120,2304,1,3,0.369082,10.4874
Q2K,5120,2304,32,3,0.380722,10.1668
Iq3Xxs,2304,5120,1,3,1.570457,2.8755
Iq3Xxs,2304,5120,32,3,1.603228,2.8167
Iq3Xxs,5120,2304,1,3,1.505095,3.0004
Iq3Xxs,5120,2304,32,3,1.630666,2.7693
cpu kernels: Vnni512
```

### Run 3: baseline

```text
ISA=Vnni512; one thread, T=1; activation quantization and allocation excluded
format,rows,cols,pool,repeats,median_ms_per_matrix,nominal_weight_GB_s
Q2K,2304,5120,1,3,0.857017,4.5165
Q2K,2304,5120,32,3,0.834987,4.6357
Q2K,5120,2304,1,3,0.848193,4.5635
Q2K,5120,2304,32,3,0.821585,4.7113
Iq3Xxs,2304,5120,1,3,4.822479,0.9364
Iq3Xxs,2304,5120,32,3,4.869413,0.9274
Iq3Xxs,5120,2304,1,3,4.524703,0.9980
Iq3Xxs,5120,2304,32,3,4.616514,0.9782
cpu kernels: Vnni512
```

### Run 4: optimized

```text
ISA=Vnni512; one thread, T=1; activation quantization and allocation excluded
format,rows,cols,pool,repeats,median_ms_per_matrix,nominal_weight_GB_s
Q2K,2304,5120,1,3,0.405527,9.5449
Q2K,2304,5120,32,3,0.404217,9.5758
Q2K,5120,2304,1,3,0.401421,9.6425
Q2K,5120,2304,32,3,0.385978,10.0283
Iq3Xxs,2304,5120,1,3,1.562833,2.8895
Iq3Xxs,2304,5120,32,3,1.551408,2.9108
Iq3Xxs,5120,2304,1,3,1.570794,2.8749
Iq3Xxs,5120,2304,32,3,1.586187,2.8470
cpu kernels: Vnni512
```

## Executable identities

- baseline: `137375485a6200e00f7891be0c2861942507caf06063d9c93346aff086650663`
- optimized: `b4b3d1b9f50ab7587c73bbb513615481c4eb6765a0bbac33d7e8ebf37ac71f7a`
