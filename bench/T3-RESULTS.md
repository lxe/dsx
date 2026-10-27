# T3: decode path live on SM70 — runtime auto-select, correctness check, throughput probe

Date: 2026-10-27. Host: `debianv100s` (`ssh dogda@192.168.2.216`), Tesla V100-PCIE-32GB
(UUID `GPU-4e4e3d86-f648-3706-6489-f5e8e17fcab8`, 80 SMs, driver 580.173.02), CUDA 12.9
(`/usr/local/cuda/bin`), Rust at `~/.cargo/bin`. Worktree `~/dsx-sm70-t3`, fatbin built with
`CUDA_ARCHS=70`; every GPU run under `flock /tmp/dsx-gpu.lock`.

## Runtime auto-select

- `nvidia/src/gpu.rs`: `Gpu::is_sm70()` reads the arch stored at context init (`pub arch`), no new probes.
- `nvidia/src/experts.rs` `run_t16_on`: `old = g.is_sm70() || DSX_T16_GEMV == 0` — on Volta the plain
  `t16d_gemv0_*` kernels are picked at run time (the SM70 fatbin exports `t16d_gemv_*` over the same plain body
  per the compile-time guard, so this is belt and suspenders for multi-arch fatbins); `DSX_T16_GEMV=0` still
  forces them for A/Bs.
- Verified in passing: the probe reports `"sm70": true` from the new accessor, and the check bin confirms
  `t16d_gemv_4` and `t16d_gemv0_4` outputs are bit-identical on this fatbin/device.

## Correctness (`t16-decode-check`, EXEC:VM) — PASS

CPU reference: `nvidia/src/cpu/t16k.rs` (`prep`/`gemv`/`post_block`, AVX2 kernels on this host), synthetic experts
generated in the stored trellis layout, repacked on the host into the GPU store layout, launched exactly as
`run_t16_on` launches them (`t16d_prep` -> `t16d_gemv_4` -> `t16d_post_gu` / `t16d_post_down`).

| Check | Result | Gate |
|---|---|---|
| host repack vs `repack_t16` kernel (k2 = 3, 4, 5; 512x128, 512x512) | bit-exact | exact |
| part, task shape k2=4 k=512 n=128 (T=1, K=1): CPU pipeline vs GPU | maxrel **2.87e-7** | 1e-6 |
| GEMV only, identical activations (scalar reference) | maxrel **3.83e-7** | 1e-6 |
| S=4 k-split sum vs whole-range part | maxrel **3.83e-7** | 1e-6 |
| int8 codes / group scales / group sums (GPU prep vs CPU prep) | 0 / 1 low-bit scale diff / 0 | info |
| post_gu h (bf16-rounded SwiGLU) | maxrel **0.0** (bit-equal) | 1e-2 |
| full 8-launch chain (dim=inter=512, Sgu=4, Sd=3): down part | maxrel **0.0** | 1e-6 |
| full chain final out | maxrel **4.87e-7** | 1e-2 |

The integer dp4a accumulators are identical (the layout mixup the task warns about would have shown up here as
~1.0 rel error); the residual few-1e-7 is float affine/FMA ordering plus one group scale differing in its last
mantissa bit (Hadamard tree order), which flipped no int8 code. Max rel diff overall: **3.83e-7**.

Post-review hardening re-run (every GPU download gated on all-finite, per prefill-bench's convention; identical
seed, so identical numbers): part 2.874e-7 / 3.832e-7 / 3.832e-7 / h 0.0, full-chain down 0.0, out 4.873e-7,
`h_bits_mismatch` 0 — PASS.

## Throughput (`t16-decode-probe`, EXEC:VM) — PASS

One synthetic decode expert step: T=8 tokens, K=6 picks (48 distinct experts of E=384 resident), dim=7168,
inter=2048, k2=4, gate/up GEMV k-splits S=14 (as specced), down S=3 (the engine's `T16_SD`), zero-filled 3.96 GiB
store (the kernels have no value-dependent branches, so timing is data-independent), 5 warmup + 100 timed steps.

| Measurement | Result |
|---|---|
| ms per expert step, `t16d_gemv0_4` (the engine's SM70 runtime pick) | **1.172 ms** |
| ms per expert step, `t16d_gemv_4` name | 1.152 ms |
| effective trellis weight-stream (48 picks x 3 matrices x 3.67 MB = 0.53 GB/step) | **450.7 GB/s** |
| gate | <= 90 ms -> **PASS** (77x under) |

Phase breakdown (gemv0): gate/up phase (2 prep + 2 GEMV + post_gu) 753.9 us; the three GEMVs alone 1055.4 us;
the three preps alone 11.5 us; the two posts alone 42.9 us. Calibration confirmed: the math (weight streaming) is
the whole story at ~1 ms — as predicted from Phase-0 K1 (207.8 GB/s single-pass; 48 concurrent GEMVs keep HBM2
busier, hence 450 GB/s effective), and nothing near the 90 ms overhead line.

## Dual-arch build gate (VM) — PASS

- `CUDA_ARCHS=70 cargo build --release -p dsx --features cuda`: PASS (this is the fatbin the runs above used)
- `CUDA_ARCHS=86 cargo build --release -p dsx --features cuda` (compile-only): PASS

## Reproduce

```
# host side: tar the t3-decode worktree to the VM once
cd ~/dsx-sm70-t3
PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH CUDA_ARCHS=70 cargo build --release -p dsx --features cuda \
    --bin t16-decode-check --bin t16-decode-probe
flock /tmp/dsx-gpu.lock env CUDA_VISIBLE_DEVICES=GPU-4e4e3d86-f648-3706-6489-f5e8e17fcab8 \
    ./target/release/t16-decode-check
flock /tmp/dsx-gpu.lock env CUDA_VISIBLE_DEVICES=GPU-4e4e3d86-f648-3706-6489-f5e8e17fcab8 \
    ./target/release/t16-decode-probe
```

Addendum (ad30020): accessor renamed `is_pre_sm80`, predicate `arch < 80`; probe JSON key now `pre_sm80`.
