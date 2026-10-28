# T4 results — SM70 prefill attention (Option A wiring)

Date: 2026-10-27. Host: a 2x V100-PCIE-32GB host (`ssh <v100-host-user>@<v100-host>`), Tesla V100-PCIE-32GB 80 SMs,
<gpu0-uuid>, max SM clock 1380 MHz, CUDA 12.9. All GPU runs under
`flock /tmp/dsx-gpu.lock`, `CUDA_VISIBLE_DEVICES=<gpu0-uuid>`, `CUDA_ARCHS=70`.
Harness: `nvidia/tests/gpu_attn_sm70.rs` (new; accuracy + `--ignored` perf).

```
cargo test --release -p dsx --features cuda --test gpu_attn_sm70 -- --nocapture            # accuracy
cargo test --release -p dsx --features cuda --test gpu_attn_sm70 -- --ignored --nocapture  # perf
/usr/local/cuda/bin/compute-sanitizer --tool memcheck <test binary> sm70_attn_accuracy --nocapture
```

## Builds (dual-arch gate)

| Build | Result |
|---|---|
| `CUDA_ARCHS=70 cargo build --release -p dsx --features cuda` | **PASS** (1m29s) |
| `CUDA_ARCHS=86 cargo build --release -p dsx --features cuda` | **PASS** (1m00s) |

Only pre-existing warnings (core/src/server/engine.rs `unused_mut`, gpumodel.rs dead fields).

## Accuracy — wired path (`SparseAttnTc::launch` -> `sparse_attn` on arch 70) vs plain fp32 CPU reference

Random Q/K/V at realistic magnitudes (|q|, |kv| <= 10 → score std ~30 after the 1/sqrt(512) scale; one
case at |q| <= 1 → near-uniform softmax), bf16 cache rows, gathered rows with -1 skips, duplicate
indices, and an all-masked (sink-only) token. Gates: cosine >= 0.9999 per 512-vector, max-rel <= 2e-2.

| Case (T, W, K, qmag) | cos(min) | max rel | max abs | verdict |
|---|---:|---:|---:|---|
| 5, 128, 512, x10 | 1.000000 | 7.69e-3 | 6.25e-2 | PASS |
| 17, 128, 512, x10 | 1.000000 | 7.81e-3 | 6.25e-2 | PASS |
| 64, 100, 37, x10 | 1.000000 | 7.81e-3 | 6.25e-2 | PASS |
| 9, 128, 0, x10 | 1.000000 | 7.81e-3 | 6.25e-2 | PASS |
| 33, 128, 512, x1 | 1.000000 | 7.81e-3 | 3.12e-2 | PASS |

All outputs finite; the all-masked token is exactly zeros. max-rel ≈ 7.8e-3 is the kernels' bf16 output
rounding (half-ulp 3.9e-3) plus fp32 summation-order differences — comfortably inside 2e-2.

Cross-check: the pre-existing `gpu_attn` test (sparse_attn / sparse_attn_tc / sparse_attn_tc_fast vs an
f64 reference and vs each other) passes on SM70 too — 2 passed, 0 failed; through the wiring `tc` and
`tc_fast` both route to `sparse_attn`, and no in-tree caller reaches the (now trapping) stubs.

## compute-sanitizer

`--tool memcheck` over the accuracy test: **ERROR SUMMARY: 0 errors**.

## Perf — W = 128, K = 512, H = 64, hd = 512 (flop = 2 GEMMs x 2*T*H*(W+K)*hd)

| T | sparse_attn raw | wired (`tc.launch`) | TFLOP/s | µs/token | <= 10 ms/token |
|---:|---:|---:|---:|---:|---|
| 16 | 0.55 ms | 0.55 ms | 2.44 | 34.3 | PASS |
| 64 | 1.84 ms | 1.84 ms | 2.92 | 28.7 | PASS |
| 512 | 11.37 ms | 10.77 ms | 3.99 | 21.0 | PASS |
| 4096 | 82.44 ms | 82.55 ms | 4.16 | 20.2 | PASS |

Wired vs raw serial agree (same kernel; the 512-row 0.6 ms delta is run-to-run clock jitter). The
gate's other reading — ">= 2x faster than the non-TC path's naive expectation", with the naive
expectation = the 5090's measured serial time (22.2 ms at T=4096) scaled by the two parts' fp32 peaks
(~104.8 vs ~14.1 TFLOP/s → 165 ms): measured 82.6 ms = **2.0x** — PASS on that reading as well. The
kernel runs at 29% of V100's fp32 peak (4.16/14.1), vs 14.8% of peak on the 5090 (15.5/104.8).

Budget view (the 41 tok/s reference = 24.4 ms/token): one launch per layer; even summed over all ~41
layer calls of a 4096-token chunk this is 41 x 20.2 µs = **0.83 ms/token** — 12x under the 10 ms/token
line. For context, the SM80/120 tc kernel's role at the same shape is 4.2 ms per call on the 5090
(5.3x over its serial 22.2 ms); a Volta wmma port would target >= 19 TFLOP/s (≥ 40% of Phase-0 K2's
47.6) ≈ 4.4 µs/token — the documented follow-up if prefill ever becomes attention-bound
(sketch in [T4-NOTES.md](T4-NOTES.md)).

Decode is unaffected by this task (T <= 8 uses `sparse_attn_tile` + merge, SM70-clean already);
measured for reference at T=1, W+K=640 on this GPU: `sparse_attn` 377 µs, tile+merge 32.5 µs.
