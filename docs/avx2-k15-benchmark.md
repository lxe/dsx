# AVX2 T16K1.5 compact tile benchmark

The single-token AVX2 path reconstructs a 48-byte T16K1.5 tile with four in-tile loads, two low views and derived high views. Window shifts, the codebook multiply, exact integer dots and F32 group accumulation are unchanged. Other formats, VNNI, scalar and multi-token paths retain the generic implementation.

## Reproduce

Use an AVX2-capable x86_64 CPU with the features checked by `Isa::Avx2.supported()`. The repository builds for the local CPU by default. No model download or CUDA feature is needed.

```sh
cargo test --release -p dsx --lib cpu::t16k:: -- --test-threads=1 2>&1 | tee avx2-k15-tests.log
cargo run --release -p dsx --example avx2_k15_bench -- 17 | tee avx2-k15-bench.jsonl
```

The optional final argument pins the benchmark to a CPU index; choose an allowed core or omit it. The benchmark compares generic and compact implementations in A/B/B/A order, checks the final output bits, warms each case for 150 ms and times 4,096 calls per round. Each call computes 64 output columns of a 5,120 x 2,304 matrix. The fixed case reuses one region; the streaming case rotates through 48 matrices and all column regions. This measures a direct single-thread kernel, not complete expert or model throughput.

## Recorded result

AMD Ryzen Threadripper PRO 5975WX, Zen 3, AVX2 without VNNI, CPU 17, release build with `target-cpu=native`. Parent revision: `98f1259732f8c3a5f445b25e080daf70fc300967`.

| Case | Generic mean | Compact mean | Throughput gain |
|---|---:|---:|---:|
| Fixed region | 14.508 us | 13.006 us | 11.56% |
| Rotating matrices | 14.776 us | 13.202 us | 11.92% |

All 12 T16 tests passed. The two new tests make 37,120 complete-vector comparisons against the retained generic AVX2 kernel: zero/extreme/random activations, all 384 bits of the first and last tiles, every input row, unaligned storage, column slices and every group split in four matrix shapes. Equality compares F32 bits; no tolerance is used. Existing scalar, multi-token and dot-product tests also pass on this CPU. VNNI paths compile but were not executed on this host.

A separate deployment backport into an older, locally modified engine passed 4,108 full-vocabulary logit comparisons (129,280 values each) on English, Chinese, code and longer-prefix forced-token fixtures. Its whole-model gain was only 0.48% in that bounded audit. This is supporting deployment evidence, not an end-to-end performance claim for current upstream. Finite tests do not establish performance on every workload or CPU.

Raw direct-kernel results:

```jsonl
{"round":0,"streaming":false,"candidate":false,"ns_per_call":14484.7666015625,"calls":4096,"float_bits_match":true}
{"round":1,"streaming":false,"candidate":true,"ns_per_call":13008.22705078125,"calls":4096,"float_bits_match":true}
{"round":2,"streaming":false,"candidate":true,"ns_per_call":13002.789306640625,"calls":4096,"float_bits_match":true}
{"round":3,"streaming":false,"candidate":false,"ns_per_call":14531.8828125,"calls":4096,"float_bits_match":true}
{"round":0,"streaming":true,"candidate":false,"ns_per_call":14776.710693359375,"calls":4096,"float_bits_match":true}
{"round":1,"streaming":true,"candidate":true,"ns_per_call":13200.49365234375,"calls":4096,"float_bits_match":true}
{"round":2,"streaming":true,"candidate":true,"ns_per_call":13203.984375,"calls":4096,"float_bits_match":true}
{"round":3,"streaming":true,"candidate":false,"ns_per_call":14775.51416015625,"calls":4096,"float_bits_match":true}
```
