# Native Q2_K and IQ3_XXS experts

This change adds GGUF types 10 (`Q2_K`) and 18 (`IQ3_XXS`) to the existing
block-format path. It lets mixed GGUF exports retain these expert blocks instead
of requiring a conversion to one of DSX's other formats. It does not change T16,
expert placement, routing, deferral, power settings, or the server defaults.

## Implementation

- `core`: type IDs, byte sizes, scalar decoders and IQ3_XXS's codebook.
- CPU: natural-order int8 activations and AVX2 / AVX-VNNI / AVX-512-VNNI dispatch.
  Q2_K expands the code and scale bytes in registers instead of scalar stack
  arrays. Its original kernel is retained under `cfg(test)` for bit comparisons.
  IQ3_XXS expands sign masks with SIMD and applies signs to activations before
  the unsigned-magnitude dot product. Activations are quantized to [-127,127].
- CUDA fast expert store: original packed blocks, tensor-core tile dispatch,
  and scalar weight access for prefill. The old store (`DSX_OLD_MOE`) explicitly
  rejects these formats. There is no new support for arbitrary non-expert
  projection placement implied by this change.

The Q2_K optimization preserves the former SIMD lane sums and floating-point
operation order. This is bit identity to that kernel, not a claim that the int8
activation path is bit-identical to an unquantized model. The optional change
that replaces the minimum dot product with `s16` sums is not included.

## Tests

No model download is needed for CPU tests:

```sh
cargo test --release --locked -p dsx-core --test native_quant
cargo test --release --locked -p dsx --lib --no-default-features cpu::qdot::tests
cargo test --release --locked -p dsx --lib --no-default-features cpu::moe::tests
```

The Q2_K bit-identity test covers both 5120/2304 input widths, T=1..8,
random and saturated codes/scales, zero scales, signed scales and extreme
activations, on each SIMD ISA the host supports. The scalar comparison test
also covers IQ3_XXS and batches larger than MAX_T. GPU tests must be run on a
CUDA machine, for example:

```sh
CUDA_ARCHS=80,86 cargo test --release --locked -p dsx --features cuda \
  --test gpu_expert_fast fast_gemv_matches_cpu -- --nocapture
CUDA_ARCHS=80,86 cargo test --release --locked -p dsx --features cuda \
  --test gpu_expert_fast moe_fast_iq_matches_cpu -- --nocapture
```

## Fixture provenance

The little-endian `.raw` and `.f32` pairs in `core/tests/native-fixtures` are
synthetic, not model weights. They were generated with the ggml C decoders at
llama.cpp revision `da8e2a54b919557b816b1c2beb432ccb766f7ce0` (MIT).
Each pair stores 64 blocks, including signed, zero and subnormal scales.
The IQ3_XXS codebook is from the same reference; license notices are retained
in both codebook source files.

To reproduce them, build a shared ggml library exposing `dequantize_row_q2_K`
and `dequantize_row_iq3_xxs` at that revision, then run:

```sh
cargo build --release --locked -p dsx-core --bin native-decode-probe
python3 bench/native_quant_reference.py \
  --ggml-lib /path/to/libggml-base.so \
  --probe target/release/native-decode-probe \
  --fixtures core/tests/native-fixtures
```

This compares 4096 synthetic blocks per format before writing the small subset.
Use a little-endian host for the ctypes reference output.

## Reproducible kernel benchmark

```sh
cargo run --release --locked -p dsx --bin native-qdot-bench -- 3
```

The dependency-free benchmark driver (using DSX's existing dependencies) reports
five-sample median time per matrix for both projection shapes, with pools of one
and 32 matrices. It uses one thread, prequantized activations, a warm-up pass,
and consumes output with `black_box`. Its reported weight GB/s is **nominal
matrix bytes divided by elapsed time**, not measured DRAM traffic. CPU cache
size determines whether a pool is cache-resident.

For an old/new comparison, build the format-support commit and the optimized
commit separately, using the same benchmark driver, Rust version, ISA, affinity
and build flags. Alternate A-B-A-B on an otherwise idle machine. Keep raw output,
CPU information and exact commit IDs. Do not compare to upstream main, which
has no Q2_K/IQ3_XXS CPU kernels. Measure full-model decode and prompt processing
separately; this benchmark does not predict a tokens/s improvement.

## Validation status for this submission

See the PR description for commands actually run. GPU execution and full-model
quality/performance after rebasing require hardware validation. Historical
workstation runs are not a substitute for validating this exact revision.
