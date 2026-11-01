# Tesla V100 (SM70) support

Volta runs the same T16 checkpoint with `CUDA_ARCHS=70` (CUDA 12.9; CUDA 13 dropped sm_70). Each SM80-only path has an
SM70 twin, chosen at compile time (`__CUDA_ARCH__ >= 800`) or at run time (`Gpu::is_pre_sm80`):

| Path | SM80+ | SM70 |
|---|---|---|
| Decode T16 GEMV | cp.async staged kernel | the unstaged kernel (`DSX_T16_GEMV=0` on SM80), bit-identical output |
| Prefill attention | bf16 / ldmatrix tensor-core tiles | serial fp32 `sparse_attn`; the tile kernels are `__trap()` stubs |
| Expert kernels | int8 IMMA `m16n8k32` / `m16n8k16` | warp shuffles + `dp4a` over the same fragments; integer sums, identical accumulators |

## Checks

| Check | Hardware | Result |
|---|---|---|
| `gpu_attn_sm70` accuracy, `gpu_expert_fast` (every quant format vs the CPU reference) | 2x V100-SXM2-32GB | pass |
| Full model, greedy chat (1.9K and 2.0K-token prompts) | 2x V100-SXM2-32GB | coherent output |
| Same greedy requests as main | 2x RTX 3090 (sm_86) | byte-identical to main |

## Speed

| Hardware | Prefill | Decode |
|---|---:|---:|
| 2x V100-PCIE-32GB, 108 GB RAM, `--arena-gb 85` (contributor) | 467 tok/s at 123K tokens | 21-23 tok/s, flat to 123K |
| 2x V100-SXM2-32GB, `--arena-gb 180`, container capped at 15 CPUs | 583 tok/s at 9.9K tokens | 5.5 tok/s (CPU experts throttled by the quota) |

The CPU-expert pool assumes a whole desktop CPU; a CPU quota far below the visible core count slows decode, not the
GPU paths. `t16-decode-check` compares the SM70 decode GEMV against the CPU reference; `t16-decode-probe` times it.
