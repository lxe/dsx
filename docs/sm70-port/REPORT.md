# dsx SM70 Port — Final Report (2026-10-27/28)

**Outcome: COMPLETE.** DeepSeek-V4.1-Flash-1017 (552B MoE, 234 GiB T16) now
serves interactively on 2× Tesla V100-PCIE-32GB via the ported dsx engine —
**21–23 tok/s decode (0 → 128K context, no depth penalty), 467 tok/s prefill
at 123K tokens, zero host resets** including two full-128K prefills.

## Headline numbers (acceptance config: 2× V100, 85 GB arena, ctx 131072)

| Metric | Value | Gate | Reference (2× RTX 3090) |
|---|---|---|---|
| Decode, short context (warm) | **21.0 tok/s** (engine counter) | ≥10 ✅ | 41 |
| Decode at 123K-token context | **21.9–22.9 tok/s** | — | — |
| Prefill at 123K tokens | **466.6 tok/s** (263 s) | ≥150 ✅ | ~1,008 |
| Prefill ~15K, single GPU | ~370 tok/s | ✅ | — |
| Correctness | coherence (Paris/391), repeat-verbatim exact, kernel gates: bit-exact int paths, ≤3.8e-7 float paths, cos 1.000000 attention | ✅ | — |
| Stability | 2× full-128K prefill+decode cycles, sustained warm runs, no resets | ✅ | — |
| Speculation | engine-capable (n-gram context drafts, bench `--spec`), **not wired in the serving path** — all numbers are plain decode | documented | — |

Boot profile: GPU1 main (2196 experts + dense + cache), GPU0 aux static (3763
experts, streams prefill at 11.9 GB/s), 85 GB pinned arena (9,967 experts;
1,630 file-served), load ~110 s, VRAM 30.7/31.4 GiB.

## What the port changed (branch `sm70-port`, final commit 710907b; diff vs
upstream main: 18 files, +2348/−53; every SM80+ path byte-identical)

1. **Decode T16 GEMV** (t16.cuh): SM70 exports `t16d_gemv_{3,4,5}` over the
   in-tree non-cp.async kernel (author-documented bit-identical); runtime
   select via `Gpu::is_pre_sm80()` (`arch < 80`, scale major*10+minor).
2. **Expert IMMA → dp4a** (expert_fast.cuh): `mma32t/mma16t<DP>` emulate the
   int8 tensor-core fragments with warp shuffles + dp4a — identical integer
   products, bit-exact accumulators (verified max-abs 0 across 8 quant
   formats × 3 shapes). Perf 147–690 GB/s weight-stream (gate ≥104).
3. **Attention** (attn.rs/dsx.cu): arch<80 routes to the author's in-tree
   serial `sparse_attn` (fp32) — cos 1.000000 vs reference, 20.2 µs/token
   (12× under budget); the TC kernels become `__trap()` stubs, loud if ever
   misrouted.
4. Scaffold (sm70.cuh), dual-arch compile gate (SM70+SM86 fatbins), hardened
   harnesses (NaN-blind gate fix, KEEP-IN-SYNC markers), phase-0 provenance.

## Process (subagent-driven, per the approved plan)

Plan: `~/Documents/docs/superpowers/plans/2026-10-27-dsx-sm70-port.md`
(reviewed and approved before execution). Five implementation tasks executed
by fresh subagents in two parallel waves, each with two-stage review (spec
compliance + code quality); the review loops caught three real defects
(NaN-blind correctness gate, an arch-encoding bug that would have silently
slowed every non-Volta GPU, and wrong attention-function scope). Final
whole-diff review: ✅ approved. All silicon verification ran on the VM
(`debianv100s`) with a serialized GPU lock; per-task evidence in
`dsx-sm70/bench/{PHASE0,T3,T4,T5}-RESULTS.md`.

## Follow-ups (documented, not blockers)

- **Speculation in the server**: wire `spec::lookup` drafts + the rewindable
  verify that already exists in gpumodel — free speed on repetitive text.
- **wmma attention upgrade**: sketch in T4-NOTES (uses sm70.cuh's dsx_cp16);
  est. ~4.5× the serial kernel — attention is not the bottleneck, so low
  priority.
- **dp4a shuffle optimization**: butterfly-reduce variant would cut SHFL 25%
  (T5 review note) if expert headroom is ever needed.
- **Pre-upstream-PR scrub**: machine-specific strings in bench artifacts
  (hostnames/UUID), per final review note 1.

## Live service

`dsx-server` running on the VM: `http://192.168.2.216:8090` (OpenAI API,
model `deepseek-v4.1-flash`; **unauthenticated** — bind 127.0.0.1 or set
`--api-key` before wider exposure). Model at
`~/models/DeepSeek-V4.1-Flash-1017-dsx-t16/` (SHA256-verified). Rebuild/run:
`cd ~/dsx-sm70 && DSX_GPUS=0,1 ./target/release/dsx-server --model
.../DeepSeek-V4.1-Flash-1017-dsx-t16.gguf --rank .../expert-rank.json
--main-gb=-2.0 --arena-gb 85 --ctx 131072 --no-vision`.
