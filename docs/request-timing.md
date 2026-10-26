# Opt-in per-request decode diagnostics

Set `DSX_TIMING_LOG=1` on dsx-server to emit request-level prefill/decode wall time, completed decode steps, CPU worker busy/prep/forward/idle counter deltas, and CPU pick percentage. Any presence of the variable enables it; unset it to disable.

The first sampled token comes from prefill; `steps` counts successful `gm.step()` calls, not emitted tokens. Cancellation or context exhaustion can therefore produce zero steps. In that case the divisor is clamped to one and the counters are raw deltas, not a meaningful per-step rate. CPU counters are relaxed-atomic diagnostic snapshots, not exact barriers; idle can include time around request boundaries. Missing CPU workers return zero counters, not a calibrated GPU-only measurement. Do not sum overlapping CPU and GPU times to infer wall time or derive tokens/s from these counters.

If the existing `DSX_EVPROF` is also enabled, event deltas are reported per recorded event step. This patch does not enable GPU events, add synchronization, or clear cumulative event counters. CUDA-event profiling itself can perturb performance: keep it off for throughput A/B tests. No prompt text, output text, API keys or model paths are printed by these new lines.

This is a refinement of the historical workstation timing patch: counter reads and event snapshots are gated by the opt-in flag, cumulative event data is preserved, and integer deltas use saturating subtraction. It changes diagnostics only, not model arithmetic or placement.

Validation: the original patch applies cleanly to upstream aed35fc; Rust syntax is checked with rustfmt. Current CUDA compilation and GPU integration have not been run in the CPU-only review environment. This remains a draft until a real CUDA build and consecutive-request checks (including cancellation, prefix reuse and zero steps) are performed. CPU-only cargo tests do not compile this CUDA-gated module.
