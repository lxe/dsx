# Calibrate expert placement from runtime routes

DSX already supports hotness-based placement through `--rank`. This contribution adds a standard-library CLI for its **handoff.rs** route logs, complementing `quant/make_rank.py`, which consumes quantizer calibration statistics. No runtime allocator, adaptive cache, kernel or model weight is changed.

## Workflow

1. Use a separate calibration server with your normal model and placement settings and `DSX_LOG_ROUTES=/absolute/path/calibration.routes` in its environment. Use a fresh log path per server; handoff opens it with truncation. Submit varied calibration prompts, then stop the server gracefully so buffered routes are flushed. Do not use this logging run as a performance measurement.
2. Generate the ranking (defaults: 40 layers, 384 experts, six picks):

```sh
python3 quant/rank_from_routes.py --calib calibration.routes --single-token-only --out rank-coding.json --stats rank-stats.json
```

3. Start the same server with `--rank /absolute/path/rank-coding.json`, without route logging. Compare with an otherwise identical no-ranking baseline in A–B–A–B order on held-out evaluation prompts. Record TTFT and decode separately. Keep model, adapters, sampling, GPU budgets, affinity, context and power limits fixed. Save the rank alongside the exact model revision; do not reuse expert IDs across unrelated models.

Only the handoff log grammar `layer expert expert ...` is accepted. The alternate gpumodel log grammar `position layer [experts]` is deliberately rejected. No timestamps or diagnostic output are silently skipped. Multi-token lines are validated before optional exclusion. Single-token filtering is a heuristic that also includes one-token prefill. Missing layers, duplicate picks, out-of-range IDs and empty input are rejected. Unseen experts remain with count zero and deterministic expert-first/layer-second tie breaking. Inspect coverage and calibrate further if many experts are unseen. This tool does not estimate byte-based placement, CPU-pick fraction or future tokens/s.

## Historical workstation evidence

Model: **DeepSeek-V4.1-Flash-1017, Unsloth UD-IQ2_XXS mixed GGUF**, six shards, no LoRA. Hardware: i9-10900X, 128 GiB RAM, RTX 3090 24 GiB + CMP 170HX 64 GiB (RTX first). Both variants already contain the native-format/IQ3/Q2 CPU patches proposed separately in PR #12. Context 4096, chunk 128, arena 80 GB, 14 workers (CPUs 0–13), main-gb=-6, aux-gb=-3; adaptive placement and defer disabled.

| Model / quantization | Placement | Median decode tokens/s | Gain |
|---|---|---:|---:|
| DeepSeek-V4.1-Flash-1017 / UD-IQ2_XXS | Default, no rank | 9.69 | baseline |
| DeepSeek-V4.1-Flash-1017 / UD-IQ2_XXS | Coding calibration rank | 21.64 | +123.3% (2.23×) |

These are October 23–24, 2026 historical development-binary runs, not fresh GPU tests of upstream or this PR. Values are medians across eight request rates per variant (four prompts × two runs), using `(completion_tokens - 1)/(total_seconds - TTFT)`. Early requests include paging/cache effects. Evaluation prompts: German hash-table explanation, Python merge_intervals, HTML/CSS grid, repeated merge_intervals. Completion counts: 48, 195, 256, 195; HTML/CSS hits the cap. Calibration used eight different prompts: Python LRU cache, TypeScript debounce, SQL sales totals, CSS menu, transaction deadlocks, BFS, Rust parsing and upload API tests. This is a small coding-focused workload, not a general speed or quality guarantee. Placement can change floating-point order. The historical report records matching answers for its comparisons, not universal output equivalence.

| Run | Request 0 TPS | Request 1 TPS | Request 2 TPS | Request 3 TPS |
|---|---:|---:|---:|---:|
| 06-static-A | 8.904948 | 9.738745 | 9.751189 | 9.823834 |
| 07-ranked-A | 12.705539 | 21.656418 | 21.614937 | 22.163394 |
| 08-static-B | 8.849622 | 9.624683 | 9.648713 | 9.758667 |
| 09-ranked-B | 12.722346 | 21.780828 | 21.628733 | 22.358082 |

Source archive SHA256: `42f112eaf765232526327ecfad3b08bd2a4d8e64535a5eeb8cfbd58321714cbc`. The CLI was checked against the original calibration log and reproduced all **15,360** entries of the archived `rank-coding.json` exactly. Logs and workstation-specific paths are not published.

## Local tests

```sh
python3 -m unittest discover -s quant/tests -v
```

No GPU, model download, service start or hardware configuration change is performed by this tool.
