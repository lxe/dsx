# DSpark speculative decoding

`--dspark DIR` turns on speculative decoding with the draft layers (`mtp.*`) of the official DeepSeek-V4.1-Flash
checkpoint. It is **off unless given**.

## Download

The T16 GGUF does not contain the draft layers. `dsx fetch-mtp DIR` fetches them from the official checkpoint:
`config.json`, the index, and only the shards that hold `mtp.*` tensors (about 8 GB of the ~150 GB checkpoint). It is
resumable (run it again after an interruption; present shards are skipped) and checks each shard against the hub's
SHA-256. It needs `curl` and `sha256sum`; `--repo` picks another repository, `--hub` a mirror.

```
./target/release/dsx fetch-mtp /models/dsx-mtp
./target/release/dsx-server ... --dspark /models/dsx-mtp
```

## What it does

The draft stack proposes a block of tokens; the main model checks them in one pass and keeps the accepted prefix
(greedy: argmax agreement; sampled: the first rejection resamples from the rest, so the output distribution is the
main model's). Requests with images decode without drafts.

Where the stack lives: its memory (~8.5 GB at 32 slots, less at 16; `DSX_DSPARK_SLOTS`, default 16 of 128 draft experts resident; 32 measured 1.5 % slower on prose and 2.3 % on code) is set aside before
the experts are placed — on the aux GPU with the most expert room when the stack takes at most a quarter of it,
otherwise on the main GPU. The draft then runs beside the main GPU; the main GPU only writes the hidden states it needs
(`observe`), and the draft side projects that backlog in batches before it drafts. `DSX_DSPARK_GPU=<ordinal>` picks
the GPU.

## When it drafts

A verify pass is not free on a machine whose decode is bound by expert traffic, so drafting pays only where most of a
block is accepted. In measurements that was where the answer **repeats its prompt or itself** (tests for given code,
refactors, edits); on new prose or new code it cost a few percent. The controller (`nvidia/src/spec.rs`) therefore:

- keeps the trigrams of the prompt and of the output so far; drafting is considered only while the last tokens run
  along seen trigrams (two in a row, `COPY_RUN`). Otherwise neither drafts nor probes run.
- cuts each draft by DSpark's confidence head: per position and confidence bin it learns how often this machine's
  verify pass accepts (the raw scores run overconfident; prior: the sigmoid of the logit) and verifies the prefix with
  the most expected tokens per verify ms. The table is kept in `dspark-control.json`.
- while copying, sizes drafts by the measured per-position hit rate and step time, and compares wall time per token
  with and without drafting in windows of 256 copying tokens (the losing mode is retried every 6 windows).
- remembers what it learned in `DSX_STATE_DIR` (or `DIR`)`/dspark-control.json`. After 30 000 copying tokens in which
  it never paid, the next start leaves the stack unloaded for 7 days, and its memory goes to experts. New text never
  counts toward this, so a mostly-chat workload does not switch it off.

## Measured

DeepSeek-V4.1-Flash, T16 experts, RTX 3090 (210 W) as main GPU + CMP 170HX (64 GB) as aux GPU holding the draft,
automatic expert tiering. A/B/A/B phases of the same server build with `--dspark` off/on, after a warm-up of 25
requests; medians of 9 requests per phase (3 per task); per-task rows are means of the 6 requests per mode.

| Workload | off | on | |
|---|---|---|---|
| Code: write a new function | 42.3 | 42.1 | −0.5 % |
| Code: refactor a given module | 42.4 | 45.5 | +7.4 % |
| Code: write tests for given code | 43.6 | 49.1 | +12.4 % |
| **Code, all three** | **42.4** | **45.4** | **+7.0 %** (both "on" phases above both "off") |
| Prose | 43.3 | 43.0 | −0.8 % (both "off" phases above both "on") |

tokens/s. The remaining prose cost is memory: the reserved stack keeps some experts off the GPU. Earlier versions of
the controller without the copy gate measured −2.2 % and −4.4 % on prose. Numbers come from one machine; on two
identical GPUs, or without a GPU with spare memory, the trade-off may differ.

Later changes, each A/B/A/B on the same machine against the step before (25 warm-up minutes per phase):

| Change | Workload | before | after | |
|---|---|---|---|---|
| 16 instead of 32 resident draft experts | prose | 42.5 | 43.2 | +1.5 % (both phases) |
| | code | 46.5 | 47.5 | +2.3 % (phases overlap) |
| confidence cut | code | 45.9 | 47.2 | +2.8 % (both phases) |
| | prose | 43.0 | 42.9 | −0.1 % (phases overlap) |

## Without `--dspark`

Nothing changes: no draft memory is set aside, multi-token GEMVs (prefill chunks) keep their kernels, and every GPU
loads the kernels as built. The kernel image without FMA is a separate opt-in (`DSX_NOFMA`, feature `nofma`).

## Knobs for development

| | |
|---|---|
| `DSX_DSPARK_GPU=N` | GPU for the draft stack |
| `DSX_DSPARK_SLOTS=N` | resident draft experts (1–128) |
| `DSX_DSPARK_CUT=0` | verify every chosen draft, without the confidence cut |
| `DSX_DSPARK_PROF=1` | per-draft timing in the log |
| `DSX_DSPARK_DEBUG=1` | norms at fixed points on GPU and in the CPU reference (`core/src/dspark.rs`) for parity checks |
| `DSX_MAIN_GPU=N` | main GPU instead of the measured pick |
| `DSX_Q8R4=1` / `0` | multi-token Q8_0 GEMVs (verify passes, drafts) on `gemv_q8r4`: on with `--dspark`, off without it (main's numerics for prefill chunks) |
| `DSX_NOFMA=1,2` / `auto` | GPU ordinals that load the kernel image built without fused multiply-add, or `auto` for a timing pick per GPU at startup. Off by default; needs a build with `--features cuda,nofma` (a second nvcc pass). A CMP 170HX throttles FFMA and decodes faster without it |
| `dsx spec-draft`, `dsx bench-gemv` | offline acceptance and GEMV measurements |
