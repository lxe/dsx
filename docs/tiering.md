# Automatic expert tiering

dsx places routed experts over the GPUs, a RAM arena and the model file at startup, in the order of `--rank`
([placement](placement.md)). Adaptive GPU stores then swap experts by recent use; every other store keeps the set it
got at startup, and what the server learned about this machine's traffic is gone after a restart.

`DSX_TIER=0` turns all of it off, the learned startup placement included, for comparing with and without on one build:
the log then says `tier: off` and has no `tier: placement from` line, and the saved state stays as it was.

`nvidia/src/tier.rs` learns which experts this machine uses, measures which device each layer waits for, moves work
toward the devices with room while the server runs, and keeps all of it next to the model for the next start. It needs
no option and adapts to whatever GPUs, RAM and disks a start finds.

## What it does

| When | What |
|---|---|
| every decode pass | the CPU worker counts each layer's picks (GPU-resident ones included) and which tier served them: a GPU store, the arena, the SSD buffer or the file |
| every 64th decode token | one layer (rotating through all of them) is timed with CUDA events: each GPU's expert kernels, and how long the main GPU waits for the CPU and then for each aux GPU |
| every 10 s | a background thread folds the counts into a long-term score per expert (half-life 200k decode tokens) |
| while a request runs, at most one per 16 tokens | a planned move or exchange starts (see below) |
| in every pause of 2 s | the engine thread runs planned moves and exchanges back to back and lets hot experts of the file side take the place of cold ones in the RAM arena; the background thread warms the page cache with the hottest experts the CPU still reads from the file side and, on a slow medium, copies them into an SSD buffer. All of it stops when a request arrives |
| every 2 min while busy, 10 s into a pause | the state is saved atomically next to the model if it changed; the previous version stays as `.prev`. The server does not save on a signal: stopping it mid-request loses up to 2 min of learning |
| next startup | the learned score is blended into the ranking before placement |

Routing is never changed: only where an expert's bytes are and which device computes it. As with any placement, a
different device can change the floating-point summation order of an expert's output.

## Who is waited for

In every layer the main GPU, the CPU worker and the aux GPUs compute their experts in parallel; the layer is done when
the last of them is. After its own work the main GPU waits for the CPU's experts, then for each aux GPU in turn. Each
wait is how much later that device finished than everything waited for before it. Smoothed over the samples (from 16
on) this says which device is on the critical path, and tiering moves work away from it:

| The main GPU waits | Meaning | What tiering does |
|---|---|---|
| for an aux GPU (> 20 µs after the CPU) | that aux GPU finishes last | **exchanges**: its hottest experts swap places with colder ones on a fixed-set GPU that is not late: the main GPU, or an aux GPU the main GPU does not wait for |
| for the CPU (> 20 µs), not for aux GPUs | the CPU finishes last | **moves**: hot experts the CPU computes go onto GPUs with a fixed set that are not late |
| for nobody | the main GPU is last or all are even | nothing |

The waits, the CPU's picks per token and decode tokens/s are kept per GPU combination in the state file.

## Moves and exchanges

The background thread plans them and copies the experts into one of four pinned prefetch slots: from the arena when it
holds them (read under the arena's lock, so an arena move waits for the copy), else from the SSD buffer, else from the
file. Only a move into the arena reads the file side; once everything in use is on GPUs and in the arena, tiering
reads nothing from disk. Between two tokens, or in a pause, the engine checks the plan against the current state
(experts where planned, nothing in flight for them, same quant types, room to copy) and starts it with the protocol of
adaptive replacement (`GpuExperts::begin_swap` / `commit_ready`):

- **Move from the CPU** (into a store that keeps a fixed set: the main GPU's with `DSX_ADAPT=0`, aux GPUs on slower
  links): with an arena the victim first gets its arena copy in a pause, so the CPU never computes it from the file
  side. Then the victim leaves the store's table first and the CPU computes it from the next token on; the new expert's
  bytes go to the victim's slot on the side stream after the current token's kernels and enter the table only when the
  copy has completed.
- **Exchange between GPUs** (a late GPU and one that is not, both with a fixed set; any number and mix of GPUs, of
  any size and speed): the hot expert leaves the late GPU's table and the other GPU's colder victim leaves its table, both before
  either copy starts. Until the copies land, the CPU computes both. Then the hot expert is resident on the GPU with
  room and the victim in the hot expert's old slot. Both stores stay full; no expert is ever resident on two GPUs.
- **Move into the arena** (pauses only): an expert the CPU reads from the file side takes the place of another one's
  arena copy (same quant types, so the same size). First to go are copies nobody reads: those of experts resident on a
  fixed-set GPU. Then the coldest experts the CPU computes from the arena. It happens on the engine thread while no
  pass runs, the CPU worker is idle and every stream is synchronized; the background thread reads the arena only under
  its lock. With enough pauses the experts in use end up on GPUs and in RAM, and the disk is no longer read.

An adaptive store (`DSX_ADAPT`) may hand any resident back to the CPU at any token, so dsx keeps an arena copy of each
of them. Tiering keeps that rule: it moves nothing into an adaptive store and never takes its residents' copies.
Exchanges into an adaptive store were tried and removed: on a 3090 (adaptive) + CMP 170HX they took the CMP off the
critical path, but the store handed more experts back to the CPU (5.8 -> 8.2 and 6.9 -> 9.2 picks per token) and the
decode rate did not rise. An exchanged expert rests for 15 minutes before it can be exchanged again.
`adaptive_swaps` in the stats counts an adaptive store's own replacements.

GPUs differ in size and speed. A larger store holds more experts and gets more picks; a slower GPU needs longer for
each. Neither is configured or assumed: an exchange is one expert for one of the same size, so every store stays as
full as placement made it, and the measured waits decide the direction and when to stop.

A candidate needs a learned score of at least 8 and a victim of at most half its score; a wasted arena copy makes
room for any expert the CPU reads from disk (the hottest residents' copies go first: cold residents are the next
victims and need theirs).

## Where a token's time goes

`token_ms` in the stats splits a decode token from what is measured anyway: `total` (decode rate), `main_gpu_experts`
(the main GPU's expert kernels), `waiting` (for the CPU and the aux GPUs) and `rest_main_gpu` (attention, router, head
and whatever else the main GPU does alone). Tiering can only shorten `waiting`. Stores that already swap by recent use (`DSX_ADAPT`) keep doing so and
take part in exchanges.

## State

`<model file>.tier.json` next to the model when its directory is writable, else in `$DSX_STATE_DIR`,
`$XDG_STATE_HOME/dsx`, `~/.local/state/dsx` or `/var/tmp/dsx`; a state an earlier start left in another of these
places is still read. With an overlay the name gets `+<overlay stem>` (`model.gguf+uncensored-overlay.tier.json`):
each model and overlay learns on its own. A change of any file (path, size or modification time) or of the model's
shape starts fresh.

```json
{
  "version": 2,
  "model": {"id": "…", "files": ["…/model.gguf", "…/overlay.gguf"]},
  "usage": {"tokens": …, "score": [ … per layer × expert … ]},
  "devices": {
    "GPU-3f2a…": {"name": "…", "link_gbs": …, "expert_layer_ms": …, "picks_per_token": …, "seen": …},
    "disk:8:16": {"read_mb_s": …, "seen": …}
  },
  "configs": {
    "GPU-3f2a…+GPU-9c11…": {"decode_tok_s": …, "cpu_picks_per_token": …, "wait_for_cpu_us": …, "wait_for_aux_us": [ … ], "share_from_file": …, "seen": …}
  },
  "stats": { … }
}
```

- `usage` does not depend on the hardware: the same state serves a start on one GPU or on several, with more or less
  free memory; placement takes whatever stores this start has, in learned order.
- `devices` are keyed by GPU UUID (the same in every `CUDA_VISIBLE_DEVICES` order) and by disk; a GPU missing in one
  start keeps its record.
- Rotation: devices unseen for 90 days are dropped, at most 16 GPU combinations are kept, two state files at most.
- Writes go to a temp file and are renamed over the state; the state before stays as `.prev` and is read when the
  state itself is missing or damaged. Two servers on one model both write it: the last one wins, nothing is merged.
- `--tier-reset` removes this model's state, its previous version and its SSD buffer at startup, then runs as usual.

## Budgets and safety

- Page cache: at most 40 GB, never below 16 GB of `MemAvailable`, never closer than 4 GB to the process's memory cgroup
  limit (`memory.max`). Warming and SSD copies run only in pauses (a request counts as busy from its first to its
  last token, prefill included): they would compete with the CPU's expert reads for memory bandwidth. The warm set is
  walked again every 10 minutes, since the kernel may have evicted pages.
- SSD buffer: when the model's medium is rotational or reads below 300 MB/s (measured with O_DIRECT at startup); in
  the first writable directory on another non-rotational device with more than 20 GB free (`$DSX_CACHE_DIR`, the
  state directory, `$XDG_CACHE_HOME/dsx`, `~/.cache/dsx`, `/var/tmp/dsx`); half its free space, at most 512 GB and at
  most the size of the file-side experts. Entries are appended, synced and only then published; writing stops when
  the disk has less than 20 GB left. A buffer of the same model is kept across starts and only grows. One process
  uses it at a time (file lock); a second server on the model reads the model file.
- The block device of a path is found by device number, else (btrfs and others) by its mount in `/proc/self/mountinfo`.
- GPU memory: one staging buffer (the largest expert) per fixed-set store; nothing else is allocated at run time.
- A failed timing query drops the sample; a failed pause slice stops that round. Only the first model of a process
  gets tiering; its background thread is stopped and joined when the model is dropped. Every other failure (no
  writable directory, no SSD, no pinned memory, unreadable state) disables only that part.

## What the log shows

Lines of this form (numbers depend on the machine):

```
tier: placement from … learned tokens (…% learned, …% shipped ranking)
experts resident: GPU0 … (… GB), GPU1 … (… GB, static, tier moves, … GB/s link)
tier: … experts on the file side (… GB), model medium … MB/s; state …/model.gguf.tier.json
tier: {"picks":{…},"share_from_file":…,"experts_on_gpus":…,"moves_planned":…,"wait_for_cpu_us":…,"wait_for_aux_us":…,…}
```

## Not done here

- The arena's size is fixed at startup (`--arena-gb`); its contents follow use in pauses.
- GPU memory budgets are taken at startup; a GPU added or freed while the server runs is used at the next start.
- An adaptive store takes no part in exchanges (see above).
- Work only leaves GPUs the main GPU waits for; when the main GPU itself finishes last, nothing measures how much
  room the others have, and nothing moves. The main GPU is still chosen by host link.
- A full SSD buffer keeps what it holds; it does not follow later changes of use.
- Loading by pattern was measured and not built: a table of the four experts most often picked at the next token
  guessed 8 % of the CPU's picks beyond plain repeats (12 %), at 7 % precision.

## Measured

One machine: RTX 3090 (main, adaptive, 24 GB) + CMP 170HX (aux, fixed set), DDR4 host, DeepSeek V4.1 Flash with a
state learned from about 82k decode tokens. A-B-A-B on one build: `DSX_TIER=0` (A) against the default (B). Each phase
restarts the server, checks the log for `tier: off` or `tier: placement from`, sends one warm-up run and then three
runs of three fixed 300-token requests (9 decode rates):

| phase | tiering | median decode tok/s | the 9 rates |
|---|---|---|---|
| A1 | off | 38.2 | 38.7 35.7 37.9 39.4 36.7 38.2 39.4 36.5 38.2 |
| B1 | on | 42.2 | 42.2 41.3 42.8 42.5 40.3 42.2 42.4 41.0 42.2 |
| A2 | off | 37.2 | 39.5 37.0 37.2 39.2 36.5 38.2 39.6 35.9 37.2 |
| B2 | on | 41.5 | 42.3 40.3 41.2 42.2 40.8 42.2 42.6 41.4 41.5 |

Off 38.0, on 42.2 tok/s: +10.9 %. Both B phases lie above both A phases.

- The gain comes from the learned placement at startup. Run-time moves lowered the CPU's picks per token (12.5 with
  the shipped ranking, 5.8-7.9 with tiering) without a further measurable gain in decode rate on this machine.
- `token_ms` after the run: 26.1 ms per token, of which the main GPU waits 4.7 ms, computes experts for 1.1 ms and
  spends 20.3 ms on the rest (a difference, not measured by itself). Tiering can only reach the waits.
- No pick was computed from the file side once the arena had followed use.

To repeat it: start the server twice with the same arguments, once with `DSX_TIER=0`, alternate A-B-A-B with a
restart per phase, and compare `decode_tok_s` of the same requests. `DSX_TIER=0` leaves the state on disk untouched.
One machine and one workload; A is this build with tiering off, not the commit before it.
