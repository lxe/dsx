# Multi-session prefix cache (NVIDIA server)

`dsx-server` keeps one physical sequence: one engine thread, one `Real`, one `GpuModel`, one request at a time. On
top of that, `--cache-sessions N > 1` retains several independent conversation lineages so sequential coding-agent
sessions can switch back to long histories without prefilling them from token 0.

```text
                    incoming prompt
                          |
                          v
                cache-context fingerprint     (model/config/vision/numeric identity)
                          |
                          v
                image-aware token key
                          |
                          v
               session/prefix selector
                          |
       +------------------+------------------+
       |                  |                  |
       v                  v                  v
 active lineage     parked lineage      common-prefix
                                           source
       |                  |                  |
       +------------------+------------------+
                          |
                          v
                 one active GpuModel
                          |
                +---------+---------+
                |                   |
                v                   v
        PromptCheckpoint       PrefixAnchor
       exact retry/branch      sparse prefix
                              sharing/replay
                          |
                          v
                bounded HostCacheArena
                          |
           +--------------+--------------+
           |                             |
           v                             v
   --cache-sessions             --session-cache-gb
      count bound                  hard byte bound
                          |
                          v
                  whole-session LRU
```

Two scenarios are covered:

1. `A -> B -> A`: A is parked when B runs, restored when A returns; only the new tail is prefilled.
2. `A` and `B` share a large system/tools prefix: B rewinds to a sparse `PrefixAnchor` of A and prefills only the
   divergent tail.

The cache is bounded by two independent limits: `--cache-sessions` (retained lineages, including the active one) and
`--session-cache-gb` (hard host-cache bytes). Admission checks the logical payload (arena extents plus heap-side
cached keys, saved logits and Engram histories) against the budget, and the arena separately refuses to commit past
it, so free space inside slabs cannot hide overshoot. Eviction only changes latency/hit rate; it never changes
correctness.

## Modes

| Mode | Behaviour |
|---|---|
| `--cache-sessions 1` (default) | Backward-compatible fast path: the KV of the current conversation stays on the GPU, prompt-boundary checkpoints are device snapshots, no host arena, no session switches. |
| `--cache-sessions N > 1` | Host-backed full parked states, host-backed prompt checkpoints and prefix anchors; one lineage live in the `GpuModel`, every other retained lineage in the arena. |

`--prefix-anchor-tokens 16384` sets the sparse anchor interval (`0` disables anchors). Anchors are only used in
multi-session mode.

## The three levels of cached state

| Type | Contents | Purpose |
|---|---|---|
| `ParkedState` | full backing state of an independent lineage: position, every window ring, the **used** compressed/index cache rows, compressor carry, `n_comp`/`n_idx`, full Engram history, raw next-token logits | `A -> B -> A` session switching; also the source backing state for a common-prefix clone |
| `PromptCheckpoint` | lightweight rewind state (`window` rings, compressor carry, counters, `engram_len`) **plus** boundary logits | exact retry, tool-call rewind, branch divergence inside one lineage |
| `PrefixAnchor` | the same lightweight rewind state **without** logits | periodic sparse points for sharing long common prefixes between independent agents |

Only used rows are copied. Row bytes come from the runtime `Config`: `n_comp * head_dim * 2`, `n_idx * idx_dim * 2`
for each KV/index source layer; rings are `window * head_dim * 2` per layer; compressor carry is
`2 * ratio * head_dim * 4` per ratio-`>1` source layer. Nothing is hardcoded for a specific checkpoint.

## Persistent state audit

Persistent (copied into `ParkedState`):

```text
pos
window ring (every layer)
used comp_cache rows, used idx_cache rows
compressor carry / state
n_comp, n_idx
engram_hist (the full Vec<i64>, not a length)
raw next-token logits
```

Not persistent, deliberately rebuilt:

```text
seq_base                 monotonic handoff sequence number; never rewound on a session switch
adaptive expert state    residency, replacement state, decayed usage/hotness, adapt flags, placement
images / img_rows        request-local image rows
dpost, post_attn_done, post_ffn_done
verify_t, vpass, vsnap, vrows
scratch buffers, CUDA events, candidate/top-k/prefill scratch
```

`Snapshot` stays the lightweight intra-lineage rewind type. It is not aliased to `ParkedState`: after `A -> B`,
B overwrites the device compressed/index/engram buffers, so a lightweight snapshot alone cannot restore A. That is
why full parking exists as a separate abstraction.

## HostCacheArena

Inactive state lives in host RAM, preferably in pinned CUDA-registered slabs:

- one arena owns all extents (no per-session `HostBuf` register/unregister churn);
- slabs are committed lazily and are clamped to the configured budget;
- extent allocation reuses the smallest fitting free span, coalesces free spans and drops a slab when its last extent
  is released, so committed bytes can fall again;
- two accountings are exposed: `payload_bytes` (extents handed out) and `committed_bytes` (all slab memory, including
  free space), and the hard budget bounds the committed bytes;
- copies are grouped: one DtoH batch per park and one HtoD batch per restore, with a single synchronization at the
  park/restore boundary (never per row or per layer).

## Admission and eviction

Admission runs **before** parking:

1. measure the payload requirement of the active session;
2. plan LRU victims with the pure policy in `server/session_policy.rs`;
3. enforce the session count cap and the byte budget together;
4. evict whole inactive sessions (`reason=session-count` or `reason=memory-budget`);
5. only then copy the active state into the arena.

A lineage whose own backing state exceeds the whole budget is not retained (`session_cache_admission result=rejected
reason=state-too-large`); its anchors/checkpoints are dropped and returning to it costs a cold prefill. Allocation
failures are treated the same way (`reason=host-allocation-failed`). Memory pressure can only lower the hit rate.

The eviction unit is a whole inactive session. `eviction_score` is the only comparison point (`last_used` in v1), so a
future policy can weigh recompute cost, session size, hit count or priority without rewriting the manager.

## Prefix anchors

Anchors are captured at prefill chunk boundaries:

- the target position is the next multiple of the interval after the resume point;
- when a chunk crosses it, the anchor is captured at the chunk end (`pos >= target`), so the documented maximum
  spacing is the interval plus at most one chunk;
- anchors are captured only for input history, never for generated tokens; the prompt boundary is covered by the
  prompt checkpoint;
- at most `MAX_PREFIX_ANCHORS` per lineage are kept; the earliest ones are the useful ones for a shared prefix, so
  capture stops when the cap is reached;
- captured anchors are written straight into the host arena.

Branch invalidation drops anchors after the divergence point. Whole-session eviction frees all of them. A new lineage
created from a shared prefix keeps the source anchors intact and gathers its own.

## Cache-context fingerprint

A lineage is a candidate only when its `CacheContextFingerprint` equals the engine's current one. The fingerprint
covers model identity/config (vocab, dimensions, layers, window, head dim, index dim, compress ratios, KV/index/Engram
source layers), Engram presence, vision presence/identity, and semantic runtime options that change numerics
(`DSX_DEFER`, `DSX_ROUTE_BIAS`, `DSX_ENGRAM_FP8`, `DSX_DENSE_Q6`, `DSX_QB_ROPE`, `DSX_OLD_ATTN`). Profiling,
logging, expert residency and performance-only fusions are intentionally excluded: they change speed, not model
state. The struct is the extension point for future LoRA/adapters or model variants.

The prompt key is token-wise with image spans replaced by image-content hashes, so two different images with the same
placeholder token ids never produce a false hit.

## Session selection and lineage rule

For every compatible session the engine computes the true longest common prefix (`raw_lcp`) and the latest usable
resume point among:

1. the full live/parked state at the cached end (`live`);
2. the newest prompt checkpoint at or before `raw_lcp` (with boundary logits);
3. the newest prefix anchor at or before `raw_lcp` (no logits).

The latest position wins; a prompt checkpoint wins a tie (it can serve an exact retry with no replay). Across
sessions the longest usable prefix wins, then the active session (no park/restore switch), then the most recently
used session, then the stable internal id.

Same-lineage vs new-lineage (deterministic, no client ids v1):

- `raw_lcp == cached_len`: the prompt carries the whole cached history (continuation/exact replay) -> mutate the same
  entry;
- the selected point is a prompt checkpoint -> tool-call retry/branch inside the same lineage -> mutate;
- otherwise only a shared prefix anchor (or nothing) matched -> a new lineage is created and the source session stays
  parked and intact.

A new lineage from A's anchor restores A's backing state, rewinds to the anchor on the GPU, prefills only B's tail and
creates a new entry; A's host copy is not modified (cheap copy-on-restore).

## Raw logits and cancellation

Persistent logits are raw model logits. The hesitation penalty is applied to a per-token sampling copy only, so a
parked/live state never stores penalized logits. A token that is emitted but does not go through `gm.step` (client
cancelled, context full) is not appended to `cached`, `live_pos` or the parked history.

## Observability

One request event (`event=session_cache`) reports the selected session/source, action, decision inputs
(`raw_lcp`, `usable_start`), the selected state kind, replayed tokens, park/restore bytes and milliseconds, arena
payload/committed/budget, retained sessions and per-session checkpoint/anchor/parked accounting. Parking, anchor
capture, evictions and admission rejections have their own structured events.

## Validation

Runtime validation was performed locally on a running server; the validation harness is not part of this change.

## Known limitations and future work

- No vLLM-style paged KV, radix tree, block refcounting, disk/remote cache or concurrent generation.
- Eviction works on whole sessions; anchors are not evicted individually.
- Whole-session parking copies used rows; restore copies the full session even when only a prefix anchor is needed for
  a clone.
- Anchor positions are chunk-aligned, not bit-exact multiples of the interval.
- No TTL: bounds are the session count, the byte budget and LRU only.
- Future optimizations: radix/block prefix sharing, prioritized `eviction_score`, anchor rotation by measured hit
  rate, piping copy batches on the side stream, and page-locked arena preallocation under a warmup flag.
