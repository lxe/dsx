//! The GPU engine behind the server: one `GpuModel` plus a bounded prefix cache.
//!
//! Two cache modes share the same physical engine (one engine thread, one `Real`, one `GpuModel`, one request at a
//! time):
//!
//! * `--cache-sessions 1` (default, backward compatible): the KV of the current conversation stays on the GPU, and a
//!   bounded history of prompt-boundary checkpoints lets a request continue, exact-retry or rewind inside that one
//!   lineage. This is the fast path and does no host parking.
//! * `--cache-sessions N > 1`: several independent lineages are retained. Exactly one of them is live in the
//!   `GpuModel`; the others are parked in a bounded [`HostCacheArena`] as full backing states, with host-backed prompt
//!   checkpoints and sparse prefix anchors. Switching `A -> B -> A` restores `A` instead of prefilling it from token
//!   0, and a new agent that shares a long system/tools prefix can rewind to a prefix anchor of another lineage.
//!
//! `ParkedState` (full backing state), `PromptCheckpoint` (rewind state + boundary logits) and `PrefixAnchor` (rewind
//! state, no logits) are different abstractions on purpose; see `gpumodel` and `session_cache`.
//!
//! The request flow is sequential: select the best session/prefix point, park the active session (memory admission
//! happens *before* parking), restore/rewind the selected point, prefill only the uncached tail, capture periodic
//! anchors while a long input advances, save the prompt-boundary checkpoint, decode, and only append tokens that went
//! through `gm.step` to the cached sequence.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;

use super::engine::{Engine, Finish, GenParams, GenStats};
use super::prefix::{
    best_rewind_point, choose_prefix_reuse, classify_lineage, longest_common_prefix, select_session, CacheAction,
    CheckpointPosition, LineageAction, RewindPoint, SessionCandidate, StateKind,
};
use super::sample::Sampler;
use super::session_cache::{CacheContextFingerprint, PrefixAnchor, PromptCheckpoint, SessionCache, SessionId};
use crate::gpumodel::{GpuModel, HostRewindState, Placement, Snapshot, VisionWeights};
use crate::hostcache::HostCacheArena;
use crate::model::Model;

/// Mirrors the `gen` command's flags.
#[derive(Debug, Clone)]
pub struct RealEngineArgs {
    pub model: PathBuf,
    pub engram: Option<PathBuf>,
    pub ctx: usize,
    pub chunk: usize,
    pub gpu0_gb: f64,
    pub gpu1_gb: f64,
    pub arena_gb: f64,
    pub rank: Option<PathBuf>,
    /// vision weights: `vision.safetensors` or the official checkpoint directory (None = text only)
    pub vision: Option<PathBuf>,
    /// retained conversation lineages, including the active one (`1` = the current single-lineage behaviour)
    pub cache_sessions: usize,
    /// hard host cache budget for inactive lineages (decimal GB): committed arena bytes and the logical host payload
    /// (arena extents, saved logits, Engram histories) are bounded by it
    pub session_cache_gb: f64,
    /// periodic sparse prefix-anchor interval in prompt tokens (`0` = anchors off)
    pub prefix_anchor_tokens: usize,
}

/// Prompt-boundary snapshots kept for rewinding in the single-lineage fast path. A snapshot copies every layer's
/// window ring and compressor state: about 5.6 MB on the shipped 40-layer model, so eight costs ~45 MB of GPU0 memory.
const PREFIX_CHECKPOINTS: usize = 8;

/// Multi-session mode keeps the same number of prompt checkpoints per lineage, but they live in the host arena.
const MAX_PROMPT_CHECKPOINTS: usize = 8;

/// Bounded prefix anchors per lineage. At 16K tokens and ~5.6 MB per anchor this caps one lineage near 90 MB; the
/// whole arena still obeys `--session-cache-gb`.
const MAX_PREFIX_ANCHORS: usize = 16;

/// Host slabs are committed lazily in this size, so a small cache never allocates the whole budget at startup.
const SESSION_CACHE_SLAB_BYTES: usize = 256 << 20;

struct PrefixCheckpoint {
    pos: usize,
    snapshot: Snapshot,
    /// next-token logits at the prompt boundary; an exact retry resumes generation without replaying the prompt
    logits: Vec<f32>,
}

impl CheckpointPosition for PrefixCheckpoint {
    fn pos(&self) -> usize {
        self.pos
    }
}

struct Real {
    gm: GpuModel<'static>,
    /// single-lineage fast path (`--cache-sessions 1`)
    cached: Vec<u32>,
    checkpoints: Vec<PrefixCheckpoint>,
    /// multi-lineage cache (`--cache-sessions > 1`); None in the fast path
    multi: Option<SessionCache>,
    stats: GenStats,
    vision: Option<VisionWeights>,
}

// GPU state is created and used only on the engine thread (EngineHandle::spawn builds us there).
unsafe impl Send for Real {}

/// Runs on the engine thread (`EngineHandle::spawn`), so GPU contexts are created there.
pub fn real_engine(args: &RealEngineArgs) -> Result<Box<dyn Engine>> {
    let m: &'static Model = Box::leak(Box::new(if args.model.is_dir() {
        Model::load_hf(&args.model)?
    } else {
        Model::load(&args.model, args.engram.as_deref())?
    }));
    let pl = Placement { gpu0_gb: args.gpu0_gb, aux_gb: args.gpu1_gb, arena_gb: args.arena_gb, prefill_chunk: args.chunk, rank: args.rank.clone() };
    let vision = args.vision.as_deref().map(VisionWeights::load).transpose()?;
    let gm = GpuModel::with_placement(m, -1, args.ctx, pl)?;
    let multi = if args.cache_sessions > 1 {
        let fingerprint = CacheContextFingerprint::build(&args.model, args.vision.as_deref(), &m.cfg, m.engram.is_some());
        let budget = (args.session_cache_gb.max(0.0) * 1e9) as usize;
        let anchors = if args.prefix_anchor_tokens == 0 { 0 } else { args.prefix_anchor_tokens };
        tracing::info!(
            event = "session_cache",
            action = "configured",
            sessions_limit = args.cache_sessions,
            arena_budget_bytes = budget,
            prefix_anchor_tokens = anchors,
            cache_context = fingerprint.describe(),
            "multi-session prefix cache"
        );
        Some(SessionCache::new(
            fingerprint,
            args.cache_sessions,
            HostCacheArena::pinned(budget, SESSION_CACHE_SLAB_BYTES),
            anchors,
            MAX_PROMPT_CHECKPOINTS,
            MAX_PREFIX_ANCHORS,
        ))
    } else {
        None
    };
    Ok(Box::new(Real { gm, cached: vec![], checkpoints: Vec::new(), multi, stats: GenStats::default(), vision }))
}

/// The image-aware prefix-cache key: every image span position carries the same token id, so key those positions by
/// the image content. Two different images with identical placeholder ids never produce a false cache hit.
fn prompt_key(prompt: &[u32], images: &[dsx_core::vision::ImageInput]) -> Vec<u32> {
    let mut key = prompt.to_vec();
    for im in images {
        if im.start + im.len() > key.len() {
            continue;
        }
        for (i, k) in key[im.start..im.start + im.len()].iter_mut().enumerate() {
            *k = 0xf000_0000 | ((im.hash as u32 ^ (im.hash >> 32) as u32 ^ i as u32) & 0x0fff_ffff);
        }
    }
    key
}

impl Real {
    /// Record the model state after an input prompt. Reusing the last checkpoint at the same position keeps an
    /// exact-retry request from allocating a duplicate snapshot.
    fn save_prompt_checkpoint(&mut self, pos: usize, logits: &[f32]) -> Result<()> {
        if self.checkpoints.last().is_some_and(|c| c.pos == pos) {
            return Ok(());
        }
        let snapshot = self.gm.snapshot()?;
        self.checkpoints.push(PrefixCheckpoint { pos, snapshot, logits: logits.to_vec() });
        while self.checkpoints.len() > PREFIX_CHECKPOINTS {
            self.checkpoints.remove(0);
        }
        Ok(())
    }

    fn generate_single(&mut self, prompt: &[u32], p: &GenParams, emit: &mut dyn FnMut(u32) -> bool) -> Result<Finish> {
        let t0 = Instant::now();
        anyhow::ensure!(!prompt.is_empty(), "empty prompt");
        let key = prompt_key(prompt, &p.images);

        let raw_lcp = longest_common_prefix(&key, &self.cached);
        let model_pos = self.gm.pos;
        let decision = choose_prefix_reuse(prompt.len(), self.cached.len(), raw_lcp, model_pos, &self.checkpoints);

        if raw_lcp < prompt.len().min(self.cached.len()) {
            tracing::info!(first_diff = raw_lcp, new_token_id = key[raw_lcp], cached_token_id = self.cached[raw_lcp], "prefix cache mismatch");
            if tracing::enabled!(tracing::Level::DEBUG) {
                let lo = raw_lcp.saturating_sub(8);
                let hi = (raw_lcp + 8).min(key.len()).min(self.cached.len());
                tracing::debug!(new_window = ?&key[lo..hi], cached_window = ?&self.cached[lo..hi], "prefix cache mismatch window");
            }
        }
        tracing::info!(
            prompt_len = prompt.len(),
            cached_len = self.cached.len(),
            raw_lcp,
            current_model_pos = model_pos,
            selected_checkpoint_pos = decision.start,
            replayed_tokens = decision.replayed_tokens,
            checkpoint_count = decision.checkpoint_count,
            retained_checkpoints = decision.retain_checkpoints,
            cache_action = decision.action.as_str(),
            exact_retry = decision.exact_retry,
            "prefix cache"
        );

        // Restore before touching the history: a failed restore leaves every checkpoint available for recovery.
        let mut saved_logits = None;
        match decision.checkpoint_index {
            Some(i) => {
                let cp = &self.checkpoints[i];
                self.gm.restore(&cp.snapshot)?;
                if decision.replayed_tokens == 0 {
                    saved_logits = Some(cp.logits.clone());
                }
            }
            None if decision.action == CacheAction::ColdReset => self.gm.reset(),
            None => {}
        }
        // Checkpoints after the divergence belong to the old branch and cannot be restored for this lineage.
        self.checkpoints.truncate(decision.retain_checkpoints);

        // images not already in the KV cache go through the vision tower; their span rows replace the embeddings
        self.gm.images.clear();
        for im in p.images.iter().filter(|im| im.start + im.len() > decision.start) {
            let vw = self.vision.as_ref().ok_or_else(|| anyhow::anyhow!("image input needs the vision weights (--vision)"))?;
            let tv = Instant::now();
            let rows = self.gm.encode_image(vw, im)?;
            tracing::info!(patches = im.n_vit_h * im.n_vit_w, tokens = im.len(), ms = tv.elapsed().as_millis() as u64, "image encoded");
            self.gm.images.push((im.start, rows));
        }
        let result = if decision.replayed_tokens == 0 {
            match saved_logits {
                Some(l) => Ok(l),
                None => match self.checkpoints.last().filter(|c| c.pos == prompt.len()) {
                    Some(cp) => Ok(cp.logits.clone()),
                    None => self.gm.read_logits(),
                },
            }
        } else {
            let pb = self.gm.pb.take().expect("prefill buffers");
            let r = self.gm.prefill(&pb, &prompt[decision.start..]);
            self.gm.pb = Some(pb);
            r
        };
        self.gm.images.clear();
        let logits = result?;
        self.save_prompt_checkpoint(prompt.len(), &logits)?;
        self.cached = key;
        let prefill = t0.elapsed();
        let td = Instant::now();
        let mut sampler = Sampler::from_params(p);
        let mut fin = Finish::Length;
        let mut hes = p.hesitation.clone();
        let mut raw = logits;
        for _ in 0..p.max_new {
            // raw model logits stay untouched; the hesitation penalty only mutates this token's sampling copy
            let tok = match &hes {
                Some(h) => {
                    let mut sampling = raw.clone();
                    for &i in h.ids.iter() {
                        if let Some(l) = sampling.get_mut(i as usize) {
                            *l -= h.lambda;
                        }
                    }
                    sampler.sample(&sampling)
                }
                None => sampler.sample(&raw),
            };
            if hes.as_ref().is_some_and(|h| tok == h.close) {
                hes = None;
            }
            if !emit(tok) {
                fin = Finish::Cancelled;
                break;
            }
            if self.gm.pos + 1 >= self.gm.max_ctx() {
                break;
            }
            raw = self.gm.step(tok)?;
            self.cached.push(tok);
        }
        self.stats = GenStats { prefill: Some(prefill), decode: Some(td.elapsed()), cached_prompt_tokens: decision.start };
        Ok(fin)
    }

    fn generate_multi(&mut self, prompt: &[u32], p: &GenParams, emit: &mut dyn FnMut(u32) -> bool) -> Result<Finish> {
        // Take the manager out so the GPU model and the cache can be borrowed independently.
        let mut cache = self.multi.take().expect("multi cache");
        let r = multi_request(&mut self.gm, self.vision.as_ref(), &mut cache, prompt, p, emit);
        if r.is_err() {
            // the device state and the active lineage's `cached` tokens may disagree after a failed prefill or rewind:
            // forget that lineage so the next request restores a parked session or prefills cold
            if let Some(id) = cache.active {
                cache.drop_session(id);
            }
        }
        self.multi = Some(cache);
        let (fin, stats) = r?;
        self.stats = stats;
        Ok(fin)
    }
}

impl Engine for Real {
    fn generate(&mut self, prompt: &[u32], p: &GenParams, emit: &mut dyn FnMut(u32) -> bool) -> Result<Finish> {
        if self.multi.is_some() {
            self.generate_multi(prompt, p, emit)
        } else {
            self.generate_single(prompt, p, emit)
        }
    }

    fn last_stats(&self) -> GenStats {
        GenStats { prefill: self.stats.prefill, decode: self.stats.decode, cached_prompt_tokens: self.stats.cached_prompt_tokens }
    }
}

/// What the request does with a matched lineage or the empty cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RequestClass {
    Cold,
    Mutate,
    NewFromPrefix,
}

impl RequestClass {
    fn as_str(self) -> &'static str {
        match self {
            Self::Cold => "cold",
            Self::Mutate => "mutate",
            Self::NewFromPrefix => "new-lineage-from-prefix",
        }
    }
}

/// Park the active lineage into the host arena. Admission runs first (count and byte caps, LRU victims), and the GPU
/// state is only copied after it succeeded. Returns the parked payload bytes and the wall time in milliseconds.
fn park_active(gm: &mut GpuModel<'static>, cache: &mut SessionCache, keep: &[SessionId], add_sessions: usize) -> (usize, u64) {
    let Some(active) = cache.active else { return (0, 0) };
    // prefill-only requests (cancelled before the first step) leave their raw logits in the session entry, not in the
    // decode logits buffer: park what the live state actually produced.
    let stored_logits = cache.session(active).and_then(|s| s.live_logits.clone());
    let required_arena = gm.parked_state_bytes();
    let required_heap = gm.parked_heap_bytes();
    let required = required_arena + required_heap;
    if let Err(e) = cache.plan_admission(keep, required_arena, required_heap, add_sessions) {
        cache.reject_admission(Some(active), e, required);
        // the session is not retainable; the caller resets/reuses the GPU state, so the lineage is dropped instead of
        // leaving useless anchors behind. Only hit rate changes, never correctness.
        cache.drop_session(active);
        cache.active = None;
        return (0, 0);
    }
    let t = Instant::now();
    let logits = match stored_logits {
        Some(l) => l,
        None => match gm.read_logits() {
            Ok(l) => l,
            Err(e) => {
                cache.reject_admission_reason(Some(active), "host-allocation-failed", required);
                tracing::warn!(error = %e, "park could not read the live logits");
                cache.drop_session(active);
                cache.active = None;
                return (0, 0);
            }
        },
    };
    let parked = match gm.park_state(&mut cache.arena, logits) {
        Ok(p) => p,
        Err(e) => {
            cache.reject_admission_reason(Some(active), "host-allocation-failed", required);
            tracing::warn!(error = %e, "park failed");
            cache.drop_session(active);
            cache.active = None;
            return (0, 0);
        }
    };
    let ms = t.elapsed().as_millis() as u64;
    let bytes = parked.payload_bytes();
    let Some(idx) = cache.session_index(active) else {
        parked.free(&mut cache.arena);
        cache.active = None;
        return (0, 0);
    };
    let pos = cache.sessions[idx].cached.len();
    let clk = cache.clock();
    cache.sessions[idx].parked = Some(parked);
    cache.sessions[idx].live_logits = None;
    cache.sessions[idx].live_pos = pos;
    cache.sessions[idx].last_used = clk;
    cache.active = None;
    cache.stats.park_count += 1;
    cache.stats.park_bytes += bytes as u64;
    cache.stats.park_ns += t.elapsed().as_nanos() as u64;
    tracing::info!(
        event = "session_cache",
        action = "park",
        session_id = active,
        park_bytes = bytes,
        park_ms = ms,
        arena_payload_bytes = cache.arena.payload_bytes(),
        arena_committed_bytes = cache.arena.committed_bytes(),
        arena_budget_bytes = cache.arena.budget_bytes(),
        sessions_retained = cache.sessions.len(),
        "session parked"
    );
    (bytes, ms)
}

/// Prefill the uncached prompt tail in chunks and capture sparse prefix anchors at chunk boundaries that cross the
/// configured interval. Anchors are captured only for input history (never for generated tokens); the prompt boundary
/// is covered by the prompt checkpoint instead. Copied anchors go straight to the host arena.
fn prefill_with_anchors(
    gm: &mut GpuModel<'static>,
    arena: &mut HostCacheArena,
    tokens: &[u32],
    anchor_interval: usize,
    existing_anchors: usize,
    max_anchors: usize,
) -> Result<(Vec<f32>, Vec<(usize, HostRewindState)>)> {
    anyhow::ensure!(!tokens.is_empty(), "empty prompt tail");
    let c = &gm.m.cfg;
    let cone = (c.n_layers - 1 - c.cand_layer) * c.window;
    let start = gm.pos;
    let final_pos = start + tokens.len();
    let pb = gm.pb.take().expect("prefill buffers");
    let mut anchors: Vec<(usize, HostRewindState)> = Vec::new();
    let mut logits: Option<Vec<f32>> = None;
    let mut s0 = 0usize;
    let mut next_target = if anchor_interval > 0 && existing_anchors < max_anchors {
        (start / anchor_interval + 1) * anchor_interval
    } else {
        usize::MAX
    };
    let mut anchor_failed = false;
    while s0 < tokens.len() {
        let mut e = (s0 + pb.tmax).min(tokens.len());
        if e < tokens.len() && tokens.len() - e < cone {
            e = tokens.len().saturating_sub(cone).max(s0 + 1);
        }
        let last = e == tokens.len();
        match gm.prefill_chunk(&pb, &tokens[s0..e], last) {
            Ok(v) => {
                if let Some(v) = v {
                    logits = Some(v);
                }
            }
            Err(e) => {
                gm.pb = Some(pb);
                for (_, st) in anchors.drain(..) {
                    st.free(arena);
                }
                return Err(e);
            }
        }
        s0 = e;
        let abs = gm.pos;
        // the final chunk ends on the prompt boundary (the prompt checkpoint covers it)
        if !anchor_failed && abs >= next_target && abs < final_pos && anchor_interval > 0 && existing_anchors + anchors.len() < max_anchors {
            match gm.capture_rewind(arena) {
                Ok(state) => {
                    let bytes = state.payload_bytes();
                    anchors.push((abs, state));
                    // deterministic maximum spacing: interval + at most one chunk
                    next_target = abs + anchor_interval;
                    tracing::debug!(event = "session_cache", action = "anchor", anchor_pos = abs, anchor_bytes = bytes, "prefix anchor captured");
                }
                Err(e) => {
                    anchor_failed = true;
                    tracing::info!(event = "session_cache_admission", result = "rejected", reason = "host-allocation-failed", anchor_pos = abs, error = %e, "prefix anchor capture stopped");
                }
            }
        }
    }
    gm.pb = Some(pb);
    let logits = logits.ok_or_else(|| anyhow::anyhow!("empty prompt tail"))?;
    Ok((logits, anchors))
}

/// Execute one request against the multi-lineage cache.
fn multi_request(
    gm: &mut GpuModel<'static>,
    vision: Option<&VisionWeights>,
    cache: &mut SessionCache,
    prompt: &[u32],
    p: &GenParams,
    emit: &mut dyn FnMut(u32) -> bool,
) -> Result<(Finish, GenStats)> {
    let t0 = Instant::now();
    anyhow::ensure!(!prompt.is_empty(), "empty prompt");
    let key = prompt_key(prompt, &p.images);
    cache.begin_request();
    let evict_base = cache.stats.session_evictions;
    let create_base = cache.stats.session_creates;
    let switch_base = cache.stats.session_switches;

    struct Match {
        cand: SessionCandidate,
        raw_lcp: usize,
        point: RewindPoint,
    }
    let mut matches: Vec<Match> = Vec::new();
    for s in cache.sessions.iter() {
        // a lineage from another execution context is not a candidate
        if s.ctx != cache.fingerprint {
            continue;
        }
        let active = cache.active == Some(s.id);
        if !active && s.parked.is_none() {
            continue;
        }
        let raw_lcp = longest_common_prefix(&key, &s.cached);
        let full_state_pos = if active { Some(gm.pos) } else { Some(s.cached.len()) };
        let cps: Vec<usize> = s.prompt_checkpoints.iter().map(|c| c.pos).collect();
        let aps: Vec<usize> = s.anchors.iter().map(|a| a.pos).collect();
        let point = best_rewind_point(prompt.len(), s.cached.len(), raw_lcp, full_state_pos, &cps, &aps);
        matches.push(Match {
            cand: SessionCandidate { session_id: s.id, usable_start: point.pos, kind: point.kind, active, last_used: s.last_used },
            raw_lcp,
            point,
        });
    }
    let selection = select_session(&matches.iter().map(|m| m.cand).collect::<Vec<_>>());
    let (class, source_id, raw_lcp, point) = match selection {
        Some(i) => {
            let m = &matches[i];
            let cached_len = cache.session(m.cand.session_id).map_or(0, |s| s.cached.len());
            let class = match classify_lineage(m.point, m.raw_lcp, cached_len) {
                LineageAction::Mutate => RequestClass::Mutate,
                LineageAction::NewFromPrefix => RequestClass::NewFromPrefix,
                LineageAction::ColdStart => RequestClass::Cold,
            };
            (class, Some(m.cand.session_id), m.raw_lcp, m.point)
        }
        None => (RequestClass::Cold, None, 0, RewindPoint { pos: 0, kind: StateKind::Cold, has_logits: false }),
    };
    let active_before = cache.active;
    let mut park_bytes = 0usize;
    let mut park_ms = 0u64;
    let mut restore_bytes = 0usize;
    let mut restore_ms = 0u64;
    let mut source_used = None;
    let mut invalidated = (0usize, 0usize);

    // ---- state preparation: park, restore, rewind ----
    let mut session_id = source_id;
    let mut usable_start = point.pos;
    let mut state_kind = point.kind;
    let mut saved_logits: Option<Vec<f32>> = None;
    let mutated = matches!(class, RequestClass::Mutate);

    if class == RequestClass::Mutate {
        let target = source_id.ok_or_else(|| anyhow::anyhow!("session cache: mutate without a session"))?;
        let target_active = cache.active == Some(target);
        if !target_active {
            // switch: park the current active lineage, then restore the target's full backing state
            let keep = [target];
            let (pb, pm) = park_active(gm, cache, &keep, 0);
            park_bytes = pb;
            park_ms = pm;
            let idx = cache.session_index(target).ok_or_else(|| anyhow::anyhow!("session cache: mutate target survives admission"))?;
            let parked = cache.sessions[idx].parked.take();
            match parked {
                Some(ps) => {
                    let t = Instant::now();
                    match gm.restore_state(&cache.arena, &ps) {
                        Ok(()) => {
                            restore_ms = t.elapsed().as_millis() as u64;
                            restore_bytes = ps.payload_bytes();
                            cache.stats.restore_count += 1;
                            cache.stats.restore_bytes += restore_bytes as u64;
                            cache.stats.restore_ns += t.elapsed().as_nanos() as u64;
                            cache.sessions[idx].live_logits = Some(ps.logits.clone());
                        }
                        Err(e) => {
                            // a corrupt host state only costs a hit: reset and prefill from token 0
                            tracing::warn!(session_id = target, error = %e, "restore failed, falling back to a cold prefill");
                            gm.reset();
                            usable_start = 0;
                            state_kind = StateKind::Cold;
                            saved_logits = None;
                        }
                    }
                    ps.free(&mut cache.arena);
                    cache.sessions[idx].parked = None;
                    cache.set_active(target);
                }
                None => {
                    gm.reset();
                    cache.set_active(target);
                }
            }
        }
        let idx = cache.session_index(target).ok_or_else(|| anyhow::anyhow!("session cache: mutate target exists"))?;
        // rewind to the selected checkpoint/anchor (the live state needs no rewind)
        match state_kind {
            StateKind::PromptCheckpoint => {
                let cpi = cache.sessions[idx]
                    .prompt_checkpoints
                    .iter()
                    .rposition(|c| c.pos == usable_start)
                    .ok_or_else(|| anyhow::anyhow!("selected prompt checkpoint vanished"))?;
                gm.restore_rewind(&cache.arena, &cache.sessions[idx].prompt_checkpoints[cpi].state)?;
                if prompt.len() == usable_start {
                    saved_logits = Some(cache.sessions[idx].prompt_checkpoints[cpi].logits.clone());
                }
            }
            StateKind::PrefixAnchor => {
                let ai = cache.sessions[idx]
                    .anchors
                    .iter()
                    .rposition(|a| a.pos == usable_start)
                    .ok_or_else(|| anyhow::anyhow!("selected prefix anchor vanished"))?;
                gm.restore_rewind(&cache.arena, &cache.sessions[idx].anchors[ai].state)?;
            }
            StateKind::Live => {
                if prompt.len() == usable_start {
                    saved_logits = cache.sessions[idx]
                        .live_logits
                        .clone()
                        .or_else(|| gm.read_logits().ok());
                }
            }
            StateKind::Cold => {
                gm.reset();
                usable_start = 0;
            }
        }
        invalidated = cache.invalidate_after(idx, raw_lcp);
        session_id = Some(target);
    } else if class == RequestClass::NewFromPrefix {
        let source = source_id.ok_or_else(|| anyhow::anyhow!("session cache: prefix source"))?;
        let source_was_active = cache.active == Some(source);
        if cache.active.is_some() {
            let keep: Vec<SessionId> = if cache.active == Some(source) { vec![] } else { vec![source] };
            let (pb, pm) = park_active(gm, cache, &keep, 1);
            park_bytes = pb;
            park_ms = pm;
        }
        source_used = Some(source);
        if cache.session_index(source).is_none() {
            // the source was not retainable: fall back to a cold start, still correct
            gm.reset();
            usable_start = 0;
            state_kind = StateKind::Cold;
            saved_logits = None;
        }
        if let Some(src_idx) = cache.session_index(source) {
            if !source_was_active {
                let parked = cache.sessions[src_idx].parked.as_ref();
                match parked {
                    Some(ps) => {
                        // the source's host copy stays intact (copy-on-restore); only the active GPU sequence diverges
                        let t = Instant::now();
                        match gm.restore_state(&cache.arena, ps) {
                            Ok(()) => {
                                restore_ms = t.elapsed().as_millis() as u64;
                                restore_bytes = ps.payload_bytes();
                                cache.stats.restore_count += 1;
                                cache.stats.restore_bytes += restore_bytes as u64;
                                cache.stats.restore_ns += t.elapsed().as_nanos() as u64;
                            }
                            Err(e) => {
                                tracing::warn!(session_id = source, error = %e, "source restore failed, falling back to a cold prefill");
                                gm.reset();
                                usable_start = 0;
                                state_kind = StateKind::Cold;
                                saved_logits = None;
                            }
                        }
                    }
                    None => {
                        gm.reset();
                        usable_start = 0;
                        state_kind = StateKind::Cold;
                    }
                }
            } else if cache.sessions[src_idx].parked.is_none() {
                // the park failed: the device state is gone and the lineage cannot serve a prefix
                gm.reset();
                usable_start = 0;
                state_kind = StateKind::Cold;
            }
            match state_kind {
                StateKind::PromptCheckpoint => {
                    let ai = cache.sessions[src_idx].prompt_checkpoints.iter().rposition(|c| c.pos == usable_start);
                    if let Some(ai) = ai {
                        gm.restore_rewind(&cache.arena, &cache.sessions[src_idx].prompt_checkpoints[ai].state)?;
                        if prompt.len() == usable_start {
                            saved_logits = Some(cache.sessions[src_idx].prompt_checkpoints[ai].logits.clone());
                        }
                    } else {
                        gm.reset();
                        usable_start = 0;
                        state_kind = StateKind::Cold;
                    }
                }
                StateKind::PrefixAnchor => {
                    let ai = cache.sessions[src_idx].anchors.iter().rposition(|a| a.pos == usable_start);
                    if let Some(ai) = ai {
                        gm.restore_rewind(&cache.arena, &cache.sessions[src_idx].anchors[ai].state)?;
                    } else {
                        gm.reset();
                        usable_start = 0;
                        state_kind = StateKind::Cold;
                    }
                }
                _ => {}
            }
        }
    } else {
        // cold start: retain the active lineage first, then reset the physical sequence
        if cache.active.is_some() {
            let (pb, pm) = park_active(gm, cache, &[], 1);
            park_bytes = pb;
            park_ms = pm;
        }
        gm.reset();
        usable_start = 0;
        state_kind = StateKind::Cold;
    }
    let replayed_tokens = prompt.len() - usable_start;
    // A cold reset leaves no append-only compressed/index rows behind, so every rewind point of a lineage that had to
    // fall back to a cold reset is invalid (the normal mutate path keeps points at or before the divergence).
    if mutated && state_kind == StateKind::Cold {
        if let Some(idx) = session_id.and_then(|sid| cache.session_index(sid)) {
            cache.invalidate_after(idx, 0);
            cache.sessions[idx].cached.clear();
        }
    }

    // ---- images: only spans that are not already in the KV cache ----
    gm.images.clear();
    for im in p.images.iter().filter(|im| im.start + im.len() > usable_start) {
        let vw = vision.ok_or_else(|| anyhow::anyhow!("image input needs the vision weights (--vision)"))?;
        let tv = Instant::now();
        let rows = gm.encode_image(vw, im)?;
        tracing::info!(patches = im.n_vit_h * im.n_vit_w, tokens = im.len(), ms = tv.elapsed().as_millis() as u64, "image encoded");
        gm.images.push((im.start, rows));
    }

    // ---- prefill only the uncached tail (or reuse saved raw logits) ----
    let mut anchors: Vec<(usize, HostRewindState)> = Vec::new();
    let logits = if usable_start == prompt.len() {
        match saved_logits.take() {
            Some(l) => l,
            None => gm.read_logits()?,
        }
    } else {
        // only a mutated lineage continues its own anchor series; a new lineage starts one of its own
        let existing = if mutated { session_id.and_then(|id| cache.session(id)).map_or(0, |s| s.anchors.len()) } else { 0 };
        let (l, a) = prefill_with_anchors(gm, &mut cache.arena, &prompt[usable_start..], cache.anchor_interval, existing, cache.max_anchors)?;
        anchors = a;
        l
    };
    gm.images.clear();

    // ---- session entry: mutate the existing lineage, or start a new one ----
    if let Some(sid) = session_id {
        if mutated {
            let idx = cache.session_index(sid).ok_or_else(|| anyhow::anyhow!("session cache: mutated session exists"))?;
            for (pos, st) in anchors.drain(..) {
                cache.push_anchor(idx, PrefixAnchor { pos, state: st });
            }
            cache.sessions[idx].cached = key.clone();
            cache.sessions[idx].live_pos = gm.pos;
        }
    }
    if !mutated || session_id.is_none() {
        let key_heap = key.len() * std::mem::size_of::<u32>();
        if cache.sessions.len() >= cache.max_sessions {
            cache.stats.session_uncacheable += 1;
            cache.reject_admission_reason(None, "session-count-full", 0);
            cache.active = None;
            session_id = None;
        } else if !cache.fits_combined(key_heap) {
            cache.stats.session_uncacheable += 1;
            cache.reject_admission_reason(None, "cache-budget", key_heap);
            cache.active = None;
            session_id = None;
        } else {
            let sid = cache.create_session(key.clone(), gm.pos);
            let idx = cache.session_index(sid).ok_or_else(|| anyhow::anyhow!("session cache: new session"))?;
            for (pos, st) in anchors.drain(..) {
                cache.push_anchor(idx, PrefixAnchor { pos, state: st });
            }
            session_id = Some(sid);
        }
    }
    if !anchors.is_empty() {
        for (_, st) in anchors.drain(..) {
            st.free(&mut cache.arena);
        }
    }

    // ---- prompt-boundary checkpoint with raw logits ----
    let duplicate_boundary = mutated && state_kind == StateKind::PromptCheckpoint && usable_start == prompt.len();
    let checkpoint_heap = logits.len() * std::mem::size_of::<f32>();
    let checkpoint_arena = gm.rewind_state_bytes();
    if let Some(sid) = session_id {
        if !duplicate_boundary && !cache.fits_combined_with_arena(checkpoint_arena, checkpoint_heap) {
            cache.reject_admission_reason(Some(sid), "cache-budget", checkpoint_heap);
        } else if !duplicate_boundary {
            match gm.capture_rewind(&mut cache.arena) {
                Ok(state) => {
                    let idx = cache.session_index(sid).ok_or_else(|| anyhow::anyhow!("session cache: session for checkpoint"))?;
                    cache.push_checkpoint(idx, PromptCheckpoint { pos: prompt.len(), logits: logits.clone(), state });
                }
                Err(e) => {
                    // a full host cache only costs the checkpoint, not correctness
                    cache.reject_admission_reason(Some(sid), "host-allocation-failed", gm.rewind_state_bytes());
                    tracing::warn!(error = %e, "prompt checkpoint not stored");
                }
            }
        }
    }

    // ---- decode: raw logits stay separate from the penalized sampling copy ----
    let td = Instant::now();
    let mut sampler = Sampler::from_params(p);
    let mut fin = Finish::Length;
    let mut hes = p.hesitation.clone();
    let mut raw = logits;
    let mut generated = 0usize;
    for _ in 0..p.max_new {
        let tok = match &hes {
            Some(h) => {
                let mut sampling = raw.clone();
                for &i in h.ids.iter() {
                    if let Some(l) = sampling.get_mut(i as usize) {
                        *l -= h.lambda;
                    }
                }
                sampler.sample(&sampling)
            }
            None => sampler.sample(&raw),
        };
        if hes.as_ref().is_some_and(|h| tok == h.close) {
            hes = None;
        }
        if !emit(tok) {
            fin = Finish::Cancelled;
            break;
        }
        // a token that never went through `step` (cancelled or context full) is not part of the model state
        if gm.pos + 1 >= gm.max_ctx() {
            break;
        }
        raw = gm.step(tok)?;
        generated += 1;
        if let Some(sid) = session_id {
            if let Some(s) = cache.session_mut(sid) {
                s.cached.push(tok);
                s.live_pos = gm.pos;
            }
        }
    }
    let live_logits = raw;
    if let Some(sid) = session_id {
        cache.hit(sid);
        if let Some(s) = cache.session_mut(sid) {
            s.live_pos = gm.pos;
            s.live_logits = Some(live_logits);
        }
    }

    // ---- metadata, statistics and one structured request event ----
    let selected_state = state_kind;
    if selected_state == StateKind::Cold {
        cache.stats.session_misses += 1;
        cache.stats.cold_starts += 1;
    } else {
        cache.stats.session_hits += 1;
    }
    match selected_state {
        StateKind::Live => cache.stats.live_hits += 1,
        StateKind::PromptCheckpoint => cache.stats.prompt_checkpoint_hits += 1,
        StateKind::PrefixAnchor => cache.stats.prefix_anchor_hits += 1,
        StateKind::Cold => {}
    }
    cache.stats.replayed_tokens += replayed_tokens as u64;
    cache.stats.cached_prompt_tokens += usable_start as u64;

    let (retained, payload, committed, budget) = cache.accounting();
    let session_fields = session_id.and_then(|id| cache.session(id)).map(|s| {
        (
            s.cached.len(),
            s.live_pos,
            s.last_used,
            s.prompt_checkpoints.len(),
            s.checkpoint_payload_bytes(),
            s.anchors.len(),
            s.anchor_payload_bytes(),
            s.parked_payload_bytes(),
        )
    });
    let (cached_tokens, live_pos, last_used, cp_count, cp_bytes, an_count, an_bytes, parked_bytes) =
        session_fields.unwrap_or((0, 0, 0, 0, 0, 0, 0, 0));
    let s = cache.stats.clone();
    let session_created = s.session_creates.saturating_sub(create_base);
    let session_evicted = s.session_evictions.saturating_sub(evict_base);
    let session_switch = s.session_switches.saturating_sub(switch_base);
    let parked_all: usize = cache.sessions.iter().map(|x| x.parked_payload_bytes()).sum();
    let checkpoints_all: usize = cache.sessions.iter().map(|x| x.checkpoint_payload_bytes()).sum();
    let anchors_all: usize = cache.sessions.iter().map(|x| x.anchor_payload_bytes()).sum();
    let heap_all: usize = cache.sessions.iter().map(|x| x.heap_bytes()).sum();
    let heap_self = cache.session(session_id.unwrap_or(0)).map_or(0, |x| x.heap_bytes());
    tracing::info!(
        event = "session_cache",
        session_id = session_id.unwrap_or(0),
        source_session_id = source_used.unwrap_or(0),
        action = class.as_str(),
        same_lineage_mutation = mutated,
        new_lineage_from_prefix = class == RequestClass::NewFromPrefix,
        selected_was_active = active_before.is_some() && active_before == source_id,
        raw_lcp,
        usable_start,
        selected_state_kind = selected_state.as_str(),
        replayed_tokens,
        prompt_tokens = prompt.len(),
        cached_prompt_tokens = usable_start,
        cached_tokens,
        live_pos,
        last_used,
        generated_tokens = generated,
        invalidated_checkpoints = invalidated.0,
        invalidated_anchors = invalidated.1,
        park_bytes,
        park_payload_bytes = park_bytes,
        park_ms,
        restore_bytes,
        restore_payload_bytes = restore_bytes,
        restore_ms,
        arena_payload_bytes = payload,
        arena_committed_bytes = committed,
        arena_budget_bytes = budget,
        sessions_retained = retained,
        sessions_limit = cache.max_sessions,
        prompt_checkpoint_count = cp_count,
        prompt_checkpoint_bytes = cp_bytes,
        prefix_anchor_count = an_count,
        prefix_anchor_bytes = an_bytes,
        parked_payload_bytes = parked_bytes,
        parked_full_state_bytes = parked_all,
        prompt_checkpoint_bytes_all = checkpoints_all,
        prefix_anchor_bytes_all = anchors_all,
        heap_bytes = heap_self,
        heap_bytes_all = heap_all,
        session_created = session_created > 0,
        session_switch = session_switch > 0,
        session_evicted = session_evicted,
        sessions_created = s.session_creates,
        session_hits = s.session_hits,
        session_misses = s.session_misses,
        session_switches = s.session_switches,
        session_evictions = s.session_evictions,
        session_uncacheable = s.session_uncacheable,
        cache_context = cache.fingerprint.describe(),
        "session cache request"
    );
    let stats = GenStats { prefill: Some(t0.elapsed()), decode: Some(td.elapsed()), cached_prompt_tokens: usable_start };
    Ok((fin, stats))
}
