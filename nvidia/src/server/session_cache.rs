//! Multi-session prefix cache manager: bounded conversation lineages, host-backed parked states, prompt checkpoints
//! and sparse prefix anchors.
//!
//! Three levels of cached state are kept apart on purpose:
//!
//! * [`ParkedState`] — the complete backing state of an independent lineage (host arena). Needed for `A -> B -> A`.
//! * `PromptCheckpoint` — the existing prompt-boundary rewind point **with** boundary logits. Needed for exact retry,
//!   tool-call rewind and branch divergence inside one lineage.
//! * `PrefixAnchor` — a periodic lightweight rewind point **without** logits. Needed to share long common prefixes
//!   between independent agents (system/tools prefix + different histories).
//!
//! Exactly one retained session is physically live in the `GpuModel`; every other retained session lives in the
//! [`HostCacheArena`]. The manager owns metadata, admission, LRU eviction and accounting only: GPU copies are issued
//! by `gpumodel` and the request orchestration lives in `real.rs`.

use std::path::Path;

use crate::gpumodel::{HostRewindState, ParkedState};
use crate::hostcache::HostCacheArena;
use crate::model::Config;

use super::session_policy::{self, AdmissionError, Eviction, EvictionReason, SessionMeta};

/// Internal stable session id. Deterministic, monotonically increasing.
pub type SessionId = u64;

/// How the execution context of a cached lineage is identified. Lineages are only candidates when their fingerprint
/// equals the engine's current one, so a cache hit never mixes states produced under different model semantics.
///
/// Included: model identity/config, vision presence/identity and semantic runtime options that change numerics.
/// Excluded on purpose: performance-only runtime state (profiling, logging, expert residency, adaptive cache,
/// statistics), which changes speed but not model state.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CacheContextFingerprint {
    pub model: u64,
    pub vision: Option<u64>,
    /// semantic runtime options (`DSX_DEFER`, `DSX_ENGRAM_FP8`, `DSX_DENSE_Q6`, ...)
    pub numerics: u64,
}

impl CacheContextFingerprint {
    /// Build the fingerprint for one engine lifetime. The struct is an extension point for future LoRA/adapters and
    /// model variants: add another identity field and compare it in the same place.
    pub fn build(
        model_path: &Path,
        vision_path: Option<&Path>,
        cfg: &Config,
        has_engram: bool,
    ) -> Self {
        let mut model = fnv(b"dsx-model-v1");
        fnv_bytes(&mut model, model_path.to_string_lossy().as_bytes());
        file_identity(&mut model, model_path);
        fnv_usize(&mut model, cfg.vocab);
        fnv_usize(&mut model, cfg.dim);
        fnv_usize(&mut model, cfg.inter);
        fnv_usize(&mut model, cfg.n_layers);
        fnv_usize(&mut model, cfg.n_heads);
        fnv_usize(&mut model, cfg.head_dim);
        fnv_usize(&mut model, cfg.window);
        fnv_usize(&mut model, cfg.idx_dim);
        fnv_usize(&mut model, cfg.idx_heads);
        fnv_usize(&mut model, cfg.cand_layer);
        fnv_usize(&mut model, cfg.n_experts);
        fnv_usize(&mut model, cfg.top_k);
        for &r in &cfg.compress_ratios {
            fnv_usize(&mut model, r);
        }
        for &s in &cfg.kv_sources {
            fnv_usize(&mut model, s);
        }
        for &s in &cfg.index_sources {
            fnv_usize(&mut model, s);
        }
        for &s in &cfg.engram_layers {
            fnv_usize(&mut model, s);
        }
        fnv_usize(&mut model, cfg.engram_dim);
        fnv_usize(&mut model, cfg.engram_ngram);
        fnv_usize(&mut model, has_engram as usize);
        let vision = vision_path.map(|p| {
            let mut v = fnv(b"dsx-vision-v1");
            fnv_bytes(&mut v, p.to_string_lossy().as_bytes());
            file_identity(&mut v, p);
            v
        });
        let mut numerics = fnv(b"dsx-numerics-v1");
        // options that change model outputs/state; deliberately not profiling, logging, expert placement or
        // performance-only fusions (their math is exact), so changing speed knobs never invalidates a cached lineage
        for name in [
            "DSX_DEFER",
            "DSX_ROUTE_BIAS",
            "DSX_ENGRAM_FP8",
            "DSX_DENSE_Q6",
            "DSX_QB_ROPE",
            "DSX_OLD_ATTN",
        ] {
            fnv_bytes(&mut numerics, name.as_bytes());
            match std::env::var(name) {
                Ok(v) => {
                    fnv_bytes(&mut numerics, b"=");
                    fnv_bytes(&mut numerics, v.as_bytes());
                }
                Err(_) => fnv_bytes(&mut numerics, b"<unset>"),
            }
        }
        Self {
            model,
            vision,
            numerics,
        }
    }

    pub fn describe(&self) -> String {
        format!(
            "model:{:016x},vision:{},numerics:{:016x}",
            self.model,
            self.vision
                .map_or_else(|| "none".to_string(), |v| format!("{v:016x}")),
            self.numerics
        )
    }
}

/// FNV-1a 64-bit: stable across runs and machines, enough for an identity fingerprint.
fn fnv(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn fnv_bytes(h: &mut u64, bytes: &[u8]) {
    for &b in bytes {
        *h ^= b as u64;
        *h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
}

fn fnv_usize(h: &mut u64, v: usize) {
    fnv_bytes(h, &(v as u64).to_le_bytes());
}

/// Path + size + mtime: a different checkpoint at the same path invalidates the cache.
fn file_identity(h: &mut u64, path: &Path) {
    fnv_bytes(h, path.to_string_lossy().as_bytes());
    if let Ok(md) = std::fs::metadata(path) {
        fnv_bytes(h, &md.len().to_le_bytes());
        if let Ok(t) = md.modified() {
            let d = t.duration_since(std::time::UNIX_EPOCH).unwrap_or_default();
            fnv_bytes(h, &d.as_secs().to_le_bytes());
            fnv_bytes(h, &d.subsec_nanos().to_le_bytes());
        }
    }
    // a directory (HF checkpoint) lists its files
    if path.is_dir() {
        if let Ok(rd) = std::fs::read_dir(path) {
            let mut names: Vec<String> = rd
                .flatten()
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .collect();
            names.sort();
            for n in names {
                fnv_bytes(h, n.as_bytes());
            }
        }
    }
}

/// A prompt-boundary checkpoint: a lightweight rewind state plus the raw boundary logits.
pub struct PromptCheckpoint {
    pub pos: usize,
    pub logits: Vec<f32>,
    pub state: HostRewindState,
}

/// A sparse periodic prefix anchor: a lightweight rewind state without logits. Used to replay only the divergent
/// tail of another agent's shared system/tools prefix.
pub struct PrefixAnchor {
    pub pos: usize,
    pub state: HostRewindState,
}

/// One retained conversation lineage.
pub struct SessionEntry {
    pub id: SessionId,
    pub ctx: CacheContextFingerprint,
    /// image-aware cache key of the processed sequence (prompt + generated tokens that went through `gm.step`)
    pub cached: Vec<u32>,
    pub prompt_checkpoints: Vec<PromptCheckpoint>,
    pub anchors: Vec<PrefixAnchor>,
    /// full backing state while this lineage is inactive
    pub parked: Option<ParkedState>,
    /// raw next-token logits of the live state (never penalized sampling logits)
    pub live_logits: Option<Vec<f32>>,
    /// model position of the live state (equals `cached.len()` while active)
    pub live_pos: usize,
    pub last_used: u64,
    pub created: u64,
    pub hits: u64,
}

impl SessionEntry {
    pub fn payload_bytes(&self) -> usize {
        self.parked.as_ref().map_or(0, |p| p.payload_bytes())
            + self
                .prompt_checkpoints
                .iter()
                .map(|c| c.state.payload_bytes())
                .sum::<usize>()
            + self
                .anchors
                .iter()
                .map(|a| a.state.payload_bytes())
                .sum::<usize>()
    }

    pub fn parked_payload_bytes(&self) -> usize {
        self.parked.as_ref().map_or(0, |p| p.payload_bytes())
    }

    pub fn checkpoint_payload_bytes(&self) -> usize {
        self.prompt_checkpoints
            .iter()
            .map(|c| c.state.payload_bytes())
            .sum()
    }

    pub fn anchor_payload_bytes(&self) -> usize {
        self.anchors.iter().map(|a| a.state.payload_bytes()).sum()
    }

    /// Heap bytes of the session metadata (not arena extents): the cached key, raw logits and the full Engram history
    /// of a parked state. Saved logits are part of the configured budget.
    pub fn heap_bytes(&self) -> usize {
        let word = std::mem::size_of::<u32>();
        let f32b = std::mem::size_of::<f32>();
        let i64b = std::mem::size_of::<i64>();
        let mut bytes = self.cached.len() * word;
        bytes += self.live_logits.as_ref().map_or(0, |l| l.len() * f32b);
        bytes += self
            .prompt_checkpoints
            .iter()
            .map(|c| c.logits.len() * f32b)
            .sum::<usize>();
        if let Some(p) = &self.parked {
            bytes += p.engram.len() * i64b + p.logits.len() * f32b;
        }
        bytes
    }
}

/// Counters for the request log and local validation.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    pub requests: u64,
    pub session_hits: u64,
    pub session_misses: u64,
    pub session_switches: u64,
    pub session_creates: u64,
    pub session_evictions: u64,
    pub session_uncacheable: u64,
    pub live_hits: u64,
    pub prompt_checkpoint_hits: u64,
    pub prefix_anchor_hits: u64,
    pub cold_starts: u64,
    pub replayed_tokens: u64,
    pub cached_prompt_tokens: u64,
    pub park_count: u64,
    pub park_bytes: u64,
    pub park_ns: u64,
    pub restore_count: u64,
    pub restore_bytes: u64,
    pub restore_ns: u64,
    pub evictions_by_count: u64,
    pub evictions_by_memory: u64,
    pub rejected_too_large: u64,
    pub rejected_alloc: u64,
}

/// Bounds and host cache of the lineage manager.
pub struct SessionCache {
    pub fingerprint: CacheContextFingerprint,
    /// retained lineages including the active one
    pub max_sessions: usize,
    /// `0` disables prefix anchors
    pub anchor_interval: usize,
    /// prompt-boundary checkpoints kept per lineage
    pub max_prompt_checkpoints: usize,
    /// prefix anchors kept per lineage (bounded; anchors are never kept forever)
    pub max_anchors: usize,
    pub arena: HostCacheArena,
    pub sessions: Vec<SessionEntry>,
    pub active: Option<SessionId>,
    pub stats: CacheStats,
    clock: u64,
    next_id: SessionId,
}

impl SessionCache {
    pub fn new(
        fingerprint: CacheContextFingerprint,
        max_sessions: usize,
        arena: HostCacheArena,
        anchor_interval: usize,
        max_prompt_checkpoints: usize,
        max_anchors: usize,
    ) -> Self {
        Self {
            fingerprint,
            max_sessions: max_sessions.max(1),
            anchor_interval,
            max_prompt_checkpoints: max_prompt_checkpoints.max(1),
            max_anchors,
            arena,
            sessions: Vec::new(),
            active: None,
            stats: CacheStats::default(),
            clock: 0,
            next_id: 1,
        }
    }

    /// Advance the LRU clock once per request; returns the current clock value.
    pub fn begin_request(&mut self) -> u64 {
        self.clock += 1;
        self.stats.requests += 1;
        self.clock
    }

    pub fn clock(&self) -> u64 {
        self.clock
    }

    pub fn session_index(&self, id: SessionId) -> Option<usize> {
        self.sessions.iter().position(|s| s.id == id)
    }

    pub fn session(&self, id: SessionId) -> Option<&SessionEntry> {
        self.sessions.iter().find(|s| s.id == id)
    }

    pub fn session_mut(&mut self, id: SessionId) -> Option<&mut SessionEntry> {
        self.sessions.iter_mut().find(|s| s.id == id)
    }

    pub fn create_session(&mut self, cached: Vec<u32>, live_pos: usize) -> SessionId {
        let id = self.next_id;
        self.next_id += 1;
        self.stats.session_creates += 1;
        self.sessions.push(SessionEntry {
            id,
            ctx: self.fingerprint.clone(),
            cached,
            prompt_checkpoints: Vec::new(),
            anchors: Vec::new(),
            parked: None,
            live_logits: None,
            live_pos,
            last_used: self.clock,
            created: self.clock,
            hits: 0,
        });
        self.active = Some(id);
        id
    }

    pub fn touch(&mut self, id: SessionId) {
        let clock = self.clock;
        if let Some(s) = self.session_mut(id) {
            s.last_used = clock;
        }
    }

    pub fn hit(&mut self, id: SessionId) {
        let clock = self.clock;
        if let Some(s) = self.session_mut(id) {
            s.hits += 1;
            s.last_used = clock;
        }
    }

    /// Metadata for the pure admission/LRU policy.
    pub fn metas(&self) -> Vec<SessionMeta> {
        let active = self.active;
        self.sessions
            .iter()
            .map(|s| SessionMeta {
                id: s.id,
                active: Some(s.id) == active,
                last_used: s.last_used,
                payload_bytes: s.payload_bytes(),
                heap_bytes: s.heap_bytes(),
            })
            .collect()
    }

    /// Plan (and immediately apply) the evictions needed to park the active session and/or add a lineage.
    ///
    /// Runs **before** any allocation: the caller learns the victims, the arena never grows past the budget and the
    /// request still runs correctly if the session is rejected. `keep` protects sessions that the request is about to
    /// use (a prefix source, or the lineage being restored).
    pub fn plan_admission(
        &mut self,
        keep: &[SessionId],
        required_arena: usize,
        required_heap: usize,
        add_sessions: usize,
    ) -> Result<Vec<Eviction>, AdmissionError> {
        let metas = self.metas();
        let budget = self.arena.budget_bytes();
        // logical payload (arena extents + heap metadata/logits) and the committed arena bound are both checked; the
        // arena itself refuses to commit past the budget, so free space inside slabs can never hide overshoot
        let required_payload = required_arena.saturating_add(required_heap);
        let plan = session_policy::plan_admission(
            &metas,
            keep,
            required_payload,
            add_sessions,
            budget,
            self.max_sessions,
        )?;
        // committed growth a single contiguous requirement would need; the arena itself enforces the hard bound
        let planned_growth = match self.arena.plan(required_arena.max(1)) {
            Ok(crate::hostcache::AllocPlan::Grow(n)) => n,
            Ok(crate::hostcache::AllocPlan::Reuse) => 0,
            Err(_) => 0,
        };
        // The policy bounds logical payload; committed arena memory is a separate hard bound, and the arena's free
        // space is part of RAM pressure. If committing the parking requirement plus the surviving heap would pass the
        // budget, evict more LRU victims (conservative: slab memory freed by evictions is not assumed).
        let mut victims = plan.victims.clone();
        let heap_of = |id: SessionId| {
            metas
                .iter()
                .find(|m| m.id == id)
                .map_or(0, |m| m.heap_bytes)
        };
        let mut heap_after = metas.iter().map(|m| m.heap_bytes).sum::<usize>() + required_heap;
        for v in &victims {
            heap_after = heap_after.saturating_sub(heap_of(v.id));
        }
        let growth = planned_growth;
        let committed = self.arena.committed_bytes();
        if committed + growth + heap_after > budget {
            for m in session_policy::eviction_order(&metas, keep) {
                if victims.iter().any(|v| v.id == m.id) {
                    continue;
                }
                victims.push(Eviction {
                    id: m.id,
                    payload_bytes: m.payload_bytes,
                    reason: EvictionReason::MemoryBudget,
                });
                heap_after = heap_after.saturating_sub(m.heap_bytes);
                if committed + growth + heap_after <= budget {
                    break;
                }
            }
            if committed + growth + heap_after > budget {
                return Err(AdmissionError::BudgetExhausted {
                    payload: plan.payload,
                    budget,
                });
            }
        }
        for v in &victims {
            self.evict_session(v.id, v.reason);
        }
        self.arena.release_empty_slabs();
        tracing::debug!(
            event = "session_cache_admission",
            result = "accepted",
            retained = plan.retained,
            payload_after = plan.payload,
            planned_committed_growth = planned_growth,
            arena_payload_bytes = self.arena.payload_bytes(),
            arena_committed_bytes = self.arena.committed_bytes(),
            arena_budget_bytes = budget,
            "session cache admission accepted"
        );
        Ok(victims)
    }

    /// Log a rejected admission (the request still runs; if the active session cannot be parked its GPU state is
    /// simply reused/reset, which loses a future hit but never correctness).
    pub fn reject_admission(
        &mut self,
        session_id: Option<SessionId>,
        err: AdmissionError,
        required: usize,
    ) {
        self.reject_admission_reason(session_id, err.as_str(), required);
    }

    pub fn reject_admission_reason(
        &mut self,
        session_id: Option<SessionId>,
        reason: &'static str,
        required: usize,
    ) {
        if reason == "state-too-large" {
            self.stats.rejected_too_large += 1;
        } else {
            self.stats.rejected_alloc += 1;
        }
        self.stats.session_uncacheable += 1;
        tracing::info!(
            event = "session_cache_admission",
            result = "rejected",
            reason = reason,
            session_id = session_id.unwrap_or(0),
            required_bytes = required,
            arena_payload_bytes = self.arena.payload_bytes(),
            arena_committed_bytes = self.arena.committed_bytes(),
            arena_budget_bytes = self.arena.budget_bytes(),
            "session cache admission rejected"
        );
    }

    /// Free a session without counting it as an LRU eviction (used when its backing state is not retainable).
    pub fn drop_session(&mut self, id: SessionId) {
        if let Some(idx) = self.session_index(id) {
            self.take_session_state(idx);
            self.sessions.remove(idx);
            if self.active == Some(id) {
                self.active = None;
            }
        }
        self.arena.release_empty_slabs();
    }

    /// Free every extent that belongs to a session and detach its metadata. Returns the freed payload bytes.
    fn take_session_state(&mut self, idx: usize) -> usize {
        let e = &mut self.sessions[idx];
        let mut freed = 0usize;
        if let Some(p) = e.parked.take() {
            freed += p.payload_bytes();
            p.free(&mut self.arena);
        }
        while let Some(cp) = e.prompt_checkpoints.pop() {
            freed += cp.state.payload_bytes();
            cp.state.free(&mut self.arena);
        }
        while let Some(a) = e.anchors.pop() {
            freed += a.state.payload_bytes();
            a.state.free(&mut self.arena);
        }
        e.live_logits = None;
        freed
    }

    /// Whole-session eviction (the v1 eviction unit).
    pub fn evict_session(&mut self, id: SessionId, reason: EvictionReason) -> usize {
        let Some(idx) = self.session_index(id) else {
            return 0;
        };
        let freed_heap = self.sessions[idx].heap_bytes();
        let freed = self.take_session_state(idx);
        self.sessions.remove(idx);
        if self.active == Some(id) {
            self.active = None;
        }
        self.stats.session_evictions += 1;
        match reason {
            EvictionReason::SessionCount => self.stats.evictions_by_count += 1,
            EvictionReason::MemoryBudget => self.stats.evictions_by_memory += 1,
        }
        self.arena.release_empty_slabs();
        tracing::info!(
            event = "session_cache_evict",
            session_id = id,
            reason = reason.as_str(),
            freed_bytes = freed,
            freed_heap_bytes = freed_heap,
            sessions_retained = self.sessions.len(),
            arena_payload_bytes = self.arena.payload_bytes(),
            arena_committed_bytes = self.arena.committed_bytes(),
            arena_budget_bytes = self.arena.budget_bytes(),
            "session cache eviction"
        );
        freed
    }

    /// Branch invalidation inside a lineage: checkpoints and anchors after the divergence point are invalid and their
    /// host extents are released. Returns how many of each were dropped.
    pub fn invalidate_after(&mut self, idx: usize, divergence: usize) -> (usize, usize) {
        let mut dropped_cp = 0usize;
        let mut dropped_an = 0usize;
        {
            let e = &mut self.sessions[idx];
            while e
                .prompt_checkpoints
                .last()
                .is_some_and(|c| c.pos > divergence)
            {
                if let Some(c) = e.prompt_checkpoints.pop() {
                    c.state.free(&mut self.arena);
                    dropped_cp += 1;
                }
            }
            let mut keep = Vec::new();
            let mut drop = Vec::new();
            for a in e.anchors.drain(..) {
                if a.pos > divergence {
                    drop.push(a);
                } else {
                    keep.push(a);
                }
            }
            e.anchors = keep;
            for a in drop {
                a.state.free(&mut self.arena);
                dropped_an += 1;
            }
        }
        (dropped_cp, dropped_an)
    }

    /// Store a prompt-boundary checkpoint; a checkpoint at the same position is replaced instead of duplicated, and
    /// the oldest one is dropped when the per-lineage bound is exceeded.
    pub fn push_checkpoint(&mut self, idx: usize, cp: PromptCheckpoint) {
        let max = self.max_prompt_checkpoints;
        let e = &mut self.sessions[idx];
        if e.prompt_checkpoints.last().is_some_and(|c| c.pos == cp.pos) {
            if let Some(old) = e.prompt_checkpoints.pop() {
                old.state.free(&mut self.arena);
            }
        }
        e.prompt_checkpoints.push(cp);
        while e.prompt_checkpoints.len() > max {
            let old = e.prompt_checkpoints.remove(0);
            old.state.free(&mut self.arena);
        }
    }

    /// Store a periodic prefix anchor; the per-lineage cap bounds anchor memory. When the cap is reached the anchor is
    /// released again (the earliest anchors are the useful ones for a shared prefix, so capture stops, not rotates).
    pub fn push_anchor(&mut self, idx: usize, anchor: PrefixAnchor) -> bool {
        let max = self.max_anchors;
        let e = &mut self.sessions[idx];
        if e.anchors.len() >= max {
            anchor.state.free(&mut self.arena);
            return false;
        }
        e.anchors.push(anchor);
        true
    }

    pub fn set_active(&mut self, id: SessionId) {
        if self.active != Some(id) {
            self.stats.session_switches += 1;
        }
        self.active = Some(id);
    }

    /// Global accounting for the request log.
    pub fn accounting(&self) -> (usize, usize, usize, usize) {
        (
            self.sessions.len(),
            self.arena.payload_bytes(),
            self.arena.committed_bytes(),
            self.arena.budget_bytes(),
        )
    }

    /// Host memory the cache already holds: committed arena slabs plus heap metadata/logits/Engram histories. Free
    /// space inside slabs is committed memory and therefore counted.
    pub fn combined_used_bytes(&self) -> usize {
        self.arena.committed_bytes() + self.sessions.iter().map(|s| s.heap_bytes()).sum::<usize>()
    }

    /// Whether `extra_heap` bytes of metadata/logits still fit the hard budget together with committed arena memory.
    pub fn fits_combined(&self, extra_heap: usize) -> bool {
        self.combined_used_bytes().saturating_add(extra_heap) <= self.arena.budget_bytes()
    }

    /// Whether an arena extent of `extra_arena` bytes plus `extra_heap` heap bytes fits the hard budget. The arena
    /// plan accounts for a new slab when no free extent is large enough.
    pub fn fits_combined_with_arena(&self, extra_arena: usize, extra_heap: usize) -> bool {
        let growth = match self.arena.plan(extra_arena.max(1)) {
            Ok(crate::hostcache::AllocPlan::Grow(n)) => n,
            Ok(crate::hostcache::AllocPlan::Reuse) => 0,
            Err(_) => return false,
        };
        self.combined_used_bytes()
            .saturating_add(growth)
            .saturating_add(extra_heap)
            <= self.arena.budget_bytes()
    }
}
