//! The batched runner: many sequences on one MI325X with continuous batching (dsx-core's BatchEngine).
//!
//! Every sequence lives in a slot of a `Batch` (packed FP4 state, kernels/batch.h) or, while it is the only one
//! decoding or while its prompt is prefilled, in the single-sequence decoder (the "lane"). Each step:
//!   1. one prompt chunk (DSXR_CHUNK tokens, default 2048) of the oldest waiting prompt runs in the lane; a prompt
//!      that extends a finished slot's history (prefix reuse) starts from that slot's state
//!   2. every decoding sequence advances one token: batched (`batch_step`) when two or more decode or a prompt is
//!      waiting, else in the lane on the decode graph
//! A finished sequence keeps its slot's history until the slot is needed again (least recently used first).

use crate::batch::Batch;
use crate::decode::Decoder;
use anyhow::Result;
use dsx_core::server::engine::{BatchEngine, GenParams};
use dsx_core::server::sample::Sampler;
use std::collections::{HashMap, VecDeque};

pub(crate) static CHUNK_OVERRIDE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
pub(crate) static IDLE_CHUNK: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
pub(crate) static DECODE_ROUNDS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

#[derive(Default)]
struct PrefillCadence {
    remaining: usize,
}

impl PrefillCadence {
    fn due(&mut self, waiting: bool, decoding: bool, rounds: usize) -> bool {
        if !waiting || !decoding || rounds == 0 {
            self.remaining = 0;
            return waiting;
        }
        self.remaining = self.remaining.min(rounds);
        if self.remaining > 0 {
            self.remaining -= 1;
            false
        } else {
            self.remaining = rounds;
            true
        }
    }
}

enum Phase {
    /// waiting for / in the lane's prefill: prompt tokens not yet fed (from `done`)
    Prefill { prompt: Vec<u32>, done: usize },
    /// decoding: `next` is the newest token (emitted, not yet fed)
    Decode { next: u32 },
}

struct Seq {
    slot: usize,
    phase: Phase,
    /// tokens fed so far (the slot's history once parked)
    fed: Vec<u32>,
    sampler: Option<Sampler>,
    /// first stop token: emitted to end a sequence at the context limit
    stop: Option<u32>,
    cached: usize,
    /// speculative decoding state (prompt lookup, DSpark acceptance), from the end of the prefill
    spec: Option<crate::spec::SpecState>,
    // dashboard bookkeeping
    started: std::time::Instant,
    first: Option<std::time::Instant>,
    generated: usize,
    prompt_len: usize,
}

struct Slot {
    owner: Option<u64>,
    /// history of the state held in the slot (valid when `owner` is None and `parked`)
    tokens: Vec<u32>,
    parked: bool,
    used: u64,
}

pub struct Runner<'a> {
    dec: &'a mut Decoder,
    b: Batch,
    seqs: HashMap<u64, Seq>,
    slots: Vec<Slot>,
    /// sequences waiting for the lane's prefill, oldest first
    waiting: VecDeque<u64>,
    /// the sequence whose state is in the lane (prefilling, or decoding alone)
    lane: Option<u64>,
    chunk: usize,
    cadence: PrefillCadence,
    clock: u64,
    /// live numbers for the admin dashboard
    pub tel: Option<std::sync::Arc<crate::telemetry::Telemetry>>,
}

// the decoder is created and used only on the engine thread
unsafe impl Send for Runner<'_> {}

impl<'a> Runner<'a> {
    pub fn new(dec: &'a mut Decoder, slots: usize, ctx: usize) -> Result<Runner<'a>> {
        anyhow::ensure!(ctx <= dec.max_ctx, "batch context {ctx} > lane context {}", dec.max_ctx);
        crate::spec::init_tuning_from_env()?;
        let b = dec.new_batch(slots, ctx)?;
        let chunk = std::env::var("DSXR_CHUNK").ok().and_then(|v| v.parse().ok()).unwrap_or(2048);
        let idle = std::env::var("DSXR_IDLE_CHUNK").ok().map(|v| v.parse::<usize>()).transpose()?.unwrap_or(0);
        let rounds = std::env::var("DSXR_DECODE_ROUNDS").ok().map(|v| v.parse::<usize>()).transpose()?.unwrap_or(0);
        anyhow::ensure!(idle <= 8192 && rounds <= 64, "DSXR_IDLE_CHUNK must be 0..8192; DSXR_DECODE_ROUNDS must be 0..64");
        IDLE_CHUNK.store(idle, std::sync::atomic::Ordering::Relaxed);
        DECODE_ROUNDS.store(rounds, std::sync::atomic::Ordering::Relaxed);
        let slots = (0..slots).map(|_| Slot { owner: None, tokens: vec![], parked: false, used: 0 }).collect();
        Ok(Runner { dec, b, seqs: HashMap::new(), slots, waiting: VecDeque::new(), lane: None, chunk, cadence: PrefillCadence::default(), clock: 0, tel: None })
    }

    /// Move the lane's decoding sequence into its slot.
    fn park_lane(&mut self) -> Result<()> {
        if let Some(id) = self.lane.take() {
            let s = &self.seqs[&id];
            if matches!(s.phase, Phase::Decode { .. }) {
                self.dec.load_slot(&mut self.b, s.slot)?;
                self.slots[s.slot].parked = true;
            }
        }
        Ok(())
    }

    fn pick(&mut self, s: &mut Seq, logits: impl FnOnce(&Decoder) -> Result<Vec<f32>>, greedy: u32) -> Result<u32> {
        match s.sampler.as_mut() {
            Some(sm) => Ok(sm.sample(&logits(self.dec)?)),
            None => Ok(greedy),
        }
    }
}

impl BatchEngine for Runner<'_> {
    fn has_room(&self) -> bool {
        self.slots.iter().any(|s| s.owner.is_none())
    }

    fn start(&mut self, id: u64, prompt: &[u32], p: &GenParams) -> Result<()> {
        anyhow::ensure!(!prompt.is_empty(), "empty prompt");
        anyhow::ensure!(prompt.len() + 1 < self.b.ctx, "prompt of {} tokens does not fit the context of {}", prompt.len(), self.b.ctx);
        // a free slot whose history the prompt extends (longest), else the least recently used free slot
        let mut best: Option<(usize, usize)> = None;
        for (i, sl) in self.slots.iter().enumerate() {
            if sl.owner.is_some() {
                continue;
            }
            let reuse = if sl.parked && sl.tokens.len() < prompt.len() && prompt.starts_with(&sl.tokens) { sl.tokens.len() } else { 0 };
            let better = match best {
                None => true,
                Some((bi, br)) => reuse > br || (reuse == br && sl.used < self.slots[bi].used),
            };
            if better {
                best = Some((i, reuse));
            }
        }
        let (slot, reuse) = best.ok_or_else(|| anyhow::anyhow!("no free slot"))?;
        self.clock += 1;
        let sl = &mut self.slots[slot];
        sl.owner = Some(id);
        sl.used = self.clock;
        if reuse == 0 {
            sl.parked = false;
            sl.tokens.clear();
        }
        let sampler = if p.temperature > 0.0 { Some(Sampler::from_params(p)) } else { None };
        self.seqs.insert(id, Seq { slot, phase: Phase::Prefill { prompt: prompt.to_vec(), done: reuse }, fed: prompt[..reuse].to_vec(), sampler, stop: p.stop_token_ids.first().copied(), cached: reuse, spec: None, started: std::time::Instant::now(), first: None, generated: 0, prompt_len: prompt.len() });
        if let Some(t) = &self.tel {
            t.admitted(reuse);
        }
        self.waiting.push_back(id);
        Ok(())
    }

    fn step(&mut self) -> Result<Vec<(u64, u32)>> {
        let t0 = std::time::Instant::now();
        let spec_sum = |r: &Runner| r.seqs.values().filter_map(|s| s.spec.as_ref()).fold((0usize, 0usize), |a, x| (a.0 + x.drafted, a.1 + x.accepted));
        let (d0, k0) = spec_sum(self);
        let fed0: usize = self.seqs.values().map(|s| match &s.phase { Phase::Prefill { done, .. } => *done, _ => 0 }).sum();
        let pre: std::collections::HashSet<u64> = self.seqs.iter().filter(|(_, s)| matches!(s.phase, Phase::Prefill { .. })).map(|(&id, _)| id).collect();
        let out = self.step_inner()?;
        let now = std::time::Instant::now();
        for &(id, _) in &out {
            if let Some(s) = self.seqs.get_mut(&id) {
                s.first.get_or_insert(now);
                s.generated += 1;
            }
        }
        if let Some(t) = self.tel.clone() {
            let (d1, k1) = spec_sum(self);
            // prompt tokens fed this step: progress of prefilling sequences (finished ones fed their whole rest)
            let fed1: usize = self.seqs.iter().map(|(id, s)| match &s.phase {
                Phase::Prefill { done, .. } => *done,
                _ if pre.contains(id) => s.prompt_len,
                _ => 0,
            }).sum();
            let infos = self.seqs.iter().map(|(&id, s)| {
                let dec_s = s.first.map(|f| (now - f).as_secs_f64()).unwrap_or(0.0);
                let (dr, kp) = s.spec.as_ref().map(|x| (x.drafted, x.accepted)).unwrap_or((0, 0));
                crate::telemetry::SessionInfo {
                    id: format!("#{id}"),
                    gpu: 0,
                    slot: s.slot,
                    state: if matches!(s.phase, Phase::Prefill { .. }) { "prefill" } else { "decode" },
                    prompt_tokens: s.prompt_len,
                    cached_tokens: s.cached,
                    generated: s.generated,
                    tok_s: if dec_s > 0.0 && s.generated > 1 { (s.generated - 1) as f64 / dec_s } else { 0.0 },
                    acceptance: if dr > 0 { kp as f64 / dr as f64 } else { 0.0 },
                    age_s: (now - s.started).as_secs_f64(),
                    sampled: s.sampler.is_some(),
                }
            }).collect();
            t.step(out.len(), fed1.saturating_sub(fed0), t0.elapsed().as_secs_f64() * 1e3, 0.0, d1.saturating_sub(d0), k1.saturating_sub(k0), infos, 0);
        }
        Ok(out)
    }

    fn finish(&mut self, id: u64) {
        if let (Some(t), Some(s)) = (&self.tel, self.seqs.get(&id)) {
            let dec_s = s.first.map(|f| f.elapsed().as_secs_f64()).unwrap_or(0.0);
            t.finished(&format!("#{id}"), s.generated, if dec_s > 0.0 && s.generated > 1 { (s.generated - 1) as f64 / dec_s } else { 0.0 });
        }
        self.finish_inner(id);
    }

    fn cached_prompt_tokens(&self, id: u64) -> usize {
        self.seqs.get(&id).map(|s| s.cached).unwrap_or(0)
    }
}

impl Runner<'_> {
    fn step_inner(&mut self) -> Result<Vec<(u64, u32)>> {
        let mut out = vec![];
        // ---- 1. prefill: one chunk of the oldest waiting prompt in the lane
        let decoding = self.seqs.values().any(|s| matches!(s.phase, Phase::Decode { .. }));
        let rounds = DECODE_ROUNDS.load(std::sync::atomic::Ordering::Relaxed);
        let prefill_due = self.cadence.due(!self.waiting.is_empty(), decoding, rounds);
        if let Some(&wid) = self.waiting.front().filter(|_| prefill_due) {
            if self.lane != Some(wid) {
                self.park_lane()?;
                let (slot, done) = match &self.seqs[&wid].phase {
                    Phase::Prefill { done, .. } => (self.seqs[&wid].slot, *done),
                    _ => unreachable!(),
                };
                if done > 0 {
                    self.dec.unload_slot(&self.b, slot)?;
                } else {
                    self.dec.reset()?;
                }
                self.lane = Some(wid);
            }
            let s = self.seqs.get_mut(&wid).unwrap();
            let Phase::Prefill { prompt, done } = &mut s.phase else { unreachable!() };
            let override_chunk = CHUNK_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed);
            let idle_chunk = IDLE_CHUNK.load(std::sync::atomic::Ordering::Relaxed);
            let chunk = if !decoding && idle_chunk > 0 { idle_chunk } else if override_chunk == 0 { self.chunk } else { override_chunk };
            let end = (*done + chunk).min(prompt.len());
            let next = self.dec.prefill(&prompt[*done..end])?;
            s.fed.extend_from_slice(&prompt[*done..end]);
            *done = end;
            if end == prompt.len() {
                self.waiting.pop_front();
                let mut s = self.seqs.remove(&wid).unwrap();
                let tok = self.pick(&mut s, |d| d.logits(), next)?;
                s.phase = Phase::Decode { next: tok };
                let mut h = s.fed.clone();
                h.push(tok);
                let mut sp = crate::spec::SpecState::new(&h);
                sp.greedy = s.sampler.is_none();
                s.spec = Some(sp);
                self.seqs.insert(wid, s);
                out.push((wid, tok));
                // stays in the lane; parked below if others decode or prompts wait
            }
        }
        // ---- 2. decode: every sequence with a token to feed
        let mut dec_ids: Vec<u64> = self.seqs.iter().filter(|(_, s)| matches!(s.phase, Phase::Decode { .. })).map(|(&id, _)| id).collect();
        // a sequence that just produced its first token is fed from the next step on
        dec_ids.retain(|id| !out.iter().any(|(o, _)| o == id));
        if dec_ids.is_empty() {
            return Ok(out);
        }
        // context limit: end with the stop token
        dec_ids.retain(|id| {
            let s = &self.seqs[id];
            if s.fed.len() + 2 >= self.b.ctx {
                if let Some(st) = s.stop {
                    out.push((*id, st));
                }
                false
            } else {
                true
            }
        });
        // DSXR_SOLO=1: a lone decoding sequence runs in the lane on the decode graph (faster than a batch of one
        // without drafts; with prompt-lookup drafts the batched path is much faster on agentic output)
        let solo_lane = std::env::var("DSXR_SOLO").map(|v| v == "1").unwrap_or(false);
        if solo_lane && dec_ids.len() == 1 && self.waiting.is_empty() && self.lane != Some(dec_ids[0]) {
            self.park_lane()?;
            let slot = self.seqs[&dec_ids[0]].slot;
            self.dec.unload_slot(&self.b, slot)?;
            self.lane = Some(dec_ids[0]);
        }
        let solo = solo_lane && dec_ids.len() == 1 && self.waiting.is_empty() && self.lane == Some(dec_ids[0]);
        if solo {
            let id = dec_ids[0];
            let mut s = self.seqs.remove(&id).unwrap();
            let Phase::Decode { next } = s.phase else { unreachable!() };
            let g = self.dec.step(next)?;
            s.fed.push(next);
            let tok = self.pick(&mut s, |d| d.logits(), g)?;
            s.phase = Phase::Decode { next: tok };
            self.seqs.insert(id, s);
            out.push((id, tok));
            return Ok(out);
        }
        // batched: the lane's decoding sequence (if any) joins the slots
        if let Some(l) = self.lane {
            if matches!(self.seqs[&l].phase, Phase::Decode { .. }) {
                self.park_lane()?;
            }
        }
        dec_ids.sort();
        // speculative step (prompt lookup / DSpark drafts for greedy sessions); at most BMAX fed tokens per step
        let mut todo = dec_ids.clone();
        while !todo.is_empty() {
            let take: Vec<u64> = todo.drain(..todo.len().min(crate::batch::BMAX / (crate::spec::MAX_DRAFTS + 1))).collect();
            let slots: Vec<usize> = take.iter().map(|id| self.seqs[id].slot).collect();
            let nexts: Vec<u32> = take.iter().map(|id| match self.seqs[id].phase { Phase::Decode { next } => next, _ => unreachable!() }).collect();
            let mut seqs: Vec<Seq> = take.iter().map(|id| self.seqs.remove(id).unwrap()).collect();
            let r = {
                let mut sts: Vec<&mut crate::spec::SpecState> = vec![];
                let mut samplers: Vec<Option<&mut Sampler>> = vec![];
                for sq in seqs.iter_mut() {
                    let Seq { spec, sampler, .. } = sq;
                    sts.push(spec.as_mut().unwrap());
                    samplers.push(sampler.as_mut().filter(|s| s.temperature > 0.0));
                }
                self.dec.spec_step_s(&mut self.b, &slots, &nexts, &mut sts, &mut samplers)
            };
            match r {
                Ok(em) => {
                    for ((id, mut s), e) in take.iter().zip(seqs).zip(em) {
                        let Phase::Decode { next } = s.phase else { unreachable!() };
                        s.fed.push(next);
                        s.fed.extend_from_slice(&e[..e.len() - 1]);
                        s.phase = Phase::Decode { next: *e.last().unwrap() };
                        for &tk in &e {
                            out.push((*id, tk));
                        }
                        self.seqs.insert(*id, s);
                    }
                }
                Err(e) => {
                    for (id, s) in take.iter().zip(seqs) {
                        self.seqs.insert(*id, s);
                    }
                    return Err(e);
                }
            }
        }
        Ok(out)
    }

    fn finish_inner(&mut self, id: u64) {
        let Some(s) = self.seqs.remove(&id) else { return };
        self.waiting.retain(|&w| w != id);
        let decoding = matches!(s.phase, Phase::Decode { .. });
        // keep a decoded sequence's state for prefix reuse (its history = the tokens fed)
        let mut parked = self.slots[s.slot].parked && decoding;
        if self.lane == Some(id) {
            self.lane = None;
            parked = decoding && self.dec.load_slot(&mut self.b, s.slot).is_ok();
        }
        let sl = &mut self.slots[s.slot];
        sl.owner = None;
        sl.parked = parked;
        sl.tokens = if parked { s.fed } else { vec![] };
    }

}

#[cfg(test)]
mod cadence_tests {
    use super::PrefillCadence;

    #[test]
    fn zero_preserves_prefill_every_step() {
        let mut c = PrefillCadence::default();
        for _ in 0..100 { assert!(c.due(true, true, 0)); }
    }

    #[test]
    fn four_decode_rounds_leave_prefill_bounded() {
        let mut c = PrefillCadence::default();
        let due: Vec<_> = (0..100).filter(|_| c.due(true, true, 4)).collect();
        assert_eq!(due, (0..100).step_by(5).collect::<Vec<_>>());
    }

    #[test]
    fn prefill_runs_immediately_without_decoders() {
        let mut c = PrefillCadence::default();
        assert!(c.due(true, true, 64));
        for _ in 0..100 { assert!(c.due(true, false, 64)); }
        assert!(c.due(true, true, 64));
    }

    #[test]
    fn removing_queue_or_disabling_rounds_clears_debt() {
        let mut c = PrefillCadence::default();
        assert!(c.due(true, true, 4));
        assert!(!c.due(false, true, 4));
        assert!(c.due(true, true, 4));
        assert!(c.due(true, true, 0));
        assert!(c.due(true, true, 4));
    }
}
