//! Speculative decoding for batched sessions: per session, drafts come from its own context (prompt lookup: the
//! tokens that followed the last earlier occurrence of its newest 3 tokens) or, while they pay off, from DSpark;
//! every session's next token and drafts are verified in one batched step and the accepted prefix is committed.
//! A session with no drafts feeds one token, as plain decoding does.

use crate::batch::Batch;
use crate::decode::{Decoder, VOCAB};
use anyhow::Result;
use dsx_core::server::sample::Sampler;
use std::collections::HashMap;

/// most drafts verified per session per step (slot rings hold 16 spare rows)
pub const MAX_DRAFTS: usize = 8;

// Optional live experiment controls. The default preserves the measured baseline policy.
pub(crate) static DRAFT_ENABLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
pub(crate) static PROB_PASSES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
pub(crate) static PROB_ACCEPTED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
pub(crate) static PROB_PROPOSED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

pub(crate) static PROB_DRAFT_OVERRIDE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);
pub(crate) static DRAFT_LIMIT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
/// Optional total verification-row budget; zero leaves draft lengths unbounded.
pub(crate) static VERIFY_ROW_BUDGET: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn row_limited_drafts(sessions: usize, budget: usize, limit: usize) -> usize {
    if budget == 0 { limit } else { limit.min((budget / sessions.max(1)).saturating_sub(1)) }
}

pub(crate) fn init_tuning_from_env() -> Result<()> {
    for (name, atom, min, max) in [
        ("DSXA_DRAFT_LIMIT", &DRAFT_LIMIT, 0, MAX_DRAFTS),
        ("DSXA_LOOKUP_LIMIT", &LOOKUP_LIMIT, 0, MAX_DRAFTS),
        ("DSXA_LOOKUP_CONTEXT", &LOOKUP_CONTEXT, 3, 12),
        ("DSXA_VERIFY_ROWS", &VERIFY_ROW_BUDGET, 0, crate::batch::BMAX),
    ] {
        if let Ok(value) = std::env::var(name) {
            let value: usize = value.parse()?;
            anyhow::ensure!((min..=max).contains(&value), "{name} must be {min}..{max}");
            atom.store(value, std::sync::atomic::Ordering::Relaxed);
        }
    }
    Ok(())
}


pub(crate) static GPU_SAMPLE_OVERRIDE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

pub(crate) static LOOKUP_LIMIT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(MAX_DRAFTS);
pub(crate) static LOOKUP_CONTEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(3);


pub struct SpecState {
    /// the session's tokens (prompt + generated, including the pending `next`)
    toks: Vec<u32>,
    /// newest earlier position following each 3-gram: (a, b, c) -> index of the token after it
    ngram: HashMap<(u32, u32, u32), usize>,
    /// running mean of DSpark drafts accepted per pass
    pub ema: f64,
    /// running mean of prompt-lookup drafts accepted per pass (sets the lookup draft length)
    pub lema: f64,
    pub passes: usize,
    pub accepted: usize,
    pub ngram_hits: usize,
    /// drafts verified (all sources)
    pub drafted: usize,
    /// sampled (temperature > 0) sessions get no drafts and sample their token from the logits
    pub greedy: bool,
}

impl SpecState {
    pub fn new(history: &[u32]) -> SpecState {
        let mut s = SpecState { toks: vec![], ngram: HashMap::new(), ema: 2.0, lema: 4.0, passes: 0, accepted: 0, ngram_hits: 0, drafted: 0, greedy: true };
        s.push(history);
        s
    }

    pub fn push(&mut self, new: &[u32]) {
        for &t in new {
            let n = self.toks.len();
            if n >= 3 {
                // the 3-gram ending at n-1 is followed by the token at n
                self.ngram.insert((self.toks[n - 3], self.toks[n - 2], self.toks[n - 1]), n);
            }
            self.toks.push(t);
        }
    }

    /// the newest `n` tokens
    pub fn tail(&self, n: usize) -> Vec<u32> {
        self.toks[self.toks.len().saturating_sub(n)..].to_vec()
    }

    /// up to `max` tokens that followed the newest earlier occurrence of the last 3 tokens
    fn lookup(&self, max: usize) -> Vec<u32> {
        self.lookup_context(max, LOOKUP_CONTEXT.load(std::sync::atomic::Ordering::Relaxed))
    }

    fn lookup_context(&self, max: usize, width: usize) -> Vec<u32> {
        let n = self.toks.len();
        if n < 4 || n < width {
            return vec![];
        }
        match self.ngram.get(&(self.toks[n - 3], self.toks[n - 2], self.toks[n - 1])) {
            Some(&i) if i < n && i >= width && self.toks[i - width..i] == self.toks[n - width..n] => self.toks[i..n.min(i + max)].to_vec(),
            _ => vec![],
        }
    }
}

/// DSpark is used for a session while its acceptance average is at least this (DSXS_MIN, default 2.0: at lower
/// acceptance a draft pass plus the longer verify costs more than it returns once a few sessions share a step)
fn dspark_min() -> f64 {
    std::env::var("DSXS_MIN").ok().and_then(|v| v.parse().ok()).unwrap_or(2.0)
}

impl Decoder {
    /// One speculative step for slots[i] whose next (emitted, not yet fed) token is nexts[i]. Returns, per slot, the
    /// tokens it emits (accepted drafts, then its new next token); st[i].push gets the same tokens.
    pub fn spec_step(&mut self, b: &mut Batch, slots: &[usize], nexts: &[u32], st: &mut [&mut SpecState]) -> Result<Vec<Vec<u32>>> {
        let mut none: Vec<Option<&mut Sampler>> = (0..slots.len()).map(|_| None).collect();
        self.spec_step_s(b, slots, nexts, st, &mut none)
    }

    /// spec_step where session i samples with samplers[i] (None: greedy). A sampled session accepts each draft with
    /// its probability or takes the resampled token (Sampler::accept_or_resample: exact speculative sampling); the
    /// verified logits come down in one copy and the sampled sessions are decided on parallel threads.
    pub fn spec_step_s(&mut self, b: &mut Batch, slots: &[usize], nexts: &[u32], st: &mut [&mut SpecState], samplers: &mut [Option<&mut Sampler>]) -> Result<Vec<Vec<u32>>> {
        let gpu_sample = match GPU_SAMPLE_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed) {
            1 => false, 2 => true, _ => std::env::var("DSXA_GPU_SAMPLE").as_deref() == Ok("1"),
        } && samplers.iter().any(Option::is_some)
          && samplers.iter().all(|s| s.as_ref().is_none_or(|s| s.untruncated()));
        let prob_draft = gpu_sample && match PROB_DRAFT_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed) {
            1 => false, 2 => true, _ => std::env::var("DSXA_PROB_DRAFT").as_deref() == Ok("1"),
        };
        let n = slots.len();
        let draft_cap = row_limited_drafts(n, VERIFY_ROW_BUDGET.load(std::sync::atomic::Ordering::Relaxed), MAX_DRAFTS);
        let rooms: Vec<usize> = slots.iter().map(|&s| (b.ctx - b.pos[s] - 2).min(draft_cap)).collect();
        // drafts: prompt lookup first, else DSpark while it pays off (a session below the bar still probes it now and then)
        // lookup draft length follows its acceptance (a wrong guess still lengthens the verify)
        let lookup_limit = LOOKUP_LIMIT.load(std::sync::atomic::Ordering::Relaxed).min(MAX_DRAFTS);
        let mut drafts: Vec<Vec<u32>> = (0..n).map(|i| st[i].lookup(rooms[i].min(lookup_limit).min((st[i].lema * 1.5 + 1.0).round().max(1.0) as usize))).collect();
        let mut from_lookup: Vec<bool> = drafts.iter().map(|d| !d.is_empty()).collect();
        let want: Vec<usize> = if self.dspark.is_some() && DRAFT_ENABLED.load(std::sync::atomic::Ordering::Relaxed) {
            (0..n).filter(|&i| drafts[i].is_empty() && rooms[i] > 0 && (st[i].ema >= dspark_min() || st[i].passes % 32 == 31)).collect()
        } else {
            vec![]
        };
        let mut proposal_base: Vec<Option<usize>> = vec![None; n];
        if !want.is_empty() {
            let ws: Vec<usize> = want.iter().map(|&i| slots[i]).collect();
            let wn: Vec<u32> = want.iter().map(|&i| nexts[i]).collect();
            let d = if prob_draft {
                let mut sms: Vec<Option<&mut Sampler>> = samplers.iter_mut().enumerate().filter(|(i, _)| want.contains(i)).map(|(_, sm)| sm.as_deref_mut()).collect();
                self.batch_draft_sampled(b, &ws, &wn, Some(&mut sms))?
            } else { self.batch_draft(b, &ws, &wn)? };
            for (k, &i) in want.iter().enumerate() {
                // draft length follows the acceptance average
                let limit = DRAFT_LIMIT.load(std::sync::atomic::Ordering::Relaxed).min(MAX_DRAFTS);
                let len = if limit > 0 { limit.min(d[k].len()) } else { ((st[i].ema + 1.5).round() as usize).clamp(1, d[k].len()) }.min(rooms[i]);
                if prob_draft && samplers[i].is_some() { proposal_base[i] = Some(k * self.dspark.as_ref().unwrap().block); }
                drafts[i] = d[k][..len].to_vec();
                from_lookup[i] = false;
            }
        }
        // verify every session's next token and drafts at once
        let feeds: Vec<Vec<u32>> = (0..n).map(|i| {
            let mut f = vec![nexts[i]];
            f.extend_from_slice(&drafts[i]);
            f
        }).collect();
        let pre: Vec<usize> = slots.iter().map(|&s| b.pos[s]).collect();
        let outs = self.batch_verify(b, slots, &feeds)?;
        // sampled sessions: (drafts accepted, next token)
        let starts: Vec<usize> = feeds.iter().scan(0, |r, f| { let s0 = *r; *r += f.len(); Some(s0) }).collect();
        let mut decided: Vec<Option<(usize, u32)>> = vec![None; n];
        if gpu_sample && proposal_base.iter().any(Option::is_some) {
            #[derive(Clone, Copy, Default)]
            #[repr(C)]
            struct Draw { temperature: f32, active: i32, qrow: i32, draft: i32, accept_uniform: f64, recover_uniform: f64 }
            let rows_n: usize = feeds.iter().map(Vec::len).sum();
            let mut draws = vec![Draw { qrow: -1, ..Draw::default() }; rows_n];
            for (i, sm) in samplers.iter_mut().enumerate() {
                if let Some(sm) = sm.as_deref_mut() {
                    for j in 0..feeds[i].len() {
                        draws[starts[i] + j] = Draw { temperature: sm.temperature, active: 1,
                            qrow: if j + 1 < feeds[i].len() { proposal_base[i].map(|base| (base + j) as i32).unwrap_or(-1) } else { -1 },
                            draft: feeds[i].get(j + 1).copied().unwrap_or(0) as i32,
                            accept_uniform: sm.uniform_draw(), recover_uniform: sm.uniform_draw() };
                    }
                }
            }
            let q = b.draft_logits.as_ref().ok_or_else(|| anyhow::anyhow!("missing proposal logits"))?;
            self.g.htod(b.sample_draws.ptr, &draws)?;
            self.g.launch("reject_logits_b", (rows_n as u32, 1, 1), 1024, 0,
                &[crate::hip::Arg::P(b.logits.ptr), crate::hip::Arg::P(q.ptr), crate::hip::Arg::I(VOCAB as i32),
                  crate::hip::Arg::P(b.sample_draws.ptr), crate::hip::Arg::P(b.next.ptr), crate::hip::Arg::P(b.sample_next.ptr)])?;
            let sampled: Vec<i32> = self.g.dtoh(b.sample_next.ptr, rows_n)?;
            for i in 0..n {
                if samplers[i].is_some() {
                    let mut acc = 0;
                    while acc + 1 < feeds[i].len() && sampled[starts[i] + acc] as u32 == feeds[i][acc + 1] { acc += 1; }
                    decided[i] = Some((acc, sampled[starts[i] + acc] as u32));
                }
            }
        } else         if gpu_sample {
            #[derive(Clone, Copy, Default)]
            #[repr(C)]
            struct Draw { temperature: f32, active: i32, uniform: f64 }
            let rows_n: usize = feeds.iter().map(Vec::len).sum();
            let mut draws = vec![Draw::default(); rows_n];
            for (i, sm) in samplers.iter_mut().enumerate() {
                if let Some(sm) = sm.as_mut() {
                    for j in starts[i]..starts[i] + feeds[i].len() {
                        draws[j] = Draw { temperature: sm.temperature, active: 1, uniform: sm.uniform_draw() };
                    }
                }
            }
            self.g.htod(b.sample_draws.ptr, &draws)?;
            self.g.launch("sample_logits_b", (rows_n as u32, 1, 1), 1024, 0,
                &[crate::hip::Arg::P(b.logits.ptr), crate::hip::Arg::I(VOCAB as i32),
                  crate::hip::Arg::P(b.sample_draws.ptr), crate::hip::Arg::P(b.next.ptr),
                  crate::hip::Arg::P(b.sample_next.ptr)])?;
            let sampled: Vec<i32> = self.g.dtoh(b.sample_next.ptr, rows_n)?;
            for i in 0..n {
                if samplers[i].is_some() {
                    let mut acc = 0;
                    while acc + 1 < feeds[i].len() && sampled[starts[i] + acc] as u32 == feeds[i][acc + 1] {
                        acc += 1;
                    }
                    decided[i] = Some((acc, sampled[starts[i] + acc] as u32));
                }
            }
        } else if let Some(last) = (0..n).rev().find(|&i| samplers[i].is_some()) {
            let rows_n = starts[last] + feeds[last].len();
            let all: Vec<f32> = self.g.dtoh(b.logits.at(0), rows_n * VOCAB)?;
            std::thread::scope(|sc| {
                for ((i, sm), dc) in samplers.iter_mut().enumerate().zip(decided.iter_mut()) {
                    let Some(sm) = sm.as_mut() else { continue };
                    let (f, l) = (&feeds[i], &all[starts[i] * VOCAB..(starts[i] + feeds[i].len()) * VOCAB]);
                    sc.spawn(move || {
                        let mut acc = 0;
                        while acc + 1 < f.len() {
                            let (ok, t) = sm.accept_or_resample(&l[acc * VOCAB..(acc + 1) * VOCAB], f[acc + 1]);
                            if !ok {
                                *dc = Some((acc, t));
                                return;
                            }
                            acc += 1;
                        }
                        *dc = Some((acc, sm.sample(&l[acc * VOCAB..(acc + 1) * VOCAB])));
                    });
                }
            });
        }
        let (mut rows, mut opos, mut oslot) = (vec![], vec![], vec![]);
        let mut emitted = vec![];
        let mut row = 0;
        for i in 0..n {
            let f = &feeds[i];
            let mut acc = 0;
            let mut sampled_next = None;
            if let Some((a, t)) = decided[i] {
                acc = a;
                sampled_next = Some(t);
            } else {
                while acc + 1 < f.len() && outs[i][acc] == f[acc + 1] {
                    acc += 1;
                }
            }
            if proposal_base[i].is_some() {
                PROB_PASSES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                PROB_ACCEPTED.fetch_add(acc, std::sync::atomic::Ordering::Relaxed);
                PROB_PROPOSED.fetch_add(f.len() - 1, std::sync::atomic::Ordering::Relaxed);
            }
            self.batch_commit(b, slots[i], row, f, acc + 1)?;
            for j in 0..=acc {
                rows.push((row + j) as i32);
                opos.push((pre[i] + j) as i32);
                oslot.push(slots[i] as i32);
            }
            let mut e = f[1..=acc].to_vec();
            e.push(sampled_next.unwrap_or(outs[i][acc]));
            st[i].push(&e);
            if f.len() > 1 {
                st[i].drafted += f.len() - 1;
                st[i].passes += 1;
                st[i].accepted += acc;
                if from_lookup[i] {
                    st[i].ngram_hits += acc;
                    st[i].lema = 0.6 * st[i].lema + 0.4 * acc as f64;
                } else {
                    st[i].ema = 0.6 * st[i].ema + 0.4 * acc as f64;
                }
            } else {
                st[i].passes += 1;
            }
            emitted.push(e);
            row += f.len();
        }
        if self.dspark.is_some() {
            let t: usize = feeds.iter().map(|f| f.len()).sum();
            self.batch_observe(b, t, &rows, &opos, &oslot)?;
        }
        Ok(emitted)
    }
}

#[cfg(test)]
mod lookup_tests {
    use super::{SpecState, row_limited_drafts};
    #[test]
    fn verification_budget_preserves_four_and_bounds_larger_batches() {
        let caps: Vec<_> = (1..=8).map(|n| row_limited_drafts(n,31,5)).collect();
        assert_eq!(caps, vec![5,5,5,5,5,4,3,2]);
        for n in 1..=31 { assert!(n*(row_limited_drafts(n,31,8)+1) <= 31); }
        assert_eq!(row_limited_drafts(8,0,8),8);
        assert_eq!(row_limited_drafts(8,4,8),0);
    }
    #[test]
    fn longer_context_rejects_an_ambiguous_suffix() {
        let history = [9,1,2,3,4,5,8,1,2,3,4];
        let state = SpecState::new(&history);
        assert_eq!(state.lookup_context(8,3), vec![5,8,1,2,3,4]);
        assert_eq!(state.lookup_context(8,4), vec![5,8,1,2,3,4]);
        assert!(state.lookup_context(8,5).is_empty());
        let exact = SpecState::new(&[9,1,2,3,4,5,9,1,2,3,4]);
        assert_eq!(exact.lookup_context(2,5), vec![5,9]);
    }
}
