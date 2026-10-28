//! Live numbers from the batched runner for the admin dashboard (`dsx-amd serve --admin-port`), ported from Volta's
//! telemetry (same JSON): throughput over the last seconds and minutes, sessions and their speed, draft acceptance,
//! queue. The runner records one sample per step; the admin server reads a snapshot.

use std::collections::VecDeque;
use std::sync::Mutex;
use std::time::Instant;

use serde::Serialize;

/// one scheduler step
#[derive(Clone, Copy)]
struct Sample {
    t: f64,
    out: usize,
    prompt: usize,
    step_ms: f64,
    draft_ms: f64,
    drafted: usize,
    kept: usize,
}

#[derive(Clone, Serialize)]
pub struct SessionInfo {
    pub id: String,
    pub gpu: usize,
    pub slot: usize,
    /// "prefill" or "decode"
    pub state: &'static str,
    pub prompt_tokens: usize,
    pub cached_tokens: usize,
    pub generated: usize,
    pub tok_s: f64,
    /// recent drafts kept per draft tried
    pub acceptance: f64,
    pub age_s: f64,
    pub sampled: bool,
}

#[derive(Default)]
struct State {
    samples: VecDeque<Sample>,
    sessions: Vec<SessionInfo>,
    queued: usize,
    requests: u64,
    tokens_out: u64,
    tokens_prompt: u64,
    tokens_cached: u64,
    finished: Vec<(String, usize, f64, f64)>,
}

pub struct Telemetry {
    t0: Instant,
    pub gpus: usize,
    /// sessions the engine holds at once
    pub capacity: usize,
    pub ctx: usize,
    /// head-parallel: every GPU runs every session
    pub heads: bool,
    s: Mutex<State>,
}

#[derive(Serialize)]
pub struct Point {
    pub t: f64,
    pub out_tok_s: f64,
    pub prompt_tok_s: f64,
}

#[derive(Serialize)]
pub struct Snapshot {
    pub uptime_s: f64,
    pub gpus: usize,
    pub capacity: usize,
    pub ctx: usize,
    pub heads: bool,
    pub active: usize,
    pub prefilling: usize,
    pub queued: usize,
    pub requests: u64,
    pub tokens_out: u64,
    pub tokens_prompt: u64,
    pub tokens_cached: u64,
    /// generated tokens per second over the last 10 s
    pub out_tok_s: f64,
    pub prompt_tok_s: f64,
    pub step_ms: f64,
    pub draft_ms: f64,
    /// drafts kept per draft verified over the last 10 s
    pub acceptance: f64,
    pub per_session_median: f64,
    pub sessions: Vec<SessionInfo>,
    /// 1 s buckets over the last 5 minutes
    pub history: Vec<Point>,
    /// the last finished requests: (id, generated, decode tok/s, seconds ago)
    pub recent: Vec<(String, usize, f64, f64)>,
}

impl Telemetry {
    pub fn new(gpus: usize, capacity: usize, ctx: usize, heads: bool) -> Self {
        Self { t0: Instant::now(), gpus, capacity, ctx, heads, s: Mutex::new(State::default()) }
    }

    fn now(&self) -> f64 {
        self.t0.elapsed().as_secs_f64()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn step(&self, out: usize, prompt: usize, step_ms: f64, draft_ms: f64, drafted: usize, kept: usize, sessions: Vec<SessionInfo>, queued: usize) {
        let t = self.now();
        let mut s = self.s.lock().unwrap();
        s.samples.push_back(Sample { t, out, prompt, step_ms, draft_ms, drafted, kept });
        while s.samples.front().is_some_and(|x| x.t < t - 300.0) {
            s.samples.pop_front();
        }
        s.tokens_out += out as u64;
        s.tokens_prompt += prompt as u64;
        s.sessions = sessions;
        s.queued = queued;
    }

    pub fn admitted(&self, cached: usize) {
        let mut s = self.s.lock().unwrap();
        s.requests += 1;
        s.tokens_cached += cached as u64;
    }

    pub fn finished(&self, id: &str, generated: usize, tok_s: f64) {
        let t = self.now();
        let mut s = self.s.lock().unwrap();
        s.finished.push((id.to_string(), generated, tok_s, t));
        // drop it from the live list now (the list is otherwise only refreshed by the next step)
        s.sessions.retain(|x| x.id != id);
        let n = s.finished.len();
        if n > 50 {
            s.finished.drain(..n - 50);
        }
    }

    pub fn snapshot(&self) -> Snapshot {
        let t = self.now();
        let s = self.s.lock().unwrap();
        let recent: Vec<&Sample> = s.samples.iter().filter(|x| x.t >= t - 10.0).collect();
        let span = recent.first().map(|x| (t - x.t).max(1.0)).unwrap_or(1.0);
        let out_tok_s = recent.iter().map(|x| x.out).sum::<usize>() as f64 / span;
        let prompt_tok_s = recent.iter().map(|x| x.prompt).sum::<usize>() as f64 / span;
        let n = recent.len().max(1) as f64;
        let step_ms = recent.iter().map(|x| x.step_ms).sum::<f64>() / n;
        let draft_ms = recent.iter().map(|x| x.draft_ms).sum::<f64>() / n;
        let (dr, kp) = recent.iter().fold((0, 0), |a, x| (a.0 + x.drafted, a.1 + x.kept));
        let mut speeds: Vec<f64> = s.sessions.iter().filter(|x| x.state == "decode" && x.generated > 1).map(|x| x.tok_s).collect();
        speeds.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut history = vec![];
        let start = (t - 300.0).max(0.0).floor();
        let mut b = start;
        let mut it = s.samples.iter().peekable();
        while b < t.floor() {
            let (mut o, mut p) = (0usize, 0usize);
            while let Some(x) = it.peek() {
                if x.t >= b + 1.0 {
                    break;
                }
                if x.t >= b {
                    o += x.out;
                    p += x.prompt;
                }
                it.next();
            }
            history.push(Point { t: b - t, out_tok_s: o as f64, prompt_tok_s: p as f64 });
            b += 1.0;
        }
        Snapshot {
            uptime_s: t,
            gpus: self.gpus,
            capacity: self.capacity,
            ctx: self.ctx,
            heads: self.heads,
            active: s.sessions.len(),
            prefilling: s.sessions.iter().filter(|x| x.state == "prefill").count(),
            queued: s.queued,
            requests: s.requests,
            tokens_out: s.tokens_out,
            tokens_prompt: s.tokens_prompt,
            tokens_cached: s.tokens_cached,
            out_tok_s,
            prompt_tok_s,
            step_ms,
            draft_ms,
            acceptance: if dr > 0 { kp as f64 / dr as f64 } else { 0.0 },
            per_session_median: speeds.get(speeds.len() / 2).copied().unwrap_or(0.0),
            sessions: s.sessions.clone(),
            history,
            recent: s.finished.iter().rev().take(12).map(|(id, g, v, ft)| (id.clone(), *g, *v, t - ft)).collect(),
        }
    }
}
