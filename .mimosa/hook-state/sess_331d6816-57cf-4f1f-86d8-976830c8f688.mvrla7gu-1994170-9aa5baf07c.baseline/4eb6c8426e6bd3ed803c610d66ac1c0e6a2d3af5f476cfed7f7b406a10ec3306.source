//! The engine abstraction and the single OS thread that runs it (GPU contexts are thread-bound).

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc as std_mpsc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use tokio::sync::mpsc;

#[derive(Debug, Clone)]
pub struct GenParams {
    pub max_new: usize,
    /// 0 = greedy
    pub temperature: f32,
    pub top_p: f32,
    pub seed: Option<u64>,
    /// EOS and friends. The worker enforces these and `max_new` too, so an engine may ignore them.
    pub stop_token_ids: Vec<u32>,
    /// logit penalty on hesitation tokens while the model is reasoning (None = off)
    pub hesitation: Option<Hesitation>,
    /// images in the prompt, in order (their spans already expanded in the prompt ids)
    pub images: std::sync::Arc<Vec<crate::vision::ImageInput>>,
}

/// Subtract `lambda` from the logits of `ids` until `close` (`</think>`) is generated.
#[derive(Debug, Clone)]
pub struct Hesitation {
    pub lambda: f32,
    pub ids: std::sync::Arc<Vec<u32>>,
    pub close: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Finish {
    Stop,
    Length,
    Cancelled,
}

/// What an engine knows about its last `generate` call. Unset timings fall back to the worker's wall clock
/// (prefill = until the first token, decode = the rest).
#[derive(Debug, Clone, Copy, Default)]
pub struct GenStats {
    pub prefill: Option<Duration>,
    pub decode: Option<Duration>,
    /// prompt tokens served from a KV/prefix cache
    pub cached_prompt_tokens: usize,
}

pub trait Engine: Send {
    /// Generate after `prompt`. Call `emit(token)` per generated token; stop early when it returns false.
    fn generate(&mut self, prompt: &[u32], p: &GenParams, emit: &mut dyn FnMut(u32) -> bool) -> Result<Finish>;
    /// Timings/cache stats of the last `generate` call.
    fn last_stats(&self) -> GenStats {
        GenStats::default()
    }
}

/// An engine that runs many sequences at once (continuous batching). The worker thread admits queued requests while
/// `has_room`, then calls `step` in a loop; each step advances every running sequence (prefill chunks and/or one
/// decoded token) and returns the tokens it produced. The worker enforces stop tokens, `max_new` and disconnects,
/// and calls `finish` when a sequence ends.
pub trait BatchEngine: Send {
    /// Can another sequence start now?
    fn has_room(&self) -> bool;
    /// Start sequence `id` after `prompt` (its prefill may run over the following steps).
    fn start(&mut self, id: u64, prompt: &[u32], p: &GenParams) -> Result<()>;
    /// Advance every running sequence by one step: (id, token) for each token produced.
    fn step(&mut self) -> Result<Vec<(u64, u32)>>;
    /// Sequence `id` is done (stop, length, cancelled or failed): free its slot.
    fn finish(&mut self, id: u64);
    /// Prompt tokens of sequence `id` served from a cache (for the log).
    fn cached_prompt_tokens(&self, _id: u64) -> usize {
        0
    }
}

/// Engine thread -> request handler.
#[derive(Debug)]
pub enum Event {
    Token(u32),
    Done { finish: Finish },
    Error(String),
}

struct Job {
    id: String,
    prompt: Vec<u32>,
    params: GenParams,
    tx: mpsc::UnboundedSender<Event>,
    queued: Instant,
}

/// Cloneable handle to the engine thread: a FIFO of jobs, one at a time.
#[derive(Clone)]
pub struct EngineHandle {
    tx: std_mpsc::Sender<Job>,
    pending: Arc<AtomicUsize>,
}

impl EngineHandle {
    /// Construct the engine on a dedicated thread (`make` runs there) and start serving jobs.
    pub fn spawn(make: impl FnOnce() -> Result<Box<dyn Engine>> + Send + 'static) -> Result<Self> {
        let (tx, rx) = std_mpsc::channel::<Job>();
        let (ready_tx, ready_rx) = std_mpsc::channel::<Result<()>>();
        let pending = Arc::new(AtomicUsize::new(0));
        let pend = pending.clone();
        std::thread::Builder::new()
            .name("dsx-engine".into())
            .spawn(move || {
                let mut engine = match make() {
                    Ok(e) => {
                        let _ = ready_tx.send(Ok(()));
                        e
                    }
                    Err(e) => {
                        let _ = ready_tx.send(Err(e));
                        return;
                    }
                };
                while let Ok(job) = rx.recv() {
                    run_job(engine.as_mut(), job);
                    pend.fetch_sub(1, Ordering::SeqCst);
                }
            })?;
        ready_rx.recv().context("engine thread died during construction")??;
        Ok(Self { tx, pending })
    }

    /// Construct a batched engine on a dedicated thread and serve jobs concurrently (continuous batching).
    pub fn spawn_batch(make: impl FnOnce() -> Result<Box<dyn BatchEngine>> + Send + 'static) -> Result<Self> {
        let (tx, rx) = std_mpsc::channel::<Job>();
        let (ready_tx, ready_rx) = std_mpsc::channel::<Result<()>>();
        let pending = Arc::new(AtomicUsize::new(0));
        let pend = pending.clone();
        std::thread::Builder::new()
            .name("dsx-engine".into())
            .spawn(move || {
                let engine = match make() {
                    Ok(e) => {
                        let _ = ready_tx.send(Ok(()));
                        e
                    }
                    Err(e) => {
                        let _ = ready_tx.send(Err(e));
                        return;
                    }
                };
                run_batch(engine, rx, pend);
            })?;
        ready_rx.recv().context("engine thread died during construction")??;
        Ok(Self { tx, pending })
    }

    /// Queue a generation; tokens and the final event arrive on the returned channel.
    /// Dropping the receiver cancels the job (queued or running).
    pub fn submit(&self, id: String, prompt: Vec<u32>, params: GenParams) -> Result<mpsc::UnboundedReceiver<Event>> {
        let (tx, rx) = mpsc::unbounded_channel();
        let ahead = self.pending.fetch_add(1, Ordering::SeqCst);
        tracing::info!(%id, prompt_tokens = prompt.len(), max_new = params.max_new, queue_position = ahead, "queued");
        let job = Job { id, prompt, params, tx, queued: Instant::now() };
        if self.tx.send(job).is_err() {
            self.pending.fetch_sub(1, Ordering::SeqCst);
            anyhow::bail!("engine thread is gone");
        }
        Ok(rx)
    }

    /// Jobs queued or running.
    pub fn pending(&self) -> usize {
        self.pending.load(Ordering::SeqCst)
    }
}

fn run_job(engine: &mut dyn Engine, job: Job) {
    let Job { id, prompt, params, tx, queued } = job;
    if tx.is_closed() {
        tracing::info!(%id, "cancelled while queued");
        return;
    }
    let wait = queued.elapsed();
    let t0 = Instant::now();
    let mut first: Option<Instant> = None;
    let mut generated = 0usize;
    let (mut hit_stop, mut hit_len, mut gone) = (false, false, false);
    let r = {
        let mut emit = |tok: u32| -> bool {
            if hit_stop || hit_len || gone {
                return false;
            }
            first.get_or_insert_with(Instant::now);
            if params.stop_token_ids.contains(&tok) {
                hit_stop = true;
                return false;
            }
            generated += 1;
            if tx.send(Event::Token(tok)).is_err() {
                gone = true;
                return false;
            }
            if generated >= params.max_new {
                hit_len = true;
                return false;
            }
            true
        };
        if params.max_new == 0 { Ok(Finish::Length) } else { engine.generate(&prompt, &params, &mut emit) }
    };
    let total = t0.elapsed();
    let st = engine.last_stats();
    let ttft = first.map(|f| f - t0).unwrap_or(total);
    let prefill = st.prefill.unwrap_or(ttft);
    let decode = st.decode.unwrap_or(total.saturating_sub(ttft));
    let finish = match r {
        Ok(f) => {
            // the worker's own stop/length/disconnect checks win over what the engine says
            let f = if hit_stop { Finish::Stop } else if hit_len { Finish::Length } else if gone { Finish::Cancelled } else { f };
            let _ = tx.send(Event::Done { finish: f });
            format!("{f:?}")
        }
        Err(e) => {
            tracing::error!(%id, "engine error: {e:#}");
            let _ = tx.send(Event::Error(format!("{e:#}")));
            "Error".into()
        }
    };
    let decode_tps = if generated > 1 { (generated - 1) as f64 / decode.as_secs_f64().max(1e-9) } else { 0.0 };
    tracing::info!(
        %id,
        prompt_tokens = prompt.len(),
        cached_prompt_tokens = st.cached_prompt_tokens,
        generated,
        queue_ms = wait.as_millis() as u64,
        prefill_ms = prefill.as_millis() as u64,
        decode_ms = decode.as_millis() as u64,
        prefill_tps = format!("{:.1}", (prompt.len() - st.cached_prompt_tokens.min(prompt.len())) as f64 / prefill.as_secs_f64().max(1e-9)),
        decode_tps = format!("{decode_tps:.2}"),
        finish,
        "done"
    );
}

struct Running {
    job_id: String,
    prompt_tokens: usize,
    params: GenParams,
    tx: mpsc::UnboundedSender<Event>,
    queued: Instant,
    started: Instant,
    first: Option<Instant>,
    generated: usize,
}

/// The batched engine thread: admit, step, route tokens, end sequences.
fn run_batch(mut engine: Box<dyn BatchEngine>, rx: std_mpsc::Receiver<Job>, pend: Arc<AtomicUsize>) {
    let mut queue: std::collections::VecDeque<Job> = Default::default();
    let mut running: std::collections::HashMap<u64, Running> = Default::default();
    let mut next_id = 1u64;
    let mut done = |engine: &mut dyn BatchEngine, running: &mut std::collections::HashMap<u64, Running>, id: u64, f: Option<Finish>, err: Option<String>| {
        engine.finish(id);
        pend.fetch_sub(1, Ordering::SeqCst);
        if let Some(r) = running.remove(&id) {
            let total = r.started.elapsed();
            let ttft = r.first.map(|t| t - r.started).unwrap_or(total);
            let fin = match (&f, &err) {
                (_, Some(e)) => {
                    let _ = r.tx.send(Event::Error(e.clone()));
                    "Error".to_string()
                }
                (Some(f), None) => {
                    let _ = r.tx.send(Event::Done { finish: *f });
                    format!("{f:?}")
                }
                _ => "?".into(),
            };
            let decode = total.saturating_sub(ttft);
            let tps = if r.generated > 1 { (r.generated - 1) as f64 / decode.as_secs_f64().max(1e-9) } else { 0.0 };
            tracing::info!(id = %r.job_id, prompt_tokens = r.prompt_tokens, cached_prompt_tokens = engine.cached_prompt_tokens(id), generated = r.generated,
                queue_ms = (r.started - r.queued).as_millis() as u64, prefill_ms = ttft.as_millis() as u64, decode_ms = decode.as_millis() as u64,
                decode_tps = format!("{tps:.2}"), finish = fin, running = running.len(), "done");
        }
    };
    loop {
        // intake: block when there is nothing to do
        if running.is_empty() && queue.is_empty() {
            match rx.recv() {
                Ok(j) => queue.push_back(j),
                Err(_) => return,
            }
        }
        while let Ok(j) = rx.try_recv() {
            queue.push_back(j);
        }
        // admit
        while engine.has_room() {
            let Some(j) = queue.pop_front() else { break };
            if j.tx.is_closed() {
                tracing::info!(id = %j.id, "cancelled while queued");
                pend.fetch_sub(1, Ordering::SeqCst);
                continue;
            }
            if j.params.max_new == 0 {
                let _ = j.tx.send(Event::Done { finish: Finish::Length });
                pend.fetch_sub(1, Ordering::SeqCst);
                continue;
            }
            let id = next_id;
            next_id += 1;
            let r = Running { job_id: j.id.clone(), prompt_tokens: j.prompt.len(), params: j.params.clone(), tx: j.tx, queued: j.queued, started: Instant::now(), first: None, generated: 0 };
            running.insert(id, r);
            if let Err(e) = engine.start(id, &j.prompt, &j.params) {
                tracing::error!(id = %j.id, "start failed: {e:#}");
                done(engine.as_mut(), &mut running, id, None, Some(format!("{e:#}")));
            }
        }
        if running.is_empty() {
            continue;
        }
        // disconnects
        let gone: Vec<u64> = running.iter().filter(|(_, r)| r.tx.is_closed()).map(|(&id, _)| id).collect();
        for id in gone {
            done(engine.as_mut(), &mut running, id, Some(Finish::Cancelled), None);
        }
        if running.is_empty() {
            continue;
        }
        match engine.step() {
            Ok(out) => {
                for (id, tok) in out {
                    let Some(r) = running.get_mut(&id) else { continue };
                    r.first.get_or_insert_with(Instant::now);
                    if r.params.stop_token_ids.contains(&tok) {
                        done(engine.as_mut(), &mut running, id, Some(Finish::Stop), None);
                        continue;
                    }
                    r.generated += 1;
                    if r.tx.send(Event::Token(tok)).is_err() {
                        done(engine.as_mut(), &mut running, id, Some(Finish::Cancelled), None);
                        continue;
                    }
                    if r.generated >= r.params.max_new {
                        done(engine.as_mut(), &mut running, id, Some(Finish::Length), None);
                    }
                }
            }
            Err(e) => {
                tracing::error!("engine step failed: {e:#}");
                let ids: Vec<u64> = running.keys().copied().collect();
                for id in ids {
                    done(engine.as_mut(), &mut running, id, None, Some(format!("{e:#}")));
                }
            }
        }
    }
}

static REQ: AtomicU64 = AtomicU64::new(0);

/// Unique-enough request id: prefix + unix seconds + counter.
pub fn request_id(prefix: &str) -> String {
    let n = REQ.fetch_add(1, Ordering::Relaxed);
    let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    format!("{prefix}{t:x}{n:04x}")
}
