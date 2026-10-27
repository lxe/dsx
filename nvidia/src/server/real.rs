//! The GPU engine behind the server: GpuModel with prefix reuse.
//!
//! The KV state of the last conversation (prompt + generated ids) stays on the GPU. A request reuses the longest
//! common token prefix: if the new prompt extends everything cached, it just continues; if it diverges inside the
//! generated part (clients often re-send the assistant turn differently), it rewinds to the snapshot taken at the
//! end of the previous prompt; otherwise it starts over.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::Result;

use super::engine::{Engine, Finish, GenParams, GenStats};
use super::sample::Sampler;
use crate::gpumodel::{GpuModel, Placement, Snapshot, VisionWeights};
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
}

struct Real {
    gm: GpuModel<'static>,
    cached: Vec<u32>,
    snap: Option<Snapshot>,
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
    Ok(Box::new(Real { gm, cached: vec![], snap: None, stats: GenStats::default(), vision }))
}

impl Engine for Real {
    fn generate(&mut self, prompt: &[u32], p: &GenParams, emit: &mut dyn FnMut(u32) -> bool) -> Result<Finish> {
        let t0 = Instant::now();
        anyhow::ensure!(!prompt.is_empty(), "empty prompt");
        // the prefix-cache key: every image position carries the same token id, so key those by image content
        let mut key = prompt.to_vec();
        for im in p.images.iter() {
            for (i, k) in key[im.start..im.start + im.len()].iter_mut().enumerate() {
                *k = 0xf000_0000 | ((im.hash as u32 ^ (im.hash >> 32) as u32 ^ i as u32) & 0x0fff_ffff);
            }
        }
        // at least one prompt token must be fed to get the next-token logits
        let lcp = key.iter().zip(&self.cached).take_while(|(a, b)| a == b).count().min(prompt.len() - 1);
        let start = if lcp == self.cached.len() && self.gm.pos == self.cached.len() {
            lcp
        } else if let Some(s) = self.snap.as_ref().filter(|s| s.pos <= lcp) {
            self.gm.restore(s)?;
            s.pos
        } else {
            self.gm.reset();
            0
        };
        // images not already in the KV cache go through the vision tower; their span rows replace the embeddings
        self.gm.images.clear();
        for im in p.images.iter().filter(|im| im.start + im.len() > start) {
            let vw = self.vision.as_ref().ok_or_else(|| anyhow::anyhow!("image input needs the vision weights (--vision)"))?;
            let tv = Instant::now();
            let rows = self.gm.encode_image(vw, im)?;
            tracing::info!(patches = im.n_vit_h * im.n_vit_w, tokens = im.len(), ms = tv.elapsed().as_millis() as u64, "image encoded");
            self.gm.images.push((im.start, rows));
        }
        let pb = self.gm.pb.take().expect("prefill buffers");
        let r = self.gm.prefill(&pb, &prompt[start..]);
        self.gm.pb = Some(pb);
        self.gm.images.clear();
        let mut logits = r?;
        self.snap = Some(self.gm.snapshot()?);
        self.cached = key;
        let prefill = t0.elapsed();
        // DSX_PROFILE_SERVER_V1: per-request counters, no model math changes.
        let profile = std::env::var_os("DSX_TIMING_LOG").is_some();
        let counters0 = profile.then(|| (
            self.gm.cpu_busy_s(), self.gm.cpu_picks(), self.gm.cpu_split_s(),
        ));
        // Snapshot rather than clear: preserve cumulative event profiling for other consumers.
        let events0 = if profile { self.gm.evprof.clone() } else { None };
        let mut profile_steps = 0usize;
        let td = Instant::now();
        let mut sampler = Sampler::from_params(p);
        let mut fin = Finish::Length;
        let mut hes = p.hesitation.clone();
        for _ in 0..p.max_new {
            if let Some(h) = &hes {
                for &i in h.ids.iter() {
                    if let Some(l) = logits.get_mut(i as usize) {
                        *l -= h.lambda;
                    }
                }
            }
            let tok = sampler.sample(&logits);
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
            logits = self.gm.step(tok)?;
            if profile { profile_steps += 1; }
            self.cached.push(tok);
        }
        self.stats = GenStats { prefill: Some(prefill), decode: Some(td.elapsed()), cached_prompt_tokens: start };
        if let Some((busy0, picks0, split0)) = counters0 {
            let picks1 = self.gm.cpu_picks();
            let split1 = self.gm.cpu_split_s();
            let n = profile_steps.max(1) as f64;
            eprintln!("DSX_PROFILE_SERVER_V1 request prompt={} cached={} prefill_s={:.6} decode_s={:.6} steps={}",
                prompt.len(), start, prefill.as_secs_f64(), self.stats.decode.unwrap_or_default().as_secs_f64(), profile_steps);
            eprintln!("DSX_PROFILE_CPU busy_ms_per_step={:.6} picks_percent={:.6} idle_ms_per_step={:.6} prep_ms_per_step={:.6} forward_ms_per_step={:.6}",
                (self.gm.cpu_busy_s()-busy0)*1000.0/n,
                picks1.1.saturating_sub(picks0.1) as f64*100.0/picks1.0.saturating_sub(picks0.0).max(1) as f64,
                (split1[0]-split0[0])*1000.0/n, (split1[1]-split0[1])*1000.0/n,
                (split1[2]-split0[2])*1000.0/n);
            if let Some(events) = &self.gm.evprof {
                let before = |key: &str| events0.as_ref().and_then(|m| m.get(key)).copied().unwrap_or(0.0);
                let steps = events.get("steps").copied().unwrap_or(0.0) - before("steps");
                eprintln!("DSX_PROFILE_EVENT_STEPS {}", steps);
                if steps > 0.0 {
                    for (stage, ms) in events.iter().filter(|(k, _)| **k != "steps") {
                        eprintln!("DSX_PROFILE_EVENT {:?} ms_per_step={:.6}", stage, (ms - before(stage))/steps);
                    }
                }
            }
        }
        let _ = Duration::ZERO;
        Ok(fin)
    }

    fn last_stats(&self) -> GenStats {
        GenStats { prefill: self.stats.prefill, decode: self.stats.decode, cached_prompt_tokens: self.stats.cached_prompt_tokens }
    }
}
