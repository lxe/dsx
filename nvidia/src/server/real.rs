//! The GPU engine behind the server: GpuModel with a bounded prefix cache.
//!
//! The KV state of the current conversation (input prompt + generated ids) stays on the GPU. `cached` records that
//! token lineage, and every successful input prefill leaves a prompt-boundary checkpoint (snapshot + boundary logits)
//! in a bounded history. A request computes the true longest common prefix against `cached`, continues from the live
//! state when the prompt extends the whole cache, otherwise restores the newest checkpoint at or before that prefix
//! and prefills only the tail. Checkpoints past the prefix belong to a divergent branch and are dropped; only when no
//! checkpoint exists before the prefix does the engine reset and prefill from token 0. An exact retry of a cached
//! prompt boundary replays no prompt tokens and reuses the checkpoint's logits.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::Result;

use super::engine::{Engine, Finish, GenParams, GenStats};
use super::prefix::{choose_prefix_reuse, longest_common_prefix, CacheAction, CheckpointPosition};
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

/// Prompt-boundary snapshots kept for rewinding. A snapshot copies every layer's 128-slot window ring and ratio-2
/// compressor state: about 5.6 MB on the shipped 40-layer model, so eight costs ~45 MB of GPU0 memory. Coding-agent
/// sessions only need a few recent boundaries to recover from tool calls and retries, and the history stays bounded.
const PREFIX_CHECKPOINTS: usize = 8;

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
    cached: Vec<u32>,
    checkpoints: Vec<PrefixCheckpoint>,
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
    Ok(Box::new(Real { gm, cached: vec![], checkpoints: Vec::new(), stats: GenStats::default(), vision }))
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
            // exact retry / exact current: no prompt token needs to be replayed for the next-token logits
            match saved_logits {
                Some(l) => Ok(l),
                // A checkpoint at the cached end holds prompt-boundary logits. Without one the prompt boundary came
                // from generated tokens, so the decode buffer holds the logits after that tail.
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
        let mut logits = result?;
        self.save_prompt_checkpoint(prompt.len(), &logits)?;
        self.cached = key;
        let prefill = t0.elapsed();
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
            self.cached.push(tok);
        }
        self.stats = GenStats { prefill: Some(prefill), decode: Some(td.elapsed()), cached_prompt_tokens: decision.start };
        let _ = Duration::ZERO;
        Ok(fin)
    }

    fn last_stats(&self) -> GenStats {
        GenStats { prefill: self.stats.prefill, decode: self.stats.decode, cached_prompt_tokens: self.stats.cached_prompt_tokens }
    }
}
