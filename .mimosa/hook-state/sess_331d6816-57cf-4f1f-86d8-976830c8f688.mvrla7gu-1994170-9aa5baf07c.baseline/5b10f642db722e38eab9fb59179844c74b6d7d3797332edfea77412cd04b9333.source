//! DSpark, the checkpoint's speculative draft stack (`mtp.*`), on the GPU; the CPU reference is core/src/dspark.rs.
//!
//! Three window-only layers with their own 128-expert MoE (top-3) read the main model's residual (the mean of its 4
//! streams at the entry of layers 37-39, projected by `main_proj`) and draft a block of 5 tokens at once,
//! bidirectionally inside the block; a Markov head (bias from the previous drafted token) finishes them in order.
//! The draft layers are ordinary `Layer`s run through the prompt path (`run_chunk` in draft mode); their window
//! rings hold the main model's projected residual of the last 128 positions (`observe`).

use crate::ck::Ck;
use crate::decode::*;
use crate::hip::{Arg, Buf};
use crate::load::{self, DMat};
use anyhow::Result;
use std::path::Path;

pub struct Dspark {
    pub layers: Vec<Layer>,
    pub(crate) arena: Buf,
    pub(crate) main_proj: DMat,
    pub(crate) main_norm: Buf,
    pub(crate) norm: Buf,
    /// bf16 rows [vocab][rank]
    pub(crate) markov_embed: Buf,
    /// bf16 tiles, vocab x rank
    pub(crate) markov_head: DMat,
    pub rank: usize,
    pub block: usize,
    pub noise: u32,
    pub targets: Vec<usize>,
    pub top_k: usize,
    pub ne: usize,
}

const LAYERS: usize = 3;
const NE_D: usize = 128;

/// DSXA_DSPARK=4 preserves native MXFP4; 3 or 2 use lower-bit codebooks.
pub fn bits_from_env() -> Option<i32> {
    match std::env::var("DSXA_DSPARK").ok()?.as_str() {
        "4" => Some(E_FP4),
        "2" => Some(E_CB2),
        "" | "0" => None,
        _ => Some(E_CB3),
    }
}

/// HBM the draft stack takes (experts in `fmt`, dense FP8 layers, main_proj, Markov head)
pub fn bytes(fmt: i32) -> usize {
    let experts = LAYERS * NE_D * expert_bytes(fmt);
    // wq_a, wkv, wq_b, wo_a, wo_b, shared expert (FP8 bytes; scales and norms in the 33/32)
    let dense_layer = (1280 + 512) * D + 32768 * 1280 + 8192 * 4096 + D * 8192 + 3 * INTER * D;
    let dense = LAYERS * dense_layer * 33 / 32 + D * 3 * D * 33 / 32;
    let markov = 2 * VOCAB * 256 * 2;
    experts + dense + markov + (64 << 20)
}

impl Dspark {
    /// Load from the checkpoint directory (exports do not carry the draft layers), after the main model.
    pub fn load(dec: &Decoder, dir: &Path, fmt: i32) -> Result<Dspark> {
        let g = &dec.g;
        let t0 = std::time::Instant::now();
        // the draft tensors (official shards 44-46), not the whole checkpoint
        let st = Ck::open_prefix(dir, "mtp.")?;
        let cfg: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
        let t = &cfg["text_config"];
        let mut layers = vec![];
        for i in 0..LAYERS {
            let pre = format!("mtp.{i}");
            let p = |s: &str| format!("{pre}.{s}");
            let (fa, ha) = hc_args(&st, &pre, "attn")?;
            let (ff, hf) = hc_args(&st, &pre, "ffn")?;
            layers.push(Layer {
                id: i,
                ratio: 0,
                kv_src: false,
                idx_src: false,
                comp_src: 0,
                idxk_src: 0,
                hc_attn_fn: g.upload(&fa)?,
                hc_attn: ha,
                hc_ffn_fn: g.upload(&ff)?,
                hc_ffn: hf,
                attn_norm: load::up_f32(g, &st, &p("attn_norm.weight"))?,
                ffn_norm: load::up_f32(g, &st, &p("ffn_norm.weight"))?,
                q_norm: load::up_f32(g, &st, &p("attn.q_norm.weight"))?,
                kv_norm: load::up_f32(g, &st, &p("attn.kv_norm.weight"))?,
                sinks: load::up_f32(g, &st, &p("attn.attn_sink"))?,
                wq_a: load::up_fp8(g, &st, &p("attn.wq_a.weight"))?,
                wkv: load::up_fp8(g, &st, &p("attn.wkv.weight"))?,
                wq_b: load::up_fp8(g, &st, &p("attn.wq_b.weight"))?,
                wo_a: load::up_fp8(g, &st, &p("attn.wo_a.weight"))?,
                wo_b: load::up_fp8(g, &st, &p("attn.wo_b.weight"))?,
                comp_wkv: None,
                comp_wgate: None,
                comp_norm: None,
                idx_wq_b: None,
                idx_proj: None,
                idx_wk: None,
                idx_knorm: None,
                gate: load::up_bf16(g, &st, &p("ffn.gate.weight"))?,
                gate_bias: load::up_f32(g, &st, &p("ffn.gate.bias"))?,
                sh_w1: load::up_fp8(g, &st, &p("ffn.shared_experts.w1.weight"))?,
                sh_w3: load::up_fp8(g, &st, &p("ffn.shared_experts.w3.weight"))?,
                sh_w2: load::up_fp8(g, &st, &p("ffn.shared_experts.w2.weight"))?,
                experts: g.alloc_zeroed(NE_D * 3 * std::mem::size_of::<ExpertMat>())?,
                engram: None,
                ring: g.alloc_zeroed(WIN * HD * 2)?,
                comp_cache: None,
                idx_cache: None,
                cstate: None,
            });
        }
        let arena = g.alloc(LAYERS * NE_D * expert_bytes(fmt))?;
        let tabs: Vec<u64> = layers.iter().map(|l| l.experts.ptr).collect();
        let name = |l: usize, e: usize, wn: &str| format!("mtp.{l}.ffn.experts.{e}.{wn}.weight");
        let (_, off) = dec.upload_experts(&st, &arena, LAYERS, NE_D, &name, &|_, _| fmt, &tabs)?;
        anyhow::ensure!(off == LAYERS * NE_D * expert_bytes(fmt), "draft expert arena: wrote {off}");
        let last = LAYERS - 1;
        let ds = Dspark {
            layers,
            arena,
            main_proj: load::up_fp8(g, &st, "mtp.0.main_proj.weight")?,
            main_norm: load::up_f32(g, &st, "mtp.0.main_norm.weight")?,
            norm: load::up_f32(g, &st, &format!("mtp.{last}.norm.weight"))?,
            markov_embed: load::up_raw(g, &st, &format!("mtp.{last}.markov_head.embed.weight"))?,
            markov_head: load::up_bf16(g, &st, &format!("mtp.{last}.markov_head.head.weight"))?,
            rank: t["dspark_markov_rank"].as_u64().unwrap_or(256) as usize,
            block: t["dspark_block_size"].as_u64().unwrap_or(5) as usize,
            noise: t["dspark_noise_token_id"].as_u64().unwrap_or(128799) as u32,
            targets: t["dspark_target_layer_ids"].as_array().map(|a| a.iter().filter_map(|x| x.as_u64()).map(|x| x as usize).collect()).unwrap_or(vec![37, 38, 39]),
            top_k: t["dspark_num_experts_per_tok"].as_u64().unwrap_or(3) as usize,
            ne: NE_D,
        };
        anyhow::ensure!(ds.targets.len() == 3 && ds.top_k <= 6 && ds.block + 1 <= 16, "unexpected DSpark config");
        let (free, total) = g.mem_info()?;
        eprintln!("dspark: {} layers, block {}, experts {}: {:.1} s, {:.1} GB live HBM used, {:.1} GB free", LAYERS, ds.block, match fmt { E_FP4 => "MXFP4", E_CB3 => "3-bit", _ => "2-bit" }, t0.elapsed().as_secs_f32(), (total - free) as f64 / 1e9, free as f64 / 1e9);
        Ok(ds)
    }
}

/// Speculation counters of a generation
#[derive(Default, Debug, Clone)]
pub struct SpecStats {
    pub passes: usize,
    pub accepted: usize,
    pub tokens: usize,
    /// tokens decoded plainly (low acceptance)
    pub plain: usize,
    pub draft_us: f64,
    pub verify_us: f64,
}

impl Decoder {
    pub fn load_dspark(&mut self, dir: &Path) -> Result<()> {
        if let Some(fmt) = bits_from_env() {
            self.dspark = Some(Dspark::load(self, dir, fmt)?);
        }
        Ok(())
    }

    /// Record the main model's residual for positions p0 .. p0+n (main_hidden rows row0.. of the last chunk) in
    /// every draft layer's window ring.
    pub(crate) fn observe(&self, p0: usize, n: usize, row0: usize) -> Result<()> {
        let ds = self.dspark.as_ref().unwrap();
        let g = &self.g;
        let pf = self.pf.as_ref().unwrap();
        let (mh, aq, as_, y, kvraw, lin) = pf.dspark_bufs();
        for c0 in (0..n).step_by(16) {
            let m = 16.min(n - c0);
            let r0 = row0 + c0;
            let p = p0 + c0;
            // main_proj(main_hidden) -> rmsnorm(main_norm) -> FP8 activation
            self.prep(4, mh + (r0 * 3 * D * 4) as u64, 0, 3 * D, 0, 0, 0, 3 * D, m, aq, as_, 0, 0)?;
            let j = crate::verify::VJob { w: ds.main_proj.w.ptr, sc: ds.main_proj.sc.as_ref().unwrap().ptr, fmt: 0, n: D as i32, k: (3 * D) as i32, xq: aq, xs: as_, ldx: (3 * D) as i32, ntok: m as i32, y, ldy: D as i32, mul: 1.0, ..Default::default() };
            self.vlaunch("dspark main_proj", vec![j], self.st.ptr)?;
            self.prep(1, y, 0, D, ds.main_norm.ptr, 0, 0, D, m, aq, as_, 0, 0)?;
            for l in &ds.layers {
                let j = crate::verify::VJob { w: l.wkv.w.ptr, sc: l.wkv.sc.as_ref().unwrap().ptr, fmt: 0, n: HD as i32, k: D as i32, xq: aq, xs: as_, ldx: D as i32, ntok: m as i32, y: kvraw, ldy: HD as i32, mul: 1.0, ..Default::default() };
                self.vlaunch("dspark kv", vec![j], self.st.ptr)?;
                g.launch("kv_t", (m as u32, 1, 1), 512, 0, &[Arg::P(kvraw), Arg::P(l.kv_norm.ptr), Arg::P(self.rope_f.ptr), Arg::I(p as i32), Arg::P(lin), Arg::I(WIN as i32)])?;
                g.launch("lin_to_ring", (m as u32, 1, 1), 256, 0, &[Arg::P(lin), Arg::I(p as i32), Arg::I(m as i32), Arg::P(l.ring.ptr)])?;
            }
        }
        Ok(())
    }

    /// Draft the block after the current position, whose token (sampled, not yet fed) is `next`.
    pub fn draft(&mut self, next: u32) -> Result<Vec<u32>> {
        let ds = self.dspark.as_ref().unwrap();
        let mut ids = vec![ds.noise; ds.block];
        ids[0] = next;
        self.run_chunk_mode(&ids, true, true)
    }

    /// Greedy generation with DSpark drafts verified by the main model, from a prefilled state whose next token is
    /// `next`. `emit` gets each committed token and returns false to stop. Returns the counters and the committed
    /// tokens (positions now in the cache, possibly a few past a stop).
    pub fn spec_loop(&mut self, mut next: u32, max_new: usize, emit: &mut dyn FnMut(u32) -> bool) -> Result<(SpecStats, Vec<u32>)> {
        use std::time::Instant;
        let mut stx = SpecStats::default();
        let mut committed = vec![];
        let block = self.dspark.as_ref().unwrap().block;
        // adaptive: a pass costs ~3 decode steps, so below ~2 accepted drafts per pass decode plainly for a while
        // (the decode graph records main_hidden, observed in one batch), then probe again
        let thresh: f64 = std::env::var("DSXA_SPEC_MIN").ok().and_then(|v| v.parse().ok()).unwrap_or(2.0);
        let mut ema = thresh + 1.0;
        while stx.tokens < max_new && self.pos + 1 < self.max_ctx {
            if ema < thresh {
                let pf_mh = self.pf.as_ref().unwrap().dspark_bufs().0;
                let p_start = self.pos;
                let k = 15.min(max_new - stx.tokens).min(self.max_ctx - self.pos - 1);
                let mut stop = false;
                let mut n = 0;
                for i in 0..k {
                    stx.tokens += 1;
                    stx.plain += 1;
                    committed.push(next);
                    let go = emit(next);
                    next = self.step(next)?;
                    self.g.dtod(pf_mh + (i * 3 * D * 4) as u64, self.dmh.ptr, 3 * D * 4)?;
                    n += 1;
                    if !go {
                        stop = true;
                        break;
                    }
                }
                self.observe(p_start, n, 0)?;
                if stop {
                    break;
                }
                ema = thresh + 0.25;
                continue;
            }
            self.g.sync()?;
            let t0 = Instant::now();
            let drafts = self.draft(next)?;
            let t1 = Instant::now();
            let mut feed = vec![next];
            feed.extend_from_slice(&drafts[..block]);
            feed.truncate((self.max_ctx - self.pos - 1).min(max_new - stx.tokens).max(1));
            let p0 = self.pos;
            let out = self.verify(&feed)?;
            let mut acc = 0;
            while acc + 1 < feed.len() && out[acc] == feed[acc + 1] {
                acc += 1;
            }
            self.commit(feed.len(), acc + 1)?;
            self.observe(p0, acc + 1, 0)?;
            self.g.sync()?;
            let t2 = Instant::now();
            stx.passes += 1;
            stx.accepted += acc;
            ema = 0.6 * ema + 0.4 * acc as f64;
            stx.draft_us += (t1 - t0).as_secs_f64() * 1e6;
            stx.verify_us += (t2 - t1).as_secs_f64() * 1e6;
            committed.extend_from_slice(&feed[..=acc]);
            let mut stop = false;
            for &tk in &feed[..=acc] {
                stx.tokens += 1;
                if !emit(tk) {
                    stop = true;
                    break;
                }
            }
            next = out[acc];
            if stop {
                break;
            }
        }
        Ok((stx, committed))
    }
}
