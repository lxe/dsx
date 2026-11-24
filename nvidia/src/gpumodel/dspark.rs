//! DSpark, the checkpoint's speculative draft stack (`mtp.*`), on the main GPU or another one (`Side`); the CPU reference is core/src/dspark.rs
//! and this follows it operation by operation with the decode path's kernels.
//!
//! Three window-only layers (attention over the main model's projected residual of the last `window` positions plus
//! the block, bidirectionally) with their own 128-expert MoE (top-3, checkpoint FP4, `gemv_fp4`) draft `block` tokens
//! at once; the main model's head plus a Markov bias from the previous drafted token finish them in order.
//! `observe` keeps the draft layers' window rings current from `GpuModel::capture_hidden`'s residuals.

use super::{f32buf, DMat, GpuModel, ATTN_SPLITS, KERNELS};
use crate::gguf::GType;
use crate::gpu::{Arg, Buf, Gpu, Kernel};
use crate::model::{Config, Mat, Model};
use std::collections::HashMap;
use crate::safetensors::SafeTensors;
use anyhow::{Context, Result};
use std::path::Path;

use crate::dspark::{dbg, nrm};

/// norm of `n` floats at `ptr` from `off` (debug only)
fn dn(g: &Gpu, ptr: u64, off: usize, n: usize) -> f32 {
    g.download::<f32>(ptr + (off * 4) as u64, n).map(|v| nrm(&v)).unwrap_or(f32::NAN)
}

/// norm of a bf16 row (debug only)
fn dn16(g: &Gpu, ptr: u64, n: usize) -> f32 {
    g.download::<u16>(ptr, n).map(|v| nrm(&v.iter().map(|&b| f32::from_bits((b as u32) << 16)).collect::<Vec<_>>())).unwrap_or(f32::NAN)
}

struct DLayer {
    attn_norm: Buf,
    ffn_norm: Buf,
    q_a: DMat,
    q_a_norm: Buf,
    q_b: DMat,
    kv: DMat,
    kv_norm: Buf,
    o_a: DMat,
    o_b: DMat,
    sinks: Buf,
    hc_attn: (DMat, Buf, Buf),
    hc_ffn: (DMat, Buf, Buf),
    gate_w: DMat,
    gate_bias: Buf,
    /// expert cache: every expert's (gate w1, up w3, down w2) FP4 matrices on the host (the checkpoint mapping), the
    /// resident ones in `slots`; `wtab` / `stab` point a resident expert at its slot and any other at zeros (a draft
    /// expert that is not resident adds nothing: drafts are guesses, the verify pass keeps decoding exact)
    host: Vec<[Mat; 3]>,
    slots: Vec<[DMat; 3]>,
    slot_of: Vec<Option<usize>>,
    who: Vec<usize>,
    wtab_h: [Vec<u64>; 3],
    stab_h: [Vec<u64>; 3],
    /// per matrix (gate, up, down): device tables of every expert's weight and scale pointers, for gemv_fp4_sel
    wtab: [Buf; 3],
    stab: [Buf; 3],
    sh: [DMat; 3],
    /// bf16 [window][head_dim]: the main model's projected residual, post-RoPE, FP8 fake-quantized
    ring: Buf,
}

/// Scratch for one draft (b = block rows)
struct Bufs {
    h: Buf,       // [b][hc][d]
    x: Buf,       // [b][d]
    mix_a: Buf,   // [b][24]
    mix_f: Buf,   // [b][24]
    qr: Buf,      // [b][q_lora]
    q: Buf,       // [b][nh*hd]
    kv: Buf,      // [b][hd] f32
    blk: Buf,     // [b][hd] bf16
    o: Buf,       // [b][nh*hd]
    r: Buf,       // [b][o_groups*o_lora]
    a: Buf,       // [b][d] attention out / MoE out
    sh_out: Buf,  // [b][d]
    pos: Buf,     // [b] i32
    win_idx: Buf, // [b][w] i32
    blk_idx: Buf, // [b][b] i32
    rl: Buf,      // [b][ne] router logits
    sel: Buf,     // [b][k] i32
    wts: Buf,     // [b][k] f32
    eg: Buf,      // [b*k][inter] routed gate (shared: [b][inter])
    eu: Buf,      // [b*k][inter]
    eh: Buf,      // [b*k][inter]
    ey: Buf,      // [b*k][d]
    xq: Buf,      // [b][d] FP8-fake-quantized expert input
    ehq: Buf,     // [b*k][inter]
    toks: Buf,    // [b+1] i32: next, then the drafted tokens
    embs: Buf,    // [b][rank] Markov embeddings used
    mtab: Buf,    // bf16 [vocab][rank] Markov embedding table
    logits: Buf,  // [b][vocab]
    bias: Buf,    // [vocab]
    mh: Buf,      // [3][d]
    mhmix: Buf,   // [3][24]: 1/hc each
    mx: Buf,      // [d]
    kvo: Buf,     // [hd]
}

/// expert cache: counters folded and swaps considered every SWAP_EVERY drafts, at most SWAP_MAX experts per update
const SWAP_EVERY: usize = 16;
const SWAP_MAX: usize = 2;

/// per-section timing of one draft (synchronizes at every mark when on)
struct Prof {
    on: bool,
    t: std::time::Instant,
    v: Vec<(&'static str, f64)>,
}

impl Prof {
    fn mark(&mut self, g: &Gpu, name: &'static str) {
        if self.on {
            let _ = g.sync();
            let e = self.t.elapsed().as_secs_f64();
            match self.v.iter_mut().find(|(k, _)| *k == name) {
                Some(x) => x.1 += e,
                None => self.v.push((name, e)),
            }
            self.t = std::time::Instant::now();
        }
    }
}

/// A context of the stack's own on another GPU than the main one (typically an aux expert GPU with memory to spare):
/// its kernels and scratch, a copy of the main model's head, and on the main GPU the small buffers that hand it the
/// observed residuals. The main GPU keeps all its memory for experts; drafting and observing run on this GPU's stream.
struct Side {
    g: Gpu,
    k: HashMap<&'static str, Kernel>,
    actq: Buf,
    attn_part: Buf,
    freqs0: Buf,
    head: DMat,
    /// on the main GPU: the 1/hc mixes that make the hc-means of the captured rows
    mhmix_main: Buf,
    /// free memory before the stack's first byte on this GPU
    free0: usize,
}

impl Side {
    fn new(gm: &GpuModel, ordinal: i32) -> Result<Side> {
        let (m, c) = (gm.m, &gm.m.cfg);
        anyhow::ensure!(gm.head.q6.is_none(), "the main head's Q6 planes are not copied to another GPU");
        let g = Gpu::new(ordinal)?;
        let free0 = g.mem_info()?.0;
        let mut k = HashMap::new();
        for &n in KERNELS {
            k.insert(n, g.kernel(n)?);
        }
        // the head: the main GPU's device bytes as they are (q8r repacked at upload), once, through the host
        let hb: Vec<u8> = gm.g.download(gm.head.buf.ptr, gm.head.buf.bytes)?;
        let hs = match gm.head.scale.as_ref() {
            Some(s) => Some(g.upload(&gm.g.download::<u8>(s.ptr, s.bytes)?)?),
            None => None,
        };
        let head = DMat { buf: g.upload(&hb)?, scale: hs, ty: gm.head.ty, rows: gm.head.rows, cols: gm.head.cols, act_fp8: gm.head.act_fp8, q8r: gm.head.q8r, q6: None };
        let mut mhmix = vec![0f32; 24];
        mhmix[..4].fill(0.25);
        Ok(Side {
            actq: g.alloc_zeroed(8 * 32768 * 4)?,
            attn_part: g.alloc_zeroed(8 * ATTN_SPLITS * c.n_heads * (c.head_dim + 2) * 4)?,
            freqs0: f32buf(&g, &m.ropes.0.freqs)?,
            head,
            mhmix_main: gm.g.upload(&mhmix)?,
            free0,
            g,
            k,
        })
    }
}

/// The GPU a draft runs on, with the helpers it needs (the decode path's, for either context)
struct Dev<'a> {
    g: &'a Gpu,
    k: &'a HashMap<&'static str, Kernel>,
    actq: u64,
    attn_part: u64,
    freqs0: u64,
    head: &'a DMat,
    c: &'a Config,
    /// the main GPU's one-token kernel variants (bit-identical; GpuModel decides them)
    q8r1: bool,
    sk1: bool,
    /// Some on the main GPU: its Q6 path
    main: Option<&'a GpuModel<'a>>,
}

/// Makes the main GPU's context current again when dropped: the decode path's raw stream calls expect it after a
/// draft or observe ran on another GPU
struct Rebind<'a>(&'a Gpu);

impl Drop for Rebind<'_> {
    fn drop(&mut self) {
        self.0.bind();
    }
}

/// rows of hc-means one observe batch moves (a prompt's last window goes in a few batches)
const MH_ROWS: usize = 64;

fn dev_of<'a>(side: &'a Option<Side>, gm: &'a GpuModel<'a>) -> Dev<'a> {
    match side {
        Some(s) => Dev { g: &s.g, k: &s.k, actq: s.actq.ptr, attn_part: s.attn_part.ptr, freqs0: s.freqs0.ptr, head: &s.head, c: &gm.m.cfg, q8r1: gm.q8r1, sk1: gm.sk1, main: None },
        None => Dev { g: &gm.g, k: &gm.k, actq: gm.actq.ptr, attn_part: gm.attn_part.ptr, freqs0: gm.freqs.0.ptr, head: &gm.head, c: &gm.m.cfg, q8r1: gm.q8r1, sk1: gm.sk1, main: Some(gm) },
    }
}

impl<'a> Dev<'a> {
    fn launch(&self, name: &str, grid: (u32, u32, u32), block: u32, args: &[Arg]) -> Result<()> {
        self.g.launch(self.k[name], grid, block, 0, args)
    }

    fn gemv(&self, w: &DMat, x: u64, y: u64) -> Result<()> {
        self.gemv_rows_ld(w, 0, w.rows, x, w.cols, y, w.rows, 1)
    }

    fn gemv_t(&self, w: &DMat, x: u64, y: u64, t: usize) -> Result<()> {
        self.gemv_rows_ld(w, 0, w.rows, x, w.cols, y, w.rows, t)
    }

    /// GpuModel::gemv_rows_ld for the types the stack and the head use
    #[allow(clippy::too_many_arguments)]
    fn gemv_rows_ld(&self, w: &DMat, r0: usize, n: usize, x: u64, ldx: usize, y: u64, ldy: usize, t: usize) -> Result<()> {
        if w.q6.is_some() {
            let gm = self.main.context("DSpark: Q6 planes off the main GPU")?;
            return gm.gemv_rows_q6(w, r0, n, x, y, t);
        }
        let kname = match w.ty {
            GType::Q8_0 if w.q8r => "gemv_q8r",
            GType::Q8_0 => "gemv_q8_0",
            GType::F16 => "gemv_f16",
            GType::BF16 => "gemv_bf16",
            GType::F32 => "gemv_f32",
            GType::Fp8B32 if !w.act_fp8 => "gemv_fp8b32_o",
            GType::Fp8B32 => "gemv_fp8b32",
            GType::Fp4 => "gemv_fp4",
            t => anyhow::bail!("DSpark: no GPU gemv for {t:?}"),
        };
        let (wp, sp, _) = w.ptrs(r0);
        let x = if w.act_fp8 {
            anyhow::ensure!(ldx == w.cols, "strided GEMV input with FP8 activations");
            self.launch("fake_fp8_copy", ((((t * w.cols) / 32) as u32).div_ceil(8), 1, 1), 256, &[Arg::P(x), Arg::P(self.actq), Arg::I((t * w.cols) as i32)])?;
            self.actq
        } else {
            x
        };
        let sk = n < 2048 && matches!(kname, "gemv_q8r" | "gemv_f16" | "gemv_bf16" | "gemv_f32");
        let kname = if sk { match kname { "gemv_q8r" => "gemv_q8r_sk", "gemv_f16" => "gemv_f16_sk", "gemv_bf16" => "gemv_bf16_sk", _ => "gemv_f32_sk" } } else { kname };
        let kname = if kname == "gemv_q8r" && t > 1 && n % 16 == 0 && !std::env::var("DSX_Q8R4").is_ok_and(|v| v == "0") { "gemv_q8r4" } else { kname };
        let kname = if kname == "gemv_q8r" && t == 1 && self.q8r1 { "gemv_q8r1" } else { kname };
        let kname = match kname {
            "gemv_bf16_sk" if t == 1 && self.sk1 => "gemv_bf16_sk1",
            "gemv_f16_sk" if t == 1 && self.sk1 => "gemv_f16_sk1",
            "gemv_q8r_sk" if t == 1 && self.sk1 => "gemv_q8r_sk1",
            k => k,
        };
        let per_block = if sk { 1 } else if matches!(kname, "gemv_q8r4" | "gemv_fp8b32" | "gemv_fp4") { 16 } else { 4 };
        self.launch(kname, ((n as u32).div_ceil(per_block), 1, 1), 128, &[Arg::P(wp), Arg::P(x), Arg::P(y), Arg::I(n as i32), Arg::I(w.cols as i32), Arg::I(t as i32), Arg::I(ldx as i32), Arg::I(ldy as i32), Arg::P(sp)])
    }

    fn rms_rows(&self, x: u64, w: &Buf, out: u64, n: usize, rows: usize, round: bool) -> Result<()> {
        self.launch("rmsnorm", (rows as u32, 1, 1), 1024, &[Arg::P(x), Arg::P(w.ptr), Arg::P(out), Arg::I(n as i32), Arg::I(n as i32), Arg::I(n as i32), Arg::F(self.c.eps), Arg::I(round as i32)])
    }

    fn rope_rows(&self, x: u64, heads: usize, hd: usize, _compressed: bool, pos: u64, rows: usize, inverse: bool) -> Result<()> {
        self.launch("rope_pos", (rows as u32, 1, 1), 256, &[Arg::P(x), Arg::P(self.freqs0), Arg::I(heads as i32), Arg::I(hd as i32), Arg::I(self.c.rope_dim as i32), Arg::I((heads * hd) as i32), Arg::P(pos), Arg::I(inverse as i32)])
    }

    fn hc_mixes_rows(&self, fnw: &(DMat, Buf, Buf), h: u64, out: u64, n: usize) -> Result<()> {
        let c = self.c;
        self.launch("hc_dots_t", (6, (n as u32).div_ceil(4), 1), 256, &[Arg::P(fnw.0.buf.ptr), Arg::P(h), Arg::P(out), Arg::I((c.hc * c.dim) as i32), Arg::F(c.eps), Arg::I(n as i32)])?;
        self.launch("hc_finish_w", ((n as u32).div_ceil(4), 1, 1), 128, &[Arg::P(out), Arg::P(fnw.1.ptr), Arg::P(fnw.2.ptr), Arg::F(c.hc_eps), Arg::I(c.sinkhorn_iters as i32), Arg::I(n as i32)])
    }

    #[allow(clippy::too_many_arguments)]
    fn sparse_attn_small(&self, t: usize, q: u64, win: u64, win_idx: u64, w: usize, comp: u64, comp_idx: u64, k: usize, sinks: u64, o: u64) -> Result<()> {
        let (nh, hd) = (self.c.n_heads, self.c.head_dim);
        let scale = 1.0 / (hd as f32).sqrt();
        let s = (w + k).div_ceil(32);
        anyhow::ensure!(s <= ATTN_SPLITS, "attention over {} rows: more than {ATTN_SPLITS} tiles", w + k);
        self.launch("sparse_attn_tile", (t as u32, (nh / 16) as u32, s as u32), 256, &[Arg::P(q), Arg::P(win), Arg::P(win_idx), Arg::I(w as i32), Arg::P(comp), Arg::P(comp_idx), Arg::I(k as i32), Arg::P(self.attn_part), Arg::I(nh as i32), Arg::I(hd as i32), Arg::F(scale)])?;
        self.launch("sparse_attn_merge", (t as u32, nh as u32, 1), 256, &[Arg::P(self.attn_part), Arg::I(s as i32), Arg::P(sinks), Arg::P(o), Arg::I(nh as i32), Arg::I(hd as i32)])
    }

    fn fake_fp8(&self, x: u64, n: usize) -> Result<()> {
        self.launch("fake_fp8", ((n as u32 / 32).div_ceil(8), 1, 1), 256, &[Arg::P(x), Arg::I(n as i32)])
    }
}

/// The background read that brings the draft experts' pages into the page cache. It reads through raw pointers into the
/// checkpoint's mapping, so it must end before the mapping goes: dropping this stops and joins it (declared before the
/// mapping wherever both live, so it drops first, also when the load fails part-way).
struct Prefault {
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl Prefault {
    /// `spans` (address, length) must stay mapped until this is dropped.
    fn start(spans: Vec<(usize, usize)>) -> Prefault {
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let s2 = stop.clone();
        let handle = std::thread::Builder::new()
            .name("dspark-prefault".into())
            .spawn(move || {
                let mut s = 0u8;
                for (p, n) in spans {
                    for o in (0..n).step_by(4096) {
                        if o % (1 << 20) == 0 && s2.load(std::sync::atomic::Ordering::Relaxed) {
                            return;
                        }
                        s = s.wrapping_add(unsafe { std::ptr::read_volatile((p + o) as *const u8) });
                    }
                }
                std::hint::black_box(s);
            })
            .ok();
        Prefault { stop, handle }
    }
}

impl Drop for Prefault {
    fn drop(&mut self) {
        self.stop.store(true, std::sync::atomic::Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

pub struct DsparkGpu {
    layers: Vec<DLayer>,
    main_proj: DMat,
    main_norm: Buf,
    norm: Buf,
    /// rows read on the host per drafted token; points into `_st`'s mapping, which must outlive it
    markov_embed: Mat,
    /// reads `_st`'s mapping: dropped (stopped and joined) before it
    _prefault: Prefault,
    _st: SafeTensors,
    markov_head: DMat,
    conf: Vec<f32>,
    b: Bufs,
    pub block: usize,
    pub noise: u32,
    pub targets: Vec<usize>,
    top_k: usize,
    ne: usize,
    /// device memory the stack took (weights and scratch)
    pub bytes: usize,
    ring_saved: Vec<Buf>,
    /// (first, last) position of the contiguous run the rings hold; None = nothing usable
    seen: Option<(usize, usize)>,
    /// on the main GPU: [window][3][dim] hc-means by position % window, and the contiguous run of positions in it
    hist: Buf,
    noted: Option<(usize, usize)>,
    seen_saved: Option<(usize, usize)>,
    prof_on: bool,
    prof: Vec<(&'static str, f64)>,
    prof_n: usize,
    /// DSX_DSPARK_PROF: draft expert picks per layer
    usage: Vec<Vec<u64>>,
    /// expert cache: device pick counters [layer][expert], their decayed host sums, drafts since the last update
    counts: Buf,
    heat: Vec<Vec<f64>>,
    since_swap: usize,
    pub swaps: usize,
    drafts: usize,
    updates: usize,
    heat_file: Option<std::path::PathBuf>,
    _zero: [DMat; 3],
    /// Some: the stack lives on another GPU than the main one
    side: Option<Side>,
    /// the GPU ordinal it runs on
    pub ordinal: i32,
    /// the checkpoint directory it came from
    pub dir: std::path::PathBuf,
}

fn up(g: &Gpu, w: &Mat) -> Result<DMat> {
    let scale = if w.scale_bytes().is_empty() { None } else { Some(g.upload(w.scale_bytes())?) };
    Ok(DMat { buf: g.upload(w.bytes())?, scale, ty: w.ty, rows: w.rows, cols: w.cols, act_fp8: w.act_fp8, q8r: false, q6: None })
}

fn f32mat(g: &Gpu, v: &[f32], rows: usize) -> Result<DMat> {
    Ok(DMat { buf: g.upload(v)?, scale: None, ty: GType::F32, rows, cols: v.len() / rows, act_fp8: false, q8r: false, q6: None })
}

fn bf16mat(g: &Gpu, v: &[f32], rows: usize) -> Result<DMat> {
    let h: Vec<half::bf16> = v.iter().map(|&x| half::bf16::from_f32(x)).collect();
    Ok(DMat { buf: g.upload(&h)?, scale: None, ty: GType::BF16, rows, cols: v.len() / rows, act_fp8: false, q8r: false, q6: None })
}

impl DsparkGpu {
    /// Device bytes `load` will take (weights, the bf16 Markov table, scratch), from the index alone: the placement
    /// sets this much aside on the main GPU before it fills it with experts.
    pub fn estimate_bytes(dir: &Path) -> Result<usize> {
        let st = SafeTensors::open_prefix(dir, "mtp.")?;
        let ne = 128.0f64;
        let slots = std::env::var("DSX_DSPARK_SLOTS").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(16.0).clamp(1.0, ne);
        let w: usize = st.tensors.iter().map(|(k, t)| if k.contains(".ffn.experts.") { ((t.end - t.start) as f64 * (slots + 1.0) / ne) as usize } else { t.end - t.start }).sum();
        // F32 / BF16 vectors are uploaded as f32 (at most 2x), the Markov embedding again as a bf16 table
        let extra: usize = st.tensors.iter().filter(|(k, t)| t.dtype != "I8" && t.dtype != "F8_E4M3" && !k.contains("experts")).map(|(_, t)| t.end - t.start).sum();
        Ok(w + 2 * extra + (256 << 20))
    }

    /// Upload the draft stack from the checkpoint directory holding config.json, the index and the mtp.* shards.
    /// `ordinal`: the GPU to put it on (None or the main GPU's: the main GPU)
    pub fn load(gm: &GpuModel, dir: &Path, ordinal: Option<i32>) -> Result<DsparkGpu> {
        let _rb = Rebind(&gm.g);
        let side = match ordinal {
            Some(o) if o != gm.g.ordinal => Some(Side::new(gm, o).with_context(|| format!("DSpark context on GPU{o}"))?),
            _ => None,
        };
        let m: &Model = gm.m;
        let g: &Gpu = side.as_ref().map_or(&gm.g, |s| &s.g);
        let c = &m.cfg;
        g.bind();
        let free0 = match &side {
            Some(s) => s.free0,
            None => g.mem_info()?.0,
        };
        let st = SafeTensors::open_prefix(dir, "mtp.")?;
        let cfg: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("config.json")).context("mtp config.json")?)?;
        let t = &cfg["text_config"];
        let nl = t["num_nextn_predict_layers"].as_u64().unwrap_or(3) as usize;
        let ne = t["dspark_n_routed_experts"].as_u64().unwrap_or(128) as usize;
        let block = t["dspark_block_size"].as_u64().unwrap_or(5) as usize;
        let top_k = t["dspark_num_experts_per_tok"].as_u64().unwrap_or(3) as usize;
        anyhow::ensure!(block + 1 <= crate::handoff::MAXT && top_k <= 8 && c.hc == 4, "unexpected DSpark config (block {block}, top-k {top_k})");
        let mat = |n: &str| crate::model::hf_mat_pub(&st, n);
        let v = |n: &str| -> Result<Vec<f32>> { Ok(crate::model::mat_f32_pub(&mat(n)?)) };
        let (d, hd, w) = (c.dim, c.head_dim, c.window);
        // resident draft experts per layer (DSX_DSPARK_SLOTS, default 16 of 128: on an RTX 3090 + CMP 170HX 16 measured
        // +1.5 % on prose and +2.3 % on code over 32, the freed memory holding main-model experts)
        let nslots = std::env::var("DSX_DSPARK_SLOTS").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(16).clamp(1, ne);
        let mut zero: Option<[DMat; 3]> = None;
        // the heat a previous run saved: its hottest experts start resident (else the router bias decides)
        let saved_heat: Option<Vec<Vec<f64>>> = Self::heat_path(dir)
            .and_then(|p| std::fs::read(p).ok())
            .and_then(|b| serde_json::from_slice::<Vec<Vec<f64>>>(&b).ok())
            .filter(|h| h.len() == nl && h.iter().all(|r| r.len() == ne));
        let mut layers = vec![];
        for l in 0..nl {
            let p = |s: &str| format!("mtp.{l}.{s}");
            let mut o_a = mat(&p("attn.wo_a.weight"))?;
            o_a.act_fp8 = false; // the reference fake-quantizes o per group itself
            let ex = |e: usize, wn: &str| -> Result<Mat> { mat(&p(&format!("ffn.experts.{e}.{wn}.weight"))) };
            let host = (0..ne).map(|e| Ok([ex(e, "w1")?, ex(e, "w3")?, ex(e, "w2")?])).collect::<Result<Vec<[Mat; 3]>>>()?;
            anyhow::ensure!(host.iter().all(|x| x.iter().all(|m| m.ty == GType::Fp4)), "draft experts are not checkpoint FP4");
            if zero.is_none() {
                // one all-zero expert (weights and E8M0 scales 0 = 2^-127 ~ 0): what a non-resident pick reads
                let z = |m: &Mat| -> Result<DMat> {
                    Ok(DMat { buf: g.alloc_zeroed(m.bytes().len())?, scale: Some(g.alloc_zeroed(m.scale_bytes().len())?), ty: m.ty, rows: m.rows, cols: m.cols, act_fp8: m.act_fp8, q8r: false, q6: None })
                };
                zero = Some([z(&host[0][0])?, z(&host[0][1])?, z(&host[0][2])?]);
            }
            let zr = zero.as_ref().unwrap();
            // first residents: the router's bias favours them (no use measured yet)
            let bias = v(&p("ffn.gate.bias"))?;
            let mut order: Vec<usize> = (0..ne).collect();
            match &saved_heat {
                Some(h) => order.sort_by(|&a, &b| h[l][b].total_cmp(&h[l][a])),
                None => order.sort_by(|&a, &b| bias[b].total_cmp(&bias[a])),
            }
            let mut slots = vec![];
            let mut slot_of = vec![None; ne];
            let mut who = vec![];
            for &e in order.iter().take(nslots) {
                slot_of[e] = Some(slots.len());
                who.push(e);
                slots.push([up(g, &host[e][0])?, up(g, &host[e][1])?, up(g, &host[e][2])?]);
            }
            let ptrs = |j: usize, s: bool| -> Vec<u64> {
                (0..ne)
                    .map(|e| {
                        let m = match slot_of[e] {
                            Some(i) => &slots[i][j],
                            None => &zr[j],
                        };
                        if s { m.scale.as_ref().unwrap().ptr } else { m.buf.ptr }
                    })
                    .collect()
            };
            let (wtab_h, stab_h) = ([ptrs(0, false), ptrs(1, false), ptrs(2, false)], [ptrs(0, true), ptrs(1, true), ptrs(2, true)]);
            let (wtab, stab) = ([g.upload(&wtab_h[0])?, g.upload(&wtab_h[1])?, g.upload(&wtab_h[2])?], [g.upload(&stab_h[0])?, g.upload(&stab_h[1])?, g.upload(&stab_h[2])?]);
            layers.push(DLayer {
                attn_norm: f32buf(g, &v(&p("attn_norm.weight"))?)?,
                ffn_norm: f32buf(g, &v(&p("ffn_norm.weight"))?)?,
                q_a: up(g, &mat(&p("attn.wq_a.weight"))?)?,
                q_a_norm: f32buf(g, &v(&p("attn.q_norm.weight"))?)?,
                q_b: up(g, &mat(&p("attn.wq_b.weight"))?)?,
                kv: up(g, &mat(&p("attn.wkv.weight"))?)?,
                kv_norm: f32buf(g, &v(&p("attn.kv_norm.weight"))?)?,
                o_a: up(g, &o_a)?,
                o_b: up(g, &mat(&p("attn.wo_b.weight"))?)?,
                sinks: f32buf(g, &v(&p("attn.attn_sink"))?)?,
                hc_attn: (f32mat(g, &v(&p("hc_attn_fn"))?, 24)?, f32buf(g, &v(&p("hc_attn_scale"))?)?, f32buf(g, &v(&p("hc_attn_base"))?)?),
                hc_ffn: (f32mat(g, &v(&p("hc_ffn_fn"))?, 24)?, f32buf(g, &v(&p("hc_ffn_scale"))?)?, f32buf(g, &v(&p("hc_ffn_base"))?)?),
                gate_w: bf16mat(g, &v(&p("ffn.gate.weight"))?, ne)?,
                gate_bias: f32buf(g, &v(&p("ffn.gate.bias"))?)?,
                host,
                slots,
                slot_of,
                who,
                wtab_h,
                stab_h,
                wtab,
                stab,
                sh: [
                    up(g, &mat(&p("ffn.shared_experts.w1.weight"))?)?,
                    up(g, &mat(&p("ffn.shared_experts.w3.weight"))?)?,
                    up(g, &mat(&p("ffn.shared_experts.w2.weight"))?)?,
                ],
                ring: g.alloc_zeroed(w * hd * 2)?,
            });
        }
        // bring every draft expert's bytes into the page cache in the background, so swapping one in never waits for
        // the disk
        // (a later `?` drops `prefault` before `st`: locals drop in reverse order, so the thread ends first)
        let prefault = {
            let spans: Vec<(usize, usize)> = layers.iter().flat_map(|l: &DLayer| l.host.iter().flat_map(|x| x.iter().flat_map(|m| [(m.bytes().as_ptr() as usize, m.bytes().len()), (m.scale_bytes().as_ptr() as usize, m.scale_bytes().len())]))).collect();
            Prefault::start(spans)
        };
        let last = nl - 1;
        let markov_embed = mat(&format!("mtp.{last}.markov_head.embed.weight"))?;
        let rank = markov_embed.cols;
        let markov_head = up(g, &mat(&format!("mtp.{last}.markov_head.head.weight"))?)?;
        let (b, nh, inter, vocab) = (block, c.n_heads, c.inter, c.vocab);
        let fz = |n: usize| g.alloc_zeroed(n * 4);
        let mut mhmix = vec![0f32; 3 * 24];
        for r in 0..3 {
            for k in 0..4 {
                mhmix[r * 24 + k] = 0.25;
            }
        }
        let bufs = Bufs {
            h: fz(b * 4 * d)?,
            x: fz(b * d)?,
            mix_a: fz(b * 24)?,
            mix_f: fz(b * 24)?,
            qr: fz(b * c.q_lora)?,
            q: fz(b * nh * hd)?,
            kv: fz(b * hd)?,
            blk: g.alloc_zeroed(b * hd * 2)?,
            o: fz(b * nh * hd)?,
            r: fz(b * c.o_groups * c.o_lora)?,
            a: fz(b * d)?,
            sh_out: fz(b * d)?,
            pos: fz(b)?,
            win_idx: fz(b * w)?,
            blk_idx: g.upload(&(0..b * b).map(|i| (i % b) as i32).collect::<Vec<i32>>())?,
            rl: fz(b * ne)?,
            sel: fz(b * top_k)?,
            wts: fz(b * top_k)?,
            eg: fz(b * top_k * inter)?,
            eu: fz(b * top_k * inter)?,
            eh: fz(b * top_k * inter)?,
            ey: fz(b * top_k * d)?,
            xq: fz(b * d)?,
            ehq: fz(b * top_k * inter)?,
            toks: fz(b + 1)?,
            embs: fz(b * rank)?,
            mtab: {
                let mut h = vec![half::bf16::ZERO; markov_embed.rows * rank];
                let mut row = vec![0f32; rank];
                for r in 0..markov_embed.rows {
                    markov_embed.dequant(r, &mut row);
                    for (o, &v) in h[r * rank..(r + 1) * rank].iter_mut().zip(&row) {
                        *o = half::bf16::from_f32(v);
                    }
                }
                g.upload(&h)?
            },
            logits: fz(b * vocab.max(gm.head.rows))?,
            bias: fz(vocab.max(markov_head.rows))?,
            mh: fz(MH_ROWS * 3 * d)?,
            mhmix: g.upload(&mhmix)?,
            mx: fz(crate::handoff::MAXT * d)?,
            kvo: fz(crate::handoff::MAXT * hd)?,
        };
        let counts = g.alloc_zeroed(nl * ne * 4)?;
        let hist = gm.g.alloc_zeroed(c.window * 3 * c.dim * 4)?;
        let main_proj = up(g, &mat("mtp.0.main_proj.weight")?)?;
        let main_norm = f32buf(g, &v("mtp.0.main_norm.weight")?)?;
        let norm = f32buf(g, &v(&format!("mtp.{last}.norm.weight"))?)?;
        let (free1, _) = g.mem_info()?;
        let ordinal = g.ordinal;
        let ds = DsparkGpu {
            layers,
            main_proj,
            main_norm,
            norm,
            markov_head,
            conf: v(&format!("mtp.{last}.confidence_head.proj.weight"))?,
            markov_embed,
            _prefault: prefault,
            _st: st,
            b: bufs,
            block,
            noise: t["dspark_noise_token_id"].as_u64().unwrap_or(128799) as u32,
            targets: t["dspark_target_layer_ids"].as_array().map(|a| a.iter().filter_map(|x| x.as_u64()).map(|x| x as usize).collect()).unwrap_or(vec![37, 38, 39]),
            top_k,
            ne,
            bytes: 0,
            ring_saved: vec![],
            seen: None,
            hist,
            noted: None,
            seen_saved: None,
            prof_on: std::env::var_os("DSX_DSPARK_PROF").is_some(),
            prof: vec![],
            prof_n: 0,
            usage: vec![],
            counts,
            heat: saved_heat.unwrap_or_else(|| vec![vec![0.0; ne]; nl]),
            since_swap: 0,
            swaps: 0,
            drafts: 0,
            updates: 0,
            heat_file: Self::heat_path(dir),
            _zero: zero.context("no draft layers")?,
            side,
            ordinal,
            dir: dir.to_path_buf(),
        };
        anyhow::ensure!(ds.targets.len() == 3, "DSpark reads 3 target layers, config has {:?}", ds.targets);
        eprintln!("dspark: GPU{ordinal}, main head {:?} {}x{} (q8r {}, q6 {})", gm.head.ty, gm.head.rows, gm.head.cols, gm.head.q8r, gm.head.q6.is_some());
        Ok(DsparkGpu { bytes: free0.saturating_sub(free1), ..ds })
    }

    /// The main model just processed position `pos` (its `step` captured row 0, a verify pass row t per token, of
    /// the target layers' residuals): project row `t` into every draft layer's window ring (the reference `observe`).
    pub fn observe_row(&mut self, gm: &GpuModel, pos: usize, t: usize) -> Result<()> {
        let hid = gm.hid_buf.as_ref().context("capture_hidden(targets) first")?.ptr;
        anyhow::ensure!(t < crate::handoff::MAXT, "observe row {t}");
        self.observe_rows(gm, hid, crate::handoff::MAXT, &[(pos, t)])
    }

    /// Rows `t0..t0 + n` of a verify pass, at positions pos0.. (one batch: one hand-over to the draft GPU)
    pub fn observe_span(&mut self, gm: &GpuModel, pos0: usize, t0: usize, n: usize) -> Result<()> {
        let hid = gm.hid_buf.as_ref().context("capture_hidden(targets) first")?.ptr;
        anyhow::ensure!(t0 + n <= crate::handoff::MAXT, "observe rows {t0}+{n}");
        let items: Vec<(usize, usize)> = (0..n).map(|i| (pos0 + i, t0 + i)).collect();
        self.observe_rows(gm, hid, crate::handoff::MAXT, &items)
    }

    /// After a prompt pass (prefill): the captured last <= window rows of the prompt (GpuModel::hid_pre_span).
    pub fn observe_prefill(&mut self, gm: &GpuModel) -> Result<()> {
        let (p0, n) = gm.hid_pre_span;
        let Some(hp) = gm.hid_pre.as_ref() else { return Ok(()) };
        let items: Vec<(usize, usize)> = (0..n).map(|r| (p0 + r, r)).collect();
        self.observe_rows(gm, hp.ptr, gm.m.cfg.window, &items)
    }

    pub fn invalidate(&mut self) {
        self.seen = None;
        self.noted = None;
    }

    /// Positions the rings hold for the main model now at `pos`: the newest window of a contiguous run of observes.
    fn valid_from(&self, pos: usize, w: usize) -> Option<usize> {
        let (lo, hi) = self.seen?;
        (hi == pos).then(|| lo.max((pos + 1).saturating_sub(w)))
    }

    /// items (position, row): the main GPU reduces each captured row to its hc-means (the draft's main_hidden) into
    /// `hist`, a window of them by position (no synchronization, a few tiny kernels). The draft GPU projects them into
    /// its rings only when a draft needs them (`catch_up`): plain decoding pays almost nothing for DSpark.
    fn observe_rows(&mut self, gm: &GpuModel, base: u64, rows_per_layer: usize, items: &[(usize, usize)]) -> Result<()> {
        let _rb = Rebind(&gm.g);
        let c = &gm.m.cfg;
        let (d, w) = (c.dim, c.window);
        anyhow::ensure!(gm.hid_layers == self.targets, "captured layers {:?}, DSpark reads {:?}", gm.hid_layers, self.targets);
        let row = c.hc * d * 4;
        // main_hidden: per target layer the mean of the hc streams, bf16 (hc_pre with 1/4 weights rounds)
        let mix = match &self.side {
            Some(s) => s.mhmix_main.ptr,
            None => self.b.mhmix.ptr,
        };
        for &(pos, t) in items {
            for k in 0..3 {
                gm.g.launch(gm.k["hc_pre"], (1, 1, 1), 256, 0, &[Arg::P(base + ((k * rows_per_layer + t) * row) as u64), Arg::P(mix), Arg::I(0), Arg::P(self.hist.ptr + (((pos % w) * 3 + k) * d * 4) as u64), Arg::I(d as i32)])?;
            }
            // contiguous with what `hist` holds, else a new run starts here
            self.noted = match self.noted {
                Some((lo, hi)) if pos == hi + 1 => Some((lo, pos)),
                Some((lo, hi)) if pos <= hi && pos >= lo => Some((lo, pos)),
                _ => Some((pos, pos)),
            };
            // a rewritten position drops what the rings hold from there on
            if let Some((lo, hi)) = self.seen {
                if pos <= hi {
                    self.seen = (pos > lo).then_some((lo, pos - 1));
                }
            }
        }
        Ok(())
    }

    /// Project the noted positions the rings lack, up to `pos`, into every layer's window ring (on another draft GPU
    /// they cross over the host first, 3 x dim floats per row, in batches of MH_ROWS).
    pub fn catch_up(&mut self, gm: &GpuModel, pos: usize) -> Result<()> {
        let _rb = Rebind(&gm.g);
        let c = &gm.m.cfg;
        let (d, hd, w) = (c.dim, c.head_dim, c.window);
        let Some((nlo, nhi)) = self.noted else { return Ok(()) };
        if nhi != pos {
            return Ok(());
        }
        let need = nlo.max((pos + 1).saturating_sub(w));
        let start = match self.seen {
            Some((lo, hi)) if hi + 1 >= need && hi <= pos && lo <= need.max(lo) => hi + 1,
            _ => need,
        };
        if start > pos {
            return Ok(());
        }
        let continues = start != need || matches!(self.seen, Some((_, hi)) if hi + 1 == start);
        let positions: Vec<usize> = (start..=pos).collect();
        for chunk in positions.chunks(MH_ROWS) {
            let dv = dev_of(&self.side, gm);
            let b = &self.b;
            // the chunk's rows contiguous in b.mh on the draft GPU: from hist over the host, or on the main GPU a device copy
            {
                let mut v: Vec<f32> = Vec::new();
                let mut i = 0;
                while i < chunk.len() {
                    // contiguous slot runs (the window wraps at most once per chunk)
                    let s0 = chunk[i] % w;
                    let mut n = 1;
                    while i + n < chunk.len() && chunk[i + n] % w == s0 + n {
                        n += 1;
                    }
                    let src = self.hist.ptr + (s0 * 3 * d * 4) as u64;
                    if self.side.is_some() {
                        v.extend(gm.g.download::<f32>(src, n * 3 * d)?);
                    } else {
                        gm.g.dtod(b.mh.ptr + (i * 3 * d * 4) as u64, src, n * 3 * d * 4)?;
                    }
                    i += n;
                }
                if self.side.is_some() {
                    dv.g.htod_sync(b.mh.ptr, &v)?;
                }
            }
            // up to MAXT rows per pass: each weight is read once per batch, not once per row
            let mt = crate::handoff::MAXT;
            for s in (0..chunk.len()).step_by(mt) {
                let t = mt.min(chunk.len() - s);
                let x = b.mh.ptr + (s * 3 * d * 4) as u64;
                dv.gemv_t(&self.main_proj, x, b.mx.ptr, t)?;
                dv.rms_rows(b.mx.ptr, &self.main_norm, b.mx.ptr, d, t, true)?;
                if dbg() {
                    for r in 0..t {
                        eprintln!("dspark-dbg gpu observe p{}: main_hidden {:.4} mx {:.4}", chunk[s + r], dn(dv.g, x + (r * 3 * d * 4) as u64, 0, 3 * d), dn(dv.g, b.mx.ptr + (r * d * 4) as u64, 0, d));
                    }
                }
                for l in &self.layers {
                    dv.gemv_t(&l.kv, b.mx.ptr, b.kvo.ptr, t)?;
                    for r in 0..t {
                        let p = chunk[s + r];
                        dv.g.launch(dv.k["kv_norm_rope_store"], (1, 1, 1), 1024, 0, &[Arg::P(b.kvo.ptr + (r * hd * 4) as u64), Arg::P(l.kv_norm.ptr), Arg::I(hd as i32), Arg::F(c.eps), Arg::P(dv.freqs0), Arg::I(c.rope_dim as i32), Arg::I(p as i32), Arg::P(l.ring.ptr + ((p % w) * hd * 2) as u64)])?;
                    }
                }
            }
        }
        self.seen = match self.seen {
            Some((lo, _)) if continues => Some((lo, pos)),
            _ => Some((start, pos)),
        };
        Ok(())
    }

    pub fn observe(&mut self, gm: &GpuModel, pos: usize) -> Result<()> {
        self.observe_row(gm, pos, 0)
    }

    /// Copy the window rings aside / back (with GpuModel::snapshot / restore)
    pub fn save_rings(&mut self, gm: &GpuModel) -> Result<()> {
        let _rb = Rebind(&gm.g);
        if let Some((_, hi)) = self.noted {
            self.catch_up(gm, hi)?;
        }
        let n = gm.m.cfg.window * gm.m.cfg.head_dim * 2;
        if self.ring_saved.is_empty() {
            for _ in &self.layers {
                self.ring_saved.push(self.side.as_ref().map_or(&gm.g, |s| &s.g).alloc(n)?);
            }
        }
        for (l, s) in self.layers.iter().zip(&self.ring_saved) {
            self.side.as_ref().map_or(&gm.g, |s| &s.g).dtod(s.ptr, l.ring.ptr, n)?;
        }
        self.seen_saved = self.seen;
        Ok(())
    }

    pub fn restore_rings(&mut self, gm: &GpuModel) -> Result<()> {
        let _rb = Rebind(&gm.g);
        self.seen = self.seen_saved;
        self.noted = self.seen;
        let n = gm.m.cfg.window * gm.m.cfg.head_dim * 2;
        for (l, s) in self.layers.iter().zip(&self.ring_saved) {
            self.side.as_ref().map_or(&gm.g, |s| &s.g).dtod(l.ring.ptr, s.ptr, n)?;
        }
        Ok(())
    }

    /// section timing on/off (it synchronizes per section, so speed runs turn it off)
    pub fn set_prof(&mut self, on: bool) {
        self.prof_on = on;
    }

    /// DSX_DSPARK_PROF=1: share of the draft experts' picks the hottest 8 / 16 / 32 / 64 experts per layer took
    pub fn usage_report(&self) -> String {
        self.usage
            .iter()
            .enumerate()
            .map(|(li, u)| {
                let mut v = u.clone();
                v.sort_unstable_by(|a, b| b.cmp(a));
                let tot = v.iter().sum::<u64>().max(1) as f64;
                let top = |n: usize| 100.0 * v.iter().take(n).sum::<u64>() as f64 / tot;
                format!("L{li}: top8 {:.0}% top16 {:.0}% top32 {:.0}% top64 {:.0}%", top(8), top(16), top(32), top(64))
            })
            .collect::<Vec<_>>()
            .join("; ")
    }

    /// DSX_DSPARK_PROF=1: milliseconds per draft section (each section synchronized), averaged
    pub fn prof_report(&self) -> String {
        let n = self.prof_n.max(1) as f64;
        self.prof.iter().map(|(k, v)| format!("{k} {:.2}", v * 1e3 / n)).collect::<Vec<_>>().join(", ")
    }

    /// Draft `block` tokens after position `pos`, whose next token (sampled by the main model, not yet fed) is
    /// `next`. Returns (drafts, confidence): drafts[i] is the guess for position pos + 2 + i.
    pub fn draft(&mut self, gm: &GpuModel, pos: usize, next: u32) -> Result<(Vec<u32>, Vec<f32>)> {
        let _rb = Rebind(&gm.g);
        self.catch_up(gm, pos)?;
        let m = gm.m;
        let c = &m.cfg;
        let (d, hc, hd, nh, w, bl) = (c.dim, c.hc, c.head_dim, c.n_heads, c.window, self.block);
        let (inter, k) = (c.inter, self.top_k);
        let dv = dev_of(&self.side, gm);
        let g = dv.g;
        let b = &self.b;
        let f4 = |n: usize| n * 4;
        let mut pf = Prof { on: self.prof_on, t: std::time::Instant::now(), v: vec![] };
        if pf.on {
            let _ = g.sync();
            pf.t = std::time::Instant::now();
        }
        // inputs: embeddings (next, then noise), one-hot pre mixes, positions, window rows in use
        let mut h = vec![0f32; bl * hc * d];
        let mut e = vec![0f32; d];
        for i in 0..bl {
            m.embed.dequant(if i == 0 { next } else { self.noise } as usize, &mut e);
            for s in 0..hc {
                h[(i * hc + s) * d..(i * hc + s + 1) * d].copy_from_slice(&e);
            }
        }
        g.htod_sync(b.h.ptr, &h)?;
        let mut pre = vec![0f32; bl * 24];
        for i in 0..bl {
            pre[i * 24] = 1.0;
        }
        g.htod_sync(b.mix_f.ptr, &pre)?;
        g.htod_sync(b.pos.ptr, &(0..bl).map(|i| (pos + 1 + i) as i32).collect::<Vec<i32>>())?;
        // ring slots of the positions observed in the current run (a slot outside it holds another sequence's row)
        let lo = self.valid_from(pos, w).context("draft without observed positions (observe the main model first)")?;
        // slot s holds the newest position p <= pos with p % w == s
        let slot = |s: usize| -> i32 {
            let off = (pos % w + w - s) % w;
            if off <= pos && pos - off >= lo { s as i32 } else { -1 }
        };
        let widx: Vec<i32> = (0..bl * w).map(|j| slot(j % w)).collect();
        g.htod_sync(b.win_idx.ptr, &widx)?;
        let gsz = nh * hd / c.o_groups;
        pf.mark(g, "setup");
        let mut picks: Vec<(usize, Vec<i32>)> = vec![];
        for (li, l) in self.layers.iter().enumerate() {
            // attention half, all block rows together
            dv.hc_mixes_rows(&l.hc_attn, b.h.ptr, b.mix_a.ptr, bl)?;
            g.launch(dv.k["hc_pre"], (bl as u32, 1, 1), 256, 0, &[Arg::P(b.h.ptr), Arg::P(b.mix_f.ptr), Arg::I(0), Arg::P(b.x.ptr), Arg::I(d as i32)])?;
            dv.rms_rows(b.x.ptr, &l.attn_norm, b.x.ptr, d, bl, true)?;
            dv.gemv_t(&l.q_a, b.x.ptr, b.qr.ptr, bl)?;
            dv.rms_rows(b.qr.ptr, &l.q_a_norm, b.qr.ptr, c.q_lora, bl, true)?;
            dv.gemv_t(&l.q_b, b.qr.ptr, b.q.ptr, bl)?;
            dv.rope_rows(b.q.ptr, nh, hd, false, b.pos.ptr, bl, false)?;
            dv.gemv_t(&l.kv, b.x.ptr, b.kv.ptr, bl)?;
            for i in 0..bl {
                g.launch(dv.k["kv_norm_rope_store"], (1, 1, 1), 1024, 0, &[Arg::P(b.kv.ptr + f4(i * hd) as u64), Arg::P(l.kv_norm.ptr), Arg::I(hd as i32), Arg::F(c.eps), Arg::P(dv.freqs0), Arg::I(c.rope_dim as i32), Arg::I((pos + 1 + i) as i32), Arg::P(b.blk.ptr + (i * hd * 2) as u64)])?;
            }
            // window rows, then every block row (bidirectional), the sink in the softmax; out bf16-rounded
            if dbg() {
                let li = self.layers.iter().position(|x| std::ptr::eq(x, l)).unwrap();
                eprintln!("dspark-dbg gpu L{li}: x0 {:.4} q0 {:.4} kv0 {:.4} ring0 {:.4}", dn(g, b.x.ptr, 0, d), dn(g, b.q.ptr, 0, nh * hd), dn16(g, b.blk.ptr, hd), dn16(g, l.ring.ptr, hd));
            }
            dv.sparse_attn_small(bl, b.q.ptr, l.ring.ptr, b.win_idx.ptr, w, b.blk.ptr, b.blk_idx.ptr, bl, l.sinks.ptr, b.o.ptr)?;
            dv.rope_rows(b.o.ptr, nh, hd, false, b.pos.ptr, bl, true)?;
            dv.fake_fp8(b.o.ptr, bl * nh * hd)?;
            let rw = c.o_groups * c.o_lora;
            for gi in 0..c.o_groups {
                dv.gemv_rows_ld(&l.o_a, gi * c.o_lora, c.o_lora, b.o.ptr + f4(gi * gsz) as u64, nh * hd, b.r.ptr + f4(gi * c.o_lora) as u64, rw, bl)?;
            }
            g.launch(dv.k["round_bf16"], (((bl * rw) as u32).div_ceil(256), 1, 1), 256, 0, &[Arg::P(b.r.ptr), Arg::I((bl * rw) as i32)])?;
            dv.gemv_t(&l.o_b, b.r.ptr, b.a.ptr, bl)?;
            g.launch(dv.k["hc_post"], (bl as u32, 1, 1), 256, 0, &[Arg::P(b.h.ptr), Arg::P(b.a.ptr), Arg::P(b.mix_a.ptr), Arg::I(d as i32)])?;
            pf.mark(g, "attention");
            if dbg() {
                let li = self.layers.iter().position(|x| std::ptr::eq(x, l)).unwrap();
                eprintln!("dspark-dbg gpu L{li}: attn out0 {:.4} h0 {:.4} h4 {:.4}", dn(g, b.a.ptr, 0, d), dn(g, b.h.ptr, 0, hc * d), dn(g, b.h.ptr, (bl - 1) * hc * d, hc * d));
            }
            // ffn half: input from the attention mixes' pre, this half's mixes for its post and the next layer's pre
            g.launch(dv.k["hc_pre"], (bl as u32, 1, 1), 256, 0, &[Arg::P(b.h.ptr), Arg::P(b.mix_a.ptr), Arg::I(0), Arg::P(b.x.ptr), Arg::I(d as i32)])?;
            dv.hc_mixes_rows(&l.hc_ffn, b.h.ptr, b.mix_f.ptr, bl)?;
            dv.rms_rows(b.x.ptr, &l.ffn_norm, b.x.ptr, d, bl, true)?;
            pf.mark(g, "ffn mixes");
            self.moe(&dv, l, bl, k, inter)?;
            pf.mark(g, "moe");
            if pf.on {
                picks.push((li, g.download(b.sel.ptr, bl * k)?));
            }
            g.launch(dv.k["hc_post"], (bl as u32, 1, 1), 256, 0, &[Arg::P(b.h.ptr), Arg::P(b.a.ptr), Arg::P(b.mix_f.ptr), Arg::I(d as i32)])?;
            if dbg() {
                let li = self.layers.iter().position(|x| std::ptr::eq(x, l)).unwrap();
                eprintln!("dspark-dbg gpu L{li}: layer out h0 {:.4} h4 {:.4}", dn(g, b.h.ptr, 0, hc * d), dn(g, b.h.ptr, (bl - 1) * hc * d, hc * d));
            }
        }
        // head: the main model's logits per row, then in order the Markov bias of the previous token and greedy
        g.launch(dv.k["hc_pre"], (bl as u32, 1, 1), 256, 0, &[Arg::P(b.h.ptr), Arg::P(b.mix_f.ptr), Arg::I(0), Arg::P(b.x.ptr), Arg::I(d as i32)])?;
        dv.rms_rows(b.x.ptr, &self.norm, b.x.ptr, d, bl, false)?;
        dv.gemv_t(dv.head, b.x.ptr, b.logits.ptr, bl)?;
        pf.mark(g, "head logits");
        // in order on the device: Markov embedding of the previous token, its bias over the vocabulary, greedy
        let rank = self.markov_embed.cols;
        let hv = dv.head.rows; // logits row stride (the head's rows; an export may pad the vocabulary)
        let nv = hv.min(self.markov_embed.rows).min(self.markov_head.rows);
        g.htod_sync(b.toks.ptr, &[next as i32])?;
        for i in 0..bl {
            let emb = b.embs.ptr + f4(i * rank) as u64;
            g.launch(dv.k["dsp_gather_bf16"], (1, 1, 1), 256, 0, &[Arg::P(b.mtab.ptr), Arg::P(b.toks.ptr + f4(i) as u64), Arg::I(rank as i32), Arg::P(emb)])?;
            dv.gemv(&self.markov_head, emb, b.bias.ptr)?;
            g.launch(dv.k["dsp_argmax_add"], (1, 1, 1), 1024, 0, &[Arg::P(b.logits.ptr + f4(i * hv) as u64), Arg::P(b.bias.ptr), Arg::I(nv as i32), Arg::P(b.toks.ptr + f4(i + 1) as u64)])?;
        }
        pf.mark(g, "markov chain");
        let toks: Vec<i32> = g.download(b.toks.ptr, bl + 1)?;
        let xs: Vec<f32> = g.download(b.x.ptr, bl * d)?;
        let embs: Vec<f32> = g.download(b.embs.ptr, bl * rank)?;
        let mut conf = vec![];
        for i in 0..bl {
            anyhow::ensure!(toks[i + 1] >= 0 && (toks[i + 1] as usize) < nv, "draft head picked token {} of {nv}", toks[i + 1]);
            let (x, emb) = (&xs[i * d..(i + 1) * d], &embs[i * rank..(i + 1) * rank]);
            conf.push(x.iter().chain(emb).zip(&self.conf).map(|(a, b)| a * b).sum());
            if dbg() {
                eprintln!("dspark-dbg gpu head {i}: x {:.4} best {}", nrm(x), toks[i + 1]);
            }
        }
        let out: Vec<u32> = toks.iter().map(|&x| x as u32).collect();
        pf.mark(g, "download+conf");
        if pf.on {
            if self.usage.is_empty() {
                self.usage = vec![vec![0u64; self.ne]; self.layers.len()];
            }
            for (li, s) in picks {
                for e in s {
                    if e >= 0 && (e as usize) < self.ne {
                        self.usage[li][e as usize] += 1;
                    }
                }
            }
            self.prof_n += 1;
            for (k, v) in pf.v {
                match self.prof.iter_mut().find(|(n, _)| *n == k) {
                    Some(x) => x.1 += v,
                    None => self.prof.push((k, v)),
                }
            }
        }
        self.since_swap += 1;
        self.drafts += 1;
        Ok((out[1..].to_vec(), conf))
    }

    /// Between passes (after a verify pass is kept, outside the draft's measured cost): every SWAP_EVERY drafts fold
    /// the expert pick counters into the heat and swap hotter experts in; now and then save the heat.
    pub fn maintain(&mut self, gm: &GpuModel) -> Result<()> {
        if self.since_swap < SWAP_EVERY {
            return Ok(());
        }
        self.since_swap = 0;
        self.update_cache(gm)?;
        self.updates += 1;
        if self.updates % 16 == 0 {
            self.save_heat();
        }
        Ok(())
    }

    fn heat_path(dir: &Path) -> Option<std::path::PathBuf> {
        let d = std::env::var_os("DSX_STATE_DIR").map(std::path::PathBuf::from).unwrap_or_else(|| dir.to_path_buf());
        Some(d.join("dspark-heat.json"))
    }

    /// The draft experts' decayed pick heat, so a restart begins with the experts this machine's use favoured
    pub fn save_heat(&self) {
        if let Some(p) = &self.heat_file {
            let tmp = p.with_extension("json.tmp");
            if std::fs::write(&tmp, serde_json::to_vec(&self.heat).unwrap_or_default()).is_ok() {
                let _ = std::fs::rename(&tmp, p);
            }
        }
    }

    /// Expert cache: fold the pick counters into decayed heat and bring in up to SWAP_MAX experts that are hotter than
    /// the coldest resident ones (from the checkpoint mapping on the host; between drafts).
    fn update_cache(&mut self, gm: &GpuModel) -> Result<()> {
        let _rb = Rebind(&gm.g);
        let g: &Gpu = self.side.as_ref().map_or(&gm.g, |s| &s.g);
        let ne = self.ne;
        let c: Vec<u32> = g.download(self.counts.ptr, self.layers.len() * ne)?;
        g.memset(self.counts.ptr, self.layers.len() * ne * 4)?;
        // warming up (first 128 drafts): faster
        let mut budget = if self.drafts < 128 { 4 * SWAP_MAX } else { SWAP_MAX };
        for (li, l) in self.layers.iter_mut().enumerate() {
            let heat = &mut self.heat[li];
            for e in 0..ne {
                heat[e] = 0.8 * heat[e] + c[li * ne + e] as f64;
            }
            if l.slots.len() >= ne {
                continue;
            }
            let mut want: Vec<usize> = (0..ne).filter(|&e| l.slot_of[e].is_none()).collect();
            want.sort_by(|&a, &b| heat[b].total_cmp(&heat[a]));
            for e in want {
                if budget == 0 {
                    break;
                }
                // the coldest resident
                let (si, &victim) = l.who.iter().enumerate().min_by(|a, b| heat[*a.1].total_cmp(&heat[*b.1])).unwrap();
                if heat[e] <= 1.25 * heat[victim] + 0.5 {
                    break;
                }
                for j in 0..3 {
                    let (m, s) = (&l.host[e][j], &l.slots[si][j]);
                    g.htod_sync(s.buf.ptr, m.bytes())?;
                    g.htod_sync(s.scale.as_ref().unwrap().ptr, m.scale_bytes())?;
                    l.wtab_h[j][e] = s.buf.ptr;
                    l.stab_h[j][e] = s.scale.as_ref().unwrap().ptr;
                    l.wtab_h[j][victim] = self._zero[j].buf.ptr;
                    l.stab_h[j][victim] = self._zero[j].scale.as_ref().unwrap().ptr;
                    g.htod_sync(l.wtab[j].ptr, &l.wtab_h[j])?;
                    g.htod_sync(l.stab[j].ptr, &l.stab_h[j])?;
                }
                l.slot_of[victim] = None;
                l.slot_of[e] = Some(si);
                l.who[si] = e;
                self.swaps += 1;
                budget -= 1;
            }
        }
        Ok(())
    }

    /// x [bl][d] (normed) -> a [bl][d]: routed experts (per row and pick) then the shared expert, bf16
    fn moe(&self, dv: &Dev, l: &DLayer, bl: usize, k: usize, inter: usize) -> Result<()> {
        let c = dv.c;
        let d = c.dim;
        let (g, b) = (dv.g, &self.b);
        dv.gemv_t(&l.gate_w, b.x.ptr, b.rl.ptr, bl)?;
        g.launch(dv.k["router_topk"], (bl as u32, 1, 1), 32, 0, &[Arg::P(b.rl.ptr), Arg::P(l.gate_bias.ptr), Arg::P(b.sel.ptr), Arg::P(b.wts.ptr), Arg::I(self.ne as i32), Arg::I(k as i32), Arg::F(c.route_scale), Arg::P(0)])?;
        let li = self.layers.iter().position(|x| std::ptr::eq(x, l)).unwrap_or(0);
        g.launch(dv.k["dsp_count"], (1, 1, 1), 64, 0, &[Arg::P(b.sel.ptr), Arg::I((bl * k) as i32), Arg::P(self.counts.ptr + (li * self.ne * 4) as u64), Arg::I(self.ne as i32)])?;
        // shared expert for every row (into sh_out) while the picks come back
        dv.gemv_t(&l.sh[0], b.x.ptr, b.eg.ptr, bl)?;
        dv.gemv_t(&l.sh[1], b.x.ptr, b.eu.ptr, bl)?;
        g.launch(dv.k["swiglu"], ((inter as u32).div_ceil(256), bl as u32, 1), 256, 0, &[Arg::P(b.eg.ptr), Arg::P(b.eu.ptr), Arg::I(inter as i32), Arg::P(0), Arg::P(b.eh.ptr), Arg::I(inter as i32), Arg::F(c.swiglu_limit)])?;
        dv.gemv_t(&l.sh[2], b.eh.ptr, b.sh_out.ptr, bl)?;
        // routed picks on the device: rows y = (token, pick); gate and up, SwiGLU with the pick's weight, down, sum
        let n = bl * k;
        let fq = |x: u64, out: u64, cnt: usize| g.launch(dv.k["fake_fp8_copy"], (((cnt / 32) as u32).div_ceil(8), 1, 1), 256, 0, &[Arg::P(x), Arg::P(out), Arg::I(cnt as i32)]);
        fq(b.x.ptr, b.xq.ptr, bl * d)?;
        let sel = |j: usize, rows: usize, kk: usize, x: u64, xdiv: usize, y: u64| {
            g.launch(dv.k["gemv_fp4_sel"], ((rows as u32).div_ceil(16), n as u32, 1), 128, 0, &[Arg::P(l.wtab[j].ptr), Arg::P(l.stab[j].ptr), Arg::P(b.sel.ptr), Arg::I(rows as i32), Arg::I(kk as i32), Arg::P(x), Arg::I(xdiv as i32), Arg::P(y)])
        };
        sel(0, inter, d, b.xq.ptr, k, b.eg.ptr)?;
        sel(1, inter, d, b.xq.ptr, k, b.eu.ptr)?;
        g.launch(dv.k["swiglu"], ((inter as u32).div_ceil(256), n as u32, 1), 256, 0, &[Arg::P(b.eg.ptr), Arg::P(b.eu.ptr), Arg::I(inter as i32), Arg::P(b.wts.ptr), Arg::P(b.eh.ptr), Arg::I(inter as i32), Arg::F(c.swiglu_limit)])?;
        fq(b.eh.ptr, b.ehq.ptr, n * inter)?;
        sel(2, d, inter, b.ehq.ptr, 1, b.ey.ptr)?;
        g.launch(dv.k["dsp_moe_sum"], (bl as u32, 1, 1), 256, 0, &[Arg::P(b.a.ptr), Arg::P(b.ey.ptr), Arg::P(b.sh_out.ptr), Arg::I(d as i32), Arg::I(k as i32)])
    }
}
