//! Prefill: a chunk of prompt tokens through the encoder half (layers 0..=20) for every token, and through the
//! decoder half only for the last `tail` tokens of the prompt. Decoder layers keep just a 128-token window of their
//! own; their long-range KV and index keys come from layer 20 (model.py kv_source_layers / SWA bounded replay), so
//! skipping the decoder for older prompt tokens gives the same caches as decoding token by token.
//!
//! Dense weights are dequantized to f16 per layer and multiplied with cuBLAS (f32 accumulation). Routed experts
//! run as one grouped GEMM per expert over the tokens routed to it; experts resident on no GPU are copied in from
//! the RAM arena (hugepage, full PCIe speed) or the mmapped file, split between the main GPU and the aux GPUs whose
//! links are as fast (`AuxPrefill::stream`) in proportion to their link bandwidth.

use super::*;
use crate::gguf::GType;

/// prefill weight expansion to f16: the 8-wide kernel for the IQ codebook types (GGUF blocks or repacked), else
/// dequant_f16; same arguments
fn dq_iq8(ty: i32, cols: usize) -> bool {
    matches!(ty, 16 | 17 | 19 | 22 | 2016 | 2017 | 2019 | 2022) && cols % 256 == 0
}
pub(crate) fn dq_kernel(k: &std::collections::HashMap<&'static str, crate::gpu::Kernel>, ty: i32, cols: usize) -> crate::gpu::Kernel {
    k[if dq_iq8(ty, cols) { "dequant_f16_iq8" } else { "dequant_f16" }]
}
/// T16 experts in prefill: y = had(had(x * suh) @ W_inner) * svh, done on the activations around one GEMM.
/// `x16` [nt][k] f16 (left as is unless `tx16` == `x16`), `tx16` scratch [nt][k], `y` [nt][n] f32.
#[allow(clippy::too_many_arguments)]
pub(crate) fn t16_linear(g: &crate::gpu::Gpu, k: &std::collections::HashMap<&'static str, crate::gpu::Kernel>, ty: i32, w: u64, x16: u64, tx16: u64, w16: u64, y: u64, nt: usize, kin: usize, nout: usize, post: bool) -> Result<()> {
    let (gt, repacked) = if ty >= 2000 { (GType::from_id(ty as u32 - 2000)?, 1) } else { (GType::from_id(ty as u32)?, 0) };
    let k2 = crate::t16::k2(gt).unwrap();
    let tb = crate::t16::trellis_bytes(k2, kin, nout) as u64;
    if t16_warp_had() {
        g.launch(k["t16_pre_warp"], (((kin / 128 * nt) as u32).div_ceil(4), 1, 1), 128, 0, &[Arg::P(x16), Arg::P(tx16), Arg::P(w + tb), Arg::I(kin as i32), Arg::I(nt as i32)])?;
    } else {
        g.launch(k["t16_pre"], ((kin / 128) as u32, nt as u32, 1), 128, 0, &[Arg::P(x16), Arg::P(tx16), Arg::P(w + tb), Arg::I(kin as i32)])?;
    }
    g.launch(k[if t16_specialized_dq() { "dequant_t16_f16_specialized" } else { "dequant_t16_f16" }], ((kin as u32).div_ceil(512), (nout / 16) as u32, 1), 512, 0, &[Arg::I(k2 as i32), Arg::P(w), Arg::P(w16), Arg::I(nout as i32), Arg::I(kin as i32), Arg::I(repacked)])?;
    g.gemm_f16(tx16, w16, y, nt, kin, nout, nout)?;
    if post && t16_warp_had() {
        g.launch(k["t16_post_warp"], (((nout / 128 * nt) as u32).div_ceil(4), 1, 1), 128, 0, &[Arg::P(y), Arg::P(w + tb + 2 * kin as u64), Arg::I(nout as i32), Arg::F(t16_gain(nout < kin, k2)), Arg::I(nt as i32)])?;
    } else if post {
        g.launch(k["t16_post"], ((nout / 128) as u32, nt as u32, 1), 128, 0, &[Arg::P(y), Arg::P(w + tb + 2 * kin as u64), Arg::I(nout as i32), Arg::F(t16_gain(nout < kin, k2))])?;
    }
    Ok(())
}

fn t16_warp_had() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("DSX_T16_WARP_HAD").is_ok_and(|v| v == "1"))
}
fn t16_specialized_dq() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("DSX_T16_SPECIALIZED_DQ").is_ok_and(|v| v == "1"))
}
fn t16_fused_post_gu() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("DSX_T16_FUSED_POST_GU").is_ok_and(|v| v == "1"))
}
/// Preserve diagnostic builds using an archived kernel image when new experiments are disabled.
pub(crate) fn load_optional_kernels(g: &Gpu, k: &mut std::collections::HashMap<&'static str, crate::gpu::Kernel>) -> Result<()> {
    if t16_warp_had() {
        for name in ["t16_pre_warp", "t16_post_warp"] { k.insert(name, g.kernel(name)?); }
    }
    if t16_fused_post_gu() { k.insert("t16_post_gu_warp", g.kernel("t16_post_gu_warp")?); }
    if t16_specialized_dq() { k.insert("dequant_t16_f16_specialized", g.kernel("dequant_t16_f16_specialized")?); }
    Ok(())
}
/// experiment: extra output gain on T16 matrices in prefill, per role and bits x 2, e.g. DSX_T16_GAIN=gu4=1.06,dn5=1.02
/// (gu = gate and up, dn = down; k2 3 / 4 / 5 = 1.5 / 2 / 2.5 bits). Default 1: the shipped gain is baked into svh.
fn t16_gain(gate_up: bool, k2: usize) -> f32 {
    static G: std::sync::OnceLock<Vec<(bool, usize, f32)>> = std::sync::OnceLock::new();
    let g = G.get_or_init(|| {
        std::env::var("DSX_T16_GAIN").unwrap_or_default().split(',').filter_map(|it| {
            let (rk, v) = it.split_once('=')?;
            let gu = rk.starts_with("gu");
            Some((gu, rk[2..].parse().ok()?, v.parse().ok()?))
        }).collect()
    });
    g.iter().find(|e| e.0 == gate_up && e.1 == k2).map_or(1.0, |e| e.2)
}

/// T16 ids: 1010-1012 stored layout (streamed from RAM or the file), 3010-3012 repacked (GPU store, 2000 + id)
pub(crate) fn is_t16(ty: i32) -> bool {
    (1010..=1012).contains(&ty) || (3010..=3012).contains(&ty)
}

pub(crate) fn dq_grid(ty: i32, rows: usize, cols: usize) -> (u32, u32, u32) {
    let per = if dq_iq8(ty, cols) { 8 } else { 1 };
    (((cols / per) as u32).div_ceil(256), rows as u32, 1)
}

/// staging buffers for streamed experts (copies run ahead of compute by up to NSTAGE - 1 experts)
const NSTAGE: usize = 4;

/// one staging buffer: any expert's raw gate + up + down (at most 8.5 bits per weight)
fn stage_bytes(c: &crate::model::Config) -> usize {
    3 * c.inter * c.dim / 32 * 34 + 4096
}

/// VRAM an aux GPU's prefill buffers take (placement leaves this free); `stream` adds the expert staging buffers
pub(crate) fn aux_prefill_bytes(c: &crate::model::Config, tmax: usize, stream: bool) -> usize {
    tmax * (c.dim * 16 + c.inter * 10 + 64) + c.inter * c.dim * 2 + (256 << 20) + if stream { NSTAGE * stage_bytes(c) } else { 0 }
}

pub struct PrefillBufs {
    pub tmax: usize,
    h: Buf,
    x: Buf,
    a: Buf,
    qr: Buf,
    q: Buf,
    kv: Buf,
    o: Option<Buf>,
    r: Buf,
    mix_a: Buf,
    mix_f: Buf,
    act16: Buf,
    w16: Buf,
    wins: Buf,
    winidx: Buf,
    pos: Buf,
    lat: Buf,
    ckv: Buf,
    csc: Buf,
    iq: Buf,
    iw: Buf,
    score: Buf,
    ns: Buf,
    compidx: Buf,
    keep: Buf,
    keep_tail: Buf,
    tmp: Option<Buf>,
    // experts
    gx16: Buf,
    /// T16: the gathered rows times one matrix's suh, Hadamard-transformed
    tx16: Buf,
    gg: Buf,
    gu: Buf,
    h16: Buf,
    go: Buf,
    rows: Buf,
    wrow: Buf,
    estage: [Buf; NSTAGE],
    ev_ready: [crate::gpu::Event; NSTAGE],
    ev_free: [crate::gpu::Event; NSTAGE],
    /// x shipped to the aux GPUs (prefill_moe_aux): the copy's event and pinned f16 staging
    ev_x: crate::gpu::Event,
    x_host: crate::gpu::HostBuf,
    /// pinned staging for small per-layer uploads (positions, index rows), bump-allocated within a pass
    stage: crate::gpu::HostBuf,
    stage_off: std::cell::Cell<usize>,
    out: Buf,
    logits: Buf,
    attn_tc: crate::attn::SparseAttnTc,
}

/// An aux GPU's share of prefill: its resident experts run there on activations shipped over its link
/// (moving ~42 MB of activations per layer beats moving its ~0.6 GB of weights per layer over an x4 link). An aux GPU
/// with a link as fast as the main GPU's also takes a share of the streamed experts (its own staging buffers, copies
/// on its side stream), so both links carry the arena.
pub struct AuxPrefill {
    x16: Buf,
    rows: Buf,
    wrow: Buf,
    gx16: Buf,
    /// T16: the gathered rows times one matrix's suh, Hadamard-transformed
    tx16: Buf,
    gg: Buf,
    gu: Buf,
    h16: Buf,
    go: Buf,
    w16: Buf,
    out: Buf,
    out_host: crate::gpu::HostBuf,
    /// pinned copies of the layer's token rows and weights (uploaded without a host sync)
    idx_host: crate::gpu::HostBuf,
    /// streamed-expert staging and its events (empty when this GPU takes no share of the stream)
    estage: Vec<Buf>,
    ev_ready: Vec<crate::gpu::Event>,
    ev_free: Vec<crate::gpu::Event>,
    /// host -> device GB/s of this GPU's link: its share of the streamed experts
    pub link_gbs: f64,
    k: std::collections::HashMap<&'static str, crate::gpu::Kernel>,
}

impl AuxPrefill {
    /// `stream`: take a share of the streamed experts (link_gbs > 0 weighs it against the main GPU's link)
    pub fn new(g1: &Gpu, m: &Model, tmax: usize, stream: bool, link_gbs: f64) -> Result<AuxPrefill> {
        let c = &m.cfg;
        let (d, inter) = (c.dim, c.inter);
        let mut k = std::collections::HashMap::new();
        for n in ["gather_h", "dequant_f16", "dequant_f16_iq8", "swiglu_rows_f16", "scatter_add", "fwht_rows_f16", "add_noise_rows", "dequant_t16_f16", "t16_pre", "t16_post"] {
            k.insert(n, g1.kernel(n)?);
        }
        load_optional_kernels(g1, &mut k)?;
        Ok(AuxPrefill {
            x16: g1.alloc(tmax * d * 2)?,
            rows: g1.alloc(tmax * 8 * 4)?,
            wrow: g1.alloc(tmax * 8 * 4)?,
            gx16: g1.alloc(tmax * d * 2)?,
            tx16: g1.alloc(tmax * d * 2)?,
            gg: g1.alloc(tmax * inter * 4)?,
            gu: g1.alloc(tmax * inter * 4)?,
            h16: g1.alloc(tmax * inter * 2)?,
            go: g1.alloc(tmax * d * 4)?,
            w16: g1.alloc(inter * d * 2)?,
            out: g1.alloc(tmax * d * 4)?,
            out_host: crate::gpu::HostBuf::new(tmax * d * 4)?,
            idx_host: crate::gpu::HostBuf::new(tmax * 8 * 4 * 2)?,
            estage: if stream { (0..NSTAGE).map(|_| g1.alloc(stage_bytes(c))).collect::<Result<_>>()? } else { vec![] },
            ev_ready: if stream { (0..NSTAGE).map(|_| g1.event()).collect::<Result<_>>()? } else { vec![] },
            ev_free: if stream { (0..NSTAGE).map(|_| g1.event()).collect::<Result<_>>()? } else { vec![] },
            link_gbs,
            k,
        })
    }
}

/// One GPU's routed-expert work for a prefill layer: its buffers, store and staging (queue_expert enqueues one expert)
struct ExpQueue<'a> {
    g: &'a Gpu,
    k: &'a std::collections::HashMap<&'static str, crate::gpu::Kernel>,
    /// gathers the expert's token rows of x into gx16 (f32 x on the main GPU, f16 x16 on an aux GPU)
    gather: &'static str,
    x: u64,
    rows: u64,
    wrow: u64,
    gx16: u64,
    tx16: u64,
    gg: u64,
    gu: u64,
    h16: u64,
    go: u64,
    w16: u64,
    out: u64,
    store: Option<&'a crate::experts::GpuExperts>,
    estage: &'a [Buf],
    ev_ready: &'a [crate::gpu::Event],
    ev_free: &'a [crate::gpu::Event],
    nstaged: usize,
}

impl ExpQueue<'_> {
    /// the side stream must not overwrite a staging buffer before the main stream reaches this layer's experts
    fn start(&self) -> Result<()> {
        for e in self.ev_free {
            self.g.record(*e, false)?;
        }
        Ok(())
    }
}

/// DSX_NOISE=A-B:REL (sensitivity probe, prefill path): every routed expert output of layers A..=B gets Gaussian noise
/// of REL x its per-token RMS before it is added
fn noise_rel(li: usize) -> Option<f32> {
    static N: std::sync::OnceLock<Option<(usize, usize, f32)>> = std::sync::OnceLock::new();
    let n = N.get_or_init(|| {
        let v = std::env::var("DSX_NOISE").ok()?;
        let (r, rel) = v.split_once(':')?;
        let (a, b) = r.split_once('-').unwrap_or((r, r));
        Some((a.parse().ok()?, b.parse().ok()?, rel.parse().ok()?))
    });
    n.and_then(|(a, b, rel)| (li >= a && li <= b).then_some(rel))
}

/// DSX_STREAM_MAIN_W=x: the main GPU's link counts x times its own speed when streamed experts are split between it
/// and the streaming aux GPUs. Default 2: an aux GPU also computes its resident experts (twice the main GPU's on 2x
/// RTX 3090), and on PCIe 3.0 x16 links an expert's compute costs about as much as its copy; best of 1-3 on an 8K
/// prompt (warm prefill 15.6 s streaming on the main GPU alone, 14.4-15.4 s at 1, 12.9-13.3 s at 2, 13.5 s at 3)
/// chunks longer than this (DSX_PREFILL_CPU_MAX_N, default 512) stream every expert: on .102 the CPU share helped at
/// 60-300 tokens (server turns 1137 -> 738 ms, 2227 -> 1698 ms), was 2% slower at 1024 and 5-20% slower at 2048-3000
/// (few experts have few tokens there, the x download stalls the stream, and the CPU competes for DRAM with the link)
fn prefill_cpu_max_n() -> usize {
    static P: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *P.get_or_init(|| std::env::var("DSX_PREFILL_CPU_MAX_N").ok().and_then(|v| v.parse().ok()).unwrap_or(512))
}

/// DSX_PREFILL_CPU=0: prefill streams every expert no GPU holds (no CPU share)
fn prefill_cpu() -> bool {
    static P: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *P.get_or_init(|| std::env::var("DSX_PREFILL_CPU").map_or(true, |v| v != "0"))
}

/// DSX_PRE_CPU_US: the CPU's estimated time for one expert and one token in prefill (default 200: 1.2 ms per 6-expert
/// token at 14 threads on .102); more tokens scale by pre_cpu_cost
fn pre_cpu_us() -> f64 {
    static P: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *P.get_or_init(|| std::env::var("DSX_PRE_CPU_US").ok().and_then(|v| v.parse().ok()).unwrap_or(200.0))
}

/// An expert's CPU cost for nt tokens in units of one token's: the multi-token kernel decodes each tile once per up to
/// 8 tokens (t16k::gemv_multi; cpubench t16 at 14 threads on .102: x1.18 at 2 tokens, x1.67 at 4, x2.66 at 8, x5.29
/// at 16), linear in between and beyond
fn pre_cpu_cost(nt: usize) -> f64 {
    const P: [(f64, f64); 5] = [(1.0, 1.0), (2.0, 1.18), (4.0, 1.67), (8.0, 2.66), (16.0, 5.29)];
    let x = nt as f64;
    for w in P.windows(2) {
        if x <= w[1].0 {
            return w[0].1 + (x - w[0].0) / (w[1].0 - w[0].0) * (w[1].1 - w[0].1);
        }
    }
    P[4].1 + (x - 16.0) * (P[4].1 - P[3].1) / 8.0
}

/// tokens per pass of the decode kernels over prefill's few-token resident experts
const PRE_DEV_TB: usize = 64;

/// DSX_PRE_DEV_NT: resident experts with at most this many tokens in a prefill chunk run on decode's T16 kernels (0 =
/// off, the default: on .102 at 60-1024 tokens it was no faster, the dequant overlapping the link's copies, and the
/// int8 activations cost KL 0.1965 -> 0.2010). At 6: dequant + GEMM costs ~110 us per expert (three f16 copies of the weights at ~640 GB/s) plus ~10
/// launches; the decode kernels read the expert's ~7.4 MB once per token, ~15 us each
fn pre_dev_nt() -> usize {
    static P: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *P.get_or_init(|| std::env::var("DSX_PRE_DEV_NT").ok().and_then(|v| v.parse().ok()).unwrap_or(0))
}

fn stream_main_weight() -> f64 {
    static W: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *W.get_or_init(|| std::env::var("DSX_STREAM_MAIN_W").ok().and_then(|v| v.parse().ok()).unwrap_or(2.0))
}

pub(crate) const SUB: usize = 256; // indexer queries per pass (bounds the [queries][positions] score matrix)

impl PrefillBufs {
    fn attn_out(&self) -> u64 { self.o.as_ref().map_or(self.q.ptr, |b| b.ptr) }
    fn tmp_ptr(&self) -> u64 { self.tmp.as_ref().map_or(self.q.ptr, |b| b.ptr) }
    fn tmp_bytes(&self) -> usize { self.tmp.as_ref().map_or(self.q.bytes, |b| b.bytes) }

    pub fn new(g: &Gpu, m: &Model, tmax: usize, max_ctx: usize) -> Result<PrefillBufs> {
        let c = &m.cfg;
        let tmax = tmax.max(1);
        let f = |n: usize| g.alloc(n * 4);
        let hq = c.n_heads * c.head_dim;
        let max_in = hq.max(c.hc * c.dim).max(6144);
        let max_w = (32768 * 1280).max(c.dim * 8192).max(25600 * 6144).max(c.n_experts * c.dim);
        let nb = max_ctx / c.cand_block + 1;
        let reuse = std::env::var("DSX_PREFILL_REUSE_BUFFERS").is_ok_and(|v| v == "1");
        Ok(PrefillBufs {
            tmax,
            h: f(tmax * c.hc * c.dim)?,
            x: f(tmax * c.dim)?,
            a: f(tmax * c.dim)?,
            qr: f(tmax * c.q_lora)?,
            q: f(tmax * if reuse { hq.max(25600.max(c.n_experts)) } else { hq })?,
            kv: f(tmax * c.head_dim)?,
            // Attention CTAs own disjoint token/head slices and load their Q into registers
            // before writing O. Split-K small attention writes O only in its later merge kernel.
            o: if reuse { None } else { Some(f(tmax * hq)?) },
            r: f(tmax * c.o_groups * c.o_lora)?,
            mix_a: f(tmax * 24)?,
            mix_f: f(tmax * 24)?,
            act16: g.alloc(tmax * max_in * 2)?,
            w16: g.alloc(max_w * 2)?,
            wins: g.alloc((c.window + tmax) * c.head_dim * 2)?,
            winidx: g.alloc(tmax * c.window * 4)?,
            pos: g.alloc(tmax * 4)?,
            lat: f(tmax * c.head_dim)?,
            ckv: f(tmax * c.head_dim)?,
            csc: f(tmax * c.head_dim)?,
            iq: f(tmax * c.idx_heads * c.idx_dim)?,
            iw: f(tmax * c.idx_heads)?,
            score: f(SUB * max_ctx)?,
            ns: g.alloc(tmax * 4)?,
            compidx: g.alloc(tmax * c.idx_topk * 4)?,
            keep: g.alloc(SUB * nb)?,
            keep_tail: g.alloc((c.n_layers - 1 - c.cand_layer) * c.window * nb)?,
            // Engram/router scratch is dead before attention Q is projected.
            tmp: if reuse { None } else { Some(f(tmax * 25600.max(c.n_experts))?) },
            gx16: g.alloc(tmax * c.dim * 2)?,
            tx16: g.alloc(tmax * c.dim * 2)?,
            gg: f(tmax * c.inter)?,
            gu: f(tmax * c.inter)?,
            h16: g.alloc(tmax * c.inter * 2)?,
            go: f(tmax * c.dim)?,
            rows: g.alloc(tmax * 8 * 4)?,
            wrow: f(tmax * 8)?,
            estage: (0..NSTAGE).map(|_| g.alloc(stage_bytes(c))).collect::<Result<Vec<_>>>()?.try_into().map_err(|_| anyhow::anyhow!("estage"))?,
            ev_ready: (0..NSTAGE).map(|_| g.event()).collect::<Result<Vec<_>>>()?.try_into().map_err(|_| anyhow::anyhow!("ev_ready"))?,
            ev_free: (0..NSTAGE).map(|_| g.event()).collect::<Result<Vec<_>>>()?.try_into().map_err(|_| anyhow::anyhow!("ev_free"))?,
            ev_x: g.event()?,
            x_host: crate::gpu::HostBuf::new(tmax * c.dim * 2)?,
            stage: crate::gpu::HostBuf::new(8 << 20)?,
            stage_off: std::cell::Cell::new(0),
            out: f(tmax * c.dim)?,
            logits: f(crate::handoff::MAXT * c.vocab)?, // [rows][vocab]: a verify pass's logits
            attn_tc: crate::attn::SparseAttnTc::new(g)?,
        })
    }
}

fn type_id(t: GType) -> i32 {
    match t {
        GType::Fp8B32 => 1000,
        t => t as u32 as i32,
    }
}

impl GpuModel<'_> {
    /// Ship this layer's expert input toward the aux GPUs: f16 on the main GPU, copied into pinned host memory. Only
    /// queued here; prefill_moe_aux waits for the copy's event once the main GPU's shared expert is queued behind it.
    fn prefill_ship_x(&self, pb: &PrefillBufs, x: u64, n: usize) -> Result<()> {
        let d = self.m.cfg.dim;
        self.g.launch(self.k["to_f16"], (((n * d) as u32).div_ceil(256), 1, 1), 256, 0, &[Arg::P(x), Arg::P(pb.act16.ptr), Arg::I((n * d) as i32)])?;
        let xh = unsafe { std::slice::from_raw_parts_mut(pb.x_host.host_ptr() as *mut u16, n * d) };
        self.g.dtoh(xh, pb.act16.ptr)?;
        self.g.record(pb.ev_x, false)
    }

    /// Copy expert (li, e)'s raw weights (the arena's copy, else the file's) into staging buffer `base` on `g`'s side
    /// stream once `free` (recorded by the main stream) has passed; the main stream waits for the copy. Returns the
    /// expert's (gate, up, down) device pointers and (gate, down) type ids.
    #[allow(clippy::too_many_arguments)]
    fn stage_expert(&self, g: &Gpu, li: usize, e: usize, base: u64, free: crate::gpu::Event, ready: crate::gpu::Event) -> Result<(u64, u64, u64, i32, i32)> {
        let ex = &self.m.layers[li].experts[e];
        let (gb, ub, db) = match self.arena.and_then(|a| a.get(li, e)) {
            Some([g, u, dn]) => (g, u, dn),
            None => (ex.gate.bytes(), ex.up.bytes(), ex.down.bytes()),
        };
        g.wait(free, true)?;
        g.htod_side(base, gb)?;
        g.htod_side(base + gb.len() as u64, ub)?;
        g.htod_side(base + (gb.len() + ub.len()) as u64, db)?;
        g.record(ready, true)?;
        g.wait(ready, false)?;
        Ok((base, base + gb.len() as u64, base + (gb.len() + ub.len()) as u64, type_id(ex.gate.ty), type_id(ex.down.ty)))
    }

    /// Prepare the aux GPUs for this layer's experts: x (shipped by prefill_ship_x), the token rows and weights, a
    /// zeroed sum. Returns one queue per aux GPU with work (`spans[i]`: its resident experts, then the streamed ones it
    /// was given, so the first copies overlap the resident experts' compute).
    fn prefill_aux_begin<'a>(&'a self, pb: &PrefillBufs, li: usize, n: usize, spans: &[Vec<(usize, usize, usize)>], all_rows: &[i32], all_w: &[f32]) -> Result<Vec<(usize, ExpQueue<'a>, Vec<(usize, usize, usize)>)>> {
        let d = self.m.cfg.dim;
        // the host waits for the x copy only (an event), not for the rest of GPU0's queue: a full sync here stalled
        // every layer behind the shared expert while no weights were copying
        self.g.event_sync(pb.ev_x)?;
        let xh = unsafe { std::slice::from_raw_parts(pb.x_host.host_ptr() as *const u16, n * d) };
        let nr = all_rows.len();
        let mut qs = vec![];
        for (i, (a, spans)) in self.aux.iter().zip(spans).enumerate() {
            if spans.is_empty() {
                continue;
            }
            let (g1, gp, s1) = (&a.g, a.pre.as_ref().unwrap(), &a.exp);
            // token rows and weights through pinned memory, so the uploads queue without a host sync (the previous
            // layer's copies are done: prefill_moe synced this GPU before adding its sum)
            let (rows_h, w_h) = unsafe {
                let p = gp.idx_host.host_ptr();
                std::ptr::copy_nonoverlapping(all_rows.as_ptr() as *const u8, p, nr * 4);
                std::ptr::copy_nonoverlapping(all_w.as_ptr() as *const u8, p.add(nr * 4), nr * 4);
                (std::slice::from_raw_parts(p as *const i32, nr), std::slice::from_raw_parts(p.add(nr * 4) as *const f32, nr))
            };
            g1.htod(gp.x16.ptr, xh)?;
            g1.htod(gp.rows.ptr, rows_h)?;
            g1.htod(gp.wrow.ptr, w_h)?;
            g1.memset(gp.out.ptr, n * d * 4)?;
            let (mut list, streamed): (Vec<_>, Vec<_>) = spans.iter().partition(|s| s1.slot_ptrs(li, s.0).is_some());
            anyhow::ensure!(streamed.is_empty() || !gp.estage.is_empty(), "streamed experts given to an aux GPU without staging buffers");
            list.extend(streamed);
            let q = ExpQueue { g: g1, k: &gp.k, gather: "gather_h", x: gp.x16.ptr, rows: gp.rows.ptr, wrow: gp.wrow.ptr, gx16: gp.gx16.ptr, tx16: gp.tx16.ptr, gg: gp.gg.ptr, gu: gp.gu.ptr, h16: gp.h16.ptr, go: gp.go.ptr, w16: gp.w16.ptr, out: gp.out.ptr, store: Some(s1), estage: &gp.estage, ev_ready: &gp.ev_ready, ev_free: &gp.ev_free, nstaged: 0 };
            q.start()?;
            qs.push((i, q, list.into_iter().copied().collect()));
        }
        self.g.bind();
        Ok(qs)
    }

    /// Enqueue one routed expert on `q`'s GPU over its `nt` token rows starting at `off`: weights from the GPU's
    /// store, else staged from the arena; the weighted output is added into q.out.
    fn queue_expert(&self, q: &mut ExpQueue, li: usize, e: usize, off: usize, nt: usize) -> Result<()> {
        let c = &self.m.cfg;
        let (d, inter) = (c.dim, c.inter);
        let rot = c.expert_rot[li];
        let g = q.g;
        let rows_p = q.rows + (off * 4) as u64;
        let w_p = q.wrow + (off * 4) as u64;
        g.launch(q.k[q.gather], ((d as u32).div_ceil(256), nt as u32, 1), 256, 0, &[Arg::P(q.x), Arg::P(rows_p), Arg::P(q.gx16), Arg::I(d as i32)])?;
        if rot {
            // rotated experts store W.R: rotate the activations (R x, R h), not the weights
            g.launch(q.k["fwht_rows_f16"], ((d / 128) as u32, nt as u32, 1), 128, 0, &[Arg::P(q.gx16), Arg::I(d as i32)])?;
        }
        // expert weights on the device: the GPU's store, else copied on its side stream into one of the staging
        // buffers while the main stream computes the previous experts
        let mut staged = None;
        let (gp, up, dp, gt, dt) = match q.store.and_then(|s| s.slot_ptrs(li, e)) {
            Some(p) => p,
            None => {
                let b = q.nstaged % NSTAGE;
                q.nstaged += 1;
                staged = Some(b);
                self.stage_expert(g, li, e, q.estage[b].ptr, q.ev_free[b], q.ev_ready[b])?
            }
        };
        let fused_post = is_t16(gt) && t16_fused_post_gu();
        for (wp, yp) in [(gp, q.gg), (up, q.gu)] {
            if is_t16(gt) {
                t16_linear(g, q.k, gt, wp, q.gx16, q.tx16, q.w16, yp, nt, d, inter, !fused_post)?;
                continue;
            }
            g.launch(dq_kernel(q.k, gt, d), dq_grid(gt, inter, d), 256, 0, &[Arg::I(gt), Arg::P(wp), Arg::P(0), Arg::P(q.w16), Arg::I(inter as i32), Arg::I(d as i32)])?;
            g.gemm_f16(q.gx16, q.w16, yp, nt, d, inter, inter)?;
        }
        if fused_post {
            let ty = if gt >= 2000 { gt - 2000 } else { gt };
            let k2 = crate::t16::k2(GType::from_id(ty as u32)?).unwrap();
            let scale = crate::t16::trellis_bytes(k2, d, inter) as u64 + 2 * d as u64;
            g.launch(q.k["t16_post_gu_warp"], (((inter / 128 * nt) as u32).div_ceil(4), 1, 1), 128, 0, &[Arg::P(q.gg), Arg::P(q.gu), Arg::P(gp + scale), Arg::P(up + scale), Arg::P(w_p), Arg::P(q.h16), Arg::I(inter as i32), Arg::I(nt as i32), Arg::F(t16_gain(inter < d, k2)), Arg::F(c.swiglu_limit)])?;
        } else {
        g.launch(q.k["swiglu_rows_f16"], ((inter as u32).div_ceil(256), nt as u32, 1), 256, 0, &[Arg::P(q.gg), Arg::P(q.gu), Arg::P(w_p), Arg::P(q.h16), Arg::I(inter as i32), Arg::F(c.swiglu_limit)])?;
        }
        if rot {
            g.launch(q.k["fwht_rows_f16"], ((inter / 128) as u32, nt as u32, 1), 128, 0, &[Arg::P(q.h16), Arg::I(inter as i32)])?;
        }
        if is_t16(dt) {
            t16_linear(g, q.k, dt, dp, q.h16, q.h16, q.w16, q.go, nt, inter, d, true)?;
        } else {
            g.launch(dq_kernel(q.k, dt, inter), dq_grid(dt, d, inter), 256, 0, &[Arg::I(dt), Arg::P(dp), Arg::P(0), Arg::P(q.w16), Arg::I(d as i32), Arg::I(inter as i32)])?;
            g.gemm_f16(q.h16, q.w16, q.go, nt, inter, d, d)?;
        }
        if let Some(rel) = noise_rel(li) {
            g.launch(q.k["add_noise_rows"], (nt as u32, 1, 1), 256, 0, &[Arg::P(q.go), Arg::I(d as i32), Arg::F(rel), Arg::I((li * 1000 + e + 1) as i32)])?;
        }
        g.launch(q.k["scatter_add"], ((d as u32).div_ceil(256), nt as u32, 1), 256, 0, &[Arg::P(q.go), Arg::P(rows_p), Arg::P(q.out), Arg::I(d as i32)])?;
        if let Some(b) = staged {
            g.record(q.ev_free[b], false)?;
        }
        Ok(())
    }

    /// y[n][w.rows] = x[n][w.cols] . W^T through f16 tensor cores (W dequantized into pb.w16 first)
    /// upload `src` to `dst` in stream order without blocking the host: copied into the pass's pinned staging first
    /// (blocking upload when the staging is full)
    fn up<T: Copy>(&self, pb: &PrefillBufs, dst: u64, src: &[T]) -> Result<()> {
        let bytes = std::mem::size_of_val(src);
        let off = pb.stage_off.get();
        if off + bytes > pb.stage.bytes {
            return self.g.htod_sync(dst, src);
        }
        let host = unsafe {
            let p = pb.stage.host_ptr().add(off);
            std::ptr::copy_nonoverlapping(src.as_ptr() as *const u8, p, bytes);
            std::slice::from_raw_parts(p as *const T, src.len())
        };
        pb.stage_off.set((off + bytes).div_ceil(256) * 256);
        self.g.htod(dst, host)
    }

    fn gemm(&self, pb: &PrefillBufs, w: &DMat, x: u64, y: u64, n: usize) -> Result<()> {
        if n <= crate::handoff::MAXT {
            // a few rows (verify passes, tiny chunks): decode's GEMV with n columns reads the weight once, no f16 copy
            return self.gemv_t(w, x, y, n);
        }
        if w.q6.is_some() {
            return self.gemm_q6(pb, w, 0, w.rows, w.cols, x, w.cols, y, w.rows, n);
        }
        let (wp, sp, id) = w.ptrs(0);
        self.gemm_raw(pb, id, wp, sp, w.rows, w.cols, w.act_fp8, x, y, n)
    }

    #[allow(clippy::too_many_arguments)]
    fn gemm_raw(&self, pb: &PrefillBufs, ty: i32, wp: u64, sp: u64, rows: usize, cols: usize, act_fp8: bool, x: u64, y: u64, n: usize) -> Result<()> {
        let g = &self.g;
        let nx = n * cols;
        let mut xin = x;
        if act_fp8 {
            g.launch(self.k["fake_fp8_copy"], ((nx as u32 / 32).div_ceil(8), 1, 1), 256, 0, &[Arg::P(x), Arg::P(pb.tmp_ptr()), Arg::I(nx as i32)])?;
            xin = pb.tmp_ptr();
        }
        g.launch(self.k["to_f16"], ((nx as u32).div_ceil(256), 1, 1), 256, 0, &[Arg::P(xin), Arg::P(pb.act16.ptr), Arg::I(nx as i32)])?;
        g.launch(dq_kernel(&self.k, ty, cols), dq_grid(ty, rows, cols), 256, 0, &[Arg::I(ty), Arg::P(wp), Arg::P(sp), Arg::P(pb.w16.ptr), Arg::I(rows as i32), Arg::I(cols as i32)])?;
        g.gemm_f16(pb.act16.ptr, pb.w16.ptr, y, n, cols, rows, rows)
    }

    fn hc_mixes_rows(&self, fnw: &(DMat, Buf, Buf), h: u64, out: u64, n: usize) -> Result<()> {
        let c = &self.m.cfg;
        // tiled: 4 rows x 4 tokens per block, the same dots as hc_dots
        self.g.launch(self.k["hc_dots_t"], (6, (n as u32).div_ceil(4), 1), 256, 0, &[Arg::P(fnw.0.buf.ptr), Arg::P(h), Arg::P(out), Arg::I((c.hc * c.dim) as i32), Arg::F(c.eps), Arg::I(n as i32)])?;
        self.g.launch(self.k["hc_finish_w"], ((n as u32).div_ceil(4), 1, 1), 128, 0, &[Arg::P(out), Arg::P(fnw.1.ptr), Arg::P(fnw.2.ptr), Arg::F(c.hc_eps), Arg::I(c.sinkhorn_iters as i32), Arg::I(n as i32)])
    }

    fn rms_rows(&self, x: u64, w: &Buf, out: u64, n: usize, rows: usize, round: bool) -> Result<()> {
        self.g.launch(self.k["rmsnorm"], (rows as u32, 1, 1), 1024, 0, &[Arg::P(x), Arg::P(w.ptr), Arg::P(out), Arg::I(n as i32), Arg::I(n as i32), Arg::I(n as i32), Arg::F(self.m.cfg.eps), Arg::I(round as i32)])
    }

    fn rope_rows(&self, x: u64, heads: usize, hd: usize, compressed: bool, pos: u64, rows: usize, inverse: bool) -> Result<()> {
        let f = if compressed { &self.freqs.1 } else { &self.freqs.0 };
        self.g.launch(self.k["rope_pos"], (rows as u32, 1, 1), 256, 0, &[Arg::P(x), Arg::P(f.ptr), Arg::I(heads as i32), Arg::I(hd as i32), Arg::I(self.m.cfg.rope_dim as i32), Arg::I((heads * hd) as i32), Arg::P(pos), Arg::I(inverse as i32)])
    }

    /// Prefill `tokens` at positions self.pos.. ; `last` = this chunk ends the prompt (run the decoder tail and
    /// return the logits of the final token).
    pub fn prefill_chunk(&mut self, pb: &PrefillBufs, tokens: &[u32], last: bool) -> Result<Option<Vec<f32>>> {
        Ok(self.prefill_pass(pb, tokens, last, false)?.map(|mut v| v.pop().unwrap()))
    }

    /// Feed up to 8 tokens in one pass (speculative-decoding verify) and return the logits after each of them.
    /// Attention runs batched as in prefill; the experts go through the decode handoff (main GPU / aux GPUs / CPU).
    pub fn verify(&mut self, pb: &PrefillBufs, tokens: &[u32]) -> Result<Vec<Vec<f32>>> {
        anyhow::ensure!(!tokens.is_empty() && tokens.len() <= crate::handoff::MAXT && self.worker.is_some(), "verify: 1..=8 tokens, handoff path");
        self.verify_t = tokens.len();
        self.enqueue_aux(tokens.len())?;
        self.g.bind();
        let r = self.prefill_pass(pb, tokens, true, true);
        self.verify_t = 0;
        self.seq_base += self.m.cfg.n_layers as u32;
        Ok(r?.unwrap())
    }

    pub(super) fn prefill_pass(&mut self, pb: &PrefillBufs, tokens: &[u32], last: bool, all_logits: bool) -> Result<Option<Vec<Vec<f32>>>> {
        self.adapt_flush()?;
        if pb.stage_off.get() > 0 {
            // the previous pass's staged uploads may still be in flight
            self.g.sync()?;
            pb.stage_off.set(0);
        }
        let m = self.m;
        let c = &m.cfg;
        let (d, hc, nh, w) = (c.dim, c.hc, c.n_heads, c.window);
        let tn = tokens.len();
        assert!(tn <= pb.tmax);
        let p0 = self.pos;
        let g = &self.g;
        g.bind();
        // ---- per-chunk host tables: positions, window rows (ring slot or W + chunk row)
        let posv: Vec<i32> = (0..tn).map(|t| (p0 + t) as i32).collect();
        g.htod_sync(pb.pos.ptr, &posv)?;
        let mut widx = vec![-1i32; tn * w];
        for t in 0..tn {
            let p = p0 + t;
            for i in 0..w {
                let q = p as i64 - (w as i64 - 1) + i as i64;
                if q < 0 {
                    continue;
                }
                let q = q as usize;
                widx[t * w + i] = if q >= p0 { (w + q - p0) as i32 } else { (q % w) as i32 };
            }
        }
        g.htod_sync(pb.winidx.ptr, &widx)?;
        // ---- image rows of this chunk (chunk-local ranges), embedding + engram history
        self.img_rows.clear();
        for (start, rows) in &self.images {
            let (lo, hi) = (*start.max(&p0), (start + rows.len() / d).min(p0 + tn));
            if lo < hi {
                self.img_rows.push((lo - p0, hi - p0));
            }
        }
        let in_img = |t: usize| self.img_rows.iter().any(|&(lo, hi)| t >= lo && t < hi);
        for (t, &tk) in tokens.iter().enumerate() {
            // image positions take no part in an n-gram (model.py: DEAD)
            self.engram_hist.push(if in_img(t) { -1 } else { c.engram_token_map[tk as usize] as i64 });
        }
        let mut hbuf = vec![0f32; tn * hc * d];
        let mut e = vec![0f32; d];
        for (t, &tk) in tokens.iter().enumerate() {
            let p = p0 + t;
            match self.images.iter().find(|(s, r)| p >= *s && p < s + r.len() / d) {
                Some((s, r)) => e.copy_from_slice(&r[(p - s) * d..(p - s + 1) * d]),
                None => m.embed.dequant(tk as usize, &mut e),
            }
            for k in 0..hc {
                hbuf[(t * hc + k) * d..(t * hc + k + 1) * d].copy_from_slice(&e);
            }
        }
        g.htod_sync(pb.h.ptr, &hbuf)?;
        let mut pre = vec![0f32; tn * 24];
        for t in 0..tn {
            pre[t * 24] = 1.0;
        }
        g.htod_sync(pb.mix_f.ptr, &pre)?;
        // Engram rows depend only on the tokens: gather every Engram layer's rows for the chunk on a helper thread
        // now (layer 1's first), while the GPUs run the layers before them (DSX_ENGRAM_PREFETCH=0: inline, per layer)
        let prefetch = m.engram.is_some() && !c.engram_layers.is_empty() && std::env::var("DSX_ENGRAM_PREFETCH").map_or(true, |v| v != "0");
        let hist = if prefetch { self.engram_hist.clone() } else { vec![] };
        std::thread::scope(|sc| -> Result<()> {
            let (tx, rx) = std::sync::mpsc::channel();
            if prefetch {
                let hist = &hist;
                sc.spawn(move || {
                    for eidx in 0..c.engram_layers.len() {
                        if tx.send(engram_gather(m, hist, tn, eidx)).is_err() {
                            break;
                        }
                    }
                });
            }
            self.prefill_layers(pb, tn, p0, last, all_logits, prefetch.then_some(&rx))
        })?;
        let _ = nh;
        self.pos += tn;
        if !last {
            self.g.sync()?;
            return Ok(None);
        }
        let _ = all_logits;
        // logits of the final token (or of every token for a verify pass)
        let rows: Vec<usize> = if all_logits { (0..tn).collect() } else { vec![tn - 1] };
        for &t in &rows {
            let off = |base: u64, width: usize| base + (t * width * 4) as u64;
            self.g.launch(self.k["hc_pre"], (1, 1, 1), 256, 0, &[Arg::P(off(pb.h.ptr, hc * d)), Arg::P(off(pb.mix_f.ptr, 24)), Arg::I(0), Arg::P(off(pb.x.ptr, d)), Arg::I(d as i32)])?;
            self.rms_rows(off(pb.x.ptr, d), &self.out_norm, off(pb.x.ptr, d), d, 1, false)?;
        }
        if rows.len() > 1 && rows.len() <= crate::handoff::MAXT {
            // every row of a verify pass (rows 0..tn, contiguous): one pass over the head
            self.gemv_t(&self.head, pb.x.ptr, pb.logits.ptr, rows.len())?;
            let all: Vec<f32> = self.g.download(pb.logits.ptr, rows.len() * c.vocab)?;
            return Ok(Some(all.chunks(c.vocab).map(|r| r.to_vec()).collect()));
        }
        let mut out = vec![];
        for t in rows {
            self.gemv(&self.head, pb.x.ptr + (t * d * 4) as u64, pb.logits.ptr)?;
            out.push(self.g.download(pb.logits.ptr, c.vocab)?);
        }
        Ok(Some(out))
    }

    /// The layers of one prefill chunk; `rx` delivers the prefetched Engram rows (`engram_gather`) in layer order.
    fn prefill_layers(&mut self, pb: &PrefillBufs, tn: usize, p0: usize, last: bool, all_logits: bool, rx: Option<&std::sync::mpsc::Receiver<Result<EngramRows>>>) -> Result<()> {
        let m = self.m;
        let c = &m.cfg;
        let (d, hc, w) = (c.dim, c.hc, c.window);
        // decoder layer L recomputes the last (n_layers - L) * W tokens: exactly the cone the final token and the
        // decode windows depend on (each layer's window reaches one window further back than the next layer's)
        for li in 0..c.n_layers {
            let enc = li <= c.cand_layer; // layers 0..=20 see every token
            if !enc && !last {
                break;
            }
            // eval (logits at every position) needs the decoder on every row
            let n = if enc || all_logits { tn } else { tn.min((c.n_layers - li) * w) };
            let r0 = tn - n;
            if let Some(eidx) = c.engram_layers.iter().position(|&x| x == li) {
                let tq = std::time::Instant::now();
                // image rows pass through untouched (model.py: the gate is shut on image tokens)
                let row = hc * d * 4;
                let saved: Vec<(usize, usize, crate::gpu::Buf)> = self
                    .img_rows
                    .iter()
                    .map(|&(lo, hi)| -> Result<_> {
                        let b = self.g.alloc((hi - lo) * row)?;
                        self.g.dtod(b.ptr, pb.h.ptr + (lo * row) as u64, (hi - lo) * row)?;
                        Ok((lo, hi, b))
                    })
                    .collect::<Result<_>>()?;
                self.prefill_engram(pb, li, eidx, tn, p0, rx)?;
                for (lo, hi, b) in &saved {
                    self.g.dtod(pb.h.ptr + (lo * row) as u64, b.ptr, (hi - lo) * row)?;
                }
                if !saved.is_empty() {
                    self.g.sync()?;
                }
                if self.prof.is_some() {
                    self.g.sync()?;
                    *self.prof.as_mut().unwrap().entry("prefill engram").or_default() += tq.elapsed().as_secs_f64();
                }
            }
            let hrow = |base: u64, width: usize| base + (r0 * width * 4) as u64;
            // attention half
            let l = &self.layers[li];
            self.hc_mixes_rows(&l.hc_attn, hrow(pb.h.ptr, hc * d), hrow(pb.mix_a.ptr, 24), n)?;
            self.g.launch(self.k["hc_pre"], (n as u32, 1, 1), 256, 0, &[Arg::P(hrow(pb.h.ptr, hc * d)), Arg::P(hrow(pb.mix_f.ptr, 24)), Arg::I(0), Arg::P(hrow(pb.x.ptr, d)), Arg::I(d as i32)])?;
            self.rms_rows(hrow(pb.x.ptr, d), &l.attn_norm, hrow(pb.x.ptr, d), d, n, true)?;
            let tq = std::time::Instant::now();
            self.prefill_attention(pb, li, r0, n, tn, p0)?;
            if self.prof.is_some() {
                self.g.sync()?;
                *self.prof.as_mut().unwrap().entry("prefill attention+indexer").or_default() += tq.elapsed().as_secs_f64();
            }
            self.g.launch(self.k["hc_post"], (n as u32, 1, 1), 256, 0, &[Arg::P(hrow(pb.h.ptr, hc * d)), Arg::P(hrow(pb.a.ptr, d)), Arg::P(hrow(pb.mix_a.ptr, 24)), Arg::I(d as i32)])?;
            // ffn half
            let l = &self.layers[li];
            self.g.launch(self.k["hc_pre"], (n as u32, 1, 1), 256, 0, &[Arg::P(hrow(pb.h.ptr, hc * d)), Arg::P(hrow(pb.mix_a.ptr, 24)), Arg::I(0), Arg::P(hrow(pb.x.ptr, d)), Arg::I(d as i32)])?;
            self.hc_mixes_rows(&l.hc_ffn, hrow(pb.h.ptr, hc * d), hrow(pb.mix_f.ptr, 24), n)?;
            self.rms_rows(hrow(pb.x.ptr, d), &l.ffn_norm, hrow(pb.x.ptr, d), d, n, true)?;
            let tq = std::time::Instant::now();
            self.prefill_moe(pb, li, r0, n)?;
            if self.prof.is_some() {
                self.g.sync()?;
                *self.prof.as_mut().unwrap().entry("prefill moe").or_default() += tq.elapsed().as_secs_f64();
            }
            self.g.launch(self.k["hc_post"], (n as u32, 1, 1), 256, 0, &[Arg::P(hrow(pb.h.ptr, hc * d)), Arg::P(hrow(pb.a.ptr, d)), Arg::P(hrow(pb.mix_f.ptr, 24)), Arg::I(d as i32)])?;
        }
        Ok(())
    }

    fn prefill_engram(&mut self, pb: &PrefillBufs, li: usize, eidx: usize, tn: usize, p0: usize, rx: Option<&std::sync::mpsc::Receiver<Result<EngramRows>>>) -> Result<()> {
        let c = &self.m.cfg;
        let t_w = std::time::Instant::now();
        let r = match rx {
            Some(rx) => rx.recv().map_err(|_| anyhow::anyhow!("engram prefetch thread ended early"))??,
            None => engram_gather(self.m, &self.engram_hist, tn, eidx)?,
        };
        anyhow::ensure!(r.eidx == eidx, "engram rows for layer index {} arrived for {eidx}", r.eidx);
        let ebuf = r.rows;
        if std::env::var_os("DSX_PROF").is_some() {
            eprintln!("engram L{li}: {} rows, {} distinct, hash {:.0} ms, gather {:.0} ms, waited {:.0} ms", r.n, r.distinct, r.hash_s * 1e3, r.gather_s * 1e3, t_w.elapsed().as_secs_f64() * 1e3);
        }
        let _ = p0;
        // Tiny GEMVs read their f32 source directly; unlike GEMM they do not pack it
        // before writing the destination. Keep Engram input separate from shared Q/tmp.
        let input = if pb.tmp.is_none() && tn <= crate::handoff::MAXT { pb.w16.ptr } else { pb.attn_out() };
        self.g.htod_sync(input, &ebuf)?;
        let l = &self.layers[li];
        let kvw = l.engram_kv.as_ref().unwrap();
        self.gemm(pb, kvw, input, pb.tmp_ptr(), tn)?;
        self.g.launch(self.k["engram_gate_t"], (c.hc as u32, tn as u32, 1), 1024, 0, &[Arg::P(pb.h.ptr), Arg::P(pb.tmp_ptr()), Arg::P(l.engram_qk.as_ref().unwrap().ptr), Arg::I(c.dim as i32), Arg::F(c.eps)])
    }

    fn prefill_attention(&mut self, pb: &PrefillBufs, li: usize, r0: usize, n: usize, tn: usize, p0: usize) -> Result<()> {
        let m = self.m;
        let c = &m.cfg;
        let (d, hd, nh, w) = (c.dim, c.head_dim, c.n_heads, c.window);
        let ratio = m.layers[li].ratio;
        let compressed = ratio > 0;
        let rowp = |base: u64, width: usize, bytes: usize| base + (r0 * width * bytes) as u64;
        let x = rowp(pb.x.ptr, d, 4);
        let posr = rowp(pb.pos.ptr, 1, 4);
        let l = &self.layers[li];
        self.gemm(pb, &l.q_a, x, pb.qr.ptr, n)?;
        self.rms_rows(pb.qr.ptr, &l.q_a_norm, pb.qr.ptr, c.q_lora, n, true)?;
        if pb.tmp.is_some() {
            self.gemm(pb, &l.q_b, pb.qr.ptr, pb.q.ptr, n)?;
            self.rope_rows(pb.q.ptr, nh, hd, compressed, posr, n, false)?;
        }
        self.gemm(pb, &l.kv, x, pb.kv.ptr, n)?;
        self.rms_rows(pb.kv.ptr, &l.kv_norm, pb.kv.ptr, hd, n, false)?;
        self.rope_rows(pb.kv.ptr, 1, hd, compressed, posr, n, false)?;
        self.fake_fp8(pb.kv.ptr, n * hd)?;
        // combined window rows: [ring (W) | chunk rows]; this pass covers chunk rows r0..r0+n
        self.g.dtod(pb.wins.ptr, l.window.ptr, w * hd * 2)?;
        self.store_bf16(pb.kv.ptr, pb.wins.ptr + ((w + r0) * hd * 2) as u64, n * hd)?;
        // the ring keeps the newest W positions for decode
        self.g.launch(self.k["ring_scatter"], (n as u32, 1, 1), 128, 0, &[Arg::P(pb.wins.ptr + ((w + r0) * hd * 2) as u64), Arg::P(l.window.ptr), Arg::I((p0 + r0) as i32), Arg::I(n as i32), Arg::I(w as i32), Arg::I(hd as i32)])?;
        let mut kcomp = 0usize;
        if compressed {
            let is_src = c.kv_sources.contains(&li);
            let mut new_lat = 0usize; // latents appended by this chunk (pre-RoPE copies in pb.lat)
            let mut lat_pos: Vec<i32> = vec![];
            if is_src {
                let l = &self.layers[li];
                if ratio == 1 {
                    self.gemm(pb, l.comp_kv.as_ref().unwrap(), x, pb.lat.ptr, n)?;
                    self.rms_rows(pb.lat.ptr, l.comp_norm.as_ref().unwrap(), pb.lat.ptr, hd, n, false)?;
                    new_lat = n;
                    lat_pos = (0..n).map(|t| (p0 + r0 + t) as i32).collect();
                } else {
                    self.gemm(pb, l.comp_kv.as_ref().unwrap(), x, pb.ckv.ptr, n)?;
                    self.gemm(pb, l.comp_gate.as_ref().unwrap(), x, pb.csc.ptr, n)?;
                    if let (true, Some(v)) = (self.verify_t > 0, self.vrows.as_ref()) {
                        // speculative verify: keep this layer's per-token rows for keep()'s carry
                        let base = v.ptr + (li * 2 * crate::handoff::MAXT * hd * 4) as u64;
                        self.g.dtod(base, pb.ckv.ptr, n * hd * 4)?;
                        self.g.dtod(base + (crate::handoff::MAXT * hd * 4) as u64, pb.csc.ptr, n * hd * 4)?;
                    }
                    let (ks, ss) = l.comp_state.as_ref().unwrap();
                    let off = (p0 + r0) % 2; // 1: the first group's first token is in the carried state
                    let total = off + n;
                    let groups = total / 2;
                    if groups > 0 {
                        self.g.launch(self.k["compress_pairs"], (groups as u32, 1, 1), 512, 0, &[Arg::P(pb.ckv.ptr), Arg::P(pb.csc.ptr), Arg::P(ks.ptr), Arg::P(ss.ptr), Arg::I(off as i32), Arg::P(pb.lat.ptr), Arg::I(hd as i32)])?;
                        self.rms_rows(pb.lat.ptr, l.comp_norm.as_ref().unwrap(), pb.lat.ptr, hd, groups, false)?;
                    }
                    // a trailing half group waits in the state (slot 0 = position with p % 2 == 0)
                    if total % 2 == 1 {
                        let last = n - 1;
                        self.g.dtod(ks.ptr, pb.ckv.ptr + (last * hd * 4) as u64, hd * 4)?;
                        self.g.dtod(ss.ptr, pb.csc.ptr + (last * hd * 4) as u64, hd * 4)?;
                    }
                    new_lat = groups;
                    // group g ends at chunk row 2g + 1 - off: position p_end; the latent sits at p_end + 1 - ratio
                    lat_pos = (0..groups).map(|gi| (p0 + r0 + 2 * gi + 1 - off + 1 - 2) as i32).collect();
                }
                self.comp_src = li;
            }
            // index keys from this chunk's latents (owners), then queries for the n rows
            if c.index_sources.contains(&li) {
                let tq = std::time::Instant::now();
                self.prefill_indexer(pb, li, r0, n, p0, new_lat, &lat_pos)?;
                if self.prof.is_some() {
                    self.g.sync()?;
                    *self.prof.as_mut().unwrap().entry("prefill . indexer").or_default() += tq.elapsed().as_secs_f64();
                }
            }
            if new_lat > 0 {
                let l = &self.layers[li];
                self.up(pb, pb.ns.ptr, &lat_pos)?;
                self.rope_rows(pb.lat.ptr, 1, hd, true, pb.ns.ptr, new_lat, false)?;
                self.fake_fp4(pb.lat.ptr, new_lat * hd, 16, true)?;
                let nc = l.n_comp;
                self.store_bf16(pb.lat.ptr, l.comp_cache.as_ref().unwrap().ptr + (nc * hd * 2) as u64, new_lat * hd)?;
                self.layers[li].n_comp += new_lat;
            }
            kcomp = c.idx_topk;
        }
        // Finish every scratch user before Q overwrites the shared buffer. Indexer
        // queries use qr, so delaying q_b leaves their values and all equations unchanged.
        if pb.tmp.is_none() {
            let l = &self.layers[li];
            self.gemm(pb, &l.q_b, pb.qr.ptr, pb.q.ptr, n)?;
            self.rope_rows(pb.q.ptr, nh, hd, compressed, posr, n, false)?;
        }
        let _ = tn;
        let l = &self.layers[li];
        let comp_ptr = if kcomp > 0 { self.layers[self.comp_src].comp_cache.as_ref().unwrap().ptr } else { 0 };
        let cidx = rowp(pb.compidx.ptr, c.idx_topk, 4);
        if self.prof.is_some() {
            self.g.sync()?;
        }
        let tq = std::time::Instant::now();
        if n > 8 {
            // many queries: tensor-core kernel (5x the CUDA-core one at 4K tokens, same results to a bf16 step)
            pb.attn_tc.launch(&self.g, pb.q.ptr, pb.wins.ptr, rowp(pb.winidx.ptr, w, 4), w, comp_ptr, cidx, kcomp, l.sinks.ptr, pb.attn_out(), n, nh, hd)?;
        } else {
            self.sparse_attn_small(n, pb.q.ptr, pb.wins.ptr, rowp(pb.winidx.ptr, w, 4), w, comp_ptr, cidx, kcomp, l.sinks.ptr, pb.attn_out())?;
        }
        if self.prof.is_some() {
            self.g.sync()?;
            *self.prof.as_mut().unwrap().entry("prefill . sparse_attn kernel").or_default() += tq.elapsed().as_secs_f64();
        }
        self.rope_rows(pb.attn_out(), nh, hd, compressed, posr, n, true)?;
        self.fake_fp8(pb.attn_out(), n * nh * hd)?;
        // grouped output projection: group gi takes o[:, gi*gsz..] -> r[:, gi*olora..]; done as per-group GEMMs
        let gsz = nh * hd / c.o_groups;
        for gi in 0..c.o_groups {
            // gather the group's columns: o is [n][nh*hd]; GEMM with ld = nh*hd needs a strided input -> copy via to_f16 per row block
            self.gemm_group(pb, &l.o_a, gi * c.o_lora, c.o_lora, gsz, pb.attn_out() + (gi * gsz * 4) as u64, nh * hd, pb.r.ptr + (gi * c.o_lora * 4) as u64, c.o_groups * c.o_lora, n)?;
        }
        let nr = n * c.o_groups * c.o_lora;
        self.g.launch(self.k["round_bf16"], ((nr as u32).div_ceil(256), 1, 1), 256, 0, &[Arg::P(pb.r.ptr), Arg::I(nr as i32)])?;
        self.gemm(pb, &l.o_b, pb.r.ptr, rowp(pb.a.ptr, d, 4), n)
    }

    /// q6-plane weights: rows [r0, r0 + rows) dequantized to f16, x packed from stride ldx, y with stride ldy
    #[allow(clippy::too_many_arguments)]
    fn gemm_q6(&self, pb: &PrefillBufs, w: &DMat, r0: usize, rows: usize, k: usize, x: u64, ldx: usize, y: u64, ldy: usize, n: usize) -> Result<()> {
        let g = &self.g;
        let ([lo, hi, sc, d], _, dk) = w.planes(r0);
        let kk = w.cols;
        g.launch(self.k["pack_f16"], ((k as u32).div_ceil(256), n as u32, 1), 256, 0, &[Arg::P(x), Arg::I(ldx as i32), Arg::P(pb.act16.ptr), Arg::I(k as i32)])?;
        g.launch(self.k[dk], ((kk as u32).div_ceil(256), rows as u32, 1), 256, 0, &[Arg::P(lo), Arg::P(hi), Arg::P(sc), Arg::P(d), Arg::P(pb.w16.ptr), Arg::I(rows as i32), Arg::I(kk as i32)])?;
        g.gemm_f16(pb.act16.ptr, pb.w16.ptr, y, n, k, rows, ldy)
    }

    /// y[n][ldy] (cols y0..) = x[n][ldx] (cols x0..x0+k) . W[r0..r0+rows]^T  (strided operands for the grouped wo_a)
    #[allow(clippy::too_many_arguments)]
    fn gemm_group(&self, pb: &PrefillBufs, w: &DMat, r0: usize, rows: usize, k: usize, x: u64, ldx: usize, y: u64, ldy: usize, n: usize) -> Result<()> {
        if n <= crate::handoff::MAXT && w.q6.is_none() && k == w.cols {
            // a few rows: decode's GEMV reads the strided operands directly
            return self.gemv_rows_ld(w, r0, rows, x, ldx, y, ldy, n);
        }
        if w.q6.is_some() {
            return self.gemm_q6(pb, w, r0, rows, k, x, ldx, y, ldy, n);
        }
        let g = &self.g;
        // pack x columns into act16 [n][k]
        g.launch(self.k["pack_f16"], ((k as u32).div_ceil(256), n as u32, 1), 256, 0, &[Arg::P(x), Arg::I(ldx as i32), Arg::P(pb.act16.ptr), Arg::I(k as i32)])?;
        let (wp, sp, id) = w.ptrs(r0);
        g.launch(dq_kernel(&self.k, id, k), dq_grid(id, rows, k), 256, 0, &[Arg::I(id), Arg::P(wp), Arg::P(sp), Arg::P(pb.w16.ptr), Arg::I(rows as i32), Arg::I(k as i32)])?;
        g.gemm_f16(pb.act16.ptr, pb.w16.ptr, y, n, k, rows, ldy)
    }

    #[allow(clippy::too_many_arguments)]
    fn prefill_indexer(&mut self, pb: &PrefillBufs, li: usize, r0: usize, n: usize, p0: usize, new_lat: usize, lat_pos: &[i32]) -> Result<()> {
        let m = self.m;
        let c = &m.cfg;
        let ratio = m.layers[li].ratio;
        let (id, nhh, d) = (c.idx_dim, c.idx_heads, c.dim);
        if c.kv_sources.contains(&li) {
            if new_lat > 0 {
                let l = &self.layers[li];
                // keys from the pre-RoPE latents
                self.gemm(pb, l.idx_k.as_ref().unwrap(), pb.lat.ptr, pb.ckv.ptr, new_lat)?;
                self.rms_rows(pb.ckv.ptr, l.idx_k_norm.as_ref().unwrap(), pb.ckv.ptr, id, new_lat, false)?;
                self.up(pb, pb.ns.ptr, lat_pos)?;
                self.rope_rows(pb.ckv.ptr, 1, id, true, pb.ns.ptr, new_lat, false)?;
                self.fake_fp4(pb.ckv.ptr, new_lat * id, 32, false)?;
                let ni = l.n_idx;
                self.store_bf16(pb.ckv.ptr, l.idx_cache.as_ref().unwrap().ptr + (ni * id * 2) as u64, new_lat * id)?;
                self.layers[li].n_idx += new_lat;
            }
            self.idxk_src = li;
        }
        let l = &self.layers[li];
        let x = pb.x.ptr + (r0 * d * 4) as u64;
        self.gemm(pb, l.idx_q_b.as_ref().unwrap(), pb.qr.ptr, pb.iq.ptr, n)?;
        self.rope_rows(pb.iq.ptr, nhh, id, true, pb.pos.ptr + (r0 * 4) as u64, n, false)?;
        self.fake_fp4(pb.iq.ptr, n * nhh * id, 32, false)?;
        self.gemm(pb, l.idx_proj.as_ref().unwrap(), x, pb.iw.ptr, n)?;
        let wscale = (id as f32).powf(-0.5) * (nhh as f32).powf(-0.5);
        self.g.launch(self.k["scale_bf16"], (((n * nhh) as u32).div_ceil(256), 1, 1), 256, 0, &[Arg::P(pb.iw.ptr), Arg::I((n * nhh) as i32), Arg::F(wscale)])?;
        let kc = self.layers[self.idxk_src].idx_cache.as_ref().unwrap().ptr;
        let nbmax = self.max_ctx / c.cand_block + 1;
        let cone = (c.n_layers - 1 - c.cand_layer) * c.window; // decoder queries whose candidate masks layer 20 keeps
        let tn_all = r0 + n; // chunk length (every pass ends at the chunk's last row)
        let keep0 = tn_all.saturating_sub(cone); // chunk row of keep_tail row 0
        let tail_start = if li == c.cand_layer { keep0 } else { usize::MAX };
        for s0 in (0..n).step_by(SUB) {
            let rows = SUB.min(n - s0);
            let ns: Vec<i32> = (0..rows).map(|t| ((p0 + r0 + s0 + t + 1) / ratio) as i32).collect();
            let n_max = *ns.last().unwrap() as usize;
            if n_max == 0 {
                // nothing visible yet: no compressed rows for these queries
                let neg = vec![-1i32; rows * c.idx_topk];
                self.up(pb, pb.compidx.ptr + ((r0 + s0) * c.idx_topk * 4) as u64, &neg)?;
                continue;
            }
            self.up(pb, pb.ns.ptr, &ns)?;
            self.g.launch(
                self.k["indexer_score"],
                ((n_max as u32).div_ceil(8), rows as u32, 1),
                256,
                0,
                &[Arg::P(pb.iq.ptr + (s0 * nhh * id * 4) as u64), Arg::P(kc), Arg::P(pb.iw.ptr + (s0 * nhh * 4) as u64), Arg::P(pb.score.ptr), Arg::P(pb.ns.ptr), Arg::I(n_max as i32), Arg::I(nhh as i32), Arg::I(id as i32)],
            )?;
            let rr = crate::topk::Rows { scores: pb.score.ptr, lds: n_max, rows, ns: pb.ns.ptr, n_max };
            // the top-k scratch is shared with decode: never let a pass write past it
            let need = crate::topk::TopK::topk_scratch(n_max, c.idx_topk, rows).max(if li == c.cand_layer { crate::topk::TopK::blocks_scratch(n_max, c.cand_block, c.cand_blocks, rows) } else { 0 });
            anyhow::ensure!(need <= self.topk_scratch.bytes, "top-k scratch: {need} bytes needed for {rows} rows of {n_max}, {} allocated", self.topk_scratch.bytes);
            let out = pb.compidx.ptr + ((r0 + s0) * c.idx_topk * 4) as u64;
            if li == c.cand_layer {
                self.topk.blocks(&self.g, &rr, c.cand_block, c.cand_blocks, pb.keep.ptr, nbmax, self.topk_scratch.ptr)?;
                self.topk.topk(&self.g, &rr, c.idx_topk, 0, nbmax, c.cand_block, out, c.idx_topk, self.topk_scratch.ptr)?;
                // keep the candidate masks of the decoder tail's queries for layers 24..36
                for t in 0..rows {
                    let q = s0 + t;
                    if q >= tail_start {
                        self.g.dtod(pb.keep_tail.ptr + ((q - tail_start) * nbmax) as u64, pb.keep.ptr + (t * nbmax) as u64, nbmax)?;
                    }
                }
            } else if li > c.cand_layer {
                // decoder index layers: queries are the tail rows, masks saved by layer 20
                let mask = pb.keep_tail.ptr + ((r0 + s0 - keep0) * nbmax) as u64;
                self.topk.topk(&self.g, &rr, c.idx_topk, mask, nbmax, c.cand_block, out, c.idx_topk, self.topk_scratch.ptr)?;
            } else {
                self.topk.topk(&self.g, &rr, c.idx_topk, 0, nbmax, c.cand_block, out, c.idx_topk, self.topk_scratch.ptr)?;
            }
        }
        Ok(())
    }

    fn prefill_moe(&mut self, pb: &PrefillBufs, li: usize, r0: usize, n: usize) -> Result<()> {
        let m = self.m;
        let c = &m.cfg;
        let (d, inter) = (c.dim, c.inter);
        let x = pb.x.ptr + (r0 * d * 4) as u64;
        if self.verify_t > 0 {
            // a verify pass (<= 8 tokens): experts through the decode handoff (main GPU / aux GPUs / CPU), no weight streaming
            let a = pb.a.ptr + (r0 * d * 4) as u64;
            return self.moe_fast_t(li, x, a, n);
        }
        let tp0 = std::time::Instant::now();
        let l = &self.layers[li];
        // router (exact f32)
        // inputs are bf16-rounded, so bf16 x bf16 with f32 accumulation is exact
        self.g.launch(self.k["to_bf16"], (((n * d) as u32).div_ceil(256), 1, 1), 256, 0, &[Arg::P(x), Arg::P(pb.act16.ptr), Arg::I((n * d) as i32)])?;
        self.g.gemm_bf16(pb.act16.ptr, l.gate_w.buf.ptr, pb.tmp_ptr(), n, d, c.n_experts, c.n_experts)?;
        let (sel_buf, wts_buf) = (pb.gg.ptr, pb.go.ptr); // [n][k] each; both free until the experts run
        self.g.launch(self.k["router_topk"], (n as u32, 1, 1), 32, 0, &[Arg::P(pb.tmp_ptr()), Arg::P(l.gate_bias.ptr), Arg::P(sel_buf), Arg::P(wts_buf), Arg::I(c.n_experts as i32), Arg::I(c.top_k as i32), Arg::F(c.route_scale), Arg::P(self.bonus_ptr(li))])?;
        // tokens inside an image span select with the vision bias (model.py Gate: bias_vl)
        if let Some(bvl) = &l.gate_bias_vl {
            for &(lo, hi) in &self.img_rows {
                let (a, b) = (lo.max(r0), hi.min(r0 + n));
                if a >= b {
                    continue;
                }
                let t0 = a - r0;
                let (ek, k4) = ((t0 * c.n_experts * 4) as u64, (t0 * c.top_k * 4) as u64);
                self.g.launch(self.k["router_topk"], ((b - a) as u32, 1, 1), 32, 0, &[Arg::P(pb.tmp_ptr() + ek), Arg::P(bvl.ptr), Arg::P(sel_buf + k4), Arg::P(wts_buf + k4), Arg::I(c.n_experts as i32), Arg::I(c.top_k as i32), Arg::F(c.route_scale), Arg::P(self.bonus_ptr(li))])?;
            }
        }
        let sel: Vec<i32> = self.g.download(sel_buf, n * c.top_k)?;
        let wts: Vec<f32> = self.g.download(wts_buf, n * c.top_k)?;
        // routed experts grouped by expert
        let mut by: Vec<Vec<(i32, f32)>> = vec![vec![]; c.n_experts];
        for t in 0..n {
            for j in 0..c.top_k {
                by[sel[t * c.top_k + j] as usize].push((t as i32, wts[t * c.top_k + j]));
            }
        }
        let ml = &m.layers[li];
        // one upload of every expert's token rows and weights, then the per-expert kernels run without host syncs
        let mut all_rows: Vec<i32> = Vec::with_capacity(n * c.top_k);
        let mut all_w: Vec<f32> = Vec::with_capacity(n * c.top_k);
        let mut spans = vec![];
        for (e, toks) in by.iter().enumerate() {
            if !toks.is_empty() {
                spans.push((e, all_rows.len(), toks.len()));
                all_rows.extend(toks.iter().map(|x| x.0));
                all_w.extend(toks.iter().map(|x| x.1));
            }
        }
        self.g.htod_sync(pb.rows.ptr, &all_rows)?;
        self.g.htod_sync(pb.wrow.ptr, &all_w)?;
        // each aux GPU takes its resident experts; the experts on no GPU stream in over the main GPU's link and the
        // streaming aux GPUs' links, each to the GPU whose link frees up first (bytes / relative link speed, an aux
        // GPU starting with the f16 x it receives and the f32 sum it sends back). Link speeds count in quarters of the
        // main GPU's, so run-to-run noise in the startup measurement does not move experts between GPUs (which
        // changes the order of the f32 sums, and so the output in the last bits)
        let streamers: Vec<usize> = (0..self.aux.len()).filter(|&i| !self.skip_experts && self.aux[i].pre.as_ref().is_some_and(|p| !p.estage.is_empty())).collect();
        let rel = |gbs: f64| ((gbs / self.main_gbs * 4.0).round() / 4.0).max(0.25);
        let bw: Vec<f64> = std::iter::once(stream_main_weight()).chain(streamers.iter().map(|&i| rel(self.aux[i].pre.as_ref().unwrap().link_gbs))).collect();
        let mut queued: Vec<f64> = bw.iter().enumerate().map(|(j, b)| if j == 0 { 0.0 } else { (n * d * 6) as f64 / b }).collect();
        let mut aux_spans: Vec<Vec<(usize, usize, usize)>> = vec![vec![]; self.aux.len()];
        let mut main_spans = Vec::with_capacity(spans.len());
        // the CPU worker's share: of the experts no GPU holds, those with the fewest tokens, while the CPU's estimated
        // time (pre_cpu_us per expert-token) stays below what is left for the main GPU's link (a few-token expert
        // costs the link its whole weight, the CPU only a GEMV per token)
        let resident = |e: usize| self.exp0.as_ref().is_some_and(|s| s.has(li, e)) || self.aux.iter().any(|a| a.pre.is_some() && !self.skip_experts && a.exp.has(li, e));
        let mut on_cpu = vec![false; c.n_experts];
        let mut cpu_spans = vec![];
        if self.worker.is_some() && prefill_cpu() && n <= prefill_cpu_max_n() {
            // the link speed in 4 GB/s steps: startup measurement noise must not move experts between the CPU and the
            // GPU (a different split changes the order of the f32 sums, and so the outputs in the last bits)
            let link = ((self.main_gbs / 4.0).round() * 4.0).max(1.0) * 1e9;
            let mut cand: Vec<(usize, usize, f64)> = spans
                .iter()
                .filter(|sp| !resident(sp.0))
                .map(|sp| {
                    let ex = &ml.experts[sp.0];
                    (sp.0, sp.2, (ex.gate.bytes().len() + ex.up.bytes().len() + ex.down.bytes().len()) as f64)
                })
                .collect();
            cand.sort_by_key(|c| c.1);
            let mut link_s: f64 = cand.iter().map(|c| c.2 / link).sum();
            let mut cpu_s = 0.0;
            for &(e, nt, bytes) in &cand {
                let (cc, cl) = (pre_cpu_cost(nt) * pre_cpu_us() * 1e-6, bytes / link);
                if cpu_s + cc > link_s - cl {
                    break;
                }
                cpu_s += cc;
                link_s -= cl;
                on_cpu[e] = true;
            }
        }
        let dry = std::env::var_os("DSX_PRE_CPU_DRY").is_some();
        for sp in spans {
            let e = sp.0;
            if on_cpu[e] {
                cpu_spans.push(sp);
            }
            if on_cpu[e] && !dry {
            } else if self.exp0.as_ref().is_some_and(|s| s.has(li, e)) {
                main_spans.push(sp);
            } else if let Some(i) = self.aux.iter().position(|a| a.pre.is_some() && !self.skip_experts && a.exp.has(li, e)) {
                aux_spans[i].push(sp);
            } else {
                let ex = &ml.experts[e];
                let bytes = (ex.gate.bytes().len() + ex.up.bytes().len() + ex.down.bytes().len()) as f64;
                let j = (0..bw.len()).min_by(|&i, &j| (queued[i] + bytes / bw[i]).total_cmp(&(queued[j] + bytes / bw[j]))).unwrap();
                queued[j] += bytes / bw[j];
                if j == 0 { main_spans.push(sp) } else { aux_spans[streamers[j - 1]].push(sp) }
            }
        }
        // GPU0's resident experts with few tokens: decode's T16 kernels over all their picks at once (no f16 copy of the
        // weights, ~7 launches per 64 tokens instead of ~10 per expert); the rest keep dequant + GEMM
        let ntd = pre_dev_nt();
        let (dev_spans, spans): (Vec<_>, Vec<_>) = if ntd > 0 && self.exp0.as_ref().is_some_and(|s| s.is_t16_layer(li)) {
            main_spans.into_iter().partition(|&(e, _, nt)| nt <= ntd && self.exp0.as_ref().unwrap().has(li, e))
        } else {
            (vec![], main_spans)
        };
        let k = c.top_k;
        let al = |b: usize| b.div_ceil(256) * 256;
        let (o_sel, o_wts) = (al(n * c.n_experts * 4), al(n * c.n_experts * 4) + al(n * k * 4));
        let (o_out, o_scr) = (o_wts + al(n * k * 4), o_wts + al(n * k * 4) + al(n * d * 4));
        let need = o_scr + crate::experts::T16Ptrs::bytes(PRE_DEV_TB * k, d, inter);
        let (dev_spans, spans) = if !dev_spans.is_empty() && need > pb.tmp_bytes() {
            // no room in the scratch: everything through dequant + GEMM
            let mut all = spans;
            all.extend(dev_spans);
            (vec![], all)
        } else {
            (dev_spans, spans)
        };
        // the CPU's experts start now, in windows of MAX_TOKENS rows, and run through the GPUs' whole layer
        let cpu_job = !cpu_spans.is_empty();
        if cpu_job {
            let mt = crate::cpu::moe::MAX_TOKENS;
            let mut job = crate::handoff::PrefillJob { layer: li, n, windows: vec![], x: self.g.download::<f32>(x, n * d)?, out: vec![] };
            for t0 in (0..n).step_by(mt) {
                let t = mt.min(n - t0);
                let (mut ids, mut w) = (vec![], vec![]);
                for &(e, off, nt) in &cpu_spans {
                    let rows = &all_rows[off..off + nt];
                    if !rows.iter().any(|&r| (r as usize) >= t0 && (r as usize) < t0 + t) {
                        continue;
                    }
                    let i = ids.len();
                    ids.push(e);
                    w.resize((i + 1) * t, 0.0);
                    for (k, &r) in rows.iter().enumerate() {
                        let r = r as usize;
                        if r >= t0 && r < t0 + t {
                            w[i * t + r - t0] = all_w[off + k];
                        }
                    }
                }
                job.windows.push((t0, t, ids, w));
            }
            self.worker.as_ref().unwrap().pre.submit(job);
        }
        let use_aux = aux_spans.iter().any(|v| !v.is_empty());
        if use_aux {
            self.prefill_ship_x(pb, x, n)?;
        }
        let tp1 = std::time::Instant::now();
        // shared expert over all rows -> out
        self.gemm(pb, &l.sh_gate, x, pb.gg.ptr, n)?;
        self.gemm(pb, &l.sh_up, x, pb.gu.ptr, n)?;
        self.g.launch(self.k["swiglu_rows_f16"], ((inter as u32).div_ceil(256), n as u32, 1), 256, 0, &[Arg::P(pb.gg.ptr), Arg::P(pb.gu.ptr), Arg::P(0), Arg::P(pb.h16.ptr), Arg::I(inter as i32), Arg::F(c.swiglu_limit)])?;
        let sd = &l.sh_down;
        if sd.q6.is_some() {
            // h16 is already f16: dequant + GEMM directly
            let ([lo, hi, sc, d], _, dk) = sd.planes(0);
            let kk = sd.cols;
            self.g.launch(self.k[dk], ((kk as u32).div_ceil(256), sd.rows as u32, 1), 256, 0, &[Arg::P(lo), Arg::P(hi), Arg::P(sc), Arg::P(d), Arg::P(pb.w16.ptr), Arg::I(sd.rows as i32), Arg::I(kk as i32)])?;
        }
        let (sdw, sds, sdid) = sd.ptrs(0);
        if sd.q6.is_none() {
            self.g.launch(dq_kernel(&self.k, sdid, sd.cols), dq_grid(sdid, sd.rows, sd.cols), 256, 0, &[Arg::I(sdid), Arg::P(sdw), Arg::P(sds), Arg::P(pb.w16.ptr), Arg::I(sd.rows as i32), Arg::I(sd.cols as i32)])?;
        }
        if sd.act_fp8 {
            // FP8 checkpoint: the down input is fake-quantized to FP8 first (bf16 h -> f32 -> fp8 -> f16)
            bail!("prefill with the FP8 checkpoint's shared expert is not wired (use a GGUF export)");
        }
        self.g.gemm_f16(pb.h16.ptr, pb.w16.ptr, pb.out.ptr, n, inter, d, d)?;
        if self.skip_experts {
            // profiling only: shared expert alone
            let a = pb.a.ptr + (r0 * d * 4) as u64;
            self.g.dtod(a, pb.out.ptr, n * d * 4)?;
            return Ok(());
        }
        self.prof_mark("prefill moe . router+shared+aux ship", tp0)?;
        if !dev_spans.is_empty() {
            let tpd = std::time::Instant::now();
            let s0 = self.exp0.as_ref().unwrap();
            let mut on_dev = vec![false; c.n_experts];
            for &(e, _, _) in &dev_spans {
                on_dev[e] = true;
            }
            // picks of the other experts point at one that is not resident here, so the kernels skip them
            let sentinel = (0..c.n_experts).find(|&e| !s0.has(li, e)).unwrap_or(0) as i32;
            let sel_dev: Vec<i32> = sel.iter().map(|&e| if on_dev[e as usize] { e } else { sentinel }).collect();
            let base = pb.tmp_ptr();
            self.up(pb, base + o_sel as u64, &sel_dev)?;
            self.up(pb, base + o_wts as u64, &wts)?;
            for t0 in (0..n).step_by(PRE_DEV_TB) {
                let tb = PRE_DEV_TB.min(n - t0);
                s0.run_t16_ext(&self.g, li, base + (o_sel + t0 * k * 4) as u64, base + (o_wts + t0 * k * 4) as u64, k, tb, x + (t0 * d * 4) as u64, base + (o_out + t0 * d * 4) as u64, base + o_scr as u64)?;
            }
            self.g.launch(self.k["add_inplace"], (((n * d) as u32).div_ceil(256), 1, 1), 256, 0, &[Arg::P(pb.out.ptr), Arg::P(base + o_out as u64), Arg::I((n * d) as i32)])?;
            self.prof_mark("prefill moe . resident few-token experts (decode kernels)", tpd)?;
        }
        // the aux GPUs start once x has reached pinned memory (queued ahead of the shared expert, which covers the wait)
        let mut qs = if use_aux { self.prefill_aux_begin(pb, li, n, &aux_spans, &all_rows, &all_w)? } else { vec![] };
        let main = ExpQueue { g: &self.g, k: &self.k, gather: "gather_f16", x, rows: pb.rows.ptr, wrow: pb.wrow.ptr, gx16: pb.gx16.ptr, tx16: pb.tx16.ptr, gg: pb.gg.ptr, gu: pb.gu.ptr, h16: pb.h16.ptr, go: pb.go.ptr, w16: pb.w16.ptr, out: pb.out.ptr, store: self.exp0.as_ref(), estage: &pb.estage, ev_ready: &pb.ev_ready, ev_free: &pb.ev_free, nstaged: 0 };
        main.start()?;
        qs.insert(0, (usize::MAX, main, spans));
        // enqueue round-robin by progress through each GPU's list: a CUDA stream takes ~1K queued operations before
        // the host blocks, so queueing one GPU's whole layer first would idle the others behind it
        let mut next = vec![0usize; qs.len()];
        while let Some(j) = (0..qs.len()).filter(|&j| next[j] < qs[j].2.len()).min_by(|&a, &b| (next[a] as f64 / qs[a].2.len() as f64).total_cmp(&(next[b] as f64 / qs[b].2.len() as f64))) {
            let (e, off, nt) = qs[j].2[next[j]];
            next[j] += 1;
            self.queue_expert(&mut qs[j].1, li, e, off, nt)?;
        }
        for (i, q, _) in qs.iter().skip(1) {
            let oh = unsafe { std::slice::from_raw_parts_mut(self.aux[*i].pre.as_ref().unwrap().out_host.host_ptr() as *mut f32, n * d) };
            q.g.dtoh(oh, q.out)?;
        }
        drop(qs);
        self.g.bind();
        self.prof_mark("prefill moe . expert loops", tp1)?;
        let tp2 = std::time::Instant::now();
        for (a, sp) in self.aux.iter().zip(&aux_spans) {
            if sp.is_empty() {
                continue;
            }
            a.g.sync()?;
            self.g.bind();
            self.g.htod(pb.go.ptr, &a.pre.as_ref().unwrap().out_host.as_slice::<f32>()[..n * d])?;
            self.g.launch(self.k["add_inplace"], (((n * d) as u32).div_ceil(256), 1, 1), 256, 0, &[Arg::P(pb.out.ptr), Arg::P(pb.go.ptr), Arg::I((n * d) as i32)])?;
        }
        self.prof_mark("prefill moe . wait aux GPUs + add", tp2)?;
        if cpu_job {
            let tp3 = std::time::Instant::now();
            let job = self.worker.as_ref().unwrap().pre.wait();
            self.prof_mark("prefill moe . wait CPU experts", tp3)?;
            if std::env::var_os("DSX_PRE_CPU_DRY").is_some() {
                // debugging: the GPUs computed these experts too; the CPU's sums are dropped. DSX_PRE_CPU_CMP=1: the
                // CPU's experts once more on the GPU alone, into a zeroed buffer, against the CPU's sums
                if std::env::var_os("DSX_PRE_CPU_CMP").is_some() {
                    let cmp = self.g.alloc(n * d * 4)?;
                    self.g.memset(cmp.ptr, n * d * 4)?;
                    let mut q = ExpQueue { g: &self.g, k: &self.k, gather: "gather_f16", x, rows: pb.rows.ptr, wrow: pb.wrow.ptr, gx16: pb.gx16.ptr, tx16: pb.tx16.ptr, gg: pb.gg.ptr, gu: pb.gu.ptr, h16: pb.h16.ptr, go: pb.go.ptr, w16: pb.w16.ptr, out: cmp.ptr, store: self.exp0.as_ref(), estage: &pb.estage, ev_ready: &pb.ev_ready, ev_free: &pb.ev_free, nstaged: 0 };
                    q.start()?;
                    for &(e, off, nt) in &cpu_spans {
                        self.queue_expert(&mut q, li, e, off, nt)?;
                    }
                    let gpu: Vec<f32> = self.g.download(cmp.ptr, n * d)?;
                    let num: f64 = job.out.iter().zip(&gpu).map(|(a, b)| ((a - b) as f64).powi(2)).sum();
                    let den: f64 = gpu.iter().map(|b| (*b as f64).powi(2)).sum();
                    let (mut worst, mut wr) = (0f64, 0usize);
                    for r in 0..n {
                        let nr: f64 = (0..d).map(|i| ((job.out[r * d + i] - gpu[r * d + i]) as f64).powi(2)).sum();
                        let dr: f64 = (0..d).map(|i| (gpu[r * d + i] as f64).powi(2)).sum();
                        if dr > 0.0 && (nr / dr).sqrt() > worst {
                            worst = (nr / dr).sqrt();
                            wr = r;
                        }
                    }
                    eprintln!("pre-cpu cmp layer {li}: r0 {r0} n {n}, {} experts on the CPU, CPU vs GPU rel {:.3e}, worst row {wr} rel {worst:.3e}", cpu_spans.len(), (num / den.max(1e-30)).sqrt());
                }
                return self.finish_moe(pb, r0, n);
            }
            // in stream order: the last experts may still be using go (a synchronous copy is not ordered against the
            // stream and overwrote it, nondeterministic outputs); from pageable memory the call returns once staged
            self.g.htod(pb.go.ptr, &job.out)?;
            self.g.launch(self.k["add_inplace"], (((n * d) as u32).div_ceil(256), 1, 1), 256, 0, &[Arg::P(pb.out.ptr), Arg::P(pb.go.ptr), Arg::I((n * d) as i32)])?;
        }
        self.finish_moe(pb, r0, n)
    }

    /// a = bf16(out)
    fn finish_moe(&self, pb: &PrefillBufs, r0: usize, n: usize) -> Result<()> {
        let d = self.m.cfg.dim;
        let a = pb.a.ptr + (r0 * d * 4) as u64;
        self.g.dtod(a, pb.out.ptr, n * d * 4)?;
        self.g.launch(self.k["round_bf16"], (((n * d) as u32).div_ceil(256), 1, 1), 256, 0, &[Arg::P(a), Arg::I((n * d) as i32)])
    }
}

/// One Engram layer's rows for the last `tn` tokens of `hist`, in per-token order ([tn][(ngram-1)*heads][dim]).
pub(crate) struct EngramRows {
    eidx: usize,
    rows: Vec<f32>,
    n: usize,
    distinct: usize,
    hash_s: f64,
    gather_s: f64,
}

/// Hash the chunk's n-grams, read the distinct rows on 32 plain threads (deep NVMe queue; rayon's pool shares cores
/// with the spinning CPU-expert workers), then expand them back to per-token order.
pub(crate) fn engram_gather(m: &Model, hist: &[i64], tn: usize, eidx: usize) -> Result<EngramRows> {
    let c = &m.cfg;
    let tabs = m.engram.as_ref().ok_or_else(|| anyhow::anyhow!("engram tables not loaded"))?;
    let tab = tabs.layer(eidx, c.engram_dim)?;
    let tab = &tab;
    let hd = c.engram_dim;
    let base = hist.len() - tn; // history index of the chunk's first token
    let per = (c.engram_ngram - 1) * c.engram_heads;
    let t_h = std::time::Instant::now();
    let ids: Vec<u64> = (0..tn).flat_map(|t| m.engram_hashes_for(&hist[..base + t + 1], eidx)).collect();
    let hash_s = t_h.elapsed().as_secs_f64();
    let mut uniq: Vec<u64> = ids.clone();
    uniq.sort_unstable();
    uniq.dedup();
    let mut ubuf = vec![0f32; uniq.len() * hd];
    let nthreads = 32;
    let per_thread = uniq.len().div_ceil(nthreads).max(1);
    std::thread::scope(|sc| {
        for (chunk, idc) in ubuf.chunks_mut(per_thread * hd).zip(uniq.chunks(per_thread)) {
            sc.spawn(move || {
                crate::cpu::pool::pin_to_set(&crate::cpu::pool::ALL_CPUS);
                for (o, &row) in chunk.chunks_mut(hd).zip(idc) {
                    tab.row(row as usize, o);
                }
            });
        }
    });
    let mut rows = vec![0f32; tn * per * hd];
    for (o, id) in rows.chunks_mut(hd).zip(&ids) {
        let u = uniq.binary_search(id).unwrap();
        o.copy_from_slice(&ubuf[u * hd..(u + 1) * hd]);
    }
    let gather_s = t_h.elapsed().as_secs_f64() - hash_s;
    Ok(EngramRows { eidx, rows, n: ids.len(), distinct: uniq.len(), hash_s, gather_s })
}
