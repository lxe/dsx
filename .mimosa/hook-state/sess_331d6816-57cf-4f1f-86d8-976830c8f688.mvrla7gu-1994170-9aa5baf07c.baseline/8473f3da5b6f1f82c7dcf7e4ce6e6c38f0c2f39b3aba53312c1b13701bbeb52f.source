//! The decoder: weights in HBM, per-layer state, and one decode step recorded as a HIP graph.

use crate::hip::{Arg, Buf, Gpu, Graph};
use crate::load::{self, DMat};
use anyhow::{Context, Result, bail};
use dsx_core::model::Config;
use crate::ck::Ck;
use dsx_core::safetensors::SafeTensors;
use std::path::Path;

pub const D: usize = 5120;
pub const HC: usize = 4;
pub const NH: usize = 64;
pub const HD: usize = 512;
pub const QL: usize = 1280;
pub const NE: usize = 384;
pub const INTER: usize = 2304;
pub const WIN: usize = 128;
/// batched-decode slot window rings (kernels/prefill.h SRING): 16 spare rows for multi-token verify steps
pub const SRING: usize = WIN + 16;
pub const IH: usize = 32;
pub const ID: usize = 128;
pub const VOCAB: usize = 129280;
/// packed FP4 cache rows of the batched-decode slots (kernels/dsxa.hip CROW_B / IROW_B)
pub const CROW_B: usize = 288;
pub const IROW_B: usize = 72;
pub(crate) const ACH: usize = 16;
pub(crate) const NCHUNK: usize = (WIN + 512).div_ceil(ACH);
pub(crate) const CAND_CAP: usize = 65536;

// per-layer accumulator offsets (f32), all zeroed at the start of a step
pub(crate) const NPART: usize = D / 64;
pub(crate) const A_HMIX_A: usize = 0; // [NPART][32] per-block partial mix dots
pub(crate) const A_HMIX_F: usize = NPART * 32;
pub(crate) const A_XSS_A: usize = 2 * NPART * 32; // [NPART] partial sums of squares
pub(crate) const A_XSS_F: usize = A_XSS_A + NPART;
pub(crate) const A_ENG_SUM: usize = A_XSS_F + NPART;
pub(crate) const A_QR: usize = A_ENG_SUM + 16;
pub(crate) const A_KV: usize = A_QR + 1280;
pub(crate) const A_CKV: usize = A_KV + 512;
pub(crate) const A_CGATE: usize = A_CKV + 512;
pub(crate) const A_IW: usize = A_CGATE + 512;
pub(crate) const A_R: usize = A_IW + 32;
pub(crate) const A_A: usize = A_R + 8192;
pub(crate) const A_LOGITS: usize = A_A + 5120;
pub(crate) const A_SHG: usize = A_LOGITS + 384;
pub(crate) const A_SHU: usize = A_SHG + 2304;
pub(crate) const A_G: usize = A_SHU + 2304;
pub(crate) const A_U: usize = A_G + 6 * 2304;
pub(crate) const A_Y: usize = A_U + 6 * 2304;
pub(crate) const A_ENG_KV: usize = A_Y + 5120;
pub(crate) const A_RCNT: usize = A_ENG_KV + 25600; // router blocks-done counter
pub(crate) const A_CNT: usize = A_RCNT + 32; // last-block counters: 0 hc attn, 1 hc ffn, 2 q_a, 3 router, 4 wo_a, 5 shared, 6.. expert slots
pub(crate) const ACC: usize = A_CNT + 32;

// ---- kernel argument structs (mirror kernels/dsxa.hip)

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct HcArgs {
    pub scale: [f32; 3],
    pub base: [f32; 24],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct Job {
    w: u64,
    sc: u64,
    y: u64,
    x: u64,
    x2: u64,
    nw: u64,
    ss: u64,
    n: i32,
    k: i32,
    split: i32,
    wkind: i32,
    xkind: i32,
    blocks: i32,
    xgroup: i32,
    eslot: i32,
    wstride: i64,
    sstride: i64,
    lo: i32,
    epi: i32,
    ropec: i32,
    emat: i32,
}

pub(crate) const MAXJ: usize = 13;
pub(crate) const ROUTE_BYTES: usize = 1024; // Route struct slot per layer

/// target waves per GEMV launch for the shared k-split choice (dev-tunable)
pub static TARGET_WAVES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(2048);
/// minimum weight units (k-pairs / k128 blocks) per wave
pub static MIN_UNITS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(4);

#[repr(C)]
#[derive(Clone, Copy)]
struct Jobs {
    j: [Job; MAXJ],
    n: i32,
    eps: f32,
    lim: f32,
    cb3: [u32; 2],
    cb2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Post {
    kind: i32,
    n: i32,
    expect: i32,
    slot: i32,
    cnt: u64,
    src: u64,
    src2: u64,
    w: u64,
    oq: u64,
    os: u64,
    ob: u64,
}
pub(crate) const P_ROUTER: i32 = 1;
pub(crate) const P_RMSNORM: i32 = 2;
pub(crate) const P_BF16: i32 = 3;
pub(crate) const P_SWIGLU: i32 = 4;

#[repr(C)]
#[derive(Clone, Copy)]
struct Jobs2 {
    j: [Job; MAXJ],
    post: [Post; 7],
    jpost: [i32; MAXJ],
    n: i32,
    eps: f32,
    lim: f32,
    cb3: [u32; 2],
    cb2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct XOut {
    cnt: u64,
    nw: u64,
    oq: u64,
    os: u64,
    ob: u64,
    mixd: u64,
    ntok: i32,
    tpb: i32,
}

/// an FP8 activation buffer (codes, scales) + optional bf16 hi / lo copies
pub struct Act {
    pub q: Buf,
    pub s: Buf,
    pub h: Buf,
    pub l: Buf,
}
impl Act {
    fn new(g: &Gpu, k: usize) -> Result<Act> {
        Ok(Act { q: g.alloc_zeroed(k)?, s: g.alloc_zeroed(k / 32 * 4)?, h: g.alloc_zeroed(k * 2)?, l: g.alloc_zeroed(k * 2)? })
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct ExpertMat {
    pub woff: i64,
    pub soff: i64,
    pub mul: f32,
    pub fmt: i32,
    pad: [i32; 2],
}

pub const W_FP8: i32 = 0;
pub const W_BF16: i32 = 1;
pub const W_EXPERT: i32 = 3;
pub const X_NORM: i32 = 0;
pub const X_RMS: i32 = 1;
pub const X_PLAIN: i32 = 2;
pub const X_BF16R: i32 = 3;
pub const X_SWIGLU: i32 = 4;
pub const X_ENGRAM: i32 = 5;
pub const X_Q: i32 = 6;
pub const E_FP4: i32 = 0;
pub const E_CB3: i32 = 1;
pub const E_CB2: i32 = 2;

fn bytes_of<T: Copy>(v: &T) -> Vec<u8> {
    unsafe { std::slice::from_raw_parts(v as *const T as *const u8, std::mem::size_of::<T>()).to_vec() }
}

impl Job {
    fn new(wkind: i32, w: u64, sc: u64, n: usize, k: usize, y: u64, xkind: i32, x: u64) -> Job {
        Job {
            w,
            sc,
            y,
            x,
            x2: 0,
            nw: 0,
            ss: 0,
            n: n as i32,
            k: k as i32,
            split: 1,
            wkind,
            xkind,
            blocks: 0,
            xgroup: 0,
            eslot: -1,
            wstride: 0,
            sstride: 0,
            lo: 0,
            epi: 0,
            ropec: 0,
            emat: 0,
        }
    }
    fn dense(m: &DMat, wkind: i32, y: u64, xkind: i32, x: u64) -> Job {
        Job::new(wkind, m.w.ptr, m.sc.as_ref().map(|b| b.ptr).unwrap_or(0), m.n, m.k, y, xkind, x)
    }
    fn norm(mut self, nw: u64, ss: u64) -> Job {
        self.nw = nw;
        self.ss = ss;
        self
    }
    fn split(mut self, s: usize) -> Job {
        self.split = s as i32;
        self
    }
    /// k units per weight-tile column step
    fn ku(&self) -> usize {
        match self.wkind {
            W_FP8 => 64,
            W_BF16 => 32,
            _ => 128,
        }
    }
    fn lds(&self) -> usize {
        let kn = (self.k as usize / self.ku()).div_ceil(self.split as usize) * self.ku();
        match self.wkind {
            W_BF16 => 2 * (kn + 8) * 2,
            W_FP8 => ((kn + 8 + 15) & !15) + kn / 32 * 4 + 4 * (kn / 64) * 2 * 4,
            _ => ((kn + 8 + 15) & !15) + kn / 32 * 4,
        }
    }
}

pub struct Layer {
    pub id: usize,
    pub ratio: usize,
    pub kv_src: bool,
    pub idx_src: bool,
    pub comp_src: usize,
    pub idxk_src: usize,
    pub(crate) hc_attn_fn: Buf,
    pub(crate) hc_attn: HcArgs,
    pub(crate) hc_ffn_fn: Buf,
    pub(crate) hc_ffn: HcArgs,
    pub(crate) attn_norm: Buf,
    pub(crate) ffn_norm: Buf,
    pub(crate) q_norm: Buf,
    pub(crate) kv_norm: Buf,
    pub(crate) sinks: Buf,
    pub(crate) wq_a: DMat,
    pub(crate) wkv: DMat,
    pub(crate) wq_b: DMat,
    pub(crate) wo_a: DMat,
    pub(crate) wo_b: DMat,
    pub(crate) comp_wkv: Option<DMat>,
    pub(crate) comp_wgate: Option<DMat>,
    pub(crate) comp_norm: Option<Buf>,
    pub(crate) idx_wq_b: Option<DMat>,
    pub(crate) idx_proj: Option<DMat>,
    pub(crate) idx_wk: Option<Buf>,
    pub(crate) idx_knorm: Option<Buf>,
    pub(crate) gate: DMat,
    pub(crate) gate_bias: Buf,
    pub(crate) sh_w1: DMat,
    pub(crate) sh_w3: DMat,
    pub(crate) sh_w2: DMat,
    pub experts: Buf, // ExpertMat [384][3]
    pub(crate) engram: Option<(DMat, Buf)>,
    // state
    pub(crate) ring: Buf,
    pub(crate) comp_cache: Option<Buf>,
    pub(crate) idx_cache: Option<Buf>,
    pub(crate) cstate: Option<Buf>,
}

pub struct Decoder {
    pub g: Gpu,
    pub cfg: Config,
    pub layers: Vec<Layer>,
    pub(crate) embed: Buf,
    pub(crate) head: DMat,
    pub(crate) out_norm: Buf,
    pub arena: Buf,
    pub(crate) rope_f: Buf,
    pub(crate) rope_fc: Buf,
    pub max_ctx: usize,
    // step buffers
    pub(crate) st: Buf,
    pub(crate) h: Buf,
    pub(crate) acc: Buf,
    pub(crate) x_a: Buf,
    pub(crate) x_f: Buf,
    pub(crate) q: Buf,
    pub(crate) iq: Buf,
    pub(crate) score: Buf,
    pub(crate) bscore: Buf,
    pub(crate) hist: Buf,
    pub(crate) ctl: Buf,
    pub(crate) sel: Buf,
    pub(crate) cand: Buf,
    pub(crate) ckey: Buf,
    pub(crate) comp_idx: Buf,
    pub(crate) ncomp: Buf,
    pub(crate) kb: Buf,
    pub(crate) nkb: Buf,
    pub(crate) nb: Buf,
    pub(crate) keep: Buf,
    pub(crate) nkeep: Buf,
    pub(crate) cscore: Buf,
    pub(crate) part: Buf,
    pub(crate) oq: Buf,
    pub(crate) os: Buf,
    pub(crate) route: Buf,
    pub(crate) xfin: Buf,
    /// DSpark main_hidden of the last decoded token [3][D]
    pub(crate) dmh: Buf,
    pub logits: Buf,
    pub(crate) next: Buf,
    pub(crate) eng_rows: Buf,
    /// pinned host staging: [token i32, pad] then engram rows for the 2 engram layers
    pub host: *mut u8,
    pub(crate) graph: Option<Graph>,
    pub cb3: [u32; 2],
    pub cb2: u32,
    pub engram_st: Option<SafeTensors>,
    pub hist_tokens: Vec<i64>,
    /// DSpark draft layers (DSXA_DSPARK)
    pub(crate) dspark: Option<crate::dspark::Dspark>,
    /// verify / draft pass graphs by (tokens, draft, 16K-position bucket)
    pub(crate) vgraphs: std::cell::RefCell<std::collections::HashMap<(usize, bool, usize), Graph>>,
    /// batched decode, dense GEMMs on hipBLAS: row-major BF16 copies of the dense matrices, as (tiled buffer start,
    /// its bytes, tiled bytes per row, k, row-major copy); see Decoder::build_bdense
    pub(crate) bdense: Vec<(u64, usize, usize, usize, Buf)>,
    pub(crate) blas: Option<crate::hip::blas::Handle>,
    /// grouped expert launch in progress (batched step / prompt chunk): vgemv_xg{R}{G} tiles and token groups
    pub(crate) xg: std::cell::Cell<Option<(usize, usize)>>,
    /// the outputs of the next dense product were cleared by the prep before it (no vzero); reset by gm
    pub(crate) pz: std::cell::Cell<bool>,
    /// batched-step graphs by (tokens, draft, verify, 16K-position bucket, batch) and the shapes seen once
    pub(crate) bgraphs: std::cell::RefCell<std::collections::HashMap<(usize, bool, bool, usize, u64), Graph>>,
    pub(crate) bseen: std::cell::RefCell<std::collections::HashSet<(usize, bool, bool, usize, u64)>>,
    pub(crate) bgraph_epoch: std::cell::Cell<usize>,
    pub pos: usize,
    /// eager debugging: residual stream after every layer (downloaded with a sync per layer)
    pub dump: Option<std::cell::RefCell<Vec<Vec<f32>>>>,
    /// host copies of the per-layer expert tables (prompt path builds grouped expert jobs from them)
    pub(crate) etab: Vec<Vec<ExpertMat>>,
    pub(crate) pf: Option<crate::prefill::PfBufs>,
    pub(crate) act_a: Act,
    pub(crate) act_f: Act,
    pub(crate) act_qr: Act,
    pub(crate) act_r: Act,
    pub(crate) act_h: Act, // 7 slots of 2304: routed experts 0..5, shared 6
    pub(crate) act_head: Act,
    pub(crate) act_eng: Act,
    pub(crate) v1: bool,
    pub(crate) gtime: std::cell::RefCell<Option<Vec<(String, [f64; 5])>>>,
    pub(crate) last_blocks: std::cell::RefCell<usize>,
}

pub(crate) const ENG_ROW_BYTES: usize = 24 * 256 + 24 * 8;
pub(crate) const HOST_BYTES: usize = 256 + 2 * ENG_ROW_BYTES;

/// hyper-connection weights of the layer whose tensors start with `pre` ("layers.3", "mtp.0")
pub(crate) fn hc_args(st: &Ck, pre: &str, which: &str) -> Result<(Vec<f32>, HcArgs)> {
    let p = |s: &str| format!("{pre}.hc_{which}_{s}");
    let fnw = load::f32_of(st, &p("fn"))?;
    let scale = load::f32_of(st, &p("scale"))?;
    let base = load::f32_of(st, &p("base"))?;
    let mut a = HcArgs::default();
    a.scale.copy_from_slice(&scale);
    a.base.copy_from_slice(&base);
    Ok((fnw, a))
}

/// FNUZ fp8 byte for a value on the FNUZ grid (exact match required)
pub fn fnuz_byte(v: f32) -> u8 {
    for b in 0..=255u8 {
        if b == 0x80 {
            continue;
        }
        let s = b >> 7;
        let e = ((b >> 3) & 15) as i32;
        let m = (b & 7) as f32;
        let x = if e == 0 { m / 8.0 * 2f32.powi(-7) } else { (1.0 + m / 8.0) * 2f32.powi(e - 8) };
        let x = if s == 1 { -x } else { x };
        if x == v {
            return b;
        }
    }
    panic!("{v} is not an FNUZ fp8 value");
}

pub const CB3: [f32; 8] = [-8.0, -4.5, -2.5, -0.625, 1.25, 2.75, 4.5, 8.0];
pub const CB2: [f32; 4] = [-8.0, -1.875, 2.75, 8.0];

fn table(v: &[f32]) -> u32 {
    v.iter().enumerate().map(|(i, &x)| (fnuz_byte(x) as u32) << (8 * i)).sum()
}

/// Self-test hook: gemv2 FP8 job with an FP8 activation (codes + scales), optional RoPE epilogue
#[allow(clippy::too_many_arguments)]
pub fn test_gemv2(g: &Gpu, w: u64, sc: u64, n: usize, k: usize, xq: u64, xs: u64, y: u64, epi: i32, st: u64, rf: u64, rfc: u64) -> Result<()> {
    let mut j = Job::new(W_FP8, w, sc, n, k, y, 0, xq);
    j.x2 = xs;
    j.epi = epi;
    j.split = 1;
    j.blocks = n.div_ceil(64) as i32;
    let js = Jobs2 { j: [j; MAXJ], post: [Post::default(); 7], jpost: [-1; MAXJ], n: 1, eps: 1e-20, lim: 10.0, cb3: [0; 2], cb2: 0 };
    g.launch("gemv2", (j.blocks as u32, 1, 1), 256, 0, &[Arg::B(bytes_of(&js)), Arg::P(st), Arg::P(rf), Arg::P(rfc), Arg::P(0), Arg::F(0.0), Arg::P(0)])?;
    g.sync()
}

/// Self-test hook: one FP8 GEMV job over a plain f32 activation, with an optional RoPE epilogue.
#[allow(clippy::too_many_arguments)]
pub fn test_gemv(g: &Gpu, w: u64, sc: u64, n: usize, k: usize, x: u64, y: u64, epi: i32, ropec: i32, st: u64, rf: u64, rfc: u64) -> Result<()> {
    let mut j = Job::new(W_FP8, w, sc, n, k, y, X_PLAIN, x);
    j.epi = epi;
    j.ropec = ropec;
    j.split = 1;
    j.blocks = (n.div_ceil(64)) as i32;
    let lds = j.lds();
    let js = Jobs { j: [j; MAXJ], n: 1, eps: 1e-20, lim: 10.0, cb3: [0; 2], cb2: 0 };
    g.launch("gemv_jobs", (j.blocks as u32, 1, 1), 256, lds as u32, &[Arg::B(bytes_of(&js)), Arg::P(st), Arg::P(rf), Arg::P(rfc), Arg::I(0), Arg::P(0), Arg::P(0), Arg::F(0.0), Arg::P(0), Arg::P(0)])
}

/// Self-test hook: one job of any weight kind with an f32 activation (X_PLAIN, or X_BF16R).
#[allow(clippy::too_many_arguments)]
pub fn test_job(g: &Gpu, wkind: i32, w: u64, sc: u64, n: usize, k: usize, x: u64, y: u64, lo: i32, split: usize, table: u64) -> Result<()> {
    let mut j = Job::new(wkind, w, sc, n, k, y, X_PLAIN, x);
    j.lo = lo;
    j.split = split as i32;
    if wkind == W_EXPERT {
        j.sc = table;
        j.eslot = 0;
    }
    j.blocks = (n.div_ceil(64) * split) as i32;
    let lds = j.lds();
    let js = Jobs { j: [j; MAXJ], n: 1, eps: 1e-20, lim: 10.0, cb3: [table_of(&CB3[..4]), table_of(&CB3[4..])], cb2: table_of(&CB2) };
    let route = g.upload(&[0i32; 16])?; // expert slot 0 = expert 0, weight 0 (read with route_mode 2)
    g.launch("gemv_jobs", (j.blocks as u32, 1, 1), 256, lds as u32, &[Arg::B(bytes_of(&js)), Arg::P(0), Arg::P(0), Arg::P(0), Arg::I(2), Arg::P(0), Arg::P(0), Arg::F(0.0), Arg::P(route.ptr), Arg::P(0)])?;
    g.sync()
}

/// test_job without the sync (timing loops); the route buffer is shared and leaked
#[allow(clippy::too_many_arguments)]
pub fn test_job_nosync(g: &Gpu, wkind: i32, w: u64, sc: u64, n: usize, k: usize, x: u64, y: u64, lo: i32, split: usize, table: u64) -> Result<()> {
    static ROUTE: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    let route = *ROUTE.get_or_init(|| { let b = g.upload(&[0i32; 16]).unwrap(); let p = b.ptr; std::mem::forget(b); p });
    let mut j = Job::new(wkind, w, sc, n, k, y, X_PLAIN, x);
    j.lo = lo;
    j.split = split as i32;
    if wkind == W_EXPERT {
        j.sc = table;
        j.eslot = 0;
    }
    j.blocks = (n.div_ceil(64) * split) as i32;
    let lds = j.lds();
    let js = Jobs { j: [j; MAXJ], n: 1, eps: 1e-20, lim: 10.0, cb3: [table_of(&CB3[..4]), table_of(&CB3[4..])], cb2: table_of(&CB2) };
    g.launch("gemv_jobs", (j.blocks as u32, 1, 1), 256, lds as u32, &[Arg::B(bytes_of(&js)), Arg::P(0), Arg::P(0), Arg::P(0), Arg::I(2), Arg::P(0), Arg::P(0), Arg::F(0.0), Arg::P(route), Arg::P(0)])
}

pub fn table_of(v: &[f32]) -> u32 {
    table(v)
}

#[allow(clippy::too_many_arguments)]
pub fn test_hc_step_nosync(g: &Gpu, h: u64, y: u64, pmix: u64, pa: &HcArgs, premix: u64, prea: &HcArgs, fnw: u64, nmix: u64, x_raw: u64, xss: u64) -> Result<()> {
    g.launch(
        "hc_step",
        ((D / 64) as u32, 1, 1),
        256,
        0,
        &[Arg::P(h), Arg::P(y), Arg::I(1), Arg::P(pmix), Arg::B(bytes_of(pa)), Arg::P(premix), Arg::B(bytes_of(prea)), Arg::P(fnw), Arg::P(nmix), Arg::P(x_raw), Arg::P(xss), Arg::P(0), Arg::P(0), Arg::P(0), Arg::F(1e-6), Arg::F(1e-20), Arg::I(20), Arg::B(vec![0u8; 56])],
    )
}

#[allow(clippy::too_many_arguments)]
pub fn test_hc_step(g: &Gpu, h: u64, y: u64, pmix: u64, pa: &HcArgs, premix: u64, prea: &HcArgs, fnw: u64, nmix: u64, x_raw: u64, xss: u64) -> Result<()> {
    g.launch(
        "hc_step",
        ((D / 64) as u32, 1, 1),
        256,
        0,
        &[Arg::P(h), Arg::P(y), Arg::I(1), Arg::P(pmix), Arg::B(bytes_of(pa)), Arg::P(premix), Arg::B(bytes_of(prea)), Arg::P(fnw), Arg::P(nmix), Arg::P(x_raw), Arg::P(xss), Arg::P(0), Arg::P(0), Arg::P(0), Arg::F(1e-6), Arg::F(1e-20), Arg::I(20), Arg::B(vec![0u8; 56])],
    )?;
    g.sync()
}

/// Expert placement: format per (layer, expert)
pub struct Placement {
    pub fmt: Vec<i32>, // [layer * 384 + e]
}

pub(crate) const FP4_BYTES: usize = 3 * (INTER * D / 2 + INTER * D / 32);
pub(crate) const CB3_BYTES: usize = 3 * (INTER * D * 3 / 8 + INTER * D / 32);
pub(crate) const CB2_BYTES: usize = 3 * (INTER * D * 2 / 8 + INTER * D / 32);

pub fn expert_bytes(fmt: i32) -> usize {
    match fmt {
        E_FP4 => FP4_BYTES,
        E_CB3 => CB3_BYTES,
        _ => CB2_BYTES,
    }
}

impl Placement {
    /// Hottest experts (rank file order) stay MXFP4; then 3-bit; the coldest 2-bit if needed to fit `budget` bytes.
    /// `force_fp4`: (layer, expert) pairs that must stay exact (e.g. those a reference run routed to).
    pub fn plan(rank: &[(usize, usize)], budget: usize, force_fp4: &[(usize, usize)], n_layers: usize) -> Placement {
        let n = n_layers * NE;
        let mut order: Vec<usize> = rank.iter().filter(|&&(l, _)| l < n_layers).map(|&(l, e)| l * NE + e).collect();
        let mut seen = vec![false; n];
        for &i in &order {
            seen[i] = true;
        }
        for i in 0..n {
            if !seen[i] {
                order.push(i);
            }
        }
        let mut forced = vec![false; n];
        for &(l, e) in force_fp4 {
            if l < n_layers {
                forced[l * NE + e] = true;
            }
        }
        // start all FP4, demote from the cold end: FP4 -> CB3, then CB3 -> CB2, until it fits
        let mut fmt = vec![E_FP4; n];
        let total = |f: &Vec<i32>| f.iter().map(|&x| expert_bytes(x)).sum::<usize>();
        let mut tot = total(&fmt);
        for pass in [E_CB3, E_CB2] {
            for &i in order.iter().rev() {
                if tot <= budget {
                    break;
                }
                if forced[i] || fmt[i] >= pass {
                    continue;
                }
                tot -= expert_bytes(fmt[i]);
                fmt[i] = pass;
                tot += expert_bytes(pass);
            }
        }
        Placement { fmt }
    }
    pub fn counts(&self) -> [usize; 3] {
        let mut c = [0; 3];
        for &f in &self.fmt {
            c[f as usize] += 1;
        }
        c
    }
    pub fn bytes(&self) -> usize {
        self.fmt.iter().map(|&f| expert_bytes(f)).sum()
    }
}

impl Decoder {
    pub fn load(dir: &Path, max_ctx: usize, rank: &[(usize, usize)], force_fp4: &[(usize, usize)], n_layers_limit: Option<usize>) -> Result<Decoder> {
        let image: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/dsxa.hsaco"));
        if image.is_empty() {
            bail!("built without ROCm: no kernels");
        }
        let g = Gpu::new(image)?;
        let cfg = Config::from_hf(dir)?;
        let wpath = std::env::var("DSXA_WEIGHTS").map(std::path::PathBuf::from).unwrap_or_else(|_| dir.to_path_buf());
        let st = Ck::open(&wpath)?;
        let t0 = std::time::Instant::now();
        let nl = n_layers_limit.unwrap_or(cfg.n_layers);
        let mut layers = vec![];
        for l in 0..nl {
            let p = |s: &str| format!("layers.{l}.{s}");
            let has = |s: &str| st.tensors.contains_key(&p(s));
            let ratio = cfg.compress_ratios[l];
            let kv_src = cfg.kv_sources.contains(&l);
            let idx_src = cfg.index_sources.contains(&l);
            let comp_src = cfg.kv_sources.iter().filter(|&&s| s <= l).max().copied().unwrap_or(0);
            let idxk_src = comp_src;
            let (fa, ha) = hc_args(&st, &format!("layers.{l}"), "attn")?;
            let (ff, hf) = hc_args(&st, &format!("layers.{l}"), "ffn")?;
            let rows = |r: usize| if ratio > 0 { max_ctx / r + 1 } else { 0 };
            let engram = if has("engram.wkv.weight") {
                let q = load::f32_of(&st, &p("engram.q_weight"))?;
                let k = load::f32_of(&st, &p("engram.k_weight"))?;
                let qk: Vec<f32> = q.iter().zip(&k).map(|(a, b)| a * b).collect();
                Some((load::up_fp8(&g, &st, &p("engram.wkv.weight"))?, g.upload(&qk)?))
            } else {
                None
            };
            layers.push(Layer {
                id: l,
                ratio,
                kv_src,
                idx_src,
                comp_src,
                idxk_src,
                hc_attn_fn: g.upload(&fa)?,
                hc_attn: ha,
                hc_ffn_fn: g.upload(&ff)?,
                hc_ffn: hf,
                attn_norm: load::up_f32(&g, &st, &p("attn_norm.weight"))?,
                ffn_norm: load::up_f32(&g, &st, &p("ffn_norm.weight"))?,
                q_norm: load::up_f32(&g, &st, &p("attn.q_norm.weight"))?,
                kv_norm: load::up_f32(&g, &st, &p("attn.kv_norm.weight"))?,
                sinks: load::up_f32(&g, &st, &p("attn.attn_sink"))?,
                wq_a: load::up_fp8(&g, &st, &p("attn.wq_a.weight"))?,
                wkv: load::up_fp8(&g, &st, &p("attn.wkv.weight"))?,
                wq_b: load::up_fp8(&g, &st, &p("attn.wq_b.weight"))?,
                wo_a: load::up_fp8(&g, &st, &p("attn.wo_a.weight"))?,
                wo_b: load::up_fp8(&g, &st, &p("attn.wo_b.weight"))?,
                comp_wkv: if kv_src { Some(load::up_bf16(&g, &st, &p("attn.compressor.wkv.weight"))?) } else { None },
                comp_wgate: if kv_src && ratio > 1 { Some(load::up_bf16(&g, &st, &p("attn.compressor.wgate.weight"))?) } else { None },
                comp_norm: if kv_src { Some(load::up_f32(&g, &st, &p("attn.compressor.norm.weight"))?) } else { None },
                idx_wq_b: if idx_src { Some(load::up_fp8(&g, &st, &p("attn.indexer.wq_b.weight"))?) } else { None },
                idx_proj: if idx_src { Some(load::up_bf16(&g, &st, &p("attn.indexer.weights_proj.weight"))?) } else { None },
                idx_wk: if kv_src { Some(load::up_raw(&g, &st, &p("attn.indexer.wk.weight"))?) } else { None },
                idx_knorm: if kv_src { Some(load::up_f32(&g, &st, &p("attn.indexer.k_norm.weight"))?) } else { None },
                gate: load::up_bf16(&g, &st, &p("ffn.gate.weight"))?,
                gate_bias: load::up_f32(&g, &st, &p("ffn.gate.bias"))?,
                sh_w1: load::up_fp8(&g, &st, &p("ffn.shared_experts.w1.weight"))?,
                sh_w3: load::up_fp8(&g, &st, &p("ffn.shared_experts.w3.weight"))?,
                sh_w2: load::up_fp8(&g, &st, &p("ffn.shared_experts.w2.weight"))?,
                experts: g.alloc_zeroed(NE * 3 * std::mem::size_of::<ExpertMat>())?,
                engram,
                ring: g.alloc_zeroed(WIN * HD * 2)?,
                comp_cache: if kv_src { Some(g.alloc_zeroed(rows(ratio) * HD * 2)?) } else { None },
                idx_cache: if kv_src { Some(g.alloc_zeroed(rows(ratio) * ID * 2)?) } else { None },
                cstate: if kv_src { Some(g.alloc_zeroed(2 * 2 * HD * 4)?) } else { None },
            });
        }
        eprintln!("dense layers: {:.1} s, {:.1} GB on device", t0.elapsed().as_secs_f32(), g.allocated.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e9);
        let embed = load::up_raw(&g, &st, "embed.weight")?;
        let head_fp8 = std::env::var("DSXA_HEAD_FP8").map(|v| v == "1").unwrap_or(false);
        let head = if head_fp8 { load::up_bf16_as_fp8(&g, &st, "head.weight")? } else { load::up_bf16(&g, &st, "head.weight")? };
        let out_norm = load::up_f32(&g, &st, "norm.weight")?;
        let ropes = |orig: usize, base: f32| -> Vec<f32> {
            dsx_core::numerics::Rope::new(cfg.rope_dim, orig, base, cfg.rope_factor, cfg.beta_fast, cfg.beta_slow).freqs
        };
        let rope_f = g.upload(&ropes(0, cfg.rope_theta))?;
        let rope_fc = g.upload(&ropes(cfg.original_seq_len, cfg.compress_rope_theta))?;
        let f = |n: usize| g.alloc_zeroed(n * 4);
        let host = g.host_alloc(HOST_BYTES)?;
        unsafe { std::ptr::write_bytes(host, 0, HOST_BYTES) };

        // ---- experts
        let (free, _) = g.mem_info()?;
        // DSXA_DSPARK=4/3/2: reserve the native MXFP4 or lower-bit draft stack
        let draft = crate::dspark::bits_from_env().map(crate::dspark::bytes).unwrap_or(0);
        // DSXA_RESERVE_GB: extra HBM kept free (batched decode's per-sequence state); the coldest experts give way
        let extra = std::env::var("DSXA_RESERVE_GB").ok().and_then(|v| v.parse::<f64>().ok()).map(|g| (g * 1e9) as usize).unwrap_or(0);
        let reserve = (3usize << 30) + draft + extra;
        let budget = free.saturating_sub(reserve + (64 << 20));
        let pl = if st.is_export() {
            // an export fixes the formats (re-quantized experts carry `.cb` tiles); if the budget is smaller (draft
            // layers), the coldest MXFP4 experts are re-quantized to 3-bit at load
            let mut fmt = vec![E_FP4; nl * NE];
            for l in 0..nl {
                for e in 0..NE {
                    if let Some(&(f, _)) = st.cb.get(&format!("layers.{l}.ffn.experts.{e}.w1.cb")) {
                        fmt[l * NE + e] = f;
                    }
                }
            }
            let mut tot: usize = fmt.iter().map(|&f| expert_bytes(f)).sum();
            if (draft > 0 || extra > 0) && tot > budget {
                anyhow::ensure!(!rank.is_empty(), "export does not fit next to the draft layers / DSXA_RESERVE_GB: pass --rank to choose experts to re-quantize");
                // if even every MXFP4 expert at 3 bits would not fit, the coldest MXFP4 experts go to 2 bits first
                let n_fp4 = (0..nl * NE).filter(|&i| fmt[i] == E_FP4).count();
                let mut all3 = tot - n_fp4 * (FP4_BYTES - CB3_BYTES);
                for &(l, e) in rank.iter().rev() {
                    if all3 <= budget {
                        break;
                    }
                    let i = l * NE + e;
                    if l < nl && fmt[i] == E_FP4 {
                        fmt[i] = E_CB2;
                        tot = tot - FP4_BYTES + CB2_BYTES;
                        all3 -= CB3_BYTES - CB2_BYTES;
                    }
                }
                for &(l, e) in rank.iter().rev() {
                    if tot <= budget {
                        break;
                    }
                    let i = l * NE + e;
                    if l < nl && fmt[i] == E_FP4 {
                        fmt[i] = E_CB3;
                        tot = tot - FP4_BYTES + CB3_BYTES;
                    }
                }
            }
            Placement { fmt }
        } else {
            Placement::plan(rank, budget, force_fp4, nl)
        };
        let c = pl.counts();
        eprintln!("experts: {} MXFP4, {} 3-bit, {} 2-bit: {:.1} GB (free {:.1} GB)", c[0], c[1], c[2], pl.bytes() as f64 / 1e9, free as f64 / 1e9);
        let arena = g.alloc(pl.bytes())?;
        let mut dec = Decoder {
            embed,
            head,
            out_norm,
            arena,
            rope_f,
            rope_fc,
            max_ctx,
            st: g.alloc_zeroed(64)?,
            h: f(HC * D)?,
            acc: f(ACC * nl)?,
            x_a: f(D)?,
            x_f: f(D)?,
            q: f(NH * HD)?,
            iq: f(IH * ID)?,
            score: f(max_ctx + 64)?,
            bscore: f(max_ctx / 8 + 64)?,
            hist: f(4096)?,
            ctl: f(16)?,
            sel: f(2048)?,
            cand: f(CAND_CAP)?,
            ckey: f(CAND_CAP)?,
            comp_idx: f(512)?,
            ncomp: f(4)?,
            kb: f(2048)?,
            nkb: f(4)?,
            nb: f(4)?,
            keep: f(16384 + 64)?,
            nkeep: f(4)?,
            cscore: f(16384 + 64)?,
            part: f(NCHUNK * NH * (HD + 2))?,
            oq: f(NH * HD / 4)?,
            os: f(NH * HD / 32)?,
            route: g.alloc_zeroed(64 * ROUTE_BYTES)?,
            xfin: f(D)?,
            dmh: f(3 * D)?,
            logits: f(VOCAB)?,
            next: f(4)?,
            eng_rows: g.alloc_zeroed(2 * ENG_ROW_BYTES)?,
            host,
            graph: None,
            cb3: [table(&CB3[..4]), table(&CB3[4..])],
            cb2: table(&CB2),
            engram_st: None,
            hist_tokens: vec![],
            dspark: None,
            vgraphs: Default::default(),
            bdense: vec![],
            blas: None,
            xg: std::cell::Cell::new(None),
            pz: std::cell::Cell::new(false),
            bgraphs: Default::default(),
            bseen: Default::default(),
            bgraph_epoch: Default::default(),
            pos: 0,
            dump: None,
            etab: vec![],
            pf: None,
            act_a: Act::new(&g, D)?,
            act_f: Act::new(&g, D)?,
            act_qr: Act::new(&g, QL)?,
            act_r: Act::new(&g, 8192)?,
            act_h: Act::new(&g, 7 * INTER)?,
            act_head: Act::new(&g, D)?,
            act_eng: Act::new(&g, 6144)?,
            v1: !std::env::var("DSXA_V2").map(|v| v == "1").unwrap_or(false),
            gtime: Default::default(),
            last_blocks: Default::default(),
            layers,
            cfg,
            g,
        };
        dec.load_experts(&st, &pl)?;
        let names: Vec<String> = dec.cfg.engram_layers.iter().flat_map(|l| [format!("layers.{l}.engram.embed.weight"), format!("layers.{l}.engram.embed.scale")]).collect();
        let want: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
        let est = SafeTensors::open(dir, &want)?;
        est.advise_random();
        dec.engram_st = Some(est);
        dec.set_globals()?;
        eprintln!("loaded in {:.1} s; {:.1} GB on device", t0.elapsed().as_secs_f32(), dec.g.allocated.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e9);
        Ok(dec)
    }

    fn load_experts(&mut self, st: &Ck, pl: &Placement) -> Result<()> {
        let n_layers = self.layers.len();
        let tabs: Vec<u64> = self.layers.iter().map(|l| l.experts.ptr).collect();
        let name = |l: usize, e: usize, wn: &str| format!("layers.{l}.ffn.experts.{e}.{wn}.weight");
        let fmt = |l: usize, e: usize| pl.fmt[l * NE + e];
        let (etab, off) = self.upload_experts(st, &self.arena, n_layers, NE, &name, &fmt, &tabs)?;
        self.etab = etab;
        if off != pl.bytes() {
            bail!("expert arena: wrote {off}, planned {}", pl.bytes());
        }
        Ok(())
    }

    /// Routed experts (`n_layers` x `ne`, matrices w1/w3/w2) into `arena`, offsets relative to it: MXFP4 tiles, an
    /// export's codebook tiles, or re-quantized to a codebook on the GPU. Writes each layer's device table to
    /// `tables[l]`; returns the host tables and the bytes written.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn upload_experts(
        &self,
        st: &Ck,
        arena: &Buf,
        n_layers: usize,
        ne: usize,
        name: &(dyn Fn(usize, usize, &str) -> String + Sync),
        fmt_of: &(dyn Fn(usize, usize) -> i32 + Sync),
        tables: &[u64],
    ) -> Result<(Vec<Vec<ExpertMat>>, usize)> {
        use rayon::prelude::*;
        let t0 = std::time::Instant::now();
        let g = &self.g;
        let mut etab = vec![];
        // GPU staging for re-quantization: SLOTS matrices in flight (raw fp4 + scales, codes, scale bytes)
        const SLOTS: usize = 16;
        let mx = INTER * D;
        let slot = mx / 2 + mx / 32 + mx + mx / 32;
        let stage = g.alloc(SLOTS * slot)?;
        let cb3 = g.upload(&CB3)?;
        let cb2 = g.upload(&CB2)?;
        let mut off = 0usize;
        let mut nslot = 0usize;
        // pinned staging for a layer whose tiles are all host-made (FP4 or export): filled on all cores, then one DMA
        // (pageable copies go through one CPU thread at ~2 GB/s)
        let mut pin: (usize, usize) = (0, 0);
        for l in 0..n_layers {
            // host side, all of the layer's matrices in parallel: FP4 tiles, or the raw bytes for GPU re-quantization
            let items: Vec<(usize, usize, i32)> = (0..ne).flat_map(|e| (0..3).map(move |m| (e, m, 0))).map(|(e, m, _)| (e, m, fmt_of(l, e))).collect();
            let prepared: Vec<Result<(usize, usize, Vec<u8>, Vec<u8>)>> = items
                .par_iter()
                .map(|&(e, mi, fmt)| {
                    let wn = ["w1", "w3", "w2"][mi];
                    let name = name(l, e, wn);
                    if fmt != E_FP4 && st.is_export() && st.get(&name.replace(".weight", ".cb")).is_ok() {
                        // device tiles straight from the export (codes then scales); shapes from the matrix role
                        let (n, k) = if mi == 2 { (D, INTER) } else { (INTER, D) };
                        let cb = st.read(st.get(&name.replace(".weight", ".cb"))?);
                        return Ok((n, k, cb, vec![]));
                    }
                    let (n, kh) = load::shape2(st, &name)?;
                    let k = kh * 2;
                    let w = st.read(st.get(&name)?);
                    let s = st.read(st.get(&name.replace(".weight", ".scale"))?);
                    if fmt == E_FP4 {
                        let (c, sc) = load::tile_fp4(&w, &s, n, k);
                        Ok((n, k, c, sc))
                    } else {
                        Ok((n, k, w, s))
                    }
                })
                .collect();
            let mut table = vec![ExpertMat::default(); ne * 3];
            let prepared: Vec<(usize, usize, Vec<u8>, Vec<u8>)> = prepared.into_iter().collect::<Result<_>>()?;
            if prepared.iter().zip(&items).all(|(p, it)| it.2 == E_FP4 || p.3.is_empty()) {
                let lstart = off;
                let mut copies: Vec<(usize, usize, usize)> = vec![];
                for (it, (n, k, a, b)) in items.iter().zip(&prepared) {
                    let (e, mi, fmt) = *it;
                    let em = &mut table[e * 3 + mi];
                    em.fmt = fmt;
                    em.woff = off as i64;
                    copies.push((a.as_ptr() as usize, a.len(), off - lstart));
                    if fmt == E_FP4 {
                        off += a.len();
                        em.soff = off as i64;
                        copies.push((b.as_ptr() as usize, b.len(), off - lstart));
                        off += b.len();
                        em.mul = 1.0;
                    } else {
                        let bits = if fmt == E_CB3 { 3 } else { 2 };
                        em.soff = (off + n * k * bits / 8) as i64;
                        off += a.len();
                        em.mul = st.cb[&name(l, e, ["w1", "w3", "w2"][mi]).replace(".weight", ".cb")].1;
                    }
                }
                let total = off - lstart;
                if total > pin.1 {
                    if pin.0 != 0 {
                        g.host_free(pin.0 as *mut u8)?;
                    }
                    pin = (g.host_alloc(total + total / 8)? as usize, total + total / 8);
                }
                let dst = pin.0;
                copies.par_iter().for_each(|&(src, len, rel)| unsafe { std::ptr::copy_nonoverlapping(src as *const u8, (dst + rel) as *mut u8, len) });
                g.htod_async(arena.at(lstart), dst as *const u8, total)?;
                g.htod(tables[l], &table)?;
                etab.push(table.clone());
                g.sync()?;
                if l % 4 == 3 || l + 1 == n_layers {
                    eprintln!("experts: layer {l} done, {:.1} GB, {:.0} s", off as f64 / 1e9, t0.elapsed().as_secs_f32());
                }
                continue;
            }
            for (it, pr) in items.iter().zip(prepared.into_iter().map(Ok::<_, anyhow::Error>)) {
                let (e, mi, fmt) = *it;
                let (n, k, a, b) = pr?;
                let em = &mut table[e * 3 + mi];
                em.fmt = fmt;
                em.woff = off as i64;
                if fmt == E_FP4 {
                    g.htod_async(arena.at(off), a.as_ptr(), a.len())?;
                    off += a.len();
                    em.soff = off as i64;
                    g.htod_async(arena.at(off), b.as_ptr(), b.len())?;
                    off += b.len();
                    em.mul = 1.0;
                } else if b.is_empty() {
                    // export tiles: codes then scales, multiplier from the index
                    let bits = if fmt == E_CB3 { 3 } else { 2 };
                    let cbytes = n * k * bits / 8;
                    g.htod_async(arena.at(off), a.as_ptr(), a.len())?;
                    em.soff = (off + cbytes) as i64;
                    off += a.len();
                    let wn = ["w1", "w3", "w2"][mi];
                    em.mul = st.cb[&name(l, e, wn).replace(".weight", ".cb")].1;
                } else {
                    let bits = if fmt == E_CB3 { 3 } else { 2 };
                    // per-matrix multiplier: the largest group scale fits e4m3 (max 240) with headroom
                    let smax = b.iter().map(|&x| f32::from_bits((x as u32) << 23)).fold(0f32, f32::max);
                    let mul = (smax * 6.0 / 8.0 * 1.25) / 128.0;
                    let sb = stage.at((nslot % SLOTS) * slot);
                    nslot += 1;
                    let (sw, ss, sc, sbb) = (sb, sb + (mx / 2) as u64, sb + (mx / 2 + mx / 32) as u64, sb + (mx / 2 + mx / 32 + mx) as u64);
                    g.htod_async(sw, a.as_ptr(), a.len())?;
                    g.htod_async(ss, b.as_ptr(), b.len())?;
                    let (cb, nlv) = if bits == 3 { (cb3.ptr, 8) } else { (cb2.ptr, 4) };
                    let groups = n * k / 32;
                    g.launch("cb_quant", ((groups as u32).div_ceil(256), 1, 1), 256, 0, &[Arg::P(sw), Arg::P(ss), Arg::I(n as i32), Arg::I(k as i32), Arg::P(cb), Arg::I(nlv), Arg::F(mul), Arg::P(sc), Arg::P(sbb)])?;
                    let cbytes = n * k * bits / 8;
                    let threads = (n / 16) * (k / 128) * 64;
                    g.launch("cb_pack", ((threads as u32).div_ceil(256), 1, 1), 256, 0, &[Arg::P(sc), Arg::P(sbb), Arg::I(n as i32), Arg::I(k as i32), Arg::I(bits as i32), Arg::P(arena.at(off)), Arg::P(arena.at(off + cbytes))])?;
                    em.soff = (off + cbytes) as i64;
                    off += cbytes + n * k / 32;
                    em.mul = mul;
                }
            }
            g.htod(tables[l], &table)?;
            etab.push(table.clone());
            g.sync()?;
            if l % 4 == 3 || l + 1 == n_layers {
                eprintln!("experts: layer {l} done, {:.1} GB, {:.0} s", off as f64 / 1e9, t0.elapsed().as_secs_f32());
            }
        }
        if pin.0 != 0 {
            g.host_free(pin.0 as *mut u8)?;
        }
        Ok((etab, off))
    }

    pub(crate) fn accp(&self, l: usize, off: usize) -> u64 {
        self.acc.at((l * ACC + off) * 4)
    }

    fn jobs(&self, jobs: Vec<Job>, route_mode: i32, router_layer: Option<usize>) -> Result<()> {
        let lab = match (jobs[0].wkind, jobs[0].xkind, jobs[0].epi, route_mode, jobs.len()) {
            (_, _, 1, _, _) => "gemv q_b(+idx)",
            (_, X_Q, _, _, _) => "gemv wo_a",
            (_, X_BF16R, _, _, _) => "gemv wo_b",
            (_, X_PLAIN, _, _, _) => "gemv head",
            (_, X_SWIGLU, _, _, _) => "gemv experts down",
            (_, _, _, 1, _) => "gemv experts gate/up",
            (W_BF16, _, _, _, 3) => "gemv router+shared g/u",
            _ => "gemv q_a/kv/comp/idxw",
        };
        self.g.set_label(Some(lab));
        let r = self.jobs_inner(jobs, route_mode, router_layer);
        self.g.set_label(None);
        if self.gtime.borrow().is_some() {
            r?;
            self.g.sync()?;
            let (p, _) = self.g.global("g_tdbg")?;
            let t: Vec<i64> = self.g.dtoh(p, 16384 * 6)?;
            let nb = *self.last_blocks.borrow();
            let t0 = (0..nb.min(16384)).map(|b| t[b * 6]).min().unwrap_or(0);
            let mut ph = [0f64; 5];
            let mut last_start = 0i64;
            let mut last_end = 0i64;
            for b in 0..nb.min(16384) {
                let d = &t[b * 6..b * 6 + 6];
                ph[0] += (d[1] - d[0]) as f64;
                ph[1] += (d[2] - d[1]) as f64;
                ph[2] += (d[3] - d[2]) as f64;
                ph[3] += (d[4] - d[3]) as f64;
                last_start = last_start.max(d[0] - t0);
                last_end = last_end.max(d[4] - t0);
            }
            let n = nb.min(16384) as f64;
            let mut a = [ph[0] / n / 100.0, ph[1] / n / 100.0, ph[2] / n / 100.0, ph[3] / n / 100.0, last_start as f64 / 100.0];
            a[4] = last_end as f64 / 100.0;
            let z = vec![0i64; 16384 * 6];
            self.g.htod(p, &z)?;
            self.gtime.borrow_mut().as_mut().unwrap().push((lab.to_string(), a));
            return Ok(());
        }
        r
    }

    fn jobs_inner(&self, mut jobs: Vec<Job>, route_mode: i32, router_layer: Option<usize>) -> Result<()> {
        if jobs.len() > MAXJ {
            bail!("{} jobs", jobs.len());
        }
        let mut lds = 0;
        let mut blocks = 0u32;
        // jobs with split 0 share one k-split chosen so the whole launch has ~2048 waves
        let tiles: usize = jobs.iter().map(|j| (j.n as usize).div_ceil(16)).sum();
        let target = TARGET_WAVES.load(std::sync::atomic::Ordering::Relaxed);
        let minu = MIN_UNITS.load(std::sync::atomic::Ordering::Relaxed);
        let mut s = 1;
        while tiles * s < target && s < 64 {
            s *= 2;
        }
        for j in jobs.iter_mut() {
            if j.split == 0 {
                let units = j.k as usize / j.ku();
                let mut sj = s;
                while sj > 1 && units / sj < minu {
                    sj /= 2;
                }
                j.split = sj as i32;
            }
            j.blocks = ((j.n as usize).div_ceil(64) * j.split as usize) as i32;
            blocks += j.blocks as u32;
            lds = lds.max(j.lds());
        }
        *self.last_blocks.borrow_mut() = blocks as usize;
        let n = jobs.len();
        let pad = jobs[0];
        let mut arr = [pad; MAXJ];
        arr[..n].copy_from_slice(&jobs);
        let js = Jobs { j: arr, n: n as i32, eps: self.cfg.eps, lim: self.cfg.swiglu_limit, cb3: self.cb3, cb2: self.cb2 };
        let (logits, bias, route) = match router_layer {
            Some(l) => (self.accp(l, A_LOGITS), self.layers[l].gate_bias.ptr, self.route.at(l * ROUTE_BYTES)),
            None => (0, 0, self.route.ptr),
        };
        self.g.launch(
            "gemv_jobs",
            (blocks, 1, 1),
            256,
            lds as u32,
            &[
                Arg::B(bytes_of(&js)),
                Arg::P(self.st.ptr),
                Arg::P(self.rope_f.ptr),
                Arg::P(self.rope_fc.ptr),
                Arg::I(route_mode),
                Arg::P(logits),
                Arg::P(bias),
                Arg::F(self.cfg.route_scale),
                Arg::P(route),
                Arg::P(self.eng_rows.ptr),
            ],
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn hc_step(&self, y: u64, round_y: bool, pmix: u64, pa: &HcArgs, premix: u64, prea: &HcArgs, fnw: u64, nmix: u64, x_raw: u64, xss: u64, eng: Option<(u64, u64)>) -> Result<()> {
        self.hc_step_x(y, round_y, pmix, pa, premix, prea, fnw, nmix, x_raw, xss, eng, XOut::default())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn hc_step_x(&self, y: u64, round_y: bool, pmix: u64, pa: &HcArgs, premix: u64, prea: &HcArgs, fnw: u64, nmix: u64, x_raw: u64, xss: u64, eng: Option<(u64, u64)>, xo: XOut) -> Result<()> {
        let c = &self.cfg;
        let (ev, es) = eng.unwrap_or((0, 0));
        self.g.launch(
            "hc_step",
            ((D / 64) as u32, 1, 1),
            256,
            0,
            &[
                Arg::P(self.h.ptr),
                Arg::P(y),
                Arg::I(round_y as i32),
                Arg::P(pmix),
                Arg::B(bytes_of(pa)),
                Arg::P(premix),
                Arg::B(bytes_of(prea)),
                Arg::P(fnw),
                Arg::P(nmix),
                Arg::P(x_raw),
                Arg::P(xss),
                Arg::P(ev),
                Arg::P(es),
                Arg::P(0),
                Arg::F(c.hc_eps),
                Arg::F(c.eps),
                Arg::I(c.sinkhorn_iters as i32),
                Arg::B(bytes_of(&xo)),
            ],
        )
    }

    fn xout(&self, l: usize, which: usize, nw: u64, a: &Act) -> XOut {
        XOut { cnt: self.accp(l, A_CNT + which), nw, oq: a.q.ptr, os: a.s.ptr, ob: a.h.ptr, mixd: 0, ntok: 0, tpb: 0 }
    }

    /// gemv2 launch: jobs (split 0 = shared choice) with their post groups; expects are filled from block counts.
    fn jobs2(&self, label: &str, mut jobs: Vec<Job>, mut posts: Vec<(Post, Vec<usize>)>, layer: Option<usize>) -> Result<()> {
        let tiles: usize = jobs.iter().map(|j| (j.n as usize).div_ceil(16)).sum();
        let target = TARGET_WAVES.load(std::sync::atomic::Ordering::Relaxed);
        let minu = MIN_UNITS.load(std::sync::atomic::Ordering::Relaxed);
        let mut s = 1;
        while tiles * s < target && s < 64 {
            s *= 2;
        }
        let mut blocks = 0u32;
        for j in jobs.iter_mut() {
            if j.split == 0 {
                let units = j.k as usize / j.ku();
                let mut sj = s;
                while sj > 1 && units / sj < minu {
                    sj /= 2;
                }
                j.split = sj as i32;
            }
            j.blocks = ((j.n as usize).div_ceil(64) * j.split as usize) as i32;
            blocks += j.blocks as u32;
        }
        let mut jpost = [-1i32; MAXJ];
        let mut parr = [Post::default(); 7];
        for (pi, (p, members)) in posts.iter_mut().enumerate() {
            p.expect = members.iter().map(|&m| jobs[m].blocks).sum();
            for &m in members.iter() {
                jpost[m] = pi as i32;
            }
            parr[pi] = *p;
        }
        let n = jobs.len();
        let mut arr = [jobs[0]; MAXJ];
        arr[..n].copy_from_slice(&jobs);
        let js = Jobs2 { j: arr, post: parr, jpost, n: n as i32, eps: self.cfg.eps, lim: self.cfg.swiglu_limit, cb3: self.cb3, cb2: self.cb2 };
        let (bias, route) = match layer {
            Some(l) => (self.layers[l].gate_bias.ptr, self.route.at(l * ROUTE_BYTES)),
            None => (0, self.route.ptr),
        };
        self.g.set_label(Some(label));
        let r = self.g.launch(
            "gemv2",
            (blocks, 1, 1),
            256,
            0,
            &[Arg::B(bytes_of(&js)), Arg::P(self.st.ptr), Arg::P(self.rope_f.ptr), Arg::P(self.rope_fc.ptr), Arg::P(bias), Arg::F(self.cfg.route_scale), Arg::P(route)],
        );
        self.g.set_label(None);
        r
    }

    /// radix top-k: n from `n_src` (device int) or (pos+1)/ratio; writes `out` (+offset, -1 padding to kmax), *nout
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn topk(&self, scores: u64, ratio: usize, n_src: u64, k: usize, map: u64, offset: i32, out: u64, kmax: usize, nout: u64) -> Result<()> {
        // only the candidate-block list needs ascending order (expand_blocks); attention rows do not
        let sorted = (out == self.kb.ptr) as i32;
        let g = &self.g;
        g.launch("topk_hist", (128, 1, 1), 256, 0, &[Arg::P(self.st.ptr), Arg::I(ratio as i32), Arg::P(n_src), Arg::P(scores), Arg::P(self.hist.ptr), Arg::P(self.ctl.ptr)])?;
        g.launch("topk_scan", (1, 1, 1), 1024, 0, &[Arg::P(self.hist.ptr), Arg::I(k as i32), Arg::P(self.ctl.ptr)])?;
        g.launch("topk_collect", (128, 1, 1), 256, 0, &[Arg::P(scores), Arg::P(self.ctl.ptr), Arg::P(self.sel.ptr), Arg::P(self.cand.ptr), Arg::P(self.ckey.ptr), Arg::I(CAND_CAP as i32)])?;
        g.launch(
            "topk_final",
            (1, 1, 1),
            1024,
            0,
            &[Arg::P(self.ctl.ptr), Arg::P(self.sel.ptr), Arg::P(self.cand.ptr), Arg::P(self.ckey.ptr), Arg::P(map), Arg::I(offset), Arg::P(out), Arg::I(kmax as i32), Arg::P(nout), Arg::P(self.hist.ptr), Arg::I(sorted)],
        )
    }

    fn record(&self) -> Result<()> {
        if self.v1 { self.record1() } else { self.record2() }
    }

    /// Decode step on gemv2: producers write FP8/bf16 activations once, GEMV blocks read them from cache.
    fn record2(&self) -> Result<()> {
        let g = &self.g;
        let c = &self.cfg;
        let nl = self.layers.len();
        let j8 = |m: &DMat, y: u64, a: &Act| -> Job {
            let mut j = Job::dense(m, W_FP8, y, 0, a.q.ptr);
            j.x2 = a.s.ptr;
            j.split = 0;
            j
        };
        let j16 = |m: &DMat, y: u64, a: &Act, lo: bool| -> Job {
            let mut j = Job::dense(m, W_BF16, y, 0, 0);
            j.nw = a.h.ptr;
            j.ss = if lo { a.l.ptr } else { 0 };
            j.split = 0;
            j
        };
        let post = |kind: i32, n: usize, cnt: u64, src: u64, src2: u64, w: u64, a: Option<(&Act, usize)>, slot: i32| -> Post {
            let (oq, os, ob) = match a {
                Some((a, off)) => (a.q.at(off), a.s.at(off / 32 * 4), if kind == P_RMSNORM { a.h.at(off * 2) } else { 0 }),
                None => (0, 0, 0),
            };
            Post { kind, n: n as i32, expect: 0, slot, cnt, src, src2, w, oq, os, ob }
        };
        g.htod_async(self.st.at(4), unsafe { self.host.add(0) }, 4)?; // token
        g.htod_async(self.eng_rows.ptr, unsafe { self.host.add(256) }, 2 * ENG_ROW_BYTES)?;
        g.memset(self.acc.ptr, 0, ACC * nl * 4)?;
        g.launch("embed", (20, 1, 1), 256, 0, &[Arg::P(self.embed.ptr), Arg::P(self.st.ptr), Arg::P(self.h.ptr)])?;
        let none = HcArgs::default();
        for (li, l) in self.layers.iter().enumerate() {
            let xa = self.xout(li, 0, l.attn_norm.ptr, &self.act_a);
            if li == 0 {
                self.hc_step_x(0, false, 0, &none, 0, &none, l.hc_attn_fn.ptr, self.accp(0, A_HMIX_A), self.x_a.ptr, self.accp(0, A_XSS_A), None, xa)?;
            } else {
                let p = &self.layers[li - 1];
                let (pm, py) = (self.accp(li - 1, A_HMIX_F), self.accp(li - 1, A_Y));
                if let Some((wkv, qk)) = &l.engram {
                    self.hc_step(py, true, pm, &p.hc_ffn, 0, &none, 0, 0, 0, 0, None)?;
                    let ei = c.engram_layers.iter().position(|&x| x == li).unwrap();
                    g.launch("engram_act", (1, 1, 1), 256, 0, &[Arg::P(self.eng_rows.at(ei * ENG_ROW_BYTES)), Arg::P(self.act_eng.q.ptr), Arg::P(self.act_eng.s.ptr)])?;
                    let mut j = j8(wkv, self.accp(li, A_ENG_KV), &self.act_eng);
                    j.split = 1;
                    self.jobs2("gemv engram", vec![j], vec![], None)?;
                    g.launch("engram_sums", (5, 4, 1), 256, 0, &[Arg::P(self.accp(li, A_ENG_KV)), Arg::P(self.h.ptr), Arg::P(qk.ptr), Arg::P(self.accp(li, A_ENG_SUM))])?;
                    self.hc_step_x(0, false, 0, &none, pm, &p.hc_ffn, l.hc_attn_fn.ptr, self.accp(li, A_HMIX_A), self.x_a.ptr, self.accp(li, A_XSS_A), Some((self.accp(li, A_ENG_KV + 4 * D), self.accp(li, A_ENG_SUM))), xa)?;
                } else {
                    self.hc_step_x(py, true, pm, &p.hc_ffn, pm, &p.hc_ffn, l.hc_attn_fn.ptr, self.accp(li, A_HMIX_A), self.x_a.ptr, self.accp(li, A_XSS_A), None, xa)?;
                }
            }
            // ---- attention projections (q_a's last block: qr = rmsnorm -> FP8 for q_b)
            let mut jobs = vec![j8(&l.wq_a, self.accp(li, A_QR), &self.act_a), j8(&l.wkv, self.accp(li, A_KV), &self.act_a)];
            if let Some(m) = &l.comp_wkv {
                jobs.push(j16(m, self.accp(li, A_CKV), &self.act_a, false));
            }
            if let Some(m) = &l.comp_wgate {
                jobs.push(j16(m, self.accp(li, A_CGATE), &self.act_a, false));
            }
            if let Some(m) = &l.idx_proj {
                jobs.push(j16(m, self.accp(li, A_IW), &self.act_a, false));
            }
            let pq = post(P_RMSNORM, QL, self.accp(li, A_CNT + 2), self.accp(li, A_QR), 0, l.q_norm.ptr, Some((&self.act_qr, 0)), -1);
            self.jobs2("gemv q_a/kv/comp/idxw", jobs, vec![(pq, vec![0])], None)?;
            // q and the indexer q are stored, not accumulated (not in the zeroed region): no k-split
            let mut qj = j8(&l.wq_b, self.q.ptr, &self.act_qr);
            qj.split = 1;
            qj.epi = 1;
            qj.ropec = (l.ratio > 0) as i32;
            let mut jobs = vec![qj];
            if let Some(m) = &l.idx_wq_b {
                let mut j = j8(m, self.iq.ptr, &self.act_qr);
                j.split = 1;
                j.epi = 2;
                j.ropec = 1;
                jobs.push(j);
            }
            self.jobs2("gemv q_b(+idx)", jobs, vec![], None)?;
            self.attention_tail(li)?;
            let mut oa = j8(&l.wo_a, self.accp(li, A_R), &self.act_a);
            oa.x = self.oq.ptr;
            oa.x2 = self.os.ptr;
            oa.xgroup = 1024;
            let pr = post(P_BF16, 8192, self.accp(li, A_CNT + 4), self.accp(li, A_R), 0, 0, Some((&self.act_r, 0)), -1);
            self.jobs2("gemv wo_a", vec![oa], vec![(pr, vec![0])], None)?;
            self.jobs2("gemv wo_b", vec![j8(&l.wo_b, self.accp(li, A_A), &self.act_r)], vec![], None)?;
            // ---- residual step into the ffn half
            let am = self.accp(li, A_HMIX_A);
            let xf = self.xout(li, 1, l.ffn_norm.ptr, &self.act_f);
            self.hc_step_x(self.accp(li, A_A), false, am, &l.hc_attn, am, &l.hc_attn, l.hc_ffn_fn.ptr, self.accp(li, A_HMIX_F), self.x_f.ptr, self.accp(li, A_XSS_F), None, xf)?;
            // router (last block routes) + shared expert gate/up (last block: shared swiglu -> slot 6)
            let pr = post(P_ROUTER, NE, self.accp(li, A_CNT + 3), self.accp(li, A_LOGITS), 0, l.experts.ptr, None, -1);
            let ps = post(P_SWIGLU, INTER, self.accp(li, A_CNT + 5), self.accp(li, A_SHG), self.accp(li, A_SHU), 0, Some((&self.act_h, 6 * INTER)), -1);
            self.jobs2(
                "gemv router+shared g/u",
                vec![j16(&l.gate, self.accp(li, A_LOGITS), &self.act_f, false), j8(&l.sh_w1, self.accp(li, A_SHG), &self.act_f), j8(&l.sh_w3, self.accp(li, A_SHU), &self.act_f)],
                vec![(pr, vec![0]), (ps, vec![1, 2])],
                Some(li),
            )?;
            // routed experts gate/up; per-slot last block: swiglu x route weight -> FP8 for the down projection
            let mut jobs = vec![];
            let mut posts = vec![];
            for sl in 0..6 {
                for (mi, base) in [(0, A_G), (1, A_U)] {
                    let mut j = Job::new(W_EXPERT, self.arena.ptr, 0, INTER, D, self.accp(li, base + sl * INTER), 0, self.act_f.q.ptr);
                    j.x2 = self.act_f.s.ptr;
                    j.eslot = sl as i32;
                    j.emat = mi;
                    j.split = 0;
                    jobs.push(j);
                }
                posts.push((post(P_SWIGLU, INTER, self.accp(li, A_CNT + 6 + sl), self.accp(li, A_G + sl * INTER), self.accp(li, A_U + sl * INTER), 0, Some((&self.act_h, sl * INTER)), sl as i32), vec![2 * sl, 2 * sl + 1]));
            }
            // 7 posts max: slot 5 shares... (6 posts here)
            self.jobs2("gemv experts gate/up", jobs, posts, Some(li))?;
            // down projections (+ shared) into y
            let mut jobs = vec![];
            for sl in 0..6 {
                let mut j = Job::new(W_EXPERT, self.arena.ptr, 0, D, INTER, self.accp(li, A_Y), 0, self.act_h.q.at(sl * INTER));
                j.x2 = self.act_h.s.at(sl * INTER / 32 * 4);
                j.eslot = sl as i32;
                j.emat = 2;
                j.split = 2;
                jobs.push(j);
            }
            let mut sj = Job::dense(&l.sh_w2, W_FP8, self.accp(li, A_Y), 0, self.act_h.q.at(6 * INTER));
            sj.x2 = self.act_h.s.at(6 * INTER / 32 * 4);
            sj.split = 2;
            jobs.push(sj);
            self.jobs2("gemv experts down", jobs, vec![], Some(li))?;
            if let Some(d) = &self.dump {
                let tmp: Vec<f32> = g.dtoh(self.h.ptr, HC * D)?;
                self.hc_step(self.accp(li, A_Y), true, self.accp(li, A_HMIX_F), &l.hc_ffn, 0, &none, 0, 0, 0, 0, None)?;
                d.borrow_mut().push(g.dtoh(self.h.ptr, HC * D)?);
                g.htod(self.h.ptr, &tmp)?;
            }
        }
        // ---- final residual step, norm, head, argmax
        let last = &self.layers[nl - 1];
        let (pm, py) = (self.accp(nl - 1, A_HMIX_F), self.accp(nl - 1, A_Y));
        self.hc_step(py, true, pm, &last.hc_ffn, 0, &none, 0, 0, 0, 0, None)?;
        g.launch(
            "hc_final",
            (1, 1, 1),
            256,
            0,
            &[Arg::P(self.h.ptr), Arg::P(pm), Arg::B(bytes_of(&last.hc_ffn)), Arg::P(self.out_norm.ptr), Arg::P(self.xfin.ptr), Arg::F(c.hc_eps), Arg::F(c.eps), Arg::I(c.sinkhorn_iters as i32), Arg::P(self.act_head.q.ptr), Arg::P(self.act_head.s.ptr), Arg::P(self.act_head.h.ptr), Arg::P(self.act_head.l.ptr)],
        )?;
        let hj = if self.head.sc.is_some() { let mut j = j8(&self.head, self.logits.ptr, &self.act_head); j.split = 1; j } else { let mut j = j16(&self.head, self.logits.ptr, &self.act_head, true); j.split = 1; j };
        self.jobs2("gemv head", vec![hj], vec![], None)?;
        g.launch("argmax_part", (128, 1, 1), 256, 0, &[Arg::P(self.logits.ptr), Arg::I(VOCAB as i32), Arg::P(self.cand.ptr), Arg::P(self.ckey.ptr)])?;
        g.launch("argmax", (1, 1, 1), 64, 0, &[Arg::P(self.cand.ptr), Arg::P(self.ckey.ptr), Arg::I(128), Arg::P(self.next.ptr)])?;
        g.launch("advance", (1, 1, 1), 1, 0, &[Arg::P(self.st.ptr), Arg::P(self.next.ptr)])?;
        Ok(())
    }

    /// attn_kv, indexer + top-k, sparse attention + merge (writes oq/os) for layer li
    fn attention_tail(&self, li: usize) -> Result<()> {
        let r = self.attention_tail_inner(li);
        if let (Some(_), Ok(dir)) = (&self.dump, std::env::var("DSXA_DUMPQ")) {
            if li == 0 {
                let p = self.pos;
                let w = |n: &str, v: Vec<u8>| std::fs::write(format!("{dir}/{n}_p{p}.bin"), v);
                let q: Vec<f32> = self.g.dtoh(self.q.ptr, NH * HD)?;
                w("q", q.iter().flat_map(|v| v.to_le_bytes()).collect())?;
                let ring: Vec<u8> = self.g.dtoh(self.layers[0].ring.ptr, 4 * HD * 2)?;
                w("ring", ring)?;
                let oq: Vec<u8> = self.g.dtoh(self.oq.ptr, NH * HD)?;
                w("oq", oq)?;
                let qr: Vec<u8> = self.g.dtoh(self.act_qr.q.ptr, QL)?;
                w("qr", qr)?;
                let qs: Vec<u8> = self.g.dtoh(self.act_qr.s.ptr, QL / 32 * 4)?;
                w("qrs", qs)?;
                let qraw: Vec<u8> = self.g.dtoh(self.accp(0, A_QR), QL * 4)?;
                w("qraw", qraw)?;
                let cnt: Vec<u8> = self.g.dtoh(self.accp(0, A_CNT), 32 * 4)?;
                w("cnt", cnt)?;
                let xr: Vec<f32> = self.g.dtoh(self.x_a.ptr, D)?;
                w("xraw", xr.iter().flat_map(|v| v.to_le_bytes()).collect())?;
                let kv: Vec<f32> = self.g.dtoh(self.accp(0, A_KV), 512)?;
                w("kv", kv.iter().flat_map(|v| v.to_le_bytes()).collect())?;
            }
        }
        r
    }

    fn attention_tail_inner(&self, li: usize) -> Result<()> {
        let g = &self.g;
        let c = &self.cfg;
        let l = &self.layers[li];
        let wscale = (c.idx_dim as f32).powf(-0.5) * (c.idx_heads as f32).powf(-0.5);
        // window kv, compressor, indexer keys
        let opt = |b: &Option<Buf>| b.as_ref().map(|x| x.ptr).unwrap_or(0);
        g.launch(
            "attn_kv",
            (1, 1, 1),
            512,
            0,
            &[
                Arg::P(self.st.ptr),
                Arg::P(self.accp(li, A_KV)),
                Arg::P(l.kv_norm.ptr),
                Arg::P(l.ring.ptr),
                Arg::P(self.rope_f.ptr),
                Arg::I(l.ratio as i32),
                Arg::P(if l.kv_src { self.accp(li, A_CKV) } else { 0 }),
                Arg::P(self.accp(li, A_CGATE)),
                Arg::P(opt(&l.cstate)),
                Arg::P(opt(&l.comp_norm)),
                Arg::P(opt(&l.idx_wk)),
                Arg::P(opt(&l.idx_knorm)),
                Arg::P(opt(&l.comp_cache)),
                Arg::P(opt(&l.idx_cache)),
                Arg::P(self.rope_fc.ptr),
                Arg::F(c.eps),
            ],
        )?;
        // indexer + top-k
        if l.idx_src {
            let kc = self.layers[l.idxk_src].idx_cache.as_ref().context("idx cache")?.ptr;
            let after_cand = li > c.cand_layer;
            let (rows, nrows, sc) = if after_cand { (self.keep.ptr, self.nkeep.ptr, self.cscore.ptr) } else { (0, 0, self.score.ptr) };
            g.launch("idx_score", (256, 1, 1), 256, 0, &[Arg::P(self.st.ptr), Arg::I(l.ratio as i32), Arg::P(self.iq.ptr), Arg::P(self.accp(li, A_IW)), Arg::F(wscale), Arg::P(kc), Arg::P(rows), Arg::P(nrows), Arg::P(sc)])?;
            if li == c.cand_layer {
                g.launch("block_max8", (64, 1, 1), 256, 0, &[Arg::P(self.st.ptr), Arg::I(l.ratio as i32), Arg::I(c.cand_block as i32), Arg::P(self.score.ptr), Arg::P(self.bscore.ptr), Arg::P(self.nb.ptr)])?;
                self.topk(self.bscore.ptr, 0, self.nb.ptr, c.cand_blocks, 0, 0, self.kb.ptr, c.cand_blocks, self.nkb.ptr)?;
                g.launch("expand_blocks", (64, 1, 1), 256, 0, &[Arg::P(self.st.ptr), Arg::I(l.ratio as i32), Arg::I(c.cand_block as i32), Arg::P(self.kb.ptr), Arg::P(self.nkb.ptr), Arg::P(self.keep.ptr), Arg::P(self.nkeep.ptr)])?;
                g.launch("gather_f", (64, 1, 1), 256, 0, &[Arg::P(self.score.ptr), Arg::P(self.keep.ptr), Arg::P(self.nkeep.ptr), Arg::P(self.cscore.ptr)])?;
                self.topk(self.cscore.ptr, 0, self.nkeep.ptr, c.idx_topk, self.keep.ptr, WIN as i32, self.comp_idx.ptr, c.idx_topk, self.ncomp.ptr)?;
            } else if after_cand {
                self.topk(self.cscore.ptr, 0, self.nkeep.ptr, c.idx_topk, self.keep.ptr, WIN as i32, self.comp_idx.ptr, c.idx_topk, self.ncomp.ptr)?;
            } else {
                self.topk(self.score.ptr, l.ratio, 0, c.idx_topk, 0, WIN as i32, self.comp_idx.ptr, c.idx_topk, self.ncomp.ptr)?;
            }
        }
        // sparse attention
        let comp = if l.ratio > 0 { self.layers[l.comp_src].comp_cache.as_ref().context("comp cache")?.ptr } else { 0 };
        let scale = 1.0 / (HD as f32).sqrt();
        g.launch("attn_part", (NCHUNK as u32, 4, 1), 256, 0, &[Arg::P(self.st.ptr), Arg::P(self.q.ptr), Arg::P(l.ring.ptr), Arg::P(comp), Arg::P(self.comp_idx.ptr), Arg::P(self.ncomp.ptr), Arg::I(WIN as i32), Arg::F(scale), Arg::P(self.part.ptr)])?;
        let rf = if l.ratio > 0 { self.rope_fc.ptr } else { self.rope_f.ptr };
        g.launch("attn_merge", (NH as u32, 1, 1), 512, 0, &[Arg::P(self.st.ptr), Arg::P(self.part.ptr), Arg::I(NCHUNK as i32), Arg::P(l.sinks.ptr), Arg::P(rf), Arg::P(self.oq.ptr), Arg::P(self.os.ptr)])?;
        Ok(())
    }

    /// Enqueue one decode step (T = 1), first version (staged activations). Used for capture.
    fn record1(&self) -> Result<()> {
        let g = &self.g;
        let c = &self.cfg;
        let nl = self.layers.len();
        g.htod_async(self.st.at(4), unsafe { self.host.add(0) }, 4)?; // token
        g.htod_async(self.eng_rows.ptr, unsafe { self.host.add(256) }, 2 * ENG_ROW_BYTES)?;
        g.memset(self.acc.ptr, 0, ACC * nl * 4)?;
        g.launch("embed", (20, 1, 1), 256, 0, &[Arg::P(self.embed.ptr), Arg::P(self.st.ptr), Arg::P(self.h.ptr)])?;
        let _wscale = (c.idx_dim as f32).powf(-0.5) * (c.idx_heads as f32).powf(-0.5);
        for (li, l) in self.layers.iter().enumerate() {
            // ---- residual step into the attention half
            let none = HcArgs::default();
            if li == 0 {
                self.hc_step(0, false, 0, &none, 0, &none, l.hc_attn_fn.ptr, self.accp(0, A_HMIX_A), self.x_a.ptr, self.accp(0, A_XSS_A), None)?;
            } else {
                let p = &self.layers[li - 1];
                let (pm, py) = (self.accp(li - 1, A_HMIX_F), self.accp(li - 1, A_Y));
                if let Some((wkv, qk)) = &l.engram {
                    self.hc_step(py, true, pm, &p.hc_ffn, 0, &none, 0, 0, 0, 0, None)?;
                    let ei = c.engram_layers.iter().position(|&x| x == li).unwrap();
                    let mut j = Job::dense(wkv, W_FP8, self.accp(li, A_ENG_KV), X_ENGRAM, 0);
                    j.split = 1;
                    // rows buffer for this engram layer
                    self.jobs_with_rows(vec![j], self.eng_rows.at(ei * ENG_ROW_BYTES))?;
                    g.launch("engram_sums", (5, 4, 1), 256, 0, &[Arg::P(self.accp(li, A_ENG_KV)), Arg::P(self.h.ptr), Arg::P(qk.ptr), Arg::P(self.accp(li, A_ENG_SUM))])?;
                    self.hc_step(0, false, 0, &none, pm, &p.hc_ffn, l.hc_attn_fn.ptr, self.accp(li, A_HMIX_A), self.x_a.ptr, self.accp(li, A_XSS_A), Some((self.accp(li, A_ENG_KV + 4 * D), self.accp(li, A_ENG_SUM))))?;
                } else {
                    self.hc_step(py, true, pm, &p.hc_ffn, pm, &p.hc_ffn, l.hc_attn_fn.ptr, self.accp(li, A_HMIX_A), self.x_a.ptr, self.accp(li, A_XSS_A), None)?;
                }
            }
            if let Some(slot) = self.dspark.as_ref().and_then(|d| d.targets.iter().position(|&x| x == li)) {
                // DSpark main_hidden of this token (observed after plain steps)
                g.launch("mh_t", (20, 1, 1), 256, 0, &[Arg::P(self.h.ptr), Arg::P(self.dmh.ptr), Arg::I((3 * D) as i32), Arg::I((slot * D) as i32)])?;
            }
            // ---- attention projections
            let xs = self.accp(li, A_XSS_A);
            let mut jobs = vec![
                Job::dense(&l.wq_a, W_FP8, self.accp(li, A_QR), X_NORM, self.x_a.ptr).norm(l.attn_norm.ptr, xs).split(0),
                Job::dense(&l.wkv, W_FP8, self.accp(li, A_KV), X_NORM, self.x_a.ptr).norm(l.attn_norm.ptr, xs).split(0),
            ];
            if let Some(m) = &l.comp_wkv {
                jobs.push(Job::dense(m, W_BF16, self.accp(li, A_CKV), X_NORM, self.x_a.ptr).norm(l.attn_norm.ptr, xs).split(0));
            }
            if let Some(m) = &l.comp_wgate {
                jobs.push(Job::dense(m, W_BF16, self.accp(li, A_CGATE), X_NORM, self.x_a.ptr).norm(l.attn_norm.ptr, xs).split(0));
            }
            if let Some(m) = &l.idx_proj {
                jobs.push(Job::dense(m, W_BF16, self.accp(li, A_IW), X_NORM, self.x_a.ptr).norm(l.attn_norm.ptr, xs).split(0));
            }
            self.jobs(jobs, 0, None)?;
            let mut qj = Job::dense(&l.wq_b, W_FP8, self.q.ptr, X_RMS, self.accp(li, A_QR)).norm(l.q_norm.ptr, 0);
            qj.epi = 1;
            qj.ropec = (l.ratio > 0) as i32;
            let mut jobs = vec![qj];
            if let Some(m) = &l.idx_wq_b {
                let mut j = Job::dense(m, W_FP8, self.iq.ptr, X_RMS, self.accp(li, A_QR)).norm(l.q_norm.ptr, 0);
                j.epi = 2;
                j.ropec = 1;
                jobs.push(j);
            }
            self.jobs(jobs, 0, None)?;
            self.attention_tail(li)?;
            let mut oa = Job::dense(&l.wo_a, W_FP8, self.accp(li, A_R), X_Q, self.oq.ptr).split(0);
            oa.x2 = self.os.ptr;
            oa.k = (NH * HD / 8) as i32;
            oa.xgroup = 1024;
            self.jobs(vec![oa], 0, None)?;
            self.jobs(vec![Job::dense(&l.wo_b, W_FP8, self.accp(li, A_A), X_BF16R, self.accp(li, A_R)).split(0)], 0, None)?;
            // ---- residual step into the ffn half
            let am = self.accp(li, A_HMIX_A);
            self.hc_step(self.accp(li, A_A), false, am, &l.hc_attn, am, &l.hc_attn, l.hc_ffn_fn.ptr, self.accp(li, A_HMIX_F), self.x_f.ptr, self.accp(li, A_XSS_F), None)?;
            let xs = self.accp(li, A_XSS_F);
            let gj = Job::dense(&l.gate, W_BF16, self.accp(li, A_LOGITS), X_NORM, self.x_f.ptr).norm(l.ffn_norm.ptr, xs).split(0);
            self.jobs(
                vec![
                    gj,
                    Job::dense(&l.sh_w1, W_FP8, self.accp(li, A_SHG), X_NORM, self.x_f.ptr).norm(l.ffn_norm.ptr, xs).split(0),
                    Job::dense(&l.sh_w3, W_FP8, self.accp(li, A_SHU), X_NORM, self.x_f.ptr).norm(l.ffn_norm.ptr, xs).split(0),
                ],
                0,
                Some(li),
            )?;
            // routed experts: gate/up for the 6 picks
            let mut jobs = vec![];
            for s in 0..6 {
                for (mi, base) in [(0, A_G), (1, A_U)] {
                    let mut j = Job::new(W_EXPERT, self.arena.ptr, l.experts.ptr, INTER, D, self.accp(li, base + s * INTER), X_NORM, self.x_f.ptr).norm(l.ffn_norm.ptr, xs);
                    j.eslot = s as i32;
                    j.emat = mi;
                    j.split = 0;
                    jobs.push(j);
                }
            }
            self.jobs(jobs, 1, Some(li))?;
            // down projections (+ shared expert) into y
            let mut jobs = vec![];
            for s in 0..6 {
                let mut j = Job::new(W_EXPERT, self.arena.ptr, l.experts.ptr, D, INTER, self.accp(li, A_Y), X_SWIGLU, self.accp(li, A_G + s * INTER));
                j.x2 = self.accp(li, A_U + s * INTER);
                j.eslot = s as i32;
                j.emat = 2;
                j.split = 0;
                jobs.push(j);
            }
            let mut sj = Job::dense(&l.sh_w2, W_FP8, self.accp(li, A_Y), X_SWIGLU, self.accp(li, A_SHG)).split(0);
            sj.x2 = self.accp(li, A_SHU);
            jobs.push(sj);
            // every job adds into y: atomics (split >= 2 for all)
            for j in jobs.iter_mut() {
                j.split = 2;
            }
            self.jobs(jobs, 2, Some(li))?;
            if let (true, Some(_), Ok(dir)) = (li == 0, &self.dump, std::env::var("DSXA_DUMPQ")) {
                let p = self.pos;
                let sel: Vec<i32> = g.dtoh(self.route.ptr, 16)?;
                std::fs::write(format!("{dir}/sel_p{p}.bin"), sel.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())?;
                for (n, off, cnt) in [("a", A_A, D), ("logits", A_LOGITS, NE), ("y", A_Y, D), ("r", A_R, 8192), ("g", A_G, 6 * INTER), ("u", A_U, 6 * INTER), ("shg", A_SHG, INTER), ("shu", A_SHU, INTER)] {
                    let v: Vec<f32> = g.dtoh(self.accp(0, off), cnt)?;
                    std::fs::write(format!("{dir}/{n}_p{p}.bin"), v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())?;
                }
                let v: Vec<f32> = g.dtoh(self.x_f.ptr, D)?;
                std::fs::write(format!("{dir}/xf_p{p}.bin"), v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())?;
            }
            if let Some(d) = &self.dump {
                // h after layer li = the next residual step's post; run it on a copy via a post-only hc_step
                let tmp: Vec<f32> = g.dtoh(self.h.ptr, HC * D)?;
                let none = HcArgs::default();
                self.hc_step(self.accp(li, A_Y), true, self.accp(li, A_HMIX_F), &l.hc_ffn, 0, &none, 0, 0, 0, 0, None)?;
                d.borrow_mut().push(g.dtoh(self.h.ptr, HC * D)?);
                g.htod(self.h.ptr, &tmp)?;
            }
        }
        // ---- final residual step, norm, head, argmax
        let none = HcArgs::default();
        let last = &self.layers[nl - 1];
        let (pm, py) = (self.accp(nl - 1, A_HMIX_F), self.accp(nl - 1, A_Y));
        self.hc_step(py, true, pm, &last.hc_ffn, 0, &none, 0, 0, 0, 0, None)?;
        g.launch("hc_final", (1, 1, 1), 256, 0, &[Arg::P(self.h.ptr), Arg::P(pm), Arg::B(bytes_of(&last.hc_ffn)), Arg::P(self.out_norm.ptr), Arg::P(self.xfin.ptr), Arg::F(c.hc_eps), Arg::F(c.eps), Arg::I(c.sinkhorn_iters as i32), Arg::P(0), Arg::P(0), Arg::P(0), Arg::P(0)])?;
        let fp8 = self.head.sc.is_some();
        let mut hj = Job::dense(&self.head, if fp8 { W_FP8 } else { W_BF16 }, self.logits.ptr, X_PLAIN, self.xfin.ptr);
        hj.lo = 1;
        hj.split = 1;
        self.jobs(vec![hj], 0, None)?;
        g.launch("argmax_part", (128, 1, 1), 256, 0, &[Arg::P(self.logits.ptr), Arg::I(VOCAB as i32), Arg::P(self.cand.ptr), Arg::P(self.ckey.ptr)])?;
        g.launch("argmax", (1, 1, 1), 64, 0, &[Arg::P(self.cand.ptr), Arg::P(self.ckey.ptr), Arg::I(128), Arg::P(self.next.ptr)])?;
        g.launch("advance", (1, 1, 1), 1, 0, &[Arg::P(self.st.ptr), Arg::P(self.next.ptr)])?;
        Ok(())
    }

    /// head GEMV + argmax from xfin (used after the prompt path)
    pub(crate) fn head_logits(&self) -> Result<()> {
        let g = &self.g;
        let fp8 = self.head.sc.is_some();
        let mut hj = Job::dense(&self.head, if fp8 { W_FP8 } else { W_BF16 }, self.logits.ptr, X_PLAIN, self.xfin.ptr);
        hj.lo = 1;
        hj.split = 1;
        self.jobs(vec![hj], 0, None)?;
        g.launch("argmax_part", (128, 1, 1), 256, 0, &[Arg::P(self.logits.ptr), Arg::I(VOCAB as i32), Arg::P(self.cand.ptr), Arg::P(self.ckey.ptr)])?;
        g.launch("argmax", (1, 1, 1), 64, 0, &[Arg::P(self.cand.ptr), Arg::P(self.ckey.ptr), Arg::I(128), Arg::P(self.next.ptr)])
    }

    fn jobs_with_rows(&self, jobs: Vec<Job>, rows: u64) -> Result<()> {
        // same as jobs() but the engram rows pointer is this layer's slice
        let mut jobs = jobs;
        let mut lds = 0;
        let mut blocks = 0u32;
        for j in jobs.iter_mut() {
            j.blocks = ((j.n as usize).div_ceil(64) * j.split as usize) as i32;
            blocks += j.blocks as u32;
            lds = lds.max(j.lds());
        }
        let n = jobs.len();
        let mut arr = [jobs[0]; MAXJ];
        arr[..n].copy_from_slice(&jobs);
        let js = Jobs { j: arr, n: n as i32, eps: self.cfg.eps, lim: self.cfg.swiglu_limit, cb3: self.cb3, cb2: self.cb2 };
        self.g.launch(
            "gemv_jobs",
            (blocks, 1, 1),
            256,
            lds as u32,
            &[Arg::B(bytes_of(&js)), Arg::P(self.st.ptr), Arg::P(self.rope_f.ptr), Arg::P(self.rope_fc.ptr), Arg::I(0), Arg::P(0), Arg::P(0), Arg::F(0.0), Arg::P(self.route.ptr), Arg::P(rows)],
        )
    }

    pub fn reload_kernels(&mut self, image: &[u8]) -> Result<()> {
        self.graph = None;
        self.vgraphs.borrow_mut().clear();
        self.bgraphs.borrow_mut().clear();
        self.bseen.borrow_mut().clear();
        self.g.reload(image)?;
        self.set_globals()
    }

    /// device globals of the code object: the decode state's position for graph-replayed prompt kernels
    pub(crate) fn set_globals(&self) -> Result<()> {
        let (p, _) = self.g.global("g_posp")?;
        self.g.htod(p, &[self.st.ptr])
    }

    pub fn capture(&mut self) -> Result<()> {
        self.g.sync()?;
        self.g.begin_capture()?;
        let r = self.record();
        let gr = self.g.end_capture();
        r?;
        self.graph = Some(gr?);
        Ok(())
    }

    /// Host side of the Engram lookup for the newest token: hash ids -> rows (fp8 + e8m0) into pinned memory.
    /// Engram rows for a batch of new tokens into `out` ([t][2][ENG_ROW_BYTES]): hashes on the host, the 48 random
    /// row reads per token in parallel (the tables are on NVMe; serial reads cost ~100 us each on a page-cache miss).
    pub(crate) fn engram_rows_batch(&mut self, tokens: &[u32], out: &mut [u8]) -> Result<()> {
        use rayon::prelude::*;
        let c = &self.cfg;
        let base_len = self.hist_tokens.len();
        for &t in tokens {
            self.hist_tokens.push(c.engram_token_map[t as usize] as i64);
        }
        let est = self.engram_st.as_ref().context("engram tables")?;
        let hd = c.engram_dim;
        let mut tabs = vec![];
        for &l in &c.engram_layers {
            let wt = est.get(&format!("layers.{l}.engram.embed.weight"))?;
            let sct = est.get(&format!("layers.{l}.engram.embed.scale"))?;
            tabs.push((est.bytes(wt), est.bytes(sct)));
        }
        let hist = &self.hist_tokens;
        out.par_chunks_mut(2 * ENG_ROW_BYTES).enumerate().for_each(|(i, o)| {
            for (ei, (wb, sb)) in tabs.iter().enumerate() {
                let ids = engram_hashes(c, &hist[..base_len + i + 1], ei);
                let ob = &mut o[ei * ENG_ROW_BYTES..(ei + 1) * ENG_ROW_BYTES];
                for (r, &id) in ids.iter().enumerate() {
                    let id = id as usize;
                    ob[r * 256..(r + 1) * 256].copy_from_slice(&wb[id * hd..id * hd + 256]);
                    ob[24 * 256 + r * 8..24 * 256 + r * 8 + 8].copy_from_slice(&sb[id * (hd / 32)..id * (hd / 32) + 8]);
                }
            }
        });
        Ok(())
    }

    /// The Engram tables (memory-mapped) as a Send handle for fetch threads
    pub(crate) fn engram_tabs(&self) -> Result<EngTabs> {
        let c = &self.cfg;
        let est = self.engram_st.as_ref().context("engram tables")?;
        let mut tabs = vec![];
        for &l in &c.engram_layers {
            let wb = est.bytes(est.get(&format!("layers.{l}.engram.embed.weight"))?);
            let sb = est.bytes(est.get(&format!("layers.{l}.engram.embed.scale"))?);
            tabs.push((wb.as_ptr() as usize, wb.len(), sb.as_ptr() as usize, sb.len()));
        }
        Ok(EngTabs { c: c.clone(), tabs })
    }

    /// Engram rows of a batched decode step: token i's row from its own history hists[i] (ending with the token)
    pub(crate) fn engram_rows_multi(&self, hists: &[&[i64]], out: &mut [u8]) -> Result<()> {
        use rayon::prelude::*;
        let c = &self.cfg;
        let est = self.engram_st.as_ref().context("engram tables")?;
        let hd = c.engram_dim;
        let mut tabs = vec![];
        for &l in &c.engram_layers {
            let wt = est.get(&format!("layers.{l}.engram.embed.weight"))?;
            let sct = est.get(&format!("layers.{l}.engram.embed.scale"))?;
            tabs.push((est.bytes(wt), est.bytes(sct)));
        }
        // the row reads are random page-cache misses on NVMe (the tables exceed RAM): many threads keep many reads in
        // flight (rayon's ~20 left a 256-token step waiting 30-90 ms)
        let nth: usize = std::env::var("DSXA_ENG_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(128);
        let per = hists.len().div_ceil(nth).max(1);
        let one = |o: &mut [u8], h: &[i64]| {
            for (ei, (wb, sb)) in tabs.iter().enumerate() {
                let ids = engram_hashes(c, h, ei);
                let ob = &mut o[ei * ENG_ROW_BYTES..(ei + 1) * ENG_ROW_BYTES];
                for (r, &id) in ids.iter().enumerate() {
                    let id = id as usize;
                    ob[r * 256..(r + 1) * 256].copy_from_slice(&wb[id * hd..id * hd + 256]);
                    ob[24 * 256 + r * 8..24 * 256 + r * 8 + 8].copy_from_slice(&sb[id * (hd / 32)..id * (hd / 32) + 8]);
                }
            }
        };
        if hists.len() <= 4 {
            out.par_chunks_mut(2 * ENG_ROW_BYTES).zip(hists.par_iter()).for_each(|(o, h)| one(o, h));
            return Ok(());
        }
        std::thread::scope(|sc| {
            for (oc, hc) in out.chunks_mut(2 * ENG_ROW_BYTES * per).zip(hists.chunks(per)) {
                let one = &one;
                sc.spawn(move || {
                    for (o, h) in oc.chunks_mut(2 * ENG_ROW_BYTES).zip(hc) {
                        one(o, h);
                    }
                });
            }
        });
        Ok(())
    }

    pub(crate) fn engram_rows(&mut self, token: u32) -> Result<()> {
        let c = &self.cfg;
        self.hist_tokens.push(c.engram_token_map[token as usize] as i64);
        let est = self.engram_st.as_ref().context("engram tables")?;
        let hd = c.engram_dim;
        for (ei, &l) in c.engram_layers.iter().enumerate() {
            let ids = engram_hashes(c, &self.hist_tokens, ei);
            let wt = est.get(&format!("layers.{l}.engram.embed.weight"))?;
            let sct = est.get(&format!("layers.{l}.engram.embed.scale"))?;
            let (wb, sb) = (est.bytes(wt), est.bytes(sct));
            let base = unsafe { self.host.add(256 + ei * ENG_ROW_BYTES) } as usize;
            use rayon::prelude::*;
            ids.par_iter().enumerate().for_each(|(r, &id)| {
                let id = id as usize;
                let base = base as *mut u8;
                unsafe {
                    std::ptr::copy_nonoverlapping(wb[id * hd..].as_ptr(), base.add(r * 256), 256);
                    std::ptr::copy_nonoverlapping(sb[id * (hd / 32)..].as_ptr(), base.add(24 * 256 + r * 8), 8);
                }
            });
        }
        Ok(())
    }

    /// Feed one token at the current position; returns the greedy next token. Logits stay on the device.
    pub fn step(&mut self, token: u32) -> Result<u32> {
        if self.pos >= self.max_ctx {
            bail!("context full ({})", self.max_ctx);
        }
        if self.graph.is_none() {
            self.capture()?;
        }
        unsafe { *(self.host as *mut i32) = token as i32 };
        self.engram_rows(token)?;
        self.g.graph_launch(self.graph.as_ref().unwrap())?;
        let next: Vec<i32> = self.g.dtoh(self.next.ptr, 1)?;
        self.pos += 1;
        Ok(next[0] as u32)
    }

    /// One step without the graph, recording h after every layer.
    pub fn step_dump(&mut self, token: u32) -> Result<(u32, Vec<Vec<f32>>)> {
        unsafe { *(self.host as *mut i32) = token as i32 };
        self.engram_rows(token)?;
        self.dump = Some(Default::default());
        let r = self.record();
        let d = self.dump.take().unwrap().into_inner();
        r?;
        let next: Vec<i32> = self.g.dtoh(self.next.ptr, 1)?;
        self.pos += 1;
        Ok((next[0] as u32, d))
    }

    /// Eager step; after each GEMV launch, read the per-block phase stamps (GEMV_TIMING kernels) and print averages.
    pub fn gemv_timing(&mut self, ctx: usize) -> Result<()> {
        self.fill_random()?;
        self.set_pos(ctx)?;
        unsafe { *(self.host as *mut i32) = 1000 };
        self.engram_rows(1000)?;
        self.g.sync()?;
        *self.gtime.borrow_mut() = Some(vec![]);
        let r = self.record();
        let v = self.gtime.borrow_mut().take().unwrap();
        r?;
        let mut agg: std::collections::BTreeMap<String, [f64; 6]> = Default::default();
        for (lab, ph) in v {
            let e = agg.entry(lab).or_default();
            for i in 0..5 {
                e[i] += ph[i];
            }
            e[5] += 1.0;
        }
        for (lab, e) in agg {
            let n = e[5];
            println!("  {lab:<26} n {n:>3}: per block route {:.2} stage {:.2} stream {:.2} store {:.2} us; whole launch {:.2} us", e[0] / n, e[1] / n, e[2] / n, e[3] / n, e[4] / n);
        }
        self.pos += 1;
        Ok(())
    }

    /// One eager step with per-kernel timings.
    pub fn step_prof(&mut self, token: u32) -> Result<Vec<(String, f32, usize)>> {
        unsafe { *(self.host as *mut i32) = token as i32 };
        self.engram_rows(token)?;
        self.g.sync()?;
        self.g.prof_start();
        let r = self.record();
        let p = self.g.prof_end();
        r?;
        self.pos += 1;
        p
    }

    pub fn logits(&self) -> Result<Vec<f32>> {
        self.g.dtoh(self.logits.ptr, VOCAB)
    }

    /// routed experts of the last step, per layer
    pub fn routes(&self) -> Result<Vec<Vec<i32>>> {
        let v: Vec<i32> = self.g.dtoh(self.route.ptr, self.layers.len() * ROUTE_BYTES / 4)?;
        Ok(v.chunks(ROUTE_BYTES / 4).map(|c| c[..6].to_vec()).collect())
    }

    /// Reset to an empty context.
    pub fn reset(&mut self) -> Result<()> {
        self.g.memset(self.st.ptr, 0, 64)?;
        for l in &self.layers {
            self.g.memset(l.ring.ptr, 0, l.ring.bytes)?;
            if let Some(b) = &l.cstate {
                self.g.memset(b.ptr, 0, b.bytes)?;
            }
        }
        self.g.sync()?;
        self.hist_tokens.clear();
        self.pos = 0;
        Ok(())
    }

    /// Benchmark helper: pretend `pos` tokens are already in context (caches hold whatever they hold).
    pub fn set_pos(&mut self, pos: usize) -> Result<()> {
        self.g.htod(self.st.ptr, &[pos as i32])?;
        self.pos = pos;
        Ok(())
    }

    /// Fill compressed caches and the window with random bf16 values (benchmarks at long context).
    pub fn fill_random(&self) -> Result<()> {
        for l in &self.layers {
            for b in [Some(&l.ring), l.comp_cache.as_ref(), l.idx_cache.as_ref()].into_iter().flatten() {
                self.g.launch("fill_rand", (1024, 1, 1), 256, 0, &[Arg::P(b.ptr), Arg::I((b.bytes / 2) as i32), Arg::I(l.id as i32)])?;
            }
        }
        self.g.sync()
    }

    pub fn time_step(&mut self, token: u32) -> Result<f32> {
        if self.graph.is_none() {
            self.capture()?;
        }
        unsafe { *(self.host as *mut i32) = token as i32 };
        self.engram_rows(token)?;
        let gr = self.graph.as_ref().unwrap();
        let us = self.g.time(|| self.g.graph_launch(gr))?;
        self.pos += 1;
        Ok(us)
    }
}

/// model.py NgramHashState for engram layer `li`: hash ids of the 2..max_ngram n-grams ending at the newest token
/// Engram tables by raw pointer into the decoder's memory maps (which live as long as the decoder): sendable to the
/// batched step's background fetch
pub(crate) struct EngTabs {
    c: Config,
    tabs: Vec<(usize, usize, usize, usize)>,
}
unsafe impl Send for EngTabs {}
unsafe impl Sync for EngTabs {}

impl EngTabs {
    /// rows of table ei ([24 x 256 B][24 x 8 B]) for each history tail (ending with the new token) into out [n][ER]
    pub(crate) fn fetch(&self, hists: &[Vec<i64>], ei: usize, out: &mut [u8]) {
        let c = &self.c;
        let hd = c.engram_dim;
        let (wp, wl, sp, sl) = self.tabs[ei];
        let (wb, sb) = unsafe { (std::slice::from_raw_parts(wp as *const u8, wl), std::slice::from_raw_parts(sp as *const u8, sl)) };
        // random page-cache misses on NVMe (the tables exceed RAM): many threads keep many reads in flight
        let nth: usize = std::env::var("DSXA_ENG_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(256);
        let per = hists.len().div_ceil(nth).max(1);
        std::thread::scope(|sc| {
            for (oc, hc) in out.chunks_mut(ENG_ROW_BYTES * per).zip(hists.chunks(per)) {
                sc.spawn(move || {
                    for (ob, h) in oc.chunks_mut(ENG_ROW_BYTES).zip(hc) {
                        for (r, &id) in engram_hashes(c, h, ei).iter().enumerate() {
                            let id = id as usize;
                            ob[r * 256..(r + 1) * 256].copy_from_slice(&wb[id * hd..id * hd + 256]);
                            ob[24 * 256 + r * 8..24 * 256 + r * 8 + 8].copy_from_slice(&sb[id * (hd / 32)..id * (hd / 32) + 8]);
                        }
                    }
                });
            }
        });
    }
}

pub fn engram_hashes(c: &Config, hist: &[i64], li: usize) -> Vec<u64> {
    let n = c.engram_ngram;
    let t = hist.len() - 1;
    let mut toks = vec![0i64; n];
    let mut blocked = false;
    for s in 0..n {
        let src = if t >= s { hist[t - s] } else { -1 };
        blocked = blocked || t < s || src == -1;
        toks[s] = if blocked { c.engram_pad } else { src };
    }
    let mult = &c.engram_multipliers[li * n..(li + 1) * n];
    let per_layer = (n - 1) * c.engram_heads;
    let primes = &c.engram_primes[li * per_layer..(li + 1) * per_layer];
    let mut offsets = vec![0u64; per_layer];
    for i in 1..per_layer {
        offsets[i] = offsets[i - 1] + primes[i - 1];
    }
    let mut rolling = toks[0].wrapping_mul(mult[0] as i64);
    let mut out = vec![];
    for i in 1..n {
        rolling ^= toks[i].wrapping_mul(mult[i] as i64);
        for hd in 0..c.engram_heads {
            let col = (i - 1) * c.engram_heads + hd;
            out.push((rolling.rem_euclid(primes[col] as i64)) as u64 + offsets[col]);
        }
    }
    out
}
