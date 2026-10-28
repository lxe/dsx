//! Routed experts on the CPU: out[t] += down(silu(min(gate x, L)) * clamp(up x, -L, L) * weight), L = swiglu_limit.
//! The routing weight scales the SwiGLU output before down (model.py Expert.forward); h is re-quantized
//! to 256-blocks for the down projection.

use crate::numerics::fwht_blocks;
use super::pool::{Ctx, Pool};
use super::qdot::{QAct, QActRaw, QK, dot_rows, dot_rows_sel};
use crate::gguf::GType;
use crate::numerics::silu;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering::*};

pub const SWIGLU_LIMIT: f32 = 10.0;
/// tokens per call (verify batch)
pub const MAX_TOKENS: usize = 16;
const RU: usize = 64; // gate/up rows per work unit (4 units per 256-block of h)
const RD: usize = 64; // down rows per work unit

/// One expert's weights: gate/up [inter][dim] in `gu`, down [dim][inter] in `dn`, rows contiguous.
#[derive(Clone, Copy)]
pub struct ExpertW<'a> {
    pub gu: GType,
    pub gate: &'a [u8],
    pub up: &'a [u8],
    pub dn: GType,
    pub down: &'a [u8],
}

impl ExpertW<'_> {
    pub fn inter(&self, dim: usize) -> usize {
        if let Some(k2) = crate::t16::k2(self.gu) {
            // trellis | suh | svh: dim * inter * k2 / 16 + 2 dim + 2 inter bytes
            return (self.gate.len() - 2 * dim) / (dim * k2 / 16 + 2);
        }
        self.gate.len() / self.gu.row_bytes(dim)
    }
    pub fn bytes(&self) -> usize {
        self.gate.len() + self.up.len() + self.down.len()
    }
}

#[inline]
pub fn swiglu(g: f32, u: f32, w: f32) -> f32 {
    silu(g.min(SWIGLU_LIMIT)) * u.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT) * w
}

/// Single-threaded: x [t][dim], w [t], out [t][dim] +=.
pub fn expert_forward(e: &ExpertW, dim: usize, x: &[f32], t: usize, w: &[f32], out: &mut [f32]) {
    let inter = e.inter(dim);
    let xa = QAct::new(x, t, dim, &[e.gu]);
    let (mut g, mut u) = (vec![0f32; t * inter], vec![0f32; t * inter]);
    dot_rows(e.gu, e.gate, dim, &xa, &mut g);
    dot_rows(e.gu, e.up, dim, &xa, &mut u);
    let h: Vec<f32> = (0..t * inter).map(|i| swiglu(g[i], u[i], w[i / inter])).collect();
    let ha = QAct::new(&h, t, inter, &[e.dn]);
    let mut y = vec![0f32; t * dim];
    dot_rows(e.dn, e.down, inter, &ha, &mut y);
    for (o, y) in out.iter_mut().zip(&y) {
        *o += y;
    }
}

/// One layer's CPU experts on a pool: x quantized across workers, barrier, gate+up rows split across workers
/// (h quantized per 256-block by whoever finishes its last rows), barrier, then down rows (each worker owns
/// its output rows). The caller spins until done, so it should not sit on one of the pool's CPUs.
pub struct CpuMoe {
    pub pool: Pool,
    xa: QAct,
    ha: Vec<QAct>,
    h: Vec<Vec<f32>>,
    toks: Vec<Vec<usize>>,
    cnt: Vec<AtomicU32>,
    next: [Padded; 2],
    /// the experts of this call are stored in the Hadamard-rotated basis: rotate x and h before quantizing them
    pub rot: bool,
    t16: T16Scratch,
    /// forward_t16's schedule (T16Sched::from_env at new)
    pub t16_sched: T16Sched,
}

/// forward_t16's work schedule (bit-exact outputs whichever it is): each expert's last gate / up columns as a guided
/// tail of `tw`-column units, about `tper` per worker (0: none, every unit 4 columns), and down without a barrier
/// (`pipe`: each down unit takes an expert's h blocks as they become ready). Default 2 tail units of 1 column per
/// worker and pipe: on the i7-13700 at 14 threads, 6-7% faster for 1-expert decode calls, 4-5% for 2, 2.4% for 3,
/// 1-2% for 6-expert calls of 1-16 tokens. From DSX_T16_TAIL=TPER[,TW] (0 = none) and DSX_T16_PIPE=0|1.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct T16Sched {
    pub tper: usize,
    pub tw: usize,
    pub pipe: bool,
}

impl T16Sched {
    pub const BARRIER: T16Sched = T16Sched { tper: 0, tw: 1, pipe: false };
    pub fn from_env() -> T16Sched {
        let t: Vec<usize> = std::env::var("DSX_T16_TAIL").unwrap_or_default().split(',').filter_map(|v| v.trim().parse().ok()).collect();
        T16Sched {
            tper: t.first().copied().unwrap_or(2),
            tw: t.get(1).copied().unwrap_or(1).clamp(1, 2),
            pipe: std::env::var("DSX_T16_PIPE").map_or(true, |v| v != "0"),
        }
    }
}

/// DSX_T16_PROF=1 (read once): forward_t16's phase boundaries per worker and around the pool run, in ns since the
/// first mark, for cpubench's small-call breakdown. Marks per worker: 0 job start, 1 phase 1 (x prep) done, 2 barrier
/// passed, 3 phase 2 (gate / up) done, 4 barrier passed, 5 phase 3 (down) done; CALL: 0 forward_t16 entry, 1 pool.run
/// called, 2 pool.run returned.
pub mod t16prof {
    use std::sync::OnceLock;
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    use std::time::Instant;
    pub const MARKS: usize = 6;
    pub static T: [[AtomicU64; MARKS]; 64] = [const { [const { AtomicU64::new(0) }; MARKS] }; 64];
    pub static CALL: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
    pub fn on() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| std::env::var("DSX_T16_PROF").is_ok_and(|v| v == "1"))
    }
    pub fn now() -> u64 {
        static BASE: OnceLock<Instant> = OnceLock::new();
        BASE.get_or_init(Instant::now).elapsed().as_nanos() as u64
    }
    #[inline]
    pub fn mark(id: usize, k: usize) {
        if on() {
            T[id][k].store(now(), Relaxed);
        }
    }
    #[inline]
    pub fn call(k: usize) {
        if on() {
            CALL[k].store(now(), Relaxed);
        }
    }
}

/// T16 calls (t16k): per (expert, token) the int8 activations of gate, up and down, the raw gate / up outputs and h
#[derive(Default)]
struct T16Scratch {
    ag: Vec<super::t16k::Act>,
    au: Vec<super::t16k::Act>,
    ad: Vec<super::t16k::Act>,
    g: Vec<Vec<f32>>,
    u: Vec<Vec<f32>>,
    h: Vec<Vec<f32>>,
    cnt: Vec<AtomicU32>,
    /// per (expert, 256-block of h): 1 once the block's down input is prepared (phase 3 waits on it)
    ready: Vec<AtomicU32>,
    next: [Padded; 3],
}

#[repr(align(128))]
#[derive(Default)]
struct Padded(AtomicUsize);

struct Job<'a> {
    ex: &'a [ExpertW<'a>],
    dim: usize,
    inter: usize,
    x: &'a [f32],
    xa: *const QAct,
    xa_raw: QActRaw,
    ha: *const QAct,
    ha_raw: Vec<QActRaw>,
    h: Vec<*mut f32>,
    toks: &'a [Vec<usize>],
    w: &'a [f32],
    t: usize,
    cnt: &'a [AtomicU32],
    next: &'a [Padded; 2],
    out: *mut f32,
    rot: bool,
}
unsafe impl Sync for Job<'_> {}

impl CpuMoe {
    pub fn new(pool: Pool) -> CpuMoe {
        CpuMoe {
            pool,
            xa: QAct::default(),
            ha: vec![],
            h: vec![],
            toks: vec![],
            cnt: vec![],
            next: Default::default(),
            rot: false,
            t16: T16Scratch::default(),
            t16_sched: T16Sched::from_env(),
        }
    }

    /// x [t][dim]; w [expert][t] routing weights (0 = token not routed there); out [t][dim] +=.
    pub fn forward(&mut self, ex: &[ExpertW], dim: usize, x: &[f32], t: usize, w: &[f32], out: &mut [f32]) {
        assert!(t <= MAX_TOKENS && x.len() == t * dim && w.len() == ex.len() * t && out.len() == t * dim);
        if ex.is_empty() {
            return;
        }
        let inter = ex[0].inter(dim);
        assert!(inter % QK == 0 && dim % RD == 0);
        let ne = ex.len();
        if crate::t16::k2(ex[0].gu).is_some() {
            return self.forward_t16(ex, dim, inter, x, t, w, out);
        }
        let gu: Vec<GType> = ex.iter().map(|e| e.gu).collect();
        self.xa.reset(t, dim, &gu);
        self.toks.resize_with(ne, Vec::new);
        self.ha.resize_with(ne, QAct::default);
        self.h.resize_with(ne, Vec::new);
        for (e, xe) in ex.iter().enumerate() {
            assert_eq!(xe.inter(dim), inter);
            let tk = &mut self.toks[e];
            tk.clear();
            tk.extend((0..t).filter(|&j| w[e * t + j] != 0.0));
            self.ha[e].reset(tk.len(), inter, &[xe.dn]);
            if self.h[e].len() < tk.len() * inter {
                self.h[e].resize(tk.len() * inter, 0.0);
            }
        }
        let nb = inter / QK;
        self.cnt.resize_with(ne * nb, || AtomicU32::new(0));
        for c in &self.cnt[..ne * nb] {
            c.store(0, Relaxed);
        }
        for n in &self.next {
            n.0.store(0, Relaxed);
        }
        let job = Job {
            ex,
            dim,
            inter,
            x,
            xa_raw: self.xa.raw(),
            xa: &self.xa,
            ha: self.ha.as_ptr(),
            ha_raw: self.ha.iter_mut().map(|a| a.raw()).collect(),
            h: self.h.iter_mut().map(|v| v.as_mut_ptr()).collect(),
            toks: &self.toks,
            w,
            t,
            cnt: &self.cnt,
            next: &self.next,
            out: out.as_mut_ptr(),
            rot: self.rot,
        };
        self.pool.run(&|c: &Ctx| job.work(c));
    }
}

impl CpuMoe {
    /// T16 experts (t16k): workers prepare x per (expert, token, 256-group) for gate and up; barrier; gate / up in
    /// units of 4 tile columns and a finer tail (T16Sched), and whoever finishes a 256-block of an
    /// expert's outputs applies the output Hadamard, svh and SwiGLU and prepares that block of h for down; barrier
    /// (none with T16Sched::pipe); down in units of 128 outputs (8 tile columns), each worker owning its outputs.
    #[allow(clippy::too_many_arguments)]
    fn forward_t16(&mut self, ex: &[ExpertW], dim: usize, inter: usize, x: &[f32], t: usize, w: &[f32], out: &mut [f32]) {
        t16prof::call(0);
        let ne = ex.len();
        for e in ex {
            assert_eq!(e.inter(dim), inter);
            assert!(crate::t16::k2(e.gu).is_some() && crate::t16::k2(e.dn).is_some(), "T16 mixed with other expert types");
        }
        let toks: Vec<Vec<usize>> = (0..ne).map(|e| (0..t).filter(|&j| w[e * t + j] != 0.0).collect()).collect();
        let s = &mut self.t16;
        let slots = ne * MAX_TOKENS;
        for v in [&mut s.ag, &mut s.au, &mut s.ad] {
            v.resize_with(slots, Default::default);
        }
        for (e, tk) in toks.iter().enumerate() {
            for j in 0..tk.len() {
                let i = e * MAX_TOKENS + j;
                for (a, k) in [(&mut s.ag[i], dim), (&mut s.au[i], dim), (&mut s.ad[i], inter)] {
                    a.x4.resize(k / 2, 0);
                    a.x8.resize(k / 2, 0);
                    if super::t16k::mul1() {
                        a.x1.resize(k, 0);
                        a.xm1.resize(k, 0);
                    }
                    a.d.resize(k / 256, 0.0);
                    a.s.resize(k / 256, 0);
                }
            }
        }
        for v in [&mut s.g, &mut s.u, &mut s.h] {
            v.resize_with(slots, Vec::new);
            for b in v.iter_mut() {
                b.resize(inter, 0.0);
            }
        }
        let nb = inter / 256;
        s.cnt.resize_with(ne * nb, || AtomicU32::new(0));
        s.ready.resize_with(ne * nb, || AtomicU32::new(0));
        for c in s.cnt[..ne * nb].iter().chain(&s.ready[..ne * nb]) {
            c.store(0, Relaxed);
        }
        for n in &s.next {
            n.0.store(0, Relaxed);
        }
        struct P<T>(*mut T);
        unsafe impl<T> Sync for P<T> {}
        unsafe impl<T> Send for P<T> {}
        let (ag, au, ad) = (P(s.ag.as_mut_ptr()), P(s.au.as_mut_ptr()), P(s.ad.as_mut_ptr()));
        let (gb, ub, hb) = (P(s.g.as_mut_ptr()), P(s.u.as_mut_ptr()), P(s.h.as_mut_ptr()));
        let outp = P(out.as_mut_ptr());
        let (cnt, ready, next) = (&s.cnt, &s.ready, &s.next);
        let groups = dim / 256;
        // gate / up units: 4 tile columns (64 outputs), then (T16Sched) each expert's last columns in units of `tw`, about
        // `tper` per worker: workers finish within one unit of each other, ~20 us a call with 4-column units throughout
        let th = self.pool.threads();
        let cols = inter / 16;
        let T16Sched { tw, tper, pipe } = self.t16_sched;
        let tail = if tper == 0 { 0 } else { ((tper * th * tw).div_ceil(ne).div_ceil(4) * 4).clamp(4, cols) };
        let (big, nbig) = ((cols - tail) / 4, ne * (cols - tail) / 4);
        let ntail = tail / tw;
        let work = |c: &Ctx| unsafe {
            use super::t16k::{gemv_multi, gemv_multi_groups, mul1, post_block, prep_group, scales};
            let (ag, au, ad, gb, ub, hb, outp) = (&ag, &au, &ad, &gb, &ub, &hb, &outp);
            t16prof::mark(c.id, 0);
            // 1. x for gate and up
            let units: Vec<(usize, usize)> = toks.iter().enumerate().flat_map(|(e, tk)| (0..tk.len()).map(move |j| (e, j))).collect();
            loop {
                let i = next[0].0.fetch_add(1, Relaxed);
                if i >= units.len() * groups {
                    break;
                }
                let ((e, j), gr) = (units[i / groups], i % groups);
                let (k2, xr) = (crate::t16::k2(ex[e].gu).unwrap(), &x[toks[e][j] * dim..(toks[e][j] + 1) * dim]);
                let si = e * MAX_TOKENS + j;
                prep_group(xr, scales(ex[e].gate, k2, dim, inter).0, gr, &mut *ag.0.add(si));
                prep_group(xr, scales(ex[e].up, k2, dim, inter).0, gr, &mut *au.0.add(si));
            }
            t16prof::mark(c.id, 1);
            c.barrier();
            t16prof::mark(c.id, 2);
            // 2. gate / up: units 0..nbig of 4 tile columns, then ne * tail of 1 (column c0 .. c0 + wc)
            loop {
                let i = next[1].0.fetch_add(1, Relaxed);
                if i >= nbig + ne * ntail {
                    break;
                }
                let (e, c0, wc) = if i < nbig { (i / big, 4 * (i % big), 4) } else { ((i - nbig) / ntail, cols - tail + tw * ((i - nbig) % ntail), tw) };
                let tk = &toks[e];
                if tk.is_empty() {
                    continue;
                }
                let (k2, k2d) = (crate::t16::k2(ex[e].gu).unwrap(), crate::t16::k2(ex[e].dn).unwrap());
                // all of the expert's tokens per call: each tile decoded once per up to 8 tokens
                let nt = tk.len();
                let si = |j: usize| e * MAX_TOKENS + j.min(nt - 1);
                for (acts, outs, mat) in [(ag, gb, ex[e].gate), (au, ub, ex[e].up)] {
                    let a: [&super::t16k::Act; MAX_TOKENS] = std::array::from_fn(|j| &*acts.0.add(si(j)));
                    let mut o: [&mut [f32]; MAX_TOKENS] =
                        std::array::from_fn(|j| if j < nt { &mut (&mut *outs.0.add(si(j)))[16 * c0..] } else { Default::default() });
                    gemv_multi(mat, k2, dim, inter, &a[..nt], c0, c0 + wc, &mut o[..nt]);
                }
                // the worker completing a 256-output block's 16 columns finishes it
                let b = c0 / 16;
                if cnt[e * nb + b].fetch_add(wc as u32, AcqRel) == 16 - wc as u32 {
                    let (svg, svu) = (scales(ex[e].gate, k2, dim, inter).1, scales(ex[e].up, k2, dim, inter).1);
                    for (j, &tok) in tk.iter().enumerate() {
                        let si = e * MAX_TOKENS + j;
                        let (g, u, h) = (&mut *gb.0.add(si), &mut *ub.0.add(si), &mut *hb.0.add(si));
                        for o in [256 * b, 256 * b + 128] {
                            post_block(&mut g[o..], svg, o);
                            post_block(&mut u[o..], svu, o);
                        }
                        let wt = w[e * t + tok];
                        for r in 256 * b..256 * b + 256 {
                            h[r] = crate::numerics::round_bf16(swiglu(g[r], u[r], wt));
                        }
                        prep_group(h, scales(ex[e].down, k2d, inter, dim).0, b, &mut *ad.0.add(si));
                    }
                    ready[e * nb + b].store(1, Release);
                }
            }
            t16prof::mark(c.id, 3);
            if !pipe {
                c.barrier();
            }
            t16prof::mark(c.id, 4);
            // 3. down, 128 outputs per unit. With T16Sched::pipe there is no barrier before it: a unit takes each expert's
            // input groups (h blocks) in order as they become ready, so it overlaps the end of gate / up; every gate / up
            // unit is claimed before any down unit, so a down unit only waits on blocks in progress
            let rdy = |e: usize, b: usize| ready[e * nb + b].load(Acquire) == 1;
            let mut ys = [[0f32; 128]; MAX_TOKENS];
            loop {
                let i = next[2].0.fetch_add(1, Relaxed);
                if i >= dim / 128 {
                    break;
                }
                for (e, tk) in toks.iter().enumerate() {
                    let k2d = crate::t16::k2(ex[e].dn).unwrap();
                    let svd = scales(ex[e].down, k2d, inter, dim).1;
                    let nt = tk.len();
                    if nt == 0 {
                        continue;
                    }
                    let a: [&super::t16k::Act; MAX_TOKENS] = std::array::from_fn(|j| &*ad.0.add(e * MAX_TOKENS + j.min(nt - 1)));
                    {
                        let mut o: [&mut [f32]; MAX_TOKENS] = std::array::from_fn(|_| Default::default());
                        for (oj, y) in o.iter_mut().zip(ys.iter_mut()).take(nt) {
                            *oj = &mut y[..];
                        }
                        if mul1() {
                            (0..nb).for_each(|b| while !rdy(e, b) { std::hint::spin_loop() });
                            gemv_multi(ex[e].down, k2d, inter, dim, &a[..nt], 8 * i, 8 * i + 8, &mut o[..nt]);
                        } else {
                            let mut g0 = 0;
                            while g0 < nb {
                                while !rdy(e, g0) {
                                    std::hint::spin_loop();
                                }
                                let mut g1 = g0 + 1;
                                while g1 < nb && rdy(e, g1) {
                                    g1 += 1;
                                }
                                gemv_multi_groups(ex[e].down, k2d, inter, dim, &a[..nt], 8 * i, 8 * i + 8, &mut o[..nt], g0, g1);
                                g0 = g1;
                            }
                        }
                    }
                    for (j, &tok) in tk.iter().enumerate() {
                        let y = &mut ys[j];
                        post_block(y, svd, 128 * i);
                        for (r, v) in y.iter().enumerate() {
                            *outp.0.add(tok * dim + 128 * i + r) += v;
                        }
                    }
                }
            }
            t16prof::mark(c.id, 5);
        };
        t16prof::call(1);
        self.pool.run(&work);
        t16prof::call(2);
    }
}

impl Job<'_> {
    fn work(&self, c: &Ctx) {
        let (dim, inter) = (self.dim, self.inter);
        let upe = inter / RU; // units per expert
        let ne = self.ex.len();
        let mut g = [0f32; MAX_TOKENS * RU];
        let mut u = [0f32; MAX_TOKENS * RU];
        // x quantized in parallel, block i by worker i % n
        let xb = dim / QK;
        let mut rb = [0f32; QK];
        for i in (c.id..self.t * xb).step_by(c.n) {
            let blk = &self.x[i * QK..(i + 1) * QK];
            let src: &[f32] = if self.rot {
                rb.copy_from_slice(blk);
                fwht_blocks(&mut rb);
                &rb
            } else {
                blk
            };
            unsafe { self.xa_raw.quantize_block(i / xb, i % xb, src) };
        }
        c.barrier();
        let xa = unsafe { &*self.xa };
        loop {
            let i = self.next[0].0.fetch_add(1, Relaxed);
            if i >= ne * upe {
                break;
            }
            let (e, r0) = (i / upe, i % upe * RU);
            let (ex, tk) = (&self.ex[e], &self.toks[e]);
            if tk.is_empty() {
                continue;
            }
            let rb = ex.gu.row_bytes(dim);
            let rows = r0 * rb..(r0 + RU) * rb;
            dot_rows_sel(ex.gu, &ex.gate[rows.clone()], dim, xa, tk, &mut g, RU);
            dot_rows_sel(ex.gu, &ex.up[rows], dim, xa, tk, &mut u, RU);
            let h = self.h[e];
            for (j, &tok) in tk.iter().enumerate() {
                let wt = self.w[e * self.t + tok];
                for k in 0..RU {
                    unsafe { *h.add(j * inter + r0 + k) = swiglu(g[j * RU + k], u[j * RU + k], wt) };
                }
            }
            let b = r0 / QK;
            if self.cnt[e * (inter / QK) + b].fetch_add(1, AcqRel) as usize == QK / RU - 1 {
                for j in 0..tk.len() {
                    unsafe {
                        let hb = std::slice::from_raw_parts_mut(h.add(j * inter + b * QK), QK);
                        if self.rot {
                            fwht_blocks(hb);
                        }
                        self.ha_raw[e].quantize_block(j, b, hb);
                    }
                }
            }
        }
        c.barrier();
        let ha = unsafe { std::slice::from_raw_parts(self.ha, ne) };
        let y = &mut g;
        loop {
            let i = self.next[1].0.fetch_add(1, Relaxed);
            if i >= dim / RD {
                break;
            }
            let r0 = i * RD;
            for (e, ex) in self.ex.iter().enumerate() {
                let tk = &self.toks[e];
                if tk.is_empty() {
                    continue;
                }
                let rb = ex.dn.row_bytes(inter);
                let all: [usize; MAX_TOKENS] = std::array::from_fn(|j| j);
                dot_rows_sel(
                    ex.dn,
                    &ex.down[r0 * rb..(r0 + RD) * rb],
                    inter,
                    &ha[e],
                    &all[..tk.len()],
                    y,
                    RD,
                );
                for (j, &tok) in tk.iter().enumerate() {
                    for k in 0..RD {
                        unsafe { *self.out.add(tok * dim + r0 + k) += y[j * RD + k] };
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::pool::Pool;
    use super::super::qdot::tests::{Rng, rand_rows};
    use super::*;
    use crate::formats::dequant_row;
    use std::time::Duration;

    fn f32_matvec(ty: GType, rows: &[u8], cols: usize, x: &[f32]) -> Vec<f32> {
        let mut w = vec![0f32; cols];
        rows.chunks_exact(ty.row_bytes(cols))
            .map(|r| {
                dequant_row(ty, r, &mut w);
                w.iter().zip(x).map(|(&a, &b)| a as f64 * b as f64).sum::<f64>() as f32
            })
            .collect()
    }

    fn f32_expert(e: &ExpertW, dim: usize, x: &[f32], w: f32) -> Vec<f32> {
        let g = f32_matvec(e.gu, e.gate, dim, x);
        let u = f32_matvec(e.gu, e.up, dim, x);
        let h: Vec<f32> = g.iter().zip(&u).map(|(&g, &u)| swiglu(g, u, w)).collect();
        f32_matvec(e.dn, e.down, e.inter(dim), &h)
    }

    /// T16 experts over a 16-token window with sparse routing (prefill's CPU share): equal to running each token
    /// alone, and bit-identical on 40 reruns with 14 threads while other threads compete for the cores
    #[test]
    fn t16_window_reproducible_under_load() {
        let (dim, inter, ne, t) = (2048usize, 1024usize, 64usize, 16usize);
        let mut s = 0x1234_5678_9abc_def0u64;
        let mut rnd = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let mut mat = |ty: GType, k: usize, n: usize| -> Vec<u8> {
            let tb = crate::t16::trellis_bytes(crate::t16::k2(ty).unwrap(), k, n);
            let mut v: Vec<u8> = (0..crate::t16::matrix_bytes(ty, k, n)).map(|_| rnd() as u8).collect();
            for c in v[tb..].chunks_exact_mut(2) {
                c.copy_from_slice(&half::f16::from_f32(0.02).to_le_bytes());
            }
            v
        };
        let mats: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> = (0..ne).map(|_| (mat(GType::T16K15, dim, inter), mat(GType::T16K15, dim, inter), mat(GType::T16K2, inter, dim))).collect();
        let ex: Vec<ExpertW> = mats.iter().map(|m| ExpertW { gu: GType::T16K15, gate: &m.0, up: &m.1, dn: GType::T16K2, down: &m.2 }).collect();
        let mut r = 0x9e37_79b9u64;
        let mut rnd2 = || {
            r ^= r << 13;
            r ^= r >> 7;
            r ^= r << 17;
            r
        };
        let x: Vec<f32> = (0..t * dim).map(|_| (rnd2() % 2001) as f32 / 1000.0 - 1.0).collect();
        // each expert gets 1-3 of the tokens
        let mut w = vec![0f32; ne * t];
        for e in 0..ne {
            for _ in 0..1 + rnd2() % 3 {
                w[e * t + (rnd2() % t as u64) as usize] = 0.1 + (rnd2() % 100) as f32 / 200.0;
            }
        }
        let mut cm = CpuMoe::new(Pool::new(&(0..14).collect::<Vec<_>>(), Duration::from_millis(5)));
        let mut win = vec![0f32; t * dim];
        cm.forward(&ex, dim, &x, t, &w, &mut win);
        // competing load on the same cores (prefill's host thread and rayon helpers), so pool threads get preempted
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let hogs: Vec<_> = (0..20)
            .map(|_| {
                let st = stop.clone();
                std::thread::spawn(move || {
                    let mut v = 0u64;
                    while !st.load(Relaxed) {
                        v = v.wrapping_mul(6364136223846793005).wrapping_add(1);
                        std::hint::black_box(v);
                    }
                })
            })
            .collect();
        let _guard = scopeguard_stop(stop.clone());
        for rep in 0..40 {
            let mut win2 = vec![0f32; t * dim];
            cm.forward(&ex, dim, &x, t, &w, &mut win2);
            let diff = win.iter().zip(&win2).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
            assert_eq!(diff, 0, "window forward not reproducible: rerun {rep}, {diff} outputs differ");
        }
        stop.store(true, Relaxed);
        for h in hogs {
            h.join().unwrap();
        }
        let mut single = vec![0f32; t * dim];
        for j in 0..t {
            let wj: Vec<f32> = (0..ne).map(|e| w[e * t + j]).collect();
            cm.forward(&ex, dim, &x[j * dim..(j + 1) * dim], 1, &wj, &mut single[j * dim..(j + 1) * dim]);
        }
        let e = rel(&win, &single);
        println!("window vs single tokens: rel {e:.2e}");
        assert!(e < 1e-5, "window vs single tokens: rel {e}");
    }

    /// stops the hog threads if an assertion unwinds
    fn scopeguard_stop(s: std::sync::Arc<std::sync::atomic::AtomicBool>) -> impl Drop {
        struct G(std::sync::Arc<std::sync::atomic::AtomicBool>);
        impl Drop for G {
            fn drop(&mut self) {
                self.0.store(true, Relaxed);
            }
        }
        G(s)
    }

    fn rel(a: &[f32], b: &[f32]) -> f64 {
        let num: f64 = a.iter().zip(b).map(|(&x, &y)| ((x - y) as f64).powi(2)).sum();
        let den: f64 = b.iter().map(|&y| (y as f64).powi(2)).sum();
        (num / den).sqrt()
    }

    struct Ex {
        gu: GType,
        dn: GType,
        gate: Vec<u8>,
        up: Vec<u8>,
        down: Vec<u8>,
    }
    impl Ex {
        fn w(&self) -> ExpertW<'_> {
            ExpertW {
                gu: self.gu,
                gate: &self.gate,
                up: &self.up,
                dn: self.dn,
                down: &self.down,
            }
        }
    }

    // reduced inter keeps the f32 reference quick; dim is the model's
    const DIM: usize = 5120;
    const INTER: usize = 768;

    fn experts(rng: &mut Rng) -> Vec<Ex> {
        use GType::*;
        [
            (Q2_0, Q3K),
            (Ptq1_0, Q2_0),
            (Q3K, Ptq1_0),
            (Q2_0, Q2_0),
            (Ptq1_0, Ptq1_0),
            (Q8_0, Q8_0),
            (Iq1S, Iq2Xxs),
            (Iq2Xxs, Iq2Xs),
            (Iq2S, Iq1S),
            (Iq2Xs, Iq2S),
        ]
        .iter()
        .map(|&(gu, dn)| Ex {
            gu,
            dn,
            gate: rand_rows(gu, INTER, DIM, rng),
            up: rand_rows(gu, INTER, DIM, rng),
            down: rand_rows(dn, DIM, INTER, rng),
        })
        .collect()
    }

    /// f32 weights, activations rounded through the same int8 quantization at both stages
    fn mirrored_expert(e: &ExpertW, dim: usize, x: &[f32], w: f32, qx: bool, qh: bool) -> Vec<f32> {
        let xq = if qx {
            QAct::new(x, 1, dim, &[e.gu]).dequant(0)
        } else {
            x.to_vec()
        };
        let g = f32_matvec(e.gu, e.gate, dim, &xq);
        let u = f32_matvec(e.gu, e.up, dim, &xq);
        let h: Vec<f32> = g.iter().zip(&u).map(|(&g, &u)| swiglu(g, u, w)).collect();
        let hq = if qh {
            QAct::new(&h, 1, h.len(), &[e.dn]).dequant(0)
        } else {
            h
        };
        f32_matvec(e.dn, e.down, e.inter(dim), &hq)
    }

    #[test]
    fn expert_vs_f32() {
        let mut rng = Rng(42);
        let exs = experts(&mut rng);
        let t = 3;
        // |gate x| ~ 1-2 with these random weights, like the model's
        let x: Vec<f32> = (0..t * DIM).map(|_| 0.3 * rng.normal()).collect();
        let w = [0.7f32, 1.3, 0.2];
        for ex in &exs {
            let e = ex.w();
            let mut out = vec![0f32; t * DIM];
            expert_forward(&e, DIM, &x, t, &w, &mut out);
            for j in 0..t {
                let xj = &x[j * DIM..(j + 1) * DIM];
                let o = &out[j * DIM..(j + 1) * DIM];
                let r = f32_expert(&e, DIM, xj, w[j]);
                let (em, ef) = (rel(o, &mirrored_expert(&e, DIM, xj, w[j], true, true)), rel(o, &r));
                if j == 0 {
                    let ex = rel(&mirrored_expert(&e, DIM, xj, w[j], true, false), &r);
                    let eh = rel(&mirrored_expert(&e, DIM, xj, w[j], false, true), &r);
                    println!(
                        "expert {:?}/{:?}: vs int8-mirrored f32 {em:.2e}, vs pure f32 {ef:.2e} (x quant alone {ex:.2e}, h quant alone {eh:.2e})",
                        e.gu, e.dn
                    );
                }
                assert!(em < 2e-3, "{:?}/{:?} token {j}: mirrored {em}", e.gu, e.dn);
                assert!(ef < 5e-2, "{:?}/{:?} token {j}: f32 {ef}", e.gu, e.dn);
            }
        }
    }

    #[test]
    fn pooled_matches_single() {
        let mut rng = Rng(5);
        let exs = experts(&mut rng);
        let ew: Vec<ExpertW> = exs.iter().map(|e| e.w()).collect();
        let mut moe = CpuMoe::new(Pool::new(&[0, 1, 2, 3, 4, 5], Duration::from_millis(1)));
        for t in [1usize, 4, 11] {
            let x: Vec<f32> = (0..t * DIM).map(|_| rng.normal()).collect();
            // sparse routing: some tokens skip some experts
            let w: Vec<f32> = (0..ew.len() * t)
                .map(|i| if i % 3 == 1 { 0.0 } else { 0.1 + rng.f32() })
                .collect();
            let mut want = vec![0.5f32; t * DIM];
            for (e, ex) in ew.iter().enumerate() {
                for j in 0..t {
                    let wt = w[e * t + j];
                    if wt != 0.0 {
                        expert_forward(
                            ex,
                            DIM,
                            &x[j * DIM..(j + 1) * DIM],
                            1,
                            &[wt],
                            &mut want[j * DIM..(j + 1) * DIM],
                        );
                    }
                }
            }
            for _ in 0..3 {
                let mut got = vec![0.5f32; t * DIM];
                moe.forward(&ew, DIM, &x, t, &w, &mut got);
                let err = rel(&got, &want);
                assert!(err < 1e-5, "T {t}: pooled vs single {err}");
            }
        }
    }

    /// T16 experts on the pool against core's f32 reference (t16::forward), sparse routing, several token counts
    #[test]
    fn t16_pooled_matches_reference() {
        use crate::t16;
        let (dim, inter) = (1024usize, 512usize);
        let mut seed = 0x1357_9bdf_2468_ace0u64;
        let mut r = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let mut mat = |ty: GType, k: usize, n: usize| -> Vec<u8> {
            let k2 = t16::k2(ty).unwrap();
            let tb = t16::trellis_bytes(k2, k, n);
            let mut m: Vec<u8> = (0..t16::matrix_bytes(ty, k, n)).map(|_| r() as u8).collect();
            for c in m[tb..].chunks_exact_mut(2) {
                let v = 0.02 * (0.5 + (r() % 1000) as f32 / 1000.0) * if r() % 2 == 0 { 1.0 } else { -1.0 };
                c.copy_from_slice(&half::f16::from_f32(v).to_le_bytes());
            }
            m
        };
        let (gu, dn) = (GType::T16K15, GType::T16K25);
        let mats: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> = (0..4).map(|_| (mat(gu, dim, inter), mat(gu, dim, inter), mat(dn, inter, dim))).collect();
        let ew: Vec<ExpertW> = mats.iter().map(|m| ExpertW { gu, gate: &m.0, up: &m.1, dn, down: &m.2 }).collect();
        assert_eq!(ew[0].inter(dim), inter);
        let mut moe = CpuMoe::new(Pool::new(&[0, 1, 2, 3, 4, 5], Duration::from_millis(1)));
        let mut seed2 = 7u64;
        let mut f = || {
            seed2 = seed2.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed2 >> 33) % 2001) as f32 / 1000.0 - 1.0
        };
        for t in [1usize, 3] {
            let x: Vec<f32> = (0..t * dim).map(|_| f()).collect();
            let w: Vec<f32> = (0..ew.len() * t).map(|i| if i % 3 == 1 { 0.0 } else { 0.2 + 0.1 * (i % 5) as f32 }).collect();
            // outputs are ~1e-7 here: start from 0, an offset would drown them in f32 rounding
            let mut want = vec![0f32; t * dim];
            for (e, m) in mats.iter().enumerate() {
                for j in 0..t {
                    let wt = w[e * t + j];
                    if wt == 0.0 {
                        continue;
                    }
                    let xr = &x[j * dim..(j + 1) * dim];
                    let (g, u) = (t16::forward(&m.0, gu, dim, inter, xr), t16::forward(&m.1, gu, dim, inter, xr));
                    let h: Vec<f32> = g.iter().zip(&u).map(|(&g, &u)| swiglu(g, u, wt)).collect();
                    let y = t16::forward(&m.2, dn, inter, dim, &h);
                    for (o, v) in want[j * dim..].iter_mut().zip(&y) {
                        *o += v;
                    }
                }
            }
            let mut got = vec![0f32; t * dim];
            moe.forward(&ew, dim, &x, t, &w, &mut got);
            let err = rel(&got, &want);
            println!("T16 pooled T {t}: rel err {err:.2e} (int8 activations)");
            assert!(err < 0.03, "T {t}: T16 pooled vs reference {err}");
        }
    }

    #[test]
    fn t16_window_matches_single_tokens() {
        use crate::t16;
        let (dim, inter) = (1024usize, 512usize);
        let mut seed = 0x1357_9bdf_2468_ace0u64;
        let mut r = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let mut mat = |ty: GType, k: usize, n: usize| -> Vec<u8> {
            let k2 = t16::k2(ty).unwrap();
            let tb = t16::trellis_bytes(k2, k, n);
            let mut m: Vec<u8> = (0..t16::matrix_bytes(ty, k, n)).map(|_| r() as u8).collect();
            for c in m[tb..].chunks_exact_mut(2) {
                let v = 0.02 * (0.5 + (r() % 1000) as f32 / 1000.0) * if r() % 2 == 0 { 1.0 } else { -1.0 };
                c.copy_from_slice(&half::f16::from_f32(v).to_le_bytes());
            }
            m
        };
        let mut moe = CpuMoe::new(Pool::new(&[0, 1, 2, 3, 4, 5], Duration::from_millis(1)));
        let mut seed2 = 7u64;
        let mut f = || {
            seed2 = seed2.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed2 >> 33) % 2001) as f32 / 1000.0 - 1.0
        };
        // a window of t tokens (experts with 1..=t of them, so every chunk size of gemv_multi) equals t one-token calls
        for (t, gu, dn) in [(16usize, GType::T16K15, GType::T16K25), (7, GType::T16K2, GType::T16K2), (5, GType::T16K25, GType::T16K15)] {
            let mats: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> = (0..4).map(|_| (mat(gu, dim, inter), mat(gu, dim, inter), mat(dn, inter, dim))).collect();
            let ew: Vec<ExpertW> = mats.iter().map(|m| ExpertW { gu, gate: &m.0, up: &m.1, dn, down: &m.2 }).collect();
            let x: Vec<f32> = (0..t * dim).map(|_| f()).collect();
            // expert e takes every (e + 1)-th token from token e % 2: 16, 8, 6 and 4 tokens at t = 16
            let w: Vec<f32> = (0..ew.len() * t).map(|i| {
                let (e, j) = (i / t, i % t);
                if j >= e % 2 && (j - e % 2) % (e + 1) == 0 { 0.2 + 0.1 * (i % 5) as f32 } else { 0.0 }
            }).collect();
            let mut got = vec![0f32; t * dim];
            moe.forward(&ew, dim, &x, t, &w, &mut got);
            for j in 0..t {
                let wj: Vec<f32> = (0..ew.len()).map(|e| w[e * t + j]).collect();
                let mut one = vec![0f32; dim];
                moe.forward(&ew, dim, &x[j * dim..(j + 1) * dim], 1, &wj, &mut one);
                assert!(one.iter().zip(&got[j * dim..(j + 1) * dim]).all(|(a, b)| a.to_bits() == b.to_bits()), "t {t} {gu:?}/{dn:?}: token {j} differs from its one-token call");
            }
        }
    }

    #[test]
    fn t16_schedules_match() {
        use crate::t16;
        let (dim, inter) = (1024usize, 512usize);
        let mut seed = 0x1357_9bdf_2468_ace0u64;
        let mut r = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let mut mat = |ty: GType, k: usize, n: usize| -> Vec<u8> {
            let k2 = t16::k2(ty).unwrap();
            let tb = t16::trellis_bytes(k2, k, n);
            let mut m: Vec<u8> = (0..t16::matrix_bytes(ty, k, n)).map(|_| r() as u8).collect();
            for c in m[tb..].chunks_exact_mut(2) {
                let v = 0.02 * (0.5 + (r() % 1000) as f32 / 1000.0) * if r() % 2 == 0 { 1.0 } else { -1.0 };
                c.copy_from_slice(&half::f16::from_f32(v).to_le_bytes());
            }
            m
        };
        let mut moe = CpuMoe::new(Pool::new(&[0, 1, 2, 3, 4, 5], Duration::from_millis(1)));
        let mut seed2 = 7u64;
        let mut f = || {
            seed2 = seed2.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed2 >> 33) % 2001) as f32 / 1000.0 - 1.0
        };
        // every schedule gives the barrier schedule's outputs bit for bit (1-6 experts of 1 token, and windows)
        let mut base = CpuMoe::new(Pool::new(&[0, 1, 2, 3, 4, 5], Duration::from_millis(1)));
        base.t16_sched = T16Sched::BARRIER;
        for (t, ne, gu, dn) in [(1usize, 1usize, GType::T16K15, GType::T16K2), (1, 3, GType::T16K2, GType::T16K25), (1, 6, GType::T16K15, GType::T16K25), (5, 4, GType::T16K25, GType::T16K15)] {
            let mats: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> = (0..ne).map(|_| (mat(gu, dim, inter), mat(gu, dim, inter), mat(dn, inter, dim))).collect();
            let ew: Vec<ExpertW> = mats.iter().map(|m| ExpertW { gu, gate: &m.0, up: &m.1, dn, down: &m.2 }).collect();
            let x: Vec<f32> = (0..t * dim).map(|_| f()).collect();
            let w: Vec<f32> = (0..ne * t).map(|i| if i % 4 == 3 && t > 1 { 0.0 } else { 0.2 + 0.1 * (i % 5) as f32 }).collect();
            let mut want = vec![0f32; t * dim];
            base.forward(&ew, dim, &x, t, &w, &mut want);
            for sched in [T16Sched { tper: 2, tw: 1, pipe: true }, T16Sched { tper: 1, tw: 2, pipe: false }, T16Sched { tper: 0, tw: 1, pipe: true }, T16Sched { tper: 3, tw: 1, pipe: true }] {
                moe.t16_sched = sched;
                let mut got = vec![0f32; t * dim];
                moe.forward(&ew, dim, &x, t, &w, &mut got);
                assert!(got.iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits()), "{sched:?}, t {t} ne {ne}: differs from the barrier schedule");
            }
        }
    }
}
