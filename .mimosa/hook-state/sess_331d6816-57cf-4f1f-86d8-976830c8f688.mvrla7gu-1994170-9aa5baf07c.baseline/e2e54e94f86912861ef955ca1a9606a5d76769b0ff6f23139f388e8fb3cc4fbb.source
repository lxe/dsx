//! The prompt path: chunks of tokens through all 40 layers with many-token kernels (kernels/prefill.h), leaving
//! the decode state (window rings, compressed caches, indexer keys, compressor state, position) as if every token
//! had been decoded, and returning the greedy next token.

use crate::decode::*;
pub(crate) static BGRAPH_OVERRIDE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);
pub(crate) static GRAPH_EPOCH: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
pub(crate) static MERGE_OUT_OVERRIDE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);
use crate::hip::{Arg, Buf, Gpu};
use anyhow::Result;

/// tokens per chunk, and query rows per indexer sub-batch
pub const TMAX: usize = 2048;
const QB: usize = 32;

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct MmJob {
    w: u64,
    sc: u64,
    woff: i64,
    soff: i64,
    mul: f32,
    fmt: i32,
    n: i32,
    k: i32,
    xq: u64,
    xs: u64,
    toks: u64,
    tok0: i32,
    ntok: i32,
    y: u64,
    ldy: i32,
    yatomic: i32,
    yrow: u64,
    ldx: i32,
    pad: i32,
    xb: u64,
}

pub struct PfBufs {
    toks: Buf,
    h: Buf,
    y: Buf,
    a: Buf,
    mix_a: Buf,
    mix_f: Buf,
    xss_a: Buf,
    xss_f: Buf,
    xr_a: Buf,
    xr_f: Buf,
    aq: Buf,
    as_: Buf,
    ab: Buf,
    aqb: Buf,
    hqb: Buf,
    shqb: Buf,
    eqb: Buf,
    oqb: Buf,
    qraw: Buf,
    kvraw: Buf,
    ckv: Buf,
    cg: Buf,
    iw: Buf,
    q: Buf,
    iq: Buf,
    lin: Buf,
    score: Buf,
    bsc: Buf,
    kb: Buf,
    keep: Buf,
    nkeep: Buf,
    cidx: Buf,
    ncomp: Buf,
    oq: Buf,
    os: Buf,
    r: Buf,
    logits: Buf,
    sel: Buf,
    wts: Buf,
    counts: Buf,
    /// device pointers of the layers' window rings (commit)
    ringtab: Buf,
    pin_in: *mut u8,
    off: Buf,
    fill: Buf,
    tok: Buf,
    wrow: Buf,
    iota: Buf,
    gg: Buf,
    uu: Buf,
    hq: Buf,
    hs: Buf,
    sg: Buf,
    su: Buf,
    shq: Buf,
    shs: Buf,
    erows: Buf,
    eq: Buf,
    es: Buf,
    ekv: Buf,
    esum: Buf,
    mixd: Buf,
    vlin: Buf,
    vnext: Buf,
    /// verify head: final-norm outputs as bf16 hi/lo rows, logits per position
    vhh: Buf,
    vhl: Buf,
    vlog: Buf,
    /// DSpark main_hidden rows [t][3][D] (f32, bf16 values)
    pub(crate) mh: Buf,
    dyn_: Buf,
    /// device-built expert jobs of prompt chunks (bgroup, up to 12288 pairs)
    gdyn: Buf,
    jobs: Buf,
    blk: Buf,
    /// pinned staging ring for job tables (async uploads; the device side uses the same offsets)
    pin: *mut u8,
    ring: std::cell::Cell<usize>,
    max_end: usize,
    /// batched decode step in progress: dense GEMMs on bgemv_d (DSXA_BMM=1: the prompt-path mm)
    bmode: std::cell::Cell<bool>,
}

const RING: usize = 192 << 20;
/// pinned input staging of the graph-replayed passes: Engram rows, then token ids at this offset
const PIN_TOKS: usize = 16 * 2 * ER;

impl PfBufs {
    /// (main_hidden, FP8 codes, scales, f32 rows, kv rows, window rows) for the DSpark observe step
    pub(crate) fn dspark_bufs(&self) -> (u64, u64, u64, u64, u64, u64) {
        (self.mh.ptr, self.aq.ptr, self.as_.ptr, self.y.ptr, self.kvraw.ptr, self.lin.ptr)
    }
}

const ER: usize = 24 * 256 + 24 * 8;

impl PfBufs {
    fn new(g: &Gpu, max_ctx: usize) -> Result<PfBufs> {
        let t = TMAX;
        let f = |n: usize| g.alloc_zeroed(n * 4);
        let b = |n: usize| g.alloc_zeroed(n);
        let iota: Vec<i32> = (0..(t * 6) as i32).collect();
        Ok(PfBufs {
            toks: f(t)?,
            h: f(t * HC * D)?,
            y: f(t * D)?,
            a: f(t * D)?,
            mix_a: f(t * 80 * 32)?,
            mix_f: f(t * 80 * 32)?,
            xss_a: f(t * 80)?,
            xss_f: f(t * 80)?,
            xr_a: f(t * D)?,
            xr_f: f(t * D)?,
            aq: b(t * 8192)?,
            as_: f(t * 256)?,
            ab: b(t * D * 2)?,
            aqb: b(t * 8192 * 2)?,
            hqb: b(t * 6 * INTER * 2)?,
            shqb: b(t * INTER * 2)?,
            eqb: b(t * 6144 * 2)?,
            oqb: b(t * NH * HD * 2)?,
            qraw: f(t * QL)?,
            kvraw: f(t * HD)?,
            ckv: f(t * HD)?,
            cg: f(t * HD)?,
            iw: f(t * IH)?,
            q: f(t * NH * HD)?,
            iq: f(t * IH * ID)?,
            lin: b((WIN + t) * HD * 2)?,
            score: f(QB * (max_ctx + 64))?,
            bsc: f(QB * (max_ctx / 8 + 64))?,
            kb: f(QB * 2048)?,
            keep: f(t * 16384)?,
            nkeep: f(t)?,
            cidx: f(t * 512)?,
            ncomp: f(t)?,
            oq: b(t * NH * HD)?,
            os: f(t * NH * HD / 32)?,
            r: f(t * 8192)?,
            logits: f(t * NE)?,
            sel: f(t * 6)?,
            wts: f(t * 6)?,
            counts: f(NE)?,
            ringtab: f(128)?,
            pin_in: g.host_alloc(PIN_TOKS + 64 * 4)?,
            off: f(NE)?,
            fill: f(NE)?,
            tok: f(t * 6)?,
            wrow: f(t * 6)?,
            iota: g.upload(&iota)?,
            gg: f(t * 6 * INTER)?,
            uu: f(t * 6 * INTER)?,
            hq: b(t * 6 * INTER)?,
            hs: f(t * 6 * INTER / 32)?,
            sg: f(t * INTER)?,
            su: f(t * INTER)?,
            shq: b(t * INTER)?,
            shs: f(t * INTER / 32)?,
            erows: b(t * 2 * ER)?,
            eq: b(t * 6144)?,
            es: f(t * 192)?,
            ekv: f(t * 25600)?,
            esum: f(t * 16)?,
            mixd: f(t * 48)?,
            vlin: b(40 * 16 * HD * 2)?,
            vnext: f(16)?,
            vhh: b(16 * D * 2)?,
            vhl: b(16 * D * 2)?,
            vlog: f(16 * VOCAB)?,
            mh: f(t * 3 * D)?,
            dyn_: f(3 * 48 * 40)?,
            gdyn: f(3 * (NE + 12288 / 16) * (8 + 2 * 64))?,
            jobs: g.alloc(RING)?,
            blk: f(1)?,
            pin: g.host_alloc(RING)?,
            ring: std::cell::Cell::new(0),
            max_end: max_ctx,
            bmode: std::cell::Cell::new(false),
        })
    }
}

/// Self-test hook: one mm job over `ntok` token rows (activation codes/scales on the device), output [ntok][n]
#[allow(clippy::too_many_arguments)]
pub fn test_mm(g: &Gpu, fmt: i32, w: u64, sc: u64, woff: i64, soff: i64, n: usize, k: usize, xq: u64, xs: u64, xb: u64, ntok: usize, y: u64, cb: (u32, u32, u32)) -> Result<()> {
    let j = MmJob { w, sc, woff, soff, mul: 1.0, fmt, n: n as i32, k: k as i32, xq, xs, ntok: ntok as i32, y, ldy: n as i32, ldx: k as i32, xb, ..Default::default() };
    let mut blk = vec![];
    for tb in 0..ntok.div_ceil(128) {
        for rb in 0..n.div_ceil(64) {
            blk.push(((rb as i32) << 8) | tb as i32);
        }
    }
    let jb = g.upload(unsafe { std::slice::from_raw_parts(&j as *const MmJob as *const u8, std::mem::size_of::<MmJob>()) })?;
    let bb = g.upload(&blk)?;
    let kern = if xb != 0 && fmt != 4 || std::env::var_os("TEST_MM2").is_some() { "mm2" } else { "mm" };
    g.launch(kern, (blk.len() as u32, 1, 1), 256, 0, &[Arg::P(jb.ptr), Arg::P(bb.ptr), Arg::I(cb.0 as i32), Arg::I(cb.1 as i32), Arg::I(cb.2 as i32)])?;
    g.sync()
}

impl MmJob {
    fn dense(m: &crate::load::DMat, fmt: i32, xq: u64, xs: u64, ldx: usize, y: u64, ldy: usize) -> MmJob {
        MmJob { w: m.w.ptr, sc: m.sc.as_ref().map(|b| b.ptr).unwrap_or(0), fmt, n: m.n as i32, k: m.k as i32, xq, xs, y, ldy: ldy as i32, ldx: ldx as i32, mul: 1.0, ..Default::default() }
    }
}

impl Decoder {
    /// Run a batch of matrix jobs (each over `ntok` tokens) through the `mm` kernel.
    fn mm(&self, jobs: &[MmJob]) -> Result<()> {
        self.mm_l("mm", jobs)
    }

    fn mm_l(&self, label: &str, jobs: &[MmJob]) -> Result<()> {
        self.g.set_label(Some(label));
        let r = self.mm_inner(jobs);
        self.g.set_label(None);
        r
    }

    fn mm_inner(&self, jobs: &[MmJob]) -> Result<()> {
        let pf = self.pf.as_ref().unwrap();
        // mm2 multiplies bf16 copies of the FP8 activations: map each job's code buffer to its bf16 twin
        let mm2 = std::env::var("DSXA_MM1").is_err();
        let mut jobs = jobs.to_vec();
        if mm2 {
            for j in jobs.iter_mut() {
                if j.fmt == 4 {
                    continue;
                }
                let pairs = [(&pf.aq, &pf.aqb), (&pf.hq, &pf.hqb), (&pf.shq, &pf.shqb), (&pf.eq, &pf.eqb), (&pf.oq, &pf.oqb)];
                let mut found = false;
                for (q, b) in pairs {
                    if j.xq >= q.ptr && j.xq < q.ptr + q.bytes as u64 {
                        j.xb = b.ptr + (j.xq - q.ptr) * 2;
                        found = true;
                    }
                }
                anyhow::ensure!(found, "mm2: no bf16 activation for a job");
            }
        }
        let jobs = &jobs[..];
        let mut blk: Vec<i32> = vec![];
        for (ji, j) in jobs.iter().enumerate() {
            let rbs = (j.n as usize).div_ceil(64);
            let tbs = (j.ntok as usize).div_ceil(128);
            for tb in 0..tbs {
                for rb in 0..rbs {
                    debug_assert!(rb < 4096 && tb < 256);
                    blk.push(((ji as i32) << 20) | ((rb as i32) << 8) | tb as i32);
                }
            }
        }
        if blk.is_empty() {
            return Ok(());
        }
        assert!(jobs.len() <= 1024);
        let g = &self.g;
        let jb: Vec<u8> = jobs.iter().flat_map(|j| unsafe { std::slice::from_raw_parts(j as *const MmJob as *const u8, std::mem::size_of::<MmJob>()).to_vec() }).collect();
        // pinned ring: truly async copies; regions are reused only after the chunk's final sync
        let need = jb.len().div_ceil(256) * 256 + blk.len() * 4;
        let mut o = pf.ring.get();
        if o + need > RING {
            g.sync()?;
            o = 0;
        }
        let (jo, bo) = (o, o + jb.len().div_ceil(256) * 256);
        unsafe {
            std::ptr::copy_nonoverlapping(jb.as_ptr(), pf.pin.add(jo), jb.len());
            std::ptr::copy_nonoverlapping(blk.as_ptr() as *const u8, pf.pin.add(bo), blk.len() * 4);
        }
        g.htod_async(pf.jobs.at(jo), unsafe { pf.pin.add(jo) }, jb.len())?;
        g.htod_async(pf.jobs.at(bo), unsafe { pf.pin.add(bo) }, blk.len() * 4)?;
        pf.ring.set(o + need);
        g.launch(if mm2 { "mm2" } else { "mm" }, (blk.len() as u32, 1, 1), 256, 0, &[Arg::P(pf.jobs.at(jo)), Arg::P(pf.jobs.at(bo)), Arg::I(self.cb3[0] as i32), Arg::I(self.cb3[1] as i32), Arg::I(self.cb2 as i32)])
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prep(&self, mode: i32, src: u64, src2: u64, lds: usize, nw: u64, part: u64, wt: u64, n: usize, rows: usize, oq: u64, os: u64, ob: u64, obq: u64) -> Result<()> {
        self.prep_z(mode, src, src2, lds, nw, part, wt, n, rows, oq, os, ob, obq, &[])
    }

    /// prep_t that also clears `zero` (pointer, floats) ranges: the outputs of the k-split products that follow it,
    /// which then skip their own zeroing launch (Decoder::pz)
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prep_z(&self, mode: i32, src: u64, src2: u64, lds: usize, nw: u64, part: u64, wt: u64, n: usize, rows: usize, oq: u64, os: u64, ob: u64, obq: u64, zero: &[(u64, usize)]) -> Result<()> {
        let mut zl = vec![0u8; 72];
        for (k, &(p, cnt)) in zero.iter().take(6).enumerate() {
            zl[k * 8..k * 8 + 8].copy_from_slice(&p.to_le_bytes());
            zl[48 + k * 4..48 + k * 4 + 4].copy_from_slice(&(cnt as i32).to_le_bytes());
        }
        self.g.launch(
            "prep_t",
            (rows as u32, n.div_ceil(1024) as u32, 1),
            256,
            0,
            &[Arg::I(mode), Arg::P(src), Arg::P(src2), Arg::I(lds as i32), Arg::P(nw), Arg::P(part), Arg::P(wt), Arg::I(n as i32), Arg::F(self.cfg.eps), Arg::F(self.cfg.swiglu_limit), Arg::P(oq), Arg::P(os), Arg::P(ob), Arg::P(obq), Arg::B(zl)],
        )
    }

    /// Feed `tokens` at the current position; returns the greedy next token (logits of the last token on device).
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<u32> {
        if self.pf.is_none() {
            self.pf = Some(PfBufs::new(&self.g, self.max_ctx)?);
            let rings: Vec<u64> = self.layers.iter().map(|l| l.ring.ptr).collect();
            self.g.htod(self.pf.as_ref().unwrap().ringtab.ptr, &rings)?;
        }
        let mut next = 0;
        for chunk in tokens.chunks(TMAX) {
            let p0 = self.pos;
            next = self.run_chunk(chunk, false)?[0];
            if self.dspark.is_some() {
                // the draft layers' windows: the newest 128 positions
                let n = chunk.len().min(WIN);
                self.observe(p0 + chunk.len() - n, n, chunk.len() - n)?;
            }
        }
        Ok(next)
    }

    /// MmJobs as decode-style vgemv jobs (verify mode)
    fn as_v(&self, jobs: &[MmJob]) -> Vec<crate::verify::VJob> {
        jobs.iter()
            .map(|j| crate::verify::VJob {
                w: j.w, sc: j.sc, woff: j.woff, soff: j.soff, mul: j.mul, fmt: j.fmt, n: j.n, k: j.k, xq: j.xq, xs: j.xs, xb: j.xb,
                ldx: j.ldx, ntok: j.ntok, y: j.y, ldy: j.ldy, atomic: j.yatomic, ..Default::default()
            })
            .collect()
    }

    /// bf16 twin of an FP8 activation buffer (written alongside it by prep / write_act)
    fn bf16_twin(&self, xq: u64) -> Option<u64> {
        let pf = self.pf.as_ref().unwrap();
        let pairs = [(&pf.aq, &pf.aqb), (&pf.hq, &pf.hqb), (&pf.shq, &pf.shqb), (&pf.eq, &pf.eqb), (&pf.oq, &pf.oqb)];
        pairs.iter().find(|(q, _)| xq >= q.ptr && xq < q.ptr + q.bytes as u64).map(|(q, b)| b.ptr + (xq - q.ptr) * 2)
    }

    /// batched step dense jobs: hipBLAS on the BF16 copies from DSXA_BBLAS_MIN tokens (default 32) when built,
    /// else bgemv_d
    pub(crate) fn bdense_launch(&self, label: &str, mut jobs: Vec<crate::verify::VJob>) -> Result<()> {
        let min: i32 = std::env::var("DSXA_BBLAS_MIN").ok().and_then(|v| v.parse().ok()).unwrap_or(32);
        if jobs.iter().all(|j| j.ntok >= min) {
            let mut ok = true;
            for j in jobs.iter_mut() {
                if j.fmt == 0 {
                    match self.bf16_twin(j.xq) { Some(b) => j.xb = b, None => ok = false }
                }
            }
            if ok && self.bgemm(label, &jobs)? {
                return Ok(());
            }
        }
        self.blaunch(label, jobs)
    }

    fn gm(&self, verify: bool, plain: &dyn Fn(&str) -> bool, label: &str, jobs: &[MmJob]) -> Result<()> {
        let r = self.gm_inner(verify, plain, label, jobs);
        self.pz.set(false);
        r
    }

    fn gm_inner(&self, verify: bool, plain: &dyn Fn(&str) -> bool, label: &str, jobs: &[MmJob]) -> Result<()> {
        if self.pf.as_ref().unwrap().bmode.get() && !plain(label) && std::env::var_os("DSXA_BMM").is_none() {
            return self.bdense_launch(label, self.as_v(jobs));
        }
        // prompt chunks: dense GEMMs on hipBLAS too once the BF16 copies exist (DSXA_PMM=1: the mm kernels)
        if !verify && !self.bdense.is_empty() && !label.contains("experts") && jobs.iter().all(|j| j.ntok >= 64 && j.toks == 0 && j.yrow == 0)
            && std::env::var_os("DSXA_PMM").is_none()
        {
            let mut v = self.as_v(jobs);
            for (vj, j) in v.iter_mut().zip(jobs) {
                vj.split = 1;
                if j.fmt == 0 {
                    match self.bf16_twin(j.xq) { Some(b) => vj.xb = b, None => return self.mm_l(label, jobs) }
                }
            }
            if self.bgemm(label, &v)? {
                return Ok(());
            }
        }
        if verify && !plain(label) { self.vlaunch(label, self.as_v(jobs), self.st.ptr) } else { self.mm_l(label, jobs) }
    }

    /// Run `tk` at the current position. Prompt mode: advance the state, return [next token]. Verify mode: the
    /// state of the window rings and the position stay as they were (see `commit`); returns the greedy next token
    /// after each position.
    pub(crate) fn run_chunk(&mut self, tk: &[u32], verify: bool) -> Result<Vec<u32>> {
        self.run_chunk_mode(tk, verify, false)
    }

    /// `draft`: run `tk` (a DSpark block at the current position) through the draft layers instead, bidirectional
    /// inside the block, nothing committed; returns the drafted tokens.
    pub(crate) fn run_chunk_mode(&mut self, tk: &[u32], verify: bool, draft: bool) -> Result<Vec<u32>> {
        self.run_chunk_full(tk, verify, draft, None)
    }

    /// Batched decode step (crate::batch): token i at its own position and slot; returns the greedy next tokens.
    pub(crate) fn run_chunk_b(&mut self, tk: &[u32], b: &crate::batch::Batch) -> Result<Vec<u32>> {
        self.run_chunk_full(tk, false, false, Some(b))
    }

    /// Batched DSpark draft pass over the draft layers (logits of every row left in b.logits)
    pub(crate) fn run_chunk_draft_b(&mut self, tk: &[u32], b: &crate::batch::Batch) -> Result<Vec<u32>> {
        self.run_chunk_full(tk, false, true, Some(b))
    }

    fn run_chunk_full(&mut self, tk: &[u32], verify: bool, draft: bool, bt: Option<&crate::batch::Batch>) -> Result<Vec<u32>> {
        let t = tk.len();
        let p0 = self.pos;
        // DSXA_VPLAIN=a,b: verify the GEMMs whose label contains a or b ("mm" = all) with the prompt-path kernels
        let plain = |label: &str| std::env::var("DSXA_VPLAIN").map(|v| v.split(',').any(|x| !x.is_empty() && label.contains(x))).unwrap_or(false);
        let vk = verify;
        if let Some(pf) = self.pf.as_ref() {
            pf.ring.set(0); // the previous chunk ended with a sync
            pf.bmode.set(bt.is_some());
        }
        if bt.is_none() && p0 + t > self.max_ctx {
            anyhow::bail!("context full");
        }
        // Engram rows for every token (host), into the device rows buffer [t][2][ER]
        let mut rows = vec![0u8; t * 2 * ER];
        if let Some(b) = bt {
            rows.copy_from_slice(&b.erows);
        } else if !draft {
            self.engram_rows_batch(tk, &mut rows)?;
        }
        let g = &self.g;
        let c = self.cfg.clone();
        let pf = self.pf.as_ref().unwrap();
        let ti: Vec<i32> = tk.iter().map(|&x| x as i32).collect();
        // verify / draft passes replay a HIP graph per (tokens, draft, 16K-position bucket): inputs through pinned
        // memory, position-dependent kernels read the position from the decode state (p0 = -1)
        let gmode = bt.is_none() && (verify || draft) && t <= 16 && self.head.sc.is_none() && std::env::var_os("DSXA_VPLAIN").is_none() && std::env::var("DSXA_VGRAPH").map(|v| v != "0").unwrap_or(true) && !g.profiling();
        let bucket = (p0 + t).div_ceil(16384);
        let key = (t, draft, bucket);
        let cached = gmode && self.vgraphs.borrow().contains_key(&key);
        let p0a: i32 = if gmode { -1 } else { p0 as i32 };
        // batched steps (DSXA_BGRAPH=1): one HIP graph per (tokens, draft, verify, 16K-position bucket, batch),
        // captured the second time a shape is seen (the first run warms up hipBLASLt); inputs through pinned memory
        let epoch = GRAPH_EPOCH.load(std::sync::atomic::Ordering::Relaxed);
        if self.bgraph_epoch.get() != epoch {
            self.bgraphs.borrow_mut().clear();
            self.bseen.borrow_mut().clear();
            self.bgraph_epoch.set(epoch);
        }
        let bgraph = bt.is_some() && match BGRAPH_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed) { 1 => false, 2 => true, _ => std::env::var("DSXA_BGRAPH").as_deref() == Ok("1") } && !g.profiling();
        let mut bkey = (0usize, false, false, 0usize, 0u64);
        let mut bcap = false;
        if let (true, Some(b)) = (bgraph, bt) {
            let maxp = b.step_pos.iter().copied().max().unwrap_or(0) + t;
            let bk = maxp.div_ceil(16384);
            b.gbucket.set(bk);
            bkey = (t, draft, b.multi, bk, b.bpos.ptr);
            unsafe {
                std::ptr::copy_nonoverlapping(rows.as_ptr(), b.pin_rows, rows.len());
                std::ptr::copy_nonoverlapping(ti.as_ptr() as *const u8, b.pin_toks, t * 4);
            }
            if let Some(h) = b.eng1.borrow_mut().take() {
                let v = h.join().map_err(|_| anyhow::anyhow!("engram fetch thread"))?;
                unsafe { std::ptr::copy_nonoverlapping(v.as_ptr(), b.eng_pin, v.len()) };
            }
            if let Some(gr) = self.bgraphs.borrow().get(&bkey) {
                g.graph_launch(gr)?;
                let v: Vec<i32> = g.dtoh(b.next.ptr, t)?;
                return Ok(v.iter().map(|&x| x as u32).collect());
            }
            bcap = !self.bseen.borrow_mut().insert(bkey);
            if bcap {
                g.begin_capture()?;
            }
        }
        if gmode {
            unsafe {
                std::ptr::copy_nonoverlapping(rows.as_ptr(), pf.pin_in, rows.len());
                std::ptr::copy_nonoverlapping(ti.as_ptr() as *const u8, pf.pin_in.add(PIN_TOKS), t * 4);
            }
            if !cached {
                g.begin_capture()?;
            }
        }
        'body: {
        if cached {
            break 'body;
        }
        if gmode {
            g.htod_async(pf.erows.ptr, pf.pin_in, t * 2 * ER)?;
            g.htod_async(pf.toks.ptr, unsafe { pf.pin_in.add(PIN_TOKS) }, t * 4)?;
        } else if let (true, Some(b)) = (bgraph, bt) {
            g.htod_async(pf.erows.ptr, b.pin_rows, t * 2 * ER)?;
            g.htod_async(pf.toks.ptr, b.pin_toks, t * 4)?;
        } else {
            g.htod(pf.erows.ptr, &rows)?;
            g.htod(pf.toks.ptr, &ti)?;
        }
        g.launch("embed_t", (20, t as u32, 1), 256, 0, &[Arg::P(self.embed.ptr), Arg::P(pf.toks.ptr), Arg::P(pf.h.ptr)])?;
        let none = HcArgs::default();
        let wscale = (c.idx_dim as f32).powf(-0.5) * (c.idx_heads as f32).powf(-0.5);
        let dsx = self.dspark.as_ref();
        let layers: &[Layer] = if draft { &dsx.unwrap().layers } else { &self.layers };
        // DSpark main_hidden capture (prompt and verify passes): layer -> slot
        // batched decode: per-token positions / slots (kernels/batch.h), experts grouped on the device (bgroup)
        let bx = bt.is_some() && !plain("mm experts");
        // prompt chunks: experts grouped on the device too (bgroup jobs of 16 * DSXA_PG tokens; DSXA_PGROUP=0: mm)
        let pg = bt.is_none() && !verify && !draft && t >= 16 && t * 6 <= 12288 && !plain("mm experts")
            && std::env::var("DSXA_PGROUP").map(|v| v != "0").unwrap_or(true);
        // DSpark main_hidden capture (prompt, verify and batched verify passes): layer -> slot
        let mh_slot = |li: usize| if draft { None } else { dsx.and_then(|d| d.targets.iter().position(|&x| x == li)) };
        let nl = layers.len();
        let hc = |y: u64, round: bool, pm: u64, pa: &HcArgs, pre: u64, prea: &HcArgs, fnw: u64, nmix: u64, xr: u64, xss: u64, eng: Option<(u64, u64)>| -> Result<()> {
            let (ev, es) = eng.unwrap_or((0, 0));
            // finish the mixes once per token
            g.launch("hc_mix_t", (t as u32, 1, 1), 64, 0, &[Arg::P(pm), Arg::B(bytes(pa)), Arg::P(pre), Arg::B(bytes(prea)), Arg::P(pf.mixd.ptr), Arg::F(c.hc_eps), Arg::F(c.eps), Arg::I(c.sinkhorn_iters as i32)])?;
            // tokens per block: the 24 mix-weight rows are loaded once per block (2 MB per token otherwise)
            let tpb = std::env::var("DSXA_HC_TPB").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
            let mut xo = vec![0u8; 56];
            xo[40..48].copy_from_slice(&pf.mixd.ptr.to_le_bytes());
            xo[48..52].copy_from_slice(&(t as i32).to_le_bytes());
            xo[52..56].copy_from_slice(&(tpb as i32).to_le_bytes());
            g.launch(
                "hc_step",
                ((D / 64) as u32, t.div_ceil(tpb) as u32, 1),
                256,
                0,
                &[Arg::P(pf.h.ptr), Arg::P(y), Arg::I(round as i32), Arg::P(pm), Arg::B(bytes(pa)), Arg::P(pre), Arg::B(bytes(prea)), Arg::P(fnw), Arg::P(nmix), Arg::P(xr), Arg::P(xss), Arg::P(ev), Arg::P(es), Arg::P(0), Arg::F(c.hc_eps), Arg::F(c.eps), Arg::I(c.sinkhorn_iters as i32), Arg::B(xo)],
            )
        };
        for li in 0..nl {
            let l = &layers[li];
            // ---- residual step into attention
            if li == 0 {
                hc(0, false, 0, &none, 0, &none, l.hc_attn_fn.ptr, pf.mix_a.ptr, pf.xr_a.ptr, pf.xss_a.ptr, None)?;
            } else {
                let p = &layers[li - 1];
                if let Some((wkv, qk)) = &l.engram {
                    hc(pf.y.ptr, true, pf.mix_f.ptr, &p.hc_ffn, 0, &none, 0, 0, 0, 0, None)?;
                    let ei = c.engram_layers.iter().position(|&x| x == li).unwrap();
                    // batched step: rows [table][token][ER]; table 1 arrives from the background fetch now
                    let (rp, rs) = if let Some(b) = bt {
                        if ei == 1 && bgraph {
                            // joined before the graph; the copy is part of it
                            g.htod_async(pf.erows.at(t * ER), b.eng_pin, t * ER)?;
                        } else if ei == 1 {
                            if let Some(h) = b.eng1.borrow_mut().take() {
                                let v = h.join().map_err(|_| anyhow::anyhow!("engram fetch thread"))?;
                                unsafe { std::ptr::copy_nonoverlapping(v.as_ptr(), b.eng_pin, v.len()) };
                                g.htod_async(pf.erows.at(t * ER), b.eng_pin, v.len())?;
                            }
                        }
                        (pf.erows.at(ei * t * ER), ER)
                    } else {
                        (pf.erows.at(ei * ER), 2 * ER)
                    };
                    g.launch("engram_act_t", (t as u32, 1, 1), 256, 0, &[Arg::P(rp), Arg::I(rs as i32), Arg::P(pf.eq.ptr), Arg::P(pf.es.ptr), Arg::P(pf.eqb.ptr)])?;
                    let mut j = MmJob::dense(wkv, 0, pf.eq.ptr, pf.es.ptr, 6144, pf.ekv.ptr, 25600);
                    j.ntok = t as i32;
                    self.gm(vk, &plain, "mm engram", &[j])?;
                    g.memset(pf.esum.ptr, 0, t * 16 * 4)?;
                    g.launch("engram_sums", (5, 4, t as u32), 256, 0, &[Arg::P(pf.ekv.ptr), Arg::P(pf.h.ptr), Arg::P(qk.ptr), Arg::P(pf.esum.ptr)])?;
                    hc(0, false, 0, &none, pf.mix_f.ptr, &p.hc_ffn, l.hc_attn_fn.ptr, pf.mix_a.ptr, pf.xr_a.ptr, pf.xss_a.ptr, Some((pf.ekv.at(4 * D * 4), pf.esum.ptr)))?;
                } else {
                    hc(pf.y.ptr, true, pf.mix_f.ptr, &p.hc_ffn, pf.mix_f.ptr, &p.hc_ffn, l.hc_attn_fn.ptr, pf.mix_a.ptr, pf.xr_a.ptr, pf.xss_a.ptr, None)?;
                }
            }
            if let Some(slot) = mh_slot(li) {
                g.launch("mh_t", (20, t as u32, 1), 256, 0, &[Arg::P(pf.h.ptr), Arg::P(pf.mh.ptr), Arg::I((3 * D) as i32), Arg::I((slot * D) as i32)])?;
            }
            // ---- attention projections
            // small token counts (k-split products): this prep clears the outputs of q_a/kv/comp and wo_a
            let zsmall = t < 32 && std::env::var_os("DSXA_NOPZ").is_none();
            let opt = |b: &Option<crate::load::DMat>| b.is_some();
            let mut zs: Vec<(u64, usize)> = vec![];
            if zsmall {
                zs = vec![(pf.qraw.ptr, t * QL), (pf.kvraw.ptr, t * HD), (pf.r.ptr, t * 8192)];
                if opt(&l.comp_wkv) { zs.push((pf.ckv.ptr, t * HD)); }
                if opt(&l.comp_wgate) { zs.push((pf.cg.ptr, t * HD)); }
                if opt(&l.idx_proj) { zs.push((pf.iw.ptr, t * IH)); }
            }
            self.prep_z(0, pf.xr_a.ptr, 0, D, l.attn_norm.ptr, pf.xss_a.ptr, 0, D, t, pf.aq.ptr, pf.as_.ptr, pf.ab.ptr, pf.aqb.ptr, &zs)?;
            self.pz.set(zsmall);
            let mut jobs = vec![MmJob::dense(&l.wq_a, 0, pf.aq.ptr, pf.as_.ptr, D, pf.qraw.ptr, QL), MmJob::dense(&l.wkv, 0, pf.aq.ptr, pf.as_.ptr, D, pf.kvraw.ptr, HD)];
            let mut bf = |m: &crate::load::DMat, y: u64, ldy: usize| {
                let mut j = MmJob::dense(m, 4, 0, 0, D, y, ldy);
                j.xb = pf.ab.ptr;
                j
            };
            if let Some(m) = &l.comp_wkv {
                jobs.push(bf(m, pf.ckv.ptr, HD));
            }
            if let Some(m) = &l.comp_wgate {
                jobs.push(bf(m, pf.cg.ptr, HD));
            }
            if let Some(m) = &l.idx_proj {
                jobs.push(bf(m, pf.iw.ptr, IH));
            }
            for j in jobs.iter_mut() {
                j.ntok = t as i32;
            }
            self.gm(vk, &plain, "mm q_a/kv/comp", &jobs)?;
            // q_b (+ indexer q) from rmsnorm(qr)
            let zs: Vec<(u64, usize)> = if zsmall { vec![(pf.q.ptr, t * NH * HD), (pf.iq.ptr, t * IH * ID)] } else { vec![] };
            self.prep_z(1, pf.qraw.ptr, 0, QL, l.q_norm.ptr, 0, 0, QL, t, pf.aq.ptr, pf.as_.ptr, 0, pf.aqb.ptr, &zs)?;
            self.pz.set(zsmall);
            let mut jobs = vec![MmJob::dense(&l.wq_b, 0, pf.aq.ptr, pf.as_.ptr, QL, pf.q.ptr, NH * HD)];
            if let Some(m) = &l.idx_wq_b {
                jobs.push(MmJob::dense(m, 0, pf.aq.ptr, pf.as_.ptr, QL, pf.iq.ptr, IH * ID));
            }
            for j in jobs.iter_mut() {
                j.ntok = t as i32;
            }
            self.gm(vk, &plain, "mm q_b", &jobs)?;
            let rf = if l.ratio > 0 { self.rope_fc.ptr } else { self.rope_f.ptr };
            if let Some(b) = bt {
                g.launch("rope_b", (t as u32, NH as u32, 1), 32, 0, &[Arg::P(pf.q.ptr), Arg::I(NH as i32), Arg::I(HD as i32), Arg::P(rf), Arg::P(b.bpos.ptr), Arg::I(0)])?;
                if l.idx_wq_b.is_some() {
                    g.launch("rope_b", (t as u32, IH as u32, 1), 32, 0, &[Arg::P(pf.iq.ptr), Arg::I(IH as i32), Arg::I(ID as i32), Arg::P(self.rope_fc.ptr), Arg::P(b.bpos.ptr), Arg::I(0)])?;
                }
                if draft { self.batch_attn_draft(b, li, t, rf)?; } else { self.batch_attn(b, li, t, rf, wscale)?; }
            } else {
            g.launch("rope_t", (t as u32, NH as u32, 1), 32, 0, &[Arg::P(pf.q.ptr), Arg::I(NH as i32), Arg::I(HD as i32), Arg::P(rf), Arg::I(p0a), Arg::I(0)])?;
            if l.idx_wq_b.is_some() {
                g.launch("rope_t", (t as u32, IH as u32, 1), 32, 0, &[Arg::P(pf.iq.ptr), Arg::I(IH as i32), Arg::I(ID as i32), Arg::P(self.rope_fc.ptr), Arg::I(p0a), Arg::I(0)])?;
            }
            // window rows: previous 128 positions from the ring, then this chunk's
            g.launch("ring_to_lin", (WIN as u32, 1, 1), 256, 0, &[Arg::P(l.ring.ptr), Arg::I(p0a), Arg::P(pf.lin.ptr)])?;
            g.launch("kv_t", (t as u32, 1, 1), 512, 0, &[Arg::P(pf.kvraw.ptr), Arg::P(l.kv_norm.ptr), Arg::P(rf), Arg::I(p0a), Arg::P(pf.lin.ptr), Arg::I(WIN as i32)])?;
            // compressor + indexer keys
            if l.kv_src {
                let opt = |b: &Option<Buf>| b.as_ref().map(|x| x.ptr).unwrap_or(0);
                g.launch(
                    "comp_t",
                    (t as u32, 1, 1),
                    512,
                    0,
                    &[Arg::I(p0a), Arg::I(t as i32), Arg::I(l.ratio as i32), Arg::P(pf.ckv.ptr), Arg::P(pf.cg.ptr), Arg::P(opt(&l.cstate)), Arg::P(opt(&l.comp_norm)), Arg::P(opt(&l.idx_wk)), Arg::P(opt(&l.idx_knorm)), Arg::P(opt(&l.comp_cache)), Arg::P(opt(&l.idx_cache)), Arg::P(self.rope_fc.ptr), Arg::F(c.eps)],
                )?;
            }
            // indexer + top-k per query
            if l.idx_src {
                let kc = self.layers[l.idxk_src].idx_cache.as_ref().unwrap().ptr;
                let after = li > c.cand_layer;
                let lds = pf.max_end + 64;
                for q0 in (0..t).step_by(QB) {
                    let nq = QB.min(t - q0);
                    let pend = if gmode { bucket * 16384 } else { p0 + q0 + nq };
                    let chunks = ((pend / l.ratio.max(1)).max(16384)).div_ceil(1024) as u32;
                    let (rows, nrows) = if after { (pf.keep.ptr, pf.nkeep.ptr) } else { (0, 0) };
                    g.launch("iscore_t", (chunks, nq as u32, 1), 256, 0, &[Arg::I(p0a), Arg::I(q0 as i32), Arg::I(l.ratio as i32), Arg::P(pf.iq.ptr), Arg::P(pf.iw.ptr), Arg::F(wscale), Arg::P(kc), Arg::P(rows), Arg::P(nrows), Arg::I(16384), Arg::P(pf.score.ptr), Arg::I(lds as i32)])?;
                    if li == c.cand_layer {
                        g.launch("icand_t", (nq as u32, 1, 1), 1024, 0, &[Arg::I(p0a), Arg::I(q0 as i32), Arg::I(l.ratio as i32), Arg::I(c.cand_block as i32), Arg::I(c.cand_blocks as i32), Arg::P(pf.score.ptr), Arg::I(lds as i32), Arg::P(pf.bsc.ptr), Arg::P(pf.kb.ptr), Arg::P(pf.keep.ptr), Arg::P(pf.nkeep.ptr)])?;
                        // the layer's own top-512 over its kept positions: rescore on the keep list
                        g.launch("iscore_t", (16, nq as u32, 1), 256, 0, &[Arg::I(p0a), Arg::I(q0 as i32), Arg::I(l.ratio as i32), Arg::P(pf.iq.ptr), Arg::P(pf.iw.ptr), Arg::F(wscale), Arg::P(kc), Arg::P(pf.keep.ptr), Arg::P(pf.nkeep.ptr), Arg::I(16384), Arg::P(pf.score.ptr), Arg::I(lds as i32)])?;
                    }
                    let use_keep = after || li == c.cand_layer;
                    let (rows, nrows) = if use_keep { (pf.keep.ptr, pf.nkeep.ptr) } else { (0, 0) };
                    g.launch("itopk_t", (nq as u32, 1, 1), 1024, 0, &[Arg::I(p0a), Arg::I(q0 as i32), Arg::I(l.ratio as i32), Arg::P(pf.score.ptr), Arg::I(lds as i32), Arg::P(rows), Arg::P(nrows), Arg::I(16384), Arg::I(c.idx_topk as i32), Arg::I(WIN as i32), Arg::P(pf.cidx.ptr), Arg::I(512), Arg::P(pf.ncomp.ptr), Arg::I(0)])?;
                }
            }
            // sparse attention per query
            let comp = if l.ratio > 0 { self.layers[l.comp_src].comp_cache.as_ref().unwrap().ptr } else { 0 };
            if std::env::var_os("DSXA_ATTN_T").is_some() {
            g.launch(
                "attn_t",
                (t as u32, 4, 1),
                256,
                0,
                &[Arg::I(p0a), Arg::P(pf.q.ptr), Arg::P(pf.lin.ptr), Arg::P(comp), Arg::P(pf.cidx.ptr), Arg::P(pf.ncomp.ptr), Arg::I(512), Arg::I(WIN as i32), Arg::F(1.0 / (HD as f32).sqrt()), Arg::P(l.sinks.ptr), Arg::P(rf), Arg::P(pf.oq.ptr), Arg::P(pf.os.ptr), Arg::P(pf.oqb.ptr)],
            )?;
            } else {
                g.launch(
                    "attn_m",
                    (t as u32, 1, 1),
                    1024,
                    0,
                    &[Arg::I(p0a), Arg::P(pf.q.ptr), Arg::P(pf.lin.ptr), Arg::P(comp), Arg::P(pf.cidx.ptr), Arg::P(pf.ncomp.ptr), Arg::I(512), Arg::I(WIN as i32), Arg::F(1.0 / (HD as f32).sqrt()), Arg::P(l.sinks.ptr), Arg::I(if draft { t as i32 } else { 0 })],
                )?;
                g.launch("attn_m_out", (t as u32, (NH / 8) as u32, 1), 256, 0, &[Arg::I(p0a), Arg::P(pf.q.ptr), Arg::P(rf), Arg::P(pf.oq.ptr), Arg::P(pf.os.ptr), Arg::P(pf.oqb.ptr)])?;
            }
            if let (true, Ok(dir)) = (li == 0, std::env::var("DSXA_DUMPQ")) {
                let w = |n: &str, v: Vec<u8>| std::fs::write(format!("{dir}/pf_{n}_p{p0}.bin"), v);
                let f = |b: &Buf, n: usize| -> Result<Vec<u8>> { let v: Vec<f32> = g.dtoh(b.ptr, n)?; Ok(v.iter().flat_map(|x| x.to_le_bytes()).collect()) };
                w("q", f(&pf.q, NH * HD)?)?;
                w("qraw", f(&pf.qraw, QL)?)?;
                w("kv", f(&pf.kvraw, HD)?)?;
                w("xraw", f(&pf.xr_a, D)?)?;
                w("xss", f(&pf.xss_a, 80)?)?;
                let oq: Vec<u8> = g.dtoh(pf.oq.ptr, NH * HD)?;
                w("oq", oq)?;
                let lin: Vec<u8> = g.dtoh(pf.lin.at(WIN * HD * 2), HD * 2)?;
                w("lin", lin)?;
            }
            if draft {
                // the draft block is never committed
            } else if verify {
                g.dtod(pf.vlin.at(li * 16 * HD * 2), pf.lin.at(WIN * HD * 2), t * HD * 2)?;
            } else {
                g.launch("lin_to_ring", (t.min(WIN) as u32, 1, 1), 256, 0, &[Arg::P(pf.lin.ptr), Arg::I(p0a), Arg::I(t as i32), Arg::P(l.ring.ptr)])?;
            }
            } // not batched
            // wo_a per group, wo_b
            let k_a = l.wo_a.k;
            let tile_bytes = (k_a / 64) * 1024;
            let jobs: Vec<MmJob> = (0..8)
                .map(|gi| {
                    let mut j = MmJob::dense(&l.wo_a, 0, pf.oq.at(gi * 4096), pf.os.at(gi * 128 * 4), NH * HD, pf.r.at(gi * 1024 * 4), 8192);
                    j.w = l.wo_a.w.at(gi * 64 * tile_bytes);
                    j.sc = l.wo_a.sc.as_ref().unwrap().at(gi * 32 * (k_a / 32));
                    j.n = 1024;
                    j.ntok = t as i32;
                    j
                })
                .collect();
            self.pz.set(zsmall); // r was cleared by this layer's first prep
            self.gm(vk, &plain, "mm wo_a", &jobs)?;
            let zs: Vec<(u64, usize)> = if zsmall { vec![(pf.a.ptr, t * D)] } else { vec![] };
            self.prep_z(2, pf.r.ptr, 0, 8192, 0, 0, 0, 8192, t, pf.aq.ptr, pf.as_.ptr, 0, pf.aqb.ptr, &zs)?;
            self.pz.set(zsmall);
            let mut j = MmJob::dense(&l.wo_b, 0, pf.aq.ptr, pf.as_.ptr, 8192, pf.a.ptr, D);
            j.ntok = t as i32;
            self.gm(vk, &plain, "mm wo_b", &[j])?;
            // ---- ffn
            hc(pf.a.ptr, false, pf.mix_a.ptr, &l.hc_attn, pf.mix_a.ptr, &l.hc_attn, l.hc_ffn_fn.ptr, pf.mix_f.ptr, pf.xr_f.ptr, pf.xss_f.ptr, None)?;
            let zs: Vec<(u64, usize)> = if zsmall { vec![(pf.logits.ptr, t * NE), (pf.sg.ptr, t * INTER), (pf.su.ptr, t * INTER)] } else { vec![] };
            self.prep_z(0, pf.xr_f.ptr, 0, D, l.ffn_norm.ptr, pf.xss_f.ptr, 0, D, t, pf.aq.ptr, pf.as_.ptr, pf.ab.ptr, pf.aqb.ptr, &zs)?;
            self.pz.set(zsmall);
            let ne_l = if draft { dsx.unwrap().ne } else { NE };
            let mut gj = MmJob::dense(&l.gate, 4, 0, 0, D, pf.logits.ptr, ne_l);
            gj.xb = pf.ab.ptr;
            let mut jobs = vec![gj, MmJob::dense(&l.sh_w1, 0, pf.aq.ptr, pf.as_.ptr, D, pf.sg.ptr, INTER), MmJob::dense(&l.sh_w3, 0, pf.aq.ptr, pf.as_.ptr, D, pf.su.ptr, INTER)];
            for j in jobs.iter_mut() {
                j.ntok = t as i32;
            }
            self.gm(vk, &plain, "mm router+shared", &jobs)?;
            let (kk, ebase) = if draft { (dsx.unwrap().top_k, dsx.unwrap().arena.ptr) } else { (6, self.arena.ptr) };
            if draft {
                g.launch("route_k_t", (t as u32, 1, 1), 64, 0, &[Arg::P(pf.logits.ptr), Arg::I(ne_l as i32), Arg::I(kk as i32), Arg::P(l.gate_bias.ptr), Arg::F(c.route_scale), Arg::P(pf.sel.ptr), Arg::P(pf.wts.ptr)])?;
            } else if bx || pg || vk && !plain("mm experts") {
                // verify / batch: no expert counts needed (device grouping)
                g.launch("route_t", (t as u32, 1, 1), 64, 0, &[Arg::P(pf.logits.ptr), Arg::P(l.gate_bias.ptr), Arg::F(c.route_scale), Arg::P(pf.sel.ptr), Arg::P(pf.wts.ptr), Arg::P(pf.counts.ptr)])?;
            }
            if !(draft || bx || pg || (vk && !plain("mm experts"))) {
                g.memset(pf.counts.ptr, 0, NE * 4)?;
                g.launch("route_t", (t as u32, 1, 1), 64, 0, &[Arg::P(pf.logits.ptr), Arg::P(l.gate_bias.ptr), Arg::F(c.route_scale), Arg::P(pf.sel.ptr), Arg::P(pf.wts.ptr), Arg::P(pf.counts.ptr)])?;
            }
            if draft || bx || pg || (vk && !plain("mm experts")) {
                // device grouping: one job per distinct expert (batch / prompt chunks: per jt of its tokens) and matrix,
                // tokens as MFMA columns. Batch: jt 16; prompt chunks: jt 16 * DSXA_PG (default 2)
                let pgg: usize = if pg { std::env::var("DSXA_PG").ok().and_then(|v| v.parse().ok()).unwrap_or(2) } else { 1 };
                let jt = 16 * pgg;
                let js = 8 + 2 * jt;
                // job slots: at most one per expert touched (<= picks) plus one per jt picks
                let maxj = NE.min(t * kk) + (t * kk).div_ceil(jt);
                let (ndyn, dynb) = if pg { (maxj, pf.gdyn.ptr) } else if let Some(b) = bt { (maxj.min(crate::batch::BG_JOBS), b.dyn_.ptr) } else { (48, pf.dyn_.ptr) };
                if bt.is_some() || pg {
                    g.launch("bgroup", (64, 1, 1), 1024, 0, &[Arg::I(t as i32), Arg::I(kk as i32), Arg::P(pf.sel.ptr), Arg::P(pf.wts.ptr), Arg::P(l.experts.ptr), Arg::P(dynb), Arg::P(pf.wrow.ptr), Arg::P(pf.fill.ptr), Arg::P(pf.gg.ptr), Arg::P(pf.uu.ptr), Arg::P(pf.y.ptr), Arg::I(jt as i32), Arg::I(ndyn as i32)])?;
                } else {
                g.launch("vgroup", (64, 1, 1), 256, 0, &[Arg::I(t as i32), Arg::I(kk as i32), Arg::P(pf.sel.ptr), Arg::P(pf.wts.ptr), Arg::P(l.experts.ptr), Arg::P(pf.dyn_.ptr), Arg::P(pf.wrow.ptr), Arg::P(pf.fill.ptr), Arg::P(pf.gg.ptr), Arg::P(pf.uu.ptr), Arg::P(pf.y.ptr)])?;
                }
                if (bt.is_some() || pg) && std::env::var_os("DSXA_NOXP").is_none() {
                    let r: usize = std::env::var("DSXA_XR").ok().and_then(|v| v.parse().ok()).unwrap_or(if pgg >= 4 { 1 } else { 2 });
                    self.xg.set(Some((r, pgg)));
                }
                // outputs zeroed by vgroup: all expert jobs add, so k can be split (gate/up 4 ways, down 2)
                let esplit = |m: usize| std::env::var(if m < 2 { "DSXA_ESPLIT_GU" } else { "DSXA_ESPLIT_D" }).ok().and_then(|v| v.parse().ok()).unwrap_or(if pg { 1 } else if m < 2 { 4 } else { 2 });
                let ej = |m: usize, n: usize, k: usize, xq: u64, xs: u64, ldx: usize, y: u64, ldy: usize, _atomic: i32| crate::verify::VJob {
                    w: ebase, n: n as i32, k: k as i32, xq, xs, ldx: ldx as i32, y, ldy: ldy as i32, atomic: 1, split: esplit(m), ndyn: ndyn as i32, dyn_: dynb + (m * ndyn * js * 4) as u64, fmt: 1, ..Default::default()
                };
                self.vlaunch("verify experts gu", vec![ej(0, INTER, D, pf.aq.ptr, pf.as_.ptr, D, pf.gg.ptr, INTER, 0), ej(1, INTER, D, pf.aq.ptr, pf.as_.ptr, D, pf.uu.ptr, INTER, 0)], self.st.ptr)?;
                let rtot = t * kk;
                if std::env::var("DSXA_SWIGLU_PAIR").as_deref() == Ok("1") {
                    g.launch("prep_swiglu_pair_t", ((rtot + t) as u32, INTER.div_ceil(1024) as u32, 1), 256, 0,
                        &[Arg::I(rtot as i32), Arg::P(pf.gg.ptr), Arg::P(pf.uu.ptr), Arg::P(pf.wrow.ptr),
                          Arg::P(pf.hq.ptr), Arg::P(pf.hs.ptr), Arg::P(0), Arg::P(pf.hqb.ptr),
                          Arg::P(pf.sg.ptr), Arg::P(pf.su.ptr), Arg::P(pf.shq.ptr), Arg::P(pf.shs.ptr),
                          Arg::P(0), Arg::P(pf.shqb.ptr), Arg::F(c.swiglu_limit)])?;
                } else {
                    self.prep(3, pf.gg.ptr, pf.uu.ptr, INTER, 0, 0, pf.wrow.ptr, INTER, rtot, pf.hq.ptr, pf.hs.ptr, 0, pf.hqb.ptr)?;
                    self.prep(3, pf.sg.ptr, pf.su.ptr, INTER, 0, 0, 0, INTER, t, pf.shq.ptr, pf.shs.ptr, 0, pf.shqb.ptr)?;
                }
                let mut sj = MmJob::dense(&l.sh_w2, 0, pf.shq.ptr, pf.shs.ptr, INTER, pf.y.ptr, D);
                sj.ntok = t as i32;
                sj.yatomic = 1;
                let mut sv = self.as_v(&[sj]);
                sv[0].atomic = 1;
                if bt.is_some() {
                    self.bdense_launch("batch shared down", sv)?;
                } else if pg {
                    self.gm(false, &plain, "mm shared down", &[sj])?;
                } else {
                    self.vlaunch("verify shared down", sv, self.st.ptr)?;
                }
                self.vlaunch("verify experts down", vec![ej(2, D, INTER, pf.hq.ptr, pf.hs.ptr, INTER, pf.y.ptr, D, 1)], self.st.ptr)?;
                self.xg.set(None);
                continue;
            }
            let counts: Vec<i32> = g.dtoh(pf.counts.ptr, NE)?;
            let mut off = vec![0i32; NE];
            for e in 1..NE {
                off[e] = off[e - 1] + counts[e - 1];
            }
            g.htod(pf.off.ptr, &off)?;
            g.memset(pf.fill.ptr, 0, NE * 4)?;
            g.launch("route_fill_t", ((t * 6).div_ceil(256) as u32, 1, 1), 256, 0, &[Arg::I(t as i32), Arg::P(pf.sel.ptr), Arg::P(pf.wts.ptr), Arg::P(pf.off.ptr), Arg::P(pf.fill.ptr), Arg::P(pf.tok.ptr), Arg::P(pf.wrow.ptr)])?;
            // routed experts gate/up over each expert's tokens -> compact rows
            let et = &self.etab[li];
            let fmt_of = |f: i32| match f {
                E_FP4 => 1,
                E_CB3 => 2,
                _ => 3,
            };
            let mut jobs = vec![];
            for e in 0..NE {
                if counts[e] == 0 {
                    continue;
                }
                for (mi, y) in [(0usize, pf.gg.ptr), (1, pf.uu.ptr)] {
                    let em = et[e * 3 + mi];
                    jobs.push(MmJob { w: self.arena.ptr, woff: em.woff, soff: em.soff, mul: em.mul, fmt: fmt_of(em.fmt), n: INTER as i32, k: D as i32, xq: pf.aq.ptr, xs: pf.as_.ptr, toks: pf.tok.ptr, tok0: off[e], ntok: counts[e], y, ldy: INTER as i32, yrow: pf.iota.ptr, ldx: D as i32, ..Default::default() });
                }
            }
            for ch in jobs.chunks(1024) {
                self.gm(vk, &plain, "mm experts gu", ch)?;
            }
            let rtot = (t * 6) as usize;
            self.prep(3, pf.gg.ptr, pf.uu.ptr, INTER, 0, 0, pf.wrow.ptr, INTER, rtot, pf.hq.ptr, pf.hs.ptr, 0, pf.hqb.ptr)?;
            self.prep(3, pf.sg.ptr, pf.su.ptr, INTER, 0, 0, 0, INTER, t, pf.shq.ptr, pf.shs.ptr, 0, pf.shqb.ptr)?;
            // down projections, scattered into y by token
            g.memset(pf.y.ptr, 0, t * D * 4)?;
            let mut jobs = vec![];
            for e in 0..NE {
                if counts[e] == 0 {
                    continue;
                }
                let em = et[e * 3 + 2];
                jobs.push(MmJob { w: self.arena.ptr, woff: em.woff, soff: em.soff, mul: em.mul, fmt: fmt_of(em.fmt), n: D as i32, k: INTER as i32, xq: pf.hq.ptr, xs: pf.hs.ptr, toks: 0, tok0: off[e], ntok: counts[e], y: pf.y.ptr, ldy: D as i32, yatomic: 1, yrow: pf.tok.ptr, ldx: INTER as i32, ..Default::default() });
            }
            let mut sj = MmJob::dense(&l.sh_w2, 0, pf.shq.ptr, pf.shs.ptr, INTER, pf.y.ptr, D);
            sj.ntok = t as i32;
            sj.yatomic = 1;
            jobs.push(sj);
            for ch in jobs.chunks(1024) {
                self.gm(vk, &plain, "mm experts down", ch)?;
            }
            if let (true, Ok(dir)) = (li == 0, std::env::var("DSXA_DUMPQ")) {
                let w = |n: &str, v: Vec<u8>| std::fs::write(format!("{dir}/pf_{n}_p{p0}.bin"), v);
                let f = |b: &Buf, n: usize| -> Result<Vec<u8>> { let v: Vec<f32> = g.dtoh(b.ptr, n)?; Ok(v.iter().flat_map(|x| x.to_le_bytes()).collect()) };
                w("a", f(&pf.a, D)?)?;
                w("logits", f(&pf.logits, NE)?)?;
                w("y", f(&pf.y, D)?)?;
                w("r", f(&pf.r, 8192)?)?;
                w("xf", f(&pf.xr_f, D)?)?;
                w("g", f(&pf.gg, 6 * INTER)?)?;
                w("u", f(&pf.uu, 6 * INTER)?)?;
                w("shg", f(&pf.sg, INTER)?)?;
                w("shu", f(&pf.su, INTER)?)?;
                let sel: Vec<i32> = g.dtoh(pf.sel.ptr, 6)?;
                w("sel", sel.iter().flat_map(|x| x.to_le_bytes()).collect())?;
                w("wts", f(&pf.wts, 6)?)?;
                w("wrow", f(&pf.wrow, 6)?)?;
            }
            if let Ok(dir) = std::env::var("DSX_DUMP") {
                // h after this layer for every position (post step applied on a copy) vs the reference dumps
                let save: Vec<f32> = g.dtoh(pf.h.ptr, t * HC * D)?;
                hc(pf.y.ptr, true, pf.mix_f.ptr, &l.hc_ffn, 0, &none, 0, 0, 0, 0, None)?;
                let hh: Vec<f32> = g.dtoh(pf.h.ptr, t * HC * D)?;
                g.htod(pf.h.ptr, &save)?;
                for ti in 0..t {
                    if let Ok(b) = std::fs::read(format!("{dir}/ref_p{}_l{li:02}.bin", p0 + ti)) {
                        let r: Vec<f32> = b.chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
                        let x = &hh[ti * HC * D..(ti + 1) * HC * D];
                        let rn = r.iter().map(|a| a * a).sum::<f32>().sqrt();
                        let dn = r.iter().zip(x).map(|(a, b)| (a - b) * (a - b)).sum::<f32>().sqrt();
                        println!("  prefill pos {} layer {li}: rel {:.2e}", p0 + ti, dn / rn);
                    }
                }
            }
        }
        // ---- last token: final residual step, norm, head, argmax (decode kernels on the decode buffers)
        let last = &layers[nl - 1];
        let onorm = if draft { dsx.unwrap().norm.ptr } else { self.out_norm.ptr };
        hc(pf.y.ptr, true, pf.mix_f.ptr, &last.hc_ffn, 0, &none, 0, 0, 0, 0, None)?;
        if let Some(b) = bt {
            g.launch("hc_final_b", (t as u32, 1, 1), 256, 0, &[Arg::P(pf.h.ptr), Arg::P(pf.mix_f.ptr), Arg::B(bytes(&last.hc_ffn)), Arg::P(onorm), Arg::F(c.hc_eps), Arg::F(c.eps), Arg::I(c.sinkhorn_iters as i32), Arg::P(b.xb.ptr), Arg::P(b.xl.ptr)])?;
            self.batch_head(b, t)?;
            if bgraph {
                break 'body;
            }
            let v: Vec<i32> = g.dtoh(b.next.ptr, t)?;
            return Ok(v.iter().map(|&x| x as u32).collect());
        }
        if verify {
            let pm = self.accp(nl - 1, A_HMIX_F);
            let batched = self.head.sc.is_none();
            for lt in 0..t {
                let (hrow, mrow) = (pf.h.at(lt * HC * D * 4), pf.mix_f.at(lt * 80 * 32 * 4));
                if !batched {
                    g.dtod(self.h.ptr, hrow, HC * D * 4)?;
                    g.dtod(pm, mrow, 80 * 32 * 4)?;
                }
                let (oh, ol) = if batched { (pf.vhh.at(lt * D * 2), pf.vhl.at(lt * D * 2)) } else { (0, 0) };
                g.launch("hc_final", (1, 1, 1), 256, 0, &[Arg::P(hrow), Arg::P(mrow), Arg::B(bytes(&last.hc_ffn)), Arg::P(onorm), Arg::P(self.xfin.ptr), Arg::F(c.hc_eps), Arg::F(c.eps), Arg::I(c.sinkhorn_iters as i32), Arg::P(0), Arg::P(0), Arg::P(oh), Arg::P(ol)])?;
                if !batched {
                    self.head_logits()?;
                    g.dtod(pf.vnext.at(lt * 4), self.next.ptr, 4)?;
                }
            }
            if draft {
                // head for all block positions, then in order: + Markov bias from the previous token, argmax
                let ds = dsx.unwrap();
                anyhow::ensure!(batched, "DSpark needs the bf16 head");
                g.dtod(pf.vnext.ptr, pf.toks.ptr, 4)?;
                let hj = crate::verify::VJob { w: self.head.w.ptr, n: VOCAB as i32, k: D as i32, fmt: 4, xb: pf.vhh.ptr, xl: pf.vhl.ptr, ldx: D as i32, ntok: t as i32, y: pf.vlog.ptr, ldy: VOCAB as i32, split: 1, mul: 1.0, ..Default::default() };
                self.vlaunch("dspark head", vec![hj], self.st.ptr)?;
                for i in 0..t {
                    let mj = crate::verify::VJob { w: ds.markov_head.w.ptr, n: VOCAB as i32, k: ds.rank as i32, fmt: 4, xb: ds.markov_embed.ptr, rows: pf.vnext.at(i * 4), ldx: ds.rank as i32, ntok: 1, y: pf.vlog.at(i * VOCAB * 4), ldy: VOCAB as i32, atomic: 1, mul: 1.0, ..Default::default() };
                    self.vlaunch("dspark markov", vec![mj], self.st.ptr)?;
                    g.launch("argmax_part", (128, 1, 1), 256, 0, &[Arg::P(pf.vlog.at(i * VOCAB * 4)), Arg::I(VOCAB as i32), Arg::P(self.cand.ptr), Arg::P(self.ckey.ptr)])?;
                    g.launch("argmax", (1, 1, 1), 64, 0, &[Arg::P(self.cand.ptr), Arg::P(self.ckey.ptr), Arg::I(128), Arg::P(pf.vnext.at((i + 1) * 4))])?;
                }
                break 'body;
            }
            if batched {
                // one pass over the head for all positions (bf16 hi + lo activations, as decode)
                let hj = crate::verify::VJob { w: self.head.w.ptr, n: VOCAB as i32, k: D as i32, fmt: 4, xb: pf.vhh.ptr, xl: pf.vhl.ptr, ldx: D as i32, ntok: t as i32, y: pf.vlog.ptr, ldy: VOCAB as i32, split: 1, mul: 1.0, ..Default::default() };
                self.vlaunch("verify head", vec![hj], self.st.ptr)?;
                for lt in 0..t {
                    g.launch("argmax_part", (128, 1, 1), 256, 0, &[Arg::P(pf.vlog.at(lt * VOCAB * 4)), Arg::I(VOCAB as i32), Arg::P(self.cand.ptr), Arg::P(self.ckey.ptr)])?;
                    g.launch("argmax", (1, 1, 1), 64, 0, &[Arg::P(self.cand.ptr), Arg::P(self.ckey.ptr), Arg::I(128), Arg::P(pf.vnext.at(lt * 4))])?;
                }
            }
            break 'body;
        }
        let lt = t - 1;
        g.dtod(self.h.ptr, pf.h.at(lt * HC * D * 4), HC * D * 4)?;
        let pm = self.accp(nl - 1, A_HMIX_F);
        g.dtod(pm, pf.mix_f.at(lt * 80 * 32 * 4), 80 * 32 * 4)?;
        g.launch("hc_final", (1, 1, 1), 256, 0, &[Arg::P(self.h.ptr), Arg::P(pm), Arg::B(bytes(&last.hc_ffn)), Arg::P(self.out_norm.ptr), Arg::P(self.xfin.ptr), Arg::F(c.hc_eps), Arg::F(c.eps), Arg::I(c.sinkhorn_iters as i32), Arg::P(0), Arg::P(0), Arg::P(0), Arg::P(0)])?;
        self.head_logits()?;
        self.pos = p0 + t;
        g.htod(self.st.ptr, &[self.pos as i32])?;
        let next: Vec<i32> = g.dtoh(self.next.ptr, 1)?;
        return Ok(vec![next[0] as u32]);
        }
        if let (true, Some(b)) = (bgraph, bt) {
            if bcap {
                let gr = g.end_capture()?;
                g.graph_launch(&gr)?;
                self.bgraphs.borrow_mut().insert(bkey, gr);
            }
            let v: Vec<i32> = g.dtoh(b.next.ptr, t)?;
            return Ok(v.iter().map(|&x| x as u32).collect());
        }
        // verify / draft: finish the capture, replay
        if gmode {
            if !cached {
                let gr = g.end_capture()?;
                self.vgraphs.borrow_mut().insert(key, gr);
            }
            g.graph_launch(&self.vgraphs.borrow()[&key])?;
        }
        if draft {
            let v: Vec<i32> = g.dtoh(pf.vnext.ptr, t + 1)?;
            return Ok(v[1..].iter().map(|&x| x as u32).collect());
        }
        let v: Vec<i32> = g.dtoh(pf.vnext.ptr, t)?;
        Ok(v.iter().map(|&x| x as u32).collect())
    }

    /// Verify `tokens` (the newest token and drafts) at the current position: the greedy next token after each.
    pub fn verify(&mut self, tokens: &[u32]) -> Result<Vec<u32>> {
        if self.pf.is_none() {
            self.pf = Some(PfBufs::new(&self.g, self.max_ctx)?);
            let rings: Vec<u64> = self.layers.iter().map(|l| l.ring.ptr).collect();
            self.g.htod(self.pf.as_ref().unwrap().ringtab.ptr, &rings)?;
        }
        anyhow::ensure!(tokens.len() <= 16);
        self.run_chunk(tokens, true)
    }

    /// After a verify of `t` tokens at the current position: keep the first `keep` (1..=t) positions.
    pub fn commit(&mut self, t: usize, keep: usize) -> Result<()> {
        let p0 = self.pos;
        let g = &self.g;
        let pf = self.pf.as_ref().unwrap();
        // window rows of the kept positions into every layer's ring
        g.launch("commit_rings", (self.layers.len() as u32, keep as u32, 1), 256, 0, &[Arg::P(pf.ringtab.ptr), Arg::P(pf.vlin.ptr), Arg::I(p0 as i32)])?;
        for l in self.layers.iter() {
            // ratio-2 compressor state: slot 0 must hold the newest kept even position
            if l.kv_src && l.ratio == 2 {
                let p = p0 + keep - 1;
                if (p + 1) % 2 != 0 {
                    let cs = l.cstate.as_ref().unwrap().ptr;
                    g.dtod(cs, pf.ckv.at((keep - 1) * HD * 4), HD * 4)?;
                    g.dtod(cs + (HD * 4) as u64, pf.cg.at((keep - 1) * HD * 4), HD * 4)?;
                }
            }
        }
        self.hist_tokens.truncate(self.hist_tokens.len() - (t - keep));
        self.pos = p0 + keep;
        g.htod(self.st.ptr, &[self.pos as i32])?;
        Ok(())
    }
}

impl Decoder {
    /// Logits of every position of the last prompt chunk (prefill eval): final norm per row, head on hipBLAS
    /// (needs build_bdense), downloaded row by row into `f(row, logits)`.
    pub fn chunk_logits(&self, t: usize, f: &mut dyn FnMut(usize, &[f32]) -> Result<()>) -> Result<()> {
        let g = &self.g;
        let pf = self.pf.as_ref().unwrap();
        let c = &self.cfg;
        let last = &self.layers[self.layers.len() - 1];
        let rows = 64;
        let (xh, xl, lg) = (g.alloc(rows * D * 2)?, g.alloc(rows * D * 2)?, g.alloc(rows * VOCAB * 4)?);
        for r0 in (0..t).step_by(rows) {
            let n = rows.min(t - r0);
            g.launch("hc_final_b", (n as u32, 1, 1), 256, 0, &[Arg::P(pf.h.at(r0 * HC * D * 4)), Arg::P(pf.mix_f.at(r0 * 80 * 32 * 4)), Arg::B(bytes(&last.hc_ffn)), Arg::P(self.out_norm.ptr), Arg::F(c.hc_eps), Arg::F(c.eps), Arg::I(c.sinkhorn_iters as i32), Arg::P(xh.ptr), Arg::P(xl.ptr)])?;
            let hj = crate::verify::VJob { w: self.head.w.ptr, n: VOCAB as i32, k: D as i32, fmt: 4, xb: xh.ptr, xl: xl.ptr, ldx: D as i32, ntok: n as i32, y: lg.ptr, ldy: VOCAB as i32, split: 1, mul: 1.0, ..Default::default() };
            anyhow::ensure!(self.bgemm("chunk head", &[hj])?, "chunk_logits needs build_bdense");
            let v: Vec<f32> = g.dtoh(lg.ptr, n * VOCAB)?;
            for i in 0..n {
                f(r0 + i, &v[i * VOCAB..(i + 1) * VOCAB])?;
            }
        }
        Ok(())
    }
}

impl Decoder {
    /// Batched DSpark draft, draft layer li: block kv rows into b.dblk, attention over each token's slot draft ring
    /// (positions before its block) and its whole block, inverse RoPE + FP8 for wo_a
    fn batch_attn_draft(&self, b: &crate::batch::Batch, li: usize, t: usize, rf: u64) -> Result<()> {
        let g = &self.g;
        let pf = self.pf.as_ref().unwrap();
        let l = &self.dspark.as_ref().unwrap().layers[li];
        g.launch("kv_bt", (t as u32, 1, 1), 512, 0, &[Arg::P(pf.kvraw.ptr), Arg::P(l.kv_norm.ptr), Arg::P(rf), Arg::P(b.bpos.ptr), Arg::P(b.dblk.ptr)])?;
        g.launch(
            "attn_bd",
            (t as u32, 1, 1),
            1024,
            0,
            &[Arg::P(b.dstart.ptr), Arg::P(b.bslot.ptr), Arg::P(pf.q.ptr), Arg::P(b.drings[li].ptr), Arg::P((SRING * HD) as u64), Arg::P(b.dblk.ptr), Arg::P(b.dcidx.ptr), Arg::I(b.dblock as i32), Arg::F(1.0 / (HD as f32).sqrt()), Arg::P(l.sinks.ptr)],
        )?;
        g.launch("attn_out_b", (t as u32, (NH / 8) as u32, 1), 256, 0, &[Arg::P(b.bpos.ptr), Arg::P(pf.q.ptr), Arg::P(rf), Arg::P(pf.oq.ptr), Arg::P(pf.os.ptr), Arg::P(pf.oqb.ptr)])
    }

    /// After a batched verify: record the main model's residual (main_hidden, captured for every verify row) of the
    /// kept rows `rows` (positions `pos`, slots `slots`) in the slots' draft rings, as observe does for one sequence
    pub(crate) fn batch_observe(&self, b: &crate::batch::Batch, t: usize, rows: &[i32], pos: &[i32], slots: &[i32]) -> Result<()> {
        let ds = self.dspark.as_ref().unwrap();
        let g = &self.g;
        let pf = self.pf.as_ref().unwrap();
        let n = rows.len();
        if n == 0 {
            return Ok(());
        }
        let (mh, aq, as_, y, kvraw, _) = pf.dspark_bufs();
        g.htod(b.orow.ptr, rows)?;
        g.htod(b.bpos.ptr, pos)?;
        g.htod(b.bslot.ptr, slots)?;
        // main_proj(main_hidden of the kept rows) -> rmsnorm(main_norm) -> FP8
        self.prep(4, mh, 0, 3 * D, 0, 0, 0, 3 * D, t, aq, as_, 0, 0)?;
        let j = crate::verify::VJob { w: ds.main_proj.w.ptr, sc: ds.main_proj.sc.as_ref().unwrap().ptr, fmt: 0, n: D as i32, k: (3 * D) as i32, xq: aq, xs: as_, ldx: (3 * D) as i32, rows: b.orow.ptr, ntok: n as i32, y, ldy: D as i32, mul: 1.0, ..Default::default() };
        self.blaunch("dspark main_proj", vec![j])?;
        self.prep(1, y, 0, D, ds.main_norm.ptr, 0, 0, D, n, aq, as_, 0, 0)?;
        for (li, l) in ds.layers.iter().enumerate() {
            let j = crate::verify::VJob { w: l.wkv.w.ptr, sc: l.wkv.sc.as_ref().unwrap().ptr, fmt: 0, n: HD as i32, k: D as i32, xq: aq, xs: as_, ldx: D as i32, ntok: n as i32, y: kvraw, ldy: HD as i32, mul: 1.0, ..Default::default() };
            self.blaunch("dspark kv", vec![j])?;
            g.launch("kv_b", (n as u32, 1, 1), 512, 0, &[Arg::P(kvraw), Arg::P(l.kv_norm.ptr), Arg::P(self.rope_f.ptr), Arg::P(b.bpos.ptr), Arg::P(b.bslot.ptr), Arg::P(b.drings[li].ptr), Arg::P((SRING * HD) as u64)])?;
        }
        Ok(())
    }

    /// Batched decode, layer li after the q / kv / compressor projections: window row into each slot's ring,
    /// compressor + indexer keys, indexer top-k per token, sparse attention, inverse RoPE + FP8 for wo_a
    fn batch_attn(&self, b: &crate::batch::Batch, li: usize, t: usize, rf: u64, wscale: f32) -> Result<()> {
        let g = &self.g;
        let c = &self.cfg;
        let pf = self.pf.as_ref().unwrap();
        let l = &self.layers[li];
        let (bp, bs) = (b.bpos.ptr, b.bslot.ptr);
        let ring_stride = (SRING * HD) as i64;
        g.launch("kv_b", (t as u32, 1, 1), 512, 0, &[Arg::P(pf.kvraw.ptr), Arg::P(l.kv_norm.ptr), Arg::P(rf), Arg::P(bp), Arg::P(bs), Arg::P(b.rings[li].ptr), Arg::P((ring_stride) as u64)])?;
        if l.kv_src {
            let r = b.rows[li] as i64;
            g.launch(
                "comp_b",
                (t as u32, 1, 1),
                512,
                0,
                &[Arg::P(bp), Arg::P(bs), Arg::I(l.ratio as i32), Arg::P(pf.ckv.ptr), Arg::P(pf.cg.ptr), Arg::P(b.cst[li].as_ref().unwrap().ptr), Arg::P((4 * HD as i64) as u64), Arg::P(l.comp_norm.as_ref().unwrap().ptr), Arg::P(l.idx_wk.as_ref().unwrap().ptr), Arg::P(l.idx_knorm.as_ref().unwrap().ptr), Arg::P(b.comp[li].as_ref().unwrap().ptr), Arg::P((r * CROW_B as i64) as u64), Arg::P(b.idx[li].as_ref().unwrap().ptr), Arg::P((r * IROW_B as i64) as u64), Arg::P(self.rope_fc.ptr), Arg::F(c.eps), Arg::P(if b.multi { b.bprev.ptr } else { 0 }), Arg::P(if b.multi { b.stage[li].as_ref().map(|x| x.ptr).unwrap_or(0) } else { 0 })],
            )?;
        }
        let maxp = b.step_pos.iter().copied().max().unwrap_or(0) + 1;
        if l.idx_src {
            let ks = l.idxk_src;
            let kc = b.idx[ks].as_ref().unwrap().ptr;
            let kstride = (b.rows[ks] * IROW_B) as i64;
            let after = li > c.cand_layer;
            let lds = b.lds() as i32;
            let pend = if b.gbucket.get() > 0 { b.gbucket.get() * 16384 } else { maxp };
            let chunks = ((pend / l.ratio.max(1)).max(16384)).div_ceil(1024) as u32;
            let (rows, nrows) = if after { (pf.keep.ptr, pf.nkeep.ptr) } else { (0, 0) };
            let isc = |grid: u32, rows: u64, nrows: u64| {
                g.launch("iscore_b", (grid, t as u32, 1), 256, 0, &[Arg::P(bp), Arg::P(bs), Arg::I(l.ratio as i32), Arg::P(pf.iq.ptr), Arg::P(pf.iw.ptr), Arg::F(wscale), Arg::P(kc), Arg::P((kstride) as u64), Arg::P(rows), Arg::P(nrows), Arg::I(16384), Arg::P(b.score.ptr), Arg::I(lds)])
            };
            isc(chunks, rows, nrows)?;
            if li == c.cand_layer {
                g.launch("icand_b", (t as u32, 1, 1), 1024, 0, &[Arg::P(bp), Arg::I(l.ratio as i32), Arg::I(c.cand_block as i32), Arg::I(c.cand_blocks as i32), Arg::P(b.score.ptr), Arg::I(lds), Arg::P(b.bsc.ptr), Arg::P(b.kb.ptr), Arg::P(pf.keep.ptr), Arg::P(pf.nkeep.ptr)])?;
                isc(16, pf.keep.ptr, pf.nkeep.ptr)?;
            }
            let use_keep = after || li == c.cand_layer;
            let (rows, nrows) = if use_keep { (pf.keep.ptr, pf.nkeep.ptr) } else { (0, 0) };
            g.launch("itopk_b", (t as u32, 1, 1), 1024, 0, &[Arg::P(bp), Arg::I(l.ratio as i32), Arg::P(b.score.ptr), Arg::I(lds), Arg::P(rows), Arg::P(nrows), Arg::I(16384), Arg::I(c.idx_topk as i32), Arg::I(WIN as i32), Arg::P(pf.cidx.ptr), Arg::I(512), Arg::P(pf.ncomp.ptr), Arg::I(0)])?;
        }
        let (comp, cstride) = if l.ratio > 0 { (b.comp[l.comp_src].as_ref().unwrap().ptr, (b.rows[l.comp_src] * CROW_B) as i64) } else { (0, 0) };
        // small batches: each token's rows split over ATTN_SPLITS blocks, then merged (one block per token leaves
        // the GPU nearly idle: ~155 us per layer for a single token)
        let split_max: usize = std::env::var("DSXA_ATTN_SPLIT").ok().and_then(|v| v.parse().ok()).unwrap_or(128usize).min(128);
        if t <= split_max {
            g.launch(
                if b.attn_fine { "attn_bs_fine" } else { "attn_bs" },
                (t as u32, if b.attn_fine {20} else {10}, 1),
                1024,
                0,
                &[Arg::P(bp), Arg::P(bs), Arg::P(pf.q.ptr), Arg::P(b.rings[li].ptr), Arg::P((ring_stride) as u64), Arg::P(comp), Arg::P((cstride) as u64), Arg::P(pf.cidx.ptr), Arg::P(pf.ncomp.ptr), Arg::I(512), Arg::I(WIN as i32), Arg::F(1.0 / (HD as f32).sqrt()), Arg::P(l.sinks.ptr), Arg::P(b.apart.ptr)],
            )?;
            let merge_out = match MERGE_OUT_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed) {
                1 => false,
                2 => true,
                _ => std::env::var("DSXA_ATTN_MERGE_OUT").as_deref() == Ok("1"),
            };
            if merge_out {
                g.launch(if b.attn_fine {"attn_merge_out_b_fine"} else {"attn_merge_out_b"}, (t as u32, NH as u32, 1), 128, 0,
                    &[Arg::P(pf.q.ptr), Arg::P(b.apart.ptr), Arg::P(l.sinks.ptr), Arg::P(bp), Arg::P(rf), Arg::P(pf.oq.ptr), Arg::P(pf.os.ptr), Arg::P(pf.oqb.ptr)])?;
                return Ok(());
            }
            g.launch(if b.attn_fine {"attn_merge_b_fine"} else {"attn_merge_b"}, (t as u32, NH as u32, 1), 512, 0, &[Arg::P(pf.q.ptr), Arg::P(b.apart.ptr), Arg::P(l.sinks.ptr)])?;
        } else {
        g.launch(
            "attn_b",
            (t as u32, 1, 1),
            1024,
            0,
            &[Arg::P(bp), Arg::P(bs), Arg::P(pf.q.ptr), Arg::P(b.rings[li].ptr), Arg::P((ring_stride) as u64), Arg::P(comp), Arg::P((cstride) as u64), Arg::P(pf.cidx.ptr), Arg::P(pf.ncomp.ptr), Arg::I(512), Arg::I(WIN as i32), Arg::F(1.0 / (HD as f32).sqrt()), Arg::P(l.sinks.ptr)],
        )?;
        }
        g.launch("attn_out_b", (t as u32, (NH / 8) as u32, 1), 256, 0, &[Arg::P(bp), Arg::P(pf.q.ptr), Arg::P(rf), Arg::P(pf.oq.ptr), Arg::P(pf.os.ptr), Arg::P(pf.oqb.ptr)])
    }
}

fn bytes<T: Copy>(v: &T) -> Vec<u8> {
    unsafe { std::slice::from_raw_parts(v as *const T as *const u8, std::mem::size_of::<T>()).to_vec() }
}
