//! DSX_ENGRAM_PERSIST=1: decode's engram rows gathered by persistent threads (one per AUX CPU, pinned once) instead of
//! a helper thread per token that spawns 8 more per engram layer. The main thread hashes both engram layers' rows at the
//! start of the token and publishes them, layer 1's first; the workers take rows by an atomic index and count each
//! layer's finished rows, so layer 1 waits only for its own rows while the later layer's finish behind layers 1..13.
//! Rows not in the page cache are NVMe reads: each worker first asks the kernel to read its share of the token's rows
//! ahead (madvise WILLNEED, all queued at once) before copying any (DSX_ENGRAM_WILLNEED=0 skips that).
use crate::model::Model;
use anyhow::{Result, bail};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering::*};
use std::time::{Duration, Instant};

unsafe extern "C" {
    fn madvise(addr: *mut std::ffi::c_void, len: usize, advice: i32) -> i32;
    fn mincore(addr: *mut std::ffi::c_void, len: usize, vec: *mut u8) -> i32;
}
const MADV_WILLNEED: i32 = 3;
const PAGE: usize = 4096;
/// rows a token gathers, at most (2 layers x 24 rows for DeepSeek-V4.1-Flash)
const MAX_ROWS: usize = 256;

fn pages(b: &[u8]) -> (*mut std::ffi::c_void, usize) {
    let a = b.as_ptr() as usize & !(PAGE - 1);
    let e = (b.as_ptr() as usize + b.len()).div_ceil(PAGE) * PAGE;
    (a as *mut std::ffi::c_void, e - a)
}

/// where the rows come from: the model's engram tables, or a test table
pub trait RowSource: Send + Sync {
    /// (engram layers, rows per layer per token, row width)
    fn shape(&self) -> (usize, usize, usize);
    /// layer `l`'s rows for the token history `hist`
    fn hashes(&self, hist: &[i64], l: usize) -> Vec<u64>;
    /// the bytes row(l, r) reads (for readahead and residency)
    fn row_bytes(&self, l: usize, r: usize) -> &[u8];
    fn row(&self, l: usize, r: usize, out: &mut [f32]);
}

struct ModelRows {
    m: &'static Model,
    tabs: Vec<crate::model::EngramLayer<'static>>,
}

impl RowSource for ModelRows {
    fn shape(&self) -> (usize, usize, usize) {
        let c = &self.m.cfg;
        (c.engram_layers.len(), (c.engram_ngram - 1) * c.engram_heads, c.engram_dim)
    }
    fn hashes(&self, hist: &[i64], l: usize) -> Vec<u64> {
        self.m.engram_hashes_for(hist, l)
    }
    fn row_bytes(&self, l: usize, r: usize) -> &[u8] {
        self.tabs[l].row_bytes(r)
    }
    fn row(&self, l: usize, r: usize, out: &mut [f32]) {
        self.tabs[l].row(r, out)
    }
}

struct Shared {
    src: Box<dyn RowSource>,
    out: usize,
    out_len: usize,
    per: usize,
    layers: usize,
    dim: usize,
    willneed: bool,
    /// job: bumped after `rows`, `next` and `done` are set
    generation: AtomicU64,
    rows: Vec<AtomicU64>,
    next: AtomicUsize,
    done: Vec<AtomicUsize>,
    /// DSX_PROF: rows not resident when gathered, and each layer's gather time since publication (ns)
    prof: bool,
    cold: AtomicU32,
    t_pub: AtomicU64,
    t_done: Vec<AtomicU64>,
    epoch: Instant,
    stop: AtomicBool,
}

pub struct EngramGather {
    sh: Arc<Shared>,
    threads: Vec<std::thread::JoinHandle<()>>,
}

impl EngramGather {
    /// `out`: the pinned host buffer the engram stage copies from ([layer][per][dim] f32)
    pub fn new(m: &Model, out: &mut [f32], prof: bool) -> Result<EngramGather> {
        // the GpuModel that owns the gather borrows the model, and drops the gather first
        let m: &'static Model = unsafe { std::mem::transmute::<&Model, &'static Model>(m) };
        let tabs = m.engram.as_ref().ok_or_else(|| anyhow::anyhow!("no engram tables"))?;
        let tabs = (0..m.cfg.engram_layers.len()).map(|l| tabs.layer(l, m.cfg.engram_dim)).collect::<Result<Vec<_>>>()?;
        Self::with_source(Box::new(ModelRows { m, tabs }), out, prof, &crate::cpu::pool::AUX_CPUS)
    }

    /// one worker per CPU in `cpus`, each pinned to it
    pub fn with_source(src: Box<dyn RowSource>, out: &mut [f32], prof: bool, cpus: &[usize]) -> Result<EngramGather> {
        let (layers, per, dim) = src.shape();
        if per * layers > MAX_ROWS || out.len() < per * layers * dim {
            bail!("engram gather: {layers} x {per} rows do not fit");
        }
        let sh = Arc::new(Shared {
            src,
            out: out.as_mut_ptr() as usize,
            out_len: out.len(),
            per,
            layers,
            dim,
            willneed: !std::env::var("DSX_ENGRAM_WILLNEED").is_ok_and(|v| v == "0"),
            generation: AtomicU64::new(0),
            rows: (0..MAX_ROWS).map(|_| AtomicU64::new(0)).collect(),
            next: AtomicUsize::new(usize::MAX),
            done: (0..layers).map(|_| AtomicUsize::new(0)).collect(),
            prof,
            cold: AtomicU32::new(0),
            t_pub: AtomicU64::new(0),
            t_done: (0..layers).map(|_| AtomicU64::new(0)).collect(),
            epoch: Instant::now(),
            stop: AtomicBool::new(false),
        });
        let threads = cpus
            .iter()
            .enumerate()
            .map(|(w, &cpu)| {
                let (sh, nw) = (sh.clone(), cpus.len());
                std::thread::Builder::new().name(format!("engram{w}")).spawn(move || worker(&sh, w, nw, cpu)).map_err(anyhow::Error::from)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(EngramGather { sh, threads })
    }

    /// hash this token's rows (layer 1's first) and hand them to the workers; the previous token's rows must have been
    /// consumed (every engram layer waited)
    pub fn publish(&self, hist: &[i64]) {
        let sh = &*self.sh;
        for (l, d) in sh.done.iter().enumerate() {
            d.store(0, Relaxed);
            let ids = sh.src.hashes(hist, l);
            debug_assert_eq!(ids.len(), sh.per);
            for (i, &r) in ids.iter().enumerate() {
                sh.rows[l * sh.per + i].store(r, Relaxed);
            }
        }
        if sh.prof {
            sh.cold.store(0, Relaxed);
            sh.t_pub.store(sh.epoch.elapsed().as_nanos() as u64, Relaxed);
        }
        sh.next.store(0, Release);
        sh.generation.fetch_add(1, Release);
        for t in &self.threads {
            t.thread().unpark();
        }
    }

    /// wait until engram layer `l`'s rows are in `out`
    pub fn wait(&self, l: usize) -> Result<()> {
        let sh = &*self.sh;
        let t0 = Instant::now();
        while sh.done[l].load(Acquire) < sh.per {
            std::hint::spin_loop();
            if t0.elapsed() > Duration::from_secs(10) {
                bail!("engram gather: layer {l}'s rows not done after 10 s");
            }
        }
        Ok(())
    }

    /// DSX_PROF: (rows that were not in the page cache, layer `l`'s gather time in s) for the last token
    pub fn stats(&self, l: usize) -> (u32, f64) {
        let sh = &*self.sh;
        (sh.cold.load(Relaxed), sh.t_done[l].load(Relaxed) as f64 * 1e-9)
    }
}

impl Drop for EngramGather {
    fn drop(&mut self) {
        self.sh.stop.store(true, Release);
        for t in self.threads.drain(..) {
            t.thread().unpark();
            let _ = t.join();
        }
    }
}

fn worker(sh: &Shared, w: usize, nw: usize, cpu: usize) {
    crate::cpu::pool::pin_to_set(&[cpu]);
    let hd = sh.dim;
    let out = unsafe { std::slice::from_raw_parts_mut(sh.out as *mut f32, sh.out_len) };
    let n = sh.per * sh.layers;
    let mut seen = 0u64;
    loop {
        // wait for a job: spin briefly, then park (unpark() from publish wakes it)
        let mut spins = 0u32;
        loop {
            if sh.stop.load(Acquire) {
                return;
            }
            let g = sh.generation.load(Acquire);
            if g != seen {
                seen = g;
                break;
            }
            if spins < 4096 {
                std::hint::spin_loop();
                spins += 1;
            } else {
                std::thread::park_timeout(Duration::from_millis(50));
            }
        }
        let row_of = |i: usize| (i / sh.per, sh.rows[i].load(Relaxed) as usize);
        if sh.willneed || sh.prof {
            // this worker's share of the token's rows (layer 1's first): DSX_PROF counts the ones not in the page cache
            // (rows another worker already copied count as cached), then the reads are queued before blocking on any
            for i in (w..n).step_by(nw) {
                let (l, r) = row_of(i);
                let (a, len) = pages(sh.src.row_bytes(l, r));
                if sh.prof {
                    let mut v = vec![0u8; len / PAGE];
                    if unsafe { mincore(a, len, v.as_mut_ptr()) } == 0 && v.iter().any(|&b| b & 1 == 0) {
                        sh.cold.fetch_add(1, Relaxed);
                    }
                }
                if sh.willneed {
                    unsafe { madvise(a, len, MADV_WILLNEED) };
                }
            }
        }
        loop {
            let i = sh.next.fetch_add(1, AcqRel);
            if i >= n {
                break;
            }
            let (l, r) = row_of(i);
            let o = (l * sh.per + i % sh.per) * hd;
            sh.src.row(l, r, &mut out[o..o + hd]);
            if sh.done[l].fetch_add(1, Release) + 1 == sh.per && sh.prof {
                sh.t_done[l].store((sh.epoch.elapsed().as_nanos() as u64).saturating_sub(sh.t_pub.load(Relaxed)), Relaxed);
            }
        }
    }
}
