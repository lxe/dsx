//! Per-layer expert handoff through mapped host memory, with no host round trips inside a token.
//!
//! GPU0 publishes each layer's picks and expert input (`publish_route`) and raises `route`. The CPU worker thread
//! (spinning on core 16) computes the RAM-resident experts and raises `cpu`. Each aux GPU's stream waits on `route`
//! (cuStreamWaitValue32), runs its resident experts and raises its own flag (`aux_flag`). GPU0 waits on all of them
//! and adds everything (`combine_experts`). Flags are sequence numbers: token k, layer l -> k * n_layers + l + 1.

use crate::cpu::moe::{CpuMoe, ExpertW};
use crate::cpu::pool::{P_CORES, Pool, pin_to};
use crate::gpu::HostBuf;
use crate::model::Model;
use anyhow::Result;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::thread::JoinHandle;
use std::time::Duration;

pub const ROUTE: usize = 0;
pub const CPU: usize = 64;
/// aux GPU 0's flag; further aux GPUs use the free lines between WTS and X (`aux_flag`)
pub const GPU1: usize = 128;
pub const NTOK: usize = 192;
pub const SEL: usize = 256;
pub const WTS: usize = 512;
pub const X: usize = 1024;
/// tokens per handoff (verify batches)
pub const MAXT: usize = 8;
/// aux GPUs the exchange has flags and result slots for
pub const MAX_AUX: usize = 5;

/// flag offset of aux GPU `i`
pub fn aux_flag(i: usize) -> usize {
    assert!(i < MAX_AUX);
    if i == 0 { GPU1 } else { 768 + 64 * (i - 1) }
}

/// result slot offset of aux GPU `i` ([MAXT][dim] f32; combine_experts reads the same layout)
pub fn aux_y(i: usize, dim: usize) -> usize {
    X + (64 + 32 * i) * dim
}

/// DSX_DEFER: the CPU's deferred results alternate between two slots by layer (even layers here, odd in y_cpu3), after
/// the aux slots; y_cpu keeps the CPU's immediate experts
pub fn y_cpu2(dim: usize) -> usize {
    aux_y(MAX_AUX, dim)
}
pub fn y_cpu3(dim: usize) -> usize {
    y_cpu2(dim) + MAXT * 4 * dim
}

/// DSX_DEFER: the CPU raises this to a layer's sequence number when its deferred experts are in their slot
pub const CPUD: usize = DIAG - 128;

/// DSX_DEFER: the CPU raises this to a layer's sequence number once it has copied that layer's picks and input, so the
/// main GPU may publish the next layer over them
pub const XACK: usize = DIAG - 64;

/// DSX_DEFER=R: in decode, the CPU's experts the router ranked R or lower (0 = best of the 6) are added in layer L+1
/// instead of layer L, so the main GPU does not wait for them inside the layer; the last layer waits for everything.
/// Changes outputs (expert deferral as in KTransformers, SOSP '25, which keeps the top-ranked experts immediate).
pub fn defer() -> Option<usize> {
    static D: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *D.get_or_init(|| std::env::var("DSX_DEFER").ok().and_then(|v| v.parse().ok()))
}

pub struct Xch {
    pub buf: HostBuf,
    pub dim: usize,
}

impl Xch {
    pub fn new(dim: usize) -> Result<Xch> {
        let end = y_cpu3(dim) + MAXT * 4 * dim;
        anyhow::ensure!(end <= DIAG, "exchange: dim {dim} too large for {MAX_AUX} aux slots below DIAG");
        Ok(Xch { buf: HostBuf::new(DIAG + 16 * 64)?, dim }) // 64 layers of diagnostics
    }
    pub fn at<T>(&self, off: usize) -> *mut T {
        unsafe { self.buf.host_ptr().add(off) as *mut T }
    }
    pub fn flag(&self, off: usize) -> &AtomicU32 {
        unsafe { &*(self.at::<AtomicU32>(off)) }
    }
    /// tokens in the current handoff
    pub fn ntok(&self) -> usize {
        unsafe { std::ptr::read_volatile(self.at::<i32>(NTOK)) as usize }
    }
    /// [t][k] picks / weights of `n` tokens
    pub fn sel(&self, n: usize) -> &[i32] {
        unsafe { std::slice::from_raw_parts(self.at(SEL), n) }
    }
    pub fn wts(&self, n: usize) -> &[f32] {
        unsafe { std::slice::from_raw_parts(self.at(WTS), n) }
    }
    pub fn x(&self, t: usize) -> &[f32] {
        unsafe { std::slice::from_raw_parts(self.at(X), t * self.dim) }
    }
    #[allow(clippy::mut_from_ref)]
    pub fn y_cpu(&self, t: usize) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.at(X + 32 * self.dim), t * self.dim) }
    }
    /// deferred result slot of layer `l` under DSX_DEFER (even layers: y_cpu2, odd: y_cpu3)
    #[allow(clippy::mut_from_ref)]
    pub fn y_cpu_deferred(&self, l: usize, t: usize) -> &mut [f32] {
        let off = if l % 2 == 0 { y_cpu2(self.dim) } else { y_cpu3(self.dim) };
        unsafe { std::slice::from_raw_parts_mut(self.at(off), t * self.dim) }
    }
    #[allow(clippy::mut_from_ref)]
    pub fn y_aux(&self, i: usize, t: usize) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.at(aux_y(i, self.dim)), t * self.dim) }
    }
}

/// A prefill layer's share of routed experts for the CPU worker (prefill_moe): token windows of at most MAX_TOKENS
/// rows, each with its experts and their [expert][t] routing weights over the window's rows; x [n][dim] in; out
/// [n][dim] = the sum of the CPU experts' outputs.
#[derive(Default)]
pub struct PrefillJob {
    pub layer: usize,
    pub n: usize,
    pub windows: Vec<(usize, usize, Vec<usize>, Vec<f32>)>,
    pub x: Vec<f32>,
    pub out: Vec<f32>,
}

/// The worker takes a submitted job between decode handoffs (prefill has none): state 0 idle, 1 submitted, 2 done.
#[derive(Default)]
pub struct PrefillChan {
    pub state: AtomicU32,
    pub job: std::sync::Mutex<PrefillJob>,
}

impl PrefillChan {
    pub fn submit(&self, job: PrefillJob) {
        *self.job.lock().unwrap() = job;
        self.state.store(1, Ordering::Release);
    }
    /// wait for the submitted job and take it back (with its out)
    pub fn wait(&self) -> PrefillJob {
        while self.state.load(Ordering::Acquire) != 2 {
            std::hint::spin_loop();
        }
        let j = std::mem::take(&mut *self.job.lock().unwrap());
        self.state.store(0, Ordering::Release);
        j
    }
}

pub struct CpuWorker {
    pub pre: Arc<PrefillChan>,
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
    pub busy_ns: Arc<std::sync::atomic::AtomicU64>,
    /// [idle between layers, prep (copy + union), forward] in ns, summed
    pub split_ns: Arc<[std::sync::atomic::AtomicU64; 3]>,
    /// per layer: [sum of (CPU wall clock when the picks were seen - GPU globaltimer at publish) ns, count]
    pub leg1_ns: Arc<Vec<[std::sync::atomic::AtomicU64; 2]>>,
    /// [layer * E + expert]: the CPU computes it (not resident on a GPU); changed only between tokens
    pub on_cpu: Arc<Vec<AtomicBool>>,
    /// [layer * E + expert]: picks, decayed x decay() (0.97) per decode pass (f32 bits), for adaptive GPU residency
    pub score: Arc<Vec<AtomicU32>>,
    /// [routed picks, picks computed by the CPU]
    pub picks: Arc<[std::sync::atomic::AtomicU64; 2]>,
}

/// u64 slots next to the flags: GPU publish time (ROUTE + 8), CPU store time (CPU + 8); per-layer GPU-side
/// handoff sums at DIAG + 16 * layer
pub const DIAG: usize = 2 << 20;

/// usage score decay per decode pass (DSX_DECAY, default 0.97)
pub fn decay() -> f32 {
    static D: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *D.get_or_init(|| std::env::var("DSX_DECAY").ok().and_then(|v| v.parse().ok()).unwrap_or(0.97))
}

pub fn wall_ns() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos() as u64)
}

impl CpuWorker {
    /// `on_cpu[l][e]`: the CPU computes expert e of layer l (not resident on a GPU).
    pub fn spawn(m: &'static Model, xch: &'static Xch, on_cpu: Vec<Vec<bool>>, top_k: usize, arena: Option<&'static crate::experts::Arena>) -> CpuWorker {
        let ne = m.cfg.n_experts;
        let on_cpu: Arc<Vec<AtomicBool>> = Arc::new(on_cpu.iter().flatten().map(|&b| AtomicBool::new(b)).collect());
        let score: Arc<Vec<AtomicU32>> = Arc::new((0..on_cpu.len()).map(|_| AtomicU32::new(0)).collect());
        let (oc, sc) = (on_cpu.clone(), score.clone());
        let picks: Arc<[std::sync::atomic::AtomicU64; 2]> = Arc::new(Default::default());
        let pk = picks.clone();
        let stop = Arc::new(AtomicBool::new(false));
        let busy_ns = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let split_ns: Arc<[std::sync::atomic::AtomicU64; 3]> = Arc::new(Default::default());
        let leg1_ns: Arc<Vec<[std::sync::atomic::AtomicU64; 2]>> = Arc::new((0..m.cfg.n_layers).map(|_| Default::default()).collect());
        let (st, busy, split, leg1) = (stop.clone(), busy_ns.clone(), split_ns.clone(), leg1_ns.clone());
        let pre: Arc<PrefillChan> = Arc::new(PrefillChan::default());
        let pc = pre.clone();
        let handle = std::thread::Builder::new()
            .name("cpu-experts".into())
            .spawn(move || {
                pin_to(crate::cpu::pool::WORKER_CPU);
                // 14 threads on CPUs 0-13 (P-cores 0-6): 24 ms/token of CPU experts vs 30 with 8 threads (measured on the
                // 1017 quant); CPUs 14-15 stay for the thread issuing GPU work and helpers. DSX_CPU_THREADS overrides.
                // Spin 50 ms between jobs (decode layers come every ~1 ms), then sleep so prefill gets the cores.
                // DSX_CPU_LIST=0,2,4 pins the pool to exactly those CPUs (experiments with core layouts)
                let list: Option<Vec<usize>> = std::env::var("DSX_CPU_LIST").ok().map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect());
                let n = std::env::var("DSX_CPU_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(14usize).min(crate::cpu::pool::MAIN_CPU);
                let cpus: Vec<usize> = list.unwrap_or_else(|| P_CORES[..n].to_vec());
                let mut cm = CpuMoe::new(Pool::new(&cpus, Duration::from_millis(50)));
                let nl = m.cfg.n_layers;
                let d = m.cfg.dim;
                let mut seq = 0u32;
                let mut x = vec![0f32; d];
                let mut y = vec![0f32; d];
                let mut t_done = std::time::Instant::now();
                // profiling only (DSX_CPU_NOOP=1): skip the CPU experts to time the bare handoff (wrong output)
                let noop = std::env::var_os("DSX_CPU_NOOP").is_some();
                // DSX_LOG_ROUTES=path: one line per (pass, layer) with every token's picks, for placement studies
                let mut route_log = std::env::var("DSX_LOG_ROUTES").ok().and_then(|p| std::fs::File::create(p).ok()).map(std::io::BufWriter::new);
                loop {
                    seq += 1;
                    let l = (seq as usize - 1) % nl;
                    while xch.flag(ROUTE).load(Ordering::Acquire) < seq {
                        if st.load(Ordering::Relaxed) {
                            return;
                        }
                        if pc.state.load(Ordering::Acquire) == 1 {
                            // a prefill layer's CPU experts (no decode handoff is pending during prefill)
                            let mut job = std::mem::take(&mut *pc.job.lock().unwrap());
                            let ws = |l: usize, ids: &[usize]| -> Vec<ExpertW> {
                                ids.iter()
                                    .map(|&e| {
                                        let x = &m.layers[l].experts[e];
                                        match arena.and_then(|a| a.get(l, e)) {
                                            Some([g, u, dn]) => ExpertW { gu: x.gate.ty, gate: g, up: u, dn: x.down.ty, down: dn },
                                            None => ExpertW { gu: x.gate.ty, gate: x.gate.bytes(), up: x.up.bytes(), dn: x.down.ty, down: x.down.bytes() },
                                        }
                                    })
                                    .collect()
                            };
                            cm.rot = m.cfg.expert_rot[job.layer];
                            job.out.clear();
                            job.out.resize(job.n * d, 0.0);
                            for (t0, t, ids, w) in &job.windows {
                                if !ids.is_empty() && !noop {
                                    let ex = ws(job.layer, ids);
                                    cm.forward(&ex, d, &job.x[t0 * d..(t0 + t) * d], *t, w, &mut job.out[t0 * d..(t0 + t) * d]);
                                }
                            }
                            if std::env::var_os("DSX_PRE_CPU_CHECK").is_some() {
                                // debugging: the job again token by token (t = 1 calls), relative error vs the windows
                                let mut o2 = vec![0f32; job.n * d];
                                for (t0, t, ids, w) in &job.windows {
                                    if ids.is_empty() {
                                        continue;
                                    }
                                    let ex = ws(job.layer, ids);
                                    for j in 0..*t {
                                        let wj: Vec<f32> = (0..ids.len()).map(|i| w[i * t + j]).collect();
                                        let sel: Vec<usize> = (0..ids.len()).filter(|&i| wj[i] != 0.0).collect();
                                        if sel.is_empty() {
                                            continue;
                                        }
                                        let exj: Vec<ExpertW> = sel.iter().map(|&i| ex[i]).collect();
                                        let wjj: Vec<f32> = sel.iter().map(|&i| wj[i]).collect();
                                        let r = t0 + j;
                                        cm.forward(&exj, d, &job.x[r * d..(r + 1) * d], 1, &wjj, &mut o2[r * d..(r + 1) * d]);
                                    }
                                }
                                let num: f64 = job.out.iter().zip(&o2).map(|(a, b)| ((a - b) as f64).powi(2)).sum();
                                let den: f64 = o2.iter().map(|b| (*b as f64).powi(2)).sum();
                                let nexp: usize = job.windows.iter().map(|w| w.2.len()).sum();
                                let maxw = job.windows.iter().map(|w| w.2.len()).max().unwrap_or(0);
                                eprintln!("pre-cpu check layer {}: n {}, {} windows, {nexp} expert-windows (max {maxw}), windows vs single tokens rel {:.2e}", job.layer, job.n, job.windows.len(), (num / den.max(1e-30)).sqrt());
                            }
                            *pc.job.lock().unwrap() = job;
                            pc.state.store(2, Ordering::Release);
                        }
                        std::hint::spin_loop();
                    }
                    let t0 = std::time::Instant::now();
                    let gp = unsafe { std::ptr::read_volatile(xch.at::<u64>(ROUTE + 8)) };
                    leg1[l][0].fetch_add(wall_ns().wrapping_sub(gp), Ordering::Relaxed);
                    leg1[l][1].fetch_add(1, Ordering::Relaxed);
                    split[0].fetch_add((t0 - t_done).as_nanos() as u64, Ordering::Relaxed);
                    let nt = xch.ntok().clamp(1, MAXT);
                    x.resize(nt * d, 0.0);
                    y.resize(nt * d, 0.0);
                    x.copy_from_slice(xch.x(nt));
                    let (sel, wts) = (xch.sel(nt * top_k).to_vec(), xch.wts(nt * top_k).to_vec());
                    // inputs copied: under DSX_DEFER the main GPU may now publish the next layer over them
                    xch.flag(XACK).store(seq, Ordering::Release);
                    let defer_rank = if nt == 1 { defer() } else { None };
                    if let Some(f) = route_log.as_mut() {
                        use std::io::Write;
                        let _ = writeln!(f, "{l} {}", sel.iter().map(|e| e.to_string()).collect::<Vec<_>>().join(" "));
                        if l + 1 == nl {
                            let _ = f.flush();
                        }
                    }
                    // distinct CPU-side experts over the tokens, with per-token weights (0 = not routed there)
                    let mut experts: Vec<usize> = vec![];
                    let mut w: Vec<f32> = vec![];
                    let (mut dexperts, mut dw): (Vec<usize>, Vec<f32>) = (vec![], vec![]);
                    for t in 0..nt {
                        for j in 0..top_k {
                            let e = sel[t * top_k + j] as usize;
                            let a = &sc[l * ne + e];
                            a.store((f32::from_bits(a.load(Ordering::Relaxed)) + 1.0).to_bits(), Ordering::Relaxed);
                            pk[0].fetch_add(1, Ordering::Relaxed);
                            if !oc[l * ne + e].load(Ordering::Relaxed) {
                                continue;
                            }
                            pk[1].fetch_add(1, Ordering::Relaxed);
                            if defer_rank.is_some_and(|r| j >= r) {
                                // nt == 1 here: each expert appears once
                                dexperts.push(e);
                                dw.push(wts[j]);
                                continue;
                            }
                            let i = match experts.iter().position(|&x| x == e) {
                                Some(i) => i,
                                None => {
                                    experts.push(e);
                                    w.extend(std::iter::repeat_n(0.0, nt));
                                    experts.len() - 1
                                }
                            };
                            w[i * nt + t] = wts[t * top_k + j];
                        }
                    }
                    y.iter_mut().for_each(|v| *v = 0.0);
                    let tf = std::time::Instant::now();
                    split[1].fetch_add((tf - t0).as_nanos() as u64, Ordering::Relaxed);
                    let weights = |ids: &[usize]| -> Vec<ExpertW> {
                        ids.iter()
                            .map(|&e| {
                                let x = &m.layers[l].experts[e];
                                match arena.and_then(|a| a.get(l, e)) {
                                    Some([g, u, dn]) => ExpertW { gu: x.gate.ty, gate: g, up: u, dn: x.down.ty, down: dn },
                                    None => ExpertW { gu: x.gate.ty, gate: x.gate.bytes(), up: x.up.bytes(), dn: x.down.ty, down: x.down.bytes() },
                                }
                            })
                            .collect()
                    };
                    cm.rot = m.cfg.expert_rot[l];
                    if !experts.is_empty() && !noop {
                        cm.forward(&weights(&experts), d, &x, nt, &w, &mut y);
                    }
                    split[2].fetch_add(tf.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    xch.y_cpu(nt).copy_from_slice(&y);
                    unsafe { std::ptr::write_volatile(xch.at::<u64>(CPU + 8), wall_ns()) };
                    xch.flag(CPU).store(seq, Ordering::Release);
                    if defer_rank.is_some() {
                        // the deferred experts, while the main GPU runs the next layer; it adds them there
                        y.iter_mut().for_each(|v| *v = 0.0);
                        if !dexperts.is_empty() && !noop {
                            cm.forward(&weights(&dexperts), d, &x, nt, &dw, &mut y);
                        }
                        xch.y_cpu_deferred(l, nt).copy_from_slice(&y);
                        xch.flag(CPUD).store(seq, Ordering::Release);
                    }
                    if l + 1 == nl {
                        // end of a pass: decay the usage scores (after the flag, off the critical path)
                        let dk = decay();
                        for a in sc.iter() {
                            let v = f32::from_bits(a.load(Ordering::Relaxed));
                            if v != 0.0 {
                                a.store(if v < 0.01 { 0.0 } else { v * dk }.to_bits(), Ordering::Relaxed);
                            }
                        }
                    }
                    t_done = std::time::Instant::now();
                    busy.fetch_add((t_done - t0).as_nanos() as u64, Ordering::Relaxed);
                }
            })
            .expect("spawn cpu-experts");
        CpuWorker { pre, stop, handle: Some(handle), busy_ns, split_ns, leg1_ns, on_cpu, score, picks }
    }
}

impl Drop for CpuWorker {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}
