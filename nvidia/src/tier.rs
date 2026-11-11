//! Automatic expert tiering across media, learned from use, applied while the server runs and kept across restarts.
//!
//! Tiers, fastest first: GPU stores, the RAM arena, the page cache (file pages kept in RAM), an SSD buffer (when the
//! model file sits on a slow medium) and the model file itself. Nothing here changes which experts the router picks;
//! it changes where their bytes are and which device computes them.
//!
//! * **Usage.** The CPU worker counts every decode pick (`note_picks`), GPU-resident ones included, and which tier
//!   served it. A background thread folds the counts into a long-term score (`HALF_LIFE_TOKENS`). Usage does not
//!   depend on the hardware: one score serves every GPU combination the model is started with.
//! * **Who is waited for.** Every 64th decode token one layer (rotating) is timed on the main GPU: after its own work
//!   it waits for the CPU's experts, then for each aux GPU (`Waits`). The device that finishes last gets work taken
//!   away: a late aux GPU exchanges its hottest experts for colder ones of a fixed-set GPU that is not late
//!   (`plan_exchanges`); a late CPU hands hot experts to fixed-set GPU stores that are not late (`plan_moves`).
//!   Adaptive stores keep an arena copy of every resident (they hand experts back to the CPU at any token) and are
//!   left to their own rule. A GPU resident gets its arena copy before a move hands it back to the CPU, so the CPU
//!   never computes from the file side because of tiering.
//! * **Arena.** In pauses, experts the CPU reads from the file side take the place of arena copies nobody reads
//!   (experts resident on fixed-set GPUs), then of cold ones (`plan_arena`, `Arena::replace` on the engine thread), so
//!   what is in use ends up on GPUs and in RAM.
//! * **Run time.** The plans' experts are copied into pinned prefetch slots by this thread (from RAM when it holds
//!   them); the engine starts them while a request runs (at most one per `WORK_MOVE_EVERY` tokens) and in pauses
//!   back to back (`GpuModel::tier_step`, `tier_idle`), with the
//!   protocol of adaptive replacement: a victim leaves its table first and the CPU computes it meanwhile; a new expert
//!   enters a table only after its copy completed. No expert is ever resident on two GPUs.
//! * **Pauses.** In every pause of `IDLE_MS` (no request running) this thread also warms the page cache with the hottest
//!   experts the CPU reads from the file side and, on a slow medium, copies them into an SSD buffer.
//! * **Measurement.** Per GPU (by UUID): host link, expert-kernel time of a layer, picks per token; per medium: read
//!   speed; per GPU combination: decode tokens/s, the waits, CPU picks per token.
//! * **Startup.** The learned score is blended into `--rank` before placement.
//!
//! State: `<model>.tier.json` next to the model (one per model file and overlay), else in `$DSX_STATE_DIR`,
//! `$XDG_STATE_HOME/dsx`, `~/.local/state/dsx`, `/var/tmp/dsx`; saved every 2 minutes, busy or not. Previous version
//! kept as `.json.prev`; devices unseen for 90 days and all but the 16 most recent GPU combinations are dropped.
//! `--tier-reset` starts from scratch. Any failure only disables the affected part; inference never depends on it.

use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::os::unix::fs::{FileExt, MetadataExt};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU32, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

pub const HALF_LIFE_TOKENS: f64 = 200_000.0;
pub const ALPHA_FULL_TOKENS: f64 = 50_000.0;
pub const ALPHA_MAX: f64 = 0.8;
/// a pause this long (no request running) counts as idle: run-time moves in bulk, page-cache warming, SSD copies
pub const IDLE_MS: u64 = 2000;
const SAVE_EVERY_SECS: u64 = 120;
/// in a pause the state is saved this soon after it changed: the server is usually stopped while idle, and it does
/// not save on a signal
const SAVE_IDLE_SECS: u64 = 10;
const REPORT_EVERY_SECS: u64 = 600;
const TICK_MS: u64 = 200;
const FOLD_EVERY_SECS: u64 = 10;
const WARM_MAX_BYTES: u64 = 40 << 30;
const MEM_RESERVE_BYTES: u64 = 16 << 30;
const CGROUP_RESERVE_BYTES: u64 = 4 << 30;
const SSD_MIN_FREE_BYTES: u64 = 20 << 30;
const SSD_MAX_BYTES: u64 = 512 << 30;
const STEP_BYTES: u64 = 512 << 20;
/// below this measured read speed the model file's medium counts as slow (HDD, network share)
pub const SLOW_READ_MB_S: f64 = 300.0;
/// the page-cache warm set is walked again this often (the kernel may have evicted pages meanwhile)
const REWARM_SECS: u64 = 600;
/// a planned move needs this much learned use, and a victim at most `MOVE_RATIO` of it
const MOVE_MIN_SCORE: f64 = 8.0;
const MOVE_RATIO: f64 = 0.5;
pub const PREFETCH_SLOTS: usize = 4;
pub const MAX_STORES: usize = 9;
/// one sample (expert kernels and the main GPU's waits, one layer, rotating) per this many decode tokens
pub const SAMPLE_EVERY: u64 = 64;
/// while a request runs, at most one move or exchange starts per this many decode tokens; pauses take the rest
pub const WORK_MOVE_EVERY: u64 = 16;
/// a measured wait counts as real above this (µs per layer); below it the device was not on the critical path
const WAIT_US: f64 = 20.0;
/// an exchanged expert is not exchanged again for this long, so two GPUs near the threshold do not pass it back and forth
const REST_SECS: u32 = 900;
/// smoothing of the measured waits (weight of a new sample)
const WAIT_EMA: f64 = 0.05;
/// samples before the waits steer moves
const WAIT_MIN_SAMPLES: u64 = 16;
const DEVICE_KEEP_SECS: u64 = 90 * 86400;
const MAX_CONFIGS: usize = 16;
const STATE_VERSION: u64 = 2;

/// Where an expert's bytes are computed from: a GPU store (`LOC_GPU + store`), else the arena, else the file side
/// (SSD buffer, page cache, file).
pub const LOC_FILE: u8 = 0;
pub const LOC_ARENA: u8 = 1;
pub const LOC_GPU: u8 = 2;

/// Which tier served an expert's bytes for the CPU (GPU-resident picks are counted separately).
#[derive(Clone, Copy)]
pub enum Hit {
    Gpu = 0,
    Arena = 1,
    Ssd = 2,
    File = 3,
}

/// A GPU store as the engine built it: store 0 is the main GPU's.
#[derive(Clone, Debug)]
pub struct StoreInfo {
    pub uuid: String,
    pub name: String,
    pub link_gbs: f64,
    /// fixed set otherwise: tiering may move experts from the CPU into it at run time
    pub tier_swaps: bool,
    /// can replace experts at run time at all (adaptive, or taking tiering's moves)
    pub swaps: bool,
}

impl StoreInfo {
    /// Replaces experts by recent use on its own (DSX_ADAPT). Its residents must keep their arena copies (it hands them
    /// back to the CPU at any token), so tiering moves nothing into it and leaves its arena copies alone.
    pub fn adaptive(&self) -> bool {
        self.swaps && !self.tier_swaps
    }
}

pub struct Tier {
    pub nl: usize,
    pub ne: usize,
    /// monotonic decode-pick counts per [layer * E + expert] (the CPU worker is the only writer)
    picks: Vec<AtomicU32>,
    hits: [AtomicU64; 4],
    /// per store: decode picks it computed
    store_picks: [AtomicU64; MAX_STORES],
    /// per store: sampled expert-kernel ms of one layer, and the number of samples
    gpu_ms: Mutex<Vec<(f64, u64)>>,
    /// the main GPU's waits per layer after its own work (smoothed, µs): for the CPU's experts, then for the aux GPUs'
    waits: Mutex<Waits>,
    /// last activity (a decode pass, or a request starting or ending), µs since `t0`
    last_busy_us: AtomicU64,
    /// decode passes: all of them, and those right after another (timed: `busy_us`)
    all_passes: AtomicU64,
    passes: AtomicU64,
    busy_us: AtomicU64,
    last_pass_us: AtomicU64,
    loc: Vec<AtomicU8>,
    /// per expert: whether the RAM arena holds a copy (it may also be resident on a GPU)
    in_arena: Vec<AtomicBool>,
    /// per expert: its (gate/up, down) quant types folded into one key; experts swap only with the same key
    key: OnceLock<Vec<u32>>,
    stores: OnceLock<Vec<StoreInfo>>,
    prefetch: OnceLock<Prefetch>,
    ssd: OnceLock<SsdBuffer>,
    stop: AtomicBool,
    /// taken by the first model that sets tiering up in this process, and by its CPU worker
    claimed: AtomicBool,
    worker: AtomicBool,
    /// requests running right now (a request is busy time as a whole, its prefill included)
    active: AtomicU32,
    /// moves carried out: from the CPU onto a GPU, between GPUs, into the arena
    done: [AtomicU64; 3],
    /// per expert: when it was last exchanged between GPUs (seconds since `t0`, 0 = never)
    moved: Vec<AtomicU32>,
    /// replacements adaptive stores started on their own
    adaptive_swaps: AtomicU64,
    t0: std::time::Instant,
}

static GLOBAL: OnceLock<&'static Tier> = OnceLock::new();
static RESET: AtomicBool = AtomicBool::new(false);

/// The process-wide tier state (created by `init`), or None when tiering is off for this run.
pub fn global() -> Option<&'static Tier> {
    GLOBAL.get().copied()
}

/// The tier state for a CPU worker of a model with this shape: the first such worker only (a second model in the
/// process, e.g. in a bench, counts nothing into the first one's usage).
pub fn for_worker(nl: usize, ne: usize) -> Option<&'static Tier> {
    global().filter(|t| t.nl == nl && t.ne == ne && !t.worker.swap(true, Ordering::AcqRel))
}

/// A request in flight; dropping it marks the end of the busy time.
pub struct Busy(&'static Tier);

impl Drop for Busy {
    fn drop(&mut self) {
        self.0.note_busy();
        self.0.active.fetch_sub(1, Ordering::AcqRel);
    }
}

/// `--tier-reset`: forget this model's learned state and SSD buffer at the next load.
pub fn request_reset() {
    RESET.store(true, Ordering::Relaxed);
}

pub fn reset_requested() -> bool {
    RESET.load(Ordering::Relaxed)
}

/// False with `DSX_TIER=0`: no learned state is read or written, nothing is measured or moved, and the placement is
/// `--rank` alone. The one switch for every part of tiering, for comparing with and without on one build.
pub fn enabled() -> bool {
    enabled_for(std::env::var_os("DSX_TIER").as_deref())
}

fn enabled_for(v: Option<&std::ffi::OsStr>) -> bool {
    v.is_none_or(|v| v != "0")
}

/// What a start places from: the state where it is written now, else where an earlier start (with other permissions)
/// left it, else nothing learned. With tiering off, nothing learned whatever is on disk.
pub fn load_for_start(on: bool, state: Option<(&Path, &str)>, candidates: &[PathBuf], nl: usize, ne: usize) -> Saved {
    state
        .filter(|_| on)
        .and_then(|(st, id)| std::iter::once(st).chain(candidates.iter().map(|p| p.as_path())).find_map(|p| load(p, id, nl, ne)))
        .unwrap_or_else(|| Saved::new(nl * ne))
}

pub fn now_secs() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

impl Tier {
    pub fn new(nl: usize, ne: usize) -> Tier {
        Tier {
            nl,
            ne,
            picks: (0..nl * ne).map(|_| AtomicU32::new(0)).collect(),
            hits: Default::default(),
            store_picks: Default::default(),
            gpu_ms: Mutex::new(vec![(0.0, 0); MAX_STORES]),
            waits: Mutex::new(Waits::default()),
            last_busy_us: AtomicU64::new(0),
            all_passes: AtomicU64::new(0),
            passes: AtomicU64::new(0),
            busy_us: AtomicU64::new(0),
            last_pass_us: AtomicU64::new(0),
            loc: (0..nl * ne).map(|_| AtomicU8::new(LOC_FILE)).collect(),
            in_arena: (0..nl * ne).map(|_| AtomicBool::new(false)).collect(),
            key: OnceLock::new(),
            stores: OnceLock::new(),
            prefetch: OnceLock::new(),
            ssd: OnceLock::new(),
            stop: AtomicBool::new(false),
            claimed: AtomicBool::new(false),
            worker: AtomicBool::new(false),
            active: AtomicU32::new(0),
            done: Default::default(),
            moved: (0..nl * ne).map(|_| AtomicU32::new(0)).collect(),
            adaptive_swaps: AtomicU64::new(0),
            t0: std::time::Instant::now(),
        }
    }

    /// Register as the process-wide instance (first call wins).
    pub fn init(nl: usize, ne: usize) -> &'static Tier {
        GLOBAL.get_or_init(|| Box::leak(Box::new(Tier::new(nl, ne))))
    }

    /// True for the first caller only: tiering state belongs to one model per process. Never with tiering off
    /// (`enabled`).
    pub fn claim(&self, nl: usize, ne: usize) -> bool {
        enabled() && self.nl == nl && self.ne == ne && !self.claimed.swap(true, Ordering::AcqRel)
    }

    /// A decode pass's picks for one layer (all experts, GPU-resident ones included).
    #[inline]
    pub fn note_picks(&self, l: usize, sel: &[i32]) {
        let base = l * self.ne;
        for &e in sel {
            if let Some(c) = self.picks.get(base + e as usize) {
                c.store(c.load(Ordering::Relaxed).wrapping_add(1), Ordering::Relaxed);
            }
        }
    }

    #[inline]
    pub fn note_hit(&self, h: Hit, n: u64) {
        self.hits[h as usize].fetch_add(n, Ordering::Relaxed);
    }

    /// A pick a GPU computed: counted for the GPU tier and for the store that holds it.
    #[inline]
    pub fn note_gpu(&self, l: usize, e: usize) {
        self.hits[Hit::Gpu as usize].fetch_add(1, Ordering::Relaxed);
        let s = self.loc(l * self.ne + e);
        if s >= LOC_GPU && ((s - LOC_GPU) as usize) < MAX_STORES {
            self.store_picks[(s - LOC_GPU) as usize].fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Called once per decode pass: marks the server busy (no warming or copying meanwhile) and times the passes.
    #[inline]
    pub fn note_decode(&self) {
        self.all_passes.fetch_add(1, Ordering::Relaxed);
        let now = self.note_busy();
        let prev = self.last_pass_us.swap(now, Ordering::Relaxed);
        if prev > 0 && now - prev < 2_000_000 {
            self.busy_us.fetch_add(now - prev, Ordering::Relaxed);
            self.passes.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// One sampled layer's expert-kernel time on `store`.
    pub fn note_gpu_ms(&self, store: usize, ms: f64) {
        if store < MAX_STORES && ms.is_finite() && ms > 0.0 {
            let mut g = self.gpu_ms.lock().unwrap();
            g[store].0 += ms;
            g[store].1 += 1;
        }
    }

    /// One sampled layer: how long the main GPU waited for the CPU's experts, then for each aux GPU in turn (ms).
    pub fn note_waits(&self, cpu_ms: f64, aux_ms: &[f64]) {
        let ok = |x: f64| x.is_finite() && x >= 0.0;
        if ok(cpu_ms) && aux_ms.iter().all(|&x| ok(x)) {
            let aux: Vec<f64> = aux_ms.iter().map(|x| x * 1000.0).collect();
            self.waits.lock().unwrap().add(cpu_ms * 1000.0, &aux);
        }
    }

    pub fn waits(&self) -> Waits {
        *self.waits.lock().unwrap()
    }

    /// Marks the server busy now (a request starts or ends, a decode pass runs); returns the time in µs since `t0`.
    pub fn note_busy(&self) -> u64 {
        let now = self.t0.elapsed().as_micros() as u64;
        self.last_busy_us.store(now, Ordering::Relaxed);
        now
    }

    /// Milliseconds since the server was last busy.
    pub fn idle_ms(&self) -> u64 {
        (self.t0.elapsed().as_micros() as u64).saturating_sub(self.last_busy_us.load(Ordering::Relaxed)) / 1000
    }

    /// No request is running and none ran for `IDLE_MS`.
    pub fn is_idle(&self) -> bool {
        self.active.load(Ordering::Acquire) == 0 && self.idle_ms() >= IDLE_MS
    }

    /// A move the engine carried out (`Move::from` set: between GPUs; `store == TO_ARENA`: into the arena).
    pub fn note_done(&self, mv: &Move) {
        let k = if mv.store == TO_ARENA { 2 } else if mv.from.is_some() { 1 } else { 0 };
        self.done[k].fetch_add(1, Ordering::Relaxed);
        if k == 1 {
            let now = (self.t0.elapsed().as_secs() as u32).max(1);
            for (l, e) in [mv.new, mv.victim] {
                if let Some(x) = self.moved.get(l * self.ne + e) {
                    x.store(now, Ordering::Relaxed);
                }
            }
        }
    }

    /// An adaptive store started a replacement of its own.
    pub fn note_adaptive_swap(&self) {
        self.adaptive_swaps.fetch_add(1, Ordering::Relaxed);
    }

    /// Experts exchanged between GPUs less than `REST_SECS` ago.
    fn resting(&self) -> Vec<usize> {
        let now = self.t0.elapsed().as_secs() as u32;
        (0..self.moved.len()).filter(|&i| matches!(self.moved[i].load(Ordering::Relaxed), t if t > 0 && now.saturating_sub(t) < REST_SECS)).collect()
    }

    /// Marks a request as running until the returned guard is dropped.
    pub fn busy(&'static self) -> Busy {
        self.active.fetch_add(1, Ordering::AcqRel);
        self.note_busy();
        Busy(self)
    }

    #[inline]
    pub fn loc(&self, i: usize) -> u8 {
        self.loc.get(i).map_or(LOC_FILE, |x| x.load(Ordering::Relaxed))
    }

    #[inline]
    pub fn set_loc(&self, l: usize, e: usize, v: u8) {
        if let Some(x) = self.loc.get(l * self.ne + e) {
            x.store(v, Ordering::Relaxed);
        }
    }

    pub fn set_in_arena(&self, l: usize, e: usize, v: bool) {
        if let Some(x) = self.in_arena.get(l * self.ne + e) {
            x.store(v, Ordering::Relaxed);
        }
    }

    /// Whether this start has a RAM arena at all.
    pub fn has_arena(&self) -> bool {
        self.in_arena.iter().any(|x| x.load(Ordering::Relaxed))
    }

    pub fn in_arena(&self, i: usize) -> bool {
        self.in_arena.get(i).is_some_and(|x| x.load(Ordering::Relaxed))
    }

    pub fn set_keys(&self, keys: Vec<u32>) {
        let _ = self.key.set(keys);
    }

    pub fn set_stores(&self, s: Vec<StoreInfo>) {
        let _ = self.stores.set(s);
    }

    pub fn set_prefetch(&self, p: Prefetch) {
        let _ = self.prefetch.set(p);
    }

    pub fn prefetch(&self) -> Option<&Prefetch> {
        self.prefetch.get()
    }

    /// The SSD buffer's copy of an expert ([gate, up, down]), if it holds one.
    #[inline]
    pub fn ssd_get(&self, l: usize, e: usize) -> Option<[&[u8]; 3]> {
        self.ssd.get().and_then(|s| s.get(l * self.ne + e))
    }

    pub fn stop(&self) {
        self.stop.store(true, Ordering::Relaxed);
    }

    /// Run `bg` on its own thread until the returned handle is dropped.
    pub fn spawn(bg: Background, m: &'static dyn ExpertBytes) -> Option<Thread> {
        let tier = bg.tier;
        let handle = std::thread::Builder::new().name("tier".into()).spawn(move || bg.run(m)).ok()?;
        Some(Thread { tier, handle: Some(handle) })
    }

    fn snapshot_picks(&self) -> Vec<u32> {
        self.picks.iter().map(|c| c.load(Ordering::Relaxed)).collect()
    }

    fn hits(&self) -> [u64; 4] {
        [0, 1, 2, 3].map(|i| self.hits[i].load(Ordering::Relaxed))
    }
}

/// The background thread; dropping it (with the model it reads) stops and joins the thread, which saves once more.
pub struct Thread {
    tier: &'static Tier,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl Drop for Thread {
    fn drop(&mut self) {
        self.tier.stop();
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

// ------------------------------------------------------------------------------------------------ run-time moves

/// A planned move: `new` replaces `victim` in GPU store `store`. With `from`, it is an exchange between two GPUs: `new`
/// leaves store `from` and `victim` takes its slot there, so both stores stay full and no expert is on two GPUs. With
/// `store == TO_ARENA`, `new` (read from the file side) takes `victim`'s place in the RAM arena.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Move {
    pub new: (usize, usize),
    pub store: usize,
    pub victim: (usize, usize),
    pub from: Option<usize>,
}

/// `Move::store` of a move into the RAM arena
pub const TO_ARENA: usize = usize::MAX;
/// arena moves prefetched at a time (they run only in pauses and must not hold up moves onto GPUs)
const ARENA_SLOTS: usize = 2;

const SLOT_FREE: u8 = 0;
const SLOT_FILLING: u8 = 1;
const SLOT_READY: u8 = 2;
const SLOT_BUSY: u8 = 3;

/// Pinned slots the background thread fills with the next experts to move; the engine copies from them to a GPU and
/// releases a slot once that copy completed. At most `PREFETCH_SLOTS` experts are in transit.
pub struct Prefetch {
    base: usize,
    cap: usize,
    state: Vec<AtomicU8>,
    /// the move and the byte lengths of its expert(s): `new`, then `victim` for an exchange
    mv: Vec<Mutex<Option<(Move, [u32; 3], [u32; 3])>>>,
}

impl Prefetch {
    /// `base`: page-locked host memory of `n * cap` bytes that outlives the process's use of it; `cap` holds two experts.
    pub fn new(base: *mut u8, cap: usize, n: usize) -> Prefetch {
        Prefetch { base: base as usize, cap, state: (0..n).map(|_| AtomicU8::new(SLOT_FREE)).collect(), mv: (0..n).map(|_| Mutex::new(None)).collect() }
    }

    fn claim(&self) -> Option<usize> {
        (0..self.state.len()).find(|&i| self.state[i].compare_exchange(SLOT_FREE, SLOT_FILLING, Ordering::AcqRel, Ordering::Relaxed).is_ok())
    }

    /// Copy one expert into a claimed slot at byte `off`; its lengths, or None when it does not fit.
    fn put(&self, i: usize, off: usize, parts: [&[u8]; 3]) -> Option<[u32; 3]> {
        let total: usize = parts.iter().map(|p| p.len()).sum();
        if off + total > self.cap {
            return None;
        }
        let dst = unsafe { std::slice::from_raw_parts_mut((self.base + i * self.cap + off) as *mut u8, total) };
        let mut o = 0;
        for p in parts {
            dst[o..o + p.len()].copy_from_slice(p);
            o += p.len();
        }
        Some(parts.map(|p| p.len() as u32))
    }

    /// Publish a claimed slot whose expert(s) `put` wrote.
    fn publish(&self, i: usize, mv: Move, lens: [u32; 3], vlens: [u32; 3]) {
        *self.mv[i].lock().unwrap() = Some((mv, lens, vlens));
        self.state[i].store(SLOT_READY, Ordering::Release);
    }

    /// Copy the expert (and, for an exchange, the victim) into a claimed slot and publish the move.
    #[cfg(test)]
    fn fill(&self, i: usize, mv: Move, parts: [&[u8]; 3], victim: Option<[&[u8]; 3]>) -> bool {
        let Some(lens) = self.put(i, 0, parts) else {
            self.unclaim(i);
            return false;
        };
        let off = lens.iter().map(|&n| n as usize).sum();
        let vlens = match victim.map(|v| self.put(i, off, v)) {
            None => [0; 3],
            Some(Some(v)) => v,
            Some(None) => {
                self.unclaim(i);
                return false;
            }
        };
        self.publish(i, mv, lens, vlens);
        true
    }

    fn unclaim(&self, i: usize) {
        self.state[i].store(SLOT_FREE, Ordering::Release);
    }

    /// The engine takes the next ready move (the slot stays in use until `release`). Arena moves wait for a pause
    /// (`idle`): they need the arena unused.
    pub fn take(&self, idle: bool) -> Option<(usize, Move)> {
        for i in 0..self.state.len() {
            if self.state[i].load(Ordering::Acquire) != SLOT_READY {
                continue;
            }
            // only this thread takes slots: a ready one stays ready until taken here
            let Some((mv, _, _)) = *self.mv[i].lock().unwrap() else { continue };
            if mv.store == TO_ARENA && !idle {
                continue;
            }
            if self.state[i].compare_exchange(SLOT_READY, SLOT_BUSY, Ordering::AcqRel, Ordering::Relaxed).is_ok() {
                return Some((i, mv));
            }
        }
        None
    }

    /// The new expert's bytes in slot `i` ([gate, up, down]).
    pub fn parts(&self, i: usize) -> Option<[&[u8]; 3]> {
        let (_, l, _) = (*self.mv[i].lock().unwrap())?;
        Some(self.at(i, 0, l))
    }

    /// For an exchange: the victim's bytes in slot `i`, which go to the store the new expert leaves.
    pub fn victim_parts(&self, i: usize) -> Option<[&[u8]; 3]> {
        let (mv, l, v) = (*self.mv[i].lock().unwrap())?;
        mv.from?;
        Some(self.at(i, l.iter().map(|&n| n as usize).sum(), v))
    }

    fn at(&self, i: usize, off: usize, l: [u32; 3]) -> [&[u8]; 3] {
        let p = (self.base + i * self.cap + off) as *const u8;
        let (a, b, c) = (l[0] as usize, l[1] as usize, l[2] as usize);
        unsafe { [std::slice::from_raw_parts(p, a), std::slice::from_raw_parts(p.add(a), b), std::slice::from_raw_parts(p.add(a + b), c)] }
    }

    pub fn release(&self, i: usize) {
        *self.mv[i].lock().unwrap() = None;
        self.state[i].store(SLOT_FREE, Ordering::Release);
    }

    /// Moves planned or in transit (both experts of each are off limits for new plans).
    fn in_transit(&self) -> Vec<Move> {
        self.mv.iter().filter_map(|m| m.lock().unwrap().map(|x| x.0)).collect()
    }
}

/// Plan up to `room` moves: hottest experts on no GPU against the coldest same-key residents of stores that take
/// run-time moves, a victim at most `MOVE_RATIO` of the candidate's score. Per expert (`layer * ne + expert`):
/// `score`, `loc`, `key`; `open` per store (see `Waits::open`); moves in `busy` (planned or in transit) block both of
/// their experts.
/// The victim is the CPU's from the next token on. With an arena (`copies`: which experts it holds) a victim without a
/// copy would be read from the file side at every pick, so it is not moved yet: it is returned in the second list
/// with the candidate's score, to get its copy first (`plan_arena`).
#[allow(clippy::too_many_arguments)]
pub fn plan_moves(ne: usize, score: &[f64], loc: &[u8], key: &[u32], copies: Option<&[bool]>, stores: &[StoreInfo], open: &[bool], busy: &[Move], room: usize) -> (Vec<Move>, Vec<(usize, f64)>) {
    let ne = ne.max(1);
    let mut off: std::collections::HashSet<usize> = busy.iter().flat_map(|m| [m.new, m.victim]).map(|(l, e)| l * ne + e).collect();
    let mut victims: HashMap<(usize, u32), Vec<(f64, usize)>> = HashMap::new();
    for i in 0..score.len() {
        let s = loc[i];
        if s < LOC_GPU {
            continue;
        }
        let st = (s - LOC_GPU) as usize;
        if !stores.get(st).is_some_and(|x| x.tier_swaps) || !open.get(st).copied().unwrap_or(true) {
            continue;
        }
        victims.entry((st, key[i])).or_default().push((score[i], i));
    }
    for v in victims.values_mut() {
        v.sort_by(|a, b| b.0.total_cmp(&a.0)); // coldest last
    }
    let mut cand: Vec<(f64, usize)> = (0..score.len()).filter(|&i| loc[i] < LOC_GPU && score[i] >= MOVE_MIN_SCORE).map(|i| (score[i], i)).collect();
    cand.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut out = vec![];
    let mut stage = vec![];
    for (sc, i) in cand {
        if out.len() + stage.len() >= room {
            break;
        }
        if off.contains(&i) {
            continue;
        }
        let k = key[i];
        let best = victims
            .iter_mut()
            .filter(|((_, kk), _)| *kk == k)
            .filter_map(|((st, _), list)| {
                while list.last().is_some_and(|x| off.contains(&x.1)) {
                    list.pop();
                }
                list.last().map(|&(vs, vi)| (vs, vi, *st))
            })
            .min_by(|a, b| a.0.total_cmp(&b.0));
        let Some((vs, vi, st)) = best else { continue };
        if vs > sc * MOVE_RATIO {
            continue;
        }
        victims.get_mut(&(st, k)).unwrap().pop();
        off.insert(i);
        off.insert(vi);
        if copies.is_some_and(|c| !c.get(vi).copied().unwrap_or(false)) {
            stage.push((vi, sc));
        } else {
            out.push(Move { new: (i / ne, i % ne), store: st, victim: (vi / ne, vi % ne), from: None });
        }
    }
    (out, stage)
}

/// Plan up to `room` exchanges that take work off GPUs finishing last (`late`, store indices) onto GPUs that do not
/// (`early`: the main GPU, which waits for the others, and aux GPUs it does not wait for): the hottest expert of a late
/// store swaps places with the coldest resident of the same key in an early store, if that one has at most `MOVE_RATIO`
/// of its score. Both stores stay full; the late one keeps a colder expert. Stores differ in size and speed: the pairs
/// are one for one and of the same size, and the waits say when to stop.
/// Only fixed-set stores take part. An adaptive store replaces residents by recent use on its own and hands them back
/// to the CPU: measured on a 3090 (adaptive) + CMP 170HX, exchanges into it took the late GPU off the critical path
/// but raised the CPU's picks per token (5.8 -> 8.2, 6.9 -> 9.2) and never the decode rate.
#[allow(clippy::too_many_arguments)]
pub fn plan_exchanges(ne: usize, score: &[f64], loc: &[u8], key: &[u32], stores: &[StoreInfo], late: &[usize], early: &[usize], busy: &[Move], room: usize) -> Vec<Move> {
    let ne = ne.max(1);
    let can = |i: usize| stores.get(i).is_some_and(|s| s.tier_swaps);
    let late: Vec<usize> = late.iter().copied().filter(|&i| can(i)).collect();
    let early: Vec<usize> = early.iter().copied().filter(|&i| can(i) && !late.contains(&i)).collect();
    if room == 0 || late.is_empty() || early.is_empty() {
        return vec![];
    }
    let mut off: std::collections::HashSet<usize> = busy.iter().flat_map(|m| [m.new, m.victim]).map(|(l, e)| l * ne + e).collect();
    // early stores' residents by key, coldest last
    let mut cold: HashMap<u32, Vec<(f64, usize, usize)>> = HashMap::new();
    let mut hot: Vec<(f64, usize, usize)> = vec![];
    for i in 0..score.len() {
        let Some(st) = loc[i].checked_sub(LOC_GPU).map(usize::from) else { continue };
        if early.contains(&st) {
            cold.entry(key[i]).or_default().push((score[i], i, st));
        } else if late.contains(&st) && score[i] >= MOVE_MIN_SCORE {
            hot.push((score[i], i, st));
        }
    }
    for v in cold.values_mut() {
        v.sort_by(|a, b| b.0.total_cmp(&a.0));
    }
    hot.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut out = vec![];
    for (sc, i, src) in hot {
        if out.len() >= room {
            break;
        }
        if off.contains(&i) {
            continue;
        }
        let Some(list) = cold.get_mut(&key[i]) else { continue };
        while list.last().is_some_and(|x| off.contains(&x.1)) {
            list.pop();
        }
        let Some(&(vs, vi, dst)) = list.last() else { continue };
        if vs > sc * MOVE_RATIO {
            continue;
        }
        list.pop();
        off.insert(i);
        off.insert(vi);
        out.push(Move { new: (i / ne, i % ne), store: dst, victim: (vi / ne, vi % ne), from: Some(src) });
    }
    out
}

/// Plan up to `room` moves into the RAM arena. Candidates: first `stage` (GPU residents that need an arena copy before
/// they enter an adaptive store or go back to the CPU, each with the score of the hot expert the copy makes room
/// for), then experts the CPU computes without an arena copy (it reads them from the file side), hottest first. Victims, same key (so the same byte lengths): first arena copies of experts
/// resident on a fixed-set GPU store (the CPU never reads them; an adaptive store's residents keep theirs, since that
/// store may hand them back to the CPU at any token), then the coldest experts the CPU reads from the arena, at most
/// `MOVE_RATIO` of the candidate's score. `in_arena` per expert; `adaptive` per store.
#[allow(clippy::too_many_arguments)]
pub fn plan_arena(ne: usize, score: &[f64], loc: &[u8], key: &[u32], in_arena: &[bool], adaptive: &[bool], stage: &[(usize, f64)], busy: &[Move], room: usize) -> Vec<Move> {
    let ne = ne.max(1);
    let mut off: std::collections::HashSet<usize> = busy.iter().flat_map(|m| [m.new, m.victim]).map(|(l, e)| l * ne + e).collect();
    // per key: (wasted?, score, index), the best victim last
    let mut victims: HashMap<u32, Vec<(bool, f64, usize)>> = HashMap::new();
    let mut hot: Vec<(f64, usize)> = vec![];
    for i in 0..score.len() {
        let gpu = loc[i].checked_sub(LOC_GPU).map(usize::from);
        if in_arena[i] {
            match gpu {
                Some(st) if !adaptive.get(st).copied().unwrap_or(true) => victims.entry(key[i]).or_default().push((true, score[i], i)),
                None => victims.entry(key[i]).or_default().push((false, score[i], i)),
                Some(_) => {}
            }
        } else if gpu.is_none() {
            hot.push((score[i], i));
        }
    }
    for v in victims.values_mut() {
        // wasted copies last (taken first), the hottest of them first: a store's cold residents are the next to go back
        // to the CPU, which needs their copies. Then the copies the CPU reads, coldest first.
        v.sort_by(|a, b| a.0.cmp(&b.0).then(if a.0 { a.1.total_cmp(&b.1) } else { b.1.total_cmp(&a.1) }));
    }
    hot.sort_by(|a, b| b.0.total_cmp(&a.0));
    let staged: Vec<(f64, usize)> = stage.iter().filter(|&&(i, _)| i < score.len() && !in_arena[i] && !off.contains(&i)).map(|&(i, worth)| (worth, i)).collect();
    let mut out = vec![];
    for (sc, i) in staged.into_iter().chain(hot) {
        if out.len() >= room {
            break;
        }
        if off.contains(&i) {
            continue;
        }
        let Some(list) = victims.get_mut(&key[i]) else { continue };
        while list.last().is_some_and(|x| off.contains(&x.2)) {
            list.pop();
        }
        let Some(&(wasted, vs, vi)) = list.last() else { continue };
        // a wasted copy makes room for any expert the CPU reads from disk; a used one only for a clearly hotter one
        if !wasted && (sc < MOVE_MIN_SCORE || vs > sc * MOVE_RATIO) {
            continue;
        }
        list.pop();
        off.insert(i);
        off.insert(vi);
        out.push(Move { new: (i / ne, i % ne), store: TO_ARENA, victim: (vi / ne, vi % ne), from: None });
    }
    out
}

/// Where a layer's time goes once the main GPU has done its own part: waiting for the CPU's experts, then for each
/// aux GPU in turn (each wait is how much later that GPU finished than everything waited for before it). A device that
/// makes the main GPU wait is on the critical path; one that does not has room.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Waits {
    pub cpu_us: f64,
    /// per aux GPU (store 1, 2, …)
    pub aux_us: [f64; MAX_STORES],
    pub samples: u64,
}

impl Waits {
    fn add(&mut self, cpu_us: f64, aux_us: &[f64]) {
        let w = if self.samples == 0 { 1.0 } else { WAIT_EMA };
        self.cpu_us += (cpu_us - self.cpu_us) * w;
        for (a, &x) in self.aux_us.iter_mut().zip(aux_us) {
            *a += (x - *a) * w;
        }
        self.samples += 1;
    }

    fn ready(&self) -> bool {
        self.samples >= WAIT_MIN_SAMPLES
    }

    /// The aux GPUs' waits together (the time the main GPU waits for aux GPUs after the CPU).
    pub fn aux_total_us(&self) -> f64 {
        self.aux_us.iter().sum()
    }

    /// Which stores may take more experts from the CPU (store 0 = main GPU, then the aux GPUs). Moving a hot expert from
    /// the CPU onto a GPU only shortens a layer while the CPU is what the main GPU waits for, and only onto a GPU that is
    /// not itself late: the main GPU while the CPU makes it wait, an aux GPU while it does not finish after the CPU. Before
    /// enough samples every store is open.
    pub fn open(&self, stores: usize) -> Vec<bool> {
        if !self.ready() {
            return vec![true; stores];
        }
        let cpu_slow = self.cpu_us > WAIT_US;
        (0..stores).map(|i| cpu_slow && (i == 0 || self.aux_us.get(i - 1).is_some_and(|&a| a <= WAIT_US))).collect()
    }

    /// Aux GPUs (store indices) that finish after the CPU and everything before them: work should leave them.
    pub fn late(&self, stores: usize, by_us: f64) -> Vec<usize> {
        if !self.ready() {
            return vec![];
        }
        (1..stores).filter(|&i| self.aux_us.get(i - 1).is_some_and(|&a| a > by_us)).collect()
    }

    /// GPUs that can take work from late ones: the main GPU (it is the one waiting) and aux GPUs it does not wait for.
    pub fn early(&self, stores: usize) -> Vec<usize> {
        let late = self.late(stores, WAIT_US);
        (0..stores).filter(|i| !late.contains(i)).collect()
    }
}

// ------------------------------------------------------------------------------------------------ learned state

/// Long-term usage: decayed picks per expert and the number of decode tokens it was learned from.
#[derive(Clone, Debug, PartialEq)]
pub struct Learned {
    pub score: Vec<f64>,
    pub tokens: f64,
}

impl Learned {
    pub fn new(n: usize) -> Learned {
        Learned { score: vec![0.0; n], tokens: 0.0 }
    }

    /// Add new picks (delta of the monotonic counters) after decaying by the tokens they represent.
    pub fn fold(&mut self, delta: &[u32], top_k: usize, nl: usize) {
        let total: u64 = delta.iter().map(|&d| d as u64).sum();
        if total == 0 {
            return;
        }
        let tokens = total as f64 / (top_k.max(1) * nl.max(1)) as f64;
        let f = 0.5f64.powf(tokens / HALF_LIFE_TOKENS);
        for (s, &d) in self.score.iter_mut().zip(delta) {
            *s = *s * f + d as f64;
        }
        self.tokens += tokens;
    }

    pub fn alpha(&self) -> f64 {
        (self.tokens / ALPHA_FULL_TOKENS * ALPHA_MAX).min(ALPHA_MAX)
    }
}

/// Everything the state file keeps: usage (hardware-independent), device measurements, per-combination figures.
#[derive(Clone, Debug, PartialEq)]
pub struct Saved {
    pub learned: Learned,
    pub devices: serde_json::Map<String, serde_json::Value>,
    pub configs: serde_json::Map<String, serde_json::Value>,
}

impl Saved {
    pub fn new(n: usize) -> Saved {
        Saved { learned: Learned::new(n), devices: Default::default(), configs: Default::default() }
    }

    /// Drop devices unseen for `DEVICE_KEEP_SECS` and all but the `MAX_CONFIGS` most recent combinations.
    pub fn rotate(&mut self, now: u64) {
        self.devices.retain(|_, v| v["seen"].as_u64().is_some_and(|s| now.saturating_sub(s) <= DEVICE_KEEP_SECS));
        if self.configs.len() > MAX_CONFIGS {
            let mut seen: Vec<(u64, String)> = self.configs.iter().map(|(k, v)| (v["seen"].as_u64().unwrap_or(0), k.clone())).collect();
            seen.sort();
            for (_, k) in seen.iter().take(self.configs.len() - MAX_CONFIGS) {
                self.configs.remove(k);
            }
        }
    }
}

/// Identity of one file: canonical path, size, modification time.
pub fn model_id(path: &Path) -> String {
    let canon = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    match std::fs::metadata(path) {
        Ok(m) => format!("{}|{}|{}", canon.display(), m.len(), m.mtime()),
        Err(_) => canon.display().to_string(),
    }
}

/// Identity of a model made of `paths` (shards, then any overlay): any change of any file starts a new state.
pub fn models_id(paths: &[PathBuf]) -> String {
    paths.iter().map(|p| model_id(p)).collect::<Vec<_>>().join(";")
}

fn clean(s: &str) -> String {
    s.chars().map(|c| if c.is_ascii_alphanumeric() || c == '.' || c == '-' || c == '_' || c == '+' { c } else { '_' }).collect()
}

/// Shards of one split GGUF share the name before `-NNNNN-of-NNNNN`.
fn shard_stem(name: &str) -> Option<&str> {
    let i = name.find("-of-")?;
    let head = &name[..i];
    let j = head.rfind('-')?;
    head[j + 1..].chars().all(|c| c.is_ascii_digit()).then_some(&head[..j])
}

/// The state's base name: the first file's name, plus `+<stem>` for every file that is not a shard of it (overlays).
pub fn state_name(paths: &[PathBuf]) -> String {
    let fname = |p: &PathBuf| p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "model".into());
    let Some(first) = paths.first() else { return "model".into() };
    let f0 = fname(first);
    let stem0 = shard_stem(&f0).map(str::to_owned);
    let mut name = f0.clone();
    for p in &paths[1..] {
        let f = fname(p);
        if stem0.is_some() && shard_stem(&f).map(str::to_owned) == stem0 {
            continue;
        }
        let stem = Path::new(&f).file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or(f);
        name += &format!("+{stem}");
    }
    clean(&name)
}

/// Where the state goes when the model's directory is not writable, in order of preference.
fn state_dirs() -> Vec<PathBuf> {
    let env = |k: &str| std::env::var_os(k).filter(|v| !v.is_empty()).map(PathBuf::from);
    [
        env("DSX_STATE_DIR"),
        env("XDG_STATE_HOME").map(|p| p.join("dsx")),
        env("HOME").map(|p| p.join(".local/state/dsx")),
        Some(PathBuf::from("/var/tmp/dsx")),
    ]
    .into_iter()
    .flatten()
    .collect()
}

/// The first of `state_dirs` that can be created and written.
fn state_dir() -> PathBuf {
    let cands = state_dirs();
    for d in &cands {
        if std::fs::create_dir_all(d).is_ok() && writable(d) {
            return d.clone();
        }
    }
    cands.last().cloned().unwrap()
}

fn writable(d: &Path) -> bool {
    let probe = d.join(format!(".dsx-probe-{}", std::process::id()));
    let ok = File::create(&probe).is_ok();
    let _ = std::fs::remove_file(&probe);
    ok
}

/// `<model dir>/<name>.tier.json` when that directory is writable, else the same name in the state directory.
pub fn state_path(paths: &[PathBuf]) -> PathBuf {
    let name = format!("{}.tier.json", state_name(paths));
    match paths.first().and_then(|p| p.parent()).filter(|d| writable(d)) {
        Some(d) => d.join(name),
        None => state_dir().join(name),
    }
}

/// Every place the state of this model may be in (the model's directory was or was not writable at earlier starts).
pub fn state_candidates(paths: &[PathBuf]) -> Vec<PathBuf> {
    let name = format!("{}.tier.json", state_name(paths));
    paths.first().and_then(|p| p.parent()).map(|d| d.to_path_buf()).into_iter().chain(state_dirs()).map(|d| d.join(&name)).collect()
}

fn prev_path(path: &Path) -> PathBuf {
    path.with_extension("json.prev")
}

/// Read the state if it belongs to this model and shape; None otherwise (a new model starts fresh). An unreadable or
/// foreign file falls back to the previous version.
pub fn load(path: &Path, id: &str, nl: usize, ne: usize) -> Option<Saved> {
    load_one(path, id, nl, ne).or_else(|| load_one(&prev_path(path), id, nl, ne))
}

fn load_one(path: &Path, id: &str, nl: usize, ne: usize) -> Option<Saved> {
    let v: serde_json::Value = serde_json::from_slice(&std::fs::read(path).ok()?).ok()?;
    if v["version"].as_u64()? != STATE_VERSION || v["model"]["id"].as_str()? != id || v["layers"].as_u64()? as usize != nl || v["experts"].as_u64()? as usize != ne {
        return None;
    }
    let score: Vec<f64> = v["usage"]["score"].as_array()?.iter().map(|x| x.as_f64().unwrap_or(0.0).max(0.0)).collect();
    if score.len() != nl * ne {
        return None;
    }
    let obj = |k: &str| v.get(k).and_then(|x| x.as_object()).cloned().unwrap_or_default();
    let tokens = v["usage"]["tokens"].as_f64().unwrap_or(0.0).max(0.0);
    Some(Saved { learned: Learned { score, tokens }, devices: obj("devices"), configs: obj("configs") })
}

/// Write the state atomically: a temp file of this process, then a rename over the state. The state before that stays
/// reachable as `.prev` (a hard link, else a copy), so there is a readable state at every instant.
pub fn save(path: &Path, id: &str, files: &[String], nl: usize, ne: usize, s: &Saved, stats: serde_json::Value) -> std::io::Result<()> {
    let v = serde_json::json!({
        "version": STATE_VERSION,
        "model": {"id": id, "files": files},
        "layers": nl, "experts": ne,
        "usage": {"tokens": s.learned.tokens, "score": s.learned.score.iter().map(|x| (x * 1000.0).round() / 1000.0).collect::<Vec<_>>()},
        "devices": s.devices, "configs": s.configs, "stats": stats, "saved": now_secs(),
    });
    let tmp = path.with_extension(format!("json.tmp.{}", std::process::id()));
    let written = (|| {
        let mut f = File::create(&tmp)?;
        f.write_all(serde_json::to_string(&v)?.as_bytes())?;
        f.sync_data()
    })();
    if let Err(e) = written {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    if path.exists() {
        let prev = prev_path(path);
        let _ = std::fs::remove_file(&prev);
        if std::fs::hard_link(path, &prev).is_err() {
            let _ = std::fs::copy(path, &prev);
        }
    }
    std::fs::rename(&tmp, path).inspect_err(|_| {
        let _ = std::fs::remove_file(&tmp);
    })
}

/// Temp files a killed process left next to the state (`<state>.tmp.<pid>`).
pub fn remove_stale_temps(path: &Path) {
    let (Some(dir), Some(name)) = (path.parent(), path.file_name().map(|n| n.to_string_lossy().into_owned())) else { return };
    let prefix = format!("{name}.tmp.");
    for e in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        if e.file_name().to_string_lossy().starts_with(&prefix) {
            let _ = std::fs::remove_file(e.path());
        }
    }
}

/// `--tier-reset`: remove the state, its previous version and the SSD buffer of this model.
pub fn reset_files(state: &Path, name: &str) -> usize {
    let mut n = 0;
    let mut rm = |p: PathBuf| {
        if std::fs::remove_file(&p).is_ok() {
            n += 1;
        }
    };
    rm(state.to_path_buf());
    rm(prev_path(state));
    for d in ssd_dirs() {
        rm(d.join(format!("{name}.ssd-buffer")));
        rm(d.join(format!("{name}.ssd-buffer.json")));
    }
    n
}

/// The startup ranking: `(1 - α)` × shipped calibration counts + `α` × learned score, both normalized; experts the
/// shipped ranking does not list keep their order behind. With no learned tokens the shipped order is unchanged.
pub fn blend_rank(base: &[(usize, usize, u64)], learned: &Learned, nl: usize, ne: usize) -> Vec<(usize, usize)> {
    let a = learned.alpha();
    let mut pos: HashMap<(usize, usize), usize> = HashMap::new();
    let mut cnt = vec![0f64; nl * ne];
    for (i, &(l, e, n)) in base.iter().enumerate() {
        if l < nl && e < ne {
            pos.entry((l, e)).or_insert(i);
            cnt[l * ne + e] = n as f64;
        }
    }
    let bsum: f64 = cnt.iter().sum::<f64>().max(1e-9);
    let lsum: f64 = learned.score.iter().sum::<f64>().max(1e-9);
    let score = |i: usize| (1.0 - a) * cnt[i] / bsum + a * learned.score.get(i).copied().unwrap_or(0.0) / lsum;
    let mut keys: Vec<(usize, usize)> = (0..ne).flat_map(|e| (0..nl).map(move |l| (l, e))).collect();
    keys.sort_by(|x, y| {
        let (i, j) = (x.0 * ne + x.1, y.0 * ne + y.1);
        score(j).total_cmp(&score(i)).then(pos.get(x).unwrap_or(&usize::MAX).cmp(pos.get(y).unwrap_or(&usize::MAX)))
    });
    keys
}

// ------------------------------------------------------------------------------------------------ storage facts

fn dev_parts(dev: u64) -> (u64, u64) {
    let major = ((dev >> 8) & 0xfff) | ((dev >> 32) & !0xfff);
    let minor = (dev & 0xff) | ((dev >> 12) & !0xff);
    (major, minor)
}

/// The sysfs directory of the block device holding `path`: by its device number, else (btrfs, overlay and others
/// report an anonymous device number) by the source of its mount in /proc/self/mountinfo.
fn block_sysfs(path: &Path) -> Option<PathBuf> {
    let (ma, mi) = dev_parts(std::fs::metadata(path).ok()?.dev());
    if let Ok(p) = std::fs::canonicalize(format!("/sys/dev/block/{ma}:{mi}")) {
        return Some(p);
    }
    let canon = std::fs::canonicalize(path).ok()?;
    let info = std::fs::read_to_string("/proc/self/mountinfo").ok()?;
    let src = mount_source(&info, &canon)?;
    let dev = std::fs::canonicalize(src).ok()?;
    std::fs::canonicalize(Path::new("/sys/class/block").join(dev.file_name()?)).ok()
}

/// The source (`/dev/…`) of the mount that holds `path`, from mountinfo text: the longest mount point that is a prefix
/// of the path; the last such line wins (later mounts shadow earlier ones).
fn mount_source(mountinfo: &str, path: &Path) -> Option<String> {
    let mut best: Option<(usize, String)> = None;
    for line in mountinfo.lines() {
        let Some((left, right)) = line.split_once(" - ") else { continue };
        let Some(point) = left.split(' ').nth(4) else { continue };
        let Some(src) = right.split(' ').nth(1) else { continue };
        // mountinfo escapes space, tab, newline and backslash as octal
        let point = point.replace("\\040", " ").replace("\\011", "\t").replace("\\012", "\n").replace("\\134", "\\");
        if src.starts_with("/dev/") && path.starts_with(&point) && best.as_ref().is_none_or(|b| point.len() >= b.0) {
            best = Some((point.len(), src.to_owned()));
        }
    }
    best.map(|b| b.1)
}

/// Whether the file lives on a rotational disk (HDD), from /sys; None if unknown.
pub fn rotational(path: &Path) -> Option<bool> {
    let sys = block_sysfs(path)?;
    for d in sys.ancestors() {
        if let Ok(s) = std::fs::read_to_string(d.join("queue/rotational")) {
            return Some(s.trim() == "1");
        }
    }
    None
}

/// `disk:<block device>` of the device holding `path` (the key of its measurements), else `disk:<major>:<minor>`.
pub fn disk_key(path: &Path) -> Option<String> {
    if let Some(name) = block_sysfs(path).and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned())) {
        return Some(format!("disk:{name}"));
    }
    let (ma, mi) = dev_parts(std::fs::metadata(path).ok()?.dev());
    Some(format!("disk:{ma}:{mi}"))
}

/// Read speed of the medium under `path` in MB/s: 32 reads of 2 MiB spread over the file, bypassing the page cache
/// (O_DIRECT). None when the file is small or O_DIRECT is unavailable.
pub fn measure_read_mb_s(path: &Path) -> Option<f64> {
    use std::os::unix::fs::OpenOptionsExt;
    const O_DIRECT: i32 = if cfg!(any(target_arch = "aarch64", target_arch = "arm")) { 0o200000 } else { 0o40000 };
    const CHUNK: usize = 2 << 20;
    let len = std::fs::metadata(path).ok()?.len() as usize;
    if len < 64 * CHUNK {
        return None;
    }
    let f = std::fs::OpenOptions::new().read(true).custom_flags(O_DIRECT).open(path).ok()?;
    let layout = std::alloc::Layout::from_size_align(CHUNK, 4096).ok()?;
    let buf = unsafe { std::alloc::alloc(layout) };
    if buf.is_null() {
        return None;
    }
    let b = unsafe { std::slice::from_raw_parts_mut(buf, CHUNK) };
    let t = std::time::Instant::now();
    let mut done = 0usize;
    for k in 0..32usize {
        // spread over the file, odd multiples so repeated runs hit the same places (no cache: O_DIRECT)
        let off = ((len - CHUNK) / 33 * (k + 1)) / 4096 * 4096;
        match f.read_at(b, off as u64) {
            Ok(n) if n > 0 => done += n,
            _ => break,
        }
    }
    let s = t.elapsed().as_secs_f64();
    unsafe { std::alloc::dealloc(buf, layout) };
    (done >= 16 * CHUNK && s > 0.0).then(|| done as f64 / s / 1e6)
}

#[repr(C)]
struct StatVfs {
    bsize: u64,
    frsize: u64,
    blocks: u64,
    bfree: u64,
    bavail: u64,
    files: u64,
    ffree: u64,
    favail: u64,
    fsid: u64,
    flag: u64,
    namemax: u64,
    spare: [i32; 6],
}
unsafe extern "C" {
    fn flock(fd: i32, operation: i32) -> i32;
    fn statvfs(path: *const std::ffi::c_char, buf: *mut StatVfs) -> i32;
    fn madvise(addr: *mut std::ffi::c_void, len: usize, advice: i32) -> i32;
}

pub fn free_bytes(dir: &Path) -> Option<u64> {
    let c = std::ffi::CString::new(dir.as_os_str().as_encoded_bytes()).ok()?;
    let mut s: StatVfs = unsafe { std::mem::zeroed() };
    (unsafe { statvfs(c.as_ptr(), &mut s) } == 0).then(|| s.bavail * s.frsize.max(1))
}

/// Bytes the page cache may still take: min(MemAvailable − reserve, cgroup headroom − reserve).
pub fn ram_headroom() -> u64 {
    let avail = std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|t| t.lines().find(|l| l.starts_with("MemAvailable:")).and_then(|l| l.split_whitespace().nth(1)?.parse::<u64>().ok()))
        .map_or(0, |kb| kb * 1024);
    let mut room = avail.saturating_sub(MEM_RESERVE_BYTES);
    if let Some(cg) = std::fs::read_to_string("/proc/self/cgroup").ok().and_then(|t| t.lines().find_map(|l| l.strip_prefix("0::").map(str::to_owned))) {
        let d = Path::new("/sys/fs/cgroup").join(cg.trim_start_matches('/'));
        let rd = |n: &str| std::fs::read_to_string(d.join(n)).ok().and_then(|s| s.trim().parse::<u64>().ok());
        if let (Some(max), Some(cur)) = (rd("memory.max"), rd("memory.current")) {
            room = room.min(max.saturating_sub(cur).saturating_sub(CGROUP_RESERVE_BYTES));
        }
    }
    room
}

const MADV_WILLNEED: i32 = 3;

/// Fault the bytes into the page cache (one read per 4 KiB page); returns the bytes covered.
pub fn touch(b: &[u8]) -> u64 {
    if !b.is_empty() {
        let p = b.as_ptr() as usize;
        let start = p & !4095;
        unsafe { madvise(start as *mut _, p + b.len() - start, MADV_WILLNEED) };
    }
    let mut acc = 0u8;
    for i in (0..b.len()).step_by(4096) {
        acc ^= unsafe { std::ptr::read_volatile(b.as_ptr().add(i)) };
    }
    std::hint::black_box(acc);
    b.len() as u64
}

// ------------------------------------------------------------------------------------------------ SSD buffer

/// Copies of hot experts on an SSD when the model file sits on a slow medium. One preallocated (sparse) file,
/// mapped once and locked by the one process that uses it; entries are appended, written and synced before they
/// become visible, and an index file names them so the buffer survives restarts. Entries are never rewritten
/// (readers hold no locks); once full, it keeps what it holds.
pub struct SsdBuffer {
    file: File,
    map: memmap2::Mmap,
    path: PathBuf,
    index: PathBuf,
    cap: u64,
    used: AtomicU64,
    /// per expert: offset + 1 of its [gate | up | down] copy, 0 = not held
    slot: Vec<AtomicU64>,
    lens: Vec<[u32; 3]>,
    entries: Mutex<Vec<(usize, u64)>>,
    id: String,
    /// stop appending when the buffer's file system has less free space than this
    min_free: u64,
}

impl SsdBuffer {
    /// Open or create the buffer in `dir` for the model `id`; `lens` gives each expert's three blob sizes, `want` the
    /// size asked for. A buffer of this model is kept with its entries and only ever grows (its size depends on free
    /// space and on the placement, which differ between starts); another model's is discarded.
    pub fn open(dir: &Path, id: &str, name: &str, lens: Vec<[u32; 3]>, want: u64) -> std::io::Result<SsdBuffer> {
        std::fs::create_dir_all(dir)?;
        let path = dir.join(format!("{name}.ssd-buffer"));
        let index = dir.join(format!("{name}.ssd-buffer.json"));
        let mut entries: Vec<(usize, u64)> = vec![];
        let mut used = 0u64;
        let prior = std::fs::read(&index).ok().and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok());
        let have = std::fs::metadata(&path).map_or(0, |m| m.len());
        let reuse = have > 0 && prior.as_ref().is_some_and(|v| v["model"].as_str() == Some(id));
        let cap = if reuse { have.max(want) } else { want };
        if reuse {
            for x in prior.as_ref().unwrap()["entries"].as_array().into_iter().flatten() {
                let (Some(i), Some(off)) = (x[0].as_u64(), x[1].as_u64()) else { continue };
                let i = i as usize;
                let Some(len) = lens.get(i).map(|l| l.iter().map(|&n| n as u64).sum::<u64>()) else { continue };
                // entries are only listed after their bytes were synced; one beyond the file is from a damaged index
                if off + len <= have && !entries.iter().any(|e: &(usize, u64)| e.0 == i) {
                    entries.push((i, off));
                    used = used.max(off + len);
                }
            }
        }
        let file = std::fs::OpenOptions::new().read(true).write(true).create(true).truncate(false).open(&path)?;
        // one writer: a second server on this model reads the model file instead (two would overwrite each other's entries)
        const LOCK_EX_NB: i32 = 2 | 4;
        if unsafe { flock(std::os::fd::AsRawFd::as_raw_fd(&file), LOCK_EX_NB) } != 0 {
            return Err(std::io::Error::other("in use by another dsx process"));
        }
        if !reuse {
            file.set_len(0)?;
        }
        if have != cap || !reuse {
            file.set_len(cap)?;
        }
        let map = unsafe { memmap2::MmapOptions::new().len(cap as usize).map(&file)? };
        let slot: Vec<AtomicU64> = (0..lens.len()).map(|_| AtomicU64::new(0)).collect();
        for &(i, off) in &entries {
            slot[i].store(off + 1, Ordering::Release);
        }
        Ok(SsdBuffer { file, map, path, index, cap, used: AtomicU64::new(used.next_multiple_of(4096)), slot, lens, entries: Mutex::new(entries), id: id.to_owned(), min_free: SSD_MIN_FREE_BYTES })
    }

    pub fn get(&self, i: usize) -> Option<[&[u8]; 3]> {
        let off = self.slot.get(i)?.load(Ordering::Acquire);
        if off == 0 {
            return None;
        }
        let o = (off - 1) as usize;
        let [a, b, c] = self.lens[i].map(|n| n as usize);
        Some([&self.map[o..o + a], &self.map[o + a..o + a + b], &self.map[o + a + b..o + a + b + c]])
    }

    pub fn holds(&self, i: usize) -> bool {
        self.slot.get(i).is_some_and(|s| s.load(Ordering::Relaxed) != 0)
    }

    pub fn used(&self) -> u64 {
        self.used.load(Ordering::Relaxed)
    }

    /// Append one expert's three blobs; false when the buffer is full.
    pub fn put(&self, i: usize, parts: [&[u8]; 3]) -> std::io::Result<bool> {
        if self.holds(i) {
            return Ok(true);
        }
        let len: u64 = parts.iter().map(|p| p.len() as u64).sum();
        let off = self.used();
        // full, or the disk filled up since the (sparse) file was sized: stop here
        if off + len > self.cap || self.path.parent().and_then(free_bytes).is_some_and(|f| f < self.min_free) {
            return Ok(false);
        }
        let mut o = off;
        for p in parts {
            self.file.write_all_at(p, o)?;
            o += p.len() as u64;
        }
        self.file.sync_data()?;
        self.used.store((off + len).next_multiple_of(4096), Ordering::Relaxed);
        self.slot[i].store(off + 1, Ordering::Release);
        self.entries.lock().unwrap().push((i, off));
        Ok(true)
    }

    /// Persist the index (atomically) so the next start finds the copies.
    pub fn save_index(&self) -> std::io::Result<()> {
        let e = self.entries.lock().unwrap().clone();
        let v = serde_json::json!({"model": self.id, "cap": self.cap, "entries": e});
        let tmp = self.index.with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_vec(&v)?)?;
        std::fs::rename(tmp, &self.index)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

fn ssd_dirs() -> Vec<PathBuf> {
    let env = |k: &str| std::env::var_os(k).filter(|v| !v.is_empty()).map(PathBuf::from);
    [
        env("DSX_CACHE_DIR"),
        Some(state_dir()),
        env("XDG_CACHE_HOME").map(|p| p.join("dsx")),
        env("HOME").map(|p| p.join(".cache/dsx")),
        Some(PathBuf::from("/var/tmp/dsx")),
    ]
    .into_iter()
    .flatten()
    .collect()
}

/// Where to put the SSD buffer: the first writable directory on another, non-rotational device with room.
pub fn ssd_dir(model: &Path) -> Option<PathBuf> {
    let model_dev = std::fs::metadata(model).ok().map(|m| m.dev());
    ssd_dirs().into_iter().find(|d| {
        std::fs::create_dir_all(d).is_ok()
            && writable(d)
            && rotational(d) == Some(false)
            && std::fs::metadata(d).ok().map(|m| m.dev()) != model_dev
            && free_bytes(d).is_some_and(|f| f > SSD_MIN_FREE_BYTES)
    })
}

// ------------------------------------------------------------------------------------------------ background work

/// What the background thread needs from the model.
pub trait ExpertBytes: Sync {
    /// An expert's three byte slices in the mapped model file.
    fn parts(&self, l: usize, e: usize) -> [&[u8]; 3];
    /// Hand the expert's bytes to `f` from RAM when possible (the arena, read under its lock), else from the file.
    fn read(&self, l: usize, e: usize, f: &mut dyn FnMut([&[u8]; 3])) {
        f(self.parts(l, e))
    }
}

pub struct Background {
    pub tier: &'static Tier,
    pub state: PathBuf,
    pub id: String,
    pub files: Vec<String>,
    pub top_k: usize,
    pub saved: Saved,
    pub model_disk: Option<String>,
    pub model_read_mb_s: Option<f64>,
    pub slow_media: bool,
}

#[derive(Default, Debug)]
pub struct Stats {
    pub warm_bytes: u64,
    pub warm_secs: f64,
    pub ssd_bytes: u64,
    pub ssd_secs: f64,
    pub saves: u64,
    pub moves_planned: u64,
}

impl Background {
    /// Experts the CPU reads from the file side (no GPU, no arena), hottest first.
    fn order(&self) -> Vec<(usize, usize)> {
        let ne = self.tier.ne;
        let mut v: Vec<(usize, usize)> = (0..self.tier.nl * ne).filter(|&i| self.tier.loc(i) == LOC_FILE).map(|i| (i / ne, i % ne)).collect();
        v.sort_by(|a, b| self.saved.learned.score[b.0 * ne + b.1].total_cmp(&self.saved.learned.score[a.0 * ne + a.1]));
        v
    }

    /// GPU-combination key: the stores' UUIDs, main first.
    fn config_key(&self) -> Option<String> {
        let s = self.tier.stores.get()?;
        (!s.is_empty()).then(|| s.iter().map(|x| x.uuid.clone()).collect::<Vec<_>>().join("+"))
    }

    /// Fold this run's measurements into the device and combination records.
    fn measure(&mut self) {
        let now = now_secs();
        let passes = self.tier.passes.load(Ordering::Relaxed);
        let all = self.tier.all_passes.load(Ordering::Relaxed);
        let round = |x: f64, digits: i32| (x * 10f64.powi(digits)).round() / 10f64.powi(digits);
        // a record is written anew each time; a figure this run could not measure keeps its last value
        fn record(old: Option<&serde_json::Value>, fields: Vec<(&str, Option<serde_json::Value>)>) -> serde_json::Value {
            let mut m = serde_json::Map::new();
            for (k, v) in fields {
                if let Some(v) = v.or_else(|| old.and_then(|o| o.get(k)).cloned()) {
                    m.insert(k.into(), v);
                }
            }
            serde_json::Value::Object(m)
        }
        if let Some(stores) = self.tier.stores.get() {
            let ms = self.tier.gpu_ms.lock().unwrap().clone();
            for (i, s) in stores.iter().enumerate().take(MAX_STORES) {
                let picks = self.tier.store_picks[i].load(Ordering::Relaxed) as f64;
                // the GPU's expert kernels in one (sampled) layer of a decode token
                let layer_ms = ms.get(i).filter(|x| x.1 >= 8).map(|&(sum, n)| round(sum / n as f64, 3).into());
                let r = record(self.saved.devices.get(&s.uuid), vec![
                    ("name", Some(s.name.clone().into())),
                    ("link_gbs", Some(round(s.link_gbs, 1).into())),
                    ("expert_layer_ms", layer_ms),
                    ("picks_per_token", (all > 0).then(|| round(picks / all as f64, 1).into())),
                    ("seen", Some(now.into())),
                ]);
                self.saved.devices.insert(s.uuid.clone(), r);
            }
        }
        if let (Some(k), Some(mbs)) = (&self.model_disk, self.model_read_mb_s) {
            self.saved.devices.insert(k.clone(), serde_json::json!({"read_mb_s": mbs.round(), "seen": now}));
        }
        if let Some(key) = self.config_key() {
            let busy = self.tier.busy_us.load(Ordering::Relaxed) as f64 / 1e6;
            let h = self.tier.hits();
            let tot = h.iter().sum::<u64>();
            let w = self.tier.waits();
            let aux = self.tier.stores.get().map_or(0, |s| s.len().saturating_sub(1)).min(MAX_STORES);
            let r = record(self.saved.configs.get(&key), vec![
                ("decode_tok_s", (passes >= 100 && busy > 0.0).then(|| round(passes as f64 / busy, 2).into())),
                ("cpu_picks_per_token", (all > 0).then(|| round((h[1] + h[2] + h[3]) as f64 / all as f64, 2).into())),
                ("share_from_file", (tot > 0).then(|| round(h[3] as f64 / tot as f64, 3).into())),
                ("wait_for_cpu_us", w.ready().then(|| w.cpu_us.round().into())),
                ("wait_for_aux_us", w.ready().then(|| w.aux_us[..aux].iter().map(|x| x.round()).collect::<Vec<_>>().into())),
                ("seen", Some(now.into())),
            ]);
            self.saved.configs.insert(key, r);
        }
        self.saved.rotate(now);
    }

    /// Where a decode token's time goes, from what is measured anyway: the whole token (decode rate), the main GPU's
    /// expert kernels and its waits for the CPU and the aux GPUs (sampled layers, times the layer count), and the rest:
    /// attention, router, head and everything else the main GPU does alone. Tiering can only shorten the waits.
    fn token_ms(&self) -> serde_json::Value {
        let (passes, busy) = (self.tier.passes.load(Ordering::Relaxed), self.tier.busy_us.load(Ordering::Relaxed) as f64 / 1e3);
        let main = self.tier.gpu_ms.lock().unwrap()[0];
        let w = self.tier.waits();
        if passes < 100 || busy <= 0.0 || main.1 < 8 || !w.ready() {
            return serde_json::Value::Null;
        }
        let nl = self.tier.nl as f64;
        let r = |x: f64| (x * 10.0).round() / 10.0;
        let (total, experts, waits) = (busy / passes as f64, main.0 / main.1 as f64 * nl, (w.cpu_us + w.aux_total_us()) * nl / 1000.0);
        serde_json::json!({"total": r(total), "main_gpu_experts": r(experts), "waiting": r(waits), "rest_main_gpu": r((total - experts - waits).max(0.0))})
    }

    fn stats_json(&self, st: &Stats) -> serde_json::Value {
        let h = self.tier.hits();
        let tot = (h.iter().sum::<u64>()).max(1) as f64;
        let rate = |b: u64, s: f64| if s > 0.05 && b >= 64 << 20 { Some((b as f64 / s / 1e6).round()) } else { None };
        let ne = self.tier.ne;
        let on_gpu = (0..self.tier.nl * ne).filter(|&i| self.tier.loc(i) >= LOC_GPU).count();
        serde_json::json!({
            "picks": {"gpu": h[0], "arena": h[1], "ssd": h[2], "file": h[3]},
            "share_from_file": ((h[3] as f64 / tot) * 1000.0).round() / 1000.0,
            "experts_on_gpus": on_gpu,
            "file_side_experts": (0..self.tier.nl * ne).filter(|&i| self.tier.loc(i) == LOC_FILE).count(),
            "moves_planned": st.moves_planned,
            "moves_done": {"cpu_to_gpu": self.tier.done[0].load(Ordering::Relaxed), "between_gpus": self.tier.done[1].load(Ordering::Relaxed),
                           "file_to_ram": self.tier.done[2].load(Ordering::Relaxed)},
            "adaptive_swaps": self.tier.adaptive_swaps.load(Ordering::Relaxed),
            "wait_ms_per_token": (((self.tier.waits().cpu_us + self.tier.waits().aux_total_us()) * self.tier.nl as f64 / 1000.0) * 10.0).round() / 10.0,
            "token_ms": self.token_ms(),
            "wait_for_cpu_us": self.tier.waits().cpu_us.round(),
            "wait_for_aux_us": self.tier.waits().aux_total_us().round(),
            "wait_samples": self.tier.waits().samples,
            "page_cache_warm_gb": (st.warm_bytes as f64 / 1e9 * 10.0).round() / 10.0,
            "model_read_mb_s": self.model_read_mb_s.map(f64::round),
            "ssd_buffer": self.tier.ssd.get().map(|s| serde_json::json!({"path": s.path(), "used_gb": (s.used() as f64 / 1e9 * 10.0).round() / 10.0,
                                                                         "write_mb_s": rate(st.ssd_bytes, st.ssd_secs)})),
            "slow_media": self.slow_media,
            "learned_tokens": self.saved.learned.tokens.round(),
            "learned_share": (self.saved.learned.alpha() * 1000.0).round() / 1000.0,
        })
    }

    /// Plan and prefetch run-time moves into free slots. Returns the number planned.
    pub fn prefetch_moves(&self, m: &dyn ExpertBytes) -> usize {
        let (Some(pf), Some(stores), Some(key)) = (self.tier.prefetch.get(), self.tier.stores.get(), self.tier.key.get()) else { return 0 };
        let free = pf.state.iter().filter(|s| s.load(Ordering::Relaxed) == SLOT_FREE).count();
        if free == 0 {
            return 0;
        }
        let ne = self.tier.ne;
        let loc: Vec<u8> = (0..self.tier.nl * ne).map(|i| self.tier.loc(i)).collect();
        let mut busy = pf.in_transit();
        let w = self.tier.waits();
        let score = &self.saved.learned.score;
        // work off GPUs that finish last first, then off the CPU onto GPUs with room
        let in_arena: Vec<bool> = (0..self.tier.nl * ne).map(|i| self.tier.in_arena(i)).collect();
        // experts exchanged a short while ago rest: they count as busy for further exchanges only
        let mut held = busy.clone();
        held.extend(self.tier.resting().into_iter().map(|i| Move { new: (i / ne, i % ne), store: 0, victim: (i / ne, i % ne), from: None }));
        let n = stores.len();
        let mut moves = plan_exchanges(ne, score, &loc, key, stores, &w.late(n, WAIT_US), &w.early(n), &held, free);
        let mut stage = vec![];
        busy.extend(moves.iter().copied());
        if stores.iter().any(|s| s.tier_swaps) {
            let (more, more_stage) = plan_moves(ne, score, &loc, key, self.tier.has_arena().then_some(&in_arena[..]), stores, &w.open(n), &busy, free - moves.len());
            busy.extend(more.iter().copied());
            moves.extend(more);
            stage.extend(more_stage);
        }
        // the arena follows use in pauses: hot experts of the file side take the place of cold ones in it
        let arena_busy = busy.iter().filter(|m| m.store == TO_ARENA).count();
        if self.tier.is_idle() && arena_busy < ARENA_SLOTS {
            let adaptive: Vec<bool> = stores.iter().map(|s| s.adaptive()).collect();
            let more = plan_arena(ne, score, &loc, key, &in_arena, &adaptive, &stage, &busy, (free - moves.len()).min(ARENA_SLOTS - arena_busy));
            moves.extend(more);
        }
        let mut n = 0;
        for mv in moves {
            let Some(i) = pf.claim() else { break };
            // the bytes come from RAM when it holds them (arena), else the SSD buffer, else the file: a move into the
            // arena is the only one that has to read the file side
            let copy = |(l, e): (usize, usize), off: usize| {
                let mut got = None;
                match self.tier.ssd_get(l, e) {
                    Some(p) => got = pf.put(i, off, p),
                    None => m.read(l, e, &mut |p| got = pf.put(i, off, p)),
                }
                got
            };
            let Some(lens) = copy(mv.new, 0) else {
                pf.unclaim(i);
                continue;
            };
            let vlens = match mv.from {
                None => Some([0; 3]),
                Some(_) => copy(mv.victim, lens.iter().map(|&n| n as usize).sum()),
            };
            match vlens {
                Some(v) => {
                    pf.publish(i, mv, lens, v);
                    n += 1;
                }
                None => pf.unclaim(i),
            }
        }
        n
    }

    /// One idle step: warm the page cache, then extend the SSD buffer. Returns bytes moved.
    pub fn idle_step(&self, m: &dyn ExpertBytes, st: &mut Stats, warmed: &mut std::collections::HashSet<(usize, usize)>) -> u64 {
        let mut moved = 0u64;
        let order = self.order();
        let budget = WARM_MAX_BYTES.saturating_sub(st.warm_bytes).min(ram_headroom());
        let t = std::time::Instant::now();
        for &(l, e) in &order {
            if moved >= STEP_BYTES.min(budget) {
                break;
            }
            if warmed.contains(&(l, e)) {
                continue;
            }
            if !self.tier.is_idle() {
                break; // a request arrived: stop at once
            }
            let p = match self.tier.ssd_get(l, e) {
                Some(s) => s,
                None => m.parts(l, e),
            };
            moved += p.iter().map(|b| touch(b)).sum::<u64>();
            warmed.insert((l, e));
        }
        if moved > 0 {
            st.warm_bytes += moved;
            st.warm_secs += t.elapsed().as_secs_f64();
        }
        if let Some(ssd) = self.tier.ssd.get() {
            let t = std::time::Instant::now();
            let mut copied = 0u64;
            for &(l, e) in &order {
                if copied >= STEP_BYTES || !self.tier.is_idle() {
                    break;
                }
                let i = l * self.tier.ne + e;
                if ssd.holds(i) {
                    continue;
                }
                let p = m.parts(l, e);
                match ssd.put(i, p) {
                    Ok(true) => copied += p.iter().map(|b| b.len() as u64).sum::<u64>(),
                    _ => break,
                }
            }
            if copied > 0 {
                st.ssd_bytes += copied;
                st.ssd_secs += t.elapsed().as_secs_f64();
                let _ = ssd.save_index();
            }
            moved += copied;
        }
        moved
    }

    fn save_now(&mut self, st: &Stats) -> bool {
        self.measure();
        let stats = self.stats_json(st);
        save(&self.state, &self.id, &self.files, self.tier.nl, self.tier.ne, &self.saved, stats).is_ok()
    }

    /// The loop, every 200 ms: prefetch run-time moves; every 10 s fold usage; while idle (a 2 s pause) warm the page
    /// cache and extend the SSD buffer; every 2 min save (busy or not), every 10 min log the figures. Ends, with a
    /// last save, when the tier is stopped (the model is dropped).
    pub fn run(mut self, m: &'static dyn ExpertBytes) {
        let mut last = self.tier.snapshot_picks();
        let mut st = Stats::default();
        let mut warmed = std::collections::HashSet::new();
        let (mut last_save, mut last_fold, mut last_report) = (std::time::Instant::now(), std::time::Instant::now(), std::time::Instant::now());
        let mut last_rewarm = std::time::Instant::now();
        let mut dirty = false;
        while !self.tier.stop.load(Ordering::Relaxed) {
            std::thread::sleep(std::time::Duration::from_millis(TICK_MS));
            if last_fold.elapsed().as_secs() >= FOLD_EVERY_SECS {
                last_fold = std::time::Instant::now();
                let now = self.tier.snapshot_picks();
                let delta: Vec<u32> = now.iter().zip(&last).map(|(a, b)| a.wrapping_sub(*b)).collect();
                last = now;
                if delta.iter().any(|&d| d > 0) {
                    self.saved.learned.fold(&delta, self.top_k, self.tier.nl);
                    dirty = true;
                }
            }
            st.moves_planned += self.prefetch_moves(m) as u64;
            if last_rewarm.elapsed().as_secs() >= REWARM_SECS {
                // walk the warm set again from the hottest: pages still cached cost nothing, evicted ones come back
                last_rewarm = std::time::Instant::now();
                warmed.clear();
                st.warm_bytes = 0;
                st.warm_secs = 0.0;
            }
            if self.tier.is_idle() {
                self.idle_step(m, &mut st, &mut warmed);
            }
            if dirty && last_save.elapsed().as_secs() >= if self.tier.is_idle() { SAVE_IDLE_SECS } else { SAVE_EVERY_SECS } {
                if self.save_now(&st) {
                    st.saves += 1;
                }
                last_save = std::time::Instant::now();
                dirty = false;
            }
            if last_report.elapsed().as_secs() >= REPORT_EVERY_SECS {
                eprintln!("tier: {}", self.stats_json(&st));
                last_report = std::time::Instant::now();
            }
        }
        let _ = self.save_now(&st);
    }
}

/// Set up the SSD buffer for a model on a slow medium; returns whether one is active.
pub fn attach_ssd(tier: &Tier, model: &Path, id: &str, name: &str, lens: Vec<[u32; 3]>, file_side_bytes: u64) -> bool {
    let Some(dir) = ssd_dir(model) else {
        eprintln!("tier: model on a slow medium, but no SSD directory with room on another device (DSX_CACHE_DIR)");
        return false;
    };
    let cap = (free_bytes(&dir).unwrap_or(0) / 2).min(SSD_MAX_BYTES).min(file_side_bytes.next_multiple_of(1 << 30)).next_multiple_of(4096);
    if cap < 1 << 30 {
        return false;
    }
    match SsdBuffer::open(&dir, id, name, lens, cap) {
        Ok(b) => {
            eprintln!("tier: SSD buffer {} ({:.0} GB, {:.1} GB held)", b.path().display(), cap as f64 / 1e9, b.used() as f64 / 1e9);
            tier.ssd.set(b).is_ok()
        }
        Err(e) => {
            eprintln!("tier: SSD buffer unavailable: {e}");
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fold_decays_and_counts_tokens() {
        let mut l = Learned::new(4);
        l.fold(&[6, 0, 0, 0], 6, 1);
        assert_eq!(l.tokens, 1.0);
        assert!((l.score[0] - 6.0).abs() < 1e-9);
        l.score[1] = 10.0;
        let big = (HALF_LIFE_TOKENS * 6.0) as u32;
        l.fold(&[0, 0, big, 0], 6, 1);
        assert!((l.score[1] - 5.0).abs() < 1e-6, "{}", l.score[1]);
        assert!(l.alpha() <= ALPHA_MAX + 1e-12);
    }

    #[test]
    fn blend_keeps_shipped_order_without_learning_and_follows_use_with_it() {
        let base: Vec<(usize, usize, u64)> = vec![(0, 0, 100), (1, 0, 90), (0, 1, 80), (1, 1, 70)];
        let none = Learned::new(4);
        assert_eq!(blend_rank(&base, &none, 2, 2), vec![(0, 0), (1, 0), (0, 1), (1, 1)]);
        let mut used = Learned::new(4);
        used.score[1 * 2 + 1] = 1e6;
        used.tokens = ALPHA_FULL_TOKENS;
        let r = blend_rank(&base, &used, 2, 2);
        assert_eq!(r[0], (1, 1));
        assert_eq!(r.len(), 4);
        let r = blend_rank(&base[..1], &none, 2, 2);
        let mut s = r.clone();
        s.sort();
        s.dedup();
        assert_eq!(s.len(), 4);
        assert_eq!(r[0], (0, 0));
    }

    #[test]
    fn state_roundtrip_previous_version_and_identity() {
        let d = std::env::temp_dir().join(format!("dsx-tier-test-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let p = d.join("m.tier.json");
        let mut s = Saved::new(6);
        s.learned.score[3] = 2.5;
        s.learned.tokens = 7.0;
        s.devices.insert("GPU-a".into(), serde_json::json!({"name": "x", "seen": now_secs()}));
        save(&p, "id-a", &["m.gguf".into()], 2, 3, &s, serde_json::json!({})).unwrap();
        let mut newer = s.clone();
        newer.learned.tokens = 9.0;
        save(&p, "id-a", &["m.gguf".into()], 2, 3, &newer, serde_json::json!({})).unwrap();
        assert_eq!(load(&p, "id-a", 2, 3), Some(newer));
        assert_eq!(load(&p, "id-b", 2, 3), None, "another model");
        assert_eq!(load(&p, "id-a", 3, 3), None, "another shape");
        // a damaged or missing state falls back to the previous version
        std::fs::write(&p, b"{broken").unwrap();
        assert_eq!(load(&p, "id-a", 2, 3), Some(s.clone()));
        std::fs::remove_file(&p).unwrap();
        assert_eq!(load(&p, "id-a", 2, 3), Some(s));
        // odd values in a hand-edited file do not get in
        std::fs::write(&p, br#"{"version":2,"model":{"id":"id-a"},"layers":1,"experts":2,"usage":{"tokens":"x","score":["y",-3]}}"#).unwrap();
        let odd = load_one(&p, "id-a", 1, 2).unwrap();
        assert_eq!((odd.learned.tokens, odd.learned.score), (0.0, vec![0.0, 0.0]));
        // temp files of killed processes are removed, the state stays
        std::fs::write(d.join("m.tier.json.tmp.1"), b"x").unwrap();
        remove_stale_temps(&p);
        assert!(!d.join("m.tier.json.tmp.1").exists() && p.exists());
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn mount_source_takes_the_longest_mount_point() {
        let info = "36 1 0:31 /root / rw - btrfs /dev/nvme0n1p3 rw\n\
                    40 36 0:35 /home /home rw - btrfs /dev/nvme0n1p3 rw\n\
                    55 36 8:17 / /mnt/daten rw - ext4 /dev/sdb1 rw\n\
                    60 36 0:50 / /mnt/my\\040disk rw - btrfs /dev/sdc1 rw\n\
                    70 36 0:60 / /proc rw - proc proc rw\n";
        let src = |p: &str| mount_source(info, Path::new(p));
        assert_eq!(src("/mnt/daten/models/m.gguf").as_deref(), Some("/dev/sdb1"));
        assert_eq!(src("/home/u/m.gguf").as_deref(), Some("/dev/nvme0n1p3"));
        assert_eq!(src("/mnt/datenbank/m.gguf").as_deref(), Some("/dev/nvme0n1p3"), "a path prefix is not a name prefix");
        assert_eq!(src("/mnt/my disk/m.gguf").as_deref(), Some("/dev/sdc1"));
    }

    #[test]
    fn the_switch_turns_off_the_learned_placement_too() {
        use std::ffi::OsStr;
        assert!(enabled_for(None) && enabled_for(Some(OsStr::new("1"))) && enabled_for(Some(OsStr::new(""))));
        assert!(!enabled_for(Some(OsStr::new("0"))));
        let d = std::env::temp_dir().join(format!("dsx-tier-off-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let (st, other) = (d.join("m.tier.json"), d.join("elsewhere.tier.json"));
        let mut s = Saved::new(2);
        s.learned.tokens = 82004.0;
        s.learned.score = vec![1.0, 5.0];
        save(&other, "id", &[], 1, 2, &s, serde_json::json!({})).unwrap();
        let cands = vec![other.clone()];
        let on = load_for_start(true, Some((&st, "id")), &cands, 1, 2);
        assert_eq!((on.learned.tokens, on.learned.score.clone()), (82004.0, vec![1.0, 5.0]), "found where an earlier start left it");
        let off = load_for_start(false, Some((&st, "id")), &cands, 1, 2);
        assert_eq!((off.learned.tokens, off.learned.score), (0.0, vec![0.0, 0.0]), "off: the state on disk is not read");
        assert_eq!(load_for_start(true, None, &cands, 1, 2).learned.tokens, 0.0);
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn requests_and_pauses_decide_idleness() {
        let t: &'static Tier = Box::leak(Box::new(Tier::new(1, 1)));
        t.last_busy_us.store(0, Ordering::Relaxed);
        std::thread::sleep(std::time::Duration::from_millis(5));
        let far = t.t0.elapsed().as_micros() as u64 + IDLE_MS * 1000;
        let idle_at = |now: u64| t.active.load(Ordering::Relaxed) == 0 && (now - t.last_busy_us.load(Ordering::Relaxed)) / 1000 >= IDLE_MS;
        assert!(idle_at(far));
        let b = t.busy();
        assert!(!t.is_idle() && !idle_at(far), "a running request is busy however long it takes");
        drop(b);
        assert!(!t.is_idle(), "the pause starts when the request ends");
        assert_eq!(t.active.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn rotation_drops_old_devices_and_extra_configs() {
        let mut s = Saved::new(1);
        let now = 1_000_000_000;
        s.devices.insert("GPU-old".into(), serde_json::json!({"seen": now - DEVICE_KEEP_SECS - 1}));
        s.devices.insert("GPU-new".into(), serde_json::json!({"seen": now - 10}));
        for i in 0..20u64 {
            s.configs.insert(format!("c{i}"), serde_json::json!({"seen": now - 100 + i}));
        }
        s.rotate(now);
        assert!(!s.devices.contains_key("GPU-old") && s.devices.contains_key("GPU-new"));
        assert_eq!(s.configs.len(), MAX_CONFIGS);
        assert!(s.configs.contains_key("c19") && !s.configs.contains_key("c0"));
    }

    #[test]
    fn names_follow_model_shards_and_overlays() {
        let p = |s: &str| PathBuf::from(s);
        assert_eq!(state_name(&[p("/m/a.gguf")]), "a.gguf");
        assert_eq!(state_name(&[p("/m/a-00001-of-00003.gguf"), p("/m/a-00002-of-00003.gguf"), p("/m/a-00003-of-00003.gguf")]), "a-00001-of-00003.gguf");
        assert_eq!(state_name(&[p("/m/a.gguf"), p("/o/a-uncensored-overlay.gguf")]), "a.gguf+a-uncensored-overlay");
        assert_eq!(state_name(&[p("/m/a-00001-of-00002.gguf"), p("/m/a-00002-of-00002.gguf"), p("/o/ov.gguf")]), "a-00001-of-00002.gguf+ov");
        // next to the model when writable
        let d = std::env::temp_dir().join(format!("dsx-tier-name-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        assert_eq!(state_path(&[d.join("x.gguf")]), d.join("x.gguf.tier.json"));
        assert_ne!(state_path(&[d.join("x.gguf"), d.join("ov.gguf")]), state_path(&[d.join("x.gguf")]));
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn reset_removes_state_prev_and_buffer() {
        let d = std::env::temp_dir().join(format!("dsx-tier-reset-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let st = d.join("m.gguf.tier.json");
        std::fs::write(&st, b"{}").unwrap();
        std::fs::write(st.with_extension("json.prev"), b"{}").unwrap();
        assert_eq!(reset_files(&st, "m.gguf-not-buffered"), 2);
        assert!(!st.exists());
        request_reset();
        assert!(reset_requested());
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn only_the_first_model_of_its_shape_claims_tiering() {
        let t = Tier::new(2, 8);
        assert!(!t.claim(3, 8), "other shape");
        assert!(t.claim(2, 8));
        assert!(!t.claim(2, 8), "second model");
        assert!(!t.worker.swap(true, Ordering::AcqRel) && t.worker.load(Ordering::Relaxed), "one worker only");
    }

    #[test]
    fn picks_hits_and_store_counts() {
        let t = Tier::new(2, 8);
        t.note_picks(1, &[7, 3, 3]);
        let s = t.snapshot_picks();
        assert_eq!(s[8 + 7], 1);
        assert_eq!(s[8 + 3], 2);
        t.note_picks(1, &[99]); // out of range: ignored
        t.note_hit(Hit::File, 2);
        assert_eq!(t.hits()[3], 2);
        t.set_loc(1, 3, LOC_GPU + 1);
        t.note_gpu(1, 3);
        t.note_gpu(1, 4); // not on a GPU per loc: tier count only
        assert_eq!(t.hits()[0], 2);
        assert_eq!(t.store_picks[1].load(Ordering::Relaxed), 1);
        t.note_decode();
        assert!(t.idle_ms() < 1000 && !t.is_idle());
    }

    #[test]
    fn plan_moves_hot_off_gpu_against_clearly_colder_same_key_residents() {
        // 1 layer x 8 experts; store 0 swaps by tiering, store 1 is adaptive (left alone)
        let stores = vec![
            StoreInfo { uuid: "GPU-a".into(), name: "a".into(), link_gbs: 3.0, tier_swaps: true, swaps: true },
            StoreInfo { uuid: "GPU-b".into(), name: "b".into(), link_gbs: 24.0, tier_swaps: false, swaps: true },
        ];
        let loc = [LOC_GPU, LOC_GPU, LOC_GPU + 1, LOC_FILE, LOC_ARENA, LOC_FILE, LOC_FILE, LOC_GPU];
        let key = [1, 1, 1, 1, 1, 2, 1, 1];
        let score = [5.0, 100.0, 1.0, 50.0, 30.0, 999.0, 3.0, 20.0];
        let (m, _) = plan_moves(8, &score, &loc, &key, None, &stores, &[true, true], &[], 4);
        // 50 replaces the coldest tier store resident (5), 30 the next (20 > 15: no), 999 has no same-key victim,
        // 3 is below the minimum, store 1's resident (1) is not a tiering victim
        assert_eq!(m, vec![Move { new: (0, 3), store: 0, victim: (0, 0), from: None }]);
        // a move in transit blocks both experts
        let (m2, _) = plan_moves(8, &score, &loc, &key, None, &stores, &[true, true], &m, 4);
        assert!(m2.iter().all(|x| x.new != (0, 3) && x.victim != (0, 0)));
        // a store on the critical path takes nothing
        assert!(plan_moves(8, &score, &loc, &key, None, &stores, &[false, true], &[], 4).0.is_empty());
        // with an arena the victim (0) needs its copy before it goes back to the CPU: staged with the candidate's score
        let (m3, stage) = plan_moves(8, &score, &loc, &key, Some(&[false; 8]), &stores, &[true, true], &[], 4);
        assert!(m3.is_empty());
        assert_eq!(stage, vec![(0, 50.0)]);
        let mut c = [false; 8];
        c[0] = true;
        assert_eq!(plan_moves(8, &score, &loc, &key, Some(&c), &stores, &[true, true], &[], 4), (m.clone(), vec![]));
    }

    #[test]
    fn waits_decide_which_gpu_may_take_cpu_work_and_which_is_late() {
        let mut w = Waits::default();
        assert_eq!(w.open(2), vec![true, true], "unmeasured: open");
        assert!(w.late(2, WAIT_US).is_empty(), "unmeasured: nothing late");
        for _ in 0..WAIT_MIN_SAMPLES {
            w.add(300.0, &[0.0]); // the main GPU waits for the CPU; the aux GPU is done before it
        }
        assert_eq!(w.open(2), vec![true, true]);
        assert!(w.late(2, WAIT_US).is_empty());
        let mut a = w;
        for _ in 0..200 {
            a.add(115.0, &[90.0]); // the station on 11.11.: the aux GPU finishes last
        }
        assert_eq!(a.open(2), vec![true, false]);
        assert_eq!(a.late(2, WAIT_US), vec![1]);
        let mut c = w;
        for _ in 0..200 {
            c.add(0.0, &[0.0]); // nobody is waited for: no move helps
        }
        assert_eq!(c.open(2), vec![false, false]);
        // three GPUs: only the second aux GPU is late
        let mut t3 = Waits::default();
        for _ in 0..WAIT_MIN_SAMPLES {
            t3.add(50.0, &[0.0, 70.0]);
        }
        assert_eq!(t3.late(3, WAIT_US), vec![2]);
        assert_eq!(t3.early(3), vec![0, 1]);
        assert_eq!(t3.open(3), vec![true, true, false]);
        let t = Tier::new(1, 1);
        t.note_waits(0.3, &[0.09]);
        assert_eq!(t.waits().samples, 1);
        assert!((t.waits().cpu_us - 300.0).abs() < 1e-9 && (t.waits().aux_us[0] - 90.0).abs() < 1e-9);
    }

    #[test]
    fn exchanges_take_hot_experts_off_late_gpus_onto_the_main_one() {
        let st = |fixed| StoreInfo { uuid: "u".into(), name: "n".into(), link_gbs: 1.0, tier_swaps: fixed, swaps: fixed };
        let stores = vec![st(true), st(true)];
        // 1 layer x 8: main holds 0 (cold), 1 (warm); aux holds 2 (hot), 3 (hotter), 4 (other key); 5 on the CPU
        let loc = [LOC_GPU, LOC_GPU, LOC_GPU + 1, LOC_GPU + 1, LOC_GPU + 1, LOC_ARENA, LOC_FILE, LOC_FILE];
        let key = [1, 1, 1, 1, 2, 1, 1, 1];
        let score = [2.0, 40.0, 50.0, 90.0, 500.0, 70.0, 0.0, 0.0];
        let m = plan_exchanges(8, &score, &loc, &key, &stores, &[1], &[0], &[], 4);
        // 90 swaps with the coldest main resident (2); 50 would need one at most 25: 40 is too warm
        assert_eq!(m, vec![Move { new: (0, 3), store: 0, victim: (0, 0), from: Some(1) }]);
        assert!(plan_exchanges(8, &score, &loc, &key, &stores, &[], &[0, 1], &[], 4).is_empty(), "nothing late");
        assert!(plan_exchanges(8, &score, &loc, &key, &[st(false), st(true)], &[1], &[0], &[], 4).is_empty(), "main cannot swap");
        // an adaptive store (swaps by itself, no tier moves) takes no part: it hands experts back to the CPU by its own rule
        let adaptive = StoreInfo { uuid: "u".into(), name: "n".into(), link_gbs: 1.0, tier_swaps: false, swaps: true };
        assert!(adaptive.adaptive());
        assert!(plan_exchanges(8, &score, &loc, &key, &[adaptive, st(true)], &[1], &[0], &[], 4).is_empty(), "main is adaptive");
        assert!(plan_exchanges(8, &score, &loc, &key, &stores, &[1], &[0], &m, 4).is_empty(), "in transit");
        // three GPUs of different kinds: the late one (2) hands its hottest to whichever early GPU holds the coldest
        let stores3 = vec![st(true), st(true), st(true)];
        let loc3 = [LOC_GPU, LOC_GPU + 1, LOC_GPU + 2, LOC_GPU + 2, LOC_ARENA, LOC_FILE, LOC_FILE, LOC_FILE];
        let score3 = [30.0, 3.0, 80.0, 9.0, 0.0, 0.0, 0.0, 0.0];
        let m3 = plan_exchanges(8, &score3, &loc3, &[1; 8], &stores3, &[2], &[0, 1], &[], 4);
        assert_eq!(m3, vec![Move { new: (0, 2), store: 1, victim: (0, 1), from: Some(2) }]);
    }

    #[test]
    fn arena_takes_hot_file_side_experts_for_cold_ones() {
        // 1 layer x 6: arena holds 0 (cold), 1 (warm), 2 (other key); file side 3 (hot), 4 (lukewarm); 5 on a GPU
        let loc = [LOC_ARENA, LOC_ARENA, LOC_ARENA, LOC_FILE, LOC_FILE, LOC_GPU];
        let key = [1, 1, 2, 1, 1, 1];
        let score = [4.0, 30.0, 0.0, 50.0, 20.0, 99.0];
        let in_arena = [true, true, true, false, false, false];
        let m = plan_arena(6, &score, &loc, &key, &in_arena, &[true], &[], &[], 4);
        // 50 takes the place of 4; 20 would need a victim of at most 10: 30 is too warm
        assert_eq!(m, vec![Move { new: (0, 3), store: TO_ARENA, victim: (0, 0), from: None }]);
        assert!(plan_arena(6, &score, &loc, &key, &in_arena, &[true], &[], &m, 4).is_empty(), "in transit");
        assert!(plan_arena(6, &score, &[LOC_FILE; 6], &key, &[false; 6], &[true], &[], &[], 4).is_empty(), "no arena");
        // 5 is resident on a store and still has an arena copy: on a fixed-set store nobody reads that copy, so it goes
        // first, even for the lukewarm 4; on an adaptive store it stays (the store may hand 5 back to the CPU)
        let in_arena5 = [true, true, true, false, false, true];
        let m = plan_arena(6, &score, &loc, &key, &in_arena5, &[false], &[], &[], 4);
        assert_eq!(m, vec![
            Move { new: (0, 3), store: TO_ARENA, victim: (0, 5), from: None },
            Move { new: (0, 4), store: TO_ARENA, victim: (0, 0), from: None },
        ]);
        let m = plan_arena(6, &score, &loc, &key, &in_arena5, &[true], &[], &[], 4);
        assert_eq!(m, vec![Move { new: (0, 3), store: TO_ARENA, victim: (0, 0), from: None }]);
        // staging: GPU resident 5 (no copy yet) needs one before an exchange; it comes first and takes the cold 0
        let m = plan_arena(6, &score, &loc, &key, &in_arena, &[false], &[(5, 99.0)], &[], 1);
        assert_eq!(m, vec![Move { new: (0, 5), store: TO_ARENA, victim: (0, 0), from: None }]);
        assert!(plan_arena(6, &score, &loc, &key, &in_arena5, &[false], &[(5, 99.0)], &[], 1).iter().all(|x| x.new != (0, 5)), "has its copy");
        // two copies nobody reads (0 cold, 5 hot, both on a fixed-set GPU): the hot one's goes first, the cold resident
        // is the next to go back to the CPU and will need its own
        let loc2 = [LOC_GPU, LOC_ARENA, LOC_ARENA, LOC_FILE, LOC_FILE, LOC_GPU];
        let m = plan_arena(6, &score, &loc2, &key, &in_arena5, &[false], &[], &[], 1);
        assert_eq!(m, vec![Move { new: (0, 3), store: TO_ARENA, victim: (0, 5), from: None }]);
    }

    #[test]
    fn prefetch_slots_fill_take_release() {
        let mut mem = vec![0u8; 2 * 128];
        let pf = Prefetch::new(mem.as_mut_ptr(), 128, 2);
        let mv = Move { new: (0, 1), store: 0, victim: (0, 2), from: None };
        let i = pf.claim().unwrap();
        assert!(pf.fill(i, mv, [&[1u8; 10], &[2u8; 20], &[3u8; 30]], None));
        assert!(pf.victim_parts(i).is_none());
        assert_eq!(pf.in_transit(), vec![mv]);
        let (j, got) = pf.take(false).unwrap();
        assert_eq!((j, got), (i, mv));
        let p = pf.parts(j).unwrap();
        assert_eq!((p[0].len(), p[1][0], p[2][29]), (10, 2, 3));
        assert!(pf.take(false).is_none(), "busy until released");
        pf.release(j);
        assert!(pf.in_transit().is_empty());
        let k = pf.claim().unwrap();
        assert!(!pf.fill(k, mv, [&[0u8; 100], &[0u8; 20], &[0u8; 10]], None), "larger than a slot");
        // an exchange carries both experts
        let x = Move { new: (0, 3), store: 0, victim: (0, 4), from: Some(1) };
        let k = pf.claim().unwrap();
        assert!(pf.fill(k, x, [&[7u8; 5], &[7u8; 5], &[7u8; 5]], Some([&[9u8; 4], &[9u8; 4], &[9u8; 4]])));
        let (j, _) = pf.take(false).unwrap();
        let v = pf.victim_parts(j).unwrap();
        assert_eq!((v[0].len(), v[2][3]), (4, 9));
        assert_eq!(pf.parts(j).unwrap()[0][0], 7);
        pf.release(j);
        // a move into the arena is only handed out in a pause
        let a = Move { new: (0, 5), store: TO_ARENA, victim: (0, 6), from: None };
        let k = pf.claim().unwrap();
        assert!(pf.fill(k, a, [&[1u8; 5], &[1u8; 5], &[1u8; 5]], None));
        assert!(pf.take(false).is_none());
        assert_eq!(pf.take(true), Some((k, a)));
        drop(mem);
    }

    struct Fake(Vec<Vec<u8>>);
    impl ExpertBytes for Fake {
        fn parts(&self, l: usize, e: usize) -> [&[u8]; 3] {
            let b = &self.0[l * 4 + e];
            [&b[..100], &b[100..300], &b[300..]]
        }
    }

    #[test]
    fn ssd_buffer_put_get_persist_grow_and_refuse_when_full() {
        let d = std::env::temp_dir().join(format!("dsx-ssd-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        let lens = vec![[100u32, 200, 4000]; 8];
        let fake = Fake((0..8).map(|i| vec![i as u8 + 1; 4300]).collect());
        let open = |id: &str, want: u64| {
            let mut b = SsdBuffer::open(&d, id, "m.gguf", lens.clone(), want).unwrap();
            b.min_free = 0;
            b
        };
        let b = open("id", 3 * 8192); // entries are 4 KiB aligned: 3 fit
        assert!(b.get(5).is_none());
        assert!(b.put(5, fake.parts(1, 1)).unwrap());
        let g = b.get(5).unwrap();
        assert_eq!((g[0].len(), g[1].len(), g[2].len()), (100, 200, 4000));
        assert!(g[2].iter().all(|&x| x == 6));
        assert!(b.put(2, fake.parts(0, 2)).unwrap());
        assert!(b.put(3, fake.parts(0, 3)).unwrap());
        assert!(!b.put(4, fake.parts(1, 0)).unwrap(), "full");
        b.save_index().unwrap();
        drop(b);
        // the next start asks for less (other placement, less free space): the buffer and its entries stay
        let b2 = open("id", 8192);
        assert!(b2.holds(5) && b2.holds(2) && !b2.holds(4));
        assert!(b2.get(2).unwrap()[0].iter().all(|&x| x == 3));
        assert!(!b2.put(4, fake.parts(1, 0)).unwrap(), "still full");
        drop(b2);
        // asked for more: it grows and keeps what it holds
        let b3 = open("id", 5 * 8192);
        assert!(b3.holds(5) && b3.put(4, fake.parts(1, 0)).unwrap());
        assert!(b3.get(4).unwrap()[2].iter().all(|&x| x == 5));
        // one process at a time: a second one gets no buffer (and reads the model file)
        assert!(SsdBuffer::open(&d, "id", "m.gguf", lens.clone(), 5 * 8192).is_err());
        drop(b3);
        // too little free space left on the disk: nothing more is written
        let mut low = open("id", 5 * 8192);
        low.min_free = u64::MAX;
        assert!(!low.put(6, fake.parts(1, 2)).unwrap());
        drop(low);
        let other = open("other-model", 3 * 8192);
        assert!(!other.holds(5), "another model starts empty");
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn idle_step_warms_hottest_file_side_first_and_measure_records_devices() {
        let tier: &'static Tier = Box::leak(Box::new(Tier::new(2, 4)));
        // file side: (0,1), (1,2), (0,3); the rest on a GPU or in the arena
        for i in 0..8 {
            tier.loc[i].store(LOC_ARENA, Ordering::Relaxed);
        }
        for (l, e) in [(0, 1), (1, 2), (0, 3)] {
            tier.set_loc(l, e, LOC_FILE);
        }
        tier.set_loc(0, 0, LOC_GPU);
        tier.set_stores(vec![StoreInfo { uuid: "GPU-x".into(), name: "X".into(), link_gbs: 24.6, tier_swaps: true, swaps: true }]);
        let fake = Fake((0..8).map(|i| vec![i as u8; 4300]).collect());
        let mut saved = Saved::new(8);
        saved.learned.score[1 * 4 + 2] = 9.0;
        saved.learned.score[0 * 4 + 1] = 5.0;
        let mut bg = Background {
            tier, state: std::env::temp_dir().join("x.json"), id: "i".into(), files: vec![], top_k: 6, saved,
            model_disk: Some("disk:8:0".into()), model_read_mb_s: Some(180.0), slow_media: true,
        };
        assert_eq!(bg.order()[0], (1, 2));
        assert_eq!(bg.order().len(), 3);
        let mut st = Stats::default();
        let mut warmed = std::collections::HashSet::new();
        if ram_headroom() > 0 {
            tier.last_busy_us.store(0, Ordering::Relaxed);
            std::thread::sleep(std::time::Duration::from_millis(IDLE_MS + 50));
            assert!(tier.is_idle());
            let moved = bg.idle_step(&fake, &mut st, &mut warmed);
            assert_eq!(moved, 3 * 4300);
            assert!(warmed.contains(&(1, 2)));
        }
        for _ in 0..10 {
            tier.note_gpu_ms(0, 0.5);
        }
        tier.passes.store(200, Ordering::Relaxed);
        tier.all_passes.store(200, Ordering::Relaxed);
        tier.busy_us.store(10_000_000, Ordering::Relaxed);
        tier.store_picks[0].store(200 * 2 * 3, Ordering::Relaxed); // 3 picks per layer, 2 layers
        bg.saved.devices.insert("GPU-x".into(), serde_json::json!({"expert_us": 99.0}));
        bg.measure();
        let d = &bg.saved.devices["GPU-x"];
        assert_eq!(d["name"], "X");
        assert_eq!(d["expert_layer_ms"], 0.5);
        assert_eq!(d["picks_per_token"], 6.0);
        assert!(d.get("expert_us").is_none(), "the old per-pick cost is dropped");
        assert_eq!(bg.saved.devices["disk:8:0"]["read_mb_s"], 180.0);
        assert_eq!(bg.saved.configs["GPU-x"]["decode_tok_s"], 20.0);
        let j = bg.stats_json(&st);
        assert!(j["share_from_file"].is_number());
        assert_eq!(j["experts_on_gpus"], 1);
    }
}
