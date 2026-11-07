//! Persistent pinned worker pool: one job per call, all workers run it, spin-then-park between calls.
//! The caller does not take part; it spins until every worker is done.

use std::cell::UnsafeCell;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering::*};
use std::thread::{self, JoinHandle, Thread};
use std::time::{Duration, Instant};

#[repr(align(128))]
#[derive(Default)]
struct Line<T>(T);

type Job = *const (dyn Fn(&Ctx) + Sync);

struct Shared {
    n: usize,
    epoch: Line<AtomicU64>,
    done: Line<AtomicUsize>,
    bar_count: Line<AtomicUsize>,
    bar_gen: Line<AtomicUsize>,
    sleeping: Vec<Line<AtomicBool>>,
    stop: AtomicBool,
    spin: Duration,
    job: UnsafeCell<Option<Job>>,
}
unsafe impl Sync for Shared {}
unsafe impl Send for Shared {}

pub struct Ctx<'a> {
    pub id: usize,
    pub n: usize,
    sh: &'a Shared,
}

impl Ctx<'_> {
    /// Spin barrier across the pool's workers.
    pub fn barrier(&self) {
        let sh = self.sh;
        let g = sh.bar_gen.0.load(Acquire);
        if sh.bar_count.0.fetch_add(1, AcqRel) == sh.n - 1 {
            sh.bar_count.0.store(0, Relaxed);
            sh.bar_gen.0.store(g + 1, Release);
        } else {
            while sh.bar_gen.0.load(Acquire) == g {
                std::hint::spin_loop();
            }
        }
    }
}

pub struct Pool {
    sh: Arc<Shared>,
    threads: Vec<Thread>,
    handles: Vec<JoinHandle<()>>,
}

unsafe extern "C" {
    fn sched_setaffinity(pid: i32, cpusetsize: usize, mask: *const u64) -> i32;
    fn sched_getaffinity(pid: i32, cpusetsize: usize, mask: *mut u64) -> i32;
}

/// Pin the calling thread to one logical CPU.
pub fn pin_to(cpu: usize) -> bool {
    pin_to_set(&[cpu])
}

/// restrict the calling thread to `cpus`
pub fn pin_to_set(cpus: &[usize]) -> bool {
    let mut set = [0u64; 16];
    for &cpu in cpus {
        set[cpu / 64] |= 1 << (cpu % 64);
    }
    unsafe { sched_setaffinity(0, std::mem::size_of_val(&set), set.as_ptr()) == 0 }
}

// i7-12700K: CPUs 0-15 are the 8 P-cores (two threads each), 16-19 the E-cores. Every spinning thread gets a CPU
// of its own: two spinners on one CPU time-slice, and the loser sees its flag milliseconds late.
/// the thread that enqueues GPU work (alone on P-core 7; its sibling 15 stays nearly idle)
pub const MAIN_CPU: usize = 14;
/// the CPU expert worker's coordinator (spins on the route flag)
pub const WORKER_CPU: usize = 16;
/// the decode Engram prefetch and other helpers: off the pool's, the main thread's and the worker's CPUs
pub const AUX_CPUS: [usize; 4] = [15, 17, 18, 19];
/// every CPU (prefill helpers run while the expert pool sleeps)
pub const ALL_CPUS: [usize; 20] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19];

/// Logical CPUs 0..15: the 8 P-cores with their hyperthreads on the i7-12700K (and 13700).
pub const P_CORES: [usize; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];

/// Where the pinned threads go on this machine.
///
/// When the process may run on every CPU of the reference layout above (CPUs 0-19), that layout is used unchanged.
/// Otherwise (fewer CPUs, a VM, a container cpuset) the constants name CPUs the process cannot use: `pin_to` fails
/// and the two spinning threads (GPU issue, CPU-expert coordinator) float over the expert pool's cores. The layout is
/// then derived from the allowed CPUs: the last two go to the main and coordinator threads, the rest to the pool,
/// and the helpers share the last two (the convention `tests/engram_gather.rs` uses on small runners).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Topology {
    /// the thread that enqueues GPU work
    pub main: usize,
    /// the CPU expert worker's coordinator
    pub worker: usize,
    /// candidates for the CPU expert pool, in order (the pool takes the first `DSX_CPU_THREADS`)
    pub pool: Vec<usize>,
    /// decode Engram prefetch and other helpers
    pub aux: Vec<usize>,
    /// every CPU the process may use (prefill helpers run while the expert pool sleeps)
    pub all: Vec<usize>,
    /// true when the reference (i7-12700K) constants are in use
    pub reference: bool,
}

impl Topology {
    /// the layout for a process allowed to run on `allowed` (sorted, deduplicated)
    pub fn for_cpus(allowed: &[usize]) -> Topology {
        if ALL_CPUS.iter().all(|c| allowed.contains(c)) {
            return Topology {
                main: MAIN_CPU,
                worker: WORKER_CPU,
                pool: P_CORES.iter().copied().filter(|&c| c != MAIN_CPU).collect(),
                aux: AUX_CPUS.to_vec(),
                all: ALL_CPUS.to_vec(),
                reference: true,
            };
        }
        let n = allowed.len();
        assert!(n > 0, "no allowed CPUs");
        if n < 4 {
            // too few CPUs to give the spinners their own: everything shares
            let last = allowed[n - 1];
            return Topology { main: last, worker: last, pool: allowed.to_vec(), aux: allowed.to_vec(), all: allowed.to_vec(), reference: false };
        }
        Topology {
            main: allowed[n - 2],
            worker: allowed[n - 1],
            pool: allowed[..n - 2].to_vec(),
            aux: allowed[n - 2..].to_vec(),
            all: allowed.to_vec(),
            reference: false,
        }
    }

    /// default number of expert-pool threads (`DSX_CPU_THREADS` overrides): 14 on the reference layout
    pub fn default_threads(&self) -> usize {
        14.min(self.pool.len())
    }

    /// most expert-pool threads `DSX_CPU_THREADS` may ask for (the reference layout keeps the main thread's CPU free)
    pub fn max_threads(&self) -> usize {
        if self.reference { MAIN_CPU } else { self.pool.len() }
    }

    /// CPUs of the synchronous (non-handoff) expert pool: all 16 P-core threads on the reference layout
    pub fn sync_pool(&self) -> Vec<usize> {
        if self.reference { P_CORES.to_vec() } else { self.pool.clone() }
    }
}

/// the CPUs the calling thread may run on
pub fn allowed_cpus() -> Vec<usize> {
    let mut set = [0u64; 16];
    if unsafe { sched_getaffinity(0, std::mem::size_of_val(&set), set.as_mut_ptr()) } != 0 {
        return ALL_CPUS.to_vec();
    }
    (0..set.len() * 64).filter(|&c| set[c / 64] >> (c % 64) & 1 == 1).collect()
}

/// This machine's layout, read from the process affinity on first use. The first use must come before any thread
/// pins itself (`GpuModel::new` calls it first).
pub fn topology() -> &'static Topology {
    static T: std::sync::OnceLock<Topology> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let t = Topology::for_cpus(&allowed_cpus());
        if !t.reference {
            eprintln!(
                "cpu layout: {} allowed CPUs (not the reference 0-19): expert pool {:?}, GPU issue {}, coordinator {}, helpers {:?}",
                t.all.len(),
                t.pool,
                t.main,
                t.worker,
                t.aux
            );
        }
        t
    })
}

impl Pool {
    /// One worker per entry of `cpus`, pinned there; workers spin `spin` after a job before parking.
    pub fn new(cpus: &[usize], spin: Duration) -> Pool {
        let n = cpus.len();
        assert!(n > 0);
        let sh = Arc::new(Shared {
            n,
            epoch: Line(AtomicU64::new(0)),
            done: Default::default(),
            bar_count: Default::default(),
            bar_gen: Default::default(),
            sleeping: (0..n).map(|_| Line::default()).collect(),
            stop: AtomicBool::new(false),
            spin,
            job: UnsafeCell::new(None),
        });
        let handles: Vec<JoinHandle<()>> = cpus
            .iter()
            .enumerate()
            .map(|(id, &cpu)| {
                let sh = sh.clone();
                thread::Builder::new()
                    .name(format!("cpu-exp-{id}"))
                    .spawn(move || worker(&sh, id, cpu))
                    .expect("spawn pool worker")
            })
            .collect();
        let threads = handles.iter().map(|h| h.thread().clone()).collect();
        Pool { sh, threads, handles }
    }

    pub fn threads(&self) -> usize {
        self.sh.n
    }

    /// Run `f` on every worker (ctx.id = 0..n) and wait for all of them.
    pub fn run<F: Fn(&Ctx) + Sync>(&self, f: &F) {
        let sh = &*self.sh;
        let job: *const (dyn Fn(&Ctx) + Sync + '_) = f;
        // lifetime erased: we do not return before every worker has finished with it
        unsafe { *sh.job.get() = Some(std::mem::transmute::<*const (dyn Fn(&Ctx) + Sync + '_), Job>(job)) };
        sh.done.0.store(0, Relaxed);
        sh.epoch.0.fetch_add(1, SeqCst);
        for (i, t) in self.threads.iter().enumerate() {
            if sh.sleeping[i].0.load(SeqCst) {
                t.unpark();
            }
        }
        while sh.done.0.load(Acquire) != sh.n {
            std::hint::spin_loop();
        }
    }
}

fn worker(sh: &Shared, id: usize, cpu: usize) {
    if !pin_to(cpu) {
        eprintln!("cpu pool: could not pin worker {id} to cpu {cpu}");
    }
    let mut seen = 0u64;
    loop {
        // wait for the next generation: spin, then park
        let t0 = Instant::now();
        let mut i = 0u32;
        loop {
            if sh.epoch.0.load(Acquire) != seen {
                break;
            }
            std::hint::spin_loop();
            i = i.wrapping_add(1);
            if i % 1024 == 0 && t0.elapsed() > sh.spin {
                sh.sleeping[id].0.store(true, SeqCst);
                if sh.epoch.0.load(SeqCst) == seen {
                    thread::park();
                }
                sh.sleeping[id].0.store(false, Relaxed);
            }
        }
        seen = sh.epoch.0.load(Acquire);
        if sh.stop.load(Acquire) {
            return;
        }
        let job = unsafe { (*sh.job.get()).unwrap() };
        let ctx = Ctx { id, n: sh.n, sh };
        if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe { (*job)(&ctx) })).is_err() {
            eprintln!("cpu pool worker {id} panicked");
            std::process::abort();
        }
        sh.done.0.fetch_add(1, Release);
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.sh.stop.store(true, Release);
        self.sh.epoch.0.fetch_add(1, SeqCst);
        for t in &self.threads {
            t.unpark();
        }
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topology_reference_layout_unchanged() {
        for allowed in [(0..20).collect::<Vec<_>>(), (0..32).collect()] {
            let t = Topology::for_cpus(&allowed);
            assert!(t.reference);
            assert_eq!((t.main, t.worker), (MAIN_CPU, WORKER_CPU));
            assert_eq!(t.aux, AUX_CPUS.to_vec());
            assert_eq!(t.all, ALL_CPUS.to_vec());
            // the handoff pool's default 14 threads: CPUs 0-13, as before
            assert_eq!(&t.pool[..t.default_threads()], &(0..14).collect::<Vec<_>>()[..]);
            assert_eq!(t.max_threads(), MAIN_CPU);
            assert_eq!(t.sync_pool(), P_CORES.to_vec());
        }
    }

    #[test]
    fn topology_small_machine() {
        // a 12-CPU VM: pool 0-9, spinners on 10 and 11, nothing outside the allowed set
        let t = Topology::for_cpus(&(0..12).collect::<Vec<_>>());
        assert!(!t.reference);
        assert_eq!((t.main, t.worker), (10, 11));
        assert_eq!(t.pool, (0..10).collect::<Vec<_>>());
        assert_eq!(t.default_threads(), 10);
        assert_eq!(t.aux, vec![10, 11]);
        assert!(!t.pool.contains(&t.main) && !t.pool.contains(&t.worker));
        // a cpuset that does not start at 0
        let t = Topology::for_cpus(&(20..28).collect::<Vec<_>>());
        assert_eq!((t.main, t.worker, t.pool.clone()), (26, 27, (20..26).collect::<Vec<_>>()));
        // tiny: everything shares
        let t = Topology::for_cpus(&[0, 1]);
        assert_eq!((t.main, t.worker, t.pool.len()), (1, 1, 2));
    }

    #[test]
    fn allowed_cpus_matches_parallelism() {
        let n = std::thread::available_parallelism().unwrap().get();
        assert_eq!(allowed_cpus().len(), n);
    }

    #[test]
    fn runs_and_barriers() {
        let pool = Pool::new(&[0, 1, 2, 3], Duration::from_micros(200));
        let hits = AtomicUsize::new(0);
        for round in 0..200 {
            let before = hits.load(Relaxed);
            pool.run(&|c: &Ctx| {
                hits.fetch_add(1, Relaxed);
                c.barrier();
                // everyone incremented before anyone passes the barrier
                assert!(hits.load(Relaxed) >= before + c.n);
                c.barrier();
            });
            assert_eq!(hits.load(Relaxed), 4 * (round + 1));
            if round % 50 == 0 {
                thread::sleep(Duration::from_millis(2)); // let workers park
            }
        }
    }
}
