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
