//! engram_gather (DSX_ENGRAM_PERSIST=1) on a test table: a memory-mapped file of 2 layers x 4096 rows x 256 f32, 24
//! rows per layer per token. Checks that every token's rows land where the engram stage reads them, over many tokens
//! (layer 0 is read while layer 1's rows are still being written), and that rows dropped from the page cache count as
//! not cached. The ignored test times publish -> layer 0 ready next to the per-token threads it replaces:
//!   cargo test --release --test engram_gather -- --ignored --nocapture
use dsx::engram_gather::{EngramGather, RowSource};
use std::time::{Duration, Instant};

const LAYERS: usize = 2;
const PER: usize = 24;
const DIM: usize = 256;
const ROWS: usize = 4096;

unsafe extern "C" {
    fn posix_fadvise(fd: i32, offset: i64, len: i64, advice: i32) -> i32;
}

fn value(l: usize, r: usize, j: usize) -> f32 {
    (l * 5_000_000 + r * 1000 + j) as f32
}

struct MapRows {
    map: memmap2::Mmap,
    file: std::fs::File,
}

impl MapRows {
    fn new(name: &str) -> MapRows {
        let path = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
        let mut bytes = Vec::with_capacity(LAYERS * ROWS * DIM * 4);
        for l in 0..LAYERS {
            for r in 0..ROWS {
                for j in 0..DIM {
                    bytes.extend_from_slice(&value(l, r, j).to_le_bytes());
                }
            }
        }
        std::fs::write(&path, &bytes).unwrap();
        let file = std::fs::File::open(&path).unwrap();
        file.sync_all().unwrap();
        let map = unsafe { memmap2::Mmap::map(&file).unwrap() };
        MapRows { map, file }
    }

    /// drop the file's pages from the page cache (they are clean after sync_all)
    fn evict(&self) {
        use std::os::fd::AsRawFd;
        unsafe { posix_fadvise(self.file.as_raw_fd(), 0, 0, 4) };
    }
}

fn hashes(hist: &[i64], l: usize) -> Vec<u64> {
    let t = *hist.last().unwrap() as u64;
    (0..PER as u64).map(|i| (t.wrapping_mul(0x9e37_79b9_7f4a_7c15).wrapping_add(i * 0x1234_5679 + l as u64 * 77) >> 17) % ROWS as u64).collect()
}

impl RowSource for MapRows {
    fn shape(&self) -> (usize, usize, usize) {
        (LAYERS, PER, DIM)
    }
    fn hashes(&self, hist: &[i64], l: usize) -> Vec<u64> {
        hashes(hist, l)
    }
    fn row_bytes(&self, l: usize, r: usize) -> &[u8] {
        let o = ((l * ROWS) + r) * DIM * 4;
        &self.map[o..o + DIM * 4]
    }
    fn row(&self, l: usize, r: usize, out: &mut [f32]) {
        for (j, o) in out.iter_mut().enumerate() {
            *o = f32::from_le_bytes(self.row_bytes(l, r)[j * 4..j * 4 + 4].try_into().unwrap());
        }
    }
}

/// the staging buffer the GPU copy reads, outside Rust's borrow of it (the workers write through a raw pointer)
struct Out(*mut f32);
impl Out {
    fn new() -> Out {
        Out(Box::leak(vec![f32::NAN; LAYERS * PER * DIM].into_boxed_slice()).as_mut_ptr())
    }
    fn slice(&self) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.0, LAYERS * PER * DIM) }
    }
    fn check(&self, hist: &[i64], l: usize) {
        let o = unsafe { std::slice::from_raw_parts(self.0.add(l * PER * DIM), PER * DIM) };
        for (i, &r) in hashes(hist, l).iter().enumerate() {
            for j in [0, 1, DIM / 2, DIM - 1] {
                assert_eq!(o[i * DIM + j], value(l, r as usize, j), "token {}, layer {l}, row {i}", hist.len());
            }
        }
    }
}

fn cpus() -> Vec<usize> {
    let n = std::thread::available_parallelism().map_or(4, |n| n.get());
    dsx::cpu::pool::AUX_CPUS.iter().copied().filter(|&c| c < n).collect()
}

#[test]
fn rows_land_in_place_every_token() {
    let src = MapRows::new("engram_gather_rows.bin");
    src.evict();
    let out = Out::new();
    let eg = EngramGather::with_source(Box::new(src), out.slice(), true, &cpus()).unwrap();
    let mut hist = vec![];
    for t in 0..3000i64 {
        hist.push(t * 7 + 3);
        eg.publish(&hist);
        eg.wait(0).unwrap();
        out.check(&hist, 0);
        eg.wait(1).unwrap();
        out.check(&hist, 1);
        if t == 0 {
            let (cold, _) = eg.stats(1);
            println!("first token after evicting the file: {cold} of {} rows not cached", LAYERS * PER);
            assert!(cold > 0, "evicted rows counted as cached");
        }
        if t % 500 == 499 {
            // workers parked: the next publish has to wake them
            std::thread::sleep(Duration::from_millis(30));
        }
    }
}

/// the per-token threads DSX_ENGRAM_PERSIST replaces: a helper thread that, per engram layer, spawns 8 pinned
/// scoped threads to gather that layer's rows; the step joins the helper before layer 1
fn old_gather(src: &MapRows, hist: &[i64], out: &Out) {
    let (src, out) = (src as *const MapRows as usize, out.0 as usize);
    let hist = hist.to_vec();
    std::thread::spawn(move || {
        let src = unsafe { &*(src as *const MapRows) };
        dsx::cpu::pool::pin_to_set(&cpus());
        for l in 0..LAYERS {
            let ids = hashes(&hist, l);
            std::thread::scope(|sc| {
                for g in 0..8 {
                    let ids = &ids;
                    sc.spawn(move || {
                        dsx::cpu::pool::pin_to_set(&cpus());
                        for i in (g..PER).step_by(8) {
                            let o = unsafe { std::slice::from_raw_parts_mut((out as *mut f32).add((l * PER + i) * DIM), DIM) };
                            src.row(l, ids[i] as usize, o);
                        }
                    });
                }
            });
        }
    })
    .join()
    .unwrap();
}

#[test]
#[ignore]
fn gather_latency() {
    let pct = |v: &mut Vec<f64>, p: f64| {
        v.sort_by(f64::total_cmp);
        v[((v.len() - 1) as f64 * p) as usize]
    };
    let src = MapRows::new("engram_gather_lat.bin");
    let out_old = Out::new();
    let out = Out::new();
    let src2 = MapRows::new("engram_gather_lat2.bin");
    let eg = EngramGather::with_source(Box::new(src2), out.slice(), false, &cpus()).unwrap();
    let (mut old, mut l0, mut l1) = (vec![], vec![], vec![]);
    let mut hist = vec![];
    // interleaved; 5 ms between tokens so the workers park as they do between decode steps (~27 ms)
    for t in 0..400i64 {
        hist.push(t * 13 + 1);
        let t0 = Instant::now();
        old_gather(&src, &hist, &out_old);
        old.push(t0.elapsed().as_secs_f64() * 1e6);
        out_old.check(&hist, 1);
        std::thread::sleep(Duration::from_millis(5));
        let t0 = Instant::now();
        eg.publish(&hist);
        eg.wait(0).unwrap();
        l0.push(t0.elapsed().as_secs_f64() * 1e6);
        eg.wait(1).unwrap();
        l1.push(t0.elapsed().as_secs_f64() * 1e6);
        out.check(&hist, 1);
        std::thread::sleep(Duration::from_millis(5));
    }
    println!("rows cached; us from the token's start, median / p90 / p99 over {} tokens:", old.len());
    println!("  per-token threads, both layers: {:.0} / {:.0} / {:.0}", pct(&mut old, 0.5), pct(&mut old, 0.9), pct(&mut old, 0.99));
    println!("  persistent, layer 1 ready:      {:.0} / {:.0} / {:.0}", pct(&mut l0, 0.5), pct(&mut l0, 0.9), pct(&mut l0, 0.99));
    println!("  persistent, both layers ready:  {:.0} / {:.0} / {:.0}", pct(&mut l1, 0.5), pct(&mut l1, 0.9), pct(&mut l1, 0.99));
}
