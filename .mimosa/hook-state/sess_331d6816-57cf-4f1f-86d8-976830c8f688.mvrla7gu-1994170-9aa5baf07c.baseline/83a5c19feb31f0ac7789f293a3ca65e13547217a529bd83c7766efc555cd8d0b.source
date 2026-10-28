//! CPU expert kernel benchmark: weight GB/s per format and thread count over 64 experts (far beyond L3),
//! ms per layer for 6 mixed-type experts, and the pool's fixed per-call overhead.
//! cargo run --release --bin cpubench [-- quick] [fmt=Q2_0]

use dsx::cpu::moe::{CpuMoe, ExpertW};
use dsx::cpu::pool::{Ctx, Pool};
use dsx::cpu::qdot::{QAct, dot_rows};
use dsx::gguf::GType;
use half::f16;
use std::time::{Duration, Instant};

const DIM: usize = 5120;
const INTER: usize = 2304;

struct Rng(u64);
impl Rng {
    fn u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn normal(&mut self) -> f32 {
        let a = ((self.u64() >> 40) as f32 / (1u64 << 24) as f32).max(1e-7);
        let b = (self.u64() >> 40) as f32 / (1u64 << 24) as f32;
        (-2.0 * a.ln()).sqrt() * (std::f32::consts::TAU * b).cos()
    }
}

fn rand_rows(ty: GType, nr: usize, cols: usize, rng: &mut Rng) -> Vec<u8> {
    let n = ty.row_bytes(cols) * nr;
    let mut v = vec![0u8; n.div_ceil(8) * 8];
    for c in v.chunks_exact_mut(8) {
        c.copy_from_slice(&rng.u64().to_le_bytes());
    }
    v.truncate(n);
    let (_, bs) = ty.block();
    let (at, d) = match ty {
        GType::Q2_0 | GType::Q8_0 | GType::Iq2Xxs | GType::Iq2Xs | GType::Iq2S | GType::Iq1S => (0, 0.02),
        GType::Ptq1_0 => (26, 0.02),
        _ => (108, 0.001),
    };
    for blk in v.chunks_exact_mut(bs) {
        blk[at..at + 2].copy_from_slice(&f16::from_f32(d).to_le_bytes());
    }
    v
}

struct Ex {
    gu: GType,
    dn: GType,
    gate: Vec<u8>,
    up: Vec<u8>,
    down: Vec<u8>,
}

fn make(types: &[(GType, GType)], n: usize, rng: &mut Rng) -> Vec<Ex> {
    (0..n)
        .map(|i| {
            let (gu, dn) = types[i % types.len()];
            Ex {
                gu,
                dn,
                gate: rand_rows(gu, INTER, DIM, rng),
                up: rand_rows(gu, INTER, DIM, rng),
                down: rand_rows(dn, DIM, INTER, rng),
            }
        })
        .collect()
}

fn views(exs: &[Ex]) -> Vec<ExpertW<'_>> {
    exs.iter()
        .map(|e| ExpertW {
            gu: e.gu,
            gate: &e.gate,
            up: &e.up,
            dn: e.dn,
            down: &e.down,
        })
        .collect()
}

/// Layer calls of 6 experts each, cycling through all of them `passes` times: (GB/s of weights, ms per call).
fn layer_rate(moe: &mut CpuMoe, ew: &[ExpertW], t: usize, passes: usize, rng: &mut Rng) -> (f64, f64) {
    let x: Vec<f32> = (0..t * DIM).map(|_| rng.normal()).collect();
    let w = vec![1.0f32 / 6.0; 6 * t];
    let mut out = vec![0f32; t * DIM];
    let groups = ew.len() / 6;
    let mut call = |g: usize| moe.forward(&ew[6 * g..6 * g + 6], DIM, &x, t, &w, &mut out);
    for g in 0..groups {
        call(g); // warm-up pass (page faults, pool spin-up)
    }
    let t0 = Instant::now();
    let mut bytes = 0usize;
    for _ in 0..passes {
        for g in 0..groups {
            call(g);
            bytes += ew[6 * g..6 * g + 6].iter().map(|e| e.bytes()).sum::<usize>();
        }
    }
    let dt = t0.elapsed().as_secs_f64();
    (bytes as f64 / dt / 1e9, dt / (passes * groups) as f64 * 1e3)
}

fn cpus(n: usize) -> Vec<usize> {
    match n {
        1 => vec![0],
        8 => (0..8).map(|c| 2 * c).collect(), // one thread per P-core
        14 => (0..14).collect(),
        _ => (0..n).collect(),
    }
}

fn t16_ex(gu: GType, dn: GType, rng: &mut Rng) -> Ex {
    let mut mat = |ty: GType, k: usize, n: usize| -> Vec<u8> {
        let k2 = dsx::t16::k2(ty).unwrap();
        let tb = dsx::t16::trellis_bytes(k2, k, n);
        let mut v: Vec<u8> = (0..dsx::t16::matrix_bytes(ty, k, n)).map(|_| rng.u64() as u8).collect();
        for c in v[tb..].chunks_exact_mut(2) {
            c.copy_from_slice(&f16::from_f32(0.02).to_le_bytes());
        }
        v
    };
    Ex { gu, dn, gate: mat(gu, DIM, INTER), up: mat(gu, DIM, INTER), down: mat(dn, INTER, DIM) }
}

/// `t16`: ms per 6-expert call, T16 layouts against a mix of UD-IQ1_M's codebook types, interleaved
fn t16_bench(nexp: usize, rng: &mut Rng) {
    use GType::*;
    dsx::cpu::pool::pin_to(16);
    let sets: Vec<(&str, Vec<Ex>)> = vec![
        ("UD IQ mix", make(&[(Iq1S, Iq2Xxs), (Iq2Xxs, Iq2Xs), (Iq1S, Iq2S), (Iq2Xxs, Iq2Xxs), (Iq2Xs, Iq2S), (Iq1S, Iq2Xs)], nexp, rng)),
        ("T16 1.5/2", (0..nexp).map(|_| t16_ex(T16K15, T16K2, rng)).collect()),
        ("T16 2/2.5", (0..nexp).map(|_| t16_ex(T16K2, T16K25, rng)).collect()),
        ("K1.5", (0..nexp).map(|_| t16_ex(T16K15, T16K15, rng)).collect()),
        ("K2", (0..nexp).map(|_| t16_ex(T16K2, T16K2, rng)).collect()),
        ("K2.5", (0..nexp).map(|_| t16_ex(T16K25, T16K25, rng)).collect()),
    ];
    // DSX_T16_CODEBOOK=mul1 runs the T16 sets with EXL3's mul1 codebook (same streams); CPUB_THREADS="1,14"
    println!("codebook: {}", if dsx::cpu::t16k::mul1() { "mul1" } else { "T16" });
    let ths: Vec<usize> = std::env::var("CPUB_THREADS").unwrap_or("8,14,16".into()).split(',').filter_map(|v| v.trim().parse().ok()).collect();
    for th in ths {
        let mut moe = CpuMoe::new(Pool::new(&cpus(th), Duration::from_millis(5)));
        let mut line = format!("{th:2} threads:");
        for round in 0..2 {
            for (name, exs) in &sets {
                let ew = views(exs);
                let (g, ms) = layer_rate(&mut moe, &ew, 1, 3, rng);
                if round == 1 {
                    line += &format!("  {name} {ms:.3} ms ({g:.1} GB/s)");
                }
            }
        }
        println!("{line}");
    }
    // prefill windows: calls of 6 experts with nt tokens routed to each (CPUB_NT, default 1,2,4,8,16); per call and
    // per token, and weight-token products per clock and thread at the measured clock (CPUB_GHZ, default 4.9)
    let nts: Vec<usize> = std::env::var("CPUB_NT").unwrap_or("1,2,4,8,16".into()).split(',').filter_map(|v| v.trim().parse().ok()).collect();
    let ghz: f64 = std::env::var("CPUB_GHZ").ok().and_then(|v| v.parse().ok()).unwrap_or(4.9);
    let ths: Vec<usize> = std::env::var("CPUB_THREADS").unwrap_or("8,14,16".into()).split(',').filter_map(|v| v.trim().parse().ok()).collect();
    println!("nt sweep ({}): ms per 6-expert call / per token, weight-token products per clock per thread at {ghz} GHz",
        if dsx::cpu::t16k::multi() { "gemv_multi" } else { "per-token gemv, DSX_T16_MULTI=0" });
    for th in ths {
        let mut moe = CpuMoe::new(Pool::new(&cpus(th), Duration::from_millis(5)));
        for (name, exs) in sets.iter().skip(1).take(2) {
            let ew = views(exs);
            let mut line = format!("{th:2} threads {name}:");
            for &nt in &nts {
                let (_, ms) = layer_rate(&mut moe, &ew, nt, if th == 1 { 1 } else { 3 }, rng);
                let wpc = (6 * 3 * DIM * INTER * nt) as f64 / (ms * 1e-3 * ghz * 1e9 * th as f64);
                line += &format!("  nt={nt} {ms:.2} ({:.3}/tok, {wpc:.1} w/clk)", ms / nt as f64);
            }
            println!("{line}");
        }
    }
}

/// t16k: the T16 kernel alone, 1 thread on P-core 0, a 5120-input matrix's first 16 tile columns (L2-resident):
/// ns per call and clocks per 16x16 tile per token for nt = CPUB_NT tokens (gemv_multi; DSX_T16_MULTI=0 per token)
fn t16k_bench(rng: &mut Rng) {
    use dsx::cpu::t16k::{Act, gemv_multi, prep, scales};
    dsx::cpu::pool::pin_to(0);
    let nts: Vec<usize> = std::env::var("CPUB_NT").unwrap_or("1,2,3,4,8,16".into()).split(',').filter_map(|v| v.trim().parse().ok()).collect();
    let ghz: f64 = std::env::var("CPUB_GHZ").ok().and_then(|v| v.parse().ok()).unwrap_or(4.9);
    let reps: usize = std::env::var("CPUB_REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(400);
    for ty in [GType::T16K15, GType::T16K2, GType::T16K25] {
        let k2 = dsx::t16::k2(ty).unwrap();
        let e = t16_ex(ty, ty, rng);
        let m = &e.gate;
        let suh = scales(m, k2, DIM, INTER).0;
        let acts: Vec<Act> = (0..16).map(|_| {
            let x: Vec<f32> = (0..DIM).map(|_| rng.normal()).collect();
            let mut a = Act::default();
            prep(&x, suh, DIM, &mut a);
            a
        }).collect();
        let cols = 16;
        let tiles = cols * DIM / 16;
        let mut line = format!("{ty:?}:");
        for &nt in &nts {
            let ar: Vec<&Act> = acts[..nt].iter().collect();
            let mut ys = vec![vec![0f32; 16 * cols]; nt];
            let mut run = |n: usize| for _ in 0..n {
                let mut o: Vec<&mut [f32]> = ys.iter_mut().map(|v| &mut v[..]).collect();
                gemv_multi(m, k2, DIM, INTER, &ar, 0, cols, &mut o);
            };
            run(reps / 10);
            let t0 = Instant::now();
            run(reps);
            let ns = t0.elapsed().as_secs_f64() * 1e9 / reps as f64;
            line += &format!("  nt={nt} {:.0} us ({:.1} clk/tile/tok)", ns / 1e3, ns * ghz / (tiles * nt) as f64);
        }
        println!("{line}");
    }
}

/// small: decode-sized T16 calls as the CPU expert worker makes them (pool on CPUs 0-13, 50 ms spin, caller on CPU 16):
/// ne = CPUB_NE (default 1,2,3,4,6) experts per call, t = 1, distinct experts per call (weights from RAM). ms per call,
/// GB/s, and with DSX_T16_PROF=1 the mean phase breakdown in us from pool.run: workers' start (first / last), each
/// phase's work (mean and slowest worker), the barrier release, and pool.run's return
fn small_bench(nexp: usize, rng: &mut Rng) {
    use dsx::cpu::moe::t16prof::{CALL, MARKS, T, on};
    use std::sync::atomic::Ordering::Relaxed;
    use GType::*;
    dsx::cpu::pool::pin_to(dsx::cpu::pool::WORKER_CPU);
    let nes: Vec<usize> = std::env::var("CPUB_NE").unwrap_or("1,2,3,4,6".into()).split(',').filter_map(|v| v.trim().parse().ok()).collect();
    let calls: usize = std::env::var("CPUB_CALLS").ok().and_then(|v| v.parse().ok()).unwrap_or(600);
    let th: usize = std::env::var("CPUB_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(14);
    let mut moe = CpuMoe::new(Pool::new(&dsx::cpu::pool::P_CORES[..th], Duration::from_millis(50)));
    for (name, gu, dn) in [("T16 1.5/2", T16K15, T16K2), ("T16 2/2.5", T16K2, T16K25)] {
        let exs: Vec<Ex> = (0..nexp).map(|_| t16_ex(gu, dn, rng)).collect();
        let ew = views(&exs);
        let x: Vec<f32> = (0..DIM).map(|_| rng.normal()).collect();
        let mut out = vec![0f32; DIM];
        println!("{name}, {th} threads, t = 1, {:.1} MB per expert", ew[0].bytes() as f64 / 1e6);
        for &ne in &nes {
            let w = vec![1.0f32 / ne as f32; ne];
            let groups = nexp / ne;
            let (mut tot, mut bytes) = (0f64, 0usize);
            let mut acc = [0f64; 10];
            for c in 0..calls + groups {
                let g = c % groups;
                let sl = &ew[ne * g..ne * g + ne];
                let t0 = Instant::now();
                moe.forward(sl, DIM, &x, 1, &w, &mut out);
                let dt = t0.elapsed().as_secs_f64();
                if c < groups {
                    continue; // warm-up pass
                }
                tot += dt;
                bytes += sl.iter().map(|e| e.bytes()).sum::<usize>();
                if on() {
                    let r = CALL[1].load(Relaxed) as f64;
                    let m = |i: usize, k: usize| T[i][k].load(Relaxed) as f64 - r;
                    let col = |k: usize| (0..th).map(|i| m(i, k)).collect::<Vec<f64>>();
                    let mx = |v: &[f64]| v.iter().cloned().fold(f64::MIN, f64::max);
                    let mn = |v: &[f64]| v.iter().cloned().fold(f64::MAX, f64::min);
                    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
                    let st = col(0);
                    let w1: Vec<f64> = (0..th).map(|i| m(i, 1) - m(i, 0)).collect();
                    let w2: Vec<f64> = (0..th).map(|i| m(i, 3) - m(i, 2)).collect();
                    let w3: Vec<f64> = (0..th).map(|i| m(i, 5) - m(i, 4)).collect();
                    let v = [CALL[1].load(Relaxed) as f64 - CALL[0].load(Relaxed) as f64, mn(&st), mx(&st), mean(&w1), mx(&col(2)),
                        mean(&w2), mx(&w2), mx(&col(4)), mean(&w3), CALL[2].load(Relaxed) as f64 - r];
                    let _ = MARKS;
                    for (a, b) in acc.iter_mut().zip(v) {
                        *a += b;
                    }
                }
            }
            let ms = tot / calls as f64 * 1e3;
            println!("  ne={ne}: {ms:.3} ms per call, {:.1} GB/s", bytes as f64 / tot / 1e9);
            if on() {
                let a: Vec<f64> = acc.iter().map(|v| v / calls as f64 / 1e3).collect();
                println!("    us: setup {:.1}; start {:.1}-{:.1}; prep {:.1} -> barrier {:.1}; gate/up mean {:.1} max {:.1} -> barrier {:.1}; \
                    down mean {:.1}; return {:.1}", a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9]);
            }
        }
    }
    // the ceiling: a pure read of 1 GB (u64 sums, 64 KB per claim) on the same threads
    drop(moe);
    let pool = Pool::new(&dsx::cpu::pool::P_CORES[..th], Duration::from_millis(50));
    let buf: Vec<u64> = (0..(1usize << 27)).map(|i| i as u64).collect();
    let next = std::sync::atomic::AtomicUsize::new(0);
    let sink = std::sync::atomic::AtomicU64::new(0);
    let chunk = 8192;
    let mut best = 0f64;
    for _ in 0..5 {
        next.store(0, Relaxed);
        let t0 = Instant::now();
        pool.run(&|_c: &dsx::cpu::pool::Ctx| {
            let mut acc = 0u64;
            loop {
                let i = next.fetch_add(1, Relaxed) * chunk;
                if i >= buf.len() {
                    break;
                }
                for v in &buf[i..i + chunk] {
                    acc = acc.wrapping_add(*v);
                }
            }
            sink.fetch_add(acc, Relaxed);
        });
        best = best.max((buf.len() * 8) as f64 / t0.elapsed().as_secs_f64() / 1e9);
    }
    println!("RAM read, {th} threads: {best:.1} GB/s (best of 5, 1 GB)");
}

fn main() {
    if std::env::args().any(|a| a == "small") {
        small_bench(if std::env::args().any(|a| a == "quick") { 24 } else { 64 }, &mut Rng(0x2545f4914f6cdd1d));
        return;
    }
    if std::env::args().any(|a| a == "t16k") {
        t16k_bench(&mut Rng(0x2545f4914f6cdd1d));
        return;
    }
    let quick = std::env::args().any(|a| a == "quick");
    if std::env::args().any(|a| a == "t16") {
        t16_bench(if quick { 12 } else { 64 }, &mut Rng(0x2545f4914f6cdd1d));
        return;
    }
    let nexp = if quick { 12 } else { 64 };
    let mut rng = Rng(0x2545f4914f6cdd1d);
    use GType::*;
    // ud=MODEL.gguf: only the mixed layer, with the (gate/up, down) types of that model's 40 layers, e.g. Unsloth UD-IQ1_M
    if let Some(path) = std::env::args().find_map(|a| a.strip_prefix("ud=").map(str::to_string)) {
        let g = dsx::gguf::Gguf::open(std::path::Path::new(&path)).expect("ud= model");
        let nl = g.u64("deepseek41.num_hidden_layers").or_else(|_| g.u64("deepseek4.block_count")).unwrap_or(40) as usize;
        let pairs: Vec<(GType, GType)> = (0..nl)
            .filter_map(|l| Some((g.tensors.get(&format!("blk.{l}.ffn_gate_exps.weight"))?.ty, g.tensors.get(&format!("blk.{l}.ffn_down_exps.weight"))?.ty)))
            .collect();
        let exs = make(&pairs, 64, &mut rng);
        let ew = views(&exs);
        let per6: f64 = ew.iter().map(|e| e.bytes()).sum::<usize>() as f64 / ew.len() as f64 * 6.0 / 1e6;
        println!("ud mix of {} layers: {per6:.1} MB of weights per 6-expert call, pool of 64", pairs.len());
        for th in [14usize, 20] {
            let mut moe = CpuMoe::new(Pool::new(&cpus(th), Duration::from_millis(5)));
            for _ in 0..2 {
                let (gb, ms) = layer_rate(&mut moe, &ew, 1, 5, &mut rng);
                println!("  {th:2} threads T=1: {ms:.3} ms per 6-expert call ({gb:.1} GB/s)");
            }
        }
        return;
    }

    // in-cache single-thread kernel rate on a P-core: the compute ceiling per thread
    dsx::cpu::pool::pin_to(0);
    println!("in-cache kernel rate, 1 thread on a P-core (64 rows x 5120, L2-resident)");
    for ty in [Q2_0, Ptq1_0, Q3K, Q8_0, Iq2Xxs, Iq2Xs, Iq2S, Iq1S] {
        let rows = rand_rows(ty, 64, DIM, &mut rng);
        let mut line = format!("  {ty:?}:");
        for t in [1usize, 6] {
            let x: Vec<f32> = (0..t * DIM).map(|_| rng.normal()).collect();
            let act = QAct::new(&x, t, DIM, &[ty]);
            let mut out = vec![0f32; t * 64];
            let reps = 2000;
            let t0 = Instant::now();
            for _ in 0..reps {
                dot_rows(ty, &rows, DIM, &act, &mut out);
            }
            let gbs = (rows.len() * reps) as f64 / t0.elapsed().as_secs_f64() / 1e9;
            line += &format!("  T={t} {gbs:5.1} GB/s");
        }
        println!("{line}");
    }

    // the caller spins while workers run: keep it off the pool's CPUs (16 = first E-core)
    dsx::cpu::pool::pin_to(16);
    println!(
        "\nweights streamed from RAM, {nexp} experts per format, calls of 6 experts (all T tokens routed to each)"
    );
    println!(
        "{:8} {:>7} {:>16} {:>16}",
        "format", "threads", "T=1 GB/s (ms)", "T=6 GB/s (ms)"
    );
    let only = std::env::args().find_map(|a| a.strip_prefix("fmt=").map(str::to_string));
    for ty in [Q2_0, Ptq1_0, Q3K, Q8_0, Iq2Xxs, Iq2Xs, Iq2S, Iq1S] {
        if only.as_ref().is_some_and(|o| *o != format!("{ty:?}")) {
            continue;
        }
        let n = if ty == Q8_0 { nexp.min(24) } else { nexp };
        let exs = make(&[(ty, ty)], n, &mut rng);
        let ew = views(&exs);
        for th in [1usize, 8, 16] {
            let mut moe = CpuMoe::new(Pool::new(&cpus(th), Duration::from_millis(5)));
            let passes = if th == 1 { 1 } else { 3 };
            let (a, am) = layer_rate(&mut moe, &ew, 1, passes, &mut rng);
            let (b, bm) = layer_rate(&mut moe, &ew, 6, passes, &mut rng);
            println!(
                "{:8} {:>7} {:>8.1} ({:5.2}) {:>8.1} ({:5.2})",
                format!("{ty:?}"),
                th,
                a,
                am,
                b,
                bm
            );
        }
    }

    println!("\nmixed layer: 6 experts per call over {nexp}, gate+up / down types cycling");
    let mix = [
        (Q2_0, Q3K),
        (Ptq1_0, Q2_0),
        (Q2_0, Q2_0),
        (Q3K, Q3K),
        (Ptq1_0, Ptq1_0),
        (Q2_0, Ptq1_0),
    ];
    // the same mix with IQ2_XXS in place of Q2_0 (the codebook 2-bit tier)
    let mix_iq = mix.map(|(a, b)| (if a == Q2_0 { Iq2Xxs } else { a }, if b == Q2_0 { Iq2Xxs } else { b }));
    for (name, m) in [("Q2_0", mix), ("IQ2_XXS", mix_iq)] {
        let exs = make(&m, nexp, &mut rng);
        let ew = views(&exs);
        let per6: f64 = ew.iter().map(|e| e.bytes()).sum::<usize>() as f64 / ew.len() as f64 * 6.0 / 1e6;
        println!("  2-bit tier {name}: {per6:.1} MB of weights per layer call");
        for th in [8usize, 16] {
            let mut moe = CpuMoe::new(Pool::new(&cpus(th), Duration::from_millis(5)));
            for t in [1usize, 6] {
                let (g, ms) = layer_rate(&mut moe, &ew, t, 3, &mut rng);
                println!("  {th:2} threads T={t}: {ms:.3} ms per layer ({g:.1} GB/s)");
            }
        }
    }
    let exs = make(&mix, nexp, &mut rng);
    let ew = views(&exs);

    // the same 2 experts (~18 MB, under the 25 MB L3) as all 6 of a call: weights come from cache after the first call,
    // the ceiling for a prefetch-into-L3 scheme
    println!("\nL3-resident: 6 experts per call drawn from the same 2 (~{:.0} MB), engine's 14 threads", (ew[0].bytes() + ew[1].bytes()) as f64 / 1e6);
    let hot: Vec<ExpertW> = (0..6).map(|i| ew[i % 2].clone()).collect();
    {
        let mut moe = CpuMoe::new(Pool::new(&(0..14).collect::<Vec<_>>(), Duration::from_millis(5)));
        let (g, ms) = layer_rate(&mut moe, &hot, 1, 200, &mut rng);
        println!("  T=1: {ms:.3} ms per call ({g:.1} GB/s of weights touched)");
        let (g, ms) = layer_rate(&mut moe, &ew, 1, 3, &mut rng);
        println!("  from RAM, same threads: {ms:.3} ms per call ({g:.1} GB/s)");
    }

    println!("\nfixed overhead per call, 16 threads");
    let pool = Pool::new(&cpus(16), Duration::from_millis(2));
    let empty = |c: &Ctx| c.barrier();
    let time = |f: &mut dyn FnMut(), n: usize, gap: Duration| -> f64 {
        let mut tot = 0.0;
        for _ in 0..n {
            if gap > Duration::from_millis(1) {
                std::thread::sleep(gap);
            } else {
                let t = Instant::now();
                while t.elapsed() < gap {} // caller busy elsewhere (e.g. waiting on the GPU)
            }
            let t0 = Instant::now();
            f();
            tot += t0.elapsed().as_secs_f64();
        }
        tot / n as f64 * 1e6
    };
    for _ in 0..1000 {
        pool.run(&empty);
    }
    println!(
        "  pool.run + 1 barrier, hot:          {:6.2} us",
        time(&mut || pool.run(&empty), 20000, Duration::ZERO)
    );
    println!(
        "  pool.run + 1 barrier, 0.5 ms busy gap:{:6.2} us",
        time(&mut || pool.run(&empty), 2000, Duration::from_micros(500))
    );
    println!(
        "  pool.run + 1 barrier, parked (5 ms sleep):{:6.2} us",
        time(&mut || pool.run(&empty), 200, Duration::from_millis(5))
    );
    let mut moe = CpuMoe::new(pool);
    let few = &ew[..6];
    for t in [1usize, 6] {
        let x: Vec<f32> = (0..t * DIM).map(|_| rng.normal()).collect();
        let w0 = vec![0f32; 6 * t]; // nothing routed: quantize x, dispatch, barrier, return
        let mut out = vec![0f32; t * DIM];
        let us = time(
            &mut || moe.forward(few, DIM, &x, t, &w0, &mut out),
            5000,
            Duration::ZERO,
        );
        println!("  CpuMoe::forward with no routed rows, T={t}: {us:6.2} us (incl. quantizing x)");
    }
}
