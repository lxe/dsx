#![cfg(feature = "cuda")]
//! T4, SM70 attention: whatever prefill-attention kernel `SparseAttnTc::launch` picks on THIS GPU (mma on arch >= 80,
//! the CUDA-core `sparse_attn` on Volta) against a plain fp32 CPU reference — Q/K/V at realistic magnitudes
//! (|x| ~ 10), softmax scale 1/sqrt(512), full gathered-row semantics (window ring + compressed cache, -1 skips,
//! duplicate indices, an all-masked token).
//! Accuracy: cargo test --release --features cuda --test gpu_attn_sm70 -- --nocapture
//! Perf:     cargo test --release --features cuda --test gpu_attn_sm70 -- --ignored --nocapture
use dsx::attn::SparseAttnTc;
use dsx::gpu::*;
use half::bf16;
use std::time::Instant;

const H: usize = 64;
const HD: usize = 512;
fn scale() -> f32 {
    1.0 / (HD as f32).sqrt()
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn unif(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
}

struct Case {
    t: usize,
    w: usize,
    k: usize,
    q: Vec<f32>,
    win: Vec<bf16>,
    comp: Vec<bf16>,
    win_idx: Vec<i32>,
    comp_idx: Vec<i32>,
    sinks: Vec<f32>,
}

/// K/V rows at |x| ~ 10 (bf16, as the caches store them)
fn rows(r: &mut Rng, n: usize) -> Vec<bf16> {
    (0..n * HD)
        .map(|_| bf16::from_f32(r.unif() * 10.0))
        .collect()
}

/// plain fp32 reference over the same gathered rows: window then compressed, -1 skipped, duplicates kept,
/// sink per head; output bf16-rounded like the kernels
fn reference(c: &Case) -> Vec<f32> {
    let mut out = vec![0f32; c.t * H * HD];
    for t in 0..c.t {
        let idx: Vec<(i32, bool)> = (0..c.w)
            .map(|i| (c.win_idx[t * c.w + i], false))
            .chain((0..c.k).map(|j| (c.comp_idx[t * c.k + j], true)))
            .filter(|(i, _)| *i >= 0)
            .collect();
        for h in 0..H {
            let q = &c.q[(t * H + h) * HD..][..HD];
            let mut s = vec![0f32; idx.len()];
            for (d, (i, from_comp)) in idx.iter().enumerate() {
                let row = if *from_comp {
                    &c.comp[*i as usize * HD..]
                } else {
                    &c.win[*i as usize * HD..]
                };
                let mut acc = 0f32;
                for j in 0..HD {
                    acc += q[j] * row[j].to_f32();
                }
                s[d] = acc * scale();
            }
            let mut m = c.sinks[h];
            for &x in &s {
                m = m.max(x);
            }
            let mut l = (c.sinks[h] - m).exp();
            let mut p = vec![0f32; idx.len()];
            for d in 0..idx.len() {
                p[d] = (s[d] - m).exp();
                l += p[d];
            }
            let o = &mut out[(t * H + h) * HD..][..HD];
            for d in 0..HD {
                let mut acc = 0f32;
                for (e, (i, from_comp)) in idx.iter().enumerate() {
                    let row = if *from_comp {
                        &c.comp[*i as usize * HD..]
                    } else {
                        &c.win[*i as usize * HD..]
                    };
                    acc += p[e] * row[d].to_f32();
                }
                o[d] = bf16::from_f32(acc / l).to_f32();
            }
        }
    }
    out
}

fn random_case(r: &mut Rng, t: usize, w: usize, k: usize, qmag: f32) -> Case {
    let (nw, nc) = (300, 2000);
    let pick = |r: &mut Rng, n: usize| {
        if r.below(10) == 0 {
            -1
        } else {
            r.below(n) as i32
        }
    };
    let mut win_idx: Vec<i32> = (0..t * w).map(|_| pick(r, nw)).collect();
    let mut comp_idx: Vec<i32> = (0..t * k).map(|_| pick(r, nc)).collect();
    for x in 0..t {
        // duplicates (the compaction keeps repeated row ids)
        if w > 4 {
            win_idx[x * w + 1] = win_idx[x * w];
            win_idx[x * w + 3] = 7;
        }
        if k > 4 {
            comp_idx[x * k + 2] = comp_idx[x * k];
            comp_idx[x * k + k - 1] = 7;
        }
    }
    // token 1: no valid rows at all (sink only -> zeros); token 2: a partial tail
    if t > 1 {
        win_idx[w..2 * w].fill(-1);
        comp_idx[k..2 * k].fill(-1);
    }
    if t > 2 {
        for i in 5..w {
            win_idx[2 * w + i] = -1;
        }
        comp_idx[2 * k..3 * k].fill(-1);
    }
    Case {
        t,
        w,
        k,
        q: (0..t * H * HD).map(|_| r.unif() * qmag).collect(),
        win: rows(r, nw),
        comp: if k > 0 { rows(r, nc) } else { vec![] },
        win_idx,
        comp_idx,
        sinks: (0..H).map(|_| r.unif() * 3.0).collect(),
    }
}

/// (min cosine per 512-vector, max relative error over |ref| > 0.1, max abs error)
fn metrics(got: &[f32], want: &[f32]) -> (f32, f32, f32) {
    let (mut mincos, mut maxrel, mut maxabs) = (2.0f32, 0f32, 0f32);
    for (a, b) in got.chunks(HD).zip(want.chunks(HD)) {
        let (mut dot, mut na, mut nb) = (0f32, 0f32, 0f32);
        for (x, y) in a.iter().zip(b) {
            dot += x * y;
            na += x * x;
            nb += y * y;
            let d = (x - y).abs();
            maxabs = maxabs.max(d);
            if y.abs() > 0.1 {
                maxrel = maxrel.max(d / y.abs());
            }
        }
        mincos = mincos.min(dot / (na.sqrt() * nb.sqrt()));
    }
    (mincos, maxrel, maxabs)
}

/// the kernel `SparseAttnTc` picks here, run through the same launch helper prefill uses
fn wired(
    g: &Gpu,
    tc: &SparseAttnTc,
    c: &Case,
    o: u64,
    q: u64,
    win: u64,
    wi: u64,
    comp: u64,
    ci: u64,
    sk: u64,
) -> Vec<f32> {
    tc.launch(g, q, win, wi, c.w, comp, ci, c.k, sk, o, c.t, H, HD)
        .unwrap();
    g.download::<f32>(o, c.t * H * HD).unwrap()
}

#[test]
fn sm70_attn_accuracy() {
    let g = Gpu::new(0).unwrap();
    let tc = SparseAttnTc::new(&g).unwrap();
    eprintln!(
        "{} ({} SMs, arch {}): SparseAttnTc path = {}",
        g.name,
        g.sms,
        g.arch,
        if tc_sm70_serial(&g) {
            "sparse_attn (CUDA cores)"
        } else {
            "sparse_attn_tc (mma)"
        }
    );
    let mut r = Rng(0x9e3779b97f4a7c15);
    // qmag 10: peaky softmax (score std ~ 30); qmag 1: nearly uniform over the gathered rows
    for (t, w, k, qmag) in [
        (5usize, 128usize, 512usize, 10.0f32),
        (17, 128, 512, 10.0),
        (64, 100, 37, 10.0),
        (9, 128, 0, 10.0),
        (33, 128, 512, 1.0),
    ] {
        let c = random_case(&mut r, t, w, k, qmag);
        let refo = reference(&c);
        let q = g.upload(&c.q).unwrap();
        let win = g.upload(&c.win).unwrap();
        let comp = if c.k > 0 {
            Some(g.upload(&c.comp).unwrap())
        } else {
            None
        };
        let cp = comp.as_ref().map_or(0, |b| b.ptr);
        let wi = g.upload(&c.win_idx).unwrap();
        let ci = g.upload(&c.comp_idx).unwrap();
        let sk = g.upload(&c.sinks).unwrap();
        let o = g.alloc(c.t * H * HD * 4).unwrap();
        g.memset(o.ptr, c.t * H * HD * 4).unwrap();
        let got = wired(
            &g, &tc, &c, o.ptr, q.ptr, win.ptr, wi.ptr, cp, ci.ptr, sk.ptr,
        );
        let (mincos, maxrel, maxabs) = metrics(&got, &refo);
        eprintln!(
            "T={t} W={w} K={k} |q|,|kv|<=10 x{qmag}: cos(min) {mincos:.6}  max rel {maxrel:.2e}  max abs {maxabs:.2e}"
        );
        assert!(got.iter().all(|v| v.is_finite()), "non-finite output");
        // all-masked token -> zeros (sink only)
        if t > 1 {
            assert!(
                got[H * HD..2 * H * HD].iter().all(|&v| v == 0.0),
                "all-masked token must be zeros"
            );
        }
        assert!(mincos >= 0.9999, "cosine below gate: {mincos}");
        assert!(maxrel <= 2e-2, "max rel above gate: {maxrel}");
    }
}

fn tc_sm70_serial(g: &Gpu) -> bool {
    g.arch < 80
}

fn time(f: &mut dyn FnMut(), g: &Gpu, n: usize) -> f64 {
    f();
    g.sync().unwrap();
    let t = Instant::now();
    for _ in 0..n {
        f();
    }
    g.sync().unwrap();
    t.elapsed().as_secs_f64() / n as f64 * 1e3
}

#[test]
#[ignore]
fn sm70_attn_perf() {
    let g = Gpu::new(0).unwrap();
    let tc = SparseAttnTc::new(&g).unwrap();
    let mut r = Rng(12345);
    let (nc, w, k) = (16384usize, 128usize, 512usize);
    let serial = g.kernel("sparse_attn").unwrap();
    eprintln!(
        "{} ({} SMs, arch {}): attention path = {}",
        g.name,
        g.sms,
        g.arch,
        if tc_sm70_serial(&g) {
            "sparse_attn (CUDA cores)"
        } else {
            "sparse_attn_tc (mma)"
        }
    );
    for t in [16usize, 64, 512, 4096] {
        // window = the token's previous 128 positions in a [128 + T] ring; comp = 512 distinct of 16K per token
        let win_idx: Vec<i32> = (0..t)
            .flat_map(|x| (0..w).map(move |i| (x + 1 + i) as i32))
            .collect();
        let mut perm: Vec<i32> = (0..nc as i32).collect();
        let mut comp_idx = vec![0i32; t * k];
        for x in 0..t {
            for j in 0..k {
                let s = j + r.below(nc - j);
                perm.swap(j, s);
            }
            comp_idx[x * k..][..k].copy_from_slice(&perm[..k]);
        }
        let q = g
            .upload(
                &(0..t * H * HD)
                    .map(|_| r.unif() * 10.0)
                    .collect::<Vec<f32>>(),
            )
            .unwrap();
        let win = g.upload(&rows(&mut r, w + t)).unwrap();
        let comp = g.upload(&rows(&mut r, nc)).unwrap();
        let wi = g.upload(&win_idx).unwrap();
        let ci = g.upload(&comp_idx).unwrap();
        let sk = g
            .upload(&(0..H).map(|_| r.unif() * 3.0).collect::<Vec<f32>>())
            .unwrap();
        let o = g.alloc(t * H * HD * 4).unwrap();
        let n = if t >= 4096 { 5 } else { 20 };
        let args = [
            Arg::P(q.ptr),
            Arg::P(win.ptr),
            Arg::P(wi.ptr),
            Arg::I(w as i32),
            Arg::P(comp.ptr),
            Arg::P(ci.ptr),
            Arg::I(k as i32),
            Arg::P(sk.ptr),
            Arg::P(o.ptr),
            Arg::I(H as i32),
            Arg::I(HD as i32),
            Arg::F(scale()),
        ];
        let t_raw = time(
            &mut || {
                g.launch(serial, (t as u32, (H / 8) as u32, 1), 256, 0, &args)
                    .unwrap()
            },
            &g,
            n,
        );
        let t_wired = time(
            &mut || {
                tc.launch(
                    &g, q.ptr, win.ptr, wi.ptr, w, comp.ptr, ci.ptr, k, sk.ptr, o.ptr, t, H, HD,
                )
                .unwrap()
            },
            &g,
            n,
        );
        let flop = 2.0 * 2.0 * (t * H * (w + k) * HD) as f64;
        eprintln!(
            "T={t:5} W={w} K={k}: sparse_attn {t_raw:8.2} ms | wired {t_wired:8.2} ms  {:6.2} TFLOP/s  {:7.1} us/token  gate(<= 10 ms/token): {}",
            flop / t_wired / 1e9,
            t_wired / t as f64 * 1e3,
            if t_wired / t as f64 <= 10.0 {
                "PASS"
            } else {
                "FAIL"
            }
        );
    }
}
