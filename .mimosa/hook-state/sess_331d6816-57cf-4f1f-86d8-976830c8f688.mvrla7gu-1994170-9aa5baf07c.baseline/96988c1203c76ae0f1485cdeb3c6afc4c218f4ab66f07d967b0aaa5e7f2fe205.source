#![cfg(feature = "cuda")]
//! sparse_attn_tc (prefill, tensor cores) vs sparse_attn (f32 CUDA cores) vs an f64 CPU reference.
//! Benchmark: cargo test --release --features cuda --test gpu_attn -- --ignored --nocapture
use dsx::attn::SparseAttnTc;
use dsx::gpu::*;
use half::bf16;
use std::time::Instant;

const H: usize = 64;
const HD: usize = 512;

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

fn rows(r: &mut Rng, n: usize) -> Vec<bf16> {
    (0..n * HD).map(|_| bf16::from_f32(r.unif() * 2.0)).collect()
}

/// f64 reference: softmax over valid rows + sink, output bf16-rounded
fn reference(c: &Case) -> Vec<f32> {
    let scale = 1.0 / (HD as f64).sqrt();
    let mut out = vec![0f32; c.t * H * HD];
    for t in 0..c.t {
        let rs: Vec<&[bf16]> = (0..c.w)
            .filter_map(|i| (c.win_idx[t * c.w + i] >= 0).then(|| &c.win[c.win_idx[t * c.w + i] as usize * HD..][..HD]))
            .chain((0..c.k).filter_map(|j| (c.comp_idx[t * c.k + j] >= 0).then(|| &c.comp[c.comp_idx[t * c.k + j] as usize * HD..][..HD])))
            .collect();
        for h in 0..H {
            let q = &c.q[(t * H + h) * HD..][..HD];
            let s: Vec<f64> = rs.iter().map(|r| r.iter().zip(q).map(|(a, b)| a.to_f64() * *b as f64).sum::<f64>() * scale).collect();
            let m = s.iter().cloned().fold(c.sinks[h] as f64, f64::max);
            let l = s.iter().map(|x| (x - m).exp()).sum::<f64>() + (c.sinks[h] as f64 - m).exp();
            let o = &mut out[(t * H + h) * HD..][..HD];
            for d in 0..HD {
                let v: f64 = rs.iter().zip(&s).map(|(r, x)| (x - m).exp() * r[d].to_f64()).sum();
                o[d] = bf16::from_f64(v / l).to_f32();
            }
        }
    }
    out
}

/// runs (old, tc, tc_fast) on the GPU
fn run(g: &Gpu, tc: &mut SparseAttnTc, c: &Case) -> [Vec<f32>; 3] {
    let q = g.upload(&c.q).unwrap();
    let win = g.upload(&c.win).unwrap();
    let comp = if c.k > 0 { Some(g.upload(&c.comp).unwrap()) } else { None };
    let cp = comp.as_ref().map_or(0, |b| b.ptr);
    let wi = g.upload(&c.win_idx).unwrap();
    let ci = g.upload(&c.comp_idx).unwrap();
    let sk = g.upload(&c.sinks).unwrap();
    let o = g.alloc(c.t * H * HD * 4).unwrap();
    let old = g.kernel("sparse_attn").unwrap();
    let scale = 1.0 / (HD as f32).sqrt();
    let args = [Arg::P(q.ptr), Arg::P(win.ptr), Arg::P(wi.ptr), Arg::I(c.w as i32), Arg::P(cp), Arg::P(ci.ptr), Arg::I(c.k as i32), Arg::P(sk.ptr), Arg::P(o.ptr), Arg::I(H as i32), Arg::I(HD as i32), Arg::F(scale)];
    g.launch(old, (c.t as u32, (H / 8) as u32, 1), 256, 0, &args).unwrap();
    let a = g.download::<f32>(o.ptr, c.t * H * HD).unwrap();
    let mut tcrun = |fast| {
        tc.fast = fast;
        g.memset(o.ptr, c.t * H * HD * 4).unwrap();
        tc.launch(g, q.ptr, win.ptr, wi.ptr, c.w, cp, ci.ptr, c.k, sk.ptr, o.ptr, c.t, H, HD).unwrap();
        g.download::<f32>(o.ptr, c.t * H * HD).unwrap()
    };
    let (b, f) = (tcrun(false), tcrun(true));
    [a, b, f]
}

/// (max abs diff, max relative diff over |ref| > 0.05, mean abs diff)
fn err(a: &[f32], b: &[f32]) -> (f32, f32, f64) {
    let mut m = (0f32, 0f32, 0f64);
    for (x, y) in a.iter().zip(b) {
        let d = (x - y).abs();
        m.0 = m.0.max(d);
        if y.abs() > 0.05 {
            m.1 = m.1.max(d / y.abs());
        }
        m.2 += d as f64;
    }
    m.2 /= a.len() as f64;
    m
}

fn random_case(r: &mut Rng, t: usize, w: usize, k: usize, qscale: f32) -> Case {
    let (nw, nc) = (300, 2000);
    let pick = |r: &mut Rng, n: usize| if r.below(10) == 0 { -1 } else { r.below(n) as i32 };
    let mut win_idx: Vec<i32> = (0..t * w).map(|_| pick(r, nw)).collect();
    let mut comp_idx: Vec<i32> = (0..t * k).map(|_| pick(r, nc)).collect();
    // duplicates
    for x in 0..t {
        if w > 4 {
            win_idx[x * w + 1] = win_idx[x * w];
            win_idx[x * w + 3] = 7;
        }
        if k > 4 {
            comp_idx[x * k + 2] = comp_idx[x * k];
            comp_idx[x * k + k - 1] = 7;
        }
    }
    // token 1: no valid rows at all
    if t > 1 {
        win_idx[w..2 * w].fill(-1);
        comp_idx[k..2 * k].fill(-1);
    }
    // token 2: only a handful of rows (one partial tile)
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
        q: (0..t * H * HD).map(|_| r.unif() * qscale).collect(),
        win: rows(r, nw),
        comp: if k > 0 { rows(r, nc) } else { vec![] },
        win_idx,
        comp_idx,
        sinks: (0..H).map(|_| r.unif() * 3.0).collect(),
    }
}

#[test]
fn sparse_attn_tc_matches() {
    let g = Gpu::new(0).unwrap();
    let mut tc = SparseAttnTc::new(&g).unwrap();
    let mut r = Rng(0x9e3779b97f4a7c15);
    for (t, w, k, qs) in [(6usize, 128usize, 512usize, 2.0f32), (4, 128, 0, 2.0), (5, 128, 512, 8.0), (3, 100, 37, 4.0)] {
        let c = random_case(&mut r, t, w, k, qs);
        let refo = reference(&c);
        let outs = run(&g, &mut tc, &c);
        eprintln!("T={t} W={w} K={k} q*{qs}: max abs / max rel / mean abs error");
        let mut e = vec![];
        for (name, x) in ["sparse_attn", "sparse_attn_tc", "sparse_attn_tc_fast"].iter().zip(&outs) {
            let (r, o) = (err(x, &refo), err(x, &outs[0]));
            eprintln!("  {name:19} vs f64 {:.2e} / {:.2e} / {:.2e}   vs sparse_attn {:.2e} / {:.2e} / {:.2e}", r.0, r.1, r.2, o.0, o.1, o.2);
            e.push(r);
        }
        for x in &outs[1..] {
            assert!(x.iter().all(|v| v.is_finite()));
            // all-masked token -> zeros
            if t > 1 {
                assert!(x[H * HD..2 * H * HD].iter().all(|&v| v == 0.0));
            }
        }
        // rows are O(1): 1e-2 is about one bf16 ulp of the output; the fast variant rounds q and p to bf16
        assert!(e[0].0 < 1e-2, "sparse_attn off the reference");
        assert!(e[1].0 < 1e-2 && e[1].1 < 1e-2, "sparse_attn_tc off the reference: {:?}", e[1]);
        assert!(e[2].0 < 4e-2, "sparse_attn_tc_fast off the reference: {:?}", e[2]);
    }
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
fn sparse_attn_tc_bench() {
    let g = Gpu::new(0).unwrap();
    let mut r = Rng(12345);
    let (t, w, k, nc) = (4096usize, 128usize, 512usize, 16384usize);
    // window = the token's previous 128 positions (itself included) in a [128 + T] buffer; comp = 512 distinct of 16K
    let win_idx: Vec<i32> = (0..t).flat_map(|x| (0..w).map(move |i| (x + 1 + i) as i32)).collect();
    let mut comp_idx = vec![0i32; t * k];
    let mut perm: Vec<i32> = (0..nc as i32).collect();
    for x in 0..t {
        for j in 0..k {
            let s = j + r.below(nc - j);
            perm.swap(j, s);
        }
        comp_idx[x * k..][..k].copy_from_slice(&perm[..k]);
    }
    let q = g.upload(&(0..t * H * HD).map(|_| r.unif() * 2.0).collect::<Vec<f32>>()).unwrap();
    let win = g.upload(&rows(&mut r, w + t)).unwrap();
    let comp = g.upload(&rows(&mut r, nc)).unwrap();
    let wi = g.upload(&win_idx).unwrap();
    let ci = g.upload(&comp_idx).unwrap();
    let sk = g.upload(&(0..H).map(|_| r.unif()).collect::<Vec<f32>>()).unwrap();
    let o = g.alloc(t * H * HD * 4).unwrap();
    let old = g.kernel("sparse_attn").unwrap();
    let args = [Arg::P(q.ptr), Arg::P(win.ptr), Arg::P(wi.ptr), Arg::I(w as i32), Arg::P(comp.ptr), Arg::P(ci.ptr), Arg::I(k as i32), Arg::P(sk.ptr), Arg::P(o.ptr), Arg::I(H as i32), Arg::I(HD as i32), Arg::F(1.0 / (HD as f32).sqrt())];
    let t_old = time(&mut || g.launch(old, (t as u32, (H / 8) as u32, 1), 256, 0, &args).unwrap(), &g, 5);
    let a = g.download::<f32>(o.ptr, t * H * HD).unwrap();
    let flop = 2.0 * 2.0 * (t * H * (w + k) * HD) as f64;
    eprintln!("{} ({} SMs), T={t} W={w} K={k}", g.name, g.sms);
    eprintln!("sparse_attn         {t_old:8.2} ms  {:6.1} TFLOP/s", flop / t_old / 1e9);
    for fast in [false, true] {
        let mut tc = SparseAttnTc::new(&g).unwrap();
        tc.fast = fast;
        let t_new = time(&mut || tc.launch(&g, q.ptr, win.ptr, wi.ptr, w, comp.ptr, ci.ptr, k, sk.ptr, o.ptr, t, H, HD).unwrap(), &g, 20);
        let b = g.download::<f32>(o.ptr, t * H * HD).unwrap();
        let name = if fast { "sparse_attn_tc_fast" } else { "sparse_attn_tc" };
        eprintln!("{name:19} {t_new:8.2} ms  {:6.1} TFLOP/s (mma work x{})  {:.1}x", flop / t_new / 1e9, if fast { 1 } else { 2 }, t_old / t_new);
        let e = err(&b, &a);
        eprintln!("  vs sparse_attn: max abs {:.2e}, max rel {:.2e}, mean abs {:.2e}", e.0, e.1, e.2);
    }
}

/// sparse_attn_tile + sparse_attn_merge (split-K, T <= 8) against the f64 reference; T=1 timing next to sparse_attn
#[test]
fn sparse_attn_tile_matches() {
    let g = Gpu::new(0).unwrap();
    let mut r = Rng(0x51ab1e);
    let (tile_k, merge_k, old) = (g.kernel("sparse_attn_tile").unwrap(), g.kernel("sparse_attn_merge").unwrap(), g.kernel("sparse_attn").unwrap());
    let scale = 1.0 / (HD as f32).sqrt();
    for (t, w, k, qs) in [(1usize, 128usize, 512usize, 2.0f32), (6, 128, 512, 2.0), (4, 128, 0, 2.0), (5, 128, 512, 8.0), (3, 100, 37, 4.0)] {
        let c = random_case(&mut r, t, w, k, qs);
        let refo = reference(&c);
        let q = g.upload(&c.q).unwrap();
        let win = g.upload(&c.win).unwrap();
        let comp = if c.k > 0 { Some(g.upload(&c.comp).unwrap()) } else { None };
        let cp = comp.as_ref().map_or(0, |b| b.ptr);
        let wi = g.upload(&c.win_idx).unwrap();
        let ci = g.upload(&c.comp_idx).unwrap();
        let sk = g.upload(&c.sinks).unwrap();
        let o = g.alloc(c.t * H * HD * 4).unwrap();
        let nz = (w + k).div_ceil(32);
        let part = g.alloc(t * nz * H * (HD + 2) * 4).unwrap();
        let mut tile = || {
            g.launch(tile_k, (t as u32, (H / 16) as u32, nz as u32), 256, 0, &[Arg::P(q.ptr), Arg::P(win.ptr), Arg::P(wi.ptr), Arg::I(w as i32), Arg::P(cp), Arg::P(ci.ptr), Arg::I(k as i32), Arg::P(part.ptr), Arg::I(H as i32), Arg::I(HD as i32), Arg::F(scale)]).unwrap();
            g.launch(merge_k, (t as u32, H as u32, 1), 256, 0, &[Arg::P(part.ptr), Arg::I(nz as i32), Arg::P(sk.ptr), Arg::P(o.ptr), Arg::I(H as i32), Arg::I(HD as i32)]).unwrap();
        };
        tile();
        let x = g.download::<f32>(o.ptr, t * H * HD).unwrap();
        let e = err(&x, &refo);
        eprintln!("T={t} W={w} K={k} q*{qs}: tile vs f64 max abs {:.2e} max rel {:.2e} mean {:.2e}", e.0, e.1, e.2);
        assert!(x.iter().all(|v| v.is_finite()));
        assert!(e.0 < 1e-2, "tiled attention off the reference: {e:?}");
        if t == 1 {
            let args = [Arg::P(q.ptr), Arg::P(win.ptr), Arg::P(wi.ptr), Arg::I(w as i32), Arg::P(cp), Arg::P(ci.ptr), Arg::I(k as i32), Arg::P(sk.ptr), Arg::P(o.ptr), Arg::I(H as i32), Arg::I(HD as i32), Arg::F(scale)];
            let t_old = time(&mut || g.launch(old, (1, (H / 8) as u32, 1), 256, 0, &args).unwrap(), &g, 200);
            let t_new = time(&mut tile, &g, 200);
            eprintln!("{}: T=1 W={w} K={k}: sparse_attn {:.1} us, tile + merge {:.1} us", g.name, t_old * 1e3, t_new * 1e3);
        }
    }
}
