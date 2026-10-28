#![cfg(feature = "cuda")]
//! GPU indexer selection (topk.cuh) vs a CPU reference with model.rs indexer()'s semantics.
use dsx::gpu::*;
use dsx::topk::{Rows, TopK};

const K: usize = 512;
const BS: usize = 8;
const NBLOCKS: usize = 2048;

fn rng(seed: &mut u64) -> u64 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    *seed
}

/// Scores quantized to `levels` values (many ties), a fraction `neg` of them -inf; some -0.0 next to 0.0.
fn scores(n: usize, levels: u64, neg: u64, seed: &mut u64) -> Vec<f32> {
    (0..n)
        .map(|_| {
            let r = rng(seed);
            if r % 100 < neg {
                f32::NEG_INFINITY
            } else {
                let v = ((r >> 8) % levels) as f32 * 0.25 - levels as f32 * 0.125;
                if v == 0.0 && r & (1 << 40) != 0 { -0.0 } else { v }
            }
        })
        .collect()
}

// --- CPU reference (model.rs indexer()) ---

fn ref_blocks(s: &[f32]) -> Vec<bool> {
    let end = s.len();
    let nb = end.div_ceil(BS);
    let mut bscore: Vec<f32> = (0..nb).map(|b| s[b * BS..((b + 1) * BS).min(end)].iter().cloned().fold(f32::NEG_INFINITY, f32::max)).collect();
    bscore[(end - 1) / BS] = f32::INFINITY;
    let mut order: Vec<usize> = (0..nb).collect();
    order.sort_by(|&a, &b| bscore[b].partial_cmp(&bscore[a]).unwrap());
    let mut keep = vec![false; end];
    for &b in order.iter().take(NBLOCKS.min(nb)) {
        if bscore[b] > f32::NEG_INFINITY {
            for i in b * BS..((b + 1) * BS).min(end) {
                keep[i] = true;
            }
        }
    }
    keep
}

fn ref_topk(s: &[f32], keep: Option<&[bool]>) -> Vec<i32> {
    let end = s.len();
    let mut score = s.to_vec();
    if let Some(keep) = keep {
        for (i, v) in score.iter_mut().enumerate() {
            if !keep.get(i).copied().unwrap_or(false) {
                *v = f32::NEG_INFINITY;
            }
        }
    }
    let k = K.min(end);
    let mut order: Vec<usize> = (0..end).collect();
    order.sort_by(|&a, &b| score[b].partial_cmp(&score[a]).unwrap().then(a.cmp(&b)));
    let mut idx: Vec<usize> = order[..k].to_vec();
    idx.sort();
    idx.iter().map(|&i| if score[i] > f32::NEG_INFINITY { i as i32 } else { -1 }).collect()
}

// --- GPU ---

struct Case {
    rows: Vec<Vec<f32>>,
}

/// Runs candidate blocks (when `cand`), then top-k (masked by those blocks when `cand`) on all rows at once.
/// Returns per row (top-k, keep mask over positions).
fn gpu_select(g: &Gpu, tk: &TopK, c: &Case, cand: bool, per_row_n: bool) -> (Vec<Vec<i32>>, Vec<Vec<bool>>) {
    let rows = c.rows.len();
    let n_max = c.rows.iter().map(|r| r.len()).max().unwrap();
    let lds = n_max.next_multiple_of(4);
    let mut flat = vec![f32::NAN; rows * lds];
    for (r, v) in c.rows.iter().enumerate() {
        flat[r * lds..r * lds + v.len()].copy_from_slice(v);
    }
    let sb = g.upload(&flat).unwrap();
    let ns: Vec<i32> = c.rows.iter().map(|r| r.len() as i32).collect();
    let nsb = g.upload(&ns).unwrap();
    let rw = Rows { scores: sb.ptr, lds, rows, ns: if per_row_n { nsb.ptr } else { 0 }, n_max };
    let ldk = n_max.div_ceil(BS);
    let keepb = g.alloc(rows * ldk).unwrap();
    let out = g.alloc(rows * K * 4).unwrap();
    let scratch = g.alloc(TopK::topk_scratch(n_max, K, rows).max(TopK::blocks_scratch(n_max, BS, NBLOCKS, rows))).unwrap();
    // garbage in the outputs: the kernels must write every slot they own
    g.htod_sync(keepb.ptr, &vec![7u8; rows * ldk]).unwrap();
    g.htod_sync(out.ptr, &vec![12345i32; rows * K]).unwrap();
    if cand {
        tk.blocks(g, &rw, BS, NBLOCKS, keepb.ptr, ldk, scratch.ptr).unwrap();
    }
    tk.topk(g, &rw, K, if cand { keepb.ptr } else { 0 }, ldk, BS, out.ptr, K, scratch.ptr).unwrap();
    let o: Vec<i32> = g.download(out.ptr, rows * K).unwrap();
    let kb: Vec<u8> = g.download(keepb.ptr, rows * ldk).unwrap();
    let mut tops = vec![];
    let mut keeps = vec![];
    for (r, v) in c.rows.iter().enumerate() {
        let k = K.min(v.len());
        assert!(o[r * K + k..(r + 1) * K].iter().all(|&x| x == -1), "pad slots");
        tops.push(o[r * K..r * K + k].to_vec());
        keeps.push((0..v.len()).map(|i| kb[r * ldk + i / BS] == 1).collect());
        if cand {
            assert!(kb[r * ldk..r * ldk + v.len().div_ceil(BS)].iter().all(|&x| x <= 1), "keep bytes");
        }
    }
    (tops, keeps)
}

fn check(g: &Gpu, tk: &TopK, c: &Case, what: &str) {
    for per_row_n in [false, true] {
        if !per_row_n && c.rows.iter().any(|r| r.len() != c.rows[0].len()) {
            continue;
        }
        let (tops, _) = gpu_select(g, tk, c, false, per_row_n);
        let (ctops, keeps) = gpu_select(g, tk, c, true, per_row_n);
        for (r, v) in c.rows.iter().enumerate() {
            assert_eq!(tops[r], ref_topk(v, None), "{what} row {r} n={} top-k", v.len());
            let rk = ref_blocks(v);
            if keeps[r] != rk {
                let i = (0..v.len()).find(|&i| keeps[r][i] != rk[i]).unwrap();
                panic!("{what} row {r} n={} keep differs at {i} (block {})", v.len(), i / BS);
            }
            assert_eq!(ctops[r], ref_topk(v, Some(&rk)), "{what} row {r} n={} masked top-k", v.len());
        }
    }
}

#[test]
fn topk_matches_reference() {
    let g = Gpu::new(0).unwrap();
    let tk = TopK::new(&g).unwrap();
    let mut seed = 0x9e3779b97f4a7c15u64;
    for &n in &[1usize, 7, 8, 9, 511, 512, 513, 4096, 16384, 16385, 100_000, 1_048_576] {
        for &(levels, neg) in &[(4u64, 10u64), (50, 30), (1 << 20, 5), (3, 99), (1, 100)] {
            let c = Case { rows: vec![scores(n, levels, neg, &mut seed)] };
            check(&g, &tk, &c, &format!("levels={levels} neg={neg}%"));
        }
    }
}

#[test]
fn topk_rows() {
    // verify-batch shape: rows of different n in one launch, nblocks above and below 2048
    let g = Gpu::new(0).unwrap();
    let tk = TopK::new(&g).unwrap();
    let mut seed = 42u64;
    for &ns in &[&[1usize, 9, 513, 4096][..], &[16_000, 16_384, 16_390, 20_000, 131_072], &[100_000, 600_000, 1_000_000]] {
        let c = Case { rows: ns.iter().map(|&n| scores(n, 6, 20, &mut seed)).collect() };
        check(&g, &tk, &c, &format!("rows {ns:?}"));
    }
}

#[test]
#[ignore] // cargo test --release --features cuda --test gpu_topk -- --ignored --nocapture
fn topk_timing() {
    let g = Gpu::new(0).unwrap();
    let tk = TopK::new(&g).unwrap();
    let mut seed = 7u64;
    eprintln!("{} sm_{}", g.name, g.arch);
    let mut floor = f64::INFINITY;
    {
        let (k, b) = (g.kernel("round_bf16").unwrap(), g.alloc(4).unwrap());
        for _ in 0..5 {
            let t = std::time::Instant::now();
            for _ in 0..100 {
                g.launch(k, (1, 1, 1), 32, 0, &[Arg::P(b.ptr), Arg::I(1)]).unwrap();
            }
            g.sync().unwrap();
            floor = floor.min(t.elapsed().as_secs_f64() * 1e4);
        }
    }
    eprintln!("empty launch: {floor:.1} us");
    for (&n, ties) in [4096usize, 32_768, 131_072, 1_048_576].iter().flat_map(|n| [(n, false), (n, true)]) {
        let s: Vec<f32> = if ties { scores(n, 4, 10, &mut seed) } else { (0..n).map(|_| (rng(&mut seed) % 1_000_000) as f32 * 1e-3 - 300.0).collect() };
        let sb = g.upload(&s).unwrap();
        let rw = Rows { scores: sb.ptr, lds: n, rows: 1, ns: 0, n_max: n };
        let keep = g.alloc(n.div_ceil(BS)).unwrap();
        let out = g.alloc(K * 4).unwrap();
        let scratch = g.alloc(TopK::topk_scratch(n, K, 1).max(TopK::blocks_scratch(n, BS, NBLOCKS, 1))).unwrap();
        let time = |f: &dyn Fn()| {
            for _ in 0..10 {
                f();
            }
            g.sync().unwrap();
            let iters = 100;
            (0..10)
                .map(|_| {
                    let t = std::time::Instant::now();
                    for _ in 0..iters {
                        f();
                    }
                    g.sync().unwrap();
                    t.elapsed().as_secs_f64() * 1e6 / iters as f64
                })
                .fold(f64::INFINITY, f64::min)
        };
        let t_top = time(&|| tk.topk(&g, &rw, K, 0, 0, BS, out.ptr, K, scratch.ptr).unwrap());
        let t_blk = time(&|| tk.blocks(&g, &rw, BS, NBLOCKS, keep.ptr, n.div_ceil(BS), scratch.ptr).unwrap());
        let t_msk = time(&|| tk.topk(&g, &rw, K, keep.ptr, n.div_ceil(BS), BS, out.ptr, K, scratch.ptr).unwrap());
        eprintln!("n={n:>8}{}: top-k {t_top:6.1} us | candidate blocks {t_blk:6.1} us | masked top-k {t_msk:6.1} us", if ties { " (4 values, 10% -inf)" } else { "" });
    }
}
