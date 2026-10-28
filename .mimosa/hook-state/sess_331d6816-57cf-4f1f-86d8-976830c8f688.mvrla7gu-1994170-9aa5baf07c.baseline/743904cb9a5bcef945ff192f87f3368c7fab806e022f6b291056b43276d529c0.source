//! The indexer's selection on the GPU (kernels/topk.cuh): candidate blocks, masking and top-k, the same as
//! model.rs indexer(). Launches only, no syncs. Top-512 over n scores: 1 launch for n <= 16K, 2 up to 512K, 3 beyond;
//! top-2048 candidate blocks of 8: 1 launch for n <= 32K, 2 up to 128K, 3 up to 1M.
use crate::gpu::{Arg, Gpu, Kernel};
use anyhow::{Result, ensure};

/// Elements one block takes per level (TK_CHUNK).
pub const CHUNK: usize = 16384;

/// Score rows: row r is `scores + r * lds` floats (-inf allowed, no NaN); its length is `ns[r]` (device i32 per row)
/// or `n_max` when `ns` is 0. Grids are sized for `n_max`.
#[derive(Clone, Copy, Debug)]
pub struct Rows {
    pub scores: u64,
    pub lds: usize,
    pub rows: usize,
    pub ns: u64,
    pub n_max: usize,
}

pub struct TopK {
    level: Kernel,
    bmax: Kernel,
}

/// Candidate blocks per row above which their keys are computed grid-wide first (topk_blockmax).
const BMAX_SPLIT: usize = 4096;

/// Pairs per row of one ping-pong level buffer for selecting `kcap` of `m0` elements.
fn level_pairs(m0: usize, kcap: usize) -> usize {
    if m0 <= CHUNK { 0 } else { m0.div_ceil(CHUNK) * kcap }
}

impl TopK {
    pub fn new(g: &Gpu) -> Result<TopK> {
        Ok(TopK { level: g.kernel("topk_level")?, bmax: g.kernel("topk_blockmax")? })
    }

    /// Scratch bytes for `topk` over `rows` rows of up to `n_max` scores.
    pub fn topk_scratch(n_max: usize, k: usize, rows: usize) -> usize {
        2 * rows * level_pairs(n_max, k) * 8
    }

    /// Scratch bytes for `blocks` over `rows` rows of up to `n_max` scores.
    pub fn blocks_scratch(n_max: usize, bs: usize, nblocks: usize, rows: usize) -> usize {
        let nb = n_max.div_ceil(bs);
        rows * (if nb > BMAX_SPLIT { nb } else { 0 } + 2 * level_pairs(nb, nblocks)) * 8
    }

    /// Top-k: per row, the min(k, n) highest scores (ties: lower position first) as positions ascending, i32 at
    /// `out + r * ldo * 4`, -1 for a selected position whose score is -inf; slots [min(k, n), k) get -1.
    /// `mask` (0 = none): byte per `bs` positions, row stride `ldm`; positions whose byte is 0 count as -inf.
    pub fn topk(&self, g: &Gpu, r: &Rows, k: usize, mask: u64, ldm: usize, bs: usize, out: u64, ldo: usize, scratch: u64) -> Result<()> {
        ensure!(ldo >= k, "out stride {ldo} < k {k}");
        self.run(g, r, 0, 0, r.n_max, k, bs, mask, ldm, 1, out, ldo, 0, 0, scratch)
    }

    /// Candidate blocks: block b = positions [b*bs, b*bs+bs); block score = max over it, the block holding n-1
    /// pinned to +inf; keeps the top min(nblocks, ceil(n/bs)) (ties: lower block first), dropping -inf blocks.
    /// Writes `keep[r * ldk + b]` = 1/0 for b < ceil(n/bs): the `mask` for `topk` on the later layers.
    pub fn blocks(&self, g: &Gpu, r: &Rows, bs: usize, nblocks: usize, keep: u64, ldk: usize, scratch: u64) -> Result<()> {
        let nb = r.n_max.div_ceil(bs);
        ensure!(ldk >= nb, "keep stride {ldk} < blocks");
        if nb <= BMAX_SPLIT {
            return self.run(g, r, 1, 0, nb, nblocks, bs, 0, 0, 2, 0, 0, keep, ldk, scratch);
        }
        let i = |v: usize| Arg::I(v as i32);
        g.launch(self.bmax, (nb.div_ceil(256) as u32, r.rows as u32, 1), 256, 0, &[Arg::P(r.scores), i(r.lds), Arg::P(r.ns), i(r.n_max), i(bs), Arg::P(scratch), i(nb)])?;
        self.run(g, r, 2, scratch, nb, nblocks, bs, 0, 0, 2, 0, 0, keep, ldk, scratch + (r.rows * nb * 8) as u64)
    }

    #[allow(clippy::too_many_arguments)]
    fn run(&self, g: &Gpu, r: &Rows, in_mode: i32, input: u64, m0: usize, kcap: usize, bs: usize, mask: u64, ldm: usize, out_mode: i32, pos: u64, ldp: usize, keep: u64, ldk: usize, scratch: u64) -> Result<()> {
        ensure!(kcap <= CHUNK / 2 && kcap > 0 && bs > 0 && r.n_max < i32::MAX as usize);
        let half = r.rows * level_pairs(m0, kcap) * 8;
        let (mut src, mut dst) = (input, scratch);
        let (mut mode, mut m) = (in_mode, m0);
        let i = |v: usize| Arg::I(v as i32);
        loop {
            let last = m <= CHUNK;
            let blocks = m.div_ceil(CHUNK);
            ensure!(last || scratch != 0, "topk: scratch needed for {m0} elements");
            g.launch(
                self.level,
                (blocks as u32, r.rows as u32, 1),
                1024,
                0,
                &[Arg::P(r.scores), i(r.lds), Arg::P(mask), i(ldm), Arg::P(src), i(m), Arg::P(r.ns), i(r.n_max), i(bs), i(kcap), Arg::I(mode), Arg::I(if last { out_mode } else { 0 }), Arg::P(dst), i(blocks * kcap), Arg::P(pos), i(ldp), Arg::P(keep), i(ldk)],
            )?;
            if last {
                return Ok(());
            }
            m = blocks * kcap;
            mode = 2;
            // ping-pong between the two level buffers
            (src, dst) = (dst, if dst == scratch { scratch + half as u64 } else { scratch });
        }
    }
}
