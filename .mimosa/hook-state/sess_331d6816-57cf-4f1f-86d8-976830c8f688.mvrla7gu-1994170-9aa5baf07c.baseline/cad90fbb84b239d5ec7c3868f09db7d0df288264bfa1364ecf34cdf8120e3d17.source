//! Speculative verify: the main model over a few tokens at once (the newest token + drafts), with decode-style
//! weight streaming (kernels/verify.h) and the prompt path's per-token kernels; state is committed for the
//! accepted prefix only.

use crate::decode::*;
use crate::hip::Arg;
use anyhow::Result;

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct VJob {
    pub w: u64,
    pub sc: u64,
    pub woff: i64,
    pub soff: i64,
    pub mul: f32,
    pub fmt: i32,
    pub n: i32,
    pub k: i32,
    pub split: i32,
    pub zero: i32,
    pub xq: u64,
    pub xs: u64,
    pub xb: u64,
    pub ldx: i32,
    pub ntok: i32,
    pub rows: u64,
    pub y: u64,
    pub orows: u64,
    pub ldy: i32,
    pub atomic: i32,
    pub blocks: i32,
    pub epi: i32,
    pub ropec: i32,
    pub ndyn: i32,
    pub dyn_: u64,
    pub xl: u64,
}

const VMAX: usize = 12;
/// batched decode expert kernels: 16-row tiles per wave (DSXA_XR: 2 or 4)
fn xr_tiles() -> usize {
    std::env::var("DSXA_XR").ok().and_then(|v| v.parse().ok()).filter(|&r| r == 1 || r == 2 || r == 4).unwrap_or(2)
}
/// waves a vgemv launch aims for (k-split until reached); dev `vwaves N`
pub static VWAVES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(2048);

#[repr(C)]
#[derive(Clone, Copy)]
struct VJobs {
    j: [VJob; VMAX],
    n: i32,
    cb3lo: u32,
    cb3hi: u32,
    cb2: u32,
}

impl Decoder {
    /// one vgemv launch; split 0 = chosen so the launch has ~2048 waves
    pub(crate) fn vlaunch(&self, label: &str, mut jobs: Vec<VJob>, pos0: u64) -> Result<()> {
        for chunk in jobs.chunks_mut(VMAX) {
            let tiles: usize = chunk.iter().map(|j| (j.n as usize).div_ceil(16) * j.ndyn.max(1) as usize).sum();
            let mut blocks = 0u32;
            for j in chunk.iter_mut() {
                if j.split == 0 {
                    let ku = match j.fmt { 0 => 64, 4 => 32, _ => 128 };
                    let units = j.k as usize / ku;
                    let mut s = 1;
                    // scattered outputs that are stored (not added) cannot be zeroed up front: no k-split
                    let fixed = j.atomic == 0 && (j.orows != 0 || j.ndyn > 0);
                    while !fixed && tiles * s < VWAVES.load(std::sync::atomic::Ordering::Relaxed) && s * 2 <= units / 2 {
                        s *= 2;
                    }
                    j.split = s as i32;
                }
                if j.split > 1 && j.atomic == 0 {
                    // k-split partial sums accumulate: zero this job's output rows first (vzero), unless the prep before
                    // this product already cleared them (Decoder::pz)
                    j.atomic = 1;
                    j.zero = (!self.pz.get()) as i32;
                }
                let rows_per_block = match self.xg.get() { Some((r, _)) if j.ndyn > 0 => 64 * r, _ => 64 };
                j.blocks = ((j.n as usize).div_ceil(rows_per_block) * j.split as usize * j.ndyn.max(1) as usize) as i32;
                blocks += j.blocks as u32;
            }
            let n = chunk.len();
            let mut arr = [chunk[0]; VMAX];
            arr[..n].copy_from_slice(chunk);
            let js = VJobs { j: arr, n: n as i32, cb3lo: self.cb3[0], cb3hi: self.cb3[1], cb2: self.cb2 };
            let b = unsafe { std::slice::from_raw_parts(&js as *const VJobs as *const u8, std::mem::size_of::<VJobs>()).to_vec() };
            if chunk.iter().any(|j| j.zero != 0) {
                let elems: usize = chunk.iter().filter(|j| j.zero != 0).map(|j| (j.n * j.ntok) as usize).sum();
                self.g.launch("vzero", (elems.div_ceil(1024).clamp(1, 1216) as u32, 1, 1), 256, 0, &[Arg::B(b.clone())])?;
            }
            self.g.set_label(Some(label));
            // kernels specialized by format family (fewer registers, more waves in flight)
            let dense = chunk.iter().all(|j| j.ndyn == 0 && (j.fmt == 0 || j.fmt == 4));
            let experts = chunk.iter().all(|j| j.ndyn > 0 || (1..=3).contains(&j.fmt));
            let kn = if dense { "vgemv_d" } else if experts { "vgemv_x" } else { "vgemv" };
            // batched decode / prompt chunks: R row tiles x G token groups per wave
            let r = if let (Some((rt, gg)), true) = (self.xg.get(), chunk.iter().all(|j| j.ndyn > 0)) {
                self.g.launch(&format!("vgemv_xg{rt}{gg}"), (blocks, 1, 1), 256, 0, &[Arg::B(b)])
            } else {
                self.g.launch(kn, (blocks, 1, 1), 256, 0, &[Arg::B(b), Arg::P(pos0), Arg::P(self.rope_f.ptr), Arg::P(self.rope_fc.ptr)])
            };
            self.g.set_label(None);
            r?;
        }
        Ok(())
    }
}

impl Decoder {
    /// batched decode, dense jobs (fmt 0 / 4, static, no epilogue) over up to 256 tokens: bgemv_d{G} streams each
    /// weight once for G groups of 16 tokens; k-split toward VWAVES waves (outputs zeroed first unless they add)
    pub(crate) fn blaunch(&self, label: &str, mut jobs: Vec<VJob>) -> Result<()> {
        let nt = jobs.iter().map(|j| j.ntok).max().unwrap_or(0) as usize;
        if nt <= 16 {
            return self.vlaunch(label, jobs, self.st.ptr);
        }
        anyhow::ensure!(nt <= 256 && jobs.iter().all(|j| j.ndyn == 0 && (j.fmt == 0 || j.fmt == 4) && j.epi == 0));
        let gsz = nt.div_ceil(16).next_power_of_two().max(2);
        let target = std::env::var("DSXA_BWAVES").ok().and_then(|v| v.parse().ok()).unwrap_or(VWAVES.load(std::sync::atomic::Ordering::Relaxed));
        for chunk in jobs.chunks_mut(VMAX) {
            let tiles: usize = chunk.iter().map(|j| (j.n as usize).div_ceil(16)).sum();
            let mut blocks = 0u32;
            for j in chunk.iter_mut() {
                if j.split == 0 {
                    let units = j.k as usize / if j.fmt == 0 { 64 } else { 32 };
                    let mut s = 1;
                    while tiles * s < target && s * 2 <= units / 2 {
                        s *= 2;
                    }
                    j.split = s as i32;
                }
                if j.split > 1 && j.atomic == 0 {
                    j.atomic = 1;
                    j.zero = (!self.pz.get()) as i32;
                }
                j.blocks = ((j.n as usize).div_ceil(64) * j.split as usize) as i32;
                blocks += j.blocks as u32;
            }
            let n = chunk.len();
            let mut arr = [chunk[0]; VMAX];
            arr[..n].copy_from_slice(chunk);
            let js = VJobs { j: arr, n: n as i32, cb3lo: self.cb3[0], cb3hi: self.cb3[1], cb2: self.cb2 };
            let b = unsafe { std::slice::from_raw_parts(&js as *const VJobs as *const u8, std::mem::size_of::<VJobs>()).to_vec() };
            if chunk.iter().any(|j| j.zero != 0) {
                let elems: usize = chunk.iter().filter(|j| j.zero != 0).map(|j| (j.n * j.ntok) as usize).sum();
                self.g.launch("vzero", (elems.div_ceil(1024).clamp(1, 1216) as u32, 1, 1), 256, 0, &[Arg::B(b.clone())])?;
            }
            self.g.set_label(Some(label));
            let r = self.g.launch(&format!("bgemv_d{gsz}"), (blocks, 1, 1), 256, 0, &[Arg::B(b)]);
            self.g.set_label(None);
            r?;
        }
        Ok(())
    }
}

/// self-test hook: one static vgemv job over `ntok` activation rows
#[allow(clippy::too_many_arguments)]
pub fn test_vgemv(g: &crate::hip::Gpu, fmt: i32, w: u64, sc: u64, soff: i64, n: usize, k: usize, xq: u64, xs: u64, xb: u64, ntok: usize, y: u64, split: i32) -> Result<()> {
    let mut j = VJob { w, sc, soff, mul: 1.0, fmt, n: n as i32, k: k as i32, xq, xs, xb, ldx: k as i32, ntok: ntok as i32, y, ldy: n as i32, split, ..Default::default() };
    if split > 1 {
        j.atomic = 1;
    }
    j.blocks = (n.div_ceil(64) * split as usize) as i32;
    let js = VJobs { j: [j; VMAX], n: 1, cb3lo: 0, cb3hi: 0, cb2: 0 };
    let b = unsafe { std::slice::from_raw_parts(&js as *const VJobs as *const u8, std::mem::size_of::<VJobs>()).to_vec() };
    let pos = g.upload(&[0i32; 4])?;
    g.launch("vgemv", (j.blocks as u32, 1, 1), 256, 0, &[Arg::B(b), Arg::P(pos.ptr), Arg::P(0), Arg::P(0)])?;
    g.sync()
}
