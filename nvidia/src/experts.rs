//! Routed experts resident in a GPU's memory (the main GPU's spare VRAM, or an aux GPU as a pure expert store), computed there
//! for one token: out = sum_e w_e * down(silu(min(gate x, L)) * clamp(up x, -L, L)).
//! By default the weights are repacked at upload (same bytes) for the fast kernels (expert_fast.cuh: int8 tensor-core
//! mma, int8 activations, up to 8 tokens per weight read); DSX_OLD_MOE=1 keeps the GGUF blocks and moe_gu / moe_down.

use crate::gguf::GType;
use crate::gpu::{Arg, Buf, Event, Gpu, Kernel};
use crate::model::Model;
use anyhow::{Result, bail};
use std::collections::HashMap;

struct Slot {
    gate: u64,
    up: u64,
    down: u64,
    gu: GType,
    dn: GType,
}

/// One expert's GGUF matrices: gate/up [inter][dim] in type `gu`, down [dim][inter] in `dn`.
pub struct RawExpert<'a> {
    pub gate: &'a [u8],
    pub up: &'a [u8],
    pub down: &'a [u8],
    pub gu: GType,
    pub dn: GType,
}

#[derive(Clone, Copy)]
pub struct ExpCfg {
    pub n_layers: usize,
    pub n_experts: usize,
    pub inter: usize,
    pub dim: usize,
    pub lim: f32,
    /// bit l: layer l's experts are stored in the Hadamard-rotated input basis (quant_act rotates x and h)
    pub rot: u64,
}

/// most tokens (T) and picks per token (K) of one run_device_fast call
pub const MAXT: usize = 8;
pub const MAXK: usize = 8;
/// warps per block of the fast kernels (KS in expert_fast.cuh): they split a 16-row tile's x-blocks
/// 16-row tiles per block of the fast kernels (ROWT in expert_fast.cuh; build with DSX_ROWT=n)
pub const ROWT: u32 = match option_env!("DSX_ROWT") {
    Some(s) => (s.as_bytes()[0] - b'0') as u32,
    None => 1,
};
pub const KS: u32 = match option_env!("DSX_KS") {
    Some(s) => (s.as_bytes()[0] - b'0') as u32,
    None => 2,
};

/// scratch of the fast path: x and h quantized (quant_act layout), h in f32, the call's plan, and moe_down_fast's
/// per-entry partial sums and row-tile counters
struct Fast {
    xq: Buf,
    xd: Buf,
    xs: Buf,
    h: Buf,
    hq: Buf,
    hd: Buf,
    hs: Buf,
    plan: Buf,
    part: Buf,
    cnt: Buf,
}

pub struct GpuExperts {
    buf: Buf,
    /// device residency table [layers][E] of ExpSlot (offsets into `buf`, ~0 = not here), for moe_gu / moe_down
    tab: Buf,
    pub sel: Buf,
    pub wts: Buf,
    h: Buf,
    n_experts: usize,
    slots: HashMap<(usize, usize), Slot>,
    /// device buffers for one token: x in, out (sum), and scratch
    pub x: Buf,
    pub out: Buf,
    tmp_g: Buf,
    tmp_u: Buf,
    tmp_h: Buf,
    tmp_o: Buf,
    k: HashMap<&'static str, Kernel>,
    inter: usize,
    dim: usize,
    lim: f32,
    rot: u64,
    pub bytes: usize,
    /// dynamic shared memory of the fast kernels: the largest codebook table among the store's types
    smem: u32,
    /// weights repacked for the fast kernels (expert_fast.cuh); None = raw GGUF blocks (DSX_OLD_MOE=1)
    fast: Option<Fast>,
    /// adaptive residency (enable_swaps): in-flight replacements
    swap: Option<Swapper>,
    /// per layer: (gate / up, down) bits x 2 when the layer's experts are T16
    t16: Vec<Option<(usize, usize)>>,
    t16b: Option<T16Bufs>,
}

/// An expert replacing a same-type victim in place: its raw matrices go to a staging buffer and are repacked into the
/// victim's slot on the side stream; it becomes resident (table entry, `slots`) only at commit, between tokens.
struct Pending {
    l: usize,
    e: usize,
    slot: Slot,
    stage: usize,
}

struct Swapper {
    /// one staging buffer (gate | up | down raw) and one event per in-flight replacement
    stage: Vec<(Buf, Event)>,
    /// pinned ring of 32-byte table entries for async writes (at most a few dozen per token; the main stream is
    /// synchronized every token, so a slot is free again long before the ring wraps)
    ring: crate::gpu::HostBuf,
    ring_i: usize,
    free: Vec<usize>,
    pending: Vec<Pending>,
    /// recorded on the main stream before each replacement's copies, which the side stream waits for: a replacement
    /// can start while the main stream still runs the current token (whose kernels may read the victim's slot)
    after_main: Event,
}

fn gemv_name(t: GType) -> Result<&'static str> {
    Ok(match t {
        GType::Q2_0 => "gemv_q2_0",
        GType::Ptq1_0 => "gemv_ptq1_0",
        GType::Q3K => "gemv_q3_k",
        GType::Q8_0 => "gemv_q8_0",
        t => bail!("no GPU expert kernel for {t:?}"),
    })
}

/// Dynamic shared memory the fast kernels need for `t`'s codebook tables (expert_fast.cuh iq_tab), bytes.
pub fn tile_smem(t: GType) -> u32 {
    4 * match t {
        GType::Iq1S => 0, // its grid is read through L1
        GType::Iq2S => 2048 + 256,
        GType::Iq2Xs => 1024 + 256,
        GType::Iq2Xxs => 512 + 256,
        _ => 0,
    }
}

fn is_t16(t: GType) -> bool {
    crate::t16::k2(t).is_some()
}

/// Table / prefill id of a resident type: repacked for the fast kernels (2000 + id; T16: exllamav3's tile order for
/// t16.cuh) unless the store is raw.
fn slot_id(t: GType, fast: bool) -> u32 {
    if fast { 2000 + t as u32 } else { t as u32 }
}

/// T16 decode scratch (t16.cuh t16d_*), for T tokens x K picks
struct T16Bufs {
    xpg: Buf,
    xpu: Buf,
    xpd: Buf,
    xdg: Buf,
    xsg: Buf,
    xdu: Buf,
    xsu: Buf,
    xdd: Buf,
    xsd: Buf,
    pg: Buf,
    pu: Buf,
    pd: Buf,
    h: Buf,
}
/// device pointers of a T16 decode pass's scratch (T16Bufs, or carved from a caller's buffer)
#[derive(Clone, Copy)]
pub struct T16Ptrs {
    xpg: u64,
    xpu: u64,
    xpd: u64,
    xdg: u64,
    xsg: u64,
    xdu: u64,
    xsu: u64,
    xdd: u64,
    xsd: u64,
    pg: u64,
    pu: u64,
    pd: u64,
    h: u64,
}

impl T16Ptrs {
    /// scratch bytes for `picks` (token, expert) picks, each piece 256-byte aligned
    pub fn bytes(picks: usize, dim: usize, inter: usize) -> usize {
        Self::layout(picks, dim, inter).1
    }
    fn layout(p: usize, d: usize, i: usize) -> ([usize; 13], usize) {
        let sizes = [p * d / 2, p * d / 2, p * i / 2, p * d / 256, p * d / 256, p * d / 256, p * d / 256, p * i / 256, p * i / 256, T16_SGU * p * i, T16_SGU * p * i, T16_SD * p * d, p * i];
        let mut offs = [0usize; 13];
        let mut o = 0;
        for (k, n) in sizes.iter().enumerate() {
            offs[k] = o;
            o += (n * 4).div_ceil(256) * 256;
        }
        (offs, o)
    }
    /// carve the scratch for `picks` picks out of [base, base + T16Ptrs::bytes(..))
    pub fn carve(base: u64, picks: usize, dim: usize, inter: usize) -> T16Ptrs {
        let (o, _) = Self::layout(picks, dim, inter);
        let at = |k: usize| base + o[k] as u64;
        T16Ptrs { xpg: at(0), xpu: at(1), xpd: at(2), xdg: at(3), xsg: at(4), xdu: at(5), xsu: at(6), xdd: at(7), xsd: at(8), pg: at(9), pu: at(10), pd: at(11), h: at(12) }
    }
}

impl T16Bufs {
    fn ptrs(&self) -> T16Ptrs {
        T16Ptrs {
            xpg: self.xpg.ptr,
            xpu: self.xpu.ptr,
            xpd: self.xpd.ptr,
            xdg: self.xdg.ptr,
            xsg: self.xsg.ptr,
            xdu: self.xdu.ptr,
            xsu: self.xsu.ptr,
            xdd: self.xdd.ptr,
            xsd: self.xsd.ptr,
            pg: self.pg.ptr,
            pu: self.pu.ptr,
            pd: self.pd.ptr,
            h: self.h.ptr,
        }
    }
}

/// k-splits of the T16 GEMVs: gate / up (dim / 256 = 20 groups), down (inter / 256 = 9 groups)
const T16_SGU: usize = 4;
const T16_SD: usize = 3;

/// Type id of `t` repacked for the fast kernels (expert_fast.cuh), the same bytes per weight.
pub fn repacked_id(t: GType) -> Result<i32> {
    if !matches!(t, GType::Iq2Xxs | GType::Iq1S | GType::Iq2Xs | GType::Iq2S | GType::Q2K | GType::Iq3Xxs) && !is_t16(t) {
        gemv_name(t)?; // IQ2_XXS has only the fast (repacked) kernels
    }
    Ok(2000 + t as u32 as i32)
}

/// Enqueue: dst (a matrix of rows x k, same size as src) = src (raw GGUF blocks of type t) repacked.
pub fn repack(g: &Gpu, t: GType, src: u64, dst: u64, rows: usize, k: usize) -> Result<()> {
    repack_on(g, false, t, src, dst, rows, k)
}

fn repack_on(g: &Gpu, side: bool, t: GType, src: u64, dst: u64, rows: usize, k: usize) -> Result<()> {
    repacked_id(t)?;
    if let Some(k2) = crate::t16::k2(t) {
        // a matrix of `rows` outputs x `k` inputs: stored layout -> exllamav3's order
        let words = crate::t16::matrix_bytes(t, k, rows) / 4;
        let args = [Arg::I(k2 as i32), Arg::P(src), Arg::P(dst), Arg::I(k as i32), Arg::I(rows as i32)];
        let grid = ((words as u32).div_ceil(256), 1, 1);
        return if side { g.launch_side(g.kernel("repack_t16")?, grid, 256, 0, &args) } else { g.launch(g.kernel("repack_t16")?, grid, 256, 0, &args) };
    }
    let blocks = rows * k / t.block().0;
    let args = [Arg::I(t as u32 as i32), Arg::P(src), Arg::P(dst), Arg::I(rows as i32), Arg::I(k as i32)];
    let grid = ((blocks as u32).div_ceil(256), 1, 1);
    if side { g.launch_side(g.kernel("repack_expert")?, grid, 256, 0, &args) } else { g.launch(g.kernel("repack_expert")?, grid, 256, 0, &args) }
}

impl GpuExperts {
    /// Upload `which` (layer, expert) pairs into one device buffer on `g`, repacked for the fast kernels unless
    /// DSX_OLD_MOE is set.
    pub fn new(g: &Gpu, m: &Model, which: &[(usize, usize)]) -> Result<GpuExperts> {
        let c = &m.cfg;
        let cfg = ExpCfg { n_layers: c.n_layers, n_experts: c.n_experts, inter: c.inter, dim: c.dim, lim: c.swiglu_limit, rot: c.expert_rot.iter().enumerate().fold(0u64, |m, (l, &r)| m | ((r as u64) << l)) };
        let items: Vec<((usize, usize), RawExpert)> = which
            .iter()
            .map(|&(l, e)| {
                let x = &m.layers[l].experts[e];
                ((l, e), RawExpert { gate: x.gate.bytes(), up: x.up.bytes(), down: x.down.bytes(), gu: x.gate.ty, dn: x.down.ty })
            })
            .collect();
        Self::build(g, cfg, &items, std::env::var_os("DSX_OLD_MOE").is_none())
    }

    /// Upload experts given as raw GGUF bytes; `fast` = repacked for the fast kernels (else raw, for moe_gu / moe_down).
    pub fn build(g: &Gpu, c: ExpCfg, items: &[((usize, usize), RawExpert)], fast: bool) -> Result<GpuExperts> {
        if fast && (c.dim % 256 != 0 || c.inter % 256 != 0) {
            bail!("fast expert kernels need dim and inter multiples of 256");
        }
        // matrices 256-byte aligned (the fast kernels load 16-byte vectors)
        let al = |n: usize| n.next_multiple_of(256);
        let total: usize = items.iter().map(|(_, x)| al(x.gate.len()) + al(x.up.len()) + al(x.down.len())).sum();
        let buf = g.alloc(total.max(1))?;
        let stage = if fast { Some(g.alloc(items.iter().map(|(_, x)| x.gate.len().max(x.down.len())).max().unwrap_or(1))?) } else { None };
        let mut slots = HashMap::new();
        let mut off = 0u64;
        for ((l, e), x) in items {
            let mut put = |b: &[u8], t: GType, rows: usize, k: usize| -> Result<u64> {
                let p = buf.ptr + off;
                match &stage {
                    Some(s) => {
                        g.htod_sync(s.ptr, b)?;
                        repack(g, t, s.ptr, p, rows, k)?;
                    }
                    None => g.htod_sync(p, b)?,
                }
                off += al(b.len()) as u64;
                Ok(p)
            };
            for t in [x.gu, x.dn] {
                anyhow::ensure!(fast || !matches!(t, GType::Q2K | GType::Iq3Xxs), "Q2_K / IQ3_XXS experts require the fast store (unset DSX_OLD_MOE)");
                // a kernel exists for the layout; raw IQ stores (DSX_OLD_MOE) only have prefill's dequant_f16 so far
                if is_t16(t) {
                    anyhow::ensure!(fast, "T16 experts need the fast store (unset DSX_OLD_MOE)");
                } else if fast {
                    repacked_id(t)?;
                } else if !matches!(t, GType::Iq1S | GType::Iq2Xs | GType::Iq2S | GType::Iq2Xxs) {
                    gemv_name(t)?;
                }
            }
            let (gp, up, dp) = (put(x.gate, x.gu, c.inter, c.dim)?, put(x.up, x.gu, c.inter, c.dim)?, put(x.down, x.dn, c.dim, c.inter)?);
            slots.insert((*l, *e), Slot { gate: gp, up, down: dp, gu: x.gu, dn: x.dn });
        }
        g.sync()?;
        drop(stage);
        let id = |t: GType| slot_id(t, fast);
        // ExpSlot {u64 gate, up, down; i32 gu, dn} = 32 bytes, offsets relative to the store base
        let mut tab = vec![0u8; c.n_layers * c.n_experts * 32];
        for l in 0..c.n_layers {
            for e in 0..c.n_experts {
                let o = (l * c.n_experts + e) * 32;
                match slots.get(&(l, e)) {
                    Some(sl) => {
                        tab[o..o + 8].copy_from_slice(&(sl.gate - buf.ptr).to_le_bytes());
                        tab[o + 8..o + 16].copy_from_slice(&(sl.up - buf.ptr).to_le_bytes());
                        tab[o + 16..o + 24].copy_from_slice(&(sl.down - buf.ptr).to_le_bytes());
                        tab[o + 24..o + 28].copy_from_slice(&id(sl.gu).to_le_bytes());
                        tab[o + 28..o + 32].copy_from_slice(&id(sl.dn).to_le_bytes());
                    }
                    None => tab[o..o + 8].copy_from_slice(&u64::MAX.to_le_bytes()),
                }
            }
        }
        let tab = g.upload(&tab)?;
        let mut k = HashMap::new();
        for n in ["gemv_q2_0", "gemv_ptq1_0", "gemv_q3_k", "gemv_q8_0", "swiglu_s", "add_inplace", "moe_gu", "moe_down", "quant_act", "moe_gu_fast", "moe_down_fast"] {
            k.insert(n, g.kernel(n)?);
        }
        let mut t16 = vec![None; c.n_layers];
        for ((l, _), x) in items {
            if let (Some(a), Some(b)) = (crate::t16::k2(x.gu), crate::t16::k2(x.dn)) {
                anyhow::ensure!(t16[*l].is_none_or(|v| v == (a, b)), "layer {l}: T16 experts of different K");
                t16[*l] = Some((a, b));
            } else {
                anyhow::ensure!(!is_t16(x.gu) && !is_t16(x.dn), "layer {l}: T16 mixed with other expert types");
            }
        }
        let t16b = if t16.iter().any(|v| v.is_some()) {
            for n in ["t16d_prep", "t16d_gemv_3", "t16d_gemv_4", "t16d_gemv_5", "t16d_gemv0_3", "t16d_gemv0_4", "t16d_gemv0_5", "t16d_post_gu", "t16d_post_down"] {
                k.insert(n, g.kernel(n)?);
            }
            let (p, d, i) = (MAXT * MAXK, c.dim, c.inter);
            let f4 = |n: usize| g.alloc_zeroed(n * 4);
            Some(T16Bufs {
                xpg: f4(p * d / 2)?,
                xpu: f4(p * d / 2)?,
                xpd: f4(p * i / 2)?,
                xdg: f4(p * d / 256)?,
                xsg: f4(p * d / 256)?,
                xdu: f4(p * d / 256)?,
                xsu: f4(p * d / 256)?,
                xdd: f4(p * i / 256)?,
                xsd: f4(p * i / 256)?,
                pg: f4(T16_SGU * p * i)?,
                pu: f4(T16_SGU * p * i)?,
                pd: f4(T16_SD * p * d)?,
                h: f4(p * i)?,
            })
        } else {
            None
        };
        let f = |n: usize| g.alloc_zeroed(n * 4);
        let fast = if fast {
            let (nx, nh) = (MAXT * c.dim, MAXT * MAXK * c.inter);
            Some(Fast {
                xq: g.alloc_zeroed(nx)?,
                xd: f(nx / 256)?,
                xs: g.alloc_zeroed(nx / 64 * 2)?,
                h: f(nh)?,
                hq: g.alloc_zeroed(nh)?,
                hd: f(nh / 256)?,
                hs: g.alloc_zeroed(nh / 64 * 2)?,
                plan: f(1 + MAXT * MAXK * 10)?,
                part: f(MAXT * MAXK * 8 * c.dim)?,
                cnt: f(c.dim.div_ceil(16))?,
            })
        } else {
            None
        };
        let smem = slots.values().flat_map(|s| [tile_smem(s.gu), tile_smem(s.dn)]).max().unwrap_or(0);
        Ok(GpuExperts {
            buf,
            tab,
            sel: g.alloc_zeroed(4 * MAXT * MAXK)?,
            wts: g.alloc_zeroed(4 * MAXT * MAXK)?,
            h: f(16 * c.inter)?,
            n_experts: c.n_experts,
            slots,
            x: f(MAXT * c.dim)?,
            out: f(MAXT * c.dim)?,
            tmp_g: f(c.inter)?,
            tmp_u: f(c.inter)?,
            tmp_h: f(c.inter)?,
            tmp_o: f(c.dim)?,
            k,
            inter: c.inter,
            dim: c.dim,
            lim: c.lim,
            rot: c.rot,
            bytes: total,
            smem,
            fast,
            swap: None,
            t16,
            t16b,
        })
    }

    /// Allow up to `n` in-flight replacements (fast layout only), each staging at most `max_raw` bytes.
    pub fn enable_swaps(&mut self, g: &Gpu, n: usize, max_raw: usize) -> Result<()> {
        if self.fast.is_none() || n == 0 {
            return Ok(());
        }
        let stage = (0..n).map(|_| Ok((g.alloc(max_raw)?, g.event()?))).collect::<Result<Vec<_>>>()?;
        self.swap = Some(Swapper { stage, ring: crate::gpu::HostBuf::new(32 * 1024)?, ring_i: 0, free: (0..n).rev().collect(), pending: vec![], after_main: g.event()? });
        Ok(())
    }

    /// (gate/up type, down type) of a resident expert
    pub fn types(&self, l: usize, e: usize) -> Option<(GType, GType)> {
        self.slots.get(&(l, e)).map(|s| (s.gu, s.dn))
    }

    pub fn residents(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.slots.keys().copied()
    }

    /// replacements started and not committed yet
    pub fn pending(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.swap.iter().flat_map(|s| s.pending.iter().map(|p| (p.l, p.e)))
    }

    /// free staging buffers (how many replacements can start now)
    pub fn swap_room(&self) -> usize {
        self.swap.as_ref().map_or(0, |s| s.free.len())
    }

    /// ExpSlot entry of `sl` (None = not resident) written to the device table, ordered on the main stream
    fn write_entry(&mut self, g: &Gpu, l: usize, e: usize, sl: Option<&Slot>) -> Result<()> {
        let mut b = [0u8; 32];
        match sl {
            Some(sl) => {
                let fast = self.fast.is_some();
                let id = |t: GType| slot_id(t, fast);
                b[0..8].copy_from_slice(&(sl.gate - self.buf.ptr).to_le_bytes());
                b[8..16].copy_from_slice(&(sl.up - self.buf.ptr).to_le_bytes());
                b[16..24].copy_from_slice(&(sl.down - self.buf.ptr).to_le_bytes());
                b[24..28].copy_from_slice(&id(sl.gu).to_le_bytes());
                b[28..32].copy_from_slice(&id(sl.dn).to_le_bytes());
            }
            None => b[0..8].copy_from_slice(&u64::MAX.to_le_bytes()),
        }
        let dst = self.tab.ptr + ((l * self.n_experts + e) * 32) as u64;
        match self.swap.as_mut() {
            Some(sw) => {
                let i = sw.ring_i % 1024;
                sw.ring_i += 1;
                let r = &mut sw.ring.as_mut_slice::<u8>()[i * 32..i * 32 + 32];
                r.copy_from_slice(&b);
                g.htod(dst, r)
            }
            None => g.htod_sync(dst, &b),
        }
    }

    /// Drop `victim` and start loading `new` (raw GGUF matrices `raw`, same types as the victim) into its slot; `new` is
    /// resident after commit_ready. May run while the main stream still has the current token queued: the table write
    /// follows that work on the main stream, and the copies into the slot wait for it on the side stream.
    pub fn begin_swap(&mut self, g: &Gpu, victim: (usize, usize), new: (usize, usize), raw: [&[u8]; 3]) -> Result<bool> {
        let (Some(sw), Some(vs)) = (self.swap.as_mut(), self.slots.get(&victim)) else { return Ok(false) };
        let Some(si) = sw.free.pop() else { return Ok(false) };
        let (stage, ev, after_main) = (sw.stage[si].0.ptr, sw.stage[si].1, sw.after_main);
        let (gu, dn, inter, dim) = (vs.gu, vs.dn, self.inter, self.dim);
        let slot = Slot { gate: vs.gate, up: vs.up, down: vs.down, gu, dn };
        let _ = sw;
        self.slots.remove(&victim);
        self.write_entry(g, victim.0, victim.1, None)?;
        let (og, ou) = (raw[0].len() as u64, (raw[0].len() + raw[1].len()) as u64);
        g.record(after_main, false)?;
        g.wait(after_main, true)?;
        g.htod_side(stage, raw[0])?;
        g.htod_side(stage + og, raw[1])?;
        g.htod_side(stage + ou, raw[2])?;
        repack_on(g, true, gu, stage, slot.gate, inter, dim)?;
        repack_on(g, true, gu, stage + og, slot.up, inter, dim)?;
        repack_on(g, true, dn, stage + ou, slot.down, dim, inter)?;
        g.record(ev, true)?;
        self.swap.as_mut().unwrap().pending.push(Pending { l: new.0, e: new.1, slot, stage: si });
        Ok(true)
    }

    /// Make finished replacements resident (`all`: wait for every one). Returns the experts committed.
    pub fn commit_ready(&mut self, g: &Gpu, all: bool) -> Result<Vec<(usize, usize)>> {
        let Some(sw) = self.swap.as_mut() else { return Ok(vec![]) };
        if all && !sw.pending.is_empty() {
            g.side_sync()?;
        }
        let mut done = vec![];
        let mut i = 0;
        while i < sw.pending.len() {
            if g.done(sw.stage[sw.pending[i].stage].1)? {
                done.push(sw.pending.swap_remove(i));
            } else {
                i += 1;
            }
        }
        let mut out = vec![];
        for p in done {
            self.write_entry(g, p.l, p.e, Some(&p.slot))?;
            self.slots.insert((p.l, p.e), p.slot);
            self.swap.as_mut().unwrap().free.push(p.stage);
            out.push((p.l, p.e));
        }
        Ok(out)
    }

    /// device pointers (gate, up, down) of a resident expert and the type ids of gate/up and down as dequant_f16
    /// takes them (ggml id, or 2000 + id when repacked)
    pub fn slot_ptrs(&self, l: usize, e: usize) -> Option<(u64, u64, u64, i32, i32)> {
        let fast = self.fast.is_some();
        self.slots.get(&(l, e)).map(|s| (s.gate, s.up, s.down, slot_id(s.gu, fast) as i32, slot_id(s.dn, fast) as i32))
    }

    pub fn has(&self, l: usize, e: usize) -> bool {
        self.slots.contains_key(&(l, e))
    }

    /// true when the weights are in the fast-kernel layout
    pub fn is_fast(&self) -> bool {
        self.fast.is_some()
    }

    fn gemv(&self, g: &Gpu, ty: GType, w: u64, x: u64, y: u64, rows: usize, cols: usize) -> Result<()> {
        g.launch(
            self.k[gemv_name(ty)?],
            ((rows as u32).div_ceil(4), 1, 1),
            128,
            0,
            &[Arg::P(w), Arg::P(x), Arg::P(y), Arg::I(rows as i32), Arg::I(cols as i32), Arg::I(1), Arg::I(cols as i32), Arg::I(rows as i32)],
        )
    }

    /// Enqueue on g's stream: out (if `accumulate` is false it is zeroed first) += experts of layer `l`, input `x`
    /// (a device pointer on g).
    pub fn run(&self, g: &Gpu, l: usize, list: &[(usize, f32)], x: u64, out: u64) -> Result<()> {
        if self.fast.is_some() || std::env::var_os("DSX_MOE_DEVICE").is_some() {
            let sel: Vec<i32> = list.iter().map(|&(e, _)| e as i32).collect();
            let w: Vec<f32> = list.iter().map(|&(_, w)| w).collect();
            g.htod_sync(self.sel.ptr, &sel)?;
            g.htod_sync(self.wts.ptr, &w)?;
            return self.run_device(g, l, self.sel.ptr, self.wts.ptr, list.len(), x, out);
        }
        g.memset(out, self.dim * 4)?;
        for &(e, w) in list {
            let s = &self.slots[&(l, e)];
            self.gemv(g, s.gu, s.gate, x, self.tmp_g.ptr, self.inter, self.dim)?;
            self.gemv(g, s.gu, s.up, x, self.tmp_u.ptr, self.inter, self.dim)?;
            g.launch(
                self.k["swiglu_s"],
                ((self.inter as u32).div_ceil(256), 1, 1),
                256,
                0,
                &[Arg::P(self.tmp_g.ptr), Arg::P(self.tmp_u.ptr), Arg::P(self.tmp_h.ptr), Arg::I(self.inter as i32), Arg::F(self.lim), Arg::F(w)],
            )?;
            self.gemv(g, s.dn, s.down, self.tmp_h.ptr, self.tmp_o.ptr, self.dim, self.inter)?;
            g.launch(self.k["add_inplace"], ((self.dim as u32).div_ceil(256), 1, 1), 256, 0, &[Arg::P(out), Arg::P(self.tmp_o.ptr), Arg::I(self.dim as i32)])?;
        }
        Ok(())
    }
}

impl GpuExperts {
    /// Enqueue the resident picks of layer `l` (sel/wts device pointers, K picks) for one token: out = their
    /// weighted sum. Fast kernels when the store is repacked, else moe_gu / moe_down.
    pub fn run_device(&self, g: &Gpu, l: usize, sel: u64, wts: u64, k: usize, x: u64, out: u64) -> Result<()> {
        anyhow::ensure!(self.fast.is_some() || (self.rot >> l) & 1 == 0, "rotated experts need the repacked store (unset DSX_OLD_MOE)");
        if self.fast.is_some() {
            return self.run_device_fast(g, l, sel, wts, k, 1, x, out);
        }
        g.launch(
            self.k["moe_gu"],
            ((self.inter as u32).div_ceil(4), k as u32, 1),
            128,
            0,
            &[Arg::P(self.tab.ptr), Arg::I(l as i32), Arg::I(self.n_experts as i32), Arg::P(sel), Arg::P(wts), Arg::I(k as i32), Arg::P(self.buf.ptr), Arg::P(x), Arg::P(self.h.ptr), Arg::I(self.inter as i32), Arg::I(self.dim as i32), Arg::F(self.lim)],
        )?;
        g.launch(
            self.k["moe_down"],
            ((self.dim as u32).div_ceil(4), 1, 1),
            128,
            0,
            &[Arg::P(self.tab.ptr), Arg::I(l as i32), Arg::I(self.n_experts as i32), Arg::P(sel), Arg::I(k as i32), Arg::P(self.buf.ptr), Arg::P(self.h.ptr), Arg::P(out), Arg::I(self.inter as i32), Arg::I(self.dim as i32)],
        )
    }

    /// Enqueue T tokens' resident picks of layer `l` on the repacked store: sel/wts [T][K] and x [T][dim] device
    /// pointers, out [T][dim] = per token the weighted sum of its resident picks (0 if none). An expert picked by
    /// several tokens is read once. Four launches: quantize x (+ plan), gate/up, quantize h, down.
    #[allow(clippy::too_many_arguments)]
    pub fn run_device_fast(&self, g: &Gpu, l: usize, sel: u64, wts: u64, k: usize, t: usize, x: u64, out: u64) -> Result<()> {
        let Some(f) = self.fast.as_ref() else { bail!("expert store is not repacked (DSX_OLD_MOE)") };
        if t == 0 || t > MAXT || k == 0 || k > MAXK {
            bail!("run_device_fast: T = {t}, K = {k} (max {MAXT}, {MAXK})");
        }
        if let Some((kg, kd)) = self.t16[l] {
            return self.run_t16(g, l, sel, wts, k, t, x, out, kg, kd);
        }
        let (dim, inter) = (self.dim, self.inter);
        let (tab, e) = (Arg::P(self.tab.ptr), Arg::I(self.n_experts as i32));
        let rot = ((self.rot >> l) & 1) as i32;
        g.launch(
            self.k["quant_act"],
            ((dim / 256) as u32, t as u32, 1),
            256,
            0,
            &[Arg::P(x), Arg::I(dim as i32), Arg::P(f.xq.ptr), Arg::P(f.xd.ptr), Arg::P(f.xs.ptr), Arg::P(sel), tab, Arg::I(l as i32), e, Arg::I(k as i32), Arg::I(t as i32), Arg::P(f.plan.ptr), Arg::I(rot)],
        )?;
        g.launch(
            self.k["moe_gu_fast"],
            ((inter as u32).div_ceil(16 * ROWT), (t * k) as u32, 1),
            32 * KS,
            self.smem,
            &[tab, Arg::I(l as i32), e, Arg::P(f.plan.ptr), Arg::P(wts), Arg::I(k as i32), Arg::P(self.buf.ptr), Arg::P(f.xq.ptr), Arg::P(f.xd.ptr), Arg::P(f.xs.ptr), Arg::P(f.h.ptr), Arg::I(inter as i32), Arg::I(dim as i32), Arg::F(self.lim)],
        )?;
        g.launch(
            self.k["quant_act"],
            ((inter / 256) as u32, (t * k) as u32, 1),
            256,
            0,
            &[Arg::P(f.h.ptr), Arg::I(inter as i32), Arg::P(f.hq.ptr), Arg::P(f.hd.ptr), Arg::P(f.hs.ptr), Arg::P(0), tab, Arg::I(0), e, Arg::I(0), Arg::I(0), Arg::P(0), Arg::I(rot)],
        )?;
        g.launch(
            self.k["moe_down_fast"],
            ((dim as u32).div_ceil(16 * ROWT), (t * k) as u32, 1),
            32 * KS,
            self.smem,
            &[tab, Arg::I(l as i32), e, Arg::P(f.plan.ptr), Arg::I(k as i32), Arg::P(self.buf.ptr), Arg::P(f.hq.ptr), Arg::P(f.hd.ptr), Arg::P(f.hs.ptr), Arg::P(out), Arg::I(inter as i32), Arg::I(dim as i32), Arg::I(t as i32), Arg::P(f.part.ptr), Arg::P(f.cnt.ptr)],
        )
    }
}

impl GpuExperts {
    /// T16 layer (t16.cuh t16d_*): per pick, prepare x for gate and up, two GEMVs, swiglu, prepare h, the down
    /// GEMV, then each token's sum. Eight launches, no host round trip.
    #[allow(clippy::too_many_arguments)]
    fn run_t16(&self, g: &Gpu, l: usize, sel: u64, wts: u64, k: usize, t: usize, x: u64, out: u64, kg: usize, kd: usize) -> Result<()> {
        let b = self.t16b.as_ref().unwrap().ptrs();
        self.run_t16_on(g, l, sel, wts, k, t, x, out, kg, kd, &b)
    }

    /// whether layer `l`'s experts here are T16 (run_t16_ext applies)
    pub fn is_t16_layer(&self, l: usize) -> bool {
        self.t16.get(l).is_some_and(|v| v.is_some())
    }

    /// The T16 decode kernels over `t` tokens x `k` picks with any t, on scratch carved from `scratch`
    /// (T16Ptrs::bytes(t * k, ..) bytes): every pick whose expert is resident here, out [t][dim] = each token's sum
    /// (written, not added; 0 for tokens without one). Prefill uses it for resident experts with few tokens.
    #[allow(clippy::too_many_arguments)]
    pub fn run_t16_ext(&self, g: &Gpu, l: usize, sel: u64, wts: u64, k: usize, t: usize, x: u64, out: u64, scratch: u64) -> Result<()> {
        let Some((kg, kd)) = self.t16[l] else { bail!("run_t16_ext: layer {l} is not T16") };
        let b = T16Ptrs::carve(scratch, t * k, self.dim, self.inter);
        self.run_t16_on(g, l, sel, wts, k, t, x, out, kg, kd, &b)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_t16_on(&self, g: &Gpu, l: usize, sel: u64, wts: u64, k: usize, t: usize, x: u64, out: u64, kg: usize, kd: usize, b: &T16Ptrs) -> Result<()> {
        let (dim, inter, picks) = (self.dim, self.inter, t * k);
        let (tab, base, e) = (Arg::P(self.tab.ptr), Arg::P(self.buf.ptr), Arg::I(self.n_experts as i32));
        let (li, ki, ti) = (Arg::I(l as i32), Arg::I(k as i32), Arg::I(t as i32));
        let blocks = |warps: usize| ((warps as u32).div_ceil(4), 1, 1);
        // DSX_T16_GEMV=0: the GEMV before cp.async staging (same launch and output bits; for A/Bs)
        let old = std::env::var("DSX_T16_GEMV").is_ok_and(|v| v == "0");
        let gemv = |k2: usize| self.k[if old { ["t16d_gemv0_3", "t16d_gemv0_4", "t16d_gemv0_5"] } else { ["t16d_gemv_3", "t16d_gemv_4", "t16d_gemv_5"] }[k2 - 3]];
        for (role, xp, xd, xs) in [(0, b.xpg, b.xdg, b.xsg), (1, b.xpu, b.xdu, b.xsu)] {
            g.launch(self.k["t16d_prep"], blocks(picks * dim / 256), 128, 0, &[Arg::P(x), Arg::I(0), Arg::P(sel), tab, base, li, e, ki, ti, Arg::I(role), Arg::I(dim as i32), Arg::I(inter as i32), Arg::I(kg as i32), Arg::P(xp), Arg::P(xd), Arg::P(xs)])?;
        }
        for (role, xp, xd, xs, part) in [(0, b.xpg, b.xdg, b.xsg, b.pg), (1, b.xpu, b.xdu, b.xsu, b.pu)] {
            g.launch(gemv(kg), blocks(picks * inter / 32 * T16_SGU), 128, 0, &[Arg::P(xp), Arg::P(xd), Arg::P(xs), Arg::P(sel), tab, base, li, e, ki, ti, Arg::I(role), Arg::I(dim as i32), Arg::I(inter as i32), Arg::I(T16_SGU as i32), Arg::P(part)])?;
        }
        g.launch(self.k["t16d_post_gu"], blocks(picks * inter / 128), 128, 0, &[Arg::P(b.pg), Arg::P(b.pu), Arg::I(T16_SGU as i32), Arg::P(sel), Arg::P(wts), tab, base, li, e, ki, ti, Arg::I(dim as i32), Arg::I(inter as i32), Arg::I(kg as i32), Arg::F(self.lim), Arg::P(b.h)])?;
        g.launch(self.k["t16d_prep"], blocks(picks * inter / 256), 128, 0, &[Arg::P(b.h), Arg::I(1), Arg::P(sel), tab, base, li, e, ki, ti, Arg::I(2), Arg::I(inter as i32), Arg::I(dim as i32), Arg::I(kd as i32), Arg::P(b.xpd), Arg::P(b.xdd), Arg::P(b.xsd)])?;
        g.launch(gemv(kd), blocks(picks * dim / 32 * T16_SD), 128, 0, &[Arg::P(b.xpd), Arg::P(b.xdd), Arg::P(b.xsd), Arg::P(sel), tab, base, li, e, ki, ti, Arg::I(2), Arg::I(inter as i32), Arg::I(dim as i32), Arg::I(T16_SD as i32), Arg::P(b.pd)])?;
        g.launch(self.k["t16d_post_down"], blocks(t * dim / 128), 128, 0, &[Arg::P(b.pd), Arg::I(T16_SD as i32), Arg::P(sel), tab, base, li, e, ki, ti, Arg::I(inter as i32), Arg::I(dim as i32), Arg::I(kd as i32), Arg::P(out)])
    }
}

/// Spread experts over the stores (`caps[i]` bytes each), hottest first (a ranking from routing statistics when
/// given, else layer-major round robin from expert 0). Stores with a lower `group` fill first (static stores before
/// adaptive ones, which reshuffle at run time anyway); within a group each expert goes to the least-filled store
/// that has room, so the hot experts, and the decode work, spread evenly over GPUs that run in parallel.
pub fn place(m: &Model, rank: Option<&[(usize, usize)]>, caps: &[usize], group: &[u8]) -> Vec<Vec<(usize, usize)>> {
    let c = &m.cfg;
    let order: Vec<(usize, usize)> = match rank {
        Some(r) => r.to_vec(),
        None => (0..c.n_experts).flat_map(|e| (0..c.n_layers).map(move |l| (l, e))).collect(),
    };
    let size = |l: usize, e: usize| {
        let x = &m.layers[l].experts[e];
        x.gate.bytes().len() + x.up.bytes().len() + x.down.bytes().len()
    };
    let mut out = vec![vec![]; caps.len()];
    for ((l, e), st) in order.iter().zip(assign(order.iter().map(|&(l, e)| size(l, e)), caps, group)) {
        if let Some(i) = st {
            out[i].push((*l, *e));
        }
    }
    out
}

/// place's fill rule on sizes: the store each item goes to (None = no room anywhere)
fn assign(sizes: impl Iterator<Item = usize>, caps: &[usize], group: &[u8]) -> Vec<Option<usize>> {
    let mut used = vec![0usize; caps.len()];
    sizes
        .map(|s| {
            let fill = |i: usize| used[i] as f64 / caps[i] as f64;
            let pick = (0..caps.len()).filter(|&i| used[i] + s <= caps[i]).min_by(|&i, &j| group[i].cmp(&group[j]).then(fill(i).total_cmp(&fill(j))));
            if let Some(i) = pick {
                used[i] += s;
            }
            pick
        })
        .collect()
}

#[cfg(test)]
mod place_tests {
    #[test]
    fn static_first_then_balanced() {
        // two static aux stores of 2 and one adaptive main store of 3; 8 unit experts, the last one has no room
        let o = super::assign([1usize; 8].into_iter(), &[2, 2, 3], &[0, 0, 1]);
        assert_eq!(o, vec![Some(0), Some(1), Some(0), Some(1), Some(2), Some(2), Some(2), None]);
    }

    #[test]
    fn two_store_order_unchanged() {
        // the old two-store rule: GPU1 (static) first, GPU0 takes what does not fit, smaller ones still go to GPU1
        let o = super::assign([3usize, 3, 1].into_iter(), &[4, 10], &[0, 1]);
        assert_eq!(o, vec![Some(0), Some(1), Some(0)]);
    }
}

/// Experts copied out of the mmapped file into one hugepage-backed, CUDA-registered host buffer: the CPU reads them
/// at RAM speed and prefill DMAs them to GPU0 at full PCIe speed (24.7 GB/s on .102 vs 13 from 4 KB pages).
pub struct Arena {
    pub buf: crate::gpu::HostBuf,
    /// (layer, expert) -> (gate, up, down) byte ranges in buf
    pub slots: HashMap<(usize, usize), [(usize, usize); 3]>,
    pub bytes: usize,
}

const DIO_ALIGN: usize = 4096;

/// A page-aligned heap buffer (O_DIRECT needs aligned destinations).
struct AlignedBuf {
    ptr: *mut u8,
    len: usize,
}

unsafe impl Send for AlignedBuf {}

impl AlignedBuf {
    fn new(len: usize) -> AlignedBuf {
        let len = len.div_ceil(DIO_ALIGN) * DIO_ALIGN;
        let ptr = unsafe { std::alloc::alloc(std::alloc::Layout::from_size_align(len, DIO_ALIGN).unwrap()) };
        assert!(!ptr.is_null(), "alloc of {len} bytes failed");
        AlignedBuf { ptr, len }
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        unsafe { std::alloc::dealloc(self.ptr, std::alloc::Layout::from_size_align(self.len, DIO_ALIGN).unwrap()) };
    }
}

/// Reads byte ranges of a mapped GGUF with O_DIRECT, bypassing the page cache: the range (a slice of one of the
/// file's mmaps) is widened to 4 KiB boundaries, read into a bounce buffer and copied out.
struct DirectReader {
    files: Vec<(usize, usize, std::fs::File)>, // (map start, map len, file opened with O_DIRECT)
}

impl DirectReader {
    fn open(g: &crate::gguf::Gguf) -> Result<DirectReader> {
        use std::os::unix::fs::OpenOptionsExt;
        const O_DIRECT: i32 = 0o40000; // x86_64 Linux
        let mut files = vec![];
        for (p, map) in g.paths.iter().zip(&g.maps) {
            let f = std::fs::OpenOptions::new().read(true).custom_flags(O_DIRECT).open(p)?;
            files.push((map.as_ptr() as usize, map.len(), f));
        }
        Ok(DirectReader { files })
    }

    fn read(&self, src: &[u8], dst: &mut [u8], bounce: &mut AlignedBuf) -> std::io::Result<()> {
        use std::os::unix::fs::FileExt;
        let p = src.as_ptr() as usize;
        let Some((start, len, f)) = self.files.iter().find(|(s, n, _)| p >= *s && p + src.len() <= s + n) else {
            return Err(std::io::Error::other("range not in the model's files"));
        };
        let off = p - start;
        let a0 = off / DIO_ALIGN * DIO_ALIGN;
        let a1 = ((off + src.len()).div_ceil(DIO_ALIGN) * DIO_ALIGN).min(len.div_ceil(DIO_ALIGN) * DIO_ALIGN);
        assert!(a1 - a0 <= bounce.len);
        let b = unsafe { std::slice::from_raw_parts_mut(bounce.ptr, a1 - a0) };
        let mut done = 0;
        while done < off + src.len() - a0 {
            let n = f.read_at(&mut b[done..], (a0 + done) as u64)?;
            if n == 0 {
                return Err(std::io::Error::from(std::io::ErrorKind::UnexpectedEof));
            }
            done += n;
        }
        dst.copy_from_slice(&b[off - a0..off - a0 + src.len()]);
        Ok(())
    }
}

impl Arena {
    /// Copy `which` experts (in that order) until `cap` bytes; returns the arena and the experts that did not fit.
    pub fn new(m: &Model, which: &[(usize, usize)], cap: usize) -> Result<(Arena, Vec<(usize, usize)>)> {
        let size = |l: usize, e: usize| {
            let x = &m.layers[l].experts[e];
            x.gate.bytes().len() + x.up.bytes().len() + x.down.bytes().len()
        };
        let mut take = vec![];
        let mut rest = vec![];
        let mut tot = 0usize;
        let piece = crate::gpu::HostBuf::PIECE;
        for &(l, e) in which {
            let s = size(l, e);
            // no expert crosses a registration piece of the buffer (copies of it would fail)
            let off = if tot / piece != (tot + s - 1) / piece { (tot / piece + 1) * piece } else { tot };
            if off + s <= cap {
                take.push((l, e, off));
                tot = off + s;
            } else {
                rest.push((l, e));
            }
        }
        let t0 = std::time::Instant::now();
        let buf = crate::gpu::HostBuf::new(tot.max(1))?;
        eprintln!("arena: {:.1} GB allocated and pinned in {:.1} s", tot as f64 / 1e9, t0.elapsed().as_secs_f64());
        let base = buf.host_ptr() as usize;
        let mut slots = HashMap::new();
        for &(l, e, off) in &take {
            let x = &m.layers[l].experts[e];
            let (a, b, c) = (x.gate.bytes().len(), x.up.bytes().len(), x.down.bytes().len());
            slots.insert((l, e), [(off, a), (off + a, b), (off + a + b, c)]);
        }
        // O_DIRECT reads from the GGUF files: 5.9 GB/s on .102's NVMe for expert-sized pieces vs 3.9-4.6 through the
        // mmap, and no page cache to reclaim while the arena fills most of the server's memory cgroup (the mmap copy
        // ran at ~1 GB/s there: 107 GB in 76-121 s). Falls back to the mmap copy (e.g. a file system without O_DIRECT).
        let direct = match &m.src {
            crate::model::Source::Gguf(g) => DirectReader::open(g).ok(),
            _ => None,
        };
        use rayon::prelude::*;
        let max_mat = take.iter().map(|&(l, e, _)| size(l, e)).max().unwrap_or(0);
        let failed = std::sync::atomic::AtomicBool::new(false);
        take.par_iter().for_each_init(
            || AlignedBuf::new(max_mat + 2 * DIO_ALIGN),
            |bounce, &(l, e, off)| {
                let x = &m.layers[l].experts[e];
                let mut o = off;
                for mm in [&x.gate, &x.up, &x.down] {
                    let src = mm.bytes();
                    let dst = unsafe { std::slice::from_raw_parts_mut((base + o) as *mut u8, src.len()) };
                    if !direct.as_ref().is_some_and(|d| d.read(src, dst, bounce).is_ok()) {
                        failed.store(true, std::sync::atomic::Ordering::Relaxed);
                        dst.copy_from_slice(src);
                    }
                    o += src.len();
                }
            },
        );
        if direct.is_none() || failed.into_inner() {
            eprintln!("arena: O_DIRECT reads unavailable for some experts, copied them through the mmap");
        }
        if std::env::var_os("DSX_ARENA_VERIFY").is_some() {
            // compare every arena byte with the mmap (slow: reads the experts again through the page cache)
            let bad = take.par_iter().filter(|&&(l, e, off)| {
                let x = &m.layers[l].experts[e];
                let mut o = off;
                [&x.gate, &x.up, &x.down].iter().any(|mm| {
                    let src = mm.bytes();
                    let d = unsafe { std::slice::from_raw_parts((base + o) as *const u8, src.len()) };
                    o += src.len();
                    d != src
                })
            }).count();
            eprintln!("arena verify: {bad} of {} experts differ from the file", take.len());
        }
        Ok((Arena { buf, slots, bytes: tot }, rest))
    }

    pub fn get(&self, l: usize, e: usize) -> Option<[&[u8]; 3]> {
        let s = self.slots.get(&(l, e))?;
        let b = self.buf.as_slice::<u8>();
        Some([&b[s[0].0..s[0].0 + s[0].1], &b[s[1].0..s[1].0 + s[1].1], &b[s[2].0..s[2].0 + s[2].1]])
    }
}
