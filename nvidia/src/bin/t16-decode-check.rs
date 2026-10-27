//! T3 (SM70 port) decode-path check: the T16 decode chain t16d_prep -> t16d_gemv_4 -> t16d_post_gu /
//! t16d_post_down against the CPU reference (cpu::t16k: prep / gemv / post_block), on synthetic experts whose
//! trellis is generated in the stored layout and repacked on the host into the GPU store's layout. The host repack
//! is checked bit-exactly against the repack_t16 kernel first: the CPU reference walks the stored byte layout while
//! the GPU expert store is repacked, so a layout mixup would masquerade as an arch-guard failure. Does not load a
//! model; kernels are launched exactly as run_t16_on (experts.rs) launches them.
//!
//! Gates: the GEMV `part` floats agree with the CPU reference at rel-tol 1e-6 (integer dp4a accumulators identical;
//! only float affine ordering differs). post_gu's h additionally rounds to bf16 and post_down sums several picks'
//! parts, so those gates are 1e-2 and the part-level checks carry the correctness.
use anyhow::{Result, ensure};
use dsx::cpu::t16k::{self, Act};
use dsx::experts::{T16_SD, T16_SGU};
use dsx::gpu::{Arg, Buf, Gpu, Kernel};
use dsx::t16;
use half::f16;
use serde_json::json;
use std::collections::HashMap;

fn rng(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}

/// one random T16 matrix in the stored layout (trellis | suh | svh), the generation style of cpu/t16k.rs's tests
fn gen_matrix(s: &mut u64, k2: usize, k: usize, n: usize) -> Vec<u8> {
    let mut m: Vec<u8> = (0..t16::trellis_bytes(k2, k, n)).map(|_| rng(s) as u8).collect();
    let tb = m.len();
    m.resize(tb + 2 * k + 2 * n, 0);
    for c in m[tb..].chunks_exact_mut(2) {
        let v = 0.02 * (0.5 + (rng(s) % 1000) as f32 / 1000.0) * if rng(s) % 2 == 0 { 1.0 } else { -1.0 };
        c.copy_from_slice(&f16::from_f32(v).to_le_bytes());
    }
    m
}

/// stored -> GPU (repacked) trellis layout, the host twin of the repack_t16 kernel: tile (kb, nb) moves to
/// kb * n/16 + nb and every 4 stream-order bytes become one big-endian u32 (little-endian words read MSB first);
/// suh / svh are copied verbatim.
fn repack_host(m: &[u8], k2: usize, k: usize, n: usize) -> Vec<u8> {
    let (tb, kt, nt) = (t16::tile_bytes(k2), k / 16, n / 16);
    let mut out = m.to_vec();
    for tile in 0..kt * nt {
        let (kb, nb) = (tile / nt, tile % nt);
        for w in 0..tb / 4 {
            let s = (nb * kt + kb) * tb + 4 * w;
            let v = (m[s] as u32) << 24 | (m[s + 1] as u32) << 16 | (m[s + 2] as u32) << 8 | m[s + 3] as u32;
            out[tile * tb + 4 * w..tile * tb + 4 * w + 4].copy_from_slice(&v.to_le_bytes());
        }
    }
    out
}

/// max |a - b| / rms(a): the repo's rel-error convention (scale-robust for zero-crossing outputs)
fn rel(cpu: &[f32], gpu: &[f32]) -> f64 {
    let rms = (cpu.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / cpu.len() as f64).sqrt();
    let max = cpu.iter().zip(gpu).map(|(a, b)| (*a - *b).abs() as f64).fold(0.0, f64::max);
    max / rms.max(1e-30)
}

/// __float2bfloat16 / __bfloat162float (round to nearest even), as t16d_post_gu rounds h
fn bf16r(x: f32) -> f32 {
    half::bf16::from_f32(x).to_f32()
}

/// ExpSlot table bytes: one layer x `e` experts, (gate, up, down) byte offsets, u64::MAX = not resident
fn tab_bytes(e: usize, resident: &[(u64, u64, u64)]) -> Vec<u8> {
    let mut t = vec![0u8; e * 32];
    for (i, &(g, u, d)) in resident.iter().enumerate() {
        let o = i * 32;
        t[o..o + 8].copy_from_slice(&g.to_le_bytes());
        t[o + 8..o + 16].copy_from_slice(&u.to_le_bytes());
        t[o + 16..o + 24].copy_from_slice(&d.to_le_bytes());
    }
    for r in resident.len()..e {
        t[r * 32..r * 32 + 8].copy_from_slice(&u64::MAX.to_le_bytes());
    }
    t
}

/// T16 decode scratch (T16Bufs's pieces, part buffers sized for the run's k-splits S)
struct Scratch {
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

fn scratch(g: &Gpu, picks: usize, dim: usize, inter: usize, sgu: usize, sd: usize) -> Result<Scratch> {
    let f4 = |n: usize| g.alloc(n * 4);
    Ok(Scratch {
        xpg: f4(picks * dim / 2)?,
        xpu: f4(picks * dim / 2)?,
        xpd: f4(picks * inter / 2)?,
        xdg: f4(picks * dim / 256)?,
        xsg: f4(picks * dim / 256)?,
        xdu: f4(picks * dim / 256)?,
        xsu: f4(picks * dim / 256)?,
        xdd: f4(picks * inter / 256)?,
        xsd: f4(picks * inter / 256)?,
        pg: f4(sgu * picks * inter)?,
        pu: f4(sgu * picks * inter)?,
        pd: f4(sd * picks * dim)?,
        h: f4(picks * inter)?,
    })
}

/// the t16d_* launch sequence of run_t16_on (experts.rs), one layer's picks: prep gate/up, the two GEMVs, post_gu;
/// `down` adds prep h, the down GEMV and post_down. `gemv` is the t16d_gemv kernel for the layer's k2.
/// MUST mirror `run_t16_on` (experts.rs): launch order, grid shapes, arg lists. Update when that changes.
#[allow(clippy::too_many_arguments)]
fn chain(
    g: &Gpu,
    prep: Kernel,
    gemv: Kernel,
    post_gu: Kernel,
    post_down: Kernel,
    down: bool,
    sel: u64,
    wts: u64,
    tab: u64,
    base: u64,
    x: u64,
    out: u64,
    l: usize,
    e: usize,
    k: usize,
    t: usize,
    dim: usize,
    inter: usize,
    kg: usize,
    kd: usize,
    sgu: usize,
    sd: usize,
    lim: f32,
    s: &Scratch,
) -> Result<()> {
    let (li, ki, ti) = (Arg::I(l as i32), Arg::I(k as i32), Arg::I(t as i32));
    let blocks = |warps: usize| ((warps as u32).div_ceil(4), 1, 1);
    let (tab, base, e) = (Arg::P(tab), Arg::P(base), Arg::I(e as i32));
    let picks = t * k;
    for (role, xp, xd, xs) in [(0, s.xpg.ptr, s.xdg.ptr, s.xsg.ptr), (1, s.xpu.ptr, s.xdu.ptr, s.xsu.ptr)] {
        g.launch(
            prep,
            blocks(picks * dim / 256),
            128,
            0,
            &[Arg::P(x), Arg::I(0), Arg::P(sel), tab, base, li, e, ki, ti, Arg::I(role), Arg::I(dim as i32), Arg::I(inter as i32), Arg::I(kg as i32), Arg::P(xp), Arg::P(xd), Arg::P(xs)],
        )?;
    }
    for (role, xp, xd, xs, part) in [(0, s.xpg.ptr, s.xdg.ptr, s.xsg.ptr, s.pg.ptr), (1, s.xpu.ptr, s.xdu.ptr, s.xsu.ptr, s.pu.ptr)] {
        g.launch(
            gemv,
            blocks(picks * inter / 32 * sgu),
            128,
            0,
            &[Arg::P(xp), Arg::P(xd), Arg::P(xs), Arg::P(sel), tab, base, li, e, ki, ti, Arg::I(role), Arg::I(dim as i32), Arg::I(inter as i32), Arg::I(sgu as i32), Arg::P(part)],
        )?;
    }
    g.launch(
        post_gu,
        blocks(picks * inter / 128),
        128,
        0,
        &[Arg::P(s.pg.ptr), Arg::P(s.pu.ptr), Arg::I(sgu as i32), Arg::P(sel), Arg::P(wts), tab, base, li, e, ki, ti, Arg::I(dim as i32), Arg::I(inter as i32), Arg::I(kg as i32), Arg::F(lim), Arg::P(s.h.ptr)],
    )?;
    if down {
        g.launch(
            prep,
            blocks(picks * inter / 256),
            128,
            0,
            &[Arg::P(s.h.ptr), Arg::I(1), Arg::P(sel), tab, base, li, e, ki, ti, Arg::I(2), Arg::I(inter as i32), Arg::I(dim as i32), Arg::I(kd as i32), Arg::P(s.xpd.ptr), Arg::P(s.xdd.ptr), Arg::P(s.xsd.ptr)],
        )?;
        g.launch(
            gemv,
            blocks(picks * dim / 32 * sd),
            128,
            0,
            &[Arg::P(s.xpd.ptr), Arg::P(s.xdd.ptr), Arg::P(s.xsd.ptr), Arg::P(sel), tab, base, li, e, ki, ti, Arg::I(2), Arg::I(inter as i32), Arg::I(dim as i32), Arg::I(sd as i32), Arg::P(s.pd.ptr)],
        )?;
        g.launch(
            post_down,
            blocks(t * dim / 128),
            128,
            0,
            &[Arg::P(s.pd.ptr), Arg::I(sd as i32), Arg::P(sel), tab, base, li, e, ki, ti, Arg::I(inter as i32), Arg::I(dim as i32), Arg::I(kd as i32), Arg::P(out)],
        )?;
    }
    Ok(())
}

/// the CPU twin of t16d_post_gu for one pick: had128 * svh on both raw outputs, then the bf16-rounded SwiGLU
fn cpu_post_gu(g_raw: &[f32], u_raw: &[f32], svh_g: &[u8], svh_u: &[u8], n: usize, lim: f32, w: f32) -> Vec<f32> {
    let (mut gv, mut uv) = (g_raw.to_vec(), u_raw.to_vec());
    for o in (0..n).step_by(128) {
        t16k::post_block(&mut gv[o..], svh_g, o);
        t16k::post_block(&mut uv[o..], svh_u, o);
    }
    (0..n)
        .map(|i| {
            let (g, u) = (gv[i].min(lim), uv[i].clamp(-lim, lim));
            bf16r(g / (1.0 + (-g).exp()) * u * w)
        })
        .collect()
}

fn main() -> Result<()> {
    let g = Gpu::new(0)?;
    let kernels: HashMap<&str, Kernel> = ["repack_t16", "t16d_prep", "t16d_gemv_4", "t16d_gemv0_4", "t16d_post_gu", "t16d_post_down"]
        .into_iter()
        .map(|n| Ok((n, g.kernel(n)?)))
        .collect::<Result<_>>()?;
    let mut s = 0xd1a7_da5a_b16c_0de5u64;

    // ---- the host repack against the repack_t16 kernel (the layout the GPU store holds) ------------------------
    for (k2, k, n) in [(4usize, 512usize, 128usize), (4, 512, 512), (3, 512, 128), (5, 512, 128)] {
        let m = gen_matrix(&mut s, k2, k, n);
        let host = repack_host(&m, k2, k, n);
        let words = host.len() / 4;
        let src = g.upload(&m)?;
        let dst = g.alloc(host.len())?;
        g.launch(
            kernels["repack_t16"],
            ((words as u32).div_ceil(256), 1, 1),
            256,
            0,
            &[Arg::I(k2 as i32), Arg::P(src.ptr), Arg::P(dst.ptr), Arg::I(k as i32), Arg::I(n as i32)],
        )?;
        let dev: Vec<u8> = g.download(dst.ptr, host.len())?;
        ensure!(dev == host, "host repack != repack_t16 kernel: k2={k2} {k}x{n}");
        println!("{}", json!({"test":"repack-layout","k2":k2,"k":k,"n":n,"bytes":host.len(),"exact":true}));
    }

    // ---- part + post_gu at the task's shape: k2=4, k=512 inputs, n=128 outputs (gate/up of dim=512, inter=128) --
    {
        let (dim, inter, k2, lim, w) = (512usize, 128usize, 4usize, 40.0f32, 0.37f32);
        let mg = gen_matrix(&mut s, k2, dim, inter);
        let mu = gen_matrix(&mut s, k2, dim, inter);
        let (pg_h, pu_h) = (repack_host(&mg, k2, dim, inter), repack_host(&mu, k2, dim, inter));
        let mb = pg_h.len().div_ceil(256) * 256;
        let mut store = vec![0u8; 2 * mb]; // gate at 0, up at mb (256-aligned like the engine's store)
        store[..pg_h.len()].copy_from_slice(&pg_h);
        store[mb..mb + pu_h.len()].copy_from_slice(&pu_h);
        let base = g.upload(&store)?;
        let tab = g.upload(&tab_bytes(1, &[(0, mb as u64, u64::MAX)]))?;
        let sel = g.upload(&[0i32])?;
        let wts = g.upload(&[w])?;
        let x: Vec<f32> = (0..dim).map(|_| (rng(&mut s) % 2001) as f32 / 1000.0 - 1.0).collect();
        let xd = g.upload(&x)?;

        // CPU reference: its own prep / gemv pipeline on the stored layout
        let (suh_g, svh_g) = t16k::scales(&mg, k2, dim, inter);
        let (suh_u, svh_u) = t16k::scales(&mu, k2, dim, inter);
        let (mut ag, mut au) = (Act::default(), Act::default());
        t16k::prep(&x, suh_g, dim, &mut ag);
        t16k::prep(&x, suh_u, dim, &mut au);
        let (mut yg, mut yu) = (vec![0f32; inter], vec![0f32; inter]);
        t16k::gemv(&mg, k2, dim, inter, &ag, 0, inter / 16, &mut yg);
        t16k::gemv(&mu, k2, dim, inter, &au, 0, inter / 16, &mut yu);

        // GPU: the engine's split count for gate/up (T16_SGU) end to end, and S=1 for a directly comparable part
        let s1 = scratch(&g, 1, dim, inter, 1, 1)?;
        chain(&g, kernels["t16d_prep"], kernels["t16d_gemv_4"], kernels["t16d_post_gu"], kernels["t16d_post_down"], false, sel.ptr, wts.ptr, tab.ptr, base.ptr, xd.ptr, 0, 0, 1, 1, 1, dim, inter, k2, k2, 1, 1, lim, &s1)?;
        let part1: Vec<f32> = g.download(s1.pg.ptr, inter)?;
        ensure!(part1.iter().all(|v| v.is_finite()), "nonfinite GPU output (part, S=1)");
        // the same with the plain (SM70-selected) kernel: same output bits
        let s1b = scratch(&g, 1, dim, inter, 1, 1)?;
        chain(&g, kernels["t16d_prep"], kernels["t16d_gemv0_4"], kernels["t16d_post_gu"], kernels["t16d_post_down"], false, sel.ptr, wts.ptr, tab.ptr, base.ptr, xd.ptr, 0, 0, 1, 1, 1, dim, inter, k2, k2, 1, 1, lim, &s1b)?;
        let part1b: Vec<f32> = g.download(s1b.pg.ptr, inter)?;
        ensure!(part1b.iter().all(|v| v.is_finite()), "nonfinite GPU output (part, gemv0)");
        let gemv0_bits_eq = part1.iter().zip(&part1b).all(|(a, b)| a.to_bits() == b.to_bits());
        ensure!(gemv0_bits_eq, "t16d_gemv_4 and t16d_gemv0_4 outputs differ (SM70 runs the same body)");

        let s4 = scratch(&g, 1, dim, inter, T16_SGU, 1)?;
        chain(&g, kernels["t16d_prep"], kernels["t16d_gemv_4"], kernels["t16d_post_gu"], kernels["t16d_post_down"], false, sel.ptr, wts.ptr, tab.ptr, base.ptr, xd.ptr, 0, 0, 1, 1, 1, dim, inter, k2, k2, T16_SGU, 1, lim, &s4)?;
        let part4: Vec<f32> = g.download(s4.pg.ptr, T16_SGU * inter)?;
        ensure!(part4.iter().all(|v| v.is_finite()), "nonfinite GPU output (part, S=T16_SGU)");
        let sum: Vec<f32> = (0..inter).map(|i| (0..T16_SGU).map(|q| part4[q * inter + i]).sum()).collect();
        let h_gpu: Vec<f32> = g.download(s4.h.ptr, inter)?;
        ensure!(h_gpu.iter().all(|v| v.is_finite()), "nonfinite GPU output (post_gu h)");

        // the GPU's own activations: same int8 codes / scales as the CPU prep?
        let xp: Vec<u32> = g.download(s4.xpg.ptr, dim / 2)?;
        let xdv: Vec<f32> = g.download(s4.xdg.ptr, dim / 256)?;
        ensure!(xdv.iter().all(|v| v.is_finite()), "nonfinite GPU output (prep group scales)");
        let xsv: Vec<i32> = g.download(s4.xsg.ptr, dim / 256)?;
        let codes_differ = xp.iter().zip(&ag.x4).filter(|(a, b)| **a as i32 != **b).count();
        let scale_bits_differ = xdv.iter().zip(&ag.d).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        let sums_differ = xsv.iter().zip(&ag.s).filter(|(a, b)| a != b).count();

        // the GEMV alone on the GPU's activations (the Phase-0 exact-integer-path check, scalar reference)
        let mut ax = Act::default();
        ax.x4 = xp.iter().map(|&v| v as i32).collect();
        ax.d = xdv.clone();
        ax.s = xsv.clone();
        let mut yx = vec![0f32; inter];
        t16k::gemv_scalar(&mg, k2, dim, inter, &ax, 0, inter / 16, &mut yx);

        let h_cpu = cpu_post_gu(&yg, &yu, svh_g, svh_u, inter, lim, w);
        let (r_part, r_gemv, r_s4, r_h) = (rel(&yg, &part1), rel(&yx, &part1), rel(&yg, &sum), rel(&h_cpu, &h_gpu));
        let pass = r_gemv < 1e-6 && r_part < 1e-6 && r_s4 < 1e-6 && r_h < 1e-2;
        println!(
            "{}",
            json!({"test":"part","k2":k2,"k":dim,"n":inter,"T":1,"K":1,
                   "maxrel_part":r_part,"maxrel_gemv_same_acts":r_gemv,"maxrel_split_s4":r_s4,"maxrel_post_gu_h":r_h,
                   "gemv0_bits_equal":gemv0_bits_eq,"codes_differ":codes_differ,"scale_bits_differ":scale_bits_differ,"group_sums_differ":sums_differ,
                   "gate":1e-6,"pass":pass,
                   "cpu_part":yg,"gpu_part":part1,"gpu_h_first16":h_gpu[..16.min(h_gpu.len())]})
        );
        ensure!(pass, "part check failed: gemv {r_gemv:.3e}, part {r_part:.3e}, split {r_s4:.3e}, h {r_h:.3e}");
    }

    // ---- the full 8-launch chain (incl. prep h, down GEMV, post_down): dim = inter = 512, K2 = 4 ----------------
    {
        let (dim, inter, k2, lim, w) = (512usize, 512usize, 4usize, 40.0f32, 0.37f32);
        let (mg, mu, md) = (gen_matrix(&mut s, k2, dim, inter), gen_matrix(&mut s, k2, dim, inter), gen_matrix(&mut s, k2, inter, dim));
        let (pg_h, pu_h, pd_h) = (repack_host(&mg, k2, dim, inter), repack_host(&mu, k2, dim, inter), repack_host(&md, k2, inter, dim));
        let mb = pg_h.len().div_ceil(256) * 256;
        let mut store = vec![0u8; 3 * mb];
        store[..pg_h.len()].copy_from_slice(&pg_h);
        store[mb..mb + pu_h.len()].copy_from_slice(&pu_h);
        store[2 * mb..2 * mb + pd_h.len()].copy_from_slice(&pd_h);
        let base = g.upload(&store)?;
        let tab = g.upload(&tab_bytes(1, &[(0, mb as u64, 2 * mb as u64)]))?;
        let sel = g.upload(&[0i32])?;
        let wts = g.upload(&[w])?;
        let x: Vec<f32> = (0..dim).map(|_| (rng(&mut s) % 2001) as f32 / 1000.0 - 1.0).collect();
        let xb = g.upload(&x)?;
        let out = g.alloc(dim * 4)?;

        // CPU: the whole decode step
        let (suh_g, svh_g) = t16k::scales(&mg, k2, dim, inter);
        let (suh_u, svh_u) = t16k::scales(&mu, k2, dim, inter);
        let (suh_d, svh_d) = t16k::scales(&md, k2, inter, dim);
        let (mut ag, mut au) = (Act::default(), Act::default());
        t16k::prep(&x, suh_g, dim, &mut ag);
        t16k::prep(&x, suh_u, dim, &mut au);
        let (mut yg, mut yu) = (vec![0f32; inter], vec![0f32; inter]);
        t16k::gemv(&mg, k2, dim, inter, &ag, 0, inter / 16, &mut yg);
        t16k::gemv(&mu, k2, dim, inter, &au, 0, inter / 16, &mut yu);
        let h_cpu = cpu_post_gu(&yg, &yu, svh_g, svh_u, inter, lim, w);
        let mut ad = Act::default();
        t16k::prep(&h_cpu, suh_d, inter, &mut ad);
        let mut yd = vec![0f32; dim];
        t16k::gemv(&md, k2, inter, dim, &ad, 0, dim / 16, &mut yd);
        let mut out_cpu = yd;
        for o in (0..dim).step_by(128) {
            t16k::post_block(&mut out_cpu[o..], svh_d, o);
        }

        // GPU: the engine's chain, engine split counts (gate/up T16_SGU, down T16_SD)
        let scr = scratch(&g, 1, dim, inter, T16_SGU, T16_SD)?;
        chain(&g, kernels["t16d_prep"], kernels["t16d_gemv_4"], kernels["t16d_post_gu"], kernels["t16d_post_down"], true, sel.ptr, wts.ptr, tab.ptr, base.ptr, xb.ptr, out.ptr, 0, 1, 1, 1, dim, inter, k2, k2, T16_SGU, T16_SD, lim, &scr)?;
        let out_gpu: Vec<f32> = g.download(out.ptr, dim)?;
        ensure!(out_gpu.iter().all(|v| v.is_finite()), "nonfinite GPU output (full-chain out)");
        let h_gpu: Vec<f32> = g.download(scr.h.ptr, inter)?;
        ensure!(h_gpu.iter().all(|v| v.is_finite()), "nonfinite GPU output (full-chain h)");
        // count of h entries whose bits differ from the CPU's (informational; the rel gates carry the check)
        let h_bits_mismatch = h_gpu.iter().zip(&h_cpu).filter(|(a, b)| a.to_bits() != b.to_bits()).count();

        // the down GEMV in isolation: the GPU prep/gemv on the CPU's h (part-level, 1e-6)
        let hb = g.upload(&h_cpu)?;
        let pd1 = g.alloc(dim * 4)?;
        g.launch(
            kernels["t16d_prep"],
            (1, 1, 1),
            128,
            0,
            &[Arg::P(hb.ptr), Arg::I(1), Arg::P(sel.ptr), Arg::P(tab.ptr), Arg::P(base.ptr), Arg::I(0), Arg::I(1), Arg::I(1), Arg::I(1), Arg::I(2), Arg::I(inter as i32), Arg::I(dim as i32), Arg::I(k2 as i32), Arg::P(scr.xpd.ptr), Arg::P(scr.xdd.ptr), Arg::P(scr.xsd.ptr)],
        )?;
        g.launch(
            kernels["t16d_gemv_4"],
            (((dim / 32) as u32).div_ceil(4), 1, 1),
            128,
            0,
            &[Arg::P(scr.xpd.ptr), Arg::P(scr.xdd.ptr), Arg::P(scr.xsd.ptr), Arg::P(sel.ptr), Arg::P(tab.ptr), Arg::P(base.ptr), Arg::I(0), Arg::I(1), Arg::I(1), Arg::I(1), Arg::I(2), Arg::I(inter as i32), Arg::I(dim as i32), Arg::I(1), Arg::P(pd1.ptr)],
        )?;
        let down1: Vec<f32> = g.download(pd1.ptr, dim)?;
        ensure!(down1.iter().all(|v| v.is_finite()), "nonfinite GPU output (down part)");
        let mut down_cpu = vec![0f32; dim];
        t16k::gemv(&md, k2, inter, dim, &ad, 0, dim / 16, &mut down_cpu);
        let r_down = rel(&down_cpu, &down1);
        let r_out = rel(&out_cpu, &out_gpu);
        let pass = r_down < 1e-6 && r_out < 1e-2;
        println!(
            "{}",
            json!({"test":"full-chain","k2":k2,"dim":dim,"inter":inter,"T":1,"K":1,"sgu":T16_SGU,"sd":T16_SD,
                   "maxrel_down_part":r_down,"maxrel_out":r_out,"h_bits_mismatch":h_bits_mismatch,
                   "gate_down":1e-6,"gate_out":1e-2,"pass":pass,
                   "cpu_out_first16":out_cpu[..16],"gpu_out_first16":out_gpu[..16]})
        );
        ensure!(pass, "full-chain check failed: down {r_down:.3e}, out {r_out:.3e}");
    }
    Ok(())
}
