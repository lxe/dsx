//! T3 throughput probe: one synthetic decode expert step (T=8 tokens x K=6 picks from E=384 resident experts,
//! dim=7168, inter=2048, T16 K2=4) through the full t16d_* chain on one GPU, launched exactly as the engine's
//! run_t16_on (experts.rs) launches it (gate/up GEMV k-splits S=14 as specced for this shape; down S = the
//! engine's T16_SD). Gate: <= 90 ms per expert step on one V100. Calibration: Phase-0's K1 micro-benchmark ran
//! the raw GEMV kernel at 207.8 GB/s weight-stream (17.66 us per k=7168 x n=2048 matrix), so the 48 picks' 144
//! matrix reads (~0.53 GB of trellis) are ~2.5 ms of streaming -- anything near 90 ms is overhead, not math.
//! The store is zero-filled: the kernels have no value-dependent branches, so timing is data-independent.
use anyhow::{Result, ensure};
use dsx::experts::T16_SD;
use dsx::gpu::{Arg, Buf, Gpu, Kernel};
use dsx::t16;
use serde_json::json;
use std::collections::HashMap;
use std::time::Instant;

fn rng(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}

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

/// the run_t16_on (experts.rs) launch sequence for one layer's expert step; `phase` selects a subset for the
/// breakdown: 0 = all eight launches, 1 = the two preps + two GEMVs + post_gu, 2 = the three GEMVs only,
/// 3 = the three preps only, 4 = the two posts only.
/// MUST mirror `run_t16_on` (experts.rs): launch order, grid shapes, arg lists. Update when that changes.
#[allow(clippy::too_many_arguments)]
fn chain(
    g: &Gpu,
    prep: Kernel,
    gemv: Kernel,
    post_gu: Kernel,
    post_down: Kernel,
    phase: u8,
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
    if phase == 0 || phase == 1 || phase == 3 {
        for (role, xp, xd, xs) in [(0, s.xpg.ptr, s.xdg.ptr, s.xsg.ptr), (1, s.xpu.ptr, s.xdu.ptr, s.xsu.ptr)] {
            g.launch(
                prep,
                blocks(picks * dim / 256),
                128,
                0,
                &[Arg::P(x), Arg::I(0), Arg::P(sel), tab, base, li, e, ki, ti, Arg::I(role), Arg::I(dim as i32), Arg::I(inter as i32), Arg::I(kg as i32), Arg::P(xp), Arg::P(xd), Arg::P(xs)],
            )?;
        }
    }
    if phase == 0 || phase == 1 || phase == 2 {
        for (role, xp, xd, xs, part) in [(0, s.xpg.ptr, s.xdg.ptr, s.xsg.ptr, s.pg.ptr), (1, s.xpu.ptr, s.xdu.ptr, s.xsu.ptr, s.pu.ptr)] {
            g.launch(
                gemv,
                blocks(picks * inter / 32 * sgu),
                128,
                0,
                &[Arg::P(xp), Arg::P(xd), Arg::P(xs), Arg::P(sel), tab, base, li, e, ki, ti, Arg::I(role), Arg::I(dim as i32), Arg::I(inter as i32), Arg::I(sgu as i32), Arg::P(part)],
            )?;
        }
    }
    if phase == 0 || phase == 1 || phase == 4 {
        g.launch(
            post_gu,
            blocks(picks * inter / 128),
            128,
            0,
            &[Arg::P(s.pg.ptr), Arg::P(s.pu.ptr), Arg::I(sgu as i32), Arg::P(sel), Arg::P(wts), tab, base, li, e, ki, ti, Arg::I(dim as i32), Arg::I(inter as i32), Arg::I(kg as i32), Arg::F(lim), Arg::P(s.h.ptr)],
        )?;
    }
    if phase == 0 {
        g.launch(
            prep,
            blocks(picks * inter / 256),
            128,
            0,
            &[Arg::P(s.h.ptr), Arg::I(1), Arg::P(sel), tab, base, li, e, ki, ti, Arg::I(2), Arg::I(inter as i32), Arg::I(dim as i32), Arg::I(kd as i32), Arg::P(s.xpd.ptr), Arg::P(s.xdd.ptr), Arg::P(s.xsd.ptr)],
        )?;
    }
    if phase == 0 || phase == 2 {
        g.launch(
            gemv,
            blocks(picks * dim / 32 * sd),
            128,
            0,
            &[Arg::P(s.xpd.ptr), Arg::P(s.xdd.ptr), Arg::P(s.xsd.ptr), Arg::P(sel), tab, base, li, e, ki, ti, Arg::I(2), Arg::I(inter as i32), Arg::I(dim as i32), Arg::I(sd as i32), Arg::P(s.pd.ptr)],
        )?;
    }
    if phase == 0 || phase == 4 {
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

fn timed(mut f: impl FnMut() -> Result<()>, g: &Gpu) -> Result<f64> {
    for _ in 0..5 {
        f()?;
    }
    g.sync()?;
    let t = Instant::now();
    for _ in 0..100 {
        f()?;
    }
    g.sync()?;
    Ok(t.elapsed().as_secs_f64() * 1e4) // microseconds per call
}

fn main() -> Result<()> {
    let (t, k, e, dim, inter, k2, sgu, sd, lim) = (8usize, 6usize, 384usize, 7168usize, 2048usize, 4usize, 14usize, T16_SD, 40.0f32);
    let g = Gpu::new(0)?;
    let kernels: HashMap<&str, Kernel> = ["t16d_prep", "t16d_gemv_4", "t16d_gemv0_4", "t16d_post_gu", "t16d_post_down"]
        .into_iter()
        .map(|n| Ok((n, g.kernel(n)?)))
        .collect::<Result<_>>()?;
    let (free, total) = g.mem_info()?;
    println!("{}", json!({"test":"env","gpu":g.name.clone(),"arch":g.arch,"sm70":g.is_sm70(),"sms":g.sms,"free_mib":free >> 20,"total_mib":total >> 20}));

    // the synthetic store: E resident experts, gate | up [inter][dim] and down [dim][inter] per expert, all T16 K2
    let mb = t16::trellis_bytes(k2, dim, inter) + 2 * dim + 2 * inter; // = matrix_bytes(ty, dim, inter)
    let per = 3 * mb.div_ceil(256) * 256;
    let store = g.alloc_zeroed(e * per)?;
    let mut tab = vec![0u8; e * 32];
    for x in 0..e {
        let (o, b) = (x * 32, (x * per) as u64);
        tab[o..o + 8].copy_from_slice(&b.to_le_bytes());
        tab[o + 8..o + 16].copy_from_slice(&(b + mb as u64).to_le_bytes());
        tab[o + 16..o + 24].copy_from_slice(&(b + 2 * mb as u64).to_le_bytes());
    }
    let tabb = g.upload(&tab)?;
    let mut s = 0x5eed_cafe_1234_5678u64;
    let sel: Vec<i32> = (0..t * k).map(|j| ((j * 8) % e) as i32).collect(); // 48 distinct experts: the heaviest case
    let wts: Vec<f32> = (0..t * k).map(|j| ((j % 7) as f32 + 1.0) / 7.0).collect();
    let x: Vec<f32> = (0..t * dim).map(|_| (rng(&mut s) % 2001) as f32 / 1000.0 - 1.0).collect();
    let (selb, wtsb, xb) = (g.upload(&sel)?, g.upload(&wts)?, g.upload(&x)?);
    let scr = scratch(&g, t * k, dim, inter, sgu, sd)?;
    let out = g.alloc(t * dim * 4)?;
    let run = |gemv: Kernel, phase: u8| {
        chain(&g, kernels["t16d_prep"], gemv, kernels["t16d_post_gu"], kernels["t16d_post_down"], phase, selb.ptr, wtsb.ptr, tabb.ptr, store.ptr, xb.ptr, out.ptr, 0, e, k, t, dim, inter, k2, k2, sgu, sd, lim, &scr)
    };

    // the engine's SM70 selection first (run_t16_on picks t16d_gemv0_* on Volta), then the staged name for the record
    let us0 = timed(|| run(kernels["t16d_gemv0_4"], 0), &g)?;
    let us1 = timed(|| run(kernels["t16d_gemv_4"], 0), &g)?;
    let gbps = |us: f64| (t * k) as f64 * 3.0 * t16::trellis_bytes(k2, dim, inter) as f64 / (us * 1e-6) / 1e9;
    let (gate, ms0, ms1) = (90.0f64, us0 / 1e3, us1 / 1e3);
    println!(
        "{}",
        json!({"test":"expert-step","T":t,"K":k,"E":e,"dim":dim,"inter":inter,"k2":k2,"sgu":sgu,"sd":sd,
               "ms_per_step_gemv0":ms0,"ms_per_step_gemv":ms1,"gbps_weight_stream_gemv0":gbps(us0),
               "gate_ms":gate,"pass":ms0 <= gate})
    );
    let phases = |name: &str, phase: u8, gemv: Kernel| -> Result<()> {
        let us = timed(|| run(gemv, phase), &g)?;
        println!("{}", json!({"test":"phase","which":name,"us":us,"ms":us / 1e3}));
        Ok(())
    };
    phases("prep-gu-gemv-post", 1, kernels["t16d_gemv0_4"])?;
    phases("gemv-only", 2, kernels["t16d_gemv0_4"])?;
    phases("prep-only", 3, kernels["t16d_gemv0_4"])?;
    phases("post-only", 4, kernels["t16d_gemv0_4"])?;
    let final_out: Vec<f32> = g.download(out.ptr, t * dim)?;
    let nonfinite = final_out.iter().filter(|v| !v.is_finite()).count();
    println!("{}", json!({"test":"sanity","out_nonfinite":nonfinite,"out_first8":final_out[..8]}));
    ensure!(nonfinite == 0, "nonfinite probe output");
    Ok(())
}
