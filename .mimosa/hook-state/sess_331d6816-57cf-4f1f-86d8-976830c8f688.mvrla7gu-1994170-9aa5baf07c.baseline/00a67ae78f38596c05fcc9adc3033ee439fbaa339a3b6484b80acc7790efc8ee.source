//! Bit-exact checks and isolated timings for opt-in prefill kernels. Does not load a model.
use anyhow::{Result, ensure};
use dsx::gpu::{Arg, Gpu};
use half::f16;
use serde_json::json;
use std::time::Instant;
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
    let g = Gpu::new(0)?;
    let kernels: std::collections::HashMap<_, _> = [
        "repack_t16",
        "dequant_t16_f16",
        "dequant_t16_f16_specialized",
        "t16_pre",
        "t16_pre_warp",
        "t16_post",
        "t16_post_warp",
    ]
    .into_iter()
    .map(|name| Ok((name, g.kernel(name)?)))
    .collect::<Result<_>>()?;
    for (n, k) in [(2048usize, 6144usize), (6144, 2048)] {
        for k2 in [3usize, 4, 5] {
            let len = n * k * k2 / 16;
            let w: Vec<u8> = (0..len)
                .map(|i| {
                    let x = (i as u64)
                        .wrapping_mul(0x9e3779b97f4a7c15)
                        .wrapping_add(0xd1b54a32d192ed03);
                    (x ^ (x >> 37) ^ (x >> 17)) as u8
                })
                .collect();
            let raw = g.alloc(len)?;
            let packed = g.alloc(len + 2 * (k + n))?;
            let input = g.alloc(len + 2 * (k + n))?;
            g.htod_sync(raw.ptr, &w)?;
            g.htod_sync(input.ptr, &vec![0u8; len + 2 * (k + n)])?;
            g.htod_sync(input.ptr, &w)?;
            let repack = kernels["repack_t16"];
            g.launch(
                repack,
                (((len + 2 * (k + n)) as u32).div_ceil(1024), 1, 1),
                256,
                0,
                &[
                    Arg::I(k2 as i32),
                    Arg::P(input.ptr),
                    Arg::P(packed.ptr),
                    Arg::I(k as i32),
                    Arg::I(n as i32),
                ],
            )?;
            let a = g.alloc(n * k * 2)?;
            let b = g.alloc(n * k * 2)?;
            for (layout, ptr) in [(0, raw.ptr), (1, packed.ptr)] {
                let run = |name: &str, out: u64| -> Result<()> {
                    g.launch(
                        kernels[name],
                        ((k as u32).div_ceil(512), (n / 16) as u32, 1),
                        512,
                        0,
                        &[
                            Arg::I(k2 as i32),
                            Arg::P(ptr),
                            Arg::P(out),
                            Arg::I(n as i32),
                            Arg::I(k as i32),
                            Arg::I(layout),
                        ],
                    )
                };
                run("dequant_t16_f16", a.ptr)?;
                run("dequant_t16_f16_specialized", b.ptr)?;
                let x: Vec<u16> = g.download(a.ptr, n * k)?;
                let y: Vec<u16> = g.download(b.ptr, n * k)?;
                let changed = x.iter().zip(&y).filter(|(a, b)| a != b).count();
                ensure!(
                    changed == 0,
                    "dequant changed {changed} weights: k2={k2}, layout={layout}"
                );
                let baseline = timed(|| run("dequant_t16_f16", a.ptr), &g)?;
                let candidate = timed(|| run("dequant_t16_f16_specialized", b.ptr), &g)?;
                println!(
                    "{}",
                    json!({"test":"dequant","n":n,"k":k,"k2":k2,"layout":layout,"values":n*k,"changed":changed,"baseline_us":baseline,"candidate_us":candidate})
                );
            }
        }
    }
    for cols in [2048usize, 6144] {
        for rows in [1usize, 3, 17, 64, 257] {
            let len = rows * cols;
            let x: Vec<u16> = (0..len)
                .map(|i| f16::from_f32(((i * 17 % 1021) as f32 - 510.) / 127.).to_bits())
                .collect();
            let scale: Vec<u16> = (0..cols)
                .map(|i| f16::from_f32(((i * 13 % 101) as f32 - 50.) / 64.).to_bits())
                .collect();
            let src = g.alloc(len * 2)?;
            g.htod_sync(src.ptr, &x)?;
            let sc = g.alloc(cols * 2)?;
            g.htod_sync(sc.ptr, &scale)?;
            let a = g.alloc(len * 4)?;
            let b = g.alloc(len * 4)?;
            let pre = |warp: bool, out: u64| -> Result<()> {
                if warp {
                    g.launch(
                        kernels["t16_pre_warp"],
                        (((cols / 128 * rows) as u32).div_ceil(4), 1, 1),
                        128,
                        0,
                        &[
                            Arg::P(src.ptr),
                            Arg::P(out),
                            Arg::P(sc.ptr),
                            Arg::I(cols as i32),
                            Arg::I(rows as i32),
                        ],
                    )
                } else {
                    g.launch(
                        kernels["t16_pre"],
                        ((cols / 128) as u32, rows as u32, 1),
                        128,
                        0,
                        &[
                            Arg::P(src.ptr),
                            Arg::P(out),
                            Arg::P(sc.ptr),
                            Arg::I(cols as i32),
                        ],
                    )
                }
            };
            pre(false, a.ptr)?;
            pre(true, b.ptr)?;
            let av: Vec<u16> = g.download(a.ptr, len)?;
            let bv: Vec<u16> = g.download(b.ptr, len)?;
            ensure!(av == bv, "pre differs: cols={cols} rows={rows}");
            let baseline = timed(|| pre(false, a.ptr), &g)?;
            let candidate = timed(|| pre(true, b.ptr), &g)?;
            // Also test the production down-projection's in-place pre transform.
            g.htod_sync(b.ptr, &x)?;
            g.launch(
                kernels["t16_pre_warp"],
                (((cols / 128 * rows) as u32).div_ceil(4), 1, 1),
                128,
                0,
                &[
                    Arg::P(b.ptr),
                    Arg::P(b.ptr),
                    Arg::P(sc.ptr),
                    Arg::I(cols as i32),
                    Arg::I(rows as i32),
                ],
            )?;
            let inplace: Vec<u16> = g.download(b.ptr, len)?;
            ensure!(av == inplace, "in-place pre differs");
            println!(
                "{}",
                json!({"test":"pre","cols":cols,"rows":rows,"values":len,"changed":0,"inplace_exact":true,"baseline_us":baseline,"candidate_us":candidate})
            );
            let input: Vec<f32> = (0..len)
                .map(|i| ((i * 37 % 2053) as f32 - 1026.) / 31.)
                .collect();
            let post = |warp: bool, out: u64| -> Result<()> {
                if warp {
                    g.launch(
                        kernels["t16_post_warp"],
                        (((cols / 128 * rows) as u32).div_ceil(4), 1, 1),
                        128,
                        0,
                        &[
                            Arg::P(out),
                            Arg::P(sc.ptr),
                            Arg::I(cols as i32),
                            Arg::F(1.03125),
                            Arg::I(rows as i32),
                        ],
                    )
                } else {
                    g.launch(
                        kernels["t16_post"],
                        ((cols / 128) as u32, rows as u32, 1),
                        128,
                        0,
                        &[
                            Arg::P(out),
                            Arg::P(sc.ptr),
                            Arg::I(cols as i32),
                            Arg::F(1.03125),
                        ],
                    )
                }
            };
            g.htod_sync(a.ptr, &input)?;
            g.htod_sync(b.ptr, &input)?;
            post(false, a.ptr)?;
            post(true, b.ptr)?;
            let av: Vec<u32> = g.download(a.ptr, len)?;
            let bv: Vec<u32> = g.download(b.ptr, len)?;
            ensure!(av == bv, "post differs: cols={cols} rows={rows}");
            // Keep timing inputs bounded; reinitialize outside each timed block.
            g.htod_sync(a.ptr, &vec![0f32; len])?;
            g.htod_sync(b.ptr, &vec![0f32; len])?;
            let baseline = timed(|| post(false, a.ptr), &g)?;
            let candidate = timed(|| post(true, b.ptr), &g)?;
            println!(
                "{}",
                json!({"test":"post","cols":cols,"rows":rows,"values":len,"changed":0,"baseline_us":baseline,"candidate_us":candidate})
            );
        }
    }
    let fused = g.kernel("t16_post_gu_warp")?;
    let swiglu = g.kernel("swiglu_rows_f16")?;
    for cols in [2048usize, 6144] {
        for rows in [1usize, 3, 17, 64, 257] {
            let len = cols * rows;
            let gate: Vec<f32> = (0..len)
                .map(|i| ((i * 37 % 2053) as f32 - 1026.) / 31.)
                .collect();
            let up: Vec<f32> = (0..len)
                .map(|i| ((i * 47 % 1021) as f32 - 510.) / 17.)
                .collect();
            let sg: Vec<u16> = (0..cols)
                .map(|i| f16::from_f32(((i * 13 % 101) as f32 - 50.) / 64.).to_bits())
                .collect();
            let su: Vec<u16> = (0..cols)
                .map(|i| f16::from_f32(((i * 19 % 127) as f32 - 63.) / 64.).to_bits())
                .collect();
            let weights: Vec<f32> = (0..rows).map(|i| (i % 13) as f32 / 12.).collect();
            let raw_g = g.alloc(len * 4)?;
            g.htod_sync(raw_g.ptr, &gate)?;
            let raw_u = g.alloc(len * 4)?;
            g.htod_sync(raw_u.ptr, &up)?;
            let a = g.alloc(len * 4)?;
            let b = g.alloc(len * 4)?;
            let gs = g.alloc(cols * 2)?;
            g.htod_sync(gs.ptr, &sg)?;
            let us = g.alloc(cols * 2)?;
            g.htod_sync(us.ptr, &su)?;
            let wt = g.alloc(rows * 4)?;
            g.htod_sync(wt.ptr, &weights)?;
            let expected = g.alloc(len * 2)?;
            let actual = g.alloc(len * 2)?;
            for gain in [1f32, 1.03125] {
                for limit in [8f32, 40.] {
                    g.htod_sync(a.ptr, &gate)?;
                    g.htod_sync(b.ptr, &up)?;
                    for (ptr, scale) in [(a.ptr, gs.ptr), (b.ptr, us.ptr)] {
                        g.launch(
                            kernels["t16_post"],
                            ((cols / 128) as u32, rows as u32, 1),
                            128,
                            0,
                            &[
                                Arg::P(ptr),
                                Arg::P(scale),
                                Arg::I(cols as i32),
                                Arg::F(gain),
                            ],
                        )?;
                    }
                    g.launch(
                        swiglu,
                        ((cols as u32).div_ceil(256), rows as u32, 1),
                        256,
                        0,
                        &[
                            Arg::P(a.ptr),
                            Arg::P(b.ptr),
                            Arg::P(wt.ptr),
                            Arg::P(expected.ptr),
                            Arg::I(cols as i32),
                            Arg::F(limit),
                        ],
                    )?;
                    g.launch(
                        fused,
                        (((cols / 128 * rows) as u32).div_ceil(4), 1, 1),
                        128,
                        0,
                        &[
                            Arg::P(raw_g.ptr),
                            Arg::P(raw_u.ptr),
                            Arg::P(gs.ptr),
                            Arg::P(us.ptr),
                            Arg::P(wt.ptr),
                            Arg::P(actual.ptr),
                            Arg::I(cols as i32),
                            Arg::I(rows as i32),
                            Arg::F(gain),
                            Arg::F(limit),
                        ],
                    )?;
                    let x: Vec<u16> = g.download(expected.ptr, len)?;
                    let y: Vec<u16> = g.download(actual.ptr, len)?;
                    ensure!(
                        x == y,
                        "fused gate/up differs: cols={cols}, rows={rows}, gain={gain}, limit={limit}"
                    );
                    println!(
                        "{}",
                        json!({"test":"fused-post-gu","cols":cols,"rows":rows,"gain":gain,"limit":limit,"values":len,"changed":0})
                    );
                }
            }
        }
    }
    // Each attention implementation must permit Q/O aliasing before enabling buffer reuse.
    let tc = dsx::attn::SparseAttnTc::new(&g)?;
    let serial = g.kernel("sparse_attn")?;
    let tile = g.kernel("sparse_attn_tile")?;
    let merge = g.kernel("sparse_attn_merge")?;
    for rows in [1usize, 8, 17, 64] {
        let (heads, hd, window, compressed) = (64usize, 512usize, 128usize, 16usize);
        let len = rows * heads * hd;
        let q: Vec<f32> = (0..len)
            .map(|i| ((i * 31 % 1021) as f32 - 510.) / 127.)
            .collect();
        let kv: Vec<u16> = (0..(window + compressed) * hd)
            .map(|i| half::bf16::from_f32(((i * 17 % 2053) as f32 - 1026.) / 257.).to_bits())
            .collect();
        let wi: Vec<i32> = (0..rows * window)
            .map(|i| {
                if i % window < 16 + i / window {
                    (i % window) as i32
                } else {
                    -1
                }
            })
            .collect();
        let ci: Vec<i32> = (0..rows * compressed)
            .map(|i| {
                if i % compressed % 3 != 0 {
                    (i % compressed) as i32
                } else {
                    -1
                }
            })
            .collect();
        let input = g.alloc(len * 4)?;
        g.htod_sync(input.ptr, &q)?;
        let weights = g.alloc(kv.len() * 2)?;
        g.htod_sync(weights.ptr, &kv)?;
        let idx = g.alloc(wi.len() * 4)?;
        g.htod_sync(idx.ptr, &wi)?;
        let cidx = g.alloc(ci.len() * 4)?;
        g.htod_sync(cidx.ptr, &ci)?;
        let sinks = g.alloc(heads * 4)?;
        g.htod_sync(sinks.ptr, &vec![0.1f32; heads])?;
        let a = g.alloc(len * 4)?;
        let b = g.alloc(len * 4)?;
        let splits = (window + compressed).div_ceil(32);
        let part = g.alloc(rows * splits * heads * (hd + 2) * 4)?;
        for path in ["tc", "serial", "split-k"] {
            let run = |src: u64, dst: u64| -> Result<()> {
                let comp = weights.ptr + (window * hd * 2) as u64;
                if path == "tc" {
                    return tc.launch(
                        &g,
                        src,
                        weights.ptr,
                        idx.ptr,
                        window,
                        comp,
                        cidx.ptr,
                        compressed,
                        sinks.ptr,
                        dst,
                        rows,
                        heads,
                        hd,
                    );
                }
                if path == "serial" {
                    return g.launch(
                        serial,
                        (rows as u32, (heads / 8) as u32, 1),
                        256,
                        0,
                        &[
                            Arg::P(src),
                            Arg::P(weights.ptr),
                            Arg::P(idx.ptr),
                            Arg::I(window as i32),
                            Arg::P(comp),
                            Arg::P(cidx.ptr),
                            Arg::I(compressed as i32),
                            Arg::P(sinks.ptr),
                            Arg::P(dst),
                            Arg::I(heads as i32),
                            Arg::I(hd as i32),
                            Arg::F(1. / (hd as f32).sqrt()),
                        ],
                    );
                }
                g.launch(
                    tile,
                    (rows as u32, (heads / 16) as u32, splits as u32),
                    256,
                    0,
                    &[
                        Arg::P(src),
                        Arg::P(weights.ptr),
                        Arg::P(idx.ptr),
                        Arg::I(window as i32),
                        Arg::P(comp),
                        Arg::P(cidx.ptr),
                        Arg::I(compressed as i32),
                        Arg::P(part.ptr),
                        Arg::I(heads as i32),
                        Arg::I(hd as i32),
                        Arg::F(1. / (hd as f32).sqrt()),
                    ],
                )?;
                g.launch(
                    merge,
                    (rows as u32, heads as u32, 1),
                    256,
                    0,
                    &[
                        Arg::P(part.ptr),
                        Arg::I(splits as i32),
                        Arg::P(sinks.ptr),
                        Arg::P(dst),
                        Arg::I(heads as i32),
                        Arg::I(hd as i32),
                    ],
                )
            };
            g.htod_sync(b.ptr, &q)?;
            run(input.ptr, a.ptr)?;
            run(b.ptr, b.ptr)?;
            let x: Vec<u32> = g.download(a.ptr, len)?;
            let y: Vec<u32> = g.download(b.ptr, len)?;
            ensure!(x == y, "attention aliasing differs: {path}, rows={rows}");
            ensure!(
                x.iter().all(|&v| f32::from_bits(v).is_finite()),
                "nonfinite attention"
            );
            println!(
                "{}",
                json!({"test":"attention-alias","path":path,"rows":rows,"heads":heads,"values":len,"changed":0})
            );
        }
    }
    Ok(())
}
