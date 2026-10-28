#![cfg(feature = "cuda")]
//! T16 prefill kernels (t16.cuh): dequant_t16_f16 + t16_pre / t16_post around a GEMM, against core's t16::forward.
use dsx::gguf::GType;
use dsx::gpu::*;
use dsx::t16;

fn rng(seed: &mut u64) -> u64 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    *seed
}

/// a T16 matrix with random trellis bits and scales near 0.02 (the size of real suh / svh)
fn random_matrix(ty: GType, k: usize, n: usize, seed: &mut u64) -> Vec<u8> {
    let k2 = t16::k2(ty).unwrap();
    let tb = t16::trellis_bytes(k2, k, n);
    let mut v: Vec<u8> = (0..t16::matrix_bytes(ty, k, n)).map(|_| rng(seed) as u8).collect();
    for c in v[tb..].chunks_exact_mut(2) {
        let s = 0.02 * (0.5 + (rng(seed) % 1000) as f32 / 1000.0) * if rng(seed) % 2 == 0 { 1.0 } else { -1.0 };
        c.copy_from_slice(&half::f16::from_f32(s).to_le_bytes());
    }
    v
}

#[test]
fn t16_prefill_matches_reference() {
    let g = Gpu::new(0).unwrap();
    let kern = |n: &str| g.kernel(n).unwrap();
    let mut seed = 0x1234_5678_9abc_def1u64;
    let nt = 5;
    for (ty, k, n) in [(GType::T16K15, 512, 256), (GType::T16K2, 256, 512), (GType::T16K25, 384, 256)] {
        let k2 = t16::k2(ty).unwrap();
        let w = random_matrix(ty, k, n, &mut seed);
        let x: Vec<f32> = (0..nt * k).map(|_| (rng(&mut seed) % 2001) as f32 / 1000.0 - 1.0).collect();
        let x16: Vec<u16> = x.iter().map(|&v| half::f16::from_f32(v).to_bits()).collect();
        // reference on the f16-rounded inputs
        let want: Vec<f32> = (0..nt)
            .flat_map(|r| {
                let xr: Vec<f32> = x16[r * k..(r + 1) * k].iter().map(|&h| half::f16::from_bits(h).to_f32()).collect();
                t16::forward(&w, ty, k, n, &xr)
            })
            .collect();
        let (wb, xb, tb, w16, yb) = (g.alloc(w.len()).unwrap(), g.alloc(nt * k * 2).unwrap(), g.alloc(nt * k * 2).unwrap(), g.alloc(n * k * 2).unwrap(), g.alloc(nt * n * 4).unwrap());
        g.htod_sync(wb.ptr, &w).unwrap();
        g.htod_sync(xb.ptr, &x16).unwrap();
        let trb = t16::trellis_bytes(k2, k, n) as u64;
        g.launch(kern("t16_pre"), ((k / 128) as u32, nt as u32, 1), 128, 0, &[Arg::P(xb.ptr), Arg::P(tb.ptr), Arg::P(wb.ptr + trb), Arg::I(k as i32)]).unwrap();
        // the decoded inner matrix must equal core's exactly, from the stored layout and after repack_t16
        let inner = t16::decode_inner(&w[..trb as usize], k2, k, n);
        let rp = g.alloc(w.len()).unwrap();
        g.launch(kern("repack_t16"), (((w.len() / 4) as u32).div_ceil(256), 1, 1), 256, 0, &[Arg::I(k2 as i32), Arg::P(wb.ptr), Arg::P(rp.ptr), Arg::I(k as i32), Arg::I(n as i32)]).unwrap();
        for (src, repacked) in [(wb.ptr, 0), (rp.ptr, 1)] {
            g.launch(kern("dequant_t16_f16"), ((k as u32).div_ceil(512), (n / 16) as u32, 1), 512, 0, &[Arg::I(k2 as i32), Arg::P(src), Arg::P(w16.ptr), Arg::I(n as i32), Arg::I(k as i32), Arg::I(repacked)]).unwrap();
            let mut wh = vec![0u16; n * k];
            g.dtoh(&mut wh, w16.ptr).unwrap();
            g.sync().unwrap();
            let bad = (0..k).flat_map(|i| (0..n).map(move |j| (i, j))).filter(|&(i, j)| half::f16::from_bits(wh[j * k + i]).to_f32() != inner[i * n + j]).count();
            assert_eq!(bad, 0, "{ty:?} (repacked {repacked}): {bad} decoded weights differ from core::t16::decode_inner");
        }
        g.gemm_f16(tb.ptr, w16.ptr, yb.ptr, nt, k, n, n).unwrap();
        g.launch(kern("t16_post"), ((n / 128) as u32, nt as u32, 1), 128, 0, &[Arg::P(yb.ptr), Arg::P(wb.ptr + trb + 2 * k as u64), Arg::I(n as i32), Arg::F(1.0)]).unwrap();
        let mut got = vec![0f32; nt * n];
        g.dtoh(&mut got, yb.ptr).unwrap();
        g.sync().unwrap();
        let rms = (want.iter().map(|v| v * v).sum::<f32>() / want.len() as f32).sqrt();
        let e = want.iter().zip(&got).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max) / rms;
        println!("{ty:?} {k}x{n}: max |err| / rms = {e:.2e}");
        assert!(e < 5e-3, "{ty:?}: prefill output vs reference {e}");
    }
}

fn bf16r(x: f32) -> f32 {
    half::bf16::from_f32(x).to_f32()
}

/// The decode path (t16d_*) of a store with T16 experts, against core's f32 reference forward. The int8 activations
/// of the GEMVs (per 256 inputs, like quant_act) leave ~1% RMS error; skipped picks (not resident) contribute 0.
#[test]
fn t16_decode_matches_reference() {
    use dsx::experts::{ExpCfg, GpuExperts, RawExpert};
    let g = Gpu::new(0).unwrap();
    let mut seed = 0x0dd5_eed5_1234_5678u64;
    let c = ExpCfg { n_layers: 2, n_experts: 10, inter: 2304, dim: 5120, lim: 10.0, rot: 0 };
    let (gu, dn) = (GType::T16K15, GType::T16K2);
    let mats: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> = (0..8)
        .map(|_| (random_matrix(gu, c.dim, c.inter, &mut seed), random_matrix(gu, c.dim, c.inter, &mut seed), random_matrix(dn, c.inter, c.dim, &mut seed)))
        .collect();
    // experts 0..8 resident on layer 1; 8 and 9 are not
    let items: Vec<((usize, usize), RawExpert)> = mats.iter().enumerate().map(|(e, m)| ((1, e), RawExpert { gate: &m.0, up: &m.1, down: &m.2, gu, dn })).collect();
    let st = GpuExperts::build(&g, c, &items, true).unwrap();
    let k = 6;
    for t in [1usize, 2] {
        let sel: Vec<i32> = (0..t * k).map(|j| [3, 9, 0, 5, 7, 1, 2, 6, 8, 4, 1, 3][j % 12]).collect();
        let wts: Vec<f32> = (0..t * k).map(|_| 0.05 + (rng(&mut seed) % 100) as f32 * 0.003).collect();
        let x: Vec<f32> = (0..t * c.dim).map(|_| (rng(&mut seed) % 2001) as f32 / 1000.0 - 1.0).collect();
        g.htod_sync(st.sel.ptr, &sel).unwrap();
        g.htod_sync(st.wts.ptr, &wts).unwrap();
        g.htod_sync(st.x.ptr, &x).unwrap();
        st.run_device_fast(&g, 1, st.sel.ptr, st.wts.ptr, k, t, st.x.ptr, st.out.ptr).unwrap();
        let got: Vec<f32> = g.download(st.out.ptr, t * c.dim).unwrap();
        let mut want = vec![0f32; t * c.dim];
        for tt in 0..t {
            let xr = &x[tt * c.dim..(tt + 1) * c.dim];
            for j in 0..k {
                let e = sel[tt * k + j] as usize;
                if e >= mats.len() {
                    continue;
                }
                let m = &mats[e];
                let (gv, uv) = (t16::forward(&m.0, gu, c.dim, c.inter, xr), t16::forward(&m.1, gu, c.dim, c.inter, xr));
                let h: Vec<f32> = gv.iter().zip(&uv).map(|(&a, &b)| {
                    let (a, b) = (a.min(c.lim), b.clamp(-c.lim, c.lim));
                    bf16r(a / (1.0 + (-a).exp()) * b * wts[tt * k + j])
                }).collect();
                let o = t16::forward(&m.2, dn, c.inter, c.dim, &h);
                for r in 0..c.dim {
                    want[tt * c.dim + r] += o[r];
                }
            }
        }
        let rms = |v: &[f32]| (v.iter().map(|a| (*a as f64).powi(2)).sum::<f64>() / v.len() as f64).sqrt() as f32;
        let d: Vec<f32> = want.iter().zip(&got).map(|(a, b)| a - b).collect();
        let e = rms(&d) / rms(&want);
        println!("T16 decode T={t}: rms err {e:.2e} of rms (int8 activations)");
        assert!(e < 0.03, "T16 decode vs reference: {e}");
        // a layer with nothing resident gives zeros
        st.run_device_fast(&g, 0, st.sel.ptr, st.wts.ptr, k, t, st.x.ptr, st.out.ptr).unwrap();
        let z: Vec<f32> = g.download(st.out.ptr, t * c.dim).unwrap();
        assert!(z.iter().all(|&v| v == 0.0), "empty layer");
    }
}

/// Decode speed of one layer's resident experts as run_device_fast runs them (T = 1, K = 6 distinct picks out of 64
/// experts, ~1 GB of weights so no call finds them in L2): T16 1.5 / 2 against a UD-IQ1_M-like codebook mix,
/// interleaved A B B A with pauses (the 3090's power cap throttles long loops).
/// cargo test --release --features cuda --test gpu_t16 -- --ignored --nocapture t16_decode_speed
#[test]
#[ignore]
fn t16_decode_speed() {
    use dsx::experts::{ExpCfg, GpuExperts, RawExpert};
    let g = Gpu::new(0).unwrap();
    let mut seed = 0xfeed_beef_1234_5678u64;
    let c = ExpCfg { n_layers: 1, n_experts: 64, inter: 2304, dim: 5120, lim: 10.0, rot: 0 };
    let iq = |ty: GType, rows: usize, k: usize, seed: &mut u64| -> Vec<u8> {
        let (_, b) = ty.block();
        let mut v: Vec<u8> = (0..ty.row_bytes(k) * rows).map(|_| rng(seed) as u8).collect();
        for blk in v.chunks_exact_mut(b) {
            blk[..2].copy_from_slice(&half::f16::from_f32(0.01).to_le_bytes());
        }
        v
    };
    let mix = [(GType::Iq1S, GType::Iq2Xxs), (GType::Iq2Xxs, GType::Iq2Xs), (GType::Iq1S, GType::Iq2S), (GType::Iq2Xxs, GType::Iq2Xxs)];
    // one type per role per layer for T16; the IQ store mixes types across experts like UD's files
    // T16_SPEED_K="gu,dn" bits per weight of the T16 store (default 1.5,2: the first half's layout)
    let ks: Vec<f32> = std::env::var("T16_SPEED_K").unwrap_or("1.5,2".into()).split(',').map(|v| v.parse().unwrap()).collect();
    let kty = |k: f32| if k == 1.5 { GType::T16K15 } else if k == 2.0 { GType::T16K2 } else { GType::T16K25 };
    let (tgu, tdn) = (kty(ks[0]), kty(ks[1]));
    println!("T16 store: gate/up K {}, down K {}", ks[0], ks[1]);
    let t16m: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> = (0..64).map(|_| (random_matrix(tgu, c.dim, c.inter, &mut seed), random_matrix(tgu, c.dim, c.inter, &mut seed), random_matrix(tdn, c.inter, c.dim, &mut seed))).collect();
    let iqm: Vec<(GType, GType, Vec<u8>, Vec<u8>, Vec<u8>)> = (0..64).map(|e| {
        let (gu, dn) = mix[e % mix.len()];
        (gu, dn, iq(gu, c.inter, c.dim, &mut seed), iq(gu, c.inter, c.dim, &mut seed), iq(dn, c.dim, c.inter, &mut seed))
    }).collect();
    let st_t16 = GpuExperts::build(&g, c, &t16m.iter().enumerate().map(|(e, m)| ((0, e), RawExpert { gate: &m.0, up: &m.1, down: &m.2, gu: tgu, dn: tdn })).collect::<Vec<_>>(), true).unwrap();
    let st_iq = GpuExperts::build(&g, c, &iqm.iter().enumerate().map(|(e, m)| ((0, e), RawExpert { gate: &m.2, up: &m.3, down: &m.4, gu: m.0, dn: m.1 })).collect::<Vec<_>>(), true).unwrap();
    let calls = 400;
    // T16_SPEED_T tokens a call (6 picks each, all distinct experts while 6 T <= 64): more warps per launch
    let tt: usize = std::env::var("T16_SPEED_T").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    let sels: Vec<Vec<i32>> = (0..calls).map(|i| (0..6 * tt).map(|j| ((i * 7 + j * 11) % 64) as i32).collect()).collect();
    let x: Vec<f32> = (0..tt * c.dim).map(|i| (i % 100) as f32 * 0.01 - 0.5).collect();
    let wts = vec![1.0f32 / 6.0; 6 * tt];
    let mut res: Vec<(&str, f64)> = vec![];
    for name in ["T16", "IQ", "IQ", "T16", "T16", "IQ"] {
        std::thread::sleep(std::time::Duration::from_secs(3));
        let st = if name == "T16" { &st_t16 } else { &st_iq };
        g.htod_sync(st.x.ptr, &x).unwrap();
        g.htod_sync(st.wts.ptr, &wts).unwrap();
        let dsel: Vec<Buf> = sels.iter().map(|s| g.upload(s).unwrap()).collect();
        let ev = (g.timed_event().unwrap(), g.timed_event().unwrap());
        st.run_device_fast(&g, 0, dsel[0].ptr, st.wts.ptr, 6, tt, st.x.ptr, st.out.ptr).unwrap();
        g.sync().unwrap();
        g.record(ev.0, false).unwrap();
        for s in &dsel {
            st.run_device_fast(&g, 0, s.ptr, st.wts.ptr, 6, tt, st.x.ptr, st.out.ptr).unwrap();
        }
        g.record(ev.1, false).unwrap();
        g.sync().unwrap();
        let us = g.elapsed_ms(ev.0, ev.1).unwrap() as f64 * 1e3 / calls as f64;
        println!("{name}: {us:.1} us per layer call (6 experts, T = 1)");
        res.push((name, us));
    }
    for n in ["T16", "IQ"] {
        let v: Vec<f64> = res.iter().filter(|r| r.0 == n).map(|r| r.1).collect();
        println!("{n}: mean {:.1} us", v.iter().sum::<f64>() / v.len() as f64);
    }
}

/// Bit pattern of the T16 decode output (run_device_fast) over the three layouts and T = 1 / 2 / 4, as one hash per
/// case: kernel restructurings must leave it unchanged (compare the printed hashes before and after).
/// cargo test --release --features cuda --test gpu_t16 -- --nocapture t16_decode_bits
#[test]
fn t16_decode_bits() {
    use dsx::experts::{ExpCfg, GpuExperts, RawExpert};
    let g = Gpu::new(0).unwrap();
    let c = ExpCfg { n_layers: 1, n_experts: 8, inter: 2304, dim: 5120, lim: 10.0, rot: 0 };
    let mut all = 0xcbf2_9ce4_8422_2325u64;
    for (gu, dn) in [(GType::T16K15, GType::T16K2), (GType::T16K2, GType::T16K25), (GType::T16K2, GType::T16K2)] {
        let mut seed = 0x5eed_b175_0000_0001u64;
        let mats: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> = (0..8)
            .map(|_| (random_matrix(gu, c.dim, c.inter, &mut seed), random_matrix(gu, c.dim, c.inter, &mut seed), random_matrix(dn, c.inter, c.dim, &mut seed)))
            .collect();
        let items: Vec<((usize, usize), RawExpert)> = mats.iter().enumerate().map(|(e, m)| ((0, e), RawExpert { gate: &m.0, up: &m.1, down: &m.2, gu, dn })).collect();
        let st = GpuExperts::build(&g, c, &items, true).unwrap();
        let k = 6;
        for t in [1usize, 2, 4] {
            let sel: Vec<i32> = (0..t * k).map(|j| ((j * 5 + j / k) % 8) as i32).collect();
            let wts: Vec<f32> = (0..t * k).map(|_| 0.05 + (rng(&mut seed) % 100) as f32 * 0.003).collect();
            let x: Vec<f32> = (0..t * c.dim).map(|_| (rng(&mut seed) % 2001) as f32 / 1000.0 - 1.0).collect();
            g.htod_sync(st.sel.ptr, &sel).unwrap();
            g.htod_sync(st.wts.ptr, &wts).unwrap();
            g.htod_sync(st.x.ptr, &x).unwrap();
            st.run_device_fast(&g, 0, st.sel.ptr, st.wts.ptr, k, t, st.x.ptr, st.out.ptr).unwrap();
            let got: Vec<f32> = g.download(st.out.ptr, t * c.dim).unwrap();
            let mut h = 0xcbf2_9ce4_8422_2325u64;
            for v in &got {
                h = (h ^ v.to_bits() as u64).wrapping_mul(0x100_0000_01b3);
            }
            all = (all ^ h).wrapping_mul(0x100_0000_01b3);
            assert!(got.iter().any(|&v| v != 0.0) && got.iter().all(|v| v.is_finite()));
            println!("T16 decode bits {gu:?}/{dn:?} T={t}: {h:016x}");
        }
    }
    println!("T16 decode bits, all cases: {all:016x}");
}
