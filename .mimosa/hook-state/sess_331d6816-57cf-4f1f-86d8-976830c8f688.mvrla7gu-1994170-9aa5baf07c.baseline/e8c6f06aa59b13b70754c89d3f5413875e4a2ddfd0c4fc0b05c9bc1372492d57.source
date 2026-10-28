#![cfg(feature = "cuda")]
//! The decode handoff in isolation, closer to the engine: GPU work between publish and wait, a pinned E-core
//! worker that copies x, 20 KB of activations through mapped memory. Prints the round trip per variant.
use dsx::cpu::pool::pin_to;
use dsx::gpu::*;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Instant;

#[test]
#[ignore]
fn handshake_variants() {
    let g = Gpu::new(0).unwrap();
    let hb = HostBuf::new(1 << 21).unwrap();
    let d = hb.dev_ptr(&g).unwrap();
    let base = hb.host_ptr() as usize;
    let sig = g.kernel("signal_host").unwrap();
    let sleep = g.kernel("gpu_sleep").unwrap();
    let n = 400u32;
    for &(pin, gap_us, copy) in &[(None, 0u32, false), (Some(16usize), 0, false), (Some(16), 100, false), (Some(16), 100, true), (Some(4), 100, true), (None, 100, true)] {
        let flags = base as *const AtomicU32;
        unsafe {
            (*flags).store(0, Ordering::SeqCst);
            (*flags.add(16)).store(0, Ordering::SeqCst);
        }
        let host = std::thread::spawn(move || {
            if let Some(c) = pin {
                pin_to(c);
            }
            let flags = base as *const AtomicU32;
            let (a, b) = unsafe { (&*flags, &*flags.add(16)) };
            let src = unsafe { std::slice::from_raw_parts((base + 4096) as *const f32, 5120) };
            let mut x = vec![0f32; 5120];
            for i in 1..=n {
                while a.load(Ordering::Acquire) < i {
                    std::hint::spin_loop();
                }
                if copy {
                    x.copy_from_slice(src);
                }
                b.store(i, Ordering::Release);
            }
            x[0]
        });
        let t = Instant::now();
        for i in 1..=n {
            g.launch(sig, (1, 1, 1), 1, 0, &[Arg::P(d), Arg::I(i as i32)]).unwrap();
            if gap_us > 0 {
                g.launch(sleep, (1, 1, 1), 1, 0, &[Arg::I((gap_us * 1000) as i32)]).unwrap();
            }
            g.wait_value(d + 64, i).unwrap();
        }
        g.sync().unwrap();
        let us = t.elapsed().as_secs_f64() * 1e6 / n as f64;
        host.join().unwrap();
        eprintln!("pin {pin:?} gap {gap_us} us copy {copy}: {us:.1} us per round (minus gap = {:.1})", us - gap_us as f64);
    }
}
