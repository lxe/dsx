#![cfg(feature = "cuda")]
//! Round trip GPU -> CPU -> GPU through flags in mapped host memory: what one layer's expert handoff costs.
use dsx::gpu::*;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Instant;

#[test]
#[ignore]
fn handshake_latency() {
    for dev in 0..2 {
        let Ok(g) = Gpu::new(dev) else { continue };
        let hb = HostBuf::new(4096).unwrap();
        let d = hb.dev_ptr(&g).unwrap();
        let flags = hb.host_ptr() as *const AtomicU32;
        let (to_cpu, to_gpu) = unsafe { (&*flags, &*flags.add(16)) };
        let sig = g.kernel("signal_host").unwrap();
        let n = 2000u32;
        let host = std::thread::spawn({
            let (a, b) = (to_cpu as *const AtomicU32 as usize, to_gpu as *const AtomicU32 as usize);
            move || {
                let (a, b) = unsafe { (&*(a as *const AtomicU32), &*(b as *const AtomicU32)) };
                for i in 1..=n {
                    while a.load(Ordering::Acquire) < i {
                        std::hint::spin_loop();
                    }
                    b.store(i, Ordering::Release);
                }
            }
        });
        let t = Instant::now();
        for i in 1..=n {
            g.launch(sig, (1, 1, 1), 1, 0, &[Arg::P(d), Arg::I(i as i32)]).unwrap();
            g.wait_value(d + 64, i).unwrap();
        }
        g.sync().unwrap();
        let us = t.elapsed().as_secs_f64() * 1e6 / n as f64;
        host.join().unwrap();
        eprintln!("GPU{dev} {}: GPU->CPU->GPU round trip {us:.1} us", g.name);
    }
}
