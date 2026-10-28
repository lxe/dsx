#![cfg(feature = "cuda")]
//! DRAM load latency for Little's law (bytes in flight per SM = GB/s x latency / SMs): one thread chases a random
//! cycle through 1 GiB at a 4 KiB-or-more stride (no cache or TLB reach), ns per dependent load.
//! cargo test --release --features cuda --test gpu_latency -- --ignored --nocapture
use dsx::gpu::*;

#[test]
#[ignore]
fn dram_load_latency() {
    let g = Gpu::new(0).unwrap();
    let n = 1usize << 28; // u32 words: 1 GiB
    let stride = 1024 + 16; // words between visited slots (4160 B): different pages, rows and sectors
    let slots = n / stride;
    let mut seed = 0x1234_5678_9abc_def0u64;
    let mut perm: Vec<usize> = (0..slots).collect();
    for i in (1..slots).rev() {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        perm.swap(i, (seed % (i as u64 + 1)) as usize);
    }
    let mut next = vec![0u32; n];
    for w in 0..slots {
        next[perm[w] * stride] = (perm[(w + 1) % slots] * stride) as u32;
    }
    let d = g.upload(&next).unwrap();
    let out = g.alloc(4).unwrap();
    let k = g.kernel("chase").unwrap();
    let (e0, e1) = (g.timed_event().unwrap(), g.timed_event().unwrap());
    let steps = 200_000;
    g.launch(k, (1, 1, 1), 1, 0, &[Arg::P(d.ptr), Arg::I(10_000), Arg::P(out.ptr)]).unwrap();
    g.record(e0, false).unwrap();
    g.launch(k, (1, 1, 1), 1, 0, &[Arg::P(d.ptr), Arg::I(steps), Arg::P(out.ptr)]).unwrap();
    g.record(e1, false).unwrap();
    g.sync().unwrap();
    let ns = g.elapsed_ms(e0, e1).unwrap() as f64 * 1e6 / steps as f64;
    println!("DRAM load latency (dependent, 1 GiB random): {ns:.0} ns per load");
}
