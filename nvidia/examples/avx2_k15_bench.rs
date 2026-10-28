//! Reproducible direct-kernel comparison; includes no model weights.
#[cfg(target_arch = "x86_64")]
use dsx::{cpu::{pool, t16k}, t16};
#[cfg(target_arch = "x86_64")]
use std::{hint::black_box, time::{Duration, Instant}};
#[cfg(target_arch = "x86_64")]
fn next(seed: &mut u64) -> u64 { *seed ^= *seed << 13; *seed ^= *seed >> 7; *seed ^= *seed << 17; *seed }
#[cfg(target_arch = "x86_64")]
fn main() {
    assert!(dsx::cpu::qdot::Isa::Avx2.supported());
    if let Some(cpu) = std::env::args().nth(1) { assert!(pool::pin_to(cpu.parse().expect("expected a CPU index"))); }
    let (k, n, calls) = (5120, 2304, 4096);
    let mut seed = 0x51d4_a33f_87a9_cc09;
    let matrices: Vec<Vec<u8>> = (0..48).map(|_| (0..t16::matrix_bytes(t16::ty_for_k2(3).unwrap(),k,n)).map(|_|next(&mut seed) as u8).collect()).collect();
    let mut a = t16k::Act::default();
    a.x8 = (0..k/2).map(|_| { let q0=next(&mut seed) as i8 as i16 as u16 as u32; let q1=next(&mut seed) as i8 as i16 as u16 as u32; (q0 | q1 << 16) as i32 }).collect();
    a.d = vec![0.017; k/256]; a.s = vec![32; k/256];
    for streaming in [false, true] {
        let mut reference = None;
        for (round, candidate) in [false,true,true,false].into_iter().enumerate() {
            let mut out = vec![0.0;64];
            let mut call = |i:usize| {
                let (m,c) = if streaming {(i%48,4*((i/48)%(n/64)))} else {(0,0)};
                unsafe { if candidate {t16k::avx2::gemv_k15(&matrices[m],k,n,&a,c,c+4,&mut out,0,k/256)}
                    else {t16k::avx2::gemv_generic(&matrices[m],3,k,n,&a,c,c+4,&mut out,0,k/256)} }
                black_box(&out);
            };
            for i in 0..48*(n/64) {call(i);}
            let warm = Instant::now(); let mut i=0;
            while warm.elapsed()<Duration::from_millis(150) {call(i);i+=1;}
            let start=Instant::now(); for i in 0..calls {call(i);}
            let ns=start.elapsed().as_secs_f64()*1e9/calls as f64;
            let bits:Vec<_>=out.iter().map(|v|v.to_bits()).collect();
            if let Some(ref want)=reference {assert_eq!(want,&bits);} else {reference=Some(bits);}
            println!("{}",serde_json::json!({"round":round,"streaming":streaming,"candidate":candidate,"ns_per_call":ns,"calls":calls,"float_bits_match":true}));
        }
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn main() { eprintln!("This benchmark requires an AVX2-capable x86_64 CPU."); std::process::exit(1); }
