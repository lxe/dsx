// rustc -O history_tail.rs -o history_tail && ./history_tail
// Isolated CPU cost and exact Engram-tail equivalence, independent of HIP.
use std::{hint::black_box, time::Instant};
fn tails(hist: &[i64], n: usize, drafts: usize, short: bool) -> Vec<Vec<i64>> {
    let start = if short { hist.len().saturating_sub(n.saturating_sub(1)) } else { 0 };
    let mut h = hist[start..].to_vec();
    let mut out = Vec::new();
    for t in 0..drafts {
        h.push(t as i64 + 1234);
        out.push(h[h.len().saturating_sub(n)..].to_vec());
    }
    out
}
fn main() {
    let mut cases = 0;
    for len in 0..=200 {
        let hist: Vec<i64> = (0..len).collect();
        for n in 0..=16 {
            for drafts in 0..=8 {
                assert_eq!(tails(&hist, n, drafts, false), tails(&hist, n, drafts, true));
                cases += 1;
            }
        }
    }
    println!("{{\"equivalence_cases\":{cases},\"all_equal\":true}}");
    for len in [4096, 65536, 262144] {
        let hist: Vec<i64> = (0..len as i64).collect();
        for round in 0..4 {
            for short in [false, true] {
                let start = Instant::now();
                for _ in 0..1000 {
                    // Match four active slots with four fed tokens per slot.
                    for _ in 0..4 { black_box(tails(black_box(&hist), black_box(3), black_box(4), short)); }
                }
                println!("{{\"context\":{len},\"round\":{round},\"short\":{short},\"us_per_four_slots\":{:.3}}}", start.elapsed().as_secs_f64() * 1000.0);
            }
        }
    }
}
