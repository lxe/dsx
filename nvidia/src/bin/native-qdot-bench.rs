//! Single-thread Q2_K / IQ3_XXS matvec benchmark; no model or GPU required.
//! cargo run --release --locked -p dsx --bin native-qdot-bench -- 3
use dsx::cpu::qdot::{QAct, dot_rows, isa};
use dsx::gguf::GType;
use half::f16;
use std::{hint::black_box, time::Instant};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

fn main() {
    let repeats: usize = std::env::args()
        .nth(1)
        .map(|s| s.parse().expect("positive repetition count"))
        .unwrap_or(3);
    assert!((1..=1000).contains(&repeats));
    println!(
        "ISA={:?}; one thread, T=1; activation quantization and allocation excluded",
        isa()
    );
    println!("format,rows,cols,pool,repeats,median_ms_per_matrix,nominal_weight_GB_s");
    for ty in [GType::Q2K, GType::Iq3Xxs] {
        for (nr, cols) in [(2304, 5120), (5120, 2304)] {
            for pool in [1, 32] {
                let mut rng = Rng(0x51ed270b9a3c4e1d);
                let matrices: Vec<Vec<u8>> = (0..pool)
                    .map(|_| {
                        let mut data: Vec<u8> = (0..ty.row_bytes(cols) * nr)
                            .map(|_| rng.next() as u8)
                            .collect();
                        let (_, block_bytes) = ty.block();
                        for block in data.chunks_exact_mut(block_bytes) {
                            let offset = if ty == GType::Q2K { 80 } else { 0 };
                            block[offset..offset + 2]
                                .copy_from_slice(&f16::from_f32(0.02).to_le_bytes());
                            if ty == GType::Q2K {
                                block[82..84].copy_from_slice(&f16::from_f32(0.003).to_le_bytes());
                            }
                        }
                        data
                    })
                    .collect();
                let x: Vec<f32> = (0..cols)
                    .map(|_| (rng.next() % 255) as f32 - 127.0)
                    .collect();
                let act = QAct::new(&x, 1, cols, &[ty]);
                let mut out = vec![0.0; nr];
                for w in &matrices {
                    dot_rows(ty, w, cols, &act, &mut out);
                    assert!(out.iter().all(|x| x.is_finite()));
                    black_box(&out);
                }
                let mut samples = Vec::new();
                for _ in 0..5 {
                    let start = Instant::now();
                    for _ in 0..repeats {
                        for w in &matrices {
                            dot_rows(ty, black_box(w), cols, black_box(&act), black_box(&mut out));
                            black_box(&out);
                        }
                    }
                    samples.push(start.elapsed().as_secs_f64() / (pool * repeats) as f64);
                }
                samples.sort_by(f64::total_cmp);
                let seconds = samples[2];
                println!(
                    "{ty:?},{nr},{cols},{pool},{repeats},{:.6},{:.4}",
                    seconds * 1000.0,
                    matrices[0].len() as f64 / seconds / 1e9
                );
            }
        }
    }
}
