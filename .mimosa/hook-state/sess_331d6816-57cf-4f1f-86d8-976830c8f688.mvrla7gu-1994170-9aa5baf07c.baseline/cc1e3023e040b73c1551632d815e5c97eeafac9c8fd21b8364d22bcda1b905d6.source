//! Full-prompt prefill timing with fixed placement and no reused KV prefix.
use anyhow::{Result, ensure};
use clap::Parser;
use dsx::{
    gpumodel::{GpuModel, Placement},
    model::Model,
};
use serde_json::json;
use std::{io::Write, path::PathBuf, time::Instant};
#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    rank: PathBuf,
    #[arg(long)]
    ids: PathBuf,
    #[arg(long, default_value_t = 262144)]
    ctx: usize,
    #[arg(long, default_value_t = 8192)]
    chunk: usize,
    #[arg(long, default_value_t = 110.)]
    arena_gb: f64,
    #[arg(long, default_value_t=-1.5, allow_negative_numbers=true)]
    main_gb: f64,
    #[arg(long, default_value_t=-1., allow_negative_numbers=true)]
    aux_gb: f64,
    #[arg(long, default_value_t = 3)]
    rounds: usize,
    /// Last round's complete final-token logits, little-endian F32.
    #[arg(long)]
    dump_logits: Option<PathBuf>,
}
fn main() -> Result<()> {
    let a = Args::parse();
    ensure!(
        a.rounds >= 2 && a.chunk > 0,
        "at least two rounds and a nonzero chunk are required"
    );
    ensure!(
        std::env::var("DSX_ADAPT").is_ok_and(|v| v == "0"),
        "set DSX_ADAPT=0 to freeze placement"
    );
    let ids: Vec<u32> = std::fs::read_to_string(&a.ids)?
        .split(|c: char| c == ',' || c.is_whitespace())
        .filter(|s| !s.is_empty())
        .map(str::parse)
        .collect::<std::result::Result<_, _>>()?;
    ensure!(
        !ids.is_empty() && ids.len() < a.ctx,
        "prompt must fit context"
    );
    let m = Model::load(&a.model, None)?;
    let mut gm = GpuModel::with_placement(
        &m,
        -1,
        a.ctx,
        Placement {
            gpu0_gb: a.main_gb,
            aux_gb: a.aux_gb,
            arena_gb: a.arena_gb,
            prefill_chunk: a.chunk,
            rank: Some(a.rank),
        },
    )?;
    let pb = gm.pb.take().expect("prefill buffers");
    let mut elapsed = 0.;
    for round in 0..a.rounds {
        gm.reset();
        gm.g.sync()?;
        let start = Instant::now();
        let logits = gm.prefill(&pb, &ids)?;
        gm.g.sync()?;
        let seconds = start.elapsed().as_secs_f64();
        ensure!(logits.iter().all(|v| v.is_finite()), "nonfinite output");
        let mut hash = 0xcbf29ce484222325u64;
        for v in &logits {
            for b in v.to_le_bytes() {
                hash ^= b as u64;
                hash = hash.wrapping_mul(0x100000001b3);
            }
        }
        println!(
            "{}",
            json!({"event":"prefill_bench","round":round,"warm":round>0,"new_tokens":ids.len(),"cached_tokens":0,"chunk":a.chunk,"ctx":a.ctx,"elapsed_s":seconds,"tok_s":ids.len() as f64/seconds,"logit_fingerprint":format!("{hash:016x}")})
        );
        if round > 0 {
            elapsed += seconds;
        }
        if round + 1 == a.rounds {
            if let Some(path) = &a.dump_logits {
                let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
                for value in logits {
                    f.write_all(&value.to_le_bytes())?;
                }
                f.flush()?;
            }
        }
    }
    println!(
        "{}",
        json!({"event":"summary","warm_rounds":a.rounds-1,"new_tokens":ids.len()*(a.rounds-1),"elapsed_s":elapsed,"warm_tok_s":(ids.len()*(a.rounds-1)) as f64/elapsed})
    );
    Ok(())
}
