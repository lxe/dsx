//! dsx-amd: DeepSeek-V4.1-Flash-1017 on one AMD Instinct MI325X.

mod ck;
mod decode;
mod export;
mod hip;
mod load;
mod prefill;
mod selftest;
mod serve;
mod verify;
mod batch;
mod runner;
mod spec;
mod telemetry;
mod dspark;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use decode::{Decoder, NE};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Parser)]
#[command(about = "DeepSeek-V4.1-Flash-1017 on one MI325X")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Write the Engram hashing constants (dsx-engram.json) next to the checkpoint, from a dsx GGUF export.
    EngramJson {
        #[arg(long)]
        gguf: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// CPU reference (FP8/FP4 checkpoint) vs the GPU decoder, token by token, on the same inputs.
    Check {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, value_delimiter = ',')]
        ids: Vec<u32>,
        #[arg(short, default_value_t = 4)]
        n: usize,
        #[arg(long)]
        rank: Option<PathBuf>,
        /// reuse a previous reference run (written by an earlier check to this path)
        #[arg(long)]
        ref_cache: Option<PathBuf>,
    },
    /// Decode speed at a given context (caches filled with synthetic values).
    Bench {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, value_delimiter = ',', default_value = "4096")]
        ctx: Vec<usize>,
        #[arg(long, default_value_t = 32)]
        steps: usize,
        #[arg(long)]
        rank: Option<PathBuf>,
    },
    /// Quality on Lucebox's 1017 eval (8 x 1024 tokens, FP8 teacher top-256 log-probs), teacher-forced through decode:
    /// KL(teacher || model) over the teacher's top-256, top-1 agreement, perplexity. Reads heldout.i32,
    /// teacher_idx.i32, teacher_lp.f32 (convert the .npy/.npz once with tools/eval_bins.py).
    Eval {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        eval: PathBuf,
        #[arg(long, default_value_t = 8)]
        seqs: usize,
        #[arg(long, default_value_t = 1024)]
        len: usize,
        #[arg(long)]
        rank: Option<PathBuf>,
    },
    /// Kernel self-tests on synthetic inputs against CPU implementations.
    Selftest,
    /// Load a build and write it as one GGUF (split it with `dsx split`); load it later with DSXA_WEIGHTS=<file>.
    Export {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        rank: Option<PathBuf>,
        #[arg(long)]
        out: PathBuf,
    },
    /// OpenAI / Anthropic compatible API server (dsx's server, llama.cpp web UI at /).
    Serve {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        rank: Option<PathBuf>,
        #[arg(long, default_value_t = 131072)]
        ctx: usize,
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        #[arg(long, default_value_t = 8090)]
        port: u16,
        #[arg(long, default_value_t = 32768)]
        max_tokens: usize,
        /// batched runner: this many concurrent sequences (0: one conversation at a time)
        #[arg(long, default_value_t = 0)]
        slots: usize,
        /// batched runner: context per slot (default: --ctx)
        #[arg(long)]
        batch_ctx: Option<usize>,
        /// admin dashboard and `/stats` (JSON, Volta's format) on this port, on --host; 0 = off (batched runner only)
        #[arg(long, default_value_t = 0)]
        admin_port: u16,
        /// sampling temperature for requests that don't set one (0 = greedy)
        #[arg(long, default_value_t = 1.0)]
        temperature: f32,
        /// top-p for requests that don't set one
        #[arg(long, default_value_t = 1.0)]
        top_p: f32,
        /// require this key on every API request (`Authorization: Bearer <key>` or `x-api-key: <key>`);
        /// `openssl rand -hex 32` makes one. Health checks and the web UI's files stay open.
        #[arg(long, env = "DSX_API_KEY")]
        api_key: Option<String>,
    },
    /// Development loop: load once, then run commands from a file as it appears (one line per command):
    /// `reload PATH.hsaco`, `bench CTX STEPS`, `prof CTX`, `eval DIR SEQS`, `check REFCACHE`, `quit`.
    Dev {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        rank: Option<PathBuf>,
        #[arg(long, default_value_t = 262200)]
        ctx: usize,
        /// command file, consumed (renamed to .done) after each command
        #[arg(long, default_value = "/root/dev.cmd")]
        cmd: PathBuf,
    },
    /// Greedy generation from token ids.
    Gen {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, value_delimiter = ',')]
        ids: Vec<u32>,
        #[arg(short, default_value_t = 32)]
        n: usize,
        #[arg(long)]
        rank: Option<PathBuf>,
        #[arg(long, default_value_t = 8192)]
        ctx: usize,
    },
}

fn read_rank(p: &Option<PathBuf>) -> Result<Vec<(usize, usize)>> {
    let Some(p) = p else { return Ok(vec![]) };
    let v: serde_json::Value = serde_json::from_slice(&std::fs::read(p)?)?;
    Ok(v.as_array().context("rank array")?.iter().map(|r| (r[0].as_u64().unwrap() as usize, r[1].as_u64().unwrap() as usize)).collect())
}

fn top(logits: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut v: Vec<(u32, f32)> = logits.iter().enumerate().map(|(i, &x)| (i as u32, x)).collect();
    v.select_nth_unstable_by(k - 1, |a, b| b.1.partial_cmp(&a.1).unwrap());
    v.truncate(k);
    v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    v
}

fn log_softmax(l: &[f32]) -> Vec<f64> {
    let mx = l.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    let lse = mx + l.iter().map(|&v| (v as f64 - mx).exp()).sum::<f64>().ln();
    l.iter().map(|&v| v as f64 - lse).collect()
}

fn engram_json(gguf: &Path, out: &Path) -> Result<()> {
    let g = dsx_core::gguf::Gguf::open(gguf)?;
    let c = dsx_core::model::Config::from_gguf(&g)?;
    let per_layer = (c.engram_ngram - 1) * c.engram_heads;
    let primes: Vec<Vec<u64>> = c.engram_primes.chunks(per_layer).map(|x| x.to_vec()).collect();
    let mults: Vec<Vec<u64>> = c.engram_multipliers.chunks(c.engram_ngram).map(|x| x.to_vec()).collect();
    let j = serde_json::json!({ "pad_id": c.engram_pad, "token_map": c.engram_token_map, "primes": primes, "multipliers": mults });
    std::fs::write(out, serde_json::to_vec(&j)?)?;
    eprintln!("wrote {}", out.display());
    Ok(())
}

fn bench(dec: &mut Decoder, c: usize, steps: usize, prof: bool) -> Result<()> {
    dec.fill_random()?;
    dec.set_pos(c)?;
    let mut tok = 1000u32;
    let mut ts = vec![];
    for s in 0..steps {
        let us = dec.time_step(tok)?;
        tok = (tok * 7 + 13) % 100000;
        if s >= 2 {
            ts.push(us);
        }
    }
    ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = ts[ts.len() / 2];
    println!("ctx {c}: decode step {med:.0} us median ({:.1} tok/s), min {:.0} max {:.0}", 1e6 / med, ts[0], ts[ts.len() - 1]);
    if prof {
        let p = dec.step_prof(tok)?;
        let tot: f32 = p.iter().map(|x| x.1).sum();
        println!("  eager step: {tot:.0} us in {} launches", p.iter().map(|x| x.2).sum::<usize>());
        for (n, t, k) in p.iter().take(30) {
            println!("  {n:<28} {t:>8.1} us {k:>5} launches {:>6.2} us each", t / *k as f32);
        }
    }
    Ok(())
}

fn eval_run(dec: &mut Decoder, eval: &Path, seqs: usize, len: usize) -> Result<String> {
    eval_run_m(dec, eval, seqs, len, false)
}

/// eval_run through the prompt path in chunks of `chunk` tokens (logits of every position from chunk_logits)
fn ceval_run(dec: &mut Decoder, eval: &Path, seqs: usize, len: usize, chunk: usize) -> Result<String> {
    let rd = |f: &str| std::fs::read(eval.join(f));
    let toks: Vec<u32> = rd("heldout.i32")?.chunks(4).map(|b| i32::from_le_bytes(b.try_into().unwrap()) as u32).collect();
    let tidx: Vec<i32> = rd("teacher_idx.i32")?.chunks(4).map(|b| i32::from_le_bytes(b.try_into().unwrap())).collect();
    let tlp: Vec<f32> = rd("teacher_lp.f32")?.chunks(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
    let topk = 256;
    dec.build_bdense()?;
    let (mut nll, mut kl, mut agree, mut cnt) = (0f64, 0f64, 0usize, 0usize);
    let t0 = Instant::now();
    for sq in 0..seqs {
        dec.reset()?;
        let seq = &toks[sq * len..(sq + 1) * len];
        for c0 in (0..len - 1).step_by(chunk) {
            let c1 = (c0 + chunk).min(len - 1);
            dec.prefill(&seq[c0..c1])?;
            dec.chunk_logits(c1 - c0, &mut |i, l| {
                let p = c0 + i;
                let lq = log_softmax(l);
                nll -= lq[seq[p + 1] as usize];
                let row = sq * (len - 1) + p;
                let (ti, tl) = (&tidx[row * topk..(row + 1) * topk], &tlp[row * topk..(row + 1) * topk]);
                kl += ti.iter().zip(tl).map(|(&i, &lt)| (lt as f64).exp() * (lt as f64 - lq[i as usize])).sum::<f64>();
                let am = l.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |a, (i, &v)| if v > a.1 { (i, v) } else { a }).0;
                agree += (am as i32 == ti[0]) as usize;
                cnt += 1;
                Ok(())
            })?;
        }
    }
    Ok(format!("{{\"chunk\": {chunk}, \"tokens\": {cnt}, \"ppl\": {:.4}, \"kl_top{topk}\": {:.6}, \"top1_agree\": {:.6}, \"s\": {:.1}}}", (nll / cnt as f64).exp(), kl / cnt as f64, agree as f64 / cnt as f64, t0.elapsed().as_secs_f64()))
}

/// `prompt`: every token through the prompt path (a one-token prefill) instead of the decode step
fn eval_run_m(dec: &mut Decoder, eval: &Path, seqs: usize, len: usize, prompt: bool) -> Result<String> {
    let rd = |f: &str| std::fs::read(eval.join(f));
    let toks: Vec<u32> = rd("heldout.i32")?.chunks(4).map(|b| i32::from_le_bytes(b.try_into().unwrap()) as u32).collect();
    let tidx: Vec<i32> = rd("teacher_idx.i32")?.chunks(4).map(|b| i32::from_le_bytes(b.try_into().unwrap())).collect();
    let tlp: Vec<f32> = rd("teacher_lp.f32")?.chunks(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
    let topk = 256;
    let (mut nll, mut kl, mut agree, mut cnt) = (0f64, 0f64, 0usize, 0usize);
    let t0 = Instant::now();
    for sq in 0..seqs {
        dec.reset()?;
        let seq = &toks[sq * len..(sq + 1) * len];
        for p in 0..len - 1 {
            if prompt { dec.prefill(&seq[p..p + 1])?; } else { dec.step(seq[p])?; }
            let l = dec.logits()?;
            let lq = log_softmax(&l);
            nll -= lq[seq[p + 1] as usize];
            let row = sq * (len - 1) + p;
            let (ti, tl) = (&tidx[row * topk..(row + 1) * topk], &tlp[row * topk..(row + 1) * topk]);
            kl += ti.iter().zip(tl).map(|(&i, &lt)| (lt as f64).exp() * (lt as f64 - lq[i as usize])).sum::<f64>();
            let am = l.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |a, (i, &v)| if v > a.1 { (i, v) } else { a }).0;
            agree += (am as i32 == ti[0]) as usize;
            cnt += 1;
        }
        eprintln!("seq {sq}: running ppl {:.3} kl {:.4} top1 {:.4} ({:.0} s)", (nll / cnt as f64).exp(), kl / cnt as f64, agree as f64 / cnt as f64, t0.elapsed().as_secs_f64());
    }
    Ok(format!("{{\"tokens\": {cnt}, \"nll\": {:.6}, \"ppl\": {:.4}, \"kl_top{topk}\": {:.6}, \"top1_agree\": {:.6}}}", nll / cnt as f64, (nll / cnt as f64).exp(), kl / cnt as f64, agree as f64 / cnt as f64))
}

/// eval_run through batched decode: the SEQS sequences (each REP times, so B = SEQS * REP) decoded together from
/// empty slots, teacher-forced, one batched step per position
fn beval_run(dec: &mut Decoder, eval: &Path, seqs: usize, rep: usize, len: usize) -> Result<String> {
    let rd = |f: &str| std::fs::read(eval.join(f));
    let toks: Vec<u32> = rd("heldout.i32")?.chunks(4).map(|b| i32::from_le_bytes(b.try_into().unwrap()) as u32).collect();
    let tidx: Vec<i32> = rd("teacher_idx.i32")?.chunks(4).map(|b| i32::from_le_bytes(b.try_into().unwrap())).collect();
    let tlp: Vec<f32> = rd("teacher_lp.f32")?.chunks(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
    let topk = 256;
    let bsz = seqs * rep;
    let mut b = dec.new_batch(bsz, len + 16)?;
    let slots: Vec<usize> = (0..bsz).collect();
    let (mut nll, mut kl, mut agree, mut cnt) = (vec![0f64; rep], vec![0f64; rep], vec![0usize; rep], 0usize);
    let t0 = Instant::now();
    let mut tstep = 0f64;
    for p in 0..len - 1 {
        let feed: Vec<u32> = (0..bsz).map(|i| toks[(i % seqs) * len + p]).collect();
        let ts = Instant::now();
        dec.batch_step(&mut b, &slots, &feed)?;
        tstep += ts.elapsed().as_secs_f64();
        for i in 0..bsz {
            let (sq, r) = (i % seqs, i / seqs);
            let l = dec.batch_logits(&b, i)?;
            let lq = log_softmax(&l);
            nll[r] -= lq[toks[sq * len + p + 1] as usize];
            let row = sq * (len - 1) + p;
            let (ti, tl) = (&tidx[row * topk..(row + 1) * topk], &tlp[row * topk..(row + 1) * topk]);
            kl[r] += ti.iter().zip(tl).map(|(&i, &lt)| (lt as f64).exp() * (lt as f64 - lq[i as usize])).sum::<f64>();
            let am = l.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |a, (i, &v)| if v > a.1 { (i, v) } else { a }).0;
            agree[r] += (am as i32 == ti[0]) as usize;
        }
        cnt += seqs;
        if (p + 1) % 256 == 0 {
            eprintln!("pos {}: kl {:.4} ({:.0} s, {:.1} ms/step)", p + 1, kl[0] / cnt as f64, t0.elapsed().as_secs_f64(), tstep * 1e3 / (p + 1) as f64);
        }
    }
    let per: Vec<String> = (0..rep).map(|r| format!("{{\"ppl\": {:.4}, \"kl_top{topk}\": {:.6}, \"top1_agree\": {:.6}}}", (nll[r] / cnt as f64).exp(), kl[r] / cnt as f64, agree[r] as f64 / cnt as f64)).collect();
    Ok(format!("{{\"batch\": {bsz}, \"tokens\": {cnt}, \"ms_per_step\": {:.2}, \"replicas\": [{}]}}", tstep * 1e3 / (len - 1) as f64, per.join(", ")))
}

/// beval through multi-token verify steps: K teacher tokens per slot per step (batch_verify + batch_commit of all K)
fn bveval_run(dec: &mut Decoder, eval: &Path, seqs: usize, k: usize, len: usize) -> Result<String> {
    let rd = |f: &str| std::fs::read(eval.join(f));
    let toks: Vec<u32> = rd("heldout.i32")?.chunks(4).map(|b| i32::from_le_bytes(b.try_into().unwrap()) as u32).collect();
    let tidx: Vec<i32> = rd("teacher_idx.i32")?.chunks(4).map(|b| i32::from_le_bytes(b.try_into().unwrap())).collect();
    let tlp: Vec<f32> = rd("teacher_lp.f32")?.chunks(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
    let topk = 256;
    let mut b = dec.new_batch(seqs, len + 16)?;
    let slots: Vec<usize> = (0..seqs).collect();
    let (mut nll, mut kl, mut agree, mut cnt) = (0f64, 0f64, 0usize, 0usize);
    let t0 = Instant::now();
    let mut p = 0;
    while p < len - 1 {
        let kk = k.min(len - 1 - p);
        let feeds: Vec<Vec<u32>> = (0..seqs).map(|s| toks[s * len + p..s * len + p + kk].to_vec()).collect();
        dec.batch_verify(&mut b, &slots, &feeds)?;
        let mut row = 0;
        for s in 0..seqs {
            for j in 0..kk {
                let l = dec.batch_logits(&b, row + j)?;
                let lq = log_softmax(&l);
                nll -= lq[toks[s * len + p + j + 1] as usize];
                let r = s * (len - 1) + p + j;
                let (ti, tl) = (&tidx[r * topk..(r + 1) * topk], &tlp[r * topk..(r + 1) * topk]);
                kl += ti.iter().zip(tl).map(|(&i, &lt)| (lt as f64).exp() * (lt as f64 - lq[i as usize])).sum::<f64>();
                let am = l.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |a, (i, &v)| if v > a.1 { (i, v) } else { a }).0;
                agree += (am as i32 == ti[0]) as usize;
                cnt += 1;
            }
            dec.batch_commit(&mut b, s, row, &feeds[s], kk)?;
            row += kk;
        }
        p += kk;
    }
    Ok(format!("{{\"k\": {k}, \"tokens\": {cnt}, \"ppl\": {:.4}, \"kl_top{topk}\": {:.6}, \"top1_agree\": {:.6}, \"s\": {:.1}}}", (nll / cnt as f64).exp(), kl / cnt as f64, agree as f64 / cnt as f64, t0.elapsed().as_secs_f64()))
}

fn dev(model: &Path, rank: &[(usize, usize)], ctx: usize, cmd: &Path) -> Result<()> {
    let nl = std::env::var("DSXA_LAYERS").ok().and_then(|v| v.parse().ok());
    let mut dec = Decoder::load(model, ctx, rank, &[], nl)?;
    dec.load_dspark(model)?;
    eprintln!("dev: ready, waiting for {}", cmd.display());
    loop {
        if !cmd.exists() {
            std::thread::sleep(std::time::Duration::from_millis(200));
            continue;
        }
        let text = std::fs::read_to_string(cmd)?;
        std::fs::rename(cmd, cmd.with_extension("done"))?;
        for line in text.lines() {
            let w: Vec<&str> = line.split_whitespace().collect();
            if w.is_empty() {
                continue;
            }
            println!("== {line}");
            // a malformed command (missing argument) must not take the loaded model down
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<()> {
                match w[0] {
                    "reload" => {
                        let img = std::fs::read(w[1])?;
                        std::fs::write("/root/cur.hsaco", &img)?;
                        dec.reload_kernels(&img)?;
                        println!("reloaded {}", w[1]);
                    }
                    "bench" => bench(&mut dec, w[1].parse()?, w.get(2).map(|s| s.parse()).transpose()?.unwrap_or(24), false)?,
                    "prof" => bench(&mut dec, w[1].parse()?, 8, true)?,
                    "pprof" => {
                        let n: usize = w[1].parse()?;
                        let toks: Vec<u32> = (0..n).map(|i| ((i * 7919 + 13) % 120000) as u32).collect();
                        dec.g.sync()?;
                        dec.g.prof_start();
                        let t0 = Instant::now();
                        let r = dec.prefill(&toks);
                        let p = dec.g.prof_end()?;
                        r?;
                        let tot: f32 = p.iter().map(|x| x.1).sum();
                        println!("prefill {n}: {:.2} s wall, {:.0} us in kernels ({} launches)", t0.elapsed().as_secs_f64(), tot, p.iter().map(|x| x.2).sum::<usize>());
                        for (nm, t, k) in p.iter().take(25) {
                            println!("  {nm:<28} {t:>10.0} us {k:>6} launches");
                        }
                    }
                    "spec" => {
                        // spec N [file]: greedy DSpark generation of N tokens from a chat prompt (file: user text)
                        let n: usize = w[1].parse()?;
                        let text = match w.get(2) { Some(f) => std::fs::read_to_string(f)?, None => "Explain how a hash map works, with a short Python example.".into() };
                        let tok = tokenizers::Tokenizer::from_file(model.join("tokenizer.json")).map_err(|e| anyhow::anyhow!("{e}"))?;
                        let chat = format!("<｜begin▁of▁sentence｜><｜User｜>{text}<｜Assistant｜></think>");
                        let ids: Vec<u32> = tok.encode(chat.as_str(), false).map_err(|e| anyhow::anyhow!("{e}"))?.get_ids().to_vec();
                        // reference: plain greedy decode
                        let t0 = Instant::now();
                        let mut next = dec.prefill(&ids)?;
                        let top2 = |lg: &[f32]| { let (mut a, mut b) = (f32::NEG_INFINITY, f32::NEG_INFINITY); for &v in lg { if v > a { b = a; a = v; } else if v > b { b = v; } } a - b };
                        let m0 = top2(&dec.logits()?);
                        let mut plain = vec![];
                        let mut margin = vec![];
                        for i in 0..n {
                            plain.push(next);
                            next = dec.step(next)?;
                            if i < 64 {
                                // top-2 gap of the logits that chose the next token (near-ties explain divergence)
                                let lg = dec.logits()?;
                                let (mut a, mut b) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
                                for &v in &lg {
                                    if v > a { b = a; a = v; } else if v > b { b = v; }
                                }
                                margin.push(a - b);
                            }
                        }
                        let tplain = t0.elapsed().as_secs_f64();
                        dec.reset()?;
                        let t1 = Instant::now();
                        let next = dec.prefill(&ids)?;
                        let m1 = top2(&dec.logits()?);
                        println!("first token: plain {} (top-2 gap {m0:.4}), spec run {next} (gap {m1:.4})", plain[0]);
                        let mut got = vec![];
                        let (sx, _) = dec.spec_loop(next, n, &mut |tk| { got.push(tk); true })?;
                        let tspec = t1.elapsed().as_secs_f64();
                        let same = got.iter().zip(&plain).take_while(|(a, b)| a == b).count();
                        if same < n && same >= 1 && same - 1 < margin.len() {
                            println!("first difference at {same}: plain {} vs spec {}; plain's top-2 logit gap there {:.4} (median gap {:.3})", plain[same], got[same], margin[same - 1], { let mut m = margin.clone(); m.sort_by(|a, b| a.partial_cmp(b).unwrap()); m[m.len() / 2] });
                        }
                        println!("prompt {} tokens; plain {n} tokens {:.2} s, spec {:.2} s ({:.2}x); {} passes, {:.2} drafts accepted per pass, {} tokens decoded plainly, draft {:.1} ms, verify {:.1} ms per pass; same as plain for {same}/{n}",
                            ids.len(), tplain, tspec, tplain / tspec, sx.passes, sx.accepted as f64 / sx.passes as f64, sx.plain, sx.draft_us / 1e3 / sx.passes as f64, sx.verify_us / 1e3 / sx.passes as f64);
                        println!("{}", tok.decode(&got, false).map_err(|e| anyhow::anyhow!("{e}"))?.chars().take(400).collect::<String>());
                    }
                    "pdet" => {
                        // prefill determinism: the same chat prompt N times, top-2 logits each time
                        let n: usize = w[1].parse()?;
                        let text = match w.get(2) { Some(f) => std::fs::read_to_string(f)?, None => "Explain how a hash map works, with a short Python example.".into() };
                        let tok = tokenizers::Tokenizer::from_file(model.join("tokenizer.json")).map_err(|e| anyhow::anyhow!("{e}"))?;
                        let chat = format!("<｜begin▁of▁sentence｜><｜User｜>{text}<｜Assistant｜></think>");
                        let ids: Vec<u32> = tok.encode(chat.as_str(), false).map_err(|e| anyhow::anyhow!("{e}"))?.get_ids().to_vec();
                        for _ in 0..n {
                            dec.reset()?;
                            let next = dec.prefill(&ids)?;
                            let lg = dec.logits()?;
                            let mut ix: Vec<usize> = (0..lg.len()).collect();
                            ix.sort_by(|&a, &b| lg[b].partial_cmp(&lg[a]).unwrap());
                            println!("next {next}: top {} {:.4}, {} {:.4}, {} {:.4}", ix[0], lg[ix[0]], ix[1], lg[ix[1]], ix[2], lg[ix[2]]);
                        }
                    }
                    "sprof" => {
                        // kernel time of DSpark generation (N tokens) after a short prompt
                        let n: usize = w[1].parse()?;
                        let prompt: Vec<u32> = vec![0, 671, 6102, 294, 8760, 344, 11111, 16, 455, 6102, 294, 10322, 344];
                        let next = dec.prefill(&prompt)?;
                        dec.g.sync()?;
                        dec.g.prof_start();
                        let t0 = Instant::now();
                        let r = dec.spec_loop(next, n, &mut |_| true);
                        let p = dec.g.prof_end()?;
                        let (sx, _) = r?;
                        let tot: f32 = p.iter().map(|x| x.1).sum();
                        println!("spec {n}: {} passes, {:.2} accepted per pass; {:.0} us wall per pass, {:.0} us in kernels per pass", sx.passes, sx.accepted as f64 / sx.passes as f64, t0.elapsed().as_secs_f64() * 1e6 / sx.passes as f64, tot / sx.passes as f32);
                        for (nm, t, k) in p.iter().take(30) {
                            println!("  {nm:<28} {:>10.0} us {:>6.1} launches per pass", t / sx.passes as f32, *k as f32 / sx.passes as f32);
                        }
                    }
                    "vprof" => {
                        // kernel time of one verify pass of T tokens after a short prompt (state left as before)
                        let t: usize = w[1].parse()?;
                        dec.reset()?;
                        let prompt: Vec<u32> = (0..64).map(|i| ((i * 7919 + 13) % 120000) as u32).collect();
                        let _ = dec.prefill(&prompt)?;
                        let feed: Vec<u32> = (0..t).map(|i| (100 + i * 37) as u32).collect();
                        let _ = dec.verify(&feed)?;
                        dec.commit(t, 1)?;
                        dec.g.sync()?;
                        dec.g.prof_start();
                        let t0 = Instant::now();
                        let r = dec.verify(&feed);
                        let p = dec.g.prof_end()?;
                        r?;
                        dec.commit(t, 1)?;
                        let tot: f32 = p.iter().map(|x| x.1).sum();
                        println!("verify {t}: {:.0} us wall, {:.0} us in kernels ({} launches)", t0.elapsed().as_secs_f64() * 1e6, tot, p.iter().map(|x| x.2).sum::<usize>());
                        for (nm, t, k) in p.iter().take(30) {
                            println!("  {nm:<28} {t:>10.0} us {k:>6} launches");
                        }
                    }
                    "bcheck" => {
                        // bcheck N [CTX]: two different prompts decoded greedily N tokens each, single-stream vs one
                        // batched pass of both (slots 1 and 3 of 4); prints where the token streams first differ
                        let n: usize = w[1].parse()?;
                        let ctx: usize = w.get(2).map(|s| s.parse()).transpose()?.unwrap_or(8192);
                        let prompts: Vec<Vec<u32>> = vec![(0..300).map(|i| ((i * 7919 + 13) % 120000) as u32).collect(), (0..1500).map(|i| ((i * 104729 + 7) % 120000) as u32).collect()];
                        let mut refs = vec![];
                        let mut b = dec.new_batch(4, ctx)?;
                        let mut first = vec![];
                        for (pi, pr) in prompts.iter().enumerate() {
                            dec.reset()?;
                            let mut tok = dec.prefill(pr)?;
                            dec.load_slot(&mut b, 1 + 2 * pi)?;
                            first.push(tok);
                            let mut out = vec![tok];
                            for _ in 1..n {
                                tok = dec.step(tok)?;
                                out.push(tok);
                            }
                            refs.push(out);
                        }
                        let mut toks = first.clone();
                        let mut got = vec![first.clone()];
                        let t0 = Instant::now();
                        for _ in 1..n {
                            toks = dec.batch_step(&mut b, &[1, 3], &toks)?;
                            got.push(toks.clone());
                        }
                        println!("batched {} steps of 2: {:.1} ms/step", n - 1, t0.elapsed().as_secs_f64() * 1e3 / (n - 1).max(1) as f64);
                        for pi in 0..2 {
                            let gs: Vec<u32> = got.iter().map(|v| v[pi]).collect();
                            let d = (0..n).find(|&i| gs[i] != refs[pi][i]);
                            println!("prompt {pi}: first difference at {:?} of {n}\n  single {:?}\n  batch  {:?}", d, &refs[pi][..n.min(24)], &gs[..n.min(24)]);
                        }
                    }
                    "blogit" => {
                        // blogit CTX [SLOT] [TEXTFILE]: logits of one token after a chat prompt: decode step vs batched step
                        // (slot SLOT of 4) vs the prompt path over prompt + token
                        let ctx: usize = w[1].parse()?;
                        let slot: usize = w.get(2).map(|s| s.parse()).transpose()?.unwrap_or(2);
                        let text = match w.get(3) { Some(f) => std::fs::read_to_string(f)?, None => "Write a Rust function that parses a CSV line with quoted fields, and explain the edge cases.".into() };
                        let tok = tokenizers::Tokenizer::from_file(model.join("tokenizer.json")).map_err(|e| anyhow::anyhow!("{e}"))?;
                        let chat = format!("<｜begin▁of▁sentence｜><｜User｜>{text}<｜Assistant｜></think>");
                        let ids: Vec<u32> = tok.encode(chat.as_str(), false).map_err(|e| anyhow::anyhow!("{e}"))?.get_ids().to_vec();
                        let mut b = dec.new_batch(4, ctx)?;
                        dec.reset()?;
                        let t1 = dec.prefill(&ids)?;
                        dec.load_slot(&mut b, slot)?;
                        let _ = dec.step(t1)?;
                        let l1 = dec.logits()?;
                        let nb = dec.batch_step(&mut b, &[slot], &[t1])?;
                        let l2 = dec.batch_logits(&b, 0)?;
                        dec.reset()?;
                        let mut ids2 = ids.clone();
                        ids2.push(t1);
                        let _ = dec.prefill(&ids2)?;
                        let l3 = dec.logits()?;
                        let rel = |a: &[f32], r: &[f32]| (a.iter().zip(r).map(|(x, y)| (x - y) * (x - y)).sum::<f32>() / r.iter().map(|y| y * y).sum::<f32>()).sqrt();
                        let top = |l: &[f32]| { let mut ix: Vec<usize> = (0..l.len()).collect(); ix.sort_by(|&a, &b| l[b].partial_cmp(&l[a]).unwrap()); ix[..5].to_vec() };
                        println!("prompt {} tokens, next {t1}; batch argmax {}", ids.len(), nb[0]);
                        println!("  decode top5 {:?}\n  batch  top5 {:?} rel {:.2e}\n  prompt top5 {:?} rel {:.2e}", top(&l1), top(&l2), rel(&l2, &l1), top(&l3), rel(&l3, &l1));
                    }
                    "bdense" => dec.build_bdense()?,
                    "mem" => {
                        let (free, total) = dec.g.mem_info()?;
                        println!("HBM live: {:.3} GB used, {:.3} GB free / {:.3} GB total", (total - free) as f64 / 1e9, free as f64 / 1e9, total as f64 / 1e9);
                        for slots in [1, 4, 8] {
                            println!("{slots} slots at 262144: {:.3} GB cache state (workspace excluded)", batch::Batch::state_bytes(&dec, slots, 262144) as f64 / 1e9);
                        }
                    }
                    "batch-mem" => {
                        let slots: usize = w[1].parse()?;
                        let ctx: usize = w[2].parse()?;
                        let b = dec.new_batch(slots, ctx)?;
                        let (free, total) = dec.g.mem_info()?;
                        println!("batch-mem slots={slots} ctx={ctx}: {:.3} GB live used, {:.3} GB free (allocated batch retained during measurement)", (total - free) as f64 / 1e9, free as f64 / 1e9);
                        drop(b);
                    }
                    "dense-clear" => {
                        dec.g.sync()?;
                        dec.graph = None;
                        dec.vgraphs.borrow_mut().clear();
                        dec.bgraphs.borrow_mut().clear();
                        dec.bseen.borrow_mut().clear();
                        dec.bdense.clear();
                        println!("duplicate BF16 dense weights released");
                    }
                    "draft-dump" => {
                        let dir = std::path::Path::new(w[1]);
                        std::fs::create_dir_all(dir)?;
                        anyhow::ensure!(dec.dspark.is_some(), "load draft first");
                        let tok = tokenizers::Tokenizer::from_file(model.join("tokenizer.json")).map_err(|e| anyhow::anyhow!("{e}"))?;
                        let chat = "<｜begin▁of▁sentence｜><｜User｜>Write a Python function that merges two sorted lists, with tests.<｜Assistant｜></think>";
                        let pr = tok.encode(chat, false).map_err(|e| anyhow::anyhow!("{e}"))?.get_ids().to_vec();
                        dec.reset()?;
                        let next = dec.prefill(&pr)?;
                        let n = dec.pos;
                        let block = dec.dspark.as_ref().unwrap().block;
                        let nwin = n.min(128);
                        let mh = dec.pf.as_ref().unwrap().dspark_bufs().0;
                        let hidden: Vec<f32> = dec.g.dtoh(mh + ((n - nwin) * 3 * decode::D * 4) as u64, nwin * 3 * decode::D)?;
                        let bytes = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
                        std::fs::write(dir.join("main-hidden.f32"), bytes(&hidden))?;
                        let mut b = dec.new_batch(1, 8192)?;
                        dec.load_slot(&mut b, 0)?;
                        for (i, ring) in b.drings.iter().enumerate() {
                            let raw: Vec<u8> = dec.g.dtoh(ring.ptr, (128 + 16) * decode::HD * 2)?;
                            std::fs::write(dir.join(format!("ring-{i}.bf16")), raw)?;
                        }
                        let drafts = dec.batch_draft(&mut b, &[0], &[next])?;
                        for (label, ptr) in [("norm-hi", b.xb.ptr), ("norm-lo", b.xl.ptr)] {
                            let raw: Vec<u8> = dec.g.dtoh(ptr, block * decode::D * 2)?;
                            std::fs::write(dir.join(format!("{label}.bf16")), raw)?;
                        }
                        let lg: Vec<f32> = dec.g.dtoh(b.logits.ptr, decode::VOCAB)?;
                        std::fs::write(dir.join("first-logits-markov.f32"), bytes(&lg))?;
                        let meta = serde_json::json!({"position": n, "anchor": next, "block": block, "drafts": drafts[0], "ring_mod": 144, "window": 128, "dim": decode::D, "head_dim": decode::HD});
                        std::fs::write(dir.join("meta.json"), serde_json::to_vec_pretty(&meta)?)?;
                        println!("frozen draft saved to {}: {meta}", dir.display());
                    }
                    "draft-load" => {
                        anyhow::ensure!(dec.dspark.is_none(), "draft stack already loaded");
                        let fmt = dspark::bits_from_env().ok_or_else(|| anyhow::anyhow!("set DSXA_DSPARK before draft-load"))?;
                        let (free, _) = dec.g.mem_info()?;
                        anyhow::ensure!(free > dspark::bytes(fmt) + (1 << 30), "not enough free HBM for draft stack and workspace");
                        dec.g.sync()?;
                        dec.graph = None;
                        dec.vgraphs.borrow_mut().clear();
                        dec.bgraphs.borrow_mut().clear();
                        dec.bseen.borrow_mut().clear();
                        dec.load_dspark(model)?;
                        println!("draft stack loaded without changing main-model expert placement");
                    }
                    "draft-unload" => {
                        dec.g.sync()?;
                        dec.graph = None;
                        dec.vgraphs.borrow_mut().clear();
                        dec.bgraphs.borrow_mut().clear();
                        dec.bseen.borrow_mut().clear();
                        dec.dspark = None;
                        println!("draft stack unloaded");
                    }
                    "plogit" => {
                        // plogit N: prefill N eval tokens from empty, print the top-5 and the rel. difference of the last
                        // logits from the previous plogit's (A/B of prompt-path settings)
                        let n: usize = w[1].parse()?;
                        let toks: Vec<u32> = std::fs::read("/data/models/lucebox/eval/heldout.i32")?.chunks(4).map(|b| i32::from_le_bytes(b.try_into().unwrap()) as u32).collect();
                        let toks: Vec<u32> = toks.iter().cycle().take(n).copied().collect();
                        dec.reset()?;
                        let t0 = Instant::now();
                        dec.prefill(&toks)?;
                        let l = dec.logits()?;
                        let dt = t0.elapsed().as_secs_f64();
                        let mut ix: Vec<usize> = (0..l.len()).collect();
                        ix.sort_by(|&a, &b| l[b].partial_cmp(&l[a]).unwrap());
                        let prev: Vec<f32> = std::fs::read("/root/plogit.bin").map(|b| b.chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()).unwrap_or_default();
                        let rel = if prev.len() == l.len() { let lp = log_softmax(&prev); let lq = log_softmax(&l); Some(lp.iter().zip(&lq).map(|(a, b)| (a.exp() as f64) * (a - b) as f64).sum::<f64>()) } else { None };
                        std::fs::write("/root/plogit.bin", l.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())?;
                        println!("plogit {n}: {:.2} s ({:.0} tok/s), top5 {:?}, KL vs previous {:?}", dt, n as f64 / dt, &ix[..5], rel);
                    }
                    "bsweep" => {
                        // bsweep CTX PLEN STEPS B1,B2,..: one PLEN-token prompt prefilled once, copied into max(B) slots, then
                        // STEPS timed batched steps at each B (tok/s per session and total) with a one-step kernel profile
                        let bctx: usize = w[1].parse()?;
                        let plen: usize = w[2].parse()?;
                        let steps: usize = w[3].parse()?;
                        let bs: Vec<usize> = w[4].split(',').map(|x| x.parse().unwrap()).collect();
                        let bmax = *bs.iter().max().unwrap();
                        let mut b = dec.new_batch(bmax, bctx)?;
                        // `distinct` (5th argument): every slot its own prompt (eval text at different offsets), so sessions route
                        // like independent ones; otherwise one prompt copied into every slot
                        let distinct = w.get(5) == Some(&"distinct");
                        let ev: Vec<u32> = std::fs::read("/data/models/lucebox/eval/heldout.i32").map(|b| b.chunks(4).map(|c| i32::from_le_bytes(c.try_into().unwrap()) as u32).collect()).unwrap_or_default();
                        let mut first = 0;
                        for s in 0..bmax {
                            if s == 0 || distinct {
                                dec.reset()?;
                                let pr: Vec<u32> = if distinct && !ev.is_empty() { (0..plen).map(|i| ev[(s * 997 + i) % ev.len()]).collect() } else { (0..plen).map(|i| ((i * 7919 + 13) % 120000) as u32).collect() };
                                first = dec.prefill(&pr)?;
                            }
                            dec.load_slot(&mut b, s)?;
                        }
                        for &bsz in &bs {
                            let slots: Vec<usize> = (0..bsz).collect();
                            let mut toks: Vec<u32> = (0..bsz).map(|i| (first + i as u32 * 31) % 120000).collect();
                            toks = dec.batch_step(&mut b, &slots, &toks)?;
                            dec.g.sync()?;
                            let t0 = Instant::now();
                            for _ in 0..steps {
                                toks = dec.batch_step(&mut b, &slots, &toks)?;
                            }
                            let st = t0.elapsed().as_secs_f64() / steps as f64;
                            dec.g.prof_start();
                            let r = dec.batch_step(&mut b, &slots, &toks);
                            let p = dec.g.prof_end()?;
                            toks = r?;
                            let _ = &toks;
                            let tot: f32 = p.iter().map(|x| x.1).sum();
                            let ntop = if std::env::var_os("DSXA_PROF_ALL").is_some() { p.len() } else { 6 };
                            let top: Vec<String> = p.iter().take(ntop).map(|(n, t, k)| format!("{n} {:.2}/{k}", t / 1e3)).collect();
                            println!("B={bsz:<3} ctx~{plen}: {:.1} ms/step, {:.1} tok/s per session, {:.0} total | kernels {:.1} ms in {} launches: {}", st * 1e3, 1.0 / st, bsz as f64 / st, tot / 1e3, p.iter().map(|x| x.2).sum::<usize>(), top.join(", "));
                            for s in 0..bsz {
                                b.pos[s] -= steps + 2;
                            }
                        }
                    }
                    "vbtest" => {
                        // vbtest B K PLEN: B distinct prompts; batch A decodes K tokens one step at a time, batch B verifies A's
                        // greedy continuation in one multi-token step, commits it, then both take one more step
                        let bsz: usize = w[1].parse()?;
                        let k: usize = w[2].parse()?;
                        let plen: usize = w[3].parse()?;
                        let ev: Vec<u32> = std::fs::read("/data/models/lucebox/eval/heldout.i32")?.chunks(4).map(|c| i32::from_le_bytes(c.try_into().unwrap()) as u32).collect();
                        let mut a = dec.new_batch(bsz, plen + 64)?;
                        let mut bb = dec.new_batch(bsz, plen + 64)?;
                        let mut first = vec![];
                        for s in 0..bsz {
                            dec.reset()?;
                            let pr: Vec<u32> = (0..plen).map(|i| ev[(s * 997 + i) % ev.len()]).collect();
                            first.push(dec.prefill(&pr)?);
                            dec.load_slot(&mut a, s)?;
                            dec.load_slot(&mut bb, s)?;
                        }
                        let slots: Vec<usize> = (0..bsz).collect();
                        let mut seq: Vec<Vec<u32>> = first.iter().map(|&f| vec![f]).collect();
                        let mut cur = first.clone();
                        let mut seql: Vec<Vec<Vec<f32>>> = vec![vec![]; bsz];
                        for _ in 0..k {
                            cur = dec.batch_step(&mut a, &slots, &cur)?;
                            for s in 0..bsz { seq[s].push(cur[s]); seql[s].push(dec.batch_logits(&a, s)?); }
                        }
                        // B: feed first + A's next k-1 tokens; outputs should be A's tokens 1..=k
                        let feeds: Vec<Vec<u32>> = seq.iter().map(|v| v[..k].to_vec()).collect();
                        let outs = dec.batch_verify(&mut bb, &slots, &feeds)?;
                        let mut same = 0;
                        let mut row = 0;
                        let (mut klmax, mut klsum) = (0f64, 0f64);
                        for s in 0..bsz {
                            for j in 0..k {
                                let lv = dec.batch_logits(&bb, row + j)?;
                                let (p, q) = (log_softmax(&seql[s][j]), log_softmax(&lv));
                                let kl: f64 = p.iter().zip(&q).map(|(a, b)| a.exp() * (a - b)).sum();
                                klmax = klmax.max(kl);
                                klsum += kl;
                            }
                            same += outs[s].iter().zip(&seq[s][1..=k]).filter(|(x, y)| x == y).count();
                            dec.batch_commit(&mut bb, s, row, &feeds[s], k)?;
                            row += k;
                        }
                        let na = dec.batch_step(&mut a, &slots, &cur)?;
                        let nb = dec.batch_step(&mut bb, &slots, &cur)?;
                        let after = na.iter().zip(&nb).filter(|(x, y)| x == y).count();
                        println!("vbtest {bsz} x {k}: verify matched {same} / {} sequential tokens (logit KL mean {:.2e}, max {:.2e}); next step after commit {after} / {bsz} equal", bsz * k, klsum / (bsz * k) as f64, klmax);
                        println!("  slot 0 sequential {:?}\n  slot 0 verify     {:?}", &seq[0][1..=k], &outs[0]);
                    }
                    "bspec" => {
                        // bspec B N PLEN [K]: B distinct prompts; N greedy tokens per slot with batched DSpark drafts (K of the
                        // block's drafts verified, default all) vs plain batched decoding from the same state
                        let bsz: usize = w[1].parse()?;
                        let n: usize = w[2].parse()?;
                        let plen: usize = w[3].parse()?;
                        let ev: Vec<u32> = std::fs::read("/data/models/lucebox/eval/heldout.i32")?.chunks(4).map(|c| i32::from_le_bytes(c.try_into().unwrap()) as u32).collect();
                        let mut a = dec.new_batch(bsz, plen + n + 64)?;
                        let mut sb = dec.new_batch(bsz, plen + n + 64)?;
                        let kd: usize = w.get(4).map(|s| s.parse()).transpose()?.unwrap_or(sb.dblock);
                        // PLEN 0: chat prompts (coding / agentic questions); else eval text slices of PLEN tokens
                        let qs = ["Write a Python function that merges two sorted lists, with tests.", "Explain how a hash map handles collisions, with a short Rust example.",
                            "Write a bash script that finds the 10 largest files under a directory.", "Implement binary search in C and explain its edge cases.",
                            "Write a SQL query that returns the top 3 customers by revenue per month.", "Refactor this JavaScript callback code into async/await: fs.readFile('a', (e, d) => { if (e) throw e; console.log(d) })",
                            "Write a Go HTTP server with one JSON endpoint that returns the current time.", "Explain what a mutex is and show a deadlock example in Python.",
                            "Write a TypeScript type for a deeply readonly object and explain it.", "Implement an LRU cache in Python with O(1) operations.",
                            "Write a Dockerfile for a Node.js app with a multi-stage build.", "Explain the difference between processes and threads, then give a C example of fork().",
                            "Write a regex that matches ISO 8601 dates and explain each part.", "Write a Rust function that parses a CSV line with quoted fields.",
                            "Implement quicksort in Java and analyze its complexity.", "Write a Python asyncio program that fetches 5 URLs concurrently."];
                        let tok = tokenizers::Tokenizer::from_file(model.join("tokenizer.json")).map_err(|e| anyhow::anyhow!("{e}"))?;
                        let mut first = vec![];
                        for s in 0..bsz {
                            dec.reset()?;
                            let pr: Vec<u32> = if plen == 0 {
                                let chat = format!("<｜begin▁of▁sentence｜><｜User｜>{}<｜Assistant｜></think>", qs[s % qs.len()]);
                                tok.encode(chat.as_str(), false).map_err(|e| anyhow::anyhow!("{e}"))?.get_ids().to_vec()
                            } else { (0..plen).map(|i| ev[(s * 997 + i) % ev.len()]).collect() };
                            first.push(dec.prefill(&pr)?);
                            dec.load_slot(&mut a, s)?;
                            dec.load_slot(&mut sb, s)?;
                        }
                        let slots: Vec<usize> = (0..bsz).collect();
                        // plain
                        let t0 = Instant::now();
                        let mut plain: Vec<Vec<u32>> = first.iter().map(|&f| vec![f]).collect();
                        let mut cur = first.clone();
                        for _ in 1..n {
                            cur = dec.batch_step(&mut a, &slots, &cur)?;
                            for s in 0..bsz { plain[s].push(cur[s]); }
                        }
                        let tp = t0.elapsed().as_secs_f64();
                        // speculative
                        let t1 = Instant::now();
                        let mut out: Vec<Vec<u32>> = first.iter().map(|&f| vec![f]).collect();
                        let mut next = first.clone();
                        let (mut passes, mut acc_tot, mut tdraft) = (0usize, 0usize, 0f64);
                        let mut live: Vec<usize> = (0..bsz).collect();
                        while !live.is_empty() {
                            let td = Instant::now();
                            let nx: Vec<u32> = live.iter().map(|&s| next[s]).collect();
                            let drafts = dec.batch_draft(&mut sb, &live, &nx)?;
                            tdraft += td.elapsed().as_secs_f64();
                            let feeds: Vec<Vec<u32>> = live.iter().enumerate().map(|(i, &s)| {
                                let mut f = vec![next[s]];
                                f.extend_from_slice(&drafts[i][..kd]);
                                f.truncate(n - out[s].len() + 1);
                                f
                            }).collect();
                            let pre: Vec<usize> = live.iter().map(|&s| sb.pos[s]).collect();
                            let outs = dec.batch_verify(&mut sb, &live, &feeds)?;
                            let (mut rows, mut opos, mut oslot) = (vec![], vec![], vec![]);
                            let mut row = 0;
                            for (i, &s) in live.iter().enumerate() {
                                let f = &feeds[i];
                                let mut acc = 0;
                                while acc + 1 < f.len() && outs[i][acc] == f[acc + 1] { acc += 1; }
                                dec.batch_commit(&mut sb, s, row, f, acc + 1)?;
                                for j in 0..=acc {
                                    rows.push((row + j) as i32);
                                    opos.push((pre[i] + j) as i32);
                                    oslot.push(s as i32);
                                    if j > 0 { out[s].push(f[j]); }
                                }
                                out[s].push(outs[i][acc]);
                                next[s] = outs[i][acc];
                                passes += 1;
                                acc_tot += acc;
                                row += f.len();
                            }
                            let tv: usize = feeds.iter().map(|f| f.len()).sum();
                            dec.batch_observe(&sb, tv, &rows, &opos, &oslot)?;
                            live.retain(|&s| out[s].len() < n);
                        }
                        let ts = t1.elapsed().as_secs_f64();
                        let mut same = 0;
                        for s in 0..bsz {
                            out[s].truncate(n);
                            same += out[s].iter().zip(&plain[s]).take_while(|(x, y)| x == y).count();
                        }
                        println!("bspec B={bsz} N={n} ctx~{plen}: plain {:.1} tok/s per session, spec {:.1} tok/s per session ({:.2}x); {:.2} drafts accepted per pass; draft {:.0}% of the time; output = plain for {same} / {} tokens",
                            (n - 1) as f64 / tp, (n - 1) as f64 / ts, tp / ts, acc_tot as f64 / passes as f64, 100.0 * tdraft / ts, bsz * n);
                    }
                    "dcmp" => {
                        // dcmp PLEN: drafts after a prompt from the single-sequence path vs batch_draft from the same state
                        let plen: usize = w[1].parse()?;
                        let ev: Vec<u32> = std::fs::read("/data/models/lucebox/eval/heldout.i32")?.chunks(4).map(|c| i32::from_le_bytes(c.try_into().unwrap()) as u32).collect();
                        let mut b = dec.new_batch(2, plen + 64)?;
                        dec.reset()?;
                        let pr: Vec<u32> = ev[..plen].to_vec();
                        let next = dec.prefill(&pr)?;
                        dec.load_slot(&mut b, 1)?;
                        // passes of draft -> verify -> commit -> observe on both paths
                        let (mut n1, mut n2) = (next, next);
                        for pass in 0..4 {
                            let d1 = dec.draft(n1)?;
                            let mut f1 = vec![n1];
                            f1.extend_from_slice(&d1);
                            let p0 = dec.pos;
                            let o1 = dec.verify(&f1)?;
                            let mut a1 = 0;
                            while a1 + 1 < f1.len() && o1[a1] == f1[a1 + 1] { a1 += 1; }
                            dec.commit(f1.len(), a1 + 1)?;
                            dec.observe(p0, a1 + 1, 0)?;
                            n1 = o1[a1];
                            let d2 = dec.batch_draft(&mut b, &[1], &[n2])?;
                            let mut f2 = vec![n2];
                            f2.extend_from_slice(&d2[0]);
                            let q0 = b.pos[1];
                            let o2 = dec.batch_verify(&mut b, &[1], &[f2.clone()])?;
                            let mut a2 = 0;
                            while a2 + 1 < f2.len() && o2[0][a2] == f2[a2 + 1] { a2 += 1; }
                            dec.batch_commit(&mut b, 1, 0, &f2, a2 + 1)?;
                            let rows: Vec<i32> = (0..=a2 as i32).collect();
                            let pos: Vec<i32> = (0..=a2).map(|j| (q0 + j) as i32).collect();
                            dec.batch_observe(&b, f2.len(), &rows, &pos, &vec![1; a2 + 1])?;
                            n2 = o2[0][a2];
                            println!("pass {pass}: single feed {:?} verify {:?} acc {a1}\n        batch  feed {:?} verify {:?} acc {a2}", f1, o1, f2, o2[0], );
                        }
                    }
                    "sspec" => {
                        // sspec B N MODE: B sessions, N tokens each, plain batched decoding vs spec_step (prompt lookup + adaptive
                        // DSpark). MODE chat: coding questions; edit: a source file to return with one change (agentic copying)
                        let bsz: usize = w[1].parse()?;
                        let n: usize = w[2].parse()?;
                        let mode = w.get(3).copied().unwrap_or("chat");
                        let tok = tokenizers::Tokenizer::from_file(model.join("tokenizer.json")).map_err(|e| anyhow::anyhow!("{e}"))?;
                        let qs = ["Write a Python function that merges two sorted lists, with tests.", "Explain how a hash map handles collisions, with a short Rust example.",
                            "Write a bash script that finds the 10 largest files under a directory.", "Implement binary search in C and explain its edge cases.",
                            "Write a SQL query that returns the top 3 customers by revenue per month.", "Write a Go HTTP server with one JSON endpoint that returns the current time.",
                            "Explain what a mutex is and show a deadlock example in Python.", "Implement an LRU cache in Python with O(1) operations."];
                        let files = ["/root/dsx/amd/src/runner.rs", "/root/dsx/amd/src/spec.rs", "/root/dsx/amd/src/serve.rs", "/root/dsx/amd/src/ck.rs", "/root/dsx/amd/src/export.rs", "/root/dsx/amd/src/load.rs"];
                        let mut prompts = vec![];
                        for s in 0..bsz {
                            let user = if mode == "edit" {
                                let f = files[s % files.len()];
                                let src = std::fs::read_to_string(f)?;
                                format!("Here is a file:\n\n```rust\n{src}\n```\n\nRewrite the whole file with every doc comment line starting with `//!` changed to start with `// NOTE:`. Output only the full file.")
                            } else { qs[s % qs.len()].to_string() };
                            let chat = format!("<｜begin▁of▁sentence｜><｜User｜>{user}<｜Assistant｜></think>");
                            prompts.push(tok.encode(chat.as_str(), false).map_err(|e| anyhow::anyhow!("{e}"))?.get_ids().to_vec());
                        }
                        let ctx = prompts.iter().map(|p| p.len()).max().unwrap() + n + 64;
                        let mut a = dec.new_batch(bsz, ctx)?;
                        let mut sb = dec.new_batch(bsz, ctx)?;
                        let mut first = vec![];
                        for s in 0..bsz {
                            dec.reset()?;
                            first.push(dec.prefill(&prompts[s])?);
                            dec.load_slot(&mut a, s)?;
                            dec.load_slot(&mut sb, s)?;
                        }
                        let slots: Vec<usize> = (0..bsz).collect();
                        let t0 = Instant::now();
                        let mut cur = first.clone();
                        for _ in 1..n {
                            cur = dec.batch_step(&mut a, &slots, &cur)?;
                        }
                        let tp = t0.elapsed().as_secs_f64();
                        let mut st: Vec<spec::SpecState> = (0..bsz).map(|s| { let mut h = prompts[s].clone(); h.push(first[s]); spec::SpecState::new(&h) }).collect();
                        let mut got = vec![1usize; bsz];
                        let mut next = first.clone();
                        let t1 = Instant::now();
                        let mut steps = 0;
                        loop {
                            let live: Vec<usize> = (0..bsz).filter(|&s| got[s] < n).collect();
                            if live.is_empty() { break; }
                            let nx: Vec<u32> = live.iter().map(|&s| next[s]).collect();
                            let mut refs: Vec<&mut spec::SpecState> = st.iter_mut().enumerate().filter(|(s, _)| got[*s] < n).map(|(_, x)| x).collect();
                            let em = dec.spec_step(&mut sb, &live, &nx, &mut refs)?;
                            for (k, &s) in live.iter().enumerate() {
                                got[s] += em[k].len();
                                next[s] = *em[k].last().unwrap();
                            }
                            steps += 1;
                        }
                        let ts = t1.elapsed().as_secs_f64();
                        let ngen: usize = got.iter().sum::<usize>() - bsz;
                        let acc: usize = st.iter().map(|x| x.accepted).sum();
                        let ng: usize = st.iter().map(|x| x.ngram_hits).sum();
                        let passes: usize = st.iter().map(|x| x.passes).sum();
                        println!("sspec B={bsz} N={n} {mode} (prompts ~{} tokens): plain {:.1} tok/s per session ({:.0} total); spec {:.1} tok/s per session ({:.0} total, {:.2}x); {steps} steps, {:.2} drafts accepted per session-step ({} from prompt lookup)",
                            prompts.iter().map(|p| p.len()).sum::<usize>() / bsz, (n - 1) as f64 / tp, (bsz * (n - 1)) as f64 / tp, ngen as f64 / bsz as f64 / ts, ngen as f64 / ts, tp / (ts * (bsz * (n - 1)) as f64 / ngen as f64), acc as f64 / passes as f64, ng);
                    }
                    "prune" => {
                        // prune FRAC: the coldest FRAC of each layer's experts (rank file order) never routed (router bias -1e30);
                        // the remaining picks renormalize as usual. prune 0 restores the router.
                        let frac: f64 = w[1].parse()?;
                        thread_local! { static ORIG: std::cell::RefCell<Vec<Vec<f32>>> = const { std::cell::RefCell::new(vec![]) }; }
                        ORIG.with(|o| -> Result<()> {
                            let mut o = o.borrow_mut();
                            if o.is_empty() {
                                for l in &dec.layers { o.push(dec.g.dtoh(l.gate_bias.ptr, NE)?); }
                            }
                            let k = (frac * NE as f64).round() as usize;
                            for (li, l) in dec.layers.iter().enumerate() {
                                let mut bias = o[li].clone();
                                let cold: Vec<usize> = rank.iter().rev().filter(|(lr, _)| *lr == li).map(|&(_, e)| e).take(k).collect();
                                for e in cold { bias[e] = -1e30; }
                                dec.g.htod(l.gate_bias.ptr, &bias)?;
                            }
                            println!("pruned {k} of {NE} experts per layer");
                            Ok(())
                        })?;
                    }
                    "lspec" => {
                        // lspec B KTOK N: a ~KTOK-thousand-token prefix of source files (prefilled once, kept in slot 0), then B
                        // sessions that each extend it with a request to rewrite a different file (prefix reuse via the lane);
                        // N tokens per session, plain batched decoding vs spec_step
                        let bsz: usize = w[1].parse()?;
                        let ktok: usize = w[2].parse()?;
                        let n: usize = w[3].parse()?;
                        let tok = tokenizers::Tokenizer::from_file(model.join("tokenizer.json")).map_err(|e| anyhow::anyhow!("{e}"))?;
                        let enc = |s: &str| -> Result<Vec<u32>> { Ok(tok.encode(s, false).map_err(|e| anyhow::anyhow!("{e}"))?.get_ids().to_vec()) };
                        let mut files: Vec<std::path::PathBuf> = vec![];
                        for d in ["/root/dsx/amd/src", "/root/dsx/amd/kernels", "/root/dsx/core/src", "/root/dsx/nvidia/src"] {
                            if let Ok(rd) = std::fs::read_dir(d) { for e in rd.flatten() { let p = e.path(); if p.extension().map(|x| x == "rs" || x == "h" || x == "cuh" || x == "hip").unwrap_or(false) { files.push(p); } } }
                        }
                        files.sort();
                        let mut prefix = String::from("<｜begin▁of▁sentence｜><｜User｜>Here is a code base.\n\n");
                        let mut fi = 0;
                        let mut ptoks = 0;
                        while ptoks < ktok * 1000 {
                            let f = &files[fi % files.len()];
                            let src = std::fs::read_to_string(f).unwrap_or_default();
                            let chunk = format!("=== {} ===\n{}\n", f.display(), src);
                            ptoks += enc(&chunk)?.len();
                            prefix.push_str(&chunk);
                            fi += 1;
                        }
                        let pre = enc(&prefix)?;
                        // files to rewrite: small ones that are in the prefix
                        let incl: Vec<&std::path::PathBuf> = files.iter().take(fi.min(files.len())).collect();
                        let mut small: Vec<&std::path::PathBuf> = incl.iter().copied().filter(|f| std::fs::metadata(f).map(|m| m.len() < 9000).unwrap_or(false)).collect();
                        if small.is_empty() { small = incl.clone(); }
                        let ctx = pre.len() + 8192 + n;
                        let mut a = dec.new_batch(bsz + 1, ctx)?;
                        let mut sb = dec.new_batch(bsz + 1, ctx)?;
                        dec.reset()?;
                        let t0 = Instant::now();
                        dec.prefill(&pre)?;
                        let tp = t0.elapsed().as_secs_f64();
                        dec.load_slot(&mut a, 0)?;
                        let mut first = vec![];
                        let mut hist = vec![];
                        let mut sufl = 0;
                        for s in 1..=bsz {
                            dec.unload_slot(&a, 0)?;
                            let f = small[(s * 7) % small.len()];
                            let suf = enc(&format!("\nNow rewrite the file {} in full, with every comment line changed to start with `// NOTE:`. Output only the file.<｜Assistant｜></think>", f.display()))?;
                            sufl += suf.len();
                            first.push(dec.prefill(&suf)?);
                            dec.load_slot(&mut a, s)?;
                            dec.load_slot(&mut sb, s)?;
                            let mut h = pre.clone();
                            h.extend_from_slice(&suf);
                            h.push(*first.last().unwrap());
                            hist.push(h);
                        }
                        let slots: Vec<usize> = (1..=bsz).collect();
                        let t1 = Instant::now();
                        let mut cur = first.clone();
                        for _ in 1..n { cur = dec.batch_step(&mut a, &slots, &cur)?; }
                        let tpl = t1.elapsed().as_secs_f64();
                        let mut st: Vec<spec::SpecState> = hist.iter().map(|h| spec::SpecState::new(h)).collect();
                        let mut got = vec![1usize; bsz];
                        let mut next = first.clone();
                        let t2 = Instant::now();
                        loop {
                            let live: Vec<usize> = (0..bsz).filter(|&i| got[i] < n).collect();
                            if live.is_empty() { break; }
                            let ls: Vec<usize> = live.iter().map(|&i| i + 1).collect();
                            let nx: Vec<u32> = live.iter().map(|&i| next[i]).collect();
                            let mut refs: Vec<&mut spec::SpecState> = st.iter_mut().enumerate().filter(|(i, _)| got[*i] < n).map(|(_, x)| x).collect();
                            let em = dec.spec_step(&mut sb, &ls, &nx, &mut refs)?;
                            for (k, &i) in live.iter().enumerate() { got[i] += em[k].len(); next[i] = *em[k].last().unwrap(); }
                        }
                        let tsp = t2.elapsed().as_secs_f64();
                        let ng: usize = got.iter().sum::<usize>() - bsz;
                        let h0 = &hist[0];
                        let out0: Vec<u32> = st[0].tail(n);
                        let _ = h0;
                        println!("session 1 output: {:?}", tok.decode(&out0, false).unwrap_or_default().chars().take(400).collect::<String>());
                        println!("session 1 lookup hits {}, passes {}", st[0].ngram_hits, st[0].passes);
                        println!("lspec B={bsz} prefix {} tokens (prefill {:.0} s, {:.0} tok/s; per-session suffix ~{} tokens): plain {:.1} tok/s per session ({:.0} total); spec {:.1} tok/s per session ({:.0} total)",
                            pre.len(), tp, pre.len() as f64 / tp, sufl / bsz, (n - 1) as f64 / tpl, (bsz * (n - 1)) as f64 / tpl, ng as f64 / bsz as f64 / tsp, ng as f64 / tsp);
                    }
                    "rbench" => {
                        // rbench SLOTS CTX N PROMPT GEN [TURNS]: N requests of PROMPT synthetic tokens (distinct per request),
                        // GEN greedy tokens each, through the batched runner; TURNS > 1 sends each conversation's
                        // follow-up (its history + 256 new tokens) after it finishes (prefix reuse)
                        use dsx_core::server::BatchEngine;
                        let slots: usize = w[1].parse()?;
                        let bctx: usize = w[2].parse()?;
                        let n: usize = w[3].parse()?;
                        let plen: usize = w[4].parse()?;
                        let ngen: usize = w[5].parse()?;
                        let turns: usize = w.get(6).map(|s| s.parse()).transpose()?.unwrap_or(1);
                        let p = dsx_core::server::GenParams { max_new: ngen, temperature: 0.0, top_p: 1.0, seed: None, stop_token_ids: vec![], hesitation: None, images: Default::default() };
                        let mut r = runner::Runner::new(&mut dec, slots, bctx)?;
                        let prompt = |i: usize, len: usize| -> Vec<u32> { (0..len).map(|j| ((j * 7919 + i * 104729 + 13) % 120000) as u32).collect() };
                        let t0 = Instant::now();
                        let mut hist: Vec<Vec<u32>> = (0..n).map(|i| prompt(i, plen)).collect();
                        let mut left: Vec<usize> = vec![turns; n];
                        let mut got = vec![0usize; n];
                        let mut first: Vec<Option<f64>> = vec![None; n];
                        let mut queue: std::collections::VecDeque<usize> = (0..n).collect();
                        let mut active = 0usize;
                        let (mut steps, mut toks) = (0usize, 0usize);
                        let mut tdec = 0f64;
                        while !queue.is_empty() || active > 0 {
                            while r.has_room() {
                                let Some(i) = queue.pop_front() else { break };
                                r.start(i as u64, &hist[i], &p)?;
                                got[i] = 0;
                                active += 1;
                            }
                            let ts = Instant::now();
                            let out = r.step()?;
                            let dt = ts.elapsed().as_secs_f64();
                            steps += 1;
                            if out.len() > 1 { tdec += dt; toks += out.len(); }
                            for (id, tk) in out {
                                let i = id as usize;
                                if got[i] >= ngen { continue; } // tokens of a step past the end
                                first[i].get_or_insert(t0.elapsed().as_secs_f64());
                                got[i] += 1;
                                hist[i].push(tk);
                                if got[i] >= ngen {
                                    // history = fed tokens: the last emitted token was not fed
                                    hist[i].pop();
                                    r.finish(id);
                                    active -= 1;
                                    left[i] -= 1;
                                    if left[i] > 0 {
                                        let more = prompt(1000 + i * 31 + left[i], 256);
                                        hist[i].extend(more);
                                        queue.push_back(i);
                                    }
                                }
                            }
                        }
                        let tot = t0.elapsed().as_secs_f64();
                        let mut ttft: Vec<f64> = first.iter().flatten().copied().collect();
                        ttft.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        let gen_tot = n * ngen * turns;
                        println!("rbench {n} x ({plen} prompt + {ngen}) x {turns} turns on {slots} slots: {tot:.1} s, {:.0} tok/s generated overall; {steps} steps; batched steps {:.0} tok/s; first token median {:.1} s, max {:.1} s",
                            gen_tot as f64 / tot, toks as f64 / tdec.max(1e-9), ttft[ttft.len() / 2], ttft[ttft.len() - 1]);
                    }
                    "bbench" => {
                        // bbench B CTX [STEPS] [PROMPT]: B slots, each holding the same PROMPT-token context (default
                        // CTX/2), STEPS batched steps; tok/s and the kernel profile of one step
                        let bsz: usize = w[1].parse()?;
                        let ctx: usize = w[2].parse()?;
                        let steps: usize = w.get(3).map(|s| s.parse()).transpose()?.unwrap_or(16);
                        let plen: usize = w.get(4).map(|s| s.parse()).transpose()?.unwrap_or(ctx / 2);
                        let need = batch::Batch::state_bytes(&dec, bsz, ctx);
                        println!("batch state: {:.1} GB", need as f64 / 1e9);
                        let mut b = dec.new_batch(bsz, ctx)?;
                        dec.reset()?;
                        let pr: Vec<u32> = (0..plen).map(|i| ((i * 7919 + 13) % 120000) as u32).collect();
                        let tp = Instant::now();
                        let first = dec.prefill(&pr)?;
                        println!("prefill {plen} tokens: {:.1} s ({:.0} tok/s)", tp.elapsed().as_secs_f64(), plen as f64 / tp.elapsed().as_secs_f64());
                        for s in 0..bsz {
                            dec.load_slot(&mut b, s)?;
                        }
                        let slots: Vec<usize> = (0..bsz).collect();
                        let mut toks: Vec<u32> = (0..bsz).map(|i| (first + i as u32 * 31) % 120000).collect();
                        toks = dec.batch_step(&mut b, &slots, &toks)?;
                        dec.g.sync()?;
                        let t0 = Instant::now();
                        for _ in 0..steps {
                            toks = dec.batch_step(&mut b, &slots, &toks)?;
                        }
                        dec.g.sync()?;
                        let s = t0.elapsed().as_secs_f64() / steps as f64;
                        println!("B={bsz} ctx~{plen}: {:.1} ms/step = {:.0} tok/s", s * 1e3, bsz as f64 / s);
                        dec.g.prof_start();
                        let r = dec.batch_step(&mut b, &slots, &toks);
                        let p = dec.g.prof_end()?;
                        r?;
                        let tot: f32 = p.iter().map(|x| x.1).sum();
                        println!("  one step: {:.0} us in kernels ({} launches)", tot, p.iter().map(|x| x.2).sum::<usize>());
                        for (nm, t, k) in p.iter().take(30) {
                            println!("  {nm:<28} {t:>10.0} us {k:>6} launches");
                        }
                    }
                    "vwaves" => dsx_amd_vwaves(w[1].parse()?),
                    "env" => {
                        // env K [V]: set (or with no value, remove) a variable read at run time by debug switches
                        unsafe {
                            match w.get(2) {
                                Some(_) => std::env::set_var(w[1], w[2..].join(" ")),
                                None => std::env::remove_var(w[1]),
                            }
                        }
                    }
                    "vtest" => {
                        // decode a greedy continuation, then replay it through verify with the true continuation as
                        // drafts: every draft should be accepted; times the verify passes
                        let t: usize = w[1].parse()?;
                        let prompt: Vec<u32> = vec![0, 671, 6102, 294, 8760, 344, 11111, 16, 455, 6102, 294, 10322, 344];
                        let mut next = dec.prefill(&prompt)?;
                        let mut cont = vec![next];
                        for _ in 0..40 {
                            next = dec.step(next)?;
                            cont.push(next);
                        }
                        dec.reset()?;
                        let _ = dec.prefill(&prompt)?;
                        let (mut i, mut passes, mut acc_tot, mut us) = (0usize, 0usize, 0usize, 0f64);
                        while i + t < cont.len() {
                            let feed = &cont[i..i + t];
                            dec.g.sync()?;
                            let t0 = Instant::now();
                            let out = dec.verify(feed)?;
                            us += t0.elapsed().as_secs_f64() * 1e6;
                            if passes < 2 {
                                println!("  feed {:?}\n  out  {:?}\n  want {:?}", feed, out, &cont[i + 1..(i + t + 1).min(cont.len())]);
                            }
                            let mut acc = 0;
                            while acc + 1 < t && out[acc] == cont[i + acc + 1] {
                                acc += 1;
                            }
                            dec.commit(t, acc + 1)?;
                            passes += 1;
                            acc_tot += acc;
                            i += acc + 1;
                        }
                        println!("verify T={t}: {passes} passes, {:.2} drafts accepted per pass (of {}), {:.0} us per pass", acc_tot as f64 / passes as f64, t - 1, us / passes as f64);
                    }
                    "pbench" => {
                        // prompt throughput from an empty context (synthetic tokens, or @FILE of comma-separated ids),
                        // then decode at that context
                        let toks: Vec<u32> = if let Some(f) = w[1].strip_prefix('@') {
                            std::fs::read_to_string(f)?.split(|c: char| c == ',' || c.is_whitespace()).filter(|x| !x.is_empty()).map(|x| x.parse().unwrap()).collect()
                        } else {
                            let n: usize = w[1].parse()?;
                            (0..n).map(|i| ((i * 7919 + 13) % 120000) as u32).collect()
                        };
                        let n = toks.len();
                        dec.g.sync()?;
                        let t0 = Instant::now();
                        let next = dec.prefill(&toks)?;
                        dec.g.sync()?;
                        let s = t0.elapsed().as_secs_f64();
                        println!("prefill {n} tokens: {s:.2} s = {:.0} tok/s (next {next})", n as f64 / s);
                        let mut ts = vec![];
                        let mut tok = next;
                        for _ in 0..16 {
                            ts.push(dec.time_step(tok)?);
                            tok = (tok * 7 + 13) % 100000;
                        }
                        ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        println!("  decode after it: {:.0} us median ({:.1} tok/s)", ts[8], 1e6 / ts[8]);
                    }
                    "pcmp" => {
                        // pcmp EVALDIR N: residual stream after every layer, decode (eager) vs the prompt path one token at a
                        // time, positions 0..N of eval sequence 0; prints rel error per layer at each position
                        let n: usize = w[2].parse()?;
                        let toks: Vec<u32> = std::fs::read(Path::new(w[1]).join("heldout.i32"))?.chunks(4).map(|b| i32::from_le_bytes(b.try_into().unwrap()) as u32).collect();
                        let dir = "/root/pcmp";
                        std::fs::create_dir_all(dir)?;
                        dec.reset()?;
                        for p in 0..n {
                            let (_, d) = dec.step_dump(toks[p])?;
                            for (li, h) in d.iter().enumerate() {
                                std::fs::write(format!("{dir}/ref_p{p}_l{li:02}.bin"), h.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())?;
                            }
                        }
                        dec.reset()?;
                        unsafe { std::env::set_var("DSX_DUMP", dir) };
                        for p in 0..n {
                            dec.prefill(&toks[p..p + 1])?;
                        }
                        unsafe { std::env::remove_var("DSX_DUMP") };
                    }
                    "ceval" => {
                        // ceval DIR SEQS CHUNK: Lucebox KL through the prompt path in chunks of CHUNK tokens
                        let r = ceval_run(&mut dec, Path::new(w[1]), w[2].parse()?, 1024, w[3].parse()?)?;
                        println!("{r}");
                    }
                    "peval" => {
                        let r = eval_run_m(&mut dec, Path::new(w[1]), w.get(2).map(|s| s.parse()).transpose()?.unwrap_or(8), 1024, true)?;
                        println!("{r}");
                    }
                    "bveval" => {
                        let r = bveval_run(&mut dec, Path::new(w[1]), w[2].parse()?, w[3].parse()?, 1024)?;
                        println!("{r}");
                    }
                    "beval" => {
                        let r = beval_run(&mut dec, Path::new(w[1]), w.get(2).map(|s| s.parse()).transpose()?.unwrap_or(8), w.get(3).map(|s| s.parse()).transpose()?.unwrap_or(1), 1024)?;
                        println!("{r}");
                    }
                    "eval" => {
                        let r = eval_run(&mut dec, Path::new(w[1]), w.get(2).map(|s| s.parse()).transpose()?.unwrap_or(8), 1024)?;
                        println!("{r}");
                    }
                    "waves" => {
                        decode::TARGET_WAVES.store(w[1].parse()?, std::sync::atomic::Ordering::Relaxed);
                        if let Some(m) = w.get(2) {
                            decode::MIN_UNITS.store(m.parse()?, std::sync::atomic::Ordering::Relaxed);
                        }
                        dec.reload_kernels(&std::fs::read(w.get(3).copied().unwrap_or("/root/cur.hsaco"))?)?;
                        println!("waves {} min units {}", w[1], w.get(2).unwrap_or(&"-"));
                    }
                    "gtime" => {
                        // GEMV phase timing: needs a kernel built with -DGEMV_TIMING loaded; prints per-launch phases
                        dec.gemv_timing(w[1].parse()?)?;
                    }
                    "quit" => std::process::exit(0),
                    c => println!("unknown command {c}"),
                }
                Ok(())
            }));
            match r {
                Ok(Err(e)) => println!("error: {e:#}"),
                Err(_) => println!("error: command panicked"),
                Ok(Ok(())) => {}
            }
            dec.reset()?;
        }
        println!("== idle");
        use std::io::Write;
        std::io::stdout().flush()?;
    }
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::EngramJson { gguf, out } => engram_json(&gguf, &out),
        Cmd::Selftest => selftest::run(),
        Cmd::Export { model, rank, out } => {
            let dec = Decoder::load(&model, 4096, &read_rank(&rank)?, &[], None)?;
            dec.export(&model, &out)
        }
        Cmd::Serve { model, rank, ctx, host, port, max_tokens, api_key, slots, batch_ctx, admin_port, temperature, top_p } => serve::serve(model, read_rank(&rank)?, ctx, host, port, max_tokens, api_key, slots, batch_ctx.unwrap_or(ctx), admin_port, temperature, top_p),
        Cmd::Dev { model, rank, ctx, cmd } => dev(&model, &read_rank(&rank)?, ctx, &cmd),
        Cmd::Check { model, ids, n, rank, ref_cache } => {
            // ---- reference: tokens fed, logits per position, routed experts
            let (feed, ref_logits, routes, pos_routes): (Vec<u32>, Vec<Vec<f32>>, Vec<(usize, usize)>, Vec<Vec<Vec<u16>>>) = match ref_cache.as_ref().filter(|p| p.exists()) {
                Some(p) => {
                    let v: serde_json::Value = serde_json::from_slice(&std::fs::read(p)?)?;
                    let feed = v["feed"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
                    let routes = v["routes"].as_array().unwrap().iter().map(|r| (r[0].as_u64().unwrap() as usize, r[1].as_u64().unwrap() as usize)).collect();
                    let pr: Vec<Vec<Vec<u16>>> = serde_json::from_value(v["pos_routes"].clone()).unwrap_or_default();
                    let lb = std::fs::read(p.with_extension("logits"))?;
                    let fl: Vec<f32> = lb.chunks(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
                    (feed, fl.chunks(decode::VOCAB).map(|c| c.to_vec()).collect(), routes, pr)
                }
                None => {
                    let t0 = Instant::now();
                    let m = dsx_core::model::Model::load_hf(&model)?;
                    eprintln!("reference loaded in {:.1} s", t0.elapsed().as_secs_f32());
                    let mut st = dsx_core::model::RefState::new(&m);
                    let mut feed = ids.clone();
                    let mut out = vec![];
                    let mut i = 0;
                    while i < feed.len() {
                        let t = Instant::now();
                        let lg = m.ref_step(&mut st, feed[i])?;
                        eprintln!("ref pos {i} tok {}: top {:?} ({:.1} s)", feed[i], top(&lg, 3), t.elapsed().as_secs_f32());
                        if i + 1 == feed.len() && feed.len() < ids.len() + n {
                            feed.push(top(&lg, 1)[0].0);
                        }
                        out.push(lg);
                        i += 1;
                    }
                    let mut routes = vec![];
                    for tr in &st.routes {
                        for (l, ex) in tr.iter().enumerate() {
                            for &e in ex {
                                routes.push((l, e as usize));
                            }
                        }
                    }
                    routes.sort();
                    routes.dedup();
                    let pr = st.routes.clone();
                    if let Some(p) = &ref_cache {
                        std::fs::write(p, serde_json::to_vec(&serde_json::json!({"feed": feed, "routes": routes, "pos_routes": pr}))?)?;
                        let b: Vec<u8> = out.iter().flatten().flat_map(|v| v.to_le_bytes()).collect();
                        std::fs::write(p.with_extension("logits"), b)?;
                    }
                    (feed, out, routes, pr)
                }
            };
            eprintln!("reference: {} positions, {} distinct routed experts", feed.len(), routes.len());
            let mut dec = Decoder::load(&model, 4096, &read_rank(&rank)?, &routes, None)?;
            let dump_dir = std::env::var("DSX_DUMP").ok();
            let npf: usize = std::env::var("DSXA_PREFILL").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
            if std::env::var_os("DSXA_PFCMP").is_some() {
                // one token through decode (eager, dumps) and through prefill (dumps), same position 0
                dec.dump = Some(Default::default());
                let _ = dec.step_dump(feed[0])?;
                dec.reset()?;
                dec.prefill(&feed[..1])?;
                return Ok(());
            }
            if npf > 0 {
                let t0 = Instant::now();
                let next = dec.prefill(&feed[..npf])?;
                let lg = dec.logits()?;
                let r = &ref_logits[npf - 1];
                let (lr, lgp) = (log_softmax(r), log_softmax(&lg));
                let kl: f64 = lr.iter().zip(&lgp).map(|(a, b)| a.exp() * (a - b)).sum();
                println!("prefill {npf} tokens ({:.2} s): next {next} top {:?} | ref top {:?} | KL {kl:.2e}", t0.elapsed().as_secs_f32(), top(&lg, 3), top(r, 3));
            }
            for (i, &tok) in feed.iter().enumerate() {
                if i < npf {
                    continue;
                }
                let next = if let (Some(dir), true) = (&dump_dir, i < 4) {
                    let (nx, hs) = dec.step_dump(tok)?;
                    for (l, h) in hs.iter().enumerate() {
                        let p = format!("{dir}/ref_p{i}_l{l:02}.bin");
                        if let Ok(b) = std::fs::read(&p) {
                            let r: Vec<f32> = b.chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
                            let md = r.iter().zip(h).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
                            let rn = r.iter().map(|a| a * a).sum::<f32>().sqrt();
                            let dn = r.iter().zip(h).map(|(a, b)| (a - b) * (a - b)).sum::<f32>().sqrt();
                            println!("  pos {i} layer {l}: max|dh| {md:.4}, rel {:.2e}", dn / rn);
                        }
                    }
                    nx
                } else {
                    dec.step(tok)?
                };
                let lg = dec.logits()?;
                let r = &ref_logits[i];
                let (lr, lgp) = (log_softmax(r), log_softmax(&lg));
                let kl: f64 = lr.iter().zip(&lgp).map(|(a, b)| a.exp() * (a - b)).sum();
                let md = r.iter().zip(&lg).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
                println!("pos {i} tok {tok}: gpu next {next} top {:?} | ref top {:?} | max|dlogit| {md:.4} KL {kl:.2e}", top(&lg, 3), top(r, 3));
                if let Some(pr) = pos_routes.get(i) {
                    let gr = dec.routes()?;
                    for (l, (a, b)) in gr.iter().zip(pr).enumerate() {
                        let mut a: Vec<i32> = a.clone();
                        let mut b: Vec<i32> = b.iter().map(|&x| x as i32).collect();
                        a.sort();
                        b.sort();
                        if a != b {
                            println!("  route differs at layer {l}: gpu {a:?} ref {b:?}");
                        }
                    }
                }
            }
            Ok(())
        }
        Cmd::Bench { model, ctx, steps, rank } => {
            let maxc = *ctx.iter().max().unwrap() + steps + 16;
            let mut dec = Decoder::load(&model, maxc, &read_rank(&rank)?, &[], None)?;
            dec.fill_random()?;
            for &c in &ctx {
                dec.set_pos(c)?;
                let mut tok = 1000u32;
                let mut ts = vec![];
                for s in 0..steps {
                    let us = dec.time_step(tok)?;
                    tok = (tok * 7 + 13) % 100000;
                    if s >= 2 {
                        ts.push(us);
                    }
                }
                ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let med = ts[ts.len() / 2];
                println!("ctx {c}: decode step {med:.0} us median ({:.1} tok/s), min {:.0} max {:.0}", 1e6 / med, ts[0], ts[ts.len() - 1]);
                if std::env::var_os("DSXA_PROF").is_some() {
                    let p = dec.step_prof(tok)?;
                    let tot: f32 = p.iter().map(|x| x.1).sum();
                    println!("  eager step: {tot:.0} us in {} launches", p.iter().map(|x| x.2).sum::<usize>());
                    for (n, t, k) in p.iter().take(30) {
                        println!("  {n:<28} {t:>8.1} us {k:>5} launches {:>6.2} us each", t / *k as f32);
                    }
                }
            }
            Ok(())
        }
        Cmd::Eval { model, eval, seqs, len, rank } => {
            let rd = |f: &str| std::fs::read(eval.join(f));
            let toks: Vec<u32> = rd("heldout.i32")?.chunks(4).map(|b| i32::from_le_bytes(b.try_into().unwrap()) as u32).collect();
            let tidx: Vec<i32> = rd("teacher_idx.i32")?.chunks(4).map(|b| i32::from_le_bytes(b.try_into().unwrap())).collect();
            let tlp: Vec<f32> = rd("teacher_lp.f32")?.chunks(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
            let topk = 256;
            let mut dec = Decoder::load(&model, len + 64, &read_rank(&rank)?, &[], None)?;
            let (mut nll, mut kl, mut agree, mut cnt) = (0f64, 0f64, 0usize, 0usize);
            let t0 = Instant::now();
            for sq in 0..seqs {
                dec.reset()?;
                let seq = &toks[sq * len..(sq + 1) * len];
                for p in 0..len - 1 {
                    dec.step(seq[p])?;
                    let l = dec.logits()?;
                    let lq = log_softmax(&l);
                    nll -= lq[seq[p + 1] as usize];
                    let row = sq * (len - 1) + p;
                    let (ti, tl) = (&tidx[row * topk..(row + 1) * topk], &tlp[row * topk..(row + 1) * topk]);
                    kl += ti.iter().zip(tl).map(|(&i, &lt)| (lt as f64).exp() * (lt as f64 - lq[i as usize])).sum::<f64>();
                    let am = l.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |a, (i, &v)| if v > a.1 { (i, v) } else { a }).0;
                    agree += (am as i32 == ti[0]) as usize;
                    cnt += 1;
                }
                eprintln!("seq {sq}: running ppl {:.3} kl {:.4} top1 {:.4} ({:.0} s)", (nll / cnt as f64).exp(), kl / cnt as f64, agree as f64 / cnt as f64, t0.elapsed().as_secs_f64());
            }
            println!("{{\"tokens\": {cnt}, \"nll\": {:.6}, \"ppl\": {:.4}, \"kl_top{topk}\": {:.6}, \"top1_agree\": {:.6}}}", nll / cnt as f64, (nll / cnt as f64).exp(), kl / cnt as f64, agree as f64 / cnt as f64);
            Ok(())
        }
        Cmd::Gen { model, ids, n, rank, ctx } => {
            let mut dec = Decoder::load(&model, ctx, &read_rank(&rank)?, &[], None)?;
            let t0 = Instant::now();
            let mut next = 0;
            for &t in &ids {
                next = dec.step(t)?;
            }
            let tp = t0.elapsed().as_secs_f64();
            let mut out = vec![next];
            let t1 = Instant::now();
            for _ in 1..n {
                next = dec.step(next)?;
                out.push(next);
            }
            let td = t1.elapsed().as_secs_f64();
            println!("ids: {out:?}");
            eprintln!("prompt {} tokens {:.2} s; {} tokens in {:.2} s = {:.1} tok/s", ids.len(), tp, n - 1, td, (n - 1) as f64 / td);
            Ok(())
        }
    }
}

fn dsx_amd_vwaves(n: usize) {
    verify::VWAVES.store(n, std::sync::atomic::Ordering::Relaxed);
}
