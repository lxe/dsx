use anyhow::Result;
use clap::{Parser, Subcommand};
use dsx::model::{Model, RefState};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser)]
#[command(about = "DeepSeek-V4.1-Flash on 2x RTX 3090")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// CPU reference forward, token by token: prints the top-5 after every prompt token, then decodes greedily.
    Ref {
        /// a GGUF export, or DeepSeek's HF checkpoint dir (FP8/FP4: the exact reference)
        #[arg(long)]
        model: PathBuf,
        /// HF checkpoint dir holding the Engram tables (GGUF models only)
        #[arg(long)]
        engram: Option<PathBuf>,
        /// prompt token ids, comma separated
        #[arg(long, value_delimiter = ',')]
        ids: Vec<u32>,
        #[arg(short, default_value_t = 8)]
        n: usize,
        /// write the final logits (f32 LE) here
        #[arg(long)]
        dump_logits: Option<PathBuf>,
    },
    /// DSpark acceptance on the CPU reference (HF checkpoint): greedy main decode, a 5-token draft after every
    /// token, scored against the tokens the main model actually produces next.
    Spec {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, value_delimiter = ',')]
        ids: Vec<u32>,
        #[arg(short, default_value_t = 32)]
        n: usize,
    },
    /// Prefill a prompt in chunks, then decode greedily; --check also runs the prompt token by token and compares.
    #[cfg(feature = "cuda")]
    Gen {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        engram: Option<PathBuf>,
        /// prompt token ids, comma separated, or @file with comma/whitespace separated ids
        #[arg(long)]
        ids: String,
        #[arg(short, default_value_t = 32)]
        n: usize,
        #[arg(long, default_value_t = 32768)]
        ctx: usize,
        #[arg(long, default_value_t = 4096)]
        chunk: usize,
        #[arg(long = "main-gb", alias = "gpu0-gb", value_name = "GB", default_value_t = -1.5, allow_negative_numbers = true)]
        gpu0_gb: f64,
        #[arg(long = "aux-gb", alias = "gpu1-gb", value_name = "GB", default_value_t = -1.0, allow_negative_numbers = true)]
        gpu1_gb: f64,
        #[arg(long, default_value_t = 0.0)]
        arena_gb: f64,
        #[arg(long)]
        check: bool,
        /// expert hotness ranking (rank.json from calibration)
        #[arg(long)]
        rank: Option<PathBuf>,
        /// after decoding, time verify passes of T tokens (speculative-decoding cost); state is not rolled back
        #[arg(long, default_value_t = 0)]
        bench_verify: usize,
        /// rebuild the model, prefill the prompt again and verify the first N decoded tokens in one pass: its logits
        /// against decode's (DSX_ADAPT=0 for a like-for-like placement)
        #[arg(long, default_value_t = 0)]
        check_verify: usize,
        /// speculative decoding with drafts looked up in the context (up to N per step), then the same tokens decoded
        /// plainly for comparison
        #[arg(long, default_value_t = 0)]
        spec: usize,
    },
    /// Quality on a held-out set against teacher top-k log-probs (Lucebox's 1017 eval: 8 x 1024 tokens, top-256):
    /// KL(teacher || model) over the teacher's top-k, top-1 agreement, perplexity.
    #[cfg(feature = "cuda")]
    Eval {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        engram: Option<PathBuf>,
        /// directory with heldout.i32 [S][L], teacher_idx.i32 / teacher_lp.f32 [S*(L-1)][K]
        #[arg(long)]
        eval: PathBuf,
        #[arg(long, default_value_t = 8)]
        seqs: usize,
        #[arg(long, default_value_t = 1024)]
        len: usize,
        #[arg(long, default_value_t = 256)]
        topk: usize,
        #[arg(long = "main-gb", alias = "gpu0-gb", value_name = "GB", default_value_t = -1.5, allow_negative_numbers = true)]
        gpu0_gb: f64,
        #[arg(long = "aux-gb", alias = "gpu1-gb", value_name = "GB", default_value_t = -1.0, allow_negative_numbers = true)]
        gpu1_gb: f64,
        #[arg(long, default_value_t = 0.0)]
        arena_gb: f64,
        #[arg(long)]
        rank: Option<PathBuf>,
        /// feed each sequence token by token through decode (step) instead of one prefill pass
        #[arg(long)]
        decode: bool,
        /// score only predictions of positions >= FROM in each sequence (llama-perplexity scores the second half)
        #[arg(long, default_value_t = 0)]
        from: usize,
    },
    /// Teacher-force a conversation through decode: per-position log-prob, rank and entropy of each continuation
    /// token, and the probability of ending the reasoning (`</think>`). TSV to stdout.
    #[cfg(feature = "cuda")]
    Score {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        engram: Option<PathBuf>,
        #[arg(long)]
        tokenizer: PathBuf,
        /// OpenAI chat-completions request (JSON); rendered with the server's V4.1 template
        #[arg(long)]
        request: PathBuf,
        /// text that follows the rendered prompt (reasoning, `</think>`, answer)
        #[arg(long)]
        continuation: PathBuf,
        #[arg(long = "main-gb", alias = "gpu0-gb", value_name = "GB", default_value_t = -1.5, allow_negative_numbers = true)]
        gpu0_gb: f64,
        #[arg(long = "aux-gb", alias = "gpu1-gb", value_name = "GB", default_value_t = -1.0, allow_negative_numbers = true)]
        gpu1_gb: f64,
        #[arg(long, default_value_t = 0.0)]
        arena_gb: f64,
        #[arg(long)]
        rank: Option<PathBuf>,
        /// also score the first N tokens in one prefill pass (column lp_prefill)
        #[arg(long, default_value_t = 0)]
        prefill_check: usize,
    },
    /// Copy the vision tower, aligner and image delimiter embeddings (bf16) from the official checkpoint into one
    /// `vision.safetensors` (what `dsx-server --vision` loads).
    VisionExtract {
        checkpoint: PathBuf,
        out: PathBuf,
    },
    /// Run the vision tower alone on one image and write its span rows (raw f32 [tokens][dim]).
    #[cfg(feature = "cuda")]
    VisionCheck {
        #[arg(long)]
        vision: PathBuf,
        #[arg(long)]
        image: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// Write a split GGUF (`<prefix>-00001-of-0000N.gguf`) as one file and check every tensor's bytes.
    Merge { input: PathBuf, out: PathBuf },
    /// Write a GGUF as llama.cpp-style split files (`<prefix>-0000k-of-0000N.gguf`) and check every tensor's bytes.
    Split {
        input: PathBuf,
        prefix: String,
        #[arg(long, default_value_t = 4)]
        shards: usize,
        /// new `general.name`
        #[arg(long)]
        name: Option<String>,
        /// write dsx.expert_rot from a quant output dir (QDIR/LNN.json "rot" per layer)
        #[arg(long)]
        expert_rot_from: Option<PathBuf>,
        /// another GGUF's tensors (e.g. the Engram table, per_layer_token_embd) as one more shard at the end
        #[arg(long)]
        add: Option<PathBuf>,
        /// store this tokenizer.json in the metadata (`dsx.tokenizer_json`): dsx-server then needs no --tokenizer
        #[arg(long)]
        tokenizer_json: Option<PathBuf>,
    },
    /// Render answered chat requests (JSON lines: system, user, effort, reasoning, answer) as training-style text with
    /// the server's V4.1 template: prompt, reasoning, `</think>`, answer, end of sentence. JSON lines {"text"} to stdout.
    Render {
        input: PathBuf,
    },
    /// GPU0 decode (experts on the CPU) next to the CPU reference, per-token logit comparison.
    #[cfg(feature = "cuda")]
    Cmp {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        engram: Option<PathBuf>,
        #[arg(long, value_delimiter = ',')]
        ids: Vec<u32>,
        #[arg(long, default_value_t = 4096)]
        ctx: usize,
        /// skip the CPU reference (GPU only, greedy)
        #[arg(long)]
        no_ref: bool,
        #[arg(short, default_value_t = 0)]
        n: usize,
        /// GB of routed experts to keep on the main GPU / on each other GPU (negative = all free memory minus that
        /// headroom; the main GPU is the best-linked one)
        #[arg(long = "main-gb", alias = "gpu0-gb", value_name = "GB", default_value_t = -1.5, allow_negative_numbers = true)]
        gpu0_gb: f64,
        #[arg(long = "aux-gb", alias = "gpu1-gb", value_name = "GB", default_value_t = -1.0, allow_negative_numbers = true)]
        gpu1_gb: f64,
        #[arg(long, default_value_t = 0.0)]
        arena_gb: f64,
    },
}

fn top(logits: &[f32], k: usize) -> Vec<(u32, f32)> {
    // partial selection, not a full sort of the 129K vocabulary
    let mut v: Vec<(u32, f32)> = logits.iter().enumerate().map(|(i, &x)| (i as u32, x)).collect();
    let k = k.min(v.len());
    v.select_nth_unstable_by(k - 1, |a, b| b.1.partial_cmp(&a.1).unwrap());
    v.truncate(k);
    v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    v
}

fn load(p: &std::path::Path, engram: Option<&std::path::Path>) -> Result<Model> {
    if p.is_dir() { Model::load_hf(p) } else { Model::load(p, engram) }
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Ref { model, engram, ids, n, dump_logits } => {
            let t0 = Instant::now();
            let m = load(&model, engram.as_deref())?;
            eprintln!("loaded in {:.1} s", t0.elapsed().as_secs_f32());
            let mut st = RefState::new(&m);
            let mut logits = vec![];
            for (i, &tok) in ids.iter().enumerate() {
                let t = Instant::now();
                logits = m.ref_step(&mut st, tok)?;
                println!("prompt[{i}] tok {tok}: top5 {:?}  ({:.2} s)", top(&logits, 5), t.elapsed().as_secs_f32());
            }
            let mut out = vec![];
            for _ in 0..n {
                let next = top(&logits, 1)[0].0;
                out.push(next);
                let t = Instant::now();
                logits = m.ref_step(&mut st, next)?;
                println!("gen tok {next}: next top5 {:?}  ({:.2} s)", top(&logits, 5), t.elapsed().as_secs_f32());
            }
            println!("generated ids: {out:?}");
            // expert overlap: distinct experts per layer over windows of w consecutive tokens (verify-batch cost)
            let n_tok = st.routes.len();
            for w in [1usize, 2, 4, 5, 6, 8] {
                if n_tok < w { break; }
                let mut tot = 0f64; let mut cnt = 0f64;
                for s0 in 0..=n_tok - w {
                    for l in 0..m.cfg.n_layers {
                        let mut set: Vec<u16> = (s0..s0 + w).flat_map(|t| st.routes[t][l].iter().cloned()).collect();
                        set.sort(); set.dedup();
                        tot += set.len() as f64; cnt += 1.0;
                    }
                }
                println!("window {w}: {:.1} distinct experts per layer (of {} picks)", tot / cnt, w * m.cfg.top_k);
            }
            if let Some(p) = dump_logits {
                let b: Vec<u8> = logits.iter().flat_map(|v| v.to_le_bytes()).collect();
                std::fs::write(p, b)?;
            }
        }
        Cmd::Spec { model, ids, n } => {
            let m = Model::load_hf(&model)?;
            let st_hf = dsx::safetensors::SafeTensors::open(&model, &[])?;
            let ds = dsx::dspark::Dspark::load(&m, &st_hf, &model)?;
            let mut dst = ds.new_state(&m);
            let mut st = RefState::new(&m);
            let mut logits = vec![];
            for &tok in &ids {
                logits = m.ref_step(&mut st, tok)?;
                ds.observe(&m, &mut dst, &st.main_hidden, st.pos - 1);
            }
            let mut drafts: Vec<(usize, Vec<u32>, Vec<f32>)> = vec![];
            let mut seq: Vec<u32> = vec![];
            for _ in 0..n {
                let next = top(&logits, 1)[0].0;
                // draft from the newest position before feeding `next`
                let (d, cf) = ds.draft(&m, &dst, st.pos - 1, next);
                drafts.push((seq.len() + 1, d, cf));
                seq.push(next);
                logits = m.ref_step(&mut st, next)?;
                ds.observe(&m, &mut dst, &st.main_hidden, st.pos - 1);
            }
            let mut hist = vec![0usize; ds.block + 1];
            for (start, d, cf) in &drafts {
                let mut a = 0;
                while a < d.len() && start + a < seq.len() && d[a] == seq[start + a] {
                    a += 1;
                }
                if start + d.len() <= seq.len() {
                    hist[a] += 1;
                }
                println!("at {start}: drafts {d:?} conf {:?} accepted {a}", cf.iter().map(|x| (x * 100.0).round() / 100.0).collect::<Vec<_>>());
            }
            let tot: usize = hist.iter().sum();
            let mean = hist.iter().enumerate().map(|(a, n)| a * n).sum::<usize>() as f64 / tot.max(1) as f64;
            println!("accepted-draft histogram {hist:?}; mean {mean:.2} drafts accepted per step (+1 main token)");
            // verify-batch cost: distinct experts per layer over windows of consecutive tokens
            let n_tok = st.routes.len();
            for w in [1usize, 2, 4, 6] {
                if n_tok < w { break; }
                let (mut tot, mut cnt) = (0f64, 0f64);
                for s0 in 0..=n_tok - w {
                    for l in 0..m.cfg.n_layers {
                        let mut set: Vec<u16> = (s0..s0 + w).flat_map(|t| st.routes[t][l].iter().cloned()).collect();
                        set.sort();
                        set.dedup();
                        tot += set.len() as f64;
                        cnt += 1.0;
                    }
                }
                println!("window {w}: {:.1} distinct experts per layer (of {} picks)", tot / cnt, w * m.cfg.top_k);
            }
            println!("generated ids: {seq:?}");
        }
        #[cfg(feature = "cuda")]
        Cmd::Gen { model, engram, ids, n, ctx, chunk, gpu0_gb, gpu1_gb, arena_gb, check, rank, bench_verify, check_verify, spec } => {
            let ids: Vec<u32> = if let Some(f) = ids.strip_prefix('@') {
                std::fs::read_to_string(f)?.split(|ch: char| ch == ',' || ch.is_whitespace()).filter(|x| !x.is_empty()).map(|x| x.parse().unwrap()).collect()
            } else {
                ids.split(',').map(|x| x.trim().parse().unwrap()).collect()
            };
            let m = load(&model, engram.as_deref())?;
            let mut gm = dsx::gpumodel::GpuModel::with_placement(&m, -1, ctx, dsx::gpumodel::Placement { gpu0_gb, aux_gb: gpu1_gb, arena_gb, prefill_chunk: chunk, rank: rank.clone() })?;
            let pb = gm.pb.take().unwrap();
            // chunk plan: the final chunk must hold the decoder cone (19 x 128 tokens) or the whole prompt
            let cone = (m.cfg.n_layers - 1 - m.cfg.cand_layer) * m.cfg.window;
            let total = ids.len();
            let mut bounds = vec![];
            let mut s0 = 0;
            while s0 < total {
                let mut e = (s0 + chunk).min(total);
                if e < total && total - e < cone {
                    // the final chunk takes the rest (it must hold the cone; shrinking this one to total - cone made
                    // 1-token chunks once s0 passed total - cone)
                    e = if total - cone > s0 { total - cone } else { total };
                }
                bounds.push((s0, e));
                s0 = e;
            }
            let t0 = Instant::now();
            let mut logits = vec![];
            for (i, &(a, b)) in bounds.iter().enumerate() {
                let tc = Instant::now();
                let r = gm.prefill_chunk(&pb, &ids[a..b], i + 1 == bounds.len())?;
                eprintln!("prefill chunk {a}..{b}: {:.2} s", tc.elapsed().as_secs_f64());
                if let Some(l) = r { logits = l; }
            }
            let tp = t0.elapsed().as_secs_f64();
            println!("prefill {total} tokens in {tp:.2} s ({:.0} tok/s); next top5 {:?}", total as f64 / tp, top(&logits, 5));
            if let Some(p) = &gm.prof {
                for (k, v) in p {
                    println!("  {k:<28} {:.2} s", v);
                }
            }
            let mut out = vec![];
            // --spec: rewind here afterwards and decode the same tokens speculatively
            let after_prompt = if spec > 0 { Some((gm.snapshot()?, logits.clone())) } else { None };
            if let Some(p) = gm.prof.as_mut() {
                p.clear();
            }
            if let Some(p) = gm.evprof.as_mut() {
                p.clear();
            }
            let busy0 = gm.cpu_busy_s();
            let (p0, c0) = gm.cpu_picks();
            let ad0 = gm.adapt_s;
            let split0 = gm.cpu_split_s();
            let td = Instant::now();
            let mut dec_logits: Vec<Vec<f32>> = vec![];
            for _ in 0..n {
                let next = top(&logits, 1)[0].0;
                out.push(next);
                logits = gm.step(next)?;
                if dec_logits.len() < check_verify {
                    dec_logits.push(logits.clone());
                }
            }
            let el = td.elapsed().as_secs_f64();
            let (p1, c1) = gm.cpu_picks();
            println!(
                "decode {n} tokens in {el:.2} s ({:.2} tok/s); CPU experts busy {:.1} ms/token, {:.1}% of picks on the CPU; GPU0 expert swaps {} ({:.2} ms/token host)",
                n as f64 / el,
                (gm.cpu_busy_s() - busy0) * 1e3 / n as f64,
                100.0 * (c1 - c0) as f64 / (p1 - p0).max(1) as f64,
                gm.adapt_swaps,
                (gm.adapt_s - ad0) * 1e3 / n as f64
            );
            if let Some(p) = &gm.prof {
                for (k, v) in p {
                    println!("  {k:<28} {:.2} ms/token", v * 1e3 / n as f64);
                }
            }
            let legs = gm.handoff_legs_us();
            if !legs.is_empty() && std::env::var_os("DSX_SPIN_WAIT").is_some() {
                let s: Vec<String> = legs.iter().enumerate().map(|(l, (a, b))| format!("{l}:{a:.0}/{b:.0}")).collect();
                println!("  handoff legs us (GPU->CPU / CPU->GPU): {}", s.join(" "));
            }
            let sp = gm.cpu_split_s();
            println!("  CPU worker idle / prep / forward: {:.2} / {:.2} / {:.2} ms/token", (sp[0] - split0[0]) * 1e3 / n as f64, (sp[1] - split0[1]) * 1e3 / n as f64, (sp[2] - split0[2]) * 1e3 / n as f64);
            if let Some(p) = &gm.evprof {
                let steps = p.get("steps").copied().unwrap_or(1.0);
                for (k, v) in p.iter().filter(|(k, _)| **k != "steps") {
                    println!("  {k:<40} {:.2} ms/token", v / steps);
                }
            }
            if gm.pregate_stats[0] > 0 {
                let p = gm.pregate_stats;
                println!("pre-gating (DSX_PREGATE=1: layer L+1's router on layer L's expert input; 2: layer L's router on its attention input): {:.1}% of picks in the predicted top-6, {:.1}% in the top-8; CPU picks {:.1}% of all, {:.1}% of them in the predicted top-8",
                    100.0 * p[1] as f64 / p[0] as f64, 100.0 * p[2] as f64 / p[0] as f64, 100.0 * p[3] as f64 / p[0] as f64, 100.0 * p[4] as f64 / p[3].max(1) as f64);
            }
            println!("generated: {out:?}");
            if spec > 0 {
                // the same prompt again, speculatively: drafts from the context, verified in one pass, the rest rewound
                let (snap, lg) = after_prompt.unwrap();
                gm.restore(&snap)?;
                let oracle = std::env::var_os("DSX_SPEC_ORACLE").is_some();
                if oracle {
                    // the reference: greedy decode through 1-token verify passes, so speculative decoding (same
                    // arithmetic) must reproduce it exactly, and its own continuation makes perfect drafts
                    let mut l = lg.clone();
                    out.clear();
                    let t = Instant::now();
                    for _ in 0..n {
                        let nx = top(&l, 1)[0].0;
                        out.push(nx);
                        l = gm.verify_spec(&pb, &[nx])?.pop().unwrap();
                        gm.keep(1)?;
                    }
                    println!("verify-path decode {n} tokens in {:.2} s ({:.2} tok/s)", t.elapsed().as_secs_f64(), n as f64 / t.elapsed().as_secs_f64());
                    gm.restore(&snap)?;
                }
                if std::env::var_os("DSX_SPEC_ROLLTEST").is_some() {
                    // rollback check that numerics cannot fool: every step verifies [next, dummy] and keeps 1; the
                    // first token cannot depend on the dummy, so two dummies must give the same tokens
                    let mut seqs = vec![];
                    for dummy in [11u32, 4242] {
                        gm.restore(&snap)?;
                        let mut l = lg.clone();
                        let mut s2 = vec![];
                        for _ in 0..n {
                            let nx = top(&l, 1)[0].0;
                            s2.push(nx);
                            l = gm.verify_spec(&pb, &[nx, dummy])?.swap_remove(0);
                            gm.keep(1)?;
                        }
                        seqs.push(s2);
                    }
                    let same = seqs[0].iter().zip(&seqs[1]).take_while(|(a, b)| a == b).count();
                    println!("rollback test: dummies 11 vs 4242 agree on the first {same}/{n} tokens");
                    return Ok(());
                }
                let (g2, pb2) = (&mut gm, &pb);
                let mut ctxt: Vec<u32> = ids.clone();
                let mut out2: Vec<u32> = vec![];
                let (mut passes, mut drafted, mut accepted) = (0usize, 0usize, 0usize);
                let t0 = Instant::now();
                let mut next = top(&lg, 1)[0].0;
                while out2.len() < n {
                    ctxt.push(next);
                    let room = (n - out2.len() - 1).min(spec).min(dsx::handoff::MAXT - 1);
                    // DSX_SPEC_ORACLE=1: drafts are plain decode's own continuation (a rollback check and the best case)
                    let drafts = if room == 0 {
                        vec![]
                    } else if oracle {
                        let i = out2.len() + 1;
                        out[i.min(out.len())..(i + room).min(out.len())].to_vec()
                    } else {
                        dsx::spec::lookup(&ctxt, room)
                    };
                    if drafts.is_empty() {
                        out2.push(next);
                        let l = g2.step(next)?;
                        next = top(&l, 1)[0].0;
                        continue;
                    }
                    let mut toks = vec![next];
                    toks.extend(&drafts);
                    let lv = g2.verify_spec(&pb2, &toks)?;
                    let mut j = 0;
                    while j < drafts.len() && top(&lv[j], 1)[0].0 == drafts[j] {
                        j += 1;
                    }
                    g2.keep(1 + j)?;
                    out2.push(next);
                    out2.extend(&drafts[..j]);
                    ctxt.extend(&drafts[..j]);
                    next = top(&lv[j], 1)[0].0;
                    passes += 1;
                    drafted += drafts.len();
                    accepted += j;
                }
                let el2 = t0.elapsed().as_secs_f64();
                out2.truncate(n);
                let same = out.iter().zip(&out2).take_while(|(a, b)| a == b).count();
                println!("spec decode {} tokens in {el2:.2} s ({:.2} tok/s vs {:.2} plain); {passes} verify passes, {accepted}/{drafted} drafts accepted; same as plain decode for the first {same}/{n} tokens",
                    out2.len(), out2.len() as f64 / el2, n as f64 / el);
                return Ok(());
            }
            if bench_verify > 0 {
                // the same tokens again, T at a time: what one speculative verify pass costs next to a decode step
                let mut times = vec![];
                if let Some(p) = gm.prof.as_mut() {
                    p.clear();
                }
                for i in 0..6 {
                    let toks: Vec<u32> = (0..bench_verify).map(|j| out[(i + j) % out.len()]).collect();
                    let t = Instant::now();
                    let lg = gm.verify(&pb, &toks)?;
                    times.push(t.elapsed().as_secs_f64());
                    let _ = lg;
                }
                times.sort_by(|a, b| a.partial_cmp(b).unwrap());
                println!("verify T={bench_verify}: median {:.1} ms per pass ({:.1} ms per token if all accepted) vs decode {:.1} ms per token",
                    times[3] * 1e3, times[3] * 1e3 / bench_verify as f64, el * 1e3 / n as f64);
                if let Some(p) = &gm.prof {
                    for (k, v) in p {
                        println!("  verify {k:<34} {:.2} ms/pass", v * 1e3 / 6.0);
                    }
                }
            }
            if check_verify > 0 {
                drop(gm);
                let mut g2 = dsx::gpumodel::GpuModel::with_placement(&m, -1, ctx, dsx::gpumodel::Placement { gpu0_gb, aux_gb: gpu1_gb, arena_gb, prefill_chunk: chunk, rank: rank.clone() })?;
                let pb2 = g2.pb.take().unwrap();
                for (i, &(a, b)) in bounds.iter().enumerate() {
                    g2.prefill_chunk(&pb2, &ids[a..b], i + 1 == bounds.len())?;
                }
                let lv = g2.verify(&pb2, &out[..check_verify])?;
                for (j, (v, d)) in lv.iter().zip(&dec_logits).enumerate() {
                    let maxd = v.iter().zip(d).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
                    println!("verify row {j}: top1 {} vs decode {}, max |logit diff| {maxd:.4}", top(v, 1)[0].0, top(d, 1)[0].0);
                }
                return Ok(());
            }
            if check {
                drop(gm);
                let mut g2 = dsx::gpumodel::GpuModel::with_placement(&m, -1, ctx, dsx::gpumodel::Placement { gpu0_gb, aux_gb: gpu1_gb, arena_gb, prefill_chunk: 0, rank: rank.clone() })?;
                let mut l2 = vec![];
                for &t in &ids { l2 = g2.step(t)?; }
                let maxd = l2.iter().zip(&logits).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
                let _ = maxd;
                println!("token-by-token next top5 {:?}", top(&l2, 5));
                let mut out2 = vec![];
                for _ in 0..n {
                    let next = top(&l2, 1)[0].0;
                    out2.push(next);
                    l2 = g2.step(next)?;
                }
                let same = out.iter().zip(&out2).take_while(|(a, b)| a == b).count();
                println!("token-by-token generated: {out2:?}\nidentical prefix: {same}/{n}");
            }
        }
        #[cfg(feature = "cuda")]
        Cmd::Eval { model, engram, eval, seqs, len, topk, gpu0_gb, gpu1_gb, arena_gb, rank, decode, from } => {
            let rd = |f: &str| std::fs::read(eval.join(f));
            let toks: Vec<u32> = rd("heldout.i32")?.chunks(4).map(|b| i32::from_le_bytes(b.try_into().unwrap()) as u32).collect();
            let tidx: Vec<i32> = rd("teacher_idx.i32")?.chunks(4).map(|b| i32::from_le_bytes(b.try_into().unwrap())).collect();
            let tlp: Vec<f32> = rd("teacher_lp.f32")?.chunks(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
            let m = load(&model, engram.as_deref())?;
            let mut gm = dsx::gpumodel::GpuModel::with_placement(&m, -1, len + 64, dsx::gpumodel::Placement { gpu0_gb, aux_gb: gpu1_gb, arena_gb, prefill_chunk: len, rank })?;
            let (mut nll, mut kl, mut agree, mut cnt) = (0f64, 0f64, 0usize, 0usize);
            let t0 = Instant::now();
            // --decode: the steps alone (teacher-forced: every arm decodes the same tokens), for speed A/Bs
            let (mut steps, mut t_steps) = (0usize, 0f64);
            for sq in 0..seqs {
                gm.reset();
                let seq = &toks[sq * len..(sq + 1) * len];
                let ts = Instant::now();
                let lg = if decode { seq[..len - 1].iter().map(|&tk| gm.step(tk)).collect::<Result<Vec<_>>>()? } else { gm.eval_logits(seq)? };
                if decode {
                    steps += len - 1;
                    t_steps += ts.elapsed().as_secs_f64();
                }
                for p in from.saturating_sub(1)..len - 1 {
                    let l = &lg[p];
                    let mx = l.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
                    let lse = mx + l.iter().map(|&v| (v as f64 - mx).exp()).sum::<f64>().ln();
                    let lq = |i: usize| l[i] as f64 - lse;
                    nll -= lq(seq[p + 1] as usize);
                    let row = sq * (len - 1) + p;
                    let (ti, tl) = (&tidx[row * topk..(row + 1) * topk], &tlp[row * topk..(row + 1) * topk]);
                    kl += ti.iter().zip(tl).map(|(&i, &lt)| (lt as f64).exp() * (lt as f64 - lq(i as usize))).sum::<f64>();
                    let am = l.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |a, (i, &v)| if v > a.1 { (i, v) } else { a }).0;
                    agree += (am as i32 == ti[0]) as usize;
                    cnt += 1;
                }
                eprintln!("seq {sq}: running ppl {:.3} kl {:.4} top1 {:.4} ({:.0} s)", (nll / cnt as f64).exp(), kl / cnt as f64, agree as f64 / cnt as f64, t0.elapsed().as_secs_f64());
            }
            if decode {
                println!("teacher-forced decode {steps} steps in {t_steps:.2} s ({:.2} tok/s)", steps as f64 / t_steps);
                if let Some(p) = &gm.prof {
                    for (k, v) in p {
                        println!("  {k:<28} {:.2} ms/token", v * 1e3 / steps as f64);
                    }
                }
            }
            println!("{{\"tokens\": {cnt}, \"nll\": {:.6}, \"ppl\": {:.4}, \"kl_top{topk}\": {:.6}, \"top1_agree\": {:.6}}}", nll / cnt as f64, (nll / cnt as f64).exp(), kl / cnt as f64, agree as f64 / cnt as f64);
        }
        #[cfg(feature = "cuda")]
        Cmd::VisionCheck { vision, image, out } => {
            let g = dsx::gpu::Gpu::new(0)?;
            let mut k = std::collections::HashMap::new();
            for &n in dsx::gpumodel::KERNELS {
                k.insert(n, g.kernel(n)?);
            }
            let t0 = Instant::now();
            let vw = dsx::gpumodel::VisionWeights::load(&vision)?;
            eprintln!("vision weights in {:.1} s", t0.elapsed().as_secs_f64());
            let im = dsx_core::vision::load(&std::fs::read(&image)?, 0)?;
            for pass in 0..2 {
                let t = Instant::now();
                let rows = dsx::gpumodel::encode_image(&g, &k, &vw, &im)?;
                eprintln!("pass {pass}: {}x{} patches -> {} tokens in {:.0} ms", im.n_vit_h, im.n_vit_w, im.len(), t.elapsed().as_secs_f64() * 1e3);
                if pass == 1 {
                    std::fs::write(&out, rows.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
                }
            }
        }
        Cmd::VisionExtract { checkpoint, out } => {
            let idx: serde_json::Value = serde_json::from_slice(&std::fs::read(checkpoint.join("model.safetensors.index.json"))?)?;
            let mut names: Vec<String> = idx["weight_map"].as_object().unwrap().keys()
                .filter(|k| k.starts_with("vision.") || k.starts_with("aligner.") || k.starts_with("image_")).cloned().collect();
            names.sort();
            anyhow::ensure!(!names.is_empty(), "no vision tensors in {}", checkpoint.display());
            let want: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
            let st = dsx::safetensors::SafeTensors::open(&checkpoint, &want)?;
            let (mut hdr, mut off) = (serde_json::Map::new(), 0usize);
            for n in &names {
                let t = st.get(n)?;
                let len = t.end - t.start;
                hdr.insert(n.clone(), serde_json::json!({"dtype": t.dtype, "shape": t.shape, "data_offsets": [off, off + len]}));
                off += len;
            }
            hdr.insert("__metadata__".into(), serde_json::json!({"format": "pt", "source": "deepseek-ai/DeepSeek-V4.1-Flash-1017"}));
            let mut h = serde_json::to_vec(&serde_json::Value::Object(hdr))?;
            while h.len() % 8 != 0 {
                h.push(b' ');
            }
            use std::io::Write;
            let mut w = std::io::BufWriter::new(std::fs::File::create(&out)?);
            w.write_all(&(h.len() as u64).to_le_bytes())?;
            w.write_all(&h)?;
            for n in &names {
                w.write_all(st.bytes(st.get(n)?))?;
            }
            w.flush()?;
            println!("{}: {} tensors, {:.2} GB", out.display(), names.len(), off as f64 / 1e9);
        }
        Cmd::Merge { input, out } => {
            dsx::gguf::merge(&input, &out)?;
            let (a, b) = (dsx::gguf::Gguf::open(&input)?, dsx::gguf::Gguf::open(&out)?);
            anyhow::ensure!(a.order == b.order, "tensor order differs");
            for k in &a.order {
                let (x, y) = (&a.tensors[k], &b.tensors[k]);
                anyhow::ensure!(x.dims == y.dims && x.ty == y.ty && a.bytes(x) == b.bytes(y), "tensor {k} differs");
            }
            // shards after the first carry general.alignment even when shard 1 has none (default 32)
            let same = |k: &String, v| k.starts_with("split.") || b.kv.get(k) == Some(v) || (k == "general.alignment" && !b.kv.contains_key(k));
            anyhow::ensure!(a.kv.iter().all(|(k, v)| same(k, v)), "metadata differs");
            anyhow::ensure!(!b.kv.keys().any(|k| k.starts_with("split.")), "split keys left");
            println!("{}: {} tensors and all metadata match {}", out.display(), b.order.len(), input.display());
        }
        Cmd::Split { input, prefix, shards, name, expert_rot_from, add, tokenizer_json } => {
            let mut arrays = vec![];
            if let Some(q) = &expert_rot_from {
                let nl = dsx::gguf::Gguf::open(&input)?.u64("deepseek41.num_hidden_layers")? as usize;
                let rot: Vec<u32> = (0..nl)
                    .map(|l| -> Result<u32> {
                        let j: serde_json::Value = serde_json::from_slice(&std::fs::read(q.join(format!("L{l:02}.json")))?)?;
                        Ok(j["rot"].as_u64().unwrap_or(0) as u32)
                    })
                    .collect::<Result<_>>()?;
                println!("dsx.expert_rot: {} of {nl} layers rotated", rot.iter().filter(|&&r| r != 0).count());
                arrays.push(("dsx.expert_rot".to_string(), rot));
            }
            let mut strs = vec![];
            if let Some(f) = &tokenizer_json {
                let s = std::fs::read_to_string(f)?;
                s.parse::<tokenizers::Tokenizer>().map_err(|e| anyhow::anyhow!("{}: {e}", f.display()))?;
                strs.push(("dsx.tokenizer_json".to_string(), s));
            }
            let paths = dsx::gguf::split(&input, &prefix, shards, name.as_deref(), &arrays, &strs, add.as_deref())?;
            let (a, b) = (dsx::gguf::Gguf::open(&input)?, dsx::gguf::Gguf::open(&paths[0])?);
            let xa = add.as_deref().map(dsx::gguf::Gguf::open).transpose()?;
            let na = a.order.len() + xa.as_ref().map_or(0, |x| x.order.len());
            anyhow::ensure!(na == b.order.len(), "tensor count {na} vs {}", b.order.len());
            for src in std::iter::once(&a).chain(xa.as_ref()) {
                for k in &src.order {
                    let (x, y) = (&src.tensors[k], b.get(k)?);
                    anyhow::ensure!(x.dims == y.dims && x.ty == y.ty && src.bytes(x) == b.bytes(y), "tensor {k} differs");
                }
            }
            let kv_same = a.kv.iter().all(|(k, v)| b.kv.get(k) == Some(v) || (name.is_some() && k == "general.name") || arrays.iter().any(|(n, _)| n == k) || strs.iter().any(|(n, _)| n == k));
            anyhow::ensure!(kv_same, "metadata differs");
            for (k, v) in &strs {
                anyhow::ensure!(b.kv.get(k).and_then(|x| x.as_str()) == Some(v.as_str()), "{k} not stored");
            }
            println!("{na} tensors and {} metadata keys match", a.kv.len());
        }
        Cmd::Render { input } => {
            use deepseek_recipe::request::{ConversionOptions, ProtocolRequest};
            use deepseek_recipe_encoding::PromptEncoding;
            let enc = deepseek_recipe_encoding::v4::dsv41::DeepseekV41Encoding::new();
            for line in std::fs::read_to_string(&input)?.lines().filter(|l| !l.trim().is_empty()) {
                let r: serde_json::Value = serde_json::from_str(line)?;
                let s = |k: &str| r.get(k).and_then(|v| v.as_str()).unwrap_or("").to_string();
                let req = serde_json::json!({"model": "deepseek-v4.1-flash", "reasoning_effort": s("effort"),
                    "messages": [{"role": "system", "content": s("system")}, {"role": "user", "content": s("user")}]});
                let req: deepseek_recipe::openai::ChatCompletionRequest = serde_json::from_value(req)?;
                let conv = req.convert(ConversionOptions::default()).map_err(|e| anyhow::anyhow!("{e:?}"))?;
                let prompt = enc.render_conversation(&conv.conversation).prompt;
                anyhow::ensure!(prompt.ends_with("<think>"), "prompt does not open a reasoning block: {:?}", &prompt[prompt.len().saturating_sub(80)..]);
                let text = format!("{prompt}{}</think>{}<｜end▁of▁sentence｜>", s("reasoning"), s("answer"));
                println!("{}", serde_json::json!({"text": text, "id": s("id")}));
            }
        }
        #[cfg(feature = "cuda")]
        Cmd::Score { model, engram, tokenizer, request, continuation, gpu0_gb, gpu1_gb, arena_gb, rank, prefill_check } => {
            use deepseek_recipe::request::{ConversionOptions, ProtocolRequest};
            use deepseek_recipe_encoding::PromptEncoding;
            let tok = tokenizers::Tokenizer::from_file(&tokenizer).map_err(|e| anyhow::anyhow!("{e}"))?;
            let req: deepseek_recipe::openai::ChatCompletionRequest = serde_json::from_slice(&std::fs::read(&request)?)?;
            let conv = req.convert(ConversionOptions::default()).map_err(|e| anyhow::anyhow!("{e:?}"))?;
            let prompt = deepseek_recipe_encoding::v4::dsv41::DeepseekV41Encoding::new().render_conversation(&conv.conversation).prompt;
            eprintln!("thinking={} effort={:?}; prompt tail: {:?}", conv.conversation.thinking_mode, conv.conversation.reasoning_effort, &prompt[prompt.len().saturating_sub(300)..]);
            let enc = |t: &str| tok.encode(t, false).map(|e| e.get_ids().to_vec()).map_err(|e| anyhow::anyhow!("{e}"));
            let mut ids = enc(&prompt)?;
            let np = ids.len();
            ids.extend(enc(&std::fs::read_to_string(&continuation)?)?);
            let end_think = tok.token_to_id("</think>").unwrap_or(u32::MAX) as usize;
            eprintln!("{np} prompt + {} continuation tokens; </think> = {end_think}", ids.len() - np);
            let m = load(&model, engram.as_deref())?;
            let chunk = prefill_check.max(4096);
            let mut gm = dsx::gpumodel::GpuModel::with_placement(&m, -1, ids.len() + 64, dsx::gpumodel::Placement { gpu0_gb, aux_gb: gpu1_gb, arena_gb, prefill_chunk: chunk, rank })?;
            let lsm = |l: &[f32]| {
                let mx = l.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
                let lse = mx + l.iter().map(|&v| (v as f64 - mx).exp()).sum::<f64>().ln();
                lse
            };
            let mut pre: Vec<f64> = vec![];
            if prefill_check > 0 {
                let n = prefill_check.min(ids.len());
                let lg = gm.eval_logits(&ids[..n])?;
                pre = (0..n - 1).map(|p| lg[p][ids[p + 1] as usize] as f64 - lsm(&lg[p])).collect();
                gm.reset();
            }
            println!("pos\ttarget\tlp\trank\ttop1\tlp_top1\tentropy\tlp_endthink\tlp_prefill");
            let t0 = Instant::now();
            for p in 0..ids.len() - 1 {
                let l = gm.step(ids[p])?;
                if p + 1 < np {
                    continue;
                }
                let lse = lsm(&l);
                let tgt = ids[p + 1] as usize;
                let lt = l[tgt];
                let rank = l.iter().filter(|&&v| v > lt).count();
                let (am, av) = l.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |a, (i, &v)| if v > a.1 { (i, v) } else { a });
                let ent: f64 = l.iter().map(|&v| { let lp = v as f64 - lse; if lp > -30.0 { -lp.exp() * lp } else { 0.0 } }).sum();
                let le = if end_think < l.len() { l[end_think] as f64 - lse } else { f64::NAN };
                let lpp = pre.get(p).copied().unwrap_or(f64::NAN);
                println!("{}\t{tgt}\t{:.4}\t{rank}\t{am}\t{:.4}\t{ent:.4}\t{le:.3}\t{lpp:.4}", p + 1 - np, lt as f64 - lse, av as f64 - lse);
                if (p + 1 - np) % 1000 == 0 {
                    eprintln!("{} / {} ({:.0} s)", p + 1 - np, ids.len() - np, t0.elapsed().as_secs_f64());
                }
            }
        }
        #[cfg(feature = "cuda")]
        Cmd::Cmp { model, engram, ids, ctx, no_ref, n, gpu0_gb, gpu1_gb, arena_gb } => {
            let m = load(&model, engram.as_deref())?;
            let mut gm = dsx::gpumodel::GpuModel::with_placement(&m, -1, ctx, dsx::gpumodel::Placement { gpu0_gb, aux_gb: gpu1_gb, arena_gb, prefill_chunk: 0, rank: None })?;
            let t_all = Instant::now();
            let mut st = RefState::new(&m);
            let mut toks = ids.clone();
            let mut i = 0;
            while i < toks.len() {
                let tok = toks[i];
                let t = Instant::now();
                let gl = gm.step(tok)?;
                let tg = t.elapsed().as_secs_f32();
                let gt = top(&gl, 5);
                if no_ref {
                    println!("[{i}] tok {tok}: gpu top5 {gt:?} ({tg:.3} s)");
                } else {
                    let t = Instant::now();
                    let rl = m.ref_step(&mut st, tok)?;
                    let tr = t.elapsed().as_secs_f32();
                    let rt = top(&rl, 5);
                    let maxd = gl.iter().zip(&rl).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
                    println!("[{i}] tok {tok}: gpu {:?} ref {:?} max|dlogit| {maxd:.4} (gpu {tg:.3} s, ref {tr:.2} s)", &gt[..3], &rt[..3]);
                }
                if i + 1 == toks.len() && toks.len() < ids.len() + n {
                    toks.push(gt[0].0);
                }
                i += 1;
            }
            println!("ids: {toks:?}");
            let el = t_all.elapsed().as_secs_f64();
            if let Some(p) = &gm.prof {
                for (k, v) in p {
                    println!("  {k:<10} {:.1} ms/token", v * 1e3 / toks.len() as f64);
                }
            }
            println!("{} tokens in {el:.1} s ({:.2} tok/s); CPU experts {:.1} s; routed evals main GPU/aux GPUs/CPU {:?}", toks.len(), toks.len() as f64 / el, gm.t_experts, gm.n_routed);
        }
    }
    Ok(())
}
