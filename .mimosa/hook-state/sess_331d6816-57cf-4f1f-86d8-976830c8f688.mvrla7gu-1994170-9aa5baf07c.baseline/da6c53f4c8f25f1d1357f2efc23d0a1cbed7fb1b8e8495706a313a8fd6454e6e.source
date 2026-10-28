//! Model config, weight handles over the GGUF mmap, and the reference forward pass.
//!
//! The reference runs on the CPU, one token at a time, in f32 with weights decoded row by row. It follows
//! DeepSeek's inference/model.py step by step (including the training-time FP8/FP4 fake quantization of the
//! caches), so every faster path in this crate is checked against it.

use crate::formats::{dequant_row, dequant_row_scaled};
use crate::gguf::{GType, Gguf, TensorInfo};
use crate::numerics::*;
use crate::safetensors::SafeTensors;
use anyhow::{Context, Result, bail};
use rayon::prelude::*;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct Config {
    pub vocab: usize,
    pub dim: usize,
    pub inter: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub rope_dim: usize,
    pub q_lora: usize,
    pub o_lora: usize,
    pub o_groups: usize,
    pub n_experts: usize,
    pub top_k: usize,
    pub window: usize,
    pub idx_heads: usize,
    pub idx_dim: usize,
    pub idx_topk: usize,
    pub cand_layer: usize,
    pub cand_blocks: usize,
    pub cand_block: usize,
    pub hc: usize,
    pub sinkhorn_iters: usize,
    pub hc_eps: f32,
    pub eps: f32,
    pub swiglu_limit: f32,
    pub route_scale: f32,
    pub rope_theta: f32,
    pub compress_rope_theta: f32,
    pub rope_factor: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
    pub original_seq_len: usize,
    pub compress_ratios: Vec<usize>,
    pub kv_sources: Vec<usize>,
    pub index_sources: Vec<usize>,
    pub engram_layers: Vec<usize>,
    pub engram_rows: Vec<u64>,
    pub engram_heads: usize,
    pub engram_dim: usize,
    pub engram_ngram: usize,
    pub engram_pad: i64,
    pub engram_token_map: Vec<u32>,
    pub engram_primes: Vec<u64>,      // [layer][ngram-1][head]
    pub engram_multipliers: Vec<u64>, // [layer][shift]
    pub dspark_targets: Vec<usize>,
    /// per layer: routed experts stored in the Hadamard-rotated input basis (numerics::fwht_blocks); GGUF dsx.expert_rot
    pub expert_rot: Vec<bool>,
}

impl Config {
    pub fn from_gguf(g: &Gguf) -> Result<Config> {
        if !g.kv.contains_key("deepseek41.config") {
            return Config::from_llama_gguf(g); // llama.cpp's deepseek41 keys (e.g. Unsloth's GGUFs, as shipped)
        }
        let k = |s: &str| format!("deepseek41.{s}");
        let u = |s: &str| g.u64(&k(s)).map(|v| v as usize);
        let f = |s: &str| g.f64(&k(s)).map(|v| v as f32);
        let arr = |s: &str| g.arr_u64(&k(s)).map(|v| v.into_iter().map(|x| x as usize).collect::<Vec<_>>());
        let cfg: serde_json::Value = serde_json::from_str(
            g.kv.get("deepseek41.config").and_then(|v| v.as_str()).context("deepseek41.config")?,
        )?;
        let tc = &cfg["text_config"];
        Ok(Config {
            vocab: u("vocab_size")?,
            dim: u("hidden_size")?,
            inter: u("moe_intermediate_size")?,
            n_layers: u("num_hidden_layers")?,
            n_heads: u("num_attention_heads")?,
            head_dim: u("head_dim")?,
            rope_dim: u("qk_rope_head_dim")?,
            q_lora: u("q_lora_rank")?,
            o_lora: u("o_lora_rank")?,
            o_groups: u("o_groups")?,
            n_experts: u("n_routed_experts")?,
            top_k: u("num_experts_per_tok")?,
            window: u("sliding_window")?,
            idx_heads: u("index_n_heads")?,
            idx_dim: u("index_head_dim")?,
            idx_topk: u("index_topk")?,
            cand_layer: u("candidate_source_layer_id")?,
            cand_blocks: u("candidate_topk_blocks")?,
            cand_block: u("candidate_block_size")?,
            hc: u("hc_mult")?,
            sinkhorn_iters: u("hc_sinkhorn_iters")?,
            hc_eps: f("hc_eps")?,
            eps: f("rms_norm_eps")?,
            swiglu_limit: f("swiglu_limit")?,
            route_scale: f("routed_scaling_factor")?,
            rope_theta: f("rope_theta")?,
            compress_rope_theta: f("compress_rope_theta")?,
            rope_factor: f("rope_scaling.factor")?,
            beta_fast: f("rope_scaling.beta_fast")?,
            beta_slow: f("rope_scaling.beta_slow")?,
            original_seq_len: f("rope_scaling.original_max_position_embeddings")? as usize,
            compress_ratios: arr("compress_ratios")?,
            kv_sources: arr("kv_source_layer_ids")?,
            index_sources: arr("index_source_layer_ids")?,
            engram_layers: arr("engram.layer_ids")?,
            engram_rows: g.arr_u64(&k("engram.rows"))?,
            engram_heads: tc["engram_n_heads"].as_u64().context("engram_n_heads")? as usize,
            engram_dim: tc["engram_head_dim"].as_u64().context("engram_head_dim")? as usize,
            engram_ngram: tc["engram_max_ngram_size"].as_u64().context("engram_max_ngram_size")? as usize,
            engram_pad: u("engram.pad_id")? as i64,
            engram_token_map: g.arr_u64(&k("engram.token_map"))?.into_iter().map(|v| v as u32).collect(),
            engram_primes: g.arr_u64(&k("engram.primes"))?,
            engram_multipliers: g.arr_u64(&k("engram.multipliers"))?,
            dspark_targets: tc["dspark_target_layer_ids"].as_array().map(|a| a.iter().filter_map(|x| x.as_u64()).map(|x| x as usize).collect()).unwrap_or_default(),
            expert_rot: {
                let n = u("num_hidden_layers")?;
                match g.arr_u64("dsx.expert_rot") {
                    Ok(v) => {
                        anyhow::ensure!(v.len() == n, "dsx.expert_rot has {} entries for {n} layers", v.len());
                        anyhow::ensure!(v.iter().all(|&b| b == 0 || b == crate::numerics::ROT as u64), "dsx.expert_rot: only block {} is supported", crate::numerics::ROT);
                        v.into_iter().map(|b| b != 0).collect()
                    }
                    Err(_) => vec![false; n],
                }
            },
        })
    }
}

impl Config {
    /// llama.cpp's deepseek41 metadata (convert_hf_to_gguf.py, PR #30817): the same model, other key names. Engram:
    /// ple.layers / ngram_size / heads_per_ngram, head_vocab_sizes (the bucket primes, [layer][ngram - 1][head]),
    /// layer_multipliers, engram.token_map and pad_token_id; each layer's rows are the sum of its heads' buckets.
    pub fn from_llama_gguf(g: &Gguf) -> Result<Config> {
        let k = |s: &str| format!("deepseek41.{s}");
        let u = |s: &str| g.u64(&k(s)).map(|v| v as usize);
        let f = |s: &str| g.f64(&k(s)).map(|v| v as f32);
        let arr = |s: &str| g.arr_u64(&k(s)).map(|v| v.into_iter().map(|x| x as usize).collect::<Vec<_>>());
        let n_layers = u("block_count")?;
        let token_map: Vec<u32> = g.arr_u64(&k("engram.token_map"))?.into_iter().map(|v| v as u32).collect();
        let (heads, ngram) = (u("ple.heads_per_ngram")?, u("ple.ngram_size")?);
        let engram_layers = arr("ple.layers")?;
        let primes = g.arr_u64(&k("ple.head_vocab_sizes"))?;
        let per = (ngram - 1) * heads;
        anyhow::ensure!(primes.len() == engram_layers.len() * per, "ple.head_vocab_sizes: {} entries for {} layers x {per} heads", primes.len(), engram_layers.len());
        let mut compress_ratios = arr("attention.compress_ratios")?;
        compress_ratios.truncate(n_layers); // the list also covers the MTP / DSpark blocks
        Ok(Config {
            vocab: token_map.len(),
            dim: u("embedding_length")?,
            inter: u("expert_feed_forward_length")?,
            n_layers,
            n_heads: u("attention.head_count")?,
            head_dim: u("attention.key_length")?,
            rope_dim: u("rope.dimension_count")?,
            q_lora: u("attention.q_lora_rank")?,
            o_lora: u("attention.output_lora_rank")?,
            o_groups: u("attention.output_group_count")?,
            n_experts: u("expert_count")?,
            top_k: u("expert_used_count")?,
            window: u("attention.sliding_window")?,
            idx_heads: u("attention.indexer.head_count")?,
            idx_dim: u("attention.indexer.key_length")?,
            idx_topk: u("attention.indexer.top_k")?,
            cand_layer: u("attention.indexer.candidate_source_layer")?,
            cand_blocks: u("attention.indexer.candidate_top_k_blocks")?,
            cand_block: u("attention.indexer.candidate_block_size")?,
            hc: u("hyper_connection.count")?,
            sinkhorn_iters: u("hyper_connection.sinkhorn_iterations")?,
            hc_eps: f("hyper_connection.epsilon")?,
            eps: f("attention.layer_norm_rms_epsilon")?,
            swiglu_limit: *g.arr_f64(&k("swiglu_clamp_exp"))?.first().context("swiglu_clamp_exp")? as f32,
            route_scale: f("expert_weights_scale")?,
            rope_theta: f("rope.freq_base")?,
            compress_rope_theta: f("attention.compress_rope_freq_base")?,
            rope_factor: f("rope.scaling.factor")?,
            beta_fast: f("rope.scaling.yarn_beta_fast")?,
            beta_slow: f("rope.scaling.yarn_beta_slow")?,
            original_seq_len: u("rope.scaling.original_context_length")?,
            compress_ratios,
            kv_sources: arr("attention.kv_source_layers")?,
            index_sources: arr("attention.indexer.source_layers")?,
            engram_rows: (0..engram_layers.len()).map(|i| primes[i * per..(i + 1) * per].iter().sum()).collect(),
            engram_layers,
            engram_heads: heads,
            engram_dim: u("embedding_length_per_layer_input")?,
            engram_ngram: ngram,
            engram_pad: u("engram.pad_token_id")? as i64,
            engram_token_map: token_map,
            engram_primes: primes,
            engram_multipliers: g.arr_u64(&k("ple.layer_multipliers"))?,
            dspark_targets: arr("dspark.target_layers").unwrap_or_default(),
            expert_rot: vec![false; n_layers],
        })
    }
}

impl Config {
    /// From an HF checkpoint's config.json plus the Engram constants sidecar (dsx-engram.json, written by
    /// xolotl's v41_gguf_meta.engram_layout: tokenizer-derived token map, bucket primes, hash multipliers).
    pub fn from_hf(dir: &Path) -> Result<Config> {
        let cfg: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
        let t = &cfg["text_config"];
        let e: serde_json::Value = serde_json::from_slice(
            &std::fs::read(dir.join("dsx-engram.json")).context("dsx-engram.json (engram constants) missing")?,
        )?;
        let u = |v: &serde_json::Value, k: &str| -> Result<usize> { v[k].as_u64().map(|x| x as usize).with_context(|| k.to_string()) };
        let f = |v: &serde_json::Value, k: &str| -> Result<f32> { v[k].as_f64().map(|x| x as f32).with_context(|| k.to_string()) };
        let arr = |v: &serde_json::Value| -> Vec<u64> { v.as_array().map(|a| a.iter().filter_map(|x| x.as_u64()).collect()).unwrap_or_default() };
        let rs = &t["rope_scaling"];
        Ok(Config {
            vocab: u(t, "vocab_size")?,
            dim: u(t, "hidden_size")?,
            inter: u(t, "moe_intermediate_size")?,
            n_layers: u(t, "num_hidden_layers")?,
            n_heads: u(t, "num_attention_heads")?,
            head_dim: u(t, "head_dim")?,
            rope_dim: u(t, "qk_rope_head_dim")?,
            q_lora: u(t, "q_lora_rank")?,
            o_lora: u(t, "o_lora_rank")?,
            o_groups: u(t, "o_groups")?,
            n_experts: u(t, "n_routed_experts")?,
            top_k: u(t, "num_experts_per_tok")?,
            window: u(t, "sliding_window")?,
            idx_heads: u(t, "index_n_heads")?,
            idx_dim: u(t, "index_head_dim")?,
            idx_topk: u(t, "index_topk")?,
            cand_layer: u(t, "candidate_source_layer_id")?,
            cand_blocks: u(t, "candidate_topk_blocks")?,
            cand_block: u(t, "candidate_block_size")?,
            hc: u(t, "hc_mult")?,
            sinkhorn_iters: u(t, "hc_sinkhorn_iters")?,
            hc_eps: f(t, "hc_eps")?,
            eps: f(t, "rms_norm_eps")?,
            swiglu_limit: f(t, "swiglu_limit")?,
            route_scale: f(t, "routed_scaling_factor")?,
            rope_theta: f(t, "rope_theta")?,
            compress_rope_theta: f(t, "compress_rope_theta")?,
            rope_factor: f(rs, "factor")?,
            beta_fast: f(rs, "beta_fast")?,
            beta_slow: f(rs, "beta_slow")?,
            original_seq_len: u(rs, "original_max_position_embeddings")?,
            compress_ratios: arr(&t["compress_ratios"]).into_iter().map(|x| x as usize).take(u(t, "num_hidden_layers")?).collect(),
            kv_sources: arr(&t["kv_source_layer_ids"]).into_iter().map(|x| x as usize).collect(),
            index_sources: arr(&t["index_source_layer_ids"]).into_iter().map(|x| x as usize).collect(),
            engram_layers: arr(&t["engram_layer_ids"]).into_iter().map(|x| x as usize).collect(),
            engram_rows: arr(&t["engram_num_embeddings"]),
            engram_heads: u(t, "engram_n_heads")?,
            engram_dim: u(t, "engram_head_dim")?,
            engram_ngram: u(t, "engram_max_ngram_size")?,
            engram_pad: e["pad_id"].as_i64().context("pad_id")?,
            engram_token_map: arr(&e["token_map"]).into_iter().map(|x| x as u32).collect(),
            engram_primes: e["primes"].as_array().context("primes")?.iter().flat_map(arr).collect(),
            engram_multipliers: e["multipliers"].as_array().context("multipliers")?.iter().flat_map(arr).collect(),
            dspark_targets: arr(&t["dspark_target_layer_ids"]).into_iter().map(|x| x as usize).collect(),
            expert_rot: vec![false; t["num_hidden_layers"].as_u64().unwrap_or(0) as usize],
        })
    }
}

/// One weight matrix: `rows` rows of `cols` weights starting at row `row0` of a tensor in an mmapped file
/// (GGUF block types, or the HF checkpoint's FP8 with 32x32-block E8M0 scales / packed FP4 with per-32 E8M0 scales).
/// Points into maps owned by the Model, which outlives every Mat.
#[derive(Clone)]
pub struct Mat {
    pub name: String,
    pub ty: GType,
    data: *const u8,
    scale: *const u8,
    pub row0: usize,
    pub rows: usize,
    pub cols: usize,
    /// FP8/FP4 checkpoint weights: DeepSeek's kernels fake-quantize the input to FP8 (block 32) first
    pub act_fp8: bool,
}

unsafe impl Send for Mat {}
unsafe impl Sync for Mat {}

impl Mat {
    pub fn from_gguf(g: &Gguf, t: &TensorInfo) -> Mat {
        Mat { name: t.name.clone(), ty: t.ty, data: g.bytes(t).as_ptr(), scale: std::ptr::null(), row0: 0, rows: t.n_rows(), cols: t.row_len(), act_fp8: false }
    }
    pub fn sub(&self, row0: usize, rows: usize) -> Mat {
        assert!(row0 + rows <= self.row0 + self.rows || self.row0 == 0, "{}: sub rows out of range", self.name);
        Mat { row0: self.row0 + row0, rows, ..self.clone() }
    }
    pub fn row_bytes(&self) -> usize {
        self.ty.row_bytes(self.cols)
    }
    /// bytes of rows [row0, row0 + rows); for T16, the whole matrix (rows x cols) starting at row0
    pub fn bytes(&self) -> &[u8] {
        if crate::t16::k2(self.ty).is_some() {
            let mb = crate::t16::matrix_bytes(self.ty, self.cols, self.rows);
            assert!(self.row0 % self.rows == 0, "{}: a T16 matrix is used whole", self.name);
            return unsafe { std::slice::from_raw_parts(self.data.add(self.row0 / self.rows * mb), mb) };
        }
        let rb = self.row_bytes();
        unsafe { std::slice::from_raw_parts(self.data.add(self.row0 * rb), self.rows * rb) }
    }
    pub fn row(&self, r: usize) -> &[u8] {
        let rb = self.row_bytes();
        unsafe { std::slice::from_raw_parts(self.data.add((self.row0 + r) * rb), rb) }
    }
    /// the E8M0 scale bytes of rows [row0, row0 + rows): FP8 = whole 32-row groups [rows/32][cols/32], FP4 = [rows][cols/32]
    pub fn scale_bytes(&self) -> &[u8] {
        let n = self.cols / 32;
        match self.ty {
            GType::Fp8B32 => unsafe { std::slice::from_raw_parts(self.scale.add(self.row0 / 32 * n), self.rows.div_ceil(32) * n) },
            GType::Fp4 => unsafe { std::slice::from_raw_parts(self.scale.add(self.row0 * n), self.rows * n) },
            _ => &[],
        }
    }
    pub fn dequant(&self, r: usize, out: &mut [f32]) {
        match self.ty {
            GType::Fp8B32 | GType::Fp4 => {
                let n = self.cols / 32;
                let sr = if self.ty == GType::Fp8B32 { (self.row0 + r) / 32 } else { self.row0 + r };
                let s = unsafe { std::slice::from_raw_parts(self.scale.add(sr * n), n) };
                dequant_row_scaled(self.ty, self.row(r), s, out);
            }
            t => dequant_row(t, self.row(r), out),
        }
    }
}

/// y = W x, rows in parallel, each decoded to f32 then dotted
pub fn matvec(w: &Mat, x: &[f32], y: &mut [f32]) {
    assert_eq!(x.len(), w.cols, "{}: input {} vs cols {}", w.name, x.len(), w.cols);
    assert_eq!(y.len(), w.rows, "{}: output", w.name);
    let xq;
    let x = if w.act_fp8 {
        let mut v = x.to_vec();
        fake_fp8(&mut v, 32);
        xq = v;
        &xq[..]
    } else {
        x
    };
    if crate::t16::k2(w.ty).is_some() {
        // no rows to decode one at a time: build the effective matrix once
        let e = crate::t16::dequant_effective(w.bytes(), w.ty, w.cols, w.rows);
        y.par_iter_mut().enumerate().for_each(|(j, yv)| *yv = dot(&e[j * w.cols..(j + 1) * w.cols], x));
        return;
    }
    y.par_chunks_mut(64).enumerate().for_each_init(
        || vec![0f32; w.cols],
        |buf, (ci, ys)| {
            for (j, yv) in ys.iter_mut().enumerate() {
                w.dequant(ci * 64 + j, buf);
                *yv = dot(buf, x);
            }
        },
    );
}

#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    // 8 independent accumulators so the compiler vectorizes
    let mut acc = [0f32; 8];
    let n = a.len() / 8 * 8;
    for i in (0..n).step_by(8) {
        for k in 0..8 {
            acc[k] += a[i + k] * b[i + k];
        }
    }
    let mut s: f32 = acc.iter().sum();
    for i in n..a.len() {
        s += a[i] * b[i];
    }
    s
}

fn mat_f32(m: &Mat) -> Vec<f32> {
    if crate::t16::k2(m.ty).is_some() {
        return crate::t16::dequant_effective(m.bytes(), m.ty, m.cols, m.rows);
    }
    let mut v = vec![0f32; m.rows * m.cols];
    for (r, chunk) in v.chunks_mut(m.cols).enumerate() {
        m.dequant(r, chunk);
    }
    v
}

pub struct Expert {
    pub gate: Mat,
    pub up: Mat,
    pub down: Mat,
}

pub struct Layer {
    pub id: usize,
    pub ratio: usize,
    pub attn_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub q_a: Mat,
    pub q_a_norm: Vec<f32>,
    pub q_b: Mat,
    pub kv: Mat,
    pub kv_norm: Vec<f32>,
    pub o_a: Mat,
    pub o_b: Mat,
    pub sinks: Vec<f32>,
    pub comp_kv: Option<Mat>,
    pub comp_gate: Option<Mat>,
    pub comp_norm: Option<Vec<f32>>,
    pub idx_q_b: Option<Mat>,
    pub idx_proj: Option<Mat>,
    pub idx_k: Option<Mat>,
    pub idx_k_norm: Option<Vec<f32>>,
    pub hc_attn_fn: Vec<f32>,
    pub hc_attn_base: Vec<f32>,
    pub hc_attn_scale: Vec<f32>,
    pub hc_ffn_fn: Vec<f32>,
    pub hc_ffn_base: Vec<f32>,
    pub hc_ffn_scale: Vec<f32>,
    pub gate_w: Vec<f32>,
    pub gate_bias: Vec<f32>,
    /// the selection bias for tokens inside an image span (`bias_vl`)
    pub gate_bias_vl: Option<Vec<f32>>,
    pub experts: Vec<Expert>,
    pub sh_gate: Mat,
    pub sh_up: Mat,
    pub sh_down: Mat,
    pub engram_kv: Option<Mat>,
    pub engram_qk: Option<Vec<f32>>, // q_weight * k_weight, [hc][dim]
}

/// The Engram embedding rows: the checkpoint's FP8 rows with an E8M0 scale per 32, or one quantized GGUF table with
/// every Engram layer's rows in layer order (llama.cpp's per_layer_token_embd, e.g. IQ4_NL).
pub enum EngramTables {
    Fp8 { st: SafeTensors, names: Vec<(String, String)> },
    /// `_file`: the GGUF holding the table when it is not the model's own (--engram FILE.gguf), kept mapped
    Table { mat: Mat, row0: Vec<usize>, _file: Option<Gguf> },
}

/// One Engram layer's rows (`EngramTables::layer`).
pub enum EngramLayer<'a> {
    Fp8 { w: &'a [u8], s: &'a [u8], dim: usize },
    Table { mat: &'a Mat, row0: usize },
}

impl EngramTables {
    pub fn layer(&self, li: usize, dim: usize) -> Result<EngramLayer<'_>> {
        Ok(match self {
            EngramTables::Fp8 { st, names } => {
                let (wn, sn) = &names[li];
                EngramLayer::Fp8 { w: st.bytes(st.get(wn)?), s: st.bytes(st.get(sn)?), dim }
            }
            EngramTables::Table { mat, row0, .. } => EngramLayer::Table { mat, row0: row0[li] },
        })
    }
}

impl EngramLayer<'_> {
    /// row `r` of this layer -> out [dim] (FP8 x E8M0 is exact in bf16, rounded as the reference does)
    pub fn row(&self, r: usize, out: &mut [f32]) {
        match self {
            EngramLayer::Fp8 { w, s, dim } => {
                for j in 0..*dim {
                    out[j] = round_bf16(fp8_e4m3_to_f32(w[r * dim + j]) * e8m0_to_f32(s[r * (dim / 32) + j / 32]));
                }
            }
            EngramLayer::Table { mat, row0 } => mat.dequant(row0 + r, out),
        }
    }
    /// the bytes `row` reads for row `r` (the codes; FP8 scales are 1/32 of them): what to prefetch or test for residency
    pub fn row_bytes(&self, r: usize) -> &[u8] {
        match self {
            EngramLayer::Fp8 { w, dim, .. } => &w[r * dim..(r + 1) * dim],
            EngramLayer::Table { mat, row0 } => mat.row(row0 + r),
        }
    }
}

/// Where the weights live: kept alive for as long as the Mats point into them.
pub enum Source {
    Gguf(Gguf),
    Hf(SafeTensors),
}

pub struct Model {
    pub src: Source,
    pub cfg: Config,
    pub embed: Mat,
    pub out_norm: Vec<f32>,
    pub head: Mat,
    pub layers: Vec<Layer>,
    pub engram: Option<EngramTables>,
    /// RoPE for pure sliding-window layers and for compressed layers (YaRN, compress_rope_theta)
    pub ropes: (Rope, Rope),
}

fn experts_of(g: &Gguf, l: usize, cfg: &Config) -> Result<Vec<Expert>> {
    let n = cfg.n_experts;
    let types = |key: &str| -> Result<Option<Vec<u32>>> {
        match g.tensors.get(&format!("blk.{l}.{key}")) {
            None => Ok(None),
            Some(t) => {
                let b = g.bytes(t);
                Ok(Some((0..n).map(|i| u32::from_le_bytes(b[4 * i..4 * i + 4].try_into().unwrap())).collect()))
            }
        }
    };
    let tname = |ty: u32| match ty {
        42 => "q2_0",
        11 => "q3_k",
        143 => "ptq1_0",
        16 => "iq2_xxs",
        1010 => "t16k15",
        1011 => "t16k2",
        1012 => "t16k25",
        _ => "?",
    };
    let gu = types("ffn_exps_gu_type")?;
    let dn = types("ffn_exps_down_type")?;
    let pick = |role: &str, map: &Option<Vec<u32>>, e: usize, rows: usize| -> Result<Mat> {
        match map {
            None => Ok(Mat::from_gguf(g, g.get(&format!("blk.{l}.ffn_{role}_exps.weight"))?).sub(e * rows, rows)),
            Some(m) => {
                let ty = m[e];
                let slot = m[..e].iter().filter(|&&x| x == ty).count();
                let t = g.get(&format!("blk.{l}.ffn_{role}_exps.{}.weight", tname(ty)))?;
                if GType::from_id(ty)? != t.ty {
                    bail!("blk.{l} expert {e}: type map {ty} vs tensor {:?}", t.ty);
                }
                Ok(Mat::from_gguf(g, t).sub(slot * rows, rows))
            }
        }
    };
    (0..n)
        .map(|e| Ok(Expert { gate: pick("gate", &gu, e, cfg.inter)?, up: pick("up", &gu, e, cfg.inter)?, down: pick("down", &dn, e, cfg.dim)? }))
        .collect()
}

fn engram_tables(cfg: &Config, dir: &Path) -> Result<EngramTables> {
    let names: Vec<(String, String)> = cfg
        .engram_layers
        .iter()
        .map(|l| (format!("layers.{l}.engram.embed.weight"), format!("layers.{l}.engram.embed.scale")))
        .collect();
    let want: Vec<&str> = names.iter().flat_map(|(a, b)| [a.as_str(), b.as_str()]).collect();
    let st = SafeTensors::open(dir, &want)?;
    st.advise_random();
    Ok(EngramTables::Fp8 { st, names })
}

fn ropes(cfg: &Config) -> (Rope, Rope) {
    (
        Rope::new(cfg.rope_dim, 0, cfg.rope_theta, cfg.rope_factor, cfg.beta_fast, cfg.beta_slow),
        Rope::new(cfg.rope_dim, cfg.original_seq_len, cfg.compress_rope_theta, cfg.rope_factor, cfg.beta_fast, cfg.beta_slow),
    )
}

impl Model {
    /// A GGUF export (dense Q8_0/F16/F32, routed experts Q2_0/PTQ1_0/Q3_K); Engram rows from the HF checkpoint dir.
    pub fn load(gguf: &Path, engram_dir: Option<&Path>) -> Result<Model> {
        let g = Gguf::open(gguf)?;
        let cfg = Config::from_gguf(&g)?;
        let m = |n: &str| -> Result<Mat> { Ok(Mat::from_gguf(&g, g.get(n)?)) };
        let om = |n: &str| -> Option<Mat> { g.tensors.get(n).map(|t| Mat::from_gguf(&g, t)) };
        let v = |n: &str| -> Result<Vec<f32>> { Ok(mat_f32(&m(n)?)) };
        let ov = |n: &str| -> Result<Option<Vec<f32>>> { om(n).map(|x| Ok(mat_f32(&x))).transpose() };
        let mut layers = vec![];
        for l in 0..cfg.n_layers {
            let p = |s: &str| format!("blk.{l}.{s}");
            let engram_qk = match (ov(&p("engram_q_norm.weight"))?, ov(&p("engram_k_norm.weight"))?) {
                (Some(q), Some(k)) => Some(q.iter().zip(&k).map(|(a, b)| a * b).collect()),
                _ => ov(&p("engram_gate.weight"))?, // llama.cpp stores the product q_weight * k_weight
            };
            layers.push(Layer {
                id: l,
                ratio: cfg.compress_ratios[l],
                attn_norm: v(&p("attn_norm.weight"))?,
                ffn_norm: v(&p("ffn_norm.weight"))?,
                q_a: m(&p("attn_q_a.weight"))?,
                q_a_norm: v(&p("attn_q_a_norm.weight"))?,
                q_b: m(&p("attn_q_b.weight"))?,
                kv: m(&p("attn_kv.weight"))?,
                kv_norm: v(&p("attn_kv_a_norm.weight"))?,
                o_a: m(&p("attn_output_a.weight"))?,
                o_b: m(&p("attn_output_b.weight"))?,
                sinks: v(&p("attn_sinks.weight"))?,
                comp_kv: om(&p("attn_compressor_kv.weight")),
                comp_gate: om(&p("attn_compressor_gate.weight")),
                comp_norm: ov(&p("attn_compressor_norm.weight"))?,
                idx_q_b: om(&p("indexer.attn_q_b.weight")),
                idx_proj: om(&p("indexer.proj.weight")),
                idx_k: om(&p("indexer.attn_k.weight")),
                idx_k_norm: ov(&p("indexer.k_norm.weight"))?,
                hc_attn_fn: v(&p("hc_attn_fn.weight"))?,
                hc_attn_base: v(&p("hc_attn_base.weight"))?,
                hc_attn_scale: v(&p("hc_attn_scale.weight"))?,
                hc_ffn_fn: v(&p("hc_ffn_fn.weight"))?,
                hc_ffn_base: v(&p("hc_ffn_base.weight"))?,
                hc_ffn_scale: v(&p("hc_ffn_scale.weight"))?,
                gate_w: v(&p("ffn_gate_inp.weight"))?,
                gate_bias: v(&p("exp_probs_b.bias"))?,
                gate_bias_vl: ov(&p("exp_probs_b_vl.bias"))?,
                experts: experts_of(&g, l, &cfg)?,
                sh_gate: m(&p("ffn_gate_shexp.weight"))?,
                sh_up: m(&p("ffn_up_shexp.weight"))?,
                sh_down: m(&p("ffn_down_shexp.weight"))?,
                engram_kv: om(&p("engram_kv.weight")),
                engram_qk,
            });
        }
        // one table with every layer's rows in layer order (llama.cpp's per_layer_token_embd, e.g. Unsloth's IQ4_NL)
        let table = |eg: &Gguf, t: &TensorInfo| -> Result<(Mat, Vec<usize>)> {
            let mut row0 = vec![];
            let mut acc = 0usize;
            for &r in &cfg.engram_rows {
                row0.push(acc);
                acc += r as usize;
            }
            anyhow::ensure!(acc == t.n_rows() && t.row_len() == cfg.engram_dim, "per_layer_token_embd: {} x {} for {acc} rows of {}", t.n_rows(), t.row_len(), cfg.engram_dim);
            eg.advise_random(t);
            Ok((Mat::from_gguf(eg, t), row0))
        };
        let engram = match (engram_dir, g.tensors.get("per_layer_token_embd.weight")) {
            // --engram FILE.gguf: the table from another GGUF (e.g. Unsloth's shard holding it), over the model's own
            (Some(d), _) if d.is_file() => {
                let eg = Gguf::open(d)?;
                let t = eg.get("per_layer_token_embd.weight")?.clone();
                let (mat, row0) = table(&eg, &t)?;
                Some(EngramTables::Table { mat, row0, _file: Some(eg) })
            }
            (_, Some(t)) => {
                let (mat, row0) = table(&g, t)?;
                Some(EngramTables::Table { mat, row0, _file: None })
            }
            (d, None) => d.map(|d| engram_tables(&cfg, d)).transpose()?,
        };
        Ok(Model {
            embed: m("token_embd.weight")?,
            out_norm: v("output_norm.weight")?,
            head: m("output.weight")?,
            layers,
            engram,
            ropes: ropes(&cfg),
            cfg,
            src: Source::Gguf(g),
        })
    }

    /// DeepSeek's own checkpoint (FP8 dense, FP4 experts): the exact reference. Engram from the same dir.
    pub fn load_hf(dir: &Path) -> Result<Model> {
        let cfg = Config::from_hf(dir)?;
        let st = SafeTensors::open(dir, &[])?;
        let m = |n: &str| -> Result<Mat> { hf_mat(&st, n) };
        let om = |n: &str| -> Result<Option<Mat>> { if st.tensors.contains_key(n) { Ok(Some(hf_mat(&st, n)?)) } else { Ok(None) } };
        let v = |n: &str| -> Result<Vec<f32>> { Ok(mat_f32(&hf_mat(&st, n)?)) };
        let ov = |n: &str| -> Result<Option<Vec<f32>>> { om(n)?.map(|x| Ok(mat_f32(&x))).transpose() };
        let mut layers = vec![];
        for l in 0..cfg.n_layers {
            let p = |s: &str| format!("layers.{l}.{s}");
            let engram_qk = match (ov(&p("engram.q_weight"))?, ov(&p("engram.k_weight"))?) {
                (Some(q), Some(k)) => Some(q.iter().zip(&k).map(|(a, b)| a * b).collect()),
                _ => None,
            };
            let experts = (0..cfg.n_experts)
                .map(|e| {
                    Ok(Expert {
                        gate: m(&p(&format!("ffn.experts.{e}.w1.weight")))?,
                        up: m(&p(&format!("ffn.experts.{e}.w3.weight")))?,
                        down: m(&p(&format!("ffn.experts.{e}.w2.weight")))?,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            layers.push(Layer {
                id: l,
                ratio: cfg.compress_ratios[l],
                attn_norm: v(&p("attn_norm.weight"))?,
                ffn_norm: v(&p("ffn_norm.weight"))?,
                q_a: m(&p("attn.wq_a.weight"))?,
                q_a_norm: v(&p("attn.q_norm.weight"))?,
                q_b: m(&p("attn.wq_b.weight"))?,
                kv: m(&p("attn.wkv.weight"))?,
                kv_norm: v(&p("attn.kv_norm.weight"))?,
                o_a: { let mut o = m(&p("attn.wo_a.weight"))?; o.act_fp8 = false; o }, // o is quantized explicitly before wo_a
                o_b: m(&p("attn.wo_b.weight"))?,
                sinks: v(&p("attn.attn_sink"))?,
                comp_kv: om(&p("attn.compressor.wkv.weight"))?,
                comp_gate: om(&p("attn.compressor.wgate.weight"))?,
                comp_norm: ov(&p("attn.compressor.norm.weight"))?,
                idx_q_b: om(&p("attn.indexer.wq_b.weight"))?,
                idx_proj: om(&p("attn.indexer.weights_proj.weight"))?,
                idx_k: om(&p("attn.indexer.wk.weight"))?,
                idx_k_norm: ov(&p("attn.indexer.k_norm.weight"))?,
                hc_attn_fn: v(&p("hc_attn_fn"))?,
                hc_attn_base: v(&p("hc_attn_base"))?,
                hc_attn_scale: v(&p("hc_attn_scale"))?,
                hc_ffn_fn: v(&p("hc_ffn_fn"))?,
                hc_ffn_base: v(&p("hc_ffn_base"))?,
                hc_ffn_scale: v(&p("hc_ffn_scale"))?,
                gate_w: v(&p("ffn.gate.weight"))?,
                gate_bias: v(&p("ffn.gate.bias"))?,
                gate_bias_vl: om(&p("ffn.gate.bias_vl"))?.map(|x| mat_f32(&x)),
                experts,
                sh_gate: m(&p("ffn.shared_experts.w1.weight"))?,
                sh_up: m(&p("ffn.shared_experts.w3.weight"))?,
                sh_down: m(&p("ffn.shared_experts.w2.weight"))?,
                engram_kv: om(&p("engram.wkv.weight"))?,
                engram_qk,
            });
        }
        let engram = Some(engram_tables(&cfg, dir)?);
        Ok(Model {
            embed: m("embed.weight")?,
            out_norm: v("norm.weight")?,
            head: m("head.weight")?,
            layers,
            engram,
            ropes: ropes(&cfg),
            cfg,
            src: Source::Hf(st),
        })
    }
}

/// An HF tensor as a Mat: F8_E4M3 (+ .scale 32x32 blocks), I8 = packed FP4 (+ .scale per 32), BF16/F16/F32.
pub fn hf_mat_pub(st: &SafeTensors, name: &str) -> Result<Mat> {
    hf_mat(st, name)
}

pub fn mat_f32_pub(m: &Mat) -> Vec<f32> {
    mat_f32(m)
}

fn hf_mat(st: &SafeTensors, name: &str) -> Result<Mat> {
    let t = st.get(name)?;
    let (rows, cols) = match t.shape.len() {
        1 => (1, t.shape[0]),
        2 => (t.shape[0], t.shape[1]),
        _ => bail!("{name}: shape {:?}", t.shape),
    };
    let scale_of = |n: &str| -> Result<*const u8> {
        let s = st.get(&(n.strip_suffix(".weight").unwrap_or(n).to_string() + ".scale"))?;
        Ok(st.bytes(s).as_ptr())
    };
    let (ty, cols, scale, act) = match t.dtype.as_str() {
        "F8_E4M3" => (GType::Fp8B32, cols, scale_of(name)?, true),
        "I8" => (GType::Fp4, cols * 2, scale_of(name)?, true),
        "BF16" => (GType::BF16, cols, std::ptr::null(), false),
        "F16" => (GType::F16, cols, std::ptr::null(), false),
        "F32" => (GType::F32, cols, std::ptr::null(), false),
        d => bail!("{name}: dtype {d}"),
    };
    Ok(Mat { name: name.to_string(), ty, data: st.bytes(t).as_ptr(), scale, row0: 0, rows, cols, act_fp8: act })
}

// ------------------------------------------------------------------ reference forward state

pub struct LayerState {
    pub window: Vec<Vec<f32>>, // ring of `window` slots, head_dim each (post-RoPE, fp8 fake-quantized)
    pub comp_kv_state: Vec<Vec<f32>>,
    pub comp_score_state: Vec<Vec<f32>>,
    pub comp_cache: Vec<Vec<f32>>, // compressed latents (post-RoPE, fp4 fake-quantized)
    pub idx_k_cache: Vec<Vec<f32>>,
}

pub struct RefState {
    pub pos: usize,
    pub h: Vec<f32>, // hc x dim residual stream
    pub layers: Vec<LayerState>,
    pub engram_hist: Vec<i64>, // compressed token ids
    // shared runtime (model.py SharedAttentionRuntime)
    comp_src: usize,
    idxk_src: usize,
    topk: Vec<i32>,
    candidates: Vec<bool>,
    /// routed experts picked per token: routes[token][layer] = top-k ids
    pub routes: Vec<Vec<Vec<u16>>>,
    /// DSpark input of the newest token: mean over hc copies of the stream entering each target layer
    pub main_hidden: Vec<f32>,
}

impl RefState {
    pub fn new(m: &Model) -> RefState {
        let c = &m.cfg;
        let layers = (0..c.n_layers)
            .map(|l| {
                let r = c.compress_ratios[l].max(1);
                LayerState {
                    window: vec![vec![0f32; c.head_dim]; c.window],
                    comp_kv_state: vec![vec![0f32; c.head_dim]; r],
                    comp_score_state: vec![vec![f32::NEG_INFINITY; c.head_dim]; r],
                    comp_cache: vec![],
                    idx_k_cache: vec![],
                }
            })
            .collect();
        RefState {
            pos: 0,
            h: vec![],
            layers,
            engram_hist: vec![],
            comp_src: usize::MAX,
            idxk_src: usize::MAX,
            topk: vec![],
            candidates: vec![],
            routes: vec![],
            main_hidden: vec![],
        }
    }
}

fn lin(_m: &Model, w: &Mat, x: &[f32]) -> Vec<f32> {
    let mut y = vec![0f32; w.rows];
    matvec(w, x, &mut y);
    y
}

/// model.py Block.hc_mixes + hc_split_sinkhorn for one token -> (pre[hc], post[hc], comb[hc][hc])
pub fn hc_mixes_pub(c: &Config, h: &[f32], fn_w: &[f32], scale: &[f32], base: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    hc_mixes(c, h, fn_w, scale, base)
}

fn hc_mixes(c: &Config, h: &[f32], fn_w: &[f32], scale: &[f32], base: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let hc = c.hc;
    let n = h.len();
    let ms = h.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let r = 1.0 / (ms + c.eps).sqrt();
    let mix_hc = (2 + hc) * hc;
    let mixes: Vec<f32> = (0..mix_hc).map(|i| dot(&fn_w[i * n..(i + 1) * n], h) * r).collect();
    let eps = c.hc_eps;
    let pre: Vec<f32> = (0..hc).map(|j| sigmoid(mixes[j] * scale[0] + base[j]) + eps).collect();
    let post: Vec<f32> = (0..hc).map(|j| 2.0 * sigmoid(mixes[j + hc] * scale[1] + base[j + hc])).collect();
    let mut comb = vec![0f32; hc * hc];
    for j in 0..hc {
        for k in 0..hc {
            comb[j * hc + k] = mixes[j * hc + k + 2 * hc] * scale[2] + base[j * hc + k + 2 * hc];
        }
    }
    // row softmax + eps
    for j in 0..hc {
        let row = &mut comb[j * hc..(j + 1) * hc];
        let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut s = 0.0;
        for v in row.iter_mut() {
            *v = (*v - mx).exp();
            s += *v;
        }
        for v in row.iter_mut() {
            *v = *v / s + eps;
        }
    }
    let col_norm = |comb: &mut Vec<f32>| {
        for k in 0..hc {
            let s: f32 = (0..hc).map(|j| comb[j * hc + k]).sum();
            for j in 0..hc {
                comb[j * hc + k] /= s + eps;
            }
        }
    };
    col_norm(&mut comb);
    for _ in 0..c.sinkhorn_iters - 1 {
        for j in 0..hc {
            let s: f32 = comb[j * hc..(j + 1) * hc].iter().sum();
            for k in 0..hc {
                comb[j * hc + k] /= s + eps;
            }
        }
        col_norm(&mut comb);
    }
    (pre, post, comb)
}

pub fn hc_pre_pub(c: &Config, h: &[f32], mix: &[f32]) -> Vec<f32> {
    hc_pre(c, h, mix)
}

fn hc_pre(c: &Config, h: &[f32], mix: &[f32]) -> Vec<f32> {
    let d = c.dim;
    let mut y = vec![0f32; d];
    for (k, w) in mix.iter().enumerate() {
        for i in 0..d {
            y[i] += w * h[k * d + i];
        }
    }
    round_bf16_slice(&mut y);
    y
}

pub fn hc_post_pub(c: &Config, x: &[f32], res: &[f32], post: &[f32], comb: &[f32]) -> Vec<f32> {
    hc_post(c, x, res, post, comb)
}

fn hc_post(c: &Config, x: &[f32], res: &[f32], post: &[f32], comb: &[f32]) -> Vec<f32> {
    let (d, hc) = (c.dim, c.hc);
    let mut y = vec![0f32; hc * d];
    for k in 0..hc {
        for i in 0..d {
            let mut v = post[k] * x[i];
            for j in 0..hc {
                v += comb[j * hc + k] * res[j * d + i];
            }
            y[k * d + i] = round_bf16(v);
        }
    }
    y
}

impl Model {
    /// hash ids of the 2..max_ngram n-grams ending at the newest position of `hist` (compressed token ids),
    /// model.py NgramHashState, for engram layer `li`
    pub fn engram_hashes_for(&self, hist: &[i64], li: usize) -> Vec<u64> {
        let c = &self.cfg;
        let n = c.engram_ngram;
        let t = hist.len() - 1;
        let mut toks = vec![0i64; n];
        let mut blocked = false;
        for s in 0..n {
            let src = if t >= s { hist[t - s] } else { -1 };
            blocked = blocked || t < s || src == -1;
            toks[s] = if blocked { c.engram_pad } else { src };
        }
        let mult = &c.engram_multipliers[li * n..(li + 1) * n];
        let per_layer = (n - 1) * c.engram_heads;
        let primes = &c.engram_primes[li * per_layer..(li + 1) * per_layer];
        let mut offsets = vec![0u64; per_layer];
        for i in 1..per_layer {
            offsets[i] = offsets[i - 1] + primes[i - 1];
        }
        let mut rolling = toks[0].wrapping_mul(mult[0] as i64);
        let mut out = vec![];
        for i in 1..n {
            rolling ^= toks[i].wrapping_mul(mult[i] as i64);
            for hd in 0..c.engram_heads {
                let col = (i - 1) * c.engram_heads + hd;
                out.push((rolling.rem_euclid(primes[col] as i64)) as u64 + offsets[col]);
            }
        }
        out
    }

    fn engram(&self, st: &mut RefState, l: &Layer) -> Result<()> {
        let c = &self.cfg;
        let li = c.engram_layers.iter().position(|&x| x == l.id).unwrap();
        let tabs = self.engram.as_ref().context("engram tables not loaded (--engram DIR)")?;
        let hd = c.engram_dim;
        let tab = tabs.layer(li, hd)?;
        let ids = self.engram_hashes_for(&st.engram_hist, li);
        let mut e = vec![0f32; ids.len() * hd];
        for (k, &row) in ids.iter().enumerate() {
            tab.row(row as usize, &mut e[k * hd..(k + 1) * hd]);
        }
        let kv = lin(self, l.engram_kv.as_ref().unwrap(), &e);
        let d = c.dim;
        let (key, value) = kv.split_at(c.hc * d);
        let w = l.engram_qk.as_ref().unwrap();
        for k in 0..c.hc {
            let h = &st.h[k * d..(k + 1) * d];
            let kk = &key[k * d..(k + 1) * d];
            let rh = 1.0 / (h.iter().map(|v| v * v).sum::<f32>() / d as f32 + c.eps).sqrt();
            let rk = 1.0 / (kk.iter().map(|v| v * v).sum::<f32>() / d as f32 + c.eps).sqrt();
            let dt: f32 = (0..d).map(|i| h[i] * w[k * d + i] * kk[i]).sum::<f32>() * rh * rk / (d as f32).sqrt();
            let gate = sigmoid(dt.abs().max(1e-6).sqrt().copysign(dt));
            for i in 0..d {
                st.h[k * d + i] = round_bf16(st.h[k * d + i] + gate * value[i]);
            }
        }
        Ok(())
    }

    fn attention(&self, st: &mut RefState, l: &Layer, x: &[f32]) -> Vec<f32> {
        let c = &self.cfg;
        let t = st.pos;
        let (hd, rd, nh) = (c.head_dim, c.rope_dim, c.n_heads);
        let rope = if l.ratio > 0 { &self.ropes.1 } else { &self.ropes.0 };
        let mut qr = lin(self, &l.q_a, x);
        rmsnorm_inplace(&mut qr, &l.q_a_norm, c.eps);
        round_bf16_slice(&mut qr);
        let mut q = lin(self, &l.q_b, &qr);
        for h in 0..nh {
            rope.apply(&mut q[h * hd + hd - rd..(h + 1) * hd], t, false);
        }
        let mut kv = lin(self, &l.kv, x);
        rmsnorm_inplace(&mut kv, &l.kv_norm, c.eps);
        rope.apply(&mut kv[hd - rd..], t, false);
        fake_fp8(&mut kv, 32);
        let w = c.window;
        st.layers[l.id].window[t % w] = kv;
        // window slots in ring order (oldest first), -1 = empty
        let oldest = t % w + 1;
        let mut slots: Vec<i64> = (oldest..w).chain(0..oldest).map(|i| i as i64).collect();
        for s in slots.iter_mut() {
            if *s as usize > t {
                *s = -1;
            }
        }
        // compressed rows this query reads
        let mut comp_rows: Vec<Vec<f32>> = vec![];
        if l.ratio > 0 {
            let ratio = l.ratio;
            let compress_len = (t + 1) / ratio;
            let is_kv_src = c.kv_sources.contains(&l.id);
            let mut latent: Option<Vec<f32>> = None;
            if is_kv_src {
                let ls = &mut st.layers[l.id];
                if ratio == 1 {
                    let mut v = lin(self, l.comp_kv.as_ref().unwrap(), x);
                    rmsnorm_inplace(&mut v, l.comp_norm.as_ref().unwrap(), c.eps);
                    latent = Some(v);
                } else {
                    let kvp = lin(self, l.comp_kv.as_ref().unwrap(), x);
                    let sc = lin(self, l.comp_gate.as_ref().unwrap(), x);
                    let slot = t % ratio;
                    ls.comp_kv_state[slot] = kvp;
                    ls.comp_score_state[slot] = sc;
                    if (t + 1) % ratio == 0 {
                        let mut v = vec![0f32; hd];
                        for j in 0..hd {
                            let mx = (0..ratio).map(|s| ls.comp_score_state[s][j]).fold(f32::NEG_INFINITY, f32::max);
                            let den: f32 = (0..ratio).map(|s| (ls.comp_score_state[s][j] - mx).exp()).sum();
                            v[j] = (0..ratio)
                                .map(|s| ls.comp_kv_state[s][j] * (ls.comp_score_state[s][j] - mx).exp() / den)
                                .sum();
                        }
                        round_bf16_slice(&mut v);
                        rmsnorm_inplace(&mut v, l.comp_norm.as_ref().unwrap(), c.eps);
                        latent = Some(v);
                    }
                }
                st.comp_src = l.id;
            }
            // indexer
            if c.index_sources.contains(&l.id) {
                if compress_len == 0 {
                    st.topk = vec![];
                } else {
                    st.topk = self.indexer(st, l, x, &qr, latent.as_deref(), w as i32);
                }
            }
            if let Some(mut lat) = latent {
                rope.apply(&mut lat[hd - rd..], t + 1 - ratio, false);
                fake_fp4(&mut lat, 16, true);
                st.layers[l.id].comp_cache.push(lat);
            }
            let src = &st.layers[st.comp_src].comp_cache;
            for &i in &st.topk {
                if i >= 0 {
                    comp_rows.push(src[i as usize - w].clone());
                }
            }
        }
        // sparse attention over window slots + compressed rows, MQA (one kv row is key and value), sink
        let scale = 1.0 / (hd as f32).sqrt();
        let ls = &st.layers[l.id];
        let rows: Vec<&Vec<f32>> = slots
            .iter()
            .filter(|&&s| s >= 0)
            .map(|&s| &ls.window[s as usize])
            .chain(comp_rows.iter())
            .collect();
        let mut o = vec![0f32; nh * hd];
        for h in 0..nh {
            let qh = &q[h * hd..(h + 1) * hd];
            let sc: Vec<f32> = rows.iter().map(|r| dot(qh, r) * scale).collect();
            let mx = sc.iter().cloned().fold(-1e30f32, f32::max);
            let ex: Vec<f32> = sc.iter().map(|s| (s - mx).exp()).collect();
            let den: f32 = ex.iter().sum::<f32>() + (l.sinks[h] - mx).exp();
            let oh = &mut o[h * hd..(h + 1) * hd];
            for (e, r) in ex.iter().zip(&rows) {
                let p = e / den;
                for j in 0..hd {
                    oh[j] += p * r[j];
                }
            }
            round_bf16_slice(oh);
            rope.apply(&mut oh[hd - rd..], t, true);
        }
        // grouped low-rank output projection
        let gsz = nh * hd / c.o_groups;
        let mut r = vec![0f32; c.o_groups * c.o_lora];
        for gi in 0..c.o_groups {
            let og = &mut o[gi * gsz..(gi + 1) * gsz];
            fake_fp8(og, 32);
            let wa = l.o_a.sub(gi * c.o_lora, c.o_lora);
            assert_eq!(wa.cols, gsz);
            matvec(&wa, og, &mut r[gi * c.o_lora..(gi + 1) * c.o_lora]);
        }
        round_bf16_slice(&mut r);
        lin(self, &l.o_b, &r)
    }

    fn indexer(&self, st: &mut RefState, l: &Layer, x: &[f32], qr: &[f32], latent: Option<&[f32]>, offset: i32) -> Vec<i32> {
        let c = &self.cfg;
        let t = st.pos;
        let ratio = l.ratio;
        let (id, rd, nh) = (c.idx_dim, c.rope_dim, c.idx_heads);
        let rope = &self.ropes.1;
        let owns_k = c.kv_sources.contains(&l.id);
        if owns_k {
            if let Some(lat) = latent {
                let mut k = lin(self, l.idx_k.as_ref().unwrap(), lat);
                rmsnorm_inplace(&mut k, l.idx_k_norm.as_ref().unwrap(), c.eps);
                rope.apply(&mut k[id - rd..], t + 1 - ratio, false);
                fake_fp4(&mut k, 32, false);
                st.layers[l.id].idx_k_cache.push(k);
            }
            st.idxk_src = l.id;
        }
        let mut q = lin(self, l.idx_q_b.as_ref().unwrap(), qr);
        for h in 0..nh {
            rope.apply(&mut q[h * id + id - rd..(h + 1) * id], t, false);
            fake_fp4(&mut q[h * id..(h + 1) * id], 32, false);
        }
        let wts = lin(self, l.idx_proj.as_ref().unwrap(), x);
        let wscale = (id as f32).powf(-0.5) * (nh as f32).powf(-0.5);
        let kc = &st.layers[st.idxk_src].idx_k_cache;
        let end = (t + 1) / ratio;
        let mut score: Vec<f32> = (0..end)
            .map(|s| {
                let k = &kc[s];
                (0..nh).map(|h| dot(&q[h * id..(h + 1) * id], k).max(0.0) * round_bf16(wts[h] * wscale)).sum()
            })
            .collect();
        let compress_len = end;
        if l.id == c.cand_layer {
            // level one: keep the best candidate blocks, pin the newest
            let bs = c.cand_block;
            let nb = end.div_ceil(bs);
            let mut bscore: Vec<f32> =
                (0..nb).map(|b| score[b * bs..((b + 1) * bs).min(end)].iter().cloned().fold(f32::NEG_INFINITY, f32::max)).collect();
            let last = (compress_len - 1) / bs;
            bscore[last] = f32::INFINITY;
            let mut order: Vec<usize> = (0..nb).collect();
            order.sort_by(|&a, &b| bscore[b].partial_cmp(&bscore[a]).unwrap());
            let mut keep = vec![false; end];
            for &b in order.iter().take(c.cand_blocks.min(nb)) {
                if bscore[b] > f32::NEG_INFINITY {
                    for i in b * bs..((b + 1) * bs).min(end) {
                        keep[i] = true;
                    }
                }
            }
            st.candidates = keep;
        } else if l.id > c.cand_layer {
            for (s, v) in score.iter_mut().enumerate() {
                if !st.candidates.get(s).copied().unwrap_or(false) {
                    *v = f32::NEG_INFINITY;
                }
            }
        }
        let k = c.idx_topk.min(end);
        let mut order: Vec<usize> = (0..end).collect();
        order.sort_by(|&a, &b| score[b].partial_cmp(&score[a]).unwrap().then(a.cmp(&b)));
        let mut idx: Vec<usize> = order[..k].to_vec();
        idx.sort();
        idx.iter().map(|&i| if i < compress_len && score[i] > f32::NEG_INFINITY { i as i32 + offset } else { -1 }).collect()
    }

    fn moe(&self, l: &Layer, x: &[f32], route_log: &mut Vec<u16>) -> Vec<f32> {
        let c = &self.cfg;
        let d = c.dim;
        let scores: Vec<f32> = (0..c.n_experts).map(|e| softplus(dot(&l.gate_w[e * d..(e + 1) * d], x)).sqrt()).collect();
        let mut order: Vec<usize> = (0..c.n_experts).collect();
        order.sort_by(|&a, &b| (scores[b] + l.gate_bias[b]).partial_cmp(&(scores[a] + l.gate_bias[a])).unwrap());
        let sel = &order[..c.top_k];
        route_log.extend(sel.iter().map(|&e| e as u16));
        if std::env::var_os("DSX_ROUTES").is_some() {
            let mut sk: Vec<f32> = order.iter().take(8).map(|&e| scores[e] + l.gate_bias[e]).collect();
            sk.iter_mut().for_each(|v| *v = (*v * 1e4).round() / 1e4);
            eprintln!("ref l{} {:?} keys {:?}", l.id, sel, sk);
        }
        let wsum: f32 = sel.iter().map(|&e| scores[e]).sum::<f32>() + 1e-20;
        let mut y = vec![0f32; d];
        // routed experts of a rotated layer are stored as W R: feed them R x and R h
        let rot = c.expert_rot.get(l.id).copied().unwrap_or(false);
        let xr: Vec<f32> = if rot { let mut v = x.to_vec(); crate::numerics::fwht_blocks(&mut v); v } else { vec![] };
        let mut run = |ex: (&Mat, &Mat, &Mat), wt: f32, routed: bool| {
            let xin: &[f32] = if routed && rot { &xr } else { x };
            let g = lin(self, ex.0, xin);
            let u = lin(self, ex.1, xin);
            let lim = c.swiglu_limit;
            let mut hmid: Vec<f32> = g
                .iter()
                .zip(&u)
                .map(|(&g, &u)| round_bf16(silu(g.min(lim)) * u.clamp(-lim, lim) * wt))
                .collect();
            if routed && rot {
                crate::numerics::fwht_blocks(&mut hmid);
            }
            let o = lin(self, ex.2, &hmid);
            for i in 0..d {
                y[i] += o[i];
            }
        };
        for &e in sel {
            let ex = &l.experts[e];
            run((&ex.gate, &ex.up, &ex.down), scores[e] / wsum * c.route_scale, true);
        }
        run((&l.sh_gate, &l.sh_up, &l.sh_down), 1.0, false);
        round_bf16_slice(&mut y);
        y
    }

    /// Feed one token, return the logits for the next one.
    pub fn ref_step(&self, st: &mut RefState, token: u32) -> Result<Vec<f32>> {
        let c = &self.cfg;
        let d = c.dim;
        st.engram_hist.push(c.engram_token_map[token as usize] as i64);
        let mut e = vec![0f32; d];
        self.embed.dequant(token as usize, &mut e);
        st.h = (0..c.hc).flat_map(|_| e.iter().cloned()).collect();
        let mut pre_mix = vec![0f32; c.hc];
        pre_mix[0] = 1.0;
        let mut routes = vec![];
        st.main_hidden.clear();
        for l in &self.layers {
            if l.engram_kv.is_some() {
                self.engram(st, l)?;
            }
            if c.dspark_targets.contains(&l.id) {
                for i in 0..d {
                    let v: f32 = (0..c.hc).map(|k| st.h[k * d + i]).sum::<f32>() / c.hc as f32;
                    st.main_hidden.push(round_bf16(v));
                }
            }
            let res = st.h.clone();
            let (a_pre, a_post, a_comb) = hc_mixes(c, &st.h, &l.hc_attn_fn, &l.hc_attn_scale, &l.hc_attn_base);
            let mut x = hc_pre(c, &st.h, &pre_mix);
            rmsnorm_inplace(&mut x, &l.attn_norm, c.eps);
            round_bf16_slice(&mut x);
            let a = self.attention(st, l, &x);
            st.h = hc_post(c, &a, &res, &a_post, &a_comb);
            let res = st.h.clone();
            let (f_pre, f_post, f_comb) = hc_mixes(c, &st.h, &l.hc_ffn_fn, &l.hc_ffn_scale, &l.hc_ffn_base);
            let mut x = hc_pre(c, &st.h, &a_pre);
            rmsnorm_inplace(&mut x, &l.ffn_norm, c.eps);
            round_bf16_slice(&mut x);
            let mut rl = vec![];
            let f = self.moe(l, &x, &mut rl);
            routes.push(rl);
            st.h = hc_post(c, &f, &res, &f_post, &f_comb);
            pre_mix = f_pre;
            dump("ref", st.pos, l.id, &st.h);
        }
        let mut x = hc_pre(c, &st.h, &pre_mix);
        rmsnorm_inplace(&mut x, &self.out_norm, c.eps);
        st.pos += 1;
        st.routes.push(routes);
        Ok(lin(self, &self.head, &x))
    }
}

pub fn fp8_e4m3_to_f32(b: u8) -> f32 {
    let s = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let e = ((b >> 3) & 0xF) as i32;
    let m = (b & 7) as f32;
    if e == 0 {
        s * m * 2f32.powi(-9)
    } else if e == 15 && (b & 7) == 7 {
        f32::NAN
    } else {
        s * (1.0 + m / 8.0) * 2f32.powi(e - 7)
    }
}

pub fn e8m0_to_f32(b: u8) -> f32 {
    f32::from_bits((b as u32) << 23)
}

/// DSX_DUMP=dir: write the residual stream after every layer for positions < 4 (debugging GPU vs reference)
pub fn dump(tag: &str, pos: usize, layer: usize, h: &[f32]) {
    if let Ok(dir) = std::env::var("DSX_DUMP") {
        if pos < 4 {
            let b: Vec<u8> = h.iter().flat_map(|v| v.to_le_bytes()).collect();
            let _ = std::fs::write(format!("{dir}/{tag}_p{pos}_l{layer:02}.bin"), b);
        }
    }
}
