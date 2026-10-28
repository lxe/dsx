//! V4.1's vision tower on GPU0 at prefill time: DeepSeek ViT (32 blocks, full attention, 2D RoPE) and the 3x3
//! aligner, as `inference/vision.py`. Weights stay in host memory and stream in one block at a time, so the
//! expert cache keeps its VRAM; activations are f32, GEMM inputs bf16.

use super::GpuModel;
use crate::gpu::{Arg, Gpu, Kernel};
use std::collections::HashMap;
use anyhow::{Context, Result, ensure};
use dsx_core::safetensors::SafeTensors;
use dsx_core::vision::{IMAGE, IMAGE_END, IMAGE_NEW_LINE, IMAGE_START, ImageInput, PATCH};
use std::path::Path;

const D: usize = 1024;
const H: usize = 16;
const INTER: usize = 2816;
const LAYERS: usize = 32;
const THETA: f32 = 10000.0;
/// query rows per attention step: scores are H x QC x n f32
const QC: usize = 256;
/// rows per MLP step
const MC: usize = 2048;

struct Block {
    norm1: Vec<f32>,
    norm2: Vec<f32>,
    wqkv: Vec<u16>,
    bqkv: Vec<f32>,
    wo: Vec<u16>,
    bo: Vec<f32>,
    w1: Vec<u16>,
    w2: Vec<u16>,
}

/// The vision weights (bf16 in the checkpoint), kept on the host.
pub struct VisionWeights {
    blocks: Vec<Block>,
    patch_w: Vec<u16>,
    patch_b: Vec<f32>,
    norm: Vec<f32>,
    al_w1: Vec<u16>,
    al_b1: Vec<f32>,
    al_w2: Vec<u16>,
    al_b2: Vec<f32>,
    start: Vec<f32>,
    end: Vec<f32>,
    newline: Vec<f32>,
    dim: usize,
}

fn bf16_to_f32(b: &[u16]) -> Vec<f32> {
    b.iter().map(|&v| f32::from_bits((v as u32) << 16)).collect()
}

impl VisionWeights {
    /// From a `vision.safetensors` sidecar, or from the official checkpoint directory (vision.*, aligner.*, image_*).
    pub fn load(path: &Path) -> Result<VisionWeights> {
        let st = if path.is_dir() {
            let idx: serde_json::Value = serde_json::from_slice(&std::fs::read(path.join("model.safetensors.index.json")).context("index json")?)?;
            let names: Vec<String> = idx["weight_map"]
                .as_object()
                .context("weight_map")?
                .keys()
                .filter(|k| k.starts_with("vision.") || k.starts_with("aligner.") || k.starts_with("image_"))
                .cloned()
                .collect();
            let want: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
            SafeTensors::open(path, &want)?
        } else {
            SafeTensors::open_file(path)?
        };
        let raw = |n: &str| -> Result<Vec<u16>> {
            let t = st.get(n)?;
            ensure!(t.dtype == "BF16", "{n}: {} (want BF16)", t.dtype);
            Ok(st.bytes(t).chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect())
        };
        let f = |n: &str| -> Result<Vec<f32>> { Ok(bf16_to_f32(&raw(n)?)) };
        let mut blocks = Vec::with_capacity(LAYERS);
        for i in 0..LAYERS {
            let p = |s: &str| format!("vision.blocks.{i}.{s}");
            blocks.push(Block {
                norm1: f(&p("norm1.weight"))?,
                norm2: f(&p("norm2.weight"))?,
                wqkv: raw(&p("attn.wqkv.weight"))?,
                bqkv: f(&p("attn.wqkv.bias"))?,
                wo: raw(&p("attn.wo.weight"))?,
                bo: f(&p("attn.wo.bias"))?,
                w1: raw(&p("mlp.w1.weight"))?,
                w2: raw(&p("mlp.w2.weight"))?,
            });
        }
        let start = f("image_start")?;
        Ok(VisionWeights {
            blocks,
            patch_w: raw("vision.patch_embed.proj.weight")?,
            patch_b: f("vision.patch_embed.proj.bias")?,
            norm: f("vision.norm.weight")?,
            al_w1: raw("aligner.w1.weight")?,
            al_b1: f("aligner.w1.bias")?,
            al_w2: raw("aligner.w2.weight")?,
            al_b2: f("aligner.w2.bias")?,
            dim: start.len(),
            start,
            end: f("image_end")?,
            newline: f("image_newline")?,
        })
    }
}

impl GpuModel<'_> {
    /// The span rows for one image, [im.len()][dim] f32 (see [`encode_image`]).
    pub fn encode_image(&self, vw: &VisionWeights, im: &ImageInput) -> Result<Vec<f32>> {
        encode_image(&self.g, &self.k, vw, im)
    }
}

fn ew(n: usize) -> (u32, u32, u32) {
    ((n as u32).div_ceil(256), 1, 1)
}

/// The span rows for one image, [im.len()][dim] f32: learned delimiter embeddings and, in reading order, the
/// aligner rows for the IMAGE slots. Needs only a GPU and the kernels (`dsx vision-check` runs it standalone).
pub fn encode_image(g: &Gpu, k: &HashMap<&'static str, Kernel>, vw: &VisionWeights, im: &ImageInput) -> Result<Vec<f32>> {
    {
        g.bind();
        let (nh, nw) = (im.n_vit_h, im.n_vit_w);
        let n = nh * nw;
        let pin = 3 * PATCH * PATCH;
        let bf = |count: usize| g.alloc(count * 2);
        let f4 = |count: usize| g.alloc(count * 4);
        // ---- patch embedding
        let patches_bf: Vec<u16> = im.patches.iter().map(|&v| half::bf16::from_f32(v).to_bits()).collect();
        let xp = g.upload(&patches_bf)?;
        let wbuf = bf((2 * INTER * D).max(3 * D * D))?; // the largest block matrix (mlp w1)
        let bbuf = f4(3 * D)?;
        let x = f4(n * D)?;
        let tmp = f4(n * 3 * D)?;
        g.htod_sync(wbuf.ptr, &vw.patch_w)?;
        g.htod_sync(bbuf.ptr, &vw.patch_b)?;
        g.gemm_bf16(xp.ptr, wbuf.ptr, tmp.ptr, n, pin, D, D)?;
        g.launch(k["vit_bias_add"], ew(n * D), 256, 0, &[Arg::P(x.ptr), Arg::P(tmp.ptr), Arg::P(bbuf.ptr), Arg::I(n as i32), Arg::I(D as i32), Arg::I(0)])?;
        drop(xp);
        // ---- blocks
        let xn = bf(n * D)?;
        let (q, kk, v) = (bf(n * D)?, bf(n * D)?, bf(n * D)?);
        let scores = f4(H * QC * n)?;
        let probs = bf(H * QC * n)?;
        let o = f4(H * QC * 64)?;
        let cat = bf(n * D)?;
        let act = bf(MC * INTER)?;
        let gu = f4(MC * 2 * INTER)?;
        let nbuf = f4(D)?;
        let w2buf = bf(D * INTER)?;
        // DSX_VIT_LAYERS=k: run only the first k blocks and return the normed ViT rows (debugging against vision.py)
        let dbg: Option<usize> = std::env::var("DSX_VIT_LAYERS").ok().and_then(|v| v.parse().ok());
        for b in vw.blocks.iter().take(dbg.unwrap_or(LAYERS)) {
            // attention
            g.htod_sync(nbuf.ptr, &b.norm1)?;
            g.launch(k["vit_rmsnorm"], (n as u32, 1, 1), 256, 0, &[Arg::P(x.ptr), Arg::P(nbuf.ptr), Arg::P(xn.ptr), Arg::I(D as i32), Arg::F(1e-6)])?;
            g.htod_sync(wbuf.ptr, &b.wqkv)?;
            g.htod_sync(bbuf.ptr, &b.bqkv)?;
            g.gemm_bf16(xn.ptr, wbuf.ptr, tmp.ptr, n, D, 3 * D, 3 * D)?;
            g.launch(k["vit_rope_split"], (n as u32, H as u32, 1), 32, 0, &[Arg::P(tmp.ptr), Arg::P(bbuf.ptr), Arg::P(q.ptr), Arg::P(kk.ptr), Arg::P(v.ptr), Arg::I(n as i32), Arg::I(H as i32), Arg::I(nw as i32), Arg::F(THETA)])?;
            let mut q0 = 0;
            while q0 < n {
                let qc = QC.min(n - q0);
                // scores[h][qc][n] = q[h][q0..][64] . k[h][n][64]^T
                g.gemm_bf16_sb(true, q.ptr + (q0 * 64 * 2) as u64, kk.ptr, scores.ptr, qc, 64, n, 64, 64, n, n * 64, n * 64, qc * n, H)?;
                g.launch(k["vit_softmax"], ((H * qc) as u32, 1, 1), 256, 0, &[Arg::P(scores.ptr), Arg::P(probs.ptr), Arg::I(n as i32), Arg::F(0.125)])?;
                // o[h][qc][64] = p[h][qc][n] . v[h][n][64]
                g.gemm_bf16_sb(false, probs.ptr, v.ptr, o.ptr, qc, n, 64, n, 64, 64, qc * n, n * 64, qc * 64, H)?;
                g.launch(k["vit_heads_merge"], (qc as u32, H as u32, 1), 64, 0, &[Arg::P(o.ptr), Arg::P(cat.ptr), Arg::I(H as i32), Arg::I(qc as i32), Arg::I(q0 as i32)])?;
                q0 += qc;
            }
            g.htod_sync(wbuf.ptr, &b.wo)?;
            g.htod_sync(bbuf.ptr, &b.bo)?;
            g.gemm_bf16(cat.ptr, wbuf.ptr, tmp.ptr, n, D, D, D)?;
            g.launch(k["vit_bias_add"], ew(n * D), 256, 0, &[Arg::P(x.ptr), Arg::P(tmp.ptr), Arg::P(bbuf.ptr), Arg::I(n as i32), Arg::I(D as i32), Arg::I(1)])?;
            // mlp
            g.htod_sync(nbuf.ptr, &b.norm2)?;
            g.launch(k["vit_rmsnorm"], (n as u32, 1, 1), 256, 0, &[Arg::P(x.ptr), Arg::P(nbuf.ptr), Arg::P(xn.ptr), Arg::I(D as i32), Arg::F(1e-6)])?;
            g.htod_sync(wbuf.ptr, &b.w1)?;
            g.htod_sync(w2buf.ptr, &b.w2)?;
            let mut r0 = 0;
            while r0 < n {
                let rc = MC.min(n - r0);
                g.gemm_bf16(xn.ptr + (r0 * D * 2) as u64, wbuf.ptr, gu.ptr, rc, D, 2 * INTER, 2 * INTER)?;
                g.launch(k["vit_swiglu"], ew(rc * INTER), 256, 0, &[Arg::P(gu.ptr), Arg::P(act.ptr), Arg::I(rc as i32), Arg::I(INTER as i32)])?;
                g.gemm_bf16(act.ptr, w2buf.ptr, tmp.ptr + (r0 * D * 4) as u64, rc, INTER, D, D)?;
                r0 += rc;
            }
            g.launch(k["vit_bias_add"], ew(n * D), 256, 0, &[Arg::P(x.ptr), Arg::P(tmp.ptr), Arg::P(0), Arg::I(n as i32), Arg::I(D as i32), Arg::I(1)])?;
        }
        drop((q, kk, v, scores, probs, o, cat, act, gu, w2buf));
        // final norm (bf16 out), back to f32 for the unfold
        g.htod_sync(nbuf.ptr, &vw.norm)?;
        g.launch(k["vit_rmsnorm"], (n as u32, 1, 1), 256, 0, &[Arg::P(x.ptr), Arg::P(nbuf.ptr), Arg::P(xn.ptr), Arg::I(D as i32), Arg::F(1e-6)])?;
        let xn_f = bf16_to_f32(&g.download::<u16>(xn.ptr, n * D)?);
        if dbg.is_some() {
            return Ok(xn_f);
        }
        g.htod_sync(x.ptr, &xn_f)?;
        // ---- aligner: 3x3 unfold (zero pad right/bottom), w1 + gelu, w2
        let (lh, lw) = (im.n_llm_h, im.n_llm_w);
        let l = lh * lw;
        let dim = vw.dim;
        let uf = bf(l * D * 9)?;
        g.launch(k["vit_unfold3"], (l as u32, 1, 1), 256, 0, &[Arg::P(x.ptr), Arg::P(uf.ptr), Arg::I(D as i32), Arg::I(nh as i32), Arg::I(nw as i32), Arg::I(lw as i32)])?;
        let wa = g.upload(&vw.al_w1)?;
        let ba = g.upload(&vw.al_b1)?;
        let y1 = f4(l * dim)?;
        g.gemm_bf16(uf.ptr, wa.ptr, y1.ptr, l, D * 9, dim, dim)?;
        let a1 = bf(l * dim)?;
        g.launch(k["vit_gelu_bias"], ew(l * dim), 256, 0, &[Arg::P(y1.ptr), Arg::P(ba.ptr), Arg::P(a1.ptr), Arg::I(l as i32), Arg::I(dim as i32)])?;
        drop((wa, uf));
        let wb = g.upload(&vw.al_w2)?;
        let bb = g.upload(&vw.al_b2)?;
        g.gemm_bf16(a1.ptr, wb.ptr, y1.ptr, l, dim, dim, dim)?;
        let y2 = f4(l * dim)?;
        g.launch(k["vit_bias_add"], ew(l * dim), 256, 0, &[Arg::P(y2.ptr), Arg::P(y1.ptr), Arg::P(bb.ptr), Arg::I(l as i32), Arg::I(dim as i32), Arg::I(0)])?;
        let rows = g.download::<f32>(y2.ptr, l * dim)?;
        // ---- the span: delimiters + aligner rows in reading order (embeddings are bf16 in the model)
        let round = |v: f32| half::bf16::from_f32(v).to_f32();
        let mut out = vec![0f32; im.len() * dim];
        let mut next = 0;
        for (i, &t) in im.types.iter().enumerate() {
            let dst = &mut out[i * dim..(i + 1) * dim];
            match t {
                IMAGE_START => dst.copy_from_slice(&vw.start),
                IMAGE_END => dst.copy_from_slice(&vw.end),
                IMAGE_NEW_LINE => dst.copy_from_slice(&vw.newline),
                IMAGE => {
                    for (o, &s) in dst.iter_mut().zip(&rows[next * dim..(next + 1) * dim]) {
                        *o = round(s);
                    }
                    next += 1;
                }
                _ => unreachable!(),
            }
        }
        ensure!(next == l, "image span has {next} IMAGE slots for {l} aligner rows");
        Ok(out)
    }
}
