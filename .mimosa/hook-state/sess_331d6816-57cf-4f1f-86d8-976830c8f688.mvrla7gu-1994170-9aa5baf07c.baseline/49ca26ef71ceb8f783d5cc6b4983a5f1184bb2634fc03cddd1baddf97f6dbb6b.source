//! DSpark, the speculative draft stack inside the checkpoint (`mtp.*`): CPU reference, model.py
//! DSparkBlock / DSparkAttention / forward_spec. Three window-only blocks with their own 128-expert MoE (top-3)
//! read the main model's attention input at layers 37-39 (`main_hidden`) and draft `block` tokens at once
//! (bidirectional inside the block), then a Markov head and a confidence head finish them one by one.

use crate::model::{Expert, Mat, Model, dot, hc_mixes_pub as hc_mixes, hc_post_pub as hc_post, hc_pre_pub as hc_pre, matvec};
use crate::numerics::*;
use crate::safetensors::SafeTensors;
use anyhow::Result;

pub struct DLayer {
    attn_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    q_a: Mat,
    q_a_norm: Vec<f32>,
    q_b: Mat,
    kv: Mat,
    kv_norm: Vec<f32>,
    o_a: Mat,
    o_b: Mat,
    sinks: Vec<f32>,
    hc_attn: (Vec<f32>, Vec<f32>, Vec<f32>), // fn, scale, base
    hc_ffn: (Vec<f32>, Vec<f32>, Vec<f32>),
    gate_w: Vec<f32>,
    gate_bias: Vec<f32>,
    experts: Vec<Expert>,
    sh: (Mat, Mat, Mat),
}

pub struct Dspark {
    pub layers: Vec<DLayer>,
    main_proj: Mat,
    main_norm: Vec<f32>,
    norm: Vec<f32>,
    markov_embed: Mat,
    markov_head: Mat,
    conf: Vec<f32>,
    pub block: usize,
    pub noise: u32,
    pub targets: Vec<usize>,
    top_k: usize,
}

pub struct DsparkState {
    window: Vec<Vec<Vec<f32>>>, // [layer][slot][head_dim]: main-model KV seen by each draft layer
}

fn lin(w: &Mat, x: &[f32]) -> Vec<f32> {
    let mut y = vec![0f32; w.rows];
    matvec(w, x, &mut y);
    y
}

impl Dspark {
    pub fn load(m: &Model, st: &SafeTensors, dir: &std::path::Path) -> Result<Dspark> {
        let cfg: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
        let t = &cfg["text_config"];
        let nl = t["num_nextn_predict_layers"].as_u64().unwrap_or(3) as usize;
        let ne = t["dspark_n_routed_experts"].as_u64().unwrap_or(128) as usize;
        let mat = |n: &str| crate::model::hf_mat_pub(st, n);
        let v = |n: &str| -> Result<Vec<f32>> { Ok(crate::model::mat_f32_pub(&mat(n)?)) };
        let mut layers = vec![];
        for l in 0..nl {
            let p = |s: &str| format!("mtp.{l}.{s}");
            let mut o_a = mat(&p("attn.wo_a.weight"))?;
            o_a.act_fp8 = false;
            layers.push(DLayer {
                attn_norm: v(&p("attn_norm.weight"))?,
                ffn_norm: v(&p("ffn_norm.weight"))?,
                q_a: mat(&p("attn.wq_a.weight"))?,
                q_a_norm: v(&p("attn.q_norm.weight"))?,
                q_b: mat(&p("attn.wq_b.weight"))?,
                kv: mat(&p("attn.wkv.weight"))?,
                kv_norm: v(&p("attn.kv_norm.weight"))?,
                o_a,
                o_b: mat(&p("attn.wo_b.weight"))?,
                sinks: v(&p("attn.attn_sink"))?,
                hc_attn: (v(&p("hc_attn_fn"))?, v(&p("hc_attn_scale"))?, v(&p("hc_attn_base"))?),
                hc_ffn: (v(&p("hc_ffn_fn"))?, v(&p("hc_ffn_scale"))?, v(&p("hc_ffn_base"))?),
                gate_w: v(&p("ffn.gate.weight"))?,
                gate_bias: v(&p("ffn.gate.bias"))?,
                experts: (0..ne)
                    .map(|e| {
                        Ok(Expert {
                            gate: mat(&p(&format!("ffn.experts.{e}.w1.weight")))?,
                            up: mat(&p(&format!("ffn.experts.{e}.w3.weight")))?,
                            down: mat(&p(&format!("ffn.experts.{e}.w2.weight")))?,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?,
                sh: (mat(&p("ffn.shared_experts.w1.weight"))?, mat(&p("ffn.shared_experts.w3.weight"))?, mat(&p("ffn.shared_experts.w2.weight"))?),
            });
        }
        let last = nl - 1;
        Ok(Dspark {
            layers,
            main_proj: mat("mtp.0.main_proj.weight")?,
            main_norm: v("mtp.0.main_norm.weight")?,
            norm: v(&format!("mtp.{last}.norm.weight"))?,
            markov_embed: mat(&format!("mtp.{last}.markov_head.embed.weight"))?,
            markov_head: mat(&format!("mtp.{last}.markov_head.head.weight"))?,
            conf: v(&format!("mtp.{last}.confidence_head.proj.weight"))?,
            block: t["dspark_block_size"].as_u64().unwrap_or(5) as usize,
            noise: t["dspark_noise_token_id"].as_u64().unwrap_or(128799) as u32,
            targets: t["dspark_target_layer_ids"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect(),
            top_k: t["dspark_num_experts_per_tok"].as_u64().unwrap_or(3) as usize,
        })
        .map(|d| {
            let _ = m;
            d
        })
    }

    pub fn new_state(&self, m: &Model) -> DsparkState {
        DsparkState { window: vec![vec![vec![0f32; m.cfg.head_dim]; m.cfg.window]; self.layers.len()] }
    }

    fn main_x(&self, m: &Model, main_hidden: &[f32]) -> Vec<f32> {
        let mut x = lin(&self.main_proj, main_hidden);
        rmsnorm_inplace(&mut x, &self.main_norm, m.cfg.eps);
        round_bf16_slice(&mut x);
        x
    }

    /// The main model just processed position `pos`; record its KV in every draft layer's window
    /// (DSparkAttention's main_kv at start_pos, also what the prefill branch seeds).
    pub fn observe(&self, m: &Model, st: &mut DsparkState, main_hidden: &[f32], pos: usize) {
        let c = &m.cfg;
        let mx = self.main_x(m, main_hidden);
        for (li, l) in self.layers.iter().enumerate() {
            let mut kv = lin(&l.kv, &mx);
            rmsnorm_inplace(&mut kv, &l.kv_norm, c.eps);
            m.ropes.0.apply(&mut kv[c.head_dim - c.rope_dim..], pos, false);
            fake_fp8(&mut kv, 32);
            st.window[li][pos % c.window] = kv;
        }
    }

    /// Draft `block` tokens after position `pos`, whose next token (sampled by the main model) is `next`.
    /// Call `observe` for `pos` first. Returns (drafts, confidence) - drafts[i] is the guess for position pos+2+i.
    pub fn draft(&self, m: &Model, st: &DsparkState, pos: usize, next: u32) -> (Vec<u32>, Vec<f32>) {
        let c = &m.cfg;
        let (d, hc, hd, rd, nh, b) = (c.dim, c.hc, c.head_dim, c.rope_dim, c.n_heads, self.block);
        let mut ids = vec![self.noise; b];
        ids[0] = next;
        let mut h: Vec<Vec<f32>> = ids
            .iter()
            .map(|&t| {
                let mut e = vec![0f32; d];
                m.embed.dequant(t as usize, &mut e);
                (0..hc).flat_map(|_| e.iter().cloned()).collect()
            })
            .collect();
        let mut pre_mix: Vec<Vec<f32>> = (0..b).map(|_| { let mut p = vec![0f32; hc]; p[0] = 1.0; p }).collect();
        let nwin = c.window.min(pos + 1);
        for (li, l) in self.layers.iter().enumerate() {
            // attention half, all draft positions together
            let res = h.clone();
            let mut mixes_a = vec![];
            let mut xs = vec![];
            for i in 0..b {
                mixes_a.push(hc_mixes(c, &h[i], &l.hc_attn.0, &l.hc_attn.1, &l.hc_attn.2));
                let mut x = hc_pre(c, &h[i], &pre_mix[i]);
                rmsnorm_inplace(&mut x, &l.attn_norm, c.eps);
                round_bf16_slice(&mut x);
                xs.push(x);
            }
            let mut qs = vec![];
            let mut kvs = vec![];
            for (i, x) in xs.iter().enumerate() {
                let mut qr = lin(&l.q_a, x);
                rmsnorm_inplace(&mut qr, &l.q_a_norm, c.eps);
                round_bf16_slice(&mut qr);
                let mut q = lin(&l.q_b, &qr);
                for hh in 0..nh {
                    m.ropes.0.apply(&mut q[hh * hd + hd - rd..(hh + 1) * hd], pos + 1 + i, false);
                }
                let mut kv = lin(&l.kv, x);
                rmsnorm_inplace(&mut kv, &l.kv_norm, c.eps);
                m.ropes.0.apply(&mut kv[hd - rd..], pos + 1 + i, false);
                fake_fp8(&mut kv, 32);
                qs.push(q);
                kvs.push(kv);
            }
            let rows: Vec<&Vec<f32>> = st.window[li][..nwin].iter().chain(kvs.iter()).collect();
            let scale = 1.0 / (hd as f32).sqrt();
            let mut outs = vec![];
            for i in 0..b {
                let mut o = vec![0f32; nh * hd];
                for hh in 0..nh {
                    let qh = &qs[i][hh * hd..(hh + 1) * hd];
                    let sc: Vec<f32> = rows.iter().map(|r| dot(qh, r) * scale).collect();
                    let mx = sc.iter().cloned().fold(-1e30f32, f32::max);
                    let ex: Vec<f32> = sc.iter().map(|s| (s - mx).exp()).collect();
                    let den: f32 = ex.iter().sum::<f32>() + (l.sinks[hh] - mx).exp();
                    let oh = &mut o[hh * hd..(hh + 1) * hd];
                    for (e, r) in ex.iter().zip(&rows) {
                        let p = e / den;
                        for j in 0..hd {
                            oh[j] += p * r[j];
                        }
                    }
                    round_bf16_slice(oh);
                    m.ropes.0.apply(&mut oh[hd - rd..], pos + 1 + i, true);
                }
                let gsz = nh * hd / c.o_groups;
                let mut r = vec![0f32; c.o_groups * c.o_lora];
                for gi in 0..c.o_groups {
                    let og = &mut o[gi * gsz..(gi + 1) * gsz];
                    fake_fp8(og, 32);
                    matvec(&l.o_a.sub(gi * c.o_lora, c.o_lora), og, &mut r[gi * c.o_lora..(gi + 1) * c.o_lora]);
                }
                round_bf16_slice(&mut r);
                outs.push(lin(&l.o_b, &r));
            }
            for i in 0..b {
                let (_, post, comb) = &mixes_a[i];
                h[i] = hc_post(c, &outs[i], &res[i], post, comb);
            }
            // ffn half, per token
            for i in 0..b {
                let res = h[i].clone();
                let (f_pre, f_post, f_comb) = hc_mixes(c, &h[i], &l.hc_ffn.0, &l.hc_ffn.1, &l.hc_ffn.2);
                let mut x = hc_pre(c, &h[i], &mixes_a[i].0);
                rmsnorm_inplace(&mut x, &l.ffn_norm, c.eps);
                round_bf16_slice(&mut x);
                let y = self.moe(m, l, &x);
                h[i] = hc_post(c, &y, &res, &f_post, &f_comb);
                pre_mix[i] = f_pre;
            }
            let _ = li;
        }
        // head: logits per position, Markov bias from the previous (drafted) token, greedy
        let mut out = vec![next];
        let mut conf = vec![];
        for i in 0..b {
            let mut x = hc_pre(c, &h[i], &pre_mix[i]);
            rmsnorm_inplace(&mut x, &self.norm, c.eps);
            let mut logits = lin(&m.head, &x);
            let mut emb = vec![0f32; self.markov_embed.cols];
            self.markov_embed.dequant(out[i] as usize, &mut emb);
            let bias = lin(&self.markov_head, &emb);
            for (lg, bv) in logits.iter_mut().zip(&bias) {
                *lg += bv;
            }
            let best = logits.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |a, (j, &v)| if v > a.1 { (j, v) } else { a }).0;
            out.push(best as u32);
            let mut cat = x.clone();
            cat.extend_from_slice(&emb);
            conf.push(dot(&self.conf, &cat));
        }
        (out[1..].to_vec(), conf)
    }

    fn moe(&self, m: &Model, l: &DLayer, x: &[f32]) -> Vec<f32> {
        let c = &m.cfg;
        let d = c.dim;
        let ne = l.experts.len();
        let scores: Vec<f32> = (0..ne).map(|e| softplus(dot(&l.gate_w[e * d..(e + 1) * d], x)).sqrt()).collect();
        let mut order: Vec<usize> = (0..ne).collect();
        order.sort_by(|&a, &b| (scores[b] + l.gate_bias[b]).partial_cmp(&(scores[a] + l.gate_bias[a])).unwrap());
        let sel = &order[..self.top_k];
        let wsum: f32 = sel.iter().map(|&e| scores[e]).sum::<f32>() + 1e-20;
        let mut y = vec![0f32; d];
        let mut run = |g: &Mat, u: &Mat, dn: &Mat, wt: f32| {
            let gv = lin(g, x);
            let uv = lin(u, x);
            let lim = c.swiglu_limit;
            let hm: Vec<f32> = gv.iter().zip(&uv).map(|(&g, &u)| round_bf16(silu(g.min(lim)) * u.clamp(-lim, lim) * wt)).collect();
            let o = lin(dn, &hm);
            for i in 0..d {
                y[i] += o[i];
            }
        };
        for &e in sel {
            let ex = &l.experts[e];
            run(&ex.gate, &ex.up, &ex.down, scores[e] / wsum * c.route_scale);
        }
        run(&l.sh.0, &l.sh.1, &l.sh.2, 1.0);
        round_bf16_slice(&mut y);
        y
    }
}
