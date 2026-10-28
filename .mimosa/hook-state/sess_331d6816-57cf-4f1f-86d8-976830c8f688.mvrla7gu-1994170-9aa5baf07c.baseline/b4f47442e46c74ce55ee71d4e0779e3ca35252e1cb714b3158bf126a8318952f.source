//! Write a loaded build as one GGUF (then `dsx split` into shards): every tensor the runner reads, under the
//! checkpoint's names and dtypes; routed experts kept in MXFP4 are the checkpoint's bytes, the re-quantized ones are
//! their device tiles (`<name>.cb`, format and multiplier in the index). Engram tables, vision and the draft layers
//! are left out (the tables come from the official checkpoint).

use crate::decode::*;
use anyhow::Result;
use std::io::{BufWriter, Seek, Write};
use std::path::Path;

fn keep(name: &str, c: &dsx_core::model::Config) -> bool {
    if name.starts_with("vision.") || name.starts_with("aligner.") || name.starts_with("image_") || name.starts_with("mtp.") {
        return false;
    }
    !c.engram_layers.iter().any(|l| name.starts_with(&format!("layers.{l}.engram.embed.")))
}

impl Decoder {
    pub fn export(&self, dir: &Path, out: &Path) -> Result<()> {
        let st = crate::ck::Ck::open(dir)?;
        let c = &self.cfg;
        // tensor list: (name, dtype, shape, source)
        enum Src {
            Ck(String),
            Arena(i64, usize),
        }
        let mut items: Vec<(String, String, Vec<usize>, Src, Option<(i32, f32)>)> = vec![];
        let mut names: Vec<&String> = st.tensors.keys().filter(|n| keep(n, c)).collect();
        names.sort();
        let ex = regex_expert();
        for n in names {
            let t = st.get(n)?;
            if let Some((l, e, m)) = ex(n) {
                if l < self.layers.len() {
                    let em = self.etab[l][e * 3 + m];
                    if em.fmt != E_FP4 {
                        if n.ends_with(".weight") {
                            let (rows, k) = (t.shape[0], t.shape[1] * 2);
                            let bits = if em.fmt == E_CB3 { 3 } else { 2 };
                            let bytes = rows * k * bits / 8 + rows * k / 32;
                            let cbn = n.replace(".weight", ".cb");
                            let dt = if em.fmt == E_CB3 { "DSXA_CB3" } else { "DSXA_CB2" };
                            items.push((cbn, dt.into(), vec![bytes], Src::Arena(em.woff, bytes), Some((em.fmt, em.mul))));
                        }
                        continue; // the scale tensor goes with the tiles
                    }
                }
            }
            items.push((n.clone(), t.dtype.clone(), t.shape.clone(), Src::Ck(n.clone()), None));
        }
        let mut index = serde_json::Map::new();
        for (n, dt, sh, _, cb) in &items {
            let mut e = serde_json::json!({"dtype": dt, "shape": sh});
            if let Some((f, m)) = cb {
                e["fmt"] = serde_json::json!(f);
                e["mul"] = serde_json::json!(m);
            }
            index.insert(n.clone(), e);
        }
        let ggml_ty = |dt: &str| -> u32 {
            match dt {
                "BF16" => 30,
                "F32" => 0,
                "F16" => 1,
                _ => 24, // bytes: F8_E4M3, I8 (packed FP4), F8_E8M0, codebook tiles
            }
        };
        let elem = |dt: &str| -> usize {
            match dt {
                "BF16" | "F16" => 2,
                "F32" => 4,
                _ => 1,
            }
        };
        let f = std::fs::File::create(out)?;
        let mut w = BufWriter::with_capacity(64 << 20, f);
        let s64 = |w: &mut BufWriter<std::fs::File>, s: &str| -> Result<()> {
            w.write_all(&(s.len() as u64).to_le_bytes())?;
            w.write_all(s.as_bytes())?;
            Ok(())
        };
        w.write_all(b"GGUF")?;
        w.write_all(&3u32.to_le_bytes())?;
        w.write_all(&(items.len() as u64).to_le_bytes())?;
        w.write_all(&4u64.to_le_bytes())?;
        s64(&mut w, "general.architecture")?;
        w.write_all(&8u32.to_le_bytes())?;
        s64(&mut w, "dsx-amd")?;
        s64(&mut w, "general.name")?;
        w.write_all(&8u32.to_le_bytes())?;
        s64(&mut w, "DeepSeek-V4.1-Flash-1017-dsx-mxfp4")?;
        s64(&mut w, "general.alignment")?;
        w.write_all(&4u32.to_le_bytes())?;
        w.write_all(&32u32.to_le_bytes())?;
        s64(&mut w, "dsxa.index")?;
        w.write_all(&8u32.to_le_bytes())?;
        s64(&mut w, &serde_json::Value::Object(index).to_string())?;
        let mut off = 0u64;
        let mut offs = vec![];
        for (n, dt, sh, _, _) in &items {
            s64(&mut w, n)?;
            // ggml dims: contiguous first; byte tensors count bytes
            let mut dims: Vec<u64> = sh.iter().rev().map(|&d| d as u64).collect();
            if dims.is_empty() {
                dims.push(1);
            }
            w.write_all(&(dims.len() as u32).to_le_bytes())?;
            for d in &dims {
                w.write_all(&d.to_le_bytes())?;
            }
            w.write_all(&ggml_ty(dt).to_le_bytes())?;
            w.write_all(&off.to_le_bytes())?;
            offs.push(off);
            let bytes = sh.iter().product::<usize>() as u64 * elem(dt) as u64;
            off = (off + bytes).div_ceil(32) * 32;
        }
        let pos = w.stream_position()?;
        let pad = pos.div_ceil(32) * 32 - pos;
        w.write_all(&vec![0u8; pad as usize])?;
        let mut written = 0u64;
        let t0 = std::time::Instant::now();
        for (i, (_, dt, sh, src, _)) in items.iter().enumerate() {
            let bytes = sh.iter().product::<usize>() as u64 * elem(dt) as u64;
            let data: Vec<u8> = match src {
                Src::Ck(n) => st.bytes(st.get(n)?).to_vec(),
                Src::Arena(o, b) => self.g.dtoh(self.arena.at(*o as usize), *b)?,
            };
            anyhow::ensure!(data.len() as u64 == bytes, "{}: {} bytes, expected {bytes}", items[i].0, data.len());
            w.write_all(&data)?;
            let next = if i + 1 < items.len() { offs[i + 1] } else { (offs[i] + bytes).div_ceil(32) * 32 };
            w.write_all(&vec![0u8; (next - offs[i] - bytes) as usize])?;
            written += bytes;
            if i % 2000 == 0 {
                eprintln!("export: {i}/{} tensors, {:.1} GB, {:.0} s", items.len(), written as f64 / 1e9, t0.elapsed().as_secs_f32());
            }
        }
        w.flush()?;
        eprintln!("export: {} tensors, {:.1} GB -> {}", items.len(), written as f64 / 1e9, out.display());
        Ok(())
    }
}

/// layers.L.ffn.experts.E.wM.(weight|scale) -> (L, E, matrix 0 w1 / 1 w3 / 2 w2)
fn regex_expert() -> impl Fn(&str) -> Option<(usize, usize, usize)> {
    |n: &str| {
        let r = n.strip_prefix("layers.")?;
        let (l, r) = r.split_once('.')?;
        let r = r.strip_prefix("ffn.experts.")?;
        let (e, r) = r.split_once('.')?;
        let m = match r.split_once('.')?.0 {
            "w1" => 0,
            "w3" => 1,
            "w2" => 2,
            _ => return None,
        };
        Some((l.parse().ok()?, e.parse().ok()?, m))
    }
}
