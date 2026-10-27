//! GGUF v3 reader over mmap, with llama.cpp split files (`-00001-of-0000N.gguf`) read as one model.

use anyhow::{Context, Result, bail};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    Arr(Vec<Value>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl Value {
    pub fn as_u64(&self) -> Option<u64> {
        Some(match *self {
            Value::U8(v) => v as u64,
            Value::U16(v) => v as u64,
            Value::U32(v) => v as u64,
            Value::U64(v) => v,
            Value::I8(v) if v >= 0 => v as u64,
            Value::I16(v) if v >= 0 => v as u64,
            Value::I32(v) if v >= 0 => v as u64,
            Value::I64(v) if v >= 0 => v as u64,
            _ => return None,
        })
    }
    pub fn as_f64(&self) -> Option<f64> {
        Some(match *self {
            Value::F32(v) => v as f64,
            Value::F64(v) => v,
            _ => return self.as_u64().map(|v| v as f64),
        })
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }
}

/// ggml type ids used by the V4.1 GSQ files (143 is llama-tq's PTQ1_0, not upstream ggml).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum GType {
    F32 = 0,
    F16 = 1,
    Q8_0 = 8,
    /// ggml Q2_K: 2-bit codes, 4-bit scales/minima per 16, two fp16 scales per 256
    Q2K = 10,
    Q3K = 11,
    /// ggml IQ3_XXS: 4-weight codebook entries, sign patterns and scale per 32
    Iq3Xxs = 18,
    /// ggml Q5_K: 5-bit codes, 6-bit scales and mins per 32 (shared experts in llama.cpp mixes)
    Q5K = 13,
    Q6K = 14,
    /// ggml IQ4_NL: 4-bit indices into a fixed non-linear table, fp16 scale per 32 (llama.cpp's Engram tables)
    Iq4Nl = 20,
    /// ggml IQ2_XXS: 8-weight groups from a 256-entry codebook with sign patterns, 4-bit scale per 32, fp16 per 256
    Iq2Xxs = 16,
    /// ggml IQ1_S: 8-weight ternary groups from a 2048-entry codebook, 3-bit scale and +-1/8 shift per 32, fp16 per 256
    Iq1S = 19,
    /// ggml IQ2_XS: 9-bit codebook index + 7-bit sign pattern per 8 weights, two 4-bit scales per 32, fp16 per 256
    Iq2Xs = 17,
    /// ggml IQ2_S: 10-bit codebook index and an 8-bit sign byte per 8 weights, two 4-bit scales per 32, fp16 per 256
    Iq2S = 22,
    I8 = 24,
    I32 = 26,
    BF16 = 30,
    Q2_0 = 42,
    Ptq1_0 = 143,
    /// HF checkpoint FP8 E4M3, E8M0 scale per 32x32 block (not a GGUF type)
    Fp8B32 = 1000,
    /// HF checkpoint FP4 E2M1, two per byte (low nibble first), E8M0 scale per 32 (not a GGUF type)
    Fp4 = 1001,
    /// T16 trellis expert matrix at 1.5 / 2 / 2.5 bits per weight: `trellis | suh | svh` per matrix (core/src/t16.rs);
    /// not a ggml type
    T16K15 = 1010,
    T16K2 = 1011,
    T16K25 = 1012,
}

impl GType {
    pub fn from_id(id: u32) -> Result<GType> {
        Ok(match id {
            0 => GType::F32,
            1 => GType::F16,
            8 => GType::Q8_0,
            10 => GType::Q2K,
            18 => GType::Iq3Xxs,
            11 => GType::Q3K,
            13 => GType::Q5K,
            14 => GType::Q6K,
            20 => GType::Iq4Nl,
            16 => GType::Iq2Xxs,
            19 => GType::Iq1S,
            17 => GType::Iq2Xs,
            22 => GType::Iq2S,
            24 => GType::I8,
            26 => GType::I32,
            30 => GType::BF16,
            42 => GType::Q2_0,
            143 => GType::Ptq1_0,
            1010 => GType::T16K15,
            1011 => GType::T16K2,
            1012 => GType::T16K25,
            _ => bail!("unsupported ggml type {id}"),
        })
    }
    /// (weights per block, bytes per block)
    pub fn block(self) -> (usize, usize) {
        match self {
            GType::F32 => (1, 4),
            GType::F16 | GType::BF16 => (1, 2),
            GType::I8 => (1, 1),
            GType::I32 => (1, 4),
            GType::Q8_0 => (32, 34),
            GType::Q2K => (256, 84),
            GType::Iq3Xxs => (256, 98),
            GType::Q3K => (256, 110),
            GType::Q5K => (256, 176),
            GType::Q6K => (256, 210),
            GType::Iq4Nl => (32, 18),
            GType::Iq2Xxs => (256, 66),
            GType::Iq1S => (256, 50),
            GType::Iq2Xs => (256, 74),
            GType::Iq2S => (256, 82),
            GType::Q2_0 => (64, 18),
            GType::Ptq1_0 => (128, 28),
            GType::Fp8B32 => (1, 1),
            GType::Fp4 => (2, 1),
            // per weight, not counting suh / svh: a T16 matrix is not a sequence of rows (see Mat::bytes)
            GType::T16K15 => (16, 3),
            GType::T16K2 => (4, 1),
            GType::T16K25 => (16, 5),
        }
    }
    pub fn row_bytes(self, n: usize) -> usize {
        let (w, b) = self.block();
        assert!(n % w == 0, "{self:?}: row of {n} is not a whole number of blocks");
        n / w * b
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    /// ggml order: dims[0] is the contiguous (row) dimension.
    pub dims: Vec<u64>,
    pub ty: GType,
    pub shard: usize,
    /// absolute byte offset in the shard file
    pub offset: u64,
    pub bytes: u64,
}

impl TensorInfo {
    pub fn n_elements(&self) -> u64 {
        self.dims.iter().product()
    }
    pub fn row_len(&self) -> usize {
        self.dims[0] as usize
    }
    pub fn n_rows(&self) -> usize {
        (self.n_elements() / self.dims[0]) as usize
    }
}

pub struct Gguf {
    pub kv: HashMap<String, Value>,
    pub tensors: HashMap<String, TensorInfo>,
    pub order: Vec<String>,
    pub paths: Vec<PathBuf>,
    pub maps: Vec<Mmap>,
}

struct Cursor<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.p + n > self.b.len() {
            bail!("gguf header truncated at {}", self.p);
        }
        let s = &self.b[self.p..self.p + n];
        self.p += n;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into()?))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into()?))
    }
    fn string(&mut self) -> Result<String> {
        let n = self.u64()? as usize;
        Ok(String::from_utf8_lossy(self.take(n)?).into_owned())
    }
    fn value(&mut self, ty: u32) -> Result<Value> {
        Ok(match ty {
            0 => Value::U8(self.take(1)?[0]),
            1 => Value::I8(self.take(1)?[0] as i8),
            2 => Value::U16(u16::from_le_bytes(self.take(2)?.try_into()?)),
            3 => Value::I16(i16::from_le_bytes(self.take(2)?.try_into()?)),
            4 => Value::U32(self.u32()?),
            5 => Value::I32(self.u32()? as i32),
            6 => Value::F32(f32::from_le_bytes(self.take(4)?.try_into()?)),
            7 => Value::Bool(self.take(1)?[0] != 0),
            8 => Value::Str(self.string()?),
            9 => {
                let et = self.u32()?;
                let n = self.u64()? as usize;
                let mut v = Vec::with_capacity(n.min(1 << 20));
                for _ in 0..n {
                    v.push(self.value(et)?);
                }
                Value::Arr(v)
            }
            10 => Value::U64(self.u64()?),
            11 => Value::I64(self.u64()? as i64),
            12 => Value::F64(f64::from_le_bytes(self.take(8)?.try_into()?)),
            _ => bail!("bad gguf value type {ty}"),
        })
    }
}

/// `a-00001-of-00003.gguf` -> all three paths; anything else -> itself.
pub fn split_paths(first: &Path) -> Vec<PathBuf> {
    let s = first.to_string_lossy();
    if let Some(i) = s.rfind("-00001-of-") {
        let rest = &s[i + 10..];
        if let Some(n) = rest.strip_suffix(".gguf").and_then(|n| n.parse::<usize>().ok()) {
            return (1..=n)
                .map(|k| PathBuf::from(format!("{}-{:05}-of-{}", &s[..i], k, rest)))
                .collect();
        }
    }
    vec![first.to_path_buf()]
}

impl Gguf {
    pub fn open(first: &Path) -> Result<Gguf> {
        let paths = split_paths(first);
        let mut g = Gguf { kv: HashMap::new(), tensors: HashMap::new(), order: vec![], paths: paths.clone(), maps: vec![] };
        for (shard, p) in paths.iter().enumerate() {
            let f = File::open(p).with_context(|| format!("open {}", p.display()))?;
            let map = unsafe { Mmap::map(&f)? };
            let mut c = Cursor { b: &map, p: 0 };
            if c.take(4)? != b"GGUF" {
                bail!("{} is not GGUF", p.display());
            }
            let ver = c.u32()?;
            if ver != 3 {
                bail!("gguf version {ver} (want 3)");
            }
            let nt = c.u64()? as usize;
            let nkv = c.u64()? as usize;
            for _ in 0..nkv {
                let k = c.string()?;
                let t = c.u32()?;
                let v = c.value(t)?;
                g.kv.entry(k).or_insert(v);
            }
            let align = g.kv.get("general.alignment").and_then(|v| v.as_u64()).unwrap_or(32);
            let mut infos = Vec::with_capacity(nt);
            for _ in 0..nt {
                let name = c.string()?;
                let nd = c.u32()? as usize;
                let dims = (0..nd).map(|_| c.u64()).collect::<Result<Vec<_>>>()?;
                let ty = GType::from_id(c.u32()?).with_context(|| format!("tensor {name}"))?;
                let off = c.u64()?;
                infos.push((name, dims, ty, off));
            }
            let data = (c.p as u64).div_ceil(align) * align;
            for (name, dims, ty, off) in infos {
                let n: u64 = dims.iter().product();
                let bytes = match crate::t16::k2(ty) {
                    // T16: [k, n, experts...], each k x n matrix stored as trellis | suh | svh
                    Some(_) => crate::t16::matrix_bytes(ty, dims[0] as usize, dims[1] as usize) as u64 * (n / dims[0] / dims[1]),
                    None => (ty.row_bytes(dims[0] as usize) as u64) * (n / dims[0]),
                };
                let ti = TensorInfo { name: name.clone(), dims, ty, shard, offset: data + off, bytes };
                if ti.offset + ti.bytes > map.len() as u64 {
                    bail!("tensor {name} runs past the end of {}", p.display());
                }
                g.order.push(name.clone());
                g.tensors.insert(name, ti);
            }
            g.maps.push(map);
        }
        Ok(g)
    }

    pub fn get(&self, name: &str) -> Result<&TensorInfo> {
        self.tensors.get(name).with_context(|| format!("tensor {name} not in the model"))
    }

    /// madvise(MADV_RANDOM) on one tensor's pages: no readahead, for tables read a row at a time (the Engram table;
    /// with readahead each 210-byte Q6_K row fault read ~3x longer than an FP8 row from the safetensors shards)
    pub fn advise_random(&self, t: &TensorInfo) {
        let page = 4096u64;
        let start = t.offset / page * page;
        let _ = self.maps[t.shard].advise_range(memmap2::Advice::Random, start as usize, (t.offset + t.bytes - start) as usize);
    }

    pub fn bytes(&self, t: &TensorInfo) -> &[u8] {
        &self.maps[t.shard][t.offset as usize..(t.offset + t.bytes) as usize]
    }

    pub fn u64(&self, key: &str) -> Result<u64> {
        self.kv.get(key).and_then(|v| v.as_u64()).with_context(|| format!("metadata {key}"))
    }
    pub fn f64(&self, key: &str) -> Result<f64> {
        self.kv.get(key).and_then(|v| v.as_f64()).with_context(|| format!("metadata {key}"))
    }
    pub fn arr_f64(&self, key: &str) -> Result<Vec<f64>> {
        match self.kv.get(key) {
            Some(Value::Arr(v)) => v.iter().map(|x| x.as_f64().context("non-numeric array item")).collect(),
            _ => bail!("metadata array {key}"),
        }
    }
    pub fn arr_u64(&self, key: &str) -> Result<Vec<u64>> {
        match self.kv.get(key) {
            Some(Value::Arr(v)) => v.iter().map(|x| x.as_u64().context("non-integer array item")).collect(),
            _ => bail!("metadata array {key}"),
        }
    }
}

/// Write `src` as `n` llama.cpp-style split files `<prefix>-0000k-of-0000n.gguf`, tensors in file order, balanced by
/// bytes. Shard 1 carries all metadata; every shard carries `split.no`, `split.count`, `split.tensors.count`.
/// `name` replaces `general.name`; `u32_arrays` adds (or replaces) u32 array keys (e.g. `dsx.expert_rot`).
/// `str_kvs` adds (or replaces) string keys (e.g. `dsx.tokenizer_json`).
/// `add`: another GGUF whose tensors (not metadata) go into one more shard at the end, e.g. an Engram table that is a
/// single tensor larger than any balanced shard; `n` counts that shard too.
pub fn split(src: &Path, prefix: &str, n: usize, name: Option<&str>, u32_arrays: &[(String, Vec<u32>)], str_kvs: &[(String, String)], add: Option<&Path>) -> Result<Vec<PathBuf>> {
    use std::io::{BufWriter, Write};
    let g = Gguf::open(src)?;
    anyhow::ensure!(g.paths.len() == 1, "{} is already split", src.display());
    let xg = add.map(Gguf::open).transpose()?;
    if let Some(x) = &xg {
        anyhow::ensure!(n >= 2, "--add needs at least 2 shards");
        anyhow::ensure!(x.order.iter().all(|k| !g.tensors.contains_key(k)), "tensor names of the added file collide");
    }
    let n_main = n - xg.is_some() as usize;
    let n_tensors = g.order.len() + xg.as_ref().map_or(0, |x| x.order.len());
    let map = &g.maps[0];
    // the metadata section verbatim: after magic, version, tensor count, kv count
    let mut c = Cursor { b: map, p: 24 };
    let mut nkv = u64::from_le_bytes(map[16..24].try_into()?) as usize;
    let mut kv_raw: Vec<u8> = vec![];
    for _ in 0..nkv {
        let p0 = c.p;
        let k = c.string()?;
        let t = c.u32()?;
        c.value(t)?;
        let replaced = u32_arrays.iter().any(|(n, _)| *n == k) || str_kvs.iter().any(|(n, _)| *n == k);
        if !(name.is_some() && k == "general.name") && !replaced {
            kv_raw.extend(&map[p0..c.p]);
        } else if replaced {
            nkv -= 1;
        }
    }
    for (key, vals) in u32_arrays {
        kv_raw.extend((key.len() as u64).to_le_bytes());
        kv_raw.extend(key.as_bytes());
        kv_raw.extend(9u32.to_le_bytes()); // array
        kv_raw.extend(4u32.to_le_bytes()); // of u32
        kv_raw.extend((vals.len() as u64).to_le_bytes());
        for v in vals {
            kv_raw.extend(v.to_le_bytes());
        }
        nkv += 1;
    }
    for (key, val) in str_kvs {
        kv_raw.extend((key.len() as u64).to_le_bytes());
        kv_raw.extend(key.as_bytes());
        kv_raw.extend(8u32.to_le_bytes()); // string
        kv_raw.extend((val.len() as u64).to_le_bytes());
        kv_raw.extend(val.as_bytes());
        nkv += 1;
    }
    if let Some(nm) = name {
        nkv = nkv - g.kv.contains_key("general.name") as usize + 1;
        for x in [&b"general.name"[..], nm.as_bytes()].iter().enumerate() {
            kv_raw.extend((x.1.len() as u64).to_le_bytes());
            kv_raw.extend(*x.1);
            if x.0 == 0 {
                kv_raw.extend(8u32.to_le_bytes());
            }
        }
    }
    let align = g.kv.get("general.alignment").and_then(|v| v.as_u64()).unwrap_or(32);
    let total: u64 = g.order.iter().map(|k| g.tensors[k].bytes).sum();
    // cut points: shard s ends at the first tensor boundary past s/n of the bytes
    let mut groups: Vec<Vec<(&Gguf, &TensorInfo)>> = vec![vec![]; n];
    let mut acc = 0u64;
    for k in &g.order {
        let t = &g.tensors[k];
        let s = ((acc as u128 * n_main as u128 / total as u128) as usize).min(n_main - 1);
        groups[s].push((&g, t));
        acc += t.bytes;
    }
    if let Some(x) = &xg {
        groups[n - 1] = x.order.iter().map(|k| (x, &x.tensors[k])).collect();
    }
    anyhow::ensure!(groups.iter().all(|v| !v.is_empty()), "too many shards for {} tensors", n_tensors);
    let s64 = |w: &mut Vec<u8>, s: &str| {
        w.extend((s.len() as u64).to_le_bytes());
        w.extend(s.as_bytes());
    };
    let mut out = vec![];
    for (si, ts) in groups.iter().enumerate() {
        let mut h: Vec<u8> = b"GGUF".to_vec();
        h.extend(3u32.to_le_bytes());
        h.extend((ts.len() as u64).to_le_bytes());
        let split_kv = 3u64;
        h.extend((if si == 0 { nkv as u64 + split_kv } else { split_kv + 1 }).to_le_bytes());
        if si == 0 {
            h.extend(&kv_raw);
        } else {
            s64(&mut h, "general.alignment");
            h.extend(4u32.to_le_bytes());
            h.extend((align as u32).to_le_bytes());
        }
        s64(&mut h, "split.no");
        h.extend(2u32.to_le_bytes());
        h.extend((si as u16).to_le_bytes());
        s64(&mut h, "split.count");
        h.extend(2u32.to_le_bytes());
        h.extend((n as u16).to_le_bytes());
        s64(&mut h, "split.tensors.count");
        h.extend(5u32.to_le_bytes());
        h.extend((n_tensors as i32).to_le_bytes());
        let mut off = 0u64;
        for (_, t) in ts {
            s64(&mut h, &t.name);
            h.extend((t.dims.len() as u32).to_le_bytes());
            for d in &t.dims {
                h.extend(d.to_le_bytes());
            }
            h.extend((t.ty as u32).to_le_bytes());
            h.extend(off.to_le_bytes());
            off = (off + t.bytes).div_ceil(align) * align;
        }
        h.resize((h.len() as u64).div_ceil(align) as usize * align as usize, 0);
        let path = PathBuf::from(format!("{prefix}-{:05}-of-{n:05}.gguf", si + 1));
        let mut w = BufWriter::with_capacity(64 << 20, File::create(&path).with_context(|| format!("create {}", path.display()))?);
        w.write_all(&h)?;
        let mut pos = 0u64;
        for (src, t) in ts {
            w.write_all(src.bytes(t))?;
            pos += t.bytes;
            let pad = pos.div_ceil(align) * align - pos;
            w.write_all(&vec![0u8; pad as usize])?;
            pos += pad;
        }
        w.into_inner().map_err(|e| anyhow::anyhow!("{e}"))?.sync_all()?;
        eprintln!("{}: {} tensors, {:.2} GB", path.display(), ts.len(), pos as f64 / 1e9);
        out.push(path);
    }
    Ok(out)
}

/// Write a split model (`<prefix>-00001-of-0000N.gguf`) as one GGUF: shard 1's metadata without the `split.*` keys,
/// then every tensor in file order.
pub fn merge(src: &Path, out: &Path) -> Result<()> {
    use std::io::{BufWriter, Write};
    let g = Gguf::open(src)?;
    let map = &g.maps[0];
    let mut c = Cursor { b: map, p: 24 };
    let nkv0 = u64::from_le_bytes(map[16..24].try_into()?) as usize;
    let (mut nkv, mut kv_raw) = (0u64, vec![]);
    for _ in 0..nkv0 {
        let p0 = c.p;
        let k = c.string()?;
        let t = c.u32()?;
        c.value(t)?;
        if !k.starts_with("split.") {
            kv_raw.extend(&map[p0..c.p]);
            nkv += 1;
        }
    }
    let align = g.kv.get("general.alignment").and_then(|v| v.as_u64()).unwrap_or(32);
    let mut h: Vec<u8> = b"GGUF".to_vec();
    h.extend(3u32.to_le_bytes());
    h.extend((g.order.len() as u64).to_le_bytes());
    h.extend(nkv.to_le_bytes());
    h.extend(&kv_raw);
    let mut off = 0u64;
    for k in &g.order {
        let t = &g.tensors[k];
        h.extend((t.name.len() as u64).to_le_bytes());
        h.extend(t.name.as_bytes());
        h.extend((t.dims.len() as u32).to_le_bytes());
        for d in &t.dims {
            h.extend(d.to_le_bytes());
        }
        h.extend((t.ty as u32).to_le_bytes());
        h.extend(off.to_le_bytes());
        off = (off + t.bytes).div_ceil(align) * align;
    }
    h.resize((h.len() as u64).div_ceil(align) as usize * align as usize, 0);
    let mut w = BufWriter::with_capacity(64 << 20, File::create(out).with_context(|| format!("create {}", out.display()))?);
    w.write_all(&h)?;
    let mut pos = 0u64;
    for k in &g.order {
        let t = &g.tensors[k];
        w.write_all(g.bytes(t))?;
        pos += t.bytes;
        let pad = pos.div_ceil(align) * align - pos;
        w.write_all(&vec![0u8; pad as usize])?;
        pos += pad;
    }
    w.into_inner().map_err(|e| anyhow::anyhow!("{e}"))?.sync_all()?;
    eprintln!("{}: {} tensors, {:.2} GB", out.display(), g.order.len(), pos as f64 / 1e9);
    Ok(())
}

#[cfg(test)]
mod split_tests {
    #[test]
    fn u32_array_round_trip() {
        let src = std::path::Path::new(concat!(env!("HOME"), "/.node-llama-cpp/models/hf_ggml-org_embeddinggemma-300M-Q8_0.gguf"));
        if !src.exists() {
            return;
        }
        let dir = std::env::temp_dir().join("dsx_split_test");
        std::fs::create_dir_all(&dir).unwrap();
        let prefix = dir.join("t").to_string_lossy().into_owned();
        let rot: Vec<u32> = (0..26).map(|l| if l % 3 == 0 { 128 } else { 0 }).collect();
        let p = super::split(src, &prefix, 2, None, &[("dsx.expert_rot".into(), rot.clone())], &[], None).unwrap();
        let g = super::Gguf::open(&p[0]).unwrap();
        assert_eq!(g.arr_u64("dsx.expert_rot").unwrap(), rot.iter().map(|&v| v as u64).collect::<Vec<_>>());
        for f in p {
            std::fs::remove_file(f).ok();
        }
    }
}
