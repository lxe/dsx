//! Weight sources: DeepSeek's HF checkpoint (safetensors) or a dsx-amd GGUF export of a build (the same tensor
//! names and dtypes, with the re-quantized experts as device tiles). Both look like `SafeTensors` to the loader.

use anyhow::{Context, Result};
use dsx_core::safetensors::{SafeTensors, StTensor};
use std::collections::HashMap;
use std::path::Path;

pub struct Ck {
    pub tensors: HashMap<String, StTensor>,
    /// codebook experts in an export: tensor name ("....w1.cb") -> (format, multiplier)
    pub cb: HashMap<String, (i32, f32)>,
    st: Option<SafeTensors>,
    gg: Option<dsx_core::gguf::Gguf>,
    /// the export's shards opened with O_DIRECT (`read` bypasses the page cache)
    direct: Vec<std::fs::File>,
}

impl Ck {
    /// A checkpoint directory's tensors starting with `prefix` (only their shards need to be downloaded)
    pub fn open_prefix(dir: &Path, prefix: &str) -> Result<Ck> {
        let st = SafeTensors::open_prefix(dir, prefix)?;
        Ok(Ck { tensors: st.tensors.clone(), cb: HashMap::new(), st: Some(st), gg: None, direct: vec![] })
    }

    pub fn open(path: &Path) -> Result<Ck> {
        if path.is_dir() {
            let st = SafeTensors::open(path, &[])?;
            return Ok(Ck { tensors: st.tensors.clone(), cb: HashMap::new(), st: Some(st), gg: None, direct: vec![] });
        }
        let g = dsx_core::gguf::Gguf::open(path)?;
        let idx: serde_json::Value = serde_json::from_str(g.kv.get("dsxa.index").and_then(|v| v.as_str()).context("dsxa.index (not a dsx-amd export)")?)?;
        let mut tensors = HashMap::new();
        let mut cb = HashMap::new();
        for (name, t) in &g.tensors {
            let e = &idx[name];
            let dtype = e["dtype"].as_str().context("dtype")?.to_string();
            let shape: Vec<usize> = e["shape"].as_array().context("shape")?.iter().map(|x| x.as_u64().unwrap() as usize).collect();
            if let Some(f) = e["fmt"].as_i64() {
                cb.insert(name.clone(), (f as i32, e["mul"].as_f64().unwrap_or(1.0) as f32));
            }
            tensors.insert(name.clone(), StTensor { dtype, shape, file: t.shard, start: t.offset as usize, end: (t.offset + t.bytes) as usize });
        }
        use std::os::unix::fs::OpenOptionsExt;
        // O_DIRECT (x86-64 Linux); without it `read` copies from the map
        let direct: Vec<std::fs::File> = g.paths.iter().map(|p| std::fs::OpenOptions::new().read(true).custom_flags(0o40000).open(p)).collect::<std::io::Result<_>>().unwrap_or_default();
        Ok(Ck { tensors, cb, st: None, gg: Some(g), direct })
    }

    pub fn is_export(&self) -> bool {
        self.gg.is_some()
    }

    pub fn get(&self, name: &str) -> Result<&StTensor> {
        self.tensors.get(name).with_context(|| format!("no tensor {name}"))
    }

    /// A tensor's bytes read from its file with O_DIRECT when the export's shards allow it: bulk loads (the experts)
    /// then skip the page cache, which can't hold them, and its lock contention when 20 threads fault in pages
    pub fn read(&self, t: &StTensor) -> Vec<u8> {
        use std::os::unix::fs::FileExt;
        let Some(f) = self.direct.get(t.file) else { return self.bytes(t).to_vec() };
        const A: usize = 4096;
        let (a0, a1) = (t.start & !(A - 1), (t.end + A - 1) & !(A - 1));
        let layout = std::alloc::Layout::from_size_align(a1 - a0, A).unwrap();
        let p = unsafe { std::alloc::alloc(layout) };
        let buf = unsafe { std::slice::from_raw_parts_mut(p, a1 - a0) };
        let mut got = 0;
        // the last read may come up short at the end of the file
        while a0 + got < t.end {
            match f.read_at(&mut buf[got..], (a0 + got) as u64) {
                Ok(0) | Err(_) => break,
                Ok(n) => got += n,
            }
        }
        let v = if a0 + got >= t.end { buf[t.start - a0..t.end - a0].to_vec() } else { self.bytes(t).to_vec() };
        unsafe { std::alloc::dealloc(p, layout) };
        v
    }

    pub fn bytes(&self, t: &StTensor) -> &[u8] {
        match (&self.st, &self.gg) {
            (Some(st), _) => st.bytes(t),
            (_, Some(g)) => &g.maps[t.file][t.start..t.end],
            _ => unreachable!(),
        }
    }
}
