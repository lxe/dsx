//! Minimal mmap reader for sharded HF safetensors checkpoints (model.safetensors.index.json).

use anyhow::{Context, Result, bail};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct StTensor {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub file: usize,
    pub start: usize,
    pub end: usize,
}

pub struct SafeTensors {
    pub files: Vec<PathBuf>,
    maps: Vec<Option<Mmap>>,
    pub tensors: HashMap<String, StTensor>,
}

impl SafeTensors {
    /// Open only the shards that hold `wanted` tensors (all shards if `wanted` is empty).
    /// Only the tensors whose names start with `prefix` (and only their shards need to be present).
    pub fn open_prefix(dir: &Path, prefix: &str) -> Result<SafeTensors> {
        let idx: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("model.safetensors.index.json")).context("index json")?)?;
        let names: Vec<String> = idx["weight_map"].as_object().context("weight_map")?.keys().filter(|k| k.starts_with(prefix)).cloned().collect();
        anyhow::ensure!(!names.is_empty(), "no tensors named {prefix}* in {}", dir.display());
        let want: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
        Self::open(dir, &want)
    }

    pub fn open(dir: &Path, wanted: &[&str]) -> Result<SafeTensors> {
        let idx: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("model.safetensors.index.json")).context("index json")?)?;
        let wm = idx["weight_map"].as_object().context("weight_map")?;
        let mut files: Vec<String> = vec![];
        for (name, f) in wm {
            if wanted.is_empty() || wanted.contains(&name.as_str()) {
                let f = f.as_str().unwrap().to_string();
                if !files.contains(&f) {
                    files.push(f);
                }
            }
        }
        let mut st = SafeTensors { files: vec![], maps: vec![], tensors: HashMap::new() };
        for f in &files {
            st.add_file(&dir.join(f))?;
        }
        Ok(st)
    }

    /// One standalone .safetensors file.
    pub fn open_file(path: &Path) -> Result<SafeTensors> {
        let mut st = SafeTensors { files: vec![], maps: vec![], tensors: HashMap::new() };
        st.add_file(path)?;
        Ok(st)
    }

    fn add_file(&mut self, p: &Path) -> Result<()> {
        let fi = self.files.len();
        let file = File::open(p).with_context(|| format!("open {}", p.display()))?;
        let map = unsafe { Mmap::map(&file)? };
        let n = u64::from_le_bytes(map[..8].try_into()?) as usize;
        let hdr: serde_json::Value = serde_json::from_slice(&map[8..8 + n])?;
        for (name, v) in hdr.as_object().context("header")? {
            if name == "__metadata__" {
                continue;
            }
            let off = v["data_offsets"].as_array().context("offsets")?;
            let t = StTensor {
                dtype: v["dtype"].as_str().unwrap_or("").to_string(),
                shape: v["shape"].as_array().unwrap().iter().map(|d| d.as_u64().unwrap() as usize).collect(),
                file: fi,
                start: 8 + n + off[0].as_u64().unwrap() as usize,
                end: 8 + n + off[1].as_u64().unwrap() as usize,
            };
            if t.end > map.len() {
                bail!("{name} runs past the end of {}", p.display());
            }
            self.tensors.insert(name.clone(), t);
        }
        self.files.push(p.to_path_buf());
        self.maps.push(Some(map));
        Ok(())
    }

    /// madvise(MADV_RANDOM) on every mapping: no readahead (the Engram tables are read 264 bytes at a time)
    pub fn advise_random(&self) {
        for m in self.maps.iter().flatten() {
            let _ = m.advise(memmap2::Advice::Random);
        }
    }

    pub fn get(&self, name: &str) -> Result<&StTensor> {
        self.tensors.get(name).with_context(|| format!("safetensors: no {name}"))
    }

    pub fn bytes(&self, t: &StTensor) -> &[u8] {
        &self.maps[t.file].as_ref().unwrap()[t.start..t.end]
    }
}
