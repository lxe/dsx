//! dsx: DeepSeek-V4.1-Flash(-1017) on one desktop (2x RTX 3090 or one RTX 5090, 128 GB RAM): dense half on the GPU,
//! routed experts spread over VRAM, a RAM arena and the file. The shared parts (config, readers, formats, CPU
//! reference, server) are in `dsx-core` and re-exported here under their old paths.

pub use dsx_core::{dspark, formats, gguf, iqgrid, model, numerics, safetensors, t16};

#[cfg(feature = "cuda")]
pub mod attn;
pub mod cpu;
pub mod engram_gather;
#[cfg(feature = "cuda")]
pub mod experts;
#[cfg(feature = "cuda")]
pub mod gpu;
#[cfg(feature = "cuda")]
pub mod gpumodel;
#[cfg(feature = "cuda")]
pub mod handoff;
/// Bounded host cache arena for parked conversation lineages (no CUDA dependency of its own).
pub mod hostcache;
pub mod q6;
pub mod spec;
#[cfg(feature = "cuda")]
pub mod topk;

/// The server (from dsx-core) and the GPU engine behind it.
pub mod server {
    pub use dsx_core::server::*;
    pub mod prefix;
    /// pure cache policy: LRU/admission arithmetic, independent of the GPU
    pub mod session_policy;
    /// host cache manager for several conversation lineages (needs the GPU state capture in `gpumodel`)
    #[cfg(feature = "cuda")]
    pub mod session_cache;
    #[cfg(feature = "cuda")]
    pub mod real;
}
