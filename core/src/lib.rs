//! dsx-core: what both runners share. DeepSeek-V4.1-Flash(-1017) config and weight handles, GGUF and safetensors
//! readers, block formats and numerics, the CPU reference forward (exact model.py semantics) and DSpark draft
//! reference, and the OpenAI / Anthropic compatible server that drives any `server::Engine`.

pub mod dspark;
pub mod formats;
pub mod gguf;
pub mod iqgrid;
pub mod model;
pub mod numerics;
mod page_cache;
pub mod safetensors;
pub mod server;
pub mod t16;
pub mod vision;
