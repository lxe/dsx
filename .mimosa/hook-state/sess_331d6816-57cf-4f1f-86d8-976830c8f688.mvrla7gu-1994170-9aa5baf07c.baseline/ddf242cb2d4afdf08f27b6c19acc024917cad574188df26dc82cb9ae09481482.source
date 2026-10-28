//! llama.cpp's web UI (tools/ui, MIT), built into webui/dist (webui/build.sh) and embedded in the binary, served at /.
//! It talks to the same OpenAI endpoints as any client, plus the llama.cpp extras answered here: /props, /slots,
//! and `timings` on streamed chat completions (mod.rs).

use std::sync::Arc;

use axum::extract::State;
use axum::http::{HeaderValue, StatusCode, Uri, header};
use axum::response::{IntoResponse, Response};
use rust_embed::RustEmbed;
use serde_json::json;

use super::AppState;

#[derive(RustEmbed)]
#[folder = "webui/dist"]
struct Assets;

/// Any GET the API routes don't claim: a UI file, or index.html (the UI routes by #hash, so this is rarely hit).
pub async fn asset(uri: Uri) -> Response {
    let path = uri.path().trim_start_matches('/');
    let path = if path.is_empty() { "index.html" } else { path };
    let (file, name) = match Assets::get(path) {
        Some(f) => (f, path),
        None if !path.contains('.') => match Assets::get("index.html") {
            Some(f) => (f, "index.html"),
            None => return StatusCode::NOT_FOUND.into_response(),
        },
        None => return StatusCode::NOT_FOUND.into_response(),
    };
    let mut res = file.data.into_owned().into_response();
    let h = res.headers_mut();
    if let Ok(v) = HeaderValue::from_str(file.metadata.mimetype()) {
        h.insert(header::CONTENT_TYPE, v);
    }
    // hashed bundle files never change; everything else is revalidated
    let cache = if name.starts_with("_app/immutable/") { "public, max-age=31536000, immutable" } else { "no-cache" };
    h.insert(header::CACHE_CONTROL, HeaderValue::from_static(cache));
    res
}

/// llama.cpp's /props, single-model mode: what the UI reads for the model name, context size and sampling defaults.
pub async fn props(State(st): State<Arc<AppState>>) -> Response {
    let c = &st.cfg;
    axum::Json(json!({
        "default_generation_settings": {
            "id": 0, "id_task": -1, "n_ctx": c.ctx, "speculative": false, "is_processing": st.engine.pending() > 0,
            "params": {
                "n_predict": -1, "seed": -1, "temperature": c.default_temperature, "top_k": 0, "top_p": c.default_top_p,
                "min_p": 0.0, "max_tokens": c.default_max_tokens, "stream": true, "stop": [], "samplers": ["top_p", "temperature"],
                "chat_format": "DeepSeek V4.1", "reasoning_format": "deepseek", "reasoning_in_content": false,
                "timings_per_token": false, "lora": []
            },
            "prompt": ""
        },
        "total_slots": 1,
        "model_path": c.model_name,
        "model_alias": c.model_name,
        "role": "model",
        "modalities": {"vision": c.vision, "audio": false, "video": false},
        "chat_template": "",
        "bos_token": "<｜begin▁of▁sentence｜>",
        "eos_token": "<｜end▁of▁sentence｜>",
        "build_info": concat!("dsx ", env!("CARGO_PKG_VERSION")),
        "ui_settings": {}
    }))
    .into_response()
}

/// llama.cpp's /slots: one slot, busy while a request runs.
pub async fn slots(State(st): State<Arc<AppState>>) -> Response {
    axum::Json(json!([{"id": 0, "n_ctx": st.cfg.ctx, "is_processing": st.engine.pending() > 0}])).into_response()
}

/// llama.cpp's built-in server tools are not offered: 403 is how llama-server says "started without --tools".
pub async fn no_tools() -> Response {
    (StatusCode::FORBIDDEN, axum::Json(json!({"error": {"message": "server tools are not enabled", "type": "forbidden"}}))).into_response()
}

/// llama.cpp's background-stream resume: nothing to resume here (streams end with their connection).
pub async fn no_streams() -> Response {
    axum::Json(json!([])).into_response()
}
