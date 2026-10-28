//! OpenAI / Anthropic compatible HTTP API over one engine thread, built on DeepSeek's `deepseek-recipe` crates:
//! request -> Conversation -> V4.1 prompt -> token ids -> engine -> token ids -> StreamProcessor -> protocol events.
//! Text only: image inputs are rejected.

pub mod auth;
pub mod engine;
pub mod mock;
pub mod sample;
pub mod webui;

use std::pin::{Pin, pin};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::Router;
use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{HeaderValue, StatusCode, header};
use axum::middleware;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use deepseek_recipe::anthropic::MessagesRequest;
use deepseek_recipe::openai::{ChatCompletionRequest, ResponsesRequest};
use deepseek_recipe::request::{ConversationRequest, ConversionError, ConversionOptions, ProtocolRequest};
use deepseek_recipe::response::ProtocolResponse;
use deepseek_recipe::stream::{ChunkGenerator, InferenceChunk, InferenceFinishReason, PromptUsage, StreamProcessor};
use deepseek_recipe::util::append_delta::AppendDelta;
use deepseek_recipe_encoding::PromptEncoding;
use deepseek_recipe_core::multimodal::ImageSource;
use crate::vision::ImageInput;
use deepseek_recipe_encoding::v4::dsv41::DeepseekV41Encoding;
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::json;
use tokenizers::Tokenizer;
use tokio_stream::{Stream, StreamExt};

pub use engine::{BatchEngine, Engine, EngineHandle, Event, Finish, GenParams, GenStats, Hesitation};

/// Tokens that end generation (never forwarded to the parser).
pub const STOP_TOKENS: [&str; 2] = ["<｜end▁of▁sentence｜>", "<｜User｜>"];

#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// reported by /v1/models and used when a request names no model
    pub model_name: String,
    /// context window: prompt + generated tokens
    pub ctx: usize,
    /// generation cap when the request sets none
    pub default_max_tokens: usize,
    pub default_temperature: f32,
    pub default_top_p: f32,
    /// hesitation-token logit penalty while reasoning; 0 = off. Requests can set `hesitation_penalty`.
    pub hesitation_penalty: f32,
    /// the engine has the vision tower loaded: image inputs are accepted
    pub vision: bool,
    /// required from every API request, as `Authorization: Bearer <key>` or `x-api-key: <key>`; None = open
    pub api_key: Option<String>,
    /// SSE comment interval while nothing else is sent (long prefills)
    pub keepalive: Duration,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            model_name: "deepseek-v4.1-flash".into(),
            ctx: 32768,
            default_max_tokens: 8192,
            default_temperature: 1.0,
            default_top_p: 1.0,
            hesitation_penalty: 0.0,
            vision: false,
            api_key: None,
            keepalive: Duration::from_secs(10),
        }
    }
}

pub struct AppState {
    pub engine: EngineHandle,
    pub tok: Arc<Tokenizer>,
    pub cfg: ServerConfig,
    stop_ids: Vec<u32>,
    /// single-token spellings of HESITATION_WORDS, and the reasoning delimiters
    hes_ids: Arc<Vec<u32>>,
    think: Option<(u32, u32)>,
}

/// Words a quantized model over-uses when it second-guesses itself (Lotfi et al., arXiv 2606.00206). Each is
/// penalized in the spellings the tokenizer has as one token: as is, lowercase, with a leading space.
pub const HESITATION_WORDS: [&str; 16] = [
    "Wait", "But", "Alternatively", "Hmm", "However", "Actually", "Hold", "Perhaps", "Maybe", "Oh", "Oops", "Instead",
    "Again", "Recheck", "Double", "Verify",
];

impl AppState {
    pub fn new(engine: EngineHandle, tok: Arc<Tokenizer>, mut cfg: ServerConfig) -> anyhow::Result<Arc<Self>> {
        // an empty --api-key would be a lock whose key is the empty string; treat it as what it is, no lock
        if cfg.api_key.as_deref().is_some_and(|k| k.trim().is_empty()) {
            tracing::warn!("api key: empty, the API is unauthenticated");
            cfg.api_key = None;
        }
        match &cfg.api_key {
            Some(k) => {
                tracing::info!("api key: required on the API routes ({} chars)", k.len());
                if k.trim() != k {
                    tracing::warn!("api key: leading or trailing whitespace, it is compared verbatim");
                }
                if k.len() < 16 {
                    tracing::warn!("api key: {} chars is short; 32 random hex chars is better (openssl rand -hex 32)", k.len());
                }
            }
            None => tracing::info!("api key: none, the API is open to anyone who can reach the port"),
        }
        let stop_ids = STOP_TOKENS
            .iter()
            .map(|t| tok.token_to_id(t).ok_or_else(|| anyhow::anyhow!("tokenizer has no {t} token: not a DeepSeek V4.1 tokenizer?")))
            .collect::<anyhow::Result<_>>()?;
        let mut hes: Vec<u32> = vec![];
        for w in HESITATION_WORDS {
            let lw = w.to_lowercase();
            for v in [w.to_string(), lw.clone(), format!(" {w}"), format!(" {lw}")] {
                if let Some(id) = tok.token_to_id(&v).or_else(|| {
                    let e = tok.encode(v.as_str(), false).ok()?;
                    (e.get_ids().len() == 1).then(|| e.get_ids()[0])
                }) {
                    hes.push(id);
                }
            }
        }
        hes.sort_unstable();
        hes.dedup();
        let think = tok.token_to_id("<think>").zip(tok.token_to_id("</think>"));
        tracing::info!(tokens = hes.len(), "hesitation penalty: {} tokens, default {}", hes.len(), cfg.hesitation_penalty);
        Ok(Arc::new(Self { engine, tok, cfg, stop_ids, hes_ids: Arc::new(hes), think }))
    }
}

pub fn router(st: Arc<AppState>) -> Router {
    // everything that runs the model or reports on it: behind the key when one is set
    let api = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/messages", post(messages))
        .route("/v1/messages/count_tokens", post(count_tokens))
        .route("/v1/responses", post(responses))
        .route("/v1/models", get(models))
        .route("/props", get(webui::props))
        .route("/slots", get(webui::slots))
        .route("/tools", get(webui::no_tools).post(webui::no_tools))
        .route("/v1/streams/lookup", post(webui::no_streams))
        .route_layer(middleware::from_fn_with_state(st.clone(), auth::require_key));
    // open: health checks, and the UI shell and its assets, which load before the UI can ask for a key
    // (a browser cannot put a header on a <script src>)
    let open = Router::new()
        .route("/health", get(health))
        .route("/v1/health", get(health))
        .fallback(get(webui::asset));
    // request bodies up to 32 MB (base64 images; axum's default is 2 MB)
    api.merge(open).layer(axum::extract::DefaultBodyLimit::max(32 * 1024 * 1024)).with_state(st)
}

pub async fn serve(listener: tokio::net::TcpListener, st: Arc<AppState>) -> std::io::Result<()> {
    if st.cfg.api_key.is_none() {
        if let Some(a) = listener.local_addr().ok().filter(|a| !a.ip().is_loopback()) {
            tracing::warn!("{a} is not loopback and there is no api key: anyone who can reach this port can use the model");
        }
    }
    // ConnectInfo hands auth::require_key the peer address for its rejection log
    axum::serve(listener, router(st).into_make_service_with_connect_info::<std::net::SocketAddr>()).await
}

// ---- errors

#[derive(Clone, Copy, PartialEq)]
enum Proto {
    OpenAi,
    Anthropic,
}

/// Which protocol's error envelope a path expects: Anthropic under /v1/messages, OpenAI everywhere else.
fn proto_for(path: &str) -> Proto {
    if path.starts_with("/v1/messages") { Proto::Anthropic } else { Proto::OpenAi }
}

struct ApiError(Proto, ConversionError);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = StatusCode::from_u16(self.1.status_code()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        let body = match self.0 {
            Proto::OpenAi => serde_json::to_value(self.1.into_response()).unwrap_or_default(),
            Proto::Anthropic => {
                let kind = if status.is_client_error() { "invalid_request_error" } else { "api_error" };
                json!({"type": "error", "error": {"type": kind, "message": self.1.to_string()}})
            }
        };
        (status, axum::Json(body)).into_response()
    }
}

fn parse<T: DeserializeOwned>(body: &[u8]) -> Result<T, ConversionError> {
    serde_json::from_slice(body).map_err(|e| ConversionError::bad_request(format!("invalid JSON request: {e}")))
}

/// parse, filling in this server's model name when the request has none (llama.cpp's web UI omits it in
/// single-model mode; there is only one model here anyway)
fn parse_with_model<T: DeserializeOwned>(st: &AppState, body: &[u8]) -> Result<T, ConversionError> {
    let mut v: serde_json::Value = parse(body)?;
    if let Some(o) = v.as_object_mut() {
        if !o.get("model").is_some_and(|m| m.as_str().is_some_and(|s| !s.is_empty())) {
            o.insert("model".into(), json!(st.cfg.model_name));
        }
    }
    serde_json::from_value(v).map_err(|e| ConversionError::bad_request(format!("invalid JSON request: {e}")))
}

/// llama.cpp's thinking switches on a chat request, mapped to the DeepSeek fields when those are absent:
/// chat_template_kwargs.enable_thinking (llama.cpp's web UI) or a boolean "thinking" -> thinking.type, and
/// chat_template_kwargs.reasoning_effort -> reasoning_effort
fn llama_thinking_compat(body: &[u8]) -> Result<Vec<u8>, ConversionError> {
    let mut v: serde_json::Value = parse(body)?;
    let Some(o) = v.as_object_mut() else { return Ok(body.to_vec()) };
    let kwargs = o.get("chat_template_kwargs").cloned().unwrap_or_default();
    let on = match o.get("thinking") {
        Some(t) => t.as_bool(),
        None => kwargs["enable_thinking"].as_bool(),
    };
    if let Some(on) = on {
        o.insert("thinking".into(), json!({"type": if on { "enabled" } else { "disabled" }}));
    }
    if !o.contains_key("reasoning_effort") {
        if let Some(effort) = kwargs["reasoning_effort"].as_str() {
            o.insert("reasoning_effort".into(), json!(effort));
        }
    }
    Ok(serde_json::to_vec(&v).expect("a JSON value serializes"))
}

// ---- routes

async fn chat_completions(State(st): State<Arc<AppState>>, body: Bytes) -> Result<Response, ApiError> {
    let e = |err| ApiError(Proto::OpenAi, err);
    let hes = hesitation_override(&body);
    let body = llama_thinking_compat(&body).map_err(e)?;
    let req: ChatCompletionRequest = parse_with_model(&st, &body).map_err(e)?;
    let (seed, include_usage) = (req.seed, req.include_usage());
    let conv = req.convert(ConversionOptions::default()).map_err(e)?;
    let streaming = conv.stream;
    run::<ChatCompletionRequest>(st, conv, "chatcmpl-", seed, hes, |g| if streaming { g.with_include_usage(include_usage) } else { g })
        .await
        .map_err(e)
}

async fn messages(State(st): State<Arc<AppState>>, body: Bytes) -> Result<Response, ApiError> {
    let e = |err| ApiError(Proto::Anthropic, err);
    let hes = hesitation_override(&body);
    let req: MessagesRequest = parse_with_model(&st, &body).map_err(e)?;
    let conv = req.convert(ConversionOptions::default()).map_err(e)?;
    run::<MessagesRequest>(st, conv, "msg_", None, hes, |g| g).await.map_err(e)
}

async fn responses(State(st): State<Arc<AppState>>, body: Bytes) -> Result<Response, ApiError> {
    let e = |err| ApiError(Proto::OpenAi, err);
    let hes = hesitation_override(&body);
    let req: ResponsesRequest = parse_with_model(&st, &body).map_err(e)?;
    let custom = req.custom_tool_names();
    let conv = req.convert(ConversionOptions::default()).map_err(e)?;
    run::<ResponsesRequest>(st, conv, "resp_", None, hes, move |g| g.with_custom_tool_names(custom)).await.map_err(e)
}

/// Anthropic token counting: the exact prompt length this server would prefill.
async fn count_tokens(State(st): State<Arc<AppState>>, body: Bytes) -> Result<Response, ApiError> {
    let e = |err| ApiError(Proto::Anthropic, err);
    let req: MessagesRequest = parse_with_model(&st, &body).map_err(e)?;
    let conv = req.convert(ConversionOptions::default()).map_err(e)?;
    let (ids, _) = encode(&st, &conv).map_err(e)?;
    Ok(axum::Json(json!({"input_tokens": ids.len()})).into_response())
}

async fn models(State(st): State<Arc<AppState>>) -> Response {
    let created = now();
    axum::Json(json!({
        "object": "list",
        "data": [{"id": st.cfg.model_name, "object": "model", "created": created, "owned_by": "dsx",
                  "display_name": st.cfg.model_name, "type": "model", "context_length": st.cfg.ctx}],
    }))
    .into_response()
}

async fn health(State(st): State<Arc<AppState>>) -> Response {
    axum::Json(json!({"status": "ok", "model": st.cfg.model_name, "pending": st.engine.pending()})).into_response()
}

// ---- the shared request path

type Gen<T> = <<T as ProtocolRequest>::Response as ProtocolResponse>::ChunkGenerator;
type Chunk<T> = <Gen<T> as ChunkGenerator>::Chunk;

/// Per-request `hesitation_penalty` (a dsx extension field; the protocol parsers ignore unknown fields).
fn hesitation_override(body: &[u8]) -> Option<f32> {
    serde_json::from_slice::<serde_json::Value>(body).ok()?.get("hesitation_penalty")?.as_f64().map(|v| v as f32)
}

/// Render + tokenize the conversation (V4.1 template). Rejects images.
pub fn encode(st: &AppState, conv: &ConversationRequest) -> Result<(Vec<u32>, Vec<ImageInput>), ConversionError> {
    let rendered = DeepseekV41Encoding::new().render_conversation(&conv.conversation);
    if !rendered.image_sources.is_empty() && !st.cfg.vision {
        return Err(ConversionError::bad_request(format!(
            "this server was started without the vision weights (--vision); the request has {} image(s)",
            rendered.image_sources.len()
        )));
    }
    tracing::debug!(prompt = rendered.prompt, "rendered prompt");
    // the template writes `<｜image｜>`; tokenizer revisions spell the image token differently (1017:
    // `<｜deepseek_image｜>`), so use whatever this tokenizer has at the image token id
    let prompt = match st.tok.id_to_token(crate::vision::IMAGE_TOKEN_ID) {
        Some(t) if !rendered.image_sources.is_empty() => rendered.prompt.replace(deepseek_recipe_core::multimodal::IMAGE_SPECIAL_TOKEN, &t),
        _ => rendered.prompt.clone(),
    };
    let enc = st.tok.encode(prompt.as_str(), false).map_err(|e| ConversionError::internal(format!("tokenizer: {e}")))?;
    let bytes = rendered
        .image_sources
        .iter()
        .map(|s| match s {
            ImageSource::DataUrl { data_url, .. } => crate::vision::data_url_bytes(data_url).map_err(|e| ConversionError::bad_request(e.to_string())),
            ImageSource::Bytes { data, .. } => Ok(data.clone()),
            ImageSource::Url { .. } => Err(ConversionError::bad_request("remote image URLs are not fetched by this server; send the image as a data URL (base64)")),
        })
        .collect::<Result<Vec<_>, _>>()?;
    crate::vision::expand(enc.get_ids(), bytes).map_err(|e| ConversionError::bad_request(e.to_string()))
}

async fn run<T>(
    st: Arc<AppState>,
    conv: ConversationRequest,
    id_prefix: &str,
    seed: Option<u64>,
    hesitation: Option<f32>,
    configure: impl FnOnce(Gen<T>) -> Gen<T>,
) -> Result<Response, ConversionError>
where
    T: ProtocolRequest,
    Chunk<T>: Serialize + Send,
{
    let (prompt, images) = encode(&st, &conv)?;
    let cfg = &st.cfg;
    let lambda = hesitation.unwrap_or(cfg.hesitation_penalty);
    let hesitation = match st.think {
        Some((open, close)) if lambda > 0.0 && prompt.last() == Some(&open) => Some(Hesitation { lambda, ids: st.hes_ids.clone(), close }),
        _ => None,
    };
    if prompt.len() >= cfg.ctx {
        return Err(ConversionError::bad_request(format!(
            "prompt is {} tokens; this server's context window is {} tokens",
            prompt.len(),
            cfg.ctx
        )));
    }
    let io = &conv.inference_options;
    let room = cfg.ctx - prompt.len();
    let params = GenParams {
        max_new: io.max_tokens.map(|m| m as usize).unwrap_or(cfg.default_max_tokens).min(room),
        temperature: io.temperature.unwrap_or(cfg.default_temperature),
        top_p: io.top_p.unwrap_or(cfg.default_top_p),
        seed,
        stop_token_ids: st.stop_ids.clone(),
        hesitation,
        images: Arc::new(images),
    };
    let id = engine::request_id(id_prefix);
    let model = conv.model.clone().filter(|m| !m.is_empty()).unwrap_or_else(|| cfg.model_name.clone());
    let c = &conv.conversation;
    tracing::info!(
        %id, %model, stream = conv.stream, thinking = c.thinking_mode, effort = ?c.reasoning_effort,
        messages = c.messages.len(), tools = c.tools.len(), temperature = params.temperature, top_p = params.top_p, hesitation = params.hesitation.as_ref().map_or(0.0, |h| h.lambda),
        "request"
    );
    let generator = configure(T::chunk_generator(&conv, id.clone(), model.clone()));
    let processor = StreamProcessor::new(generator, conv.parsing_options).with_tokenizer(st.tok.clone());
    let n_prompt = prompt.len();
    let rx = st.engine.submit(id.clone(), prompt, params).map_err(|e| ConversionError::internal(e.to_string()))?;
    let failed = Arc::new(Mutex::new(None::<String>));
    let timing = Arc::new(Mutex::new(Timing::new()));
    let inference = inference_stream(rx, n_prompt, failed.clone(), timing.clone());

    if !conv.stream {
        let mut resp = T::Response::new(id, model, now(), n_prompt, 0);
        let mut chunks = pin!(processor.process(inference));
        while let Some(chunk) = chunks.next().await {
            resp.append(chunk.map_err(|e| ConversionError::internal(e.to_string()))?);
        }
        if let Some(err) = failed.lock().unwrap().take() {
            return Err(ConversionError::internal(format!("engine error: {err}")));
        }
        return Ok(axum::Json(resp).into_response());
    }

    let events = processor.process(inference).map(|chunk| {
        let chunk = chunk.map_err(|e| std::io::Error::other(e.to_string()))?;
        let data = serde_json::to_string(&chunk).map_err(std::io::Error::other)?;
        Ok::<_, std::io::Error>(match <T::Response as ProtocolResponse>::chunk_event_type(&chunk) {
            Some(ev) => format!("event: {ev}\ndata: {data}\n\n"),
            None => format!("data: {data}\n\n"),
        })
    });
    // llama.cpp extension for chat completions: a last chunk with `timings` (its web UI shows tok/s from it)
    let llama_timings = id_prefix == "chatcmpl-";
    let (tid, tmodel) = (id.clone(), model.clone());
    let timings = tokio_stream::iter([()]).filter_map(move |_| {
        llama_timings.then(|| {
            let t = timing.lock().unwrap().json(n_prompt);
            Ok(format!("data: {}\n\n", json!({"id": tid, "object": "chat.completion.chunk", "model": tmodel, "choices": [], "timings": t})))
        })
    });
    let done = <T::Response as ProtocolResponse>::done_message().map(|m| Ok(format!("data: {m}\n\n")));
    let events = events.chain(timings).chain(tokio_stream::iter(done));
    // SSE comments keep idle connections (long prefill, slow decode) alive; parsers ignore them
    let events = events.timeout(cfg.keepalive).map(|r| r.unwrap_or_else(|_| Ok(": keep-alive\n\n".to_string())));
    let mut res = Response::new(Body::from_stream(events));
    let h = res.headers_mut();
    h.insert(header::CONTENT_TYPE, HeaderValue::from_static("text/event-stream; charset=utf-8"));
    h.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    Ok(res)
}

/// Engine events -> recipe inference chunks. Dropping the stream drops the receiver, which cancels generation.
fn inference_stream(
    rx: tokio::sync::mpsc::UnboundedReceiver<Event>,
    n_prompt: usize,
    failed: Arc<Mutex<Option<String>>>,
    timing: Arc<Mutex<Timing>>,
) -> Pin<Box<dyn Stream<Item = InferenceChunk> + Send>> {
    let ready = InferenceChunk::Ready {
        system_fingerprint: Some("dsx".into()),
        prompt_usage: PromptUsage { prompt_tokens: n_prompt, prompt_cache_hit_tokens: 0 },
    };
    let rest = tokio_stream::wrappers::UnboundedReceiverStream::new(rx).map_while(move |ev| match ev {
        Event::Token(t) => {
            timing.lock().unwrap().token();
            Some(InferenceChunk::Token { token_id: t })
        }
        Event::Done { finish: Finish::Stop } => Some(InferenceChunk::Finish { finish_reason: InferenceFinishReason::Stop }),
        Event::Done { finish: Finish::Length } => Some(InferenceChunk::Finish { finish_reason: InferenceFinishReason::Length }),
        Event::Done { finish: Finish::Cancelled } => None,
        Event::Error(e) => {
            *failed.lock().unwrap() = Some(e);
            None
        }
    });
    Box::pin(tokio_stream::iter([ready]).chain(rest))
}

/// Request timing as seen by the HTTP layer: prompt = submit -> first token, predicted = first -> last token.
struct Timing {
    start: std::time::Instant,
    first: Option<std::time::Instant>,
    last: Option<std::time::Instant>,
    n: usize,
}

impl Timing {
    fn new() -> Self {
        Self { start: std::time::Instant::now(), first: None, last: None, n: 0 }
    }
    fn token(&mut self) {
        let t = std::time::Instant::now();
        self.first.get_or_insert(t);
        self.last = Some(t);
        self.n += 1;
    }
    /// llama.cpp's `timings` object
    fn json(&self, n_prompt: usize) -> serde_json::Value {
        let ms = |a: std::time::Instant, b: std::time::Instant| (b - a).as_secs_f64() * 1e3;
        let prompt_ms = self.first.map_or(0.0, |f| ms(self.start, f));
        // the first token comes with the prefill; decode speed is over the tokens after it
        let predicted_ms = match (self.first, self.last) { (Some(f), Some(l)) => ms(f, l), _ => 0.0 };
        let dn = self.n.saturating_sub(1);
        json!({
            "prompt_n": n_prompt, "prompt_ms": prompt_ms,
            "prompt_per_second": if prompt_ms > 0.0 { n_prompt as f64 * 1e3 / prompt_ms } else { 0.0 },
            "predicted_n": self.n, "predicted_ms": predicted_ms,
            "predicted_per_second": if predicted_ms > 0.0 { dn as f64 * 1e3 / predicted_ms } else { 0.0 },
        })
    }
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}
