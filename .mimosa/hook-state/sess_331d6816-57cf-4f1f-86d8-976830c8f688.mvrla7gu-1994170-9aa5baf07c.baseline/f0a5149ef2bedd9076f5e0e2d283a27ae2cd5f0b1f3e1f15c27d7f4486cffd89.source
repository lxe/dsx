//! dsx-server: OpenAI / Anthropic compatible HTTP API for the dsx engine.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Result, anyhow};
use clap::Parser;
use dsx::server::mock::{MockEngine, MockMode};
use dsx::server::{AppState, EngineHandle, ServerConfig};
use tokenizers::Tokenizer;

#[derive(Parser)]
#[command(about = "OpenAI / Anthropic compatible API server for DeepSeek-V4.1-Flash (dsx)")]
struct Cli {
    /// DeepSeek V4.1 tokenizer.json; default: the one stored in the model GGUF (`dsx.tokenizer_json`)
    #[arg(long)]
    tokenizer: Option<PathBuf>,
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    #[arg(long, default_value_t = 8090)]
    port: u16,
    /// model id reported by /v1/models
    #[arg(long, default_value = "deepseek-v4.1-flash")]
    model_name: String,
    /// context window (prompt + generated tokens)
    #[arg(long, default_value_t = 32768)]
    ctx: usize,
    /// generation cap when a request sets no max_tokens
    #[arg(long, default_value_t = 8192)]
    max_tokens: usize,
    /// default sampling when a request sets none
    #[arg(long, default_value_t = 1.0)]
    temperature: f32,
    #[arg(long, default_value_t = 1.0)]
    top_p: f32,
    /// logit penalty on hesitation tokens ("Wait", "But", "Alternatively", ...) while reasoning; 0 = off.
    /// A request's `hesitation_penalty` field overrides it.
    #[arg(long, default_value_t = 0.0)]
    hesitation_penalty: f32,
    /// add CPU expert outputs ranked R or lower (0 = best of 6) one layer late; changes model numerics.
    /// T16 decode eval measured KL ~0.208 without deferral vs ~0.237 at R=3; speed depends on workload.
    /// Off unless set here or through DSX_DEFER=R.
    #[arg(long)]
    defer_rank: Option<usize>,
    /// run the deterministic mock engine instead of the model
    #[arg(long)]
    mock: bool,
    /// mock: repeat the prompt tokens instead of the scripted replies
    #[arg(long)]
    mock_echo: bool,
    /// mock: sleep per generated token
    #[arg(long, default_value_t = 0)]
    mock_delay_ms: u64,
    /// require this key on every API request (`Authorization: Bearer <key>` or `x-api-key: <key>`);
    /// `openssl rand -hex 32` makes one. Health checks and the web UI's files stay open.
    #[arg(long, env = "DSX_API_KEY")]
    api_key: Option<String>,
    /// model: GGUF export or HF checkpoint dir
    #[arg(long)]
    model: Option<PathBuf>,
    #[arg(long)]
    engram: Option<PathBuf>,
    #[arg(long, default_value_t = 4096)]
    chunk: usize,
    /// GB of routed experts on the main GPU (picked by host-link bandwidth); negative = its free memory minus that
    #[arg(long = "main-gb", alias = "gpu0-gb", value_name = "GB", default_value_t = -1.5, allow_negative_numbers = true)]
    gpu0_gb: f64,
    /// GB of routed experts on each other GPU; negative = its free memory minus that; 0 = main GPU only
    #[arg(long = "aux-gb", alias = "gpu1-gb", value_name = "GB", default_value_t = -1.0, allow_negative_numbers = true)]
    gpu1_gb: f64,
    #[arg(long, default_value_t = 0.0)]
    arena_gb: f64,
    /// expert hotness ranking (rank.json from calibration)
    #[arg(long)]
    rank: Option<std::path::PathBuf>,
    /// vision weights (`vision.safetensors`, or the official checkpoint directory): enables image input.
    /// Default: `vision.safetensors` next to the model, when present.
    #[arg(long)]
    vision: Option<std::path::PathBuf>,
    /// text only: don't load the vision weights even if `vision.safetensors` sits next to the model
    #[arg(long)]
    no_vision: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .init();
    let mut cli = Cli::parse();
    if let Some(r) = cli.defer_rank {
        // read once by the engine (handoff::defer) before the model starts
        unsafe { std::env::set_var("DSX_DEFER", r.to_string()) };
    }
    if cli.no_vision {
        cli.vision = None;
    } else if cli.vision.is_none() {
        // vision.safetensors next to the model (the quant repo ships it beside the shards)
        let dir = cli.model.as_ref().and_then(|m| if m.is_dir() { Some(m.clone()) } else { m.parent().map(|p| p.to_path_buf()) });
        cli.vision = dir.map(|d| d.join("vision.safetensors")).filter(|p| p.is_file());
    }
    if let Some(v) = &cli.vision {
        tracing::info!("vision: {}", v.display());
    }
    let tok = Arc::new(match (&cli.tokenizer, &cli.model) {
        (Some(t), _) => Tokenizer::from_file(t).map_err(|e| anyhow!("{}: {e}", t.display()))?,
        (None, Some(m)) if m.is_file() => {
            let g = dsx::gguf::Gguf::open(m)?;
            let s = g.kv.get("dsx.tokenizer_json").and_then(|v| v.as_str()).ok_or_else(|| anyhow!("{} has no dsx.tokenizer_json: pass --tokenizer", m.display()))?;
            s.parse::<Tokenizer>().map_err(|e| anyhow!("dsx.tokenizer_json in {}: {e}", m.display()))?
        }
        _ => return Err(anyhow!("--tokenizer is required unless the model GGUF stores dsx.tokenizer_json")),
    });
    let engine = if cli.mock {
        let (tok, mode, delay) = (tok.clone(), if cli.mock_echo { MockMode::Echo } else { MockMode::Scripted }, cli.mock_delay_ms);
        EngineHandle::spawn(move || {
            let mut m = MockEngine::new(tok, mode);
            m.delay = Duration::from_millis(delay);
            Ok(Box::new(m) as Box<dyn dsx::server::Engine>)
        })?
    } else {
        real(&cli)?
    };
    let cfg = ServerConfig {
        model_name: cli.model_name.clone(),
        ctx: cli.ctx,
        default_max_tokens: cli.max_tokens,
        default_temperature: cli.temperature,
        default_top_p: cli.top_p,
        hesitation_penalty: cli.hesitation_penalty,
        vision: cli.vision.is_some() && !cli.mock,
        api_key: cli.api_key.clone(),
        ..ServerConfig::default()
    };
    let st = AppState::new(engine, tok, cfg)?;
    let listener = tokio::net::TcpListener::bind((cli.host.as_str(), cli.port)).await?;
    tracing::info!("listening on http://{} ({})", listener.local_addr()?, if cli.mock { "mock engine" } else { "dsx engine" });
    dsx::server::serve(listener, st).await?;
    Ok(())
}

#[cfg(feature = "cuda")]
fn real(cli: &Cli) -> Result<EngineHandle> {
    let args = dsx::server::real::RealEngineArgs {
        model: cli.model.clone().ok_or_else(|| anyhow!("--model is required (or --mock)"))?,
        engram: cli.engram.clone(),
        ctx: cli.ctx,
        chunk: cli.chunk,
        gpu0_gb: cli.gpu0_gb,
        gpu1_gb: cli.gpu1_gb,
        arena_gb: cli.arena_gb,
        rank: cli.rank.clone(),
        vision: cli.vision.clone(),
    };
    EngineHandle::spawn(move || dsx::server::real::real_engine(&args))
}

#[cfg(not(feature = "cuda"))]
fn real(_: &Cli) -> Result<EngineHandle> {
    anyhow::bail!("built without the cuda feature: only --mock is available")
}
