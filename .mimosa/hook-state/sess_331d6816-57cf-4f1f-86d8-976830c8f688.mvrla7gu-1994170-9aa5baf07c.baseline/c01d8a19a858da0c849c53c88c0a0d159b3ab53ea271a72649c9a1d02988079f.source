//! HTTP API end to end against the mock engine: OpenAI chat, Anthropic messages, Responses, tool calls, limits.

use std::io::Read;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use deepseek_recipe::anthropic::MessagesRequest;
use deepseek_recipe::openai::ChatCompletionRequest;
use deepseek_recipe::request::{ConversionOptions, ProtocolRequest};
use deepseek_recipe_encoding::PromptEncoding;
use deepseek_recipe_encoding::v4::dsv41::DeepseekV41Encoding;
use dsx_core::server::mock::{MOCK_REASONING, MOCK_REPLY, MockEngine, MockMode};
use dsx_core::server::{AppState, Engine, EngineHandle, ServerConfig};
use serde_json::{Value, json};
use tokenizers::Tokenizer;

/// The bundled V4.1 tokenizer (MIT, DeepSeek; see tests/data/LICENSE-tokenizer).
fn tokenizer() -> Arc<Tokenizer> {
    static TOK: OnceLock<Arc<Tokenizer>> = OnceLock::new();
    TOK.get_or_init(|| {
        let gz = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/tokenizer-v41.json.gz")).unwrap();
        let mut json = Vec::new();
        flate2::read::GzDecoder::new(&gz[..]).read_to_end(&mut json).unwrap();
        Arc::new(Tokenizer::from_bytes(json).unwrap())
    })
    .clone()
}

async fn start(mode: MockMode, delay_ms: u64) -> String {
    start_cfg(mode, delay_ms, ServerConfig::default()).await
}

async fn start_cfg(mode: MockMode, delay_ms: u64, cfg: ServerConfig) -> String {
    let tok = tokenizer();
    let t2 = tok.clone();
    let engine = EngineHandle::spawn(move || {
        let mut m = MockEngine::new(t2, mode);
        m.delay = Duration::from_millis(delay_ms);
        Ok(Box::new(m) as Box<dyn Engine>)
    })
    .unwrap();
    let st = AppState::new(engine, tok, cfg).unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(dsx_core::server::serve(listener, st));
    base
}

async fn post(base: &str, path: &str, body: Value) -> (u16, String) {
    post_h(base, path, body, &[]).await
}

async fn post_h(base: &str, path: &str, body: Value, headers: &[(&str, &str)]) -> (u16, String) {
    let mut req = reqwest::Client::new().post(format!("{base}{path}")).json(&body);
    for (k, v) in headers {
        req = req.header(*k, *v);
    }
    let r = req.send().await.unwrap();
    (r.status().as_u16(), r.text().await.unwrap())
}

async fn get(base: &str, path: &str) -> u16 {
    reqwest::Client::new().get(format!("{base}{path}")).send().await.unwrap().status().as_u16()
}

async fn post_json(base: &str, path: &str, body: Value) -> Value {
    let (status, text) = post(base, path, body).await;
    assert_eq!(status, 200, "{text}");
    serde_json::from_str(&text).unwrap()
}

/// SSE body -> (event name, data JSON) pairs; `[DONE]` becomes a null data.
fn sse(text: &str) -> Vec<(String, Value)> {
    text.split("\n\n")
        .filter(|f| !f.trim().is_empty() && !f.starts_with(':'))
        .map(|f| {
            let mut ev = String::new();
            let mut data = Value::Null;
            for line in f.lines() {
                if let Some(e) = line.strip_prefix("event: ") {
                    ev = e.into();
                } else if let Some(d) = line.strip_prefix("data: ") {
                    data = if d == "[DONE]" { Value::Null } else { serde_json::from_str(d).unwrap() };
                }
            }
            (ev, data)
        })
        .collect()
}

fn weather_tool_openai() -> Value {
    json!([{"type": "function", "function": {"name": "get_weather", "description": "Weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}])
}

fn weather_tool_anthropic() -> Value {
    json!([{"name": "get_weather", "description": "Weather for a city",
        "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}])
}

fn chat_prompt_tokens(body: &Value) -> usize {
    let conv = serde_json::from_value::<ChatCompletionRequest>(body.clone()).unwrap().convert(ConversionOptions::default()).unwrap();
    DeepseekV41Encoding::new().with_tokenizer(tokenizer()).encode(&conv.conversation).unwrap().len()
}

#[tokio::test(flavor = "multi_thread")]
async fn openai_chat_nonstream() {
    let base = start(MockMode::Scripted, 0).await;
    let body = json!({"model": "deepseek-v4.1-flash", "messages": [{"role": "user", "content": "Hello"}]});
    let r = post_json(&base, "/v1/chat/completions", body.clone()).await;
    let m = &r["choices"][0]["message"];
    assert_eq!(m["content"], MOCK_REPLY);
    assert_eq!(m["reasoning_content"], MOCK_REASONING);
    assert_eq!(r["choices"][0]["finish_reason"], "stop");
    assert_eq!(r["usage"]["prompt_tokens"], chat_prompt_tokens(&body));
    assert!(r["usage"]["completion_tokens"].as_u64().unwrap() > 5);
    // thinking disabled: no reasoning, the prompt ends in </think>
    let r = post_json(&base, "/v1/chat/completions", json!({"model": "m", "thinking": {"type": "disabled"},
        "messages": [{"role": "user", "content": "Hello"}]})).await;
    assert_eq!(r["choices"][0]["message"]["content"], MOCK_REPLY);
    assert!(r["choices"][0]["message"]["reasoning_content"].is_null());
    // llama.cpp's switches (its web UI sends chat_template_kwargs.enable_thinking) do the same
    for off in [json!({"chat_template_kwargs": {"enable_thinking": false}}), json!({"thinking": false}),
                json!({"chat_template_kwargs": {"reasoning_effort": "none"}})] {
        let mut b = json!({"model": "m", "messages": [{"role": "user", "content": "Hello"}]});
        b.as_object_mut().unwrap().extend(off.as_object().unwrap().clone());
        let r = post_json(&base, "/v1/chat/completions", b).await;
        assert_eq!(r["choices"][0]["message"]["content"], MOCK_REPLY, "{off}");
        assert!(r["choices"][0]["message"]["reasoning_content"].is_null(), "{off}");
    }
    // an explicit thinking object wins over chat_template_kwargs
    let r = post_json(&base, "/v1/chat/completions", json!({"model": "m", "thinking": {"type": "enabled"},
        "chat_template_kwargs": {"enable_thinking": false}, "messages": [{"role": "user", "content": "Hello"}]})).await;
    assert_eq!(r["choices"][0]["message"]["reasoning_content"], MOCK_REASONING);
}

#[tokio::test(flavor = "multi_thread")]
async fn openai_chat_stream() {
    let base = start(MockMode::Scripted, 0).await;
    let (status, text) = post(&base, "/v1/chat/completions", json!({"model": "m", "stream": true,
        "stream_options": {"include_usage": true}, "messages": [{"role": "user", "content": "Hello"}]})).await;
    assert_eq!(status, 200);
    let ev = sse(&text);
    assert!(ev.last().unwrap().1.is_null(), "ends with [DONE]");
    let (mut content, mut reasoning, mut finish, mut usage) = (String::new(), String::new(), None, None);
    for (_, d) in &ev {
        if let Some(c) = d["choices"].get(0) {
            content += c["delta"]["content"].as_str().unwrap_or("");
            reasoning += c["delta"]["reasoning_content"].as_str().unwrap_or("");
            if let Some(f) = c["finish_reason"].as_str() {
                finish = Some(f.to_string());
            }
        }
        if d["usage"].is_object() {
            usage = Some(d["usage"].clone());
        }
    }
    assert_eq!(content, MOCK_REPLY);
    assert_eq!(reasoning, MOCK_REASONING);
    assert_eq!(finish.as_deref(), Some("stop"));
    assert!(usage.unwrap()["completion_tokens"].as_u64().unwrap() > 5);
}

#[tokio::test(flavor = "multi_thread")]
async fn anthropic_messages_nonstream() {
    let base = start(MockMode::Scripted, 0).await;
    let r = post_json(&base, "/v1/messages", json!({"model": "m", "max_tokens": 256,
        "messages": [{"role": "user", "content": "Hello"}]})).await;
    assert_eq!(r["type"], "message");
    assert_eq!(r["content"][0]["type"], "thinking");
    assert_eq!(r["content"][0]["thinking"], MOCK_REASONING);
    assert_eq!(r["content"][1]["type"], "text");
    assert_eq!(r["content"][1]["text"], MOCK_REPLY);
    assert_eq!(r["stop_reason"], "end_turn");
    assert!(r["usage"]["input_tokens"].as_u64().unwrap() > 5);
    // errors use Anthropic's envelope
    let (status, text) = post(&base, "/v1/messages", json!({"model": "m"})).await;
    assert_eq!(status, 400);
    let e: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(e["type"], "error");
    assert_eq!(e["error"]["type"], "invalid_request_error");
}

#[tokio::test(flavor = "multi_thread")]
async fn anthropic_messages_stream() {
    let base = start(MockMode::Scripted, 0).await;
    let (status, text) = post(&base, "/v1/messages", json!({"model": "m", "max_tokens": 256, "stream": true,
        "thinking": {"type": "disabled"}, "messages": [{"role": "user", "content": "Hello"}]})).await;
    assert_eq!(status, 200);
    let ev = sse(&text);
    let names: Vec<&str> = ev.iter().map(|(e, _)| e.as_str()).collect();
    assert_eq!(names.first(), Some(&"message_start"));
    assert_eq!(names.last(), Some(&"message_stop"));
    assert!(names.contains(&"content_block_start") && names.contains(&"message_delta"));
    let text: String = ev
        .iter()
        .filter(|(e, d)| e == "content_block_delta" && d["delta"]["type"] == "text_delta")
        .map(|(_, d)| d["delta"]["text"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(text, MOCK_REPLY);
    let delta = ev.iter().find(|(e, _)| e == "message_delta").unwrap();
    assert_eq!(delta.1["delta"]["stop_reason"], "end_turn");
}

#[tokio::test(flavor = "multi_thread")]
async fn openai_tool_call_round_trip() {
    let base = start(MockMode::Scripted, 0).await;
    let user = json!({"role": "user", "content": "Weather in Paris?"});
    let r = post_json(&base, "/v1/chat/completions", json!({"model": "m", "tools": weather_tool_openai(), "messages": [user]})).await;
    let c = &r["choices"][0];
    assert_eq!(c["finish_reason"], "tool_calls");
    let call = &c["message"]["tool_calls"][0];
    assert_eq!(call["type"], "function");
    assert_eq!(call["function"]["name"], "get_weather");
    let args: Value = serde_json::from_str(call["function"]["arguments"].as_str().unwrap()).unwrap();
    assert_eq!(args, json!({"city": "mock"}));
    // send the result back with the previous turn (thinking included) as history
    let mut assistant = c["message"].clone();
    assistant.as_object_mut().unwrap().retain(|k, _| ["role", "content", "reasoning_content", "tool_calls"].contains(&k.as_str()));
    let tool = json!({"role": "tool", "tool_call_id": call["id"], "content": "Sunny, 21C"});
    let r = post_json(&base, "/v1/chat/completions", json!({"model": "m", "tools": weather_tool_openai(),
        "messages": [user, assistant, tool]})).await;
    assert_eq!(r["choices"][0]["message"]["content"], "The tool returned: Sunny, 21C");
    assert_eq!(r["choices"][0]["finish_reason"], "stop");

    // streamed tool call: name and argument deltas reassemble
    let (_, text) = post(&base, "/v1/chat/completions", json!({"model": "m", "stream": true, "tools": weather_tool_openai(),
        "messages": [{"role": "user", "content": "Weather?"}]})).await;
    let (mut name, mut args, mut finish) = (String::new(), String::new(), String::new());
    for (_, d) in sse(&text) {
        if let Some(c) = d["choices"].get(0) {
            if let Some(tc) = c["delta"]["tool_calls"].get(0) {
                name += tc["function"]["name"].as_str().unwrap_or("");
                args += tc["function"]["arguments"].as_str().unwrap_or("");
            }
            if let Some(f) = c["finish_reason"].as_str() {
                finish = f.into();
            }
        }
    }
    assert_eq!(name, "get_weather");
    assert_eq!(serde_json::from_str::<Value>(&args).unwrap(), json!({"city": "mock"}));
    assert_eq!(finish, "tool_calls");
}

#[tokio::test(flavor = "multi_thread")]
async fn anthropic_tool_use_round_trip() {
    let base = start(MockMode::Scripted, 0).await;
    let user = json!({"role": "user", "content": "Weather in Paris?"});
    let r = post_json(&base, "/v1/messages", json!({"model": "m", "max_tokens": 512, "tools": weather_tool_anthropic(),
        "messages": [user]})).await;
    assert_eq!(r["stop_reason"], "tool_use");
    let blocks = r["content"].as_array().unwrap();
    let tu = blocks.iter().find(|b| b["type"] == "tool_use").unwrap();
    assert_eq!(tu["name"], "get_weather");
    assert_eq!(tu["input"], json!({"city": "mock"}));
    let result = json!({"role": "user", "content": [{"type": "tool_result", "tool_use_id": tu["id"], "content": "Rain"}]});
    let r = post_json(&base, "/v1/messages", json!({"model": "m", "max_tokens": 512, "tools": weather_tool_anthropic(),
        "messages": [user, {"role": "assistant", "content": r["content"]}, result]})).await;
    let text = r["content"].as_array().unwrap().iter().find(|b| b["type"] == "text").unwrap();
    assert_eq!(text["text"], "The tool returned: Rain");
    assert_eq!(r["stop_reason"], "end_turn");
}

#[tokio::test(flavor = "multi_thread")]
async fn responses_api() {
    let base = start(MockMode::Scripted, 0).await;
    let r = post_json(&base, "/v1/responses", json!({"model": "m", "input": "Hello"})).await;
    assert_eq!(r["status"], "completed");
    let out = r["output"].as_array().unwrap();
    let msg = out.iter().find(|o| o["type"] == "message").unwrap();
    assert_eq!(msg["content"][0]["text"], MOCK_REPLY);
    let (status, text) = post(&base, "/v1/responses", json!({"model": "m", "input": "Hello", "stream": true})).await;
    assert_eq!(status, 200);
    let ev = sse(&text);
    assert!(ev.iter().any(|(e, _)| e == "response.output_text.delta"));
    assert_eq!(ev.last().unwrap().0, "response.completed");
}

#[tokio::test(flavor = "multi_thread")]
async fn long_prompt_token_count() {
    let base = start(MockMode::Scripted, 0).await;
    let long: String = (0..6000).map(|i| format!("word{} ", i % 97)).collect();
    let body = json!({"model": "m", "max_tokens": 64, "messages": [{"role": "system", "content": "Be terse."},
        {"role": "user", "content": long}]});
    let r = post_json(&base, "/v1/chat/completions", body.clone()).await;
    let n = r["usage"]["prompt_tokens"].as_u64().unwrap() as usize;
    assert_eq!(n, chat_prompt_tokens(&body));
    assert!(n > 12_000 && n < 40_000, "{n}");
    // Anthropic count_tokens agrees with what /v1/messages prefills
    let mb = json!({"model": "m", "max_tokens": 64, "system": "Be terse.", "messages": [{"role": "user", "content": long}]});
    let c = post_json(&base, "/v1/messages/count_tokens", mb.clone()).await["input_tokens"].as_u64().unwrap();
    let conv = serde_json::from_value::<MessagesRequest>(mb.clone()).unwrap().convert(ConversionOptions::default()).unwrap();
    assert_eq!(c as usize, DeepseekV41Encoding::new().with_tokenizer(tokenizer()).encode(&conv.conversation).unwrap().len());
    let m = post_json(&base, "/v1/messages", mb).await;
    assert_eq!(m["usage"]["input_tokens"].as_u64().unwrap(), c);
    // over the context window: 400
    let huge: String = (0..40_000).map(|i| format!("w{i} ")).collect();
    let (status, text) = post(&base, "/v1/chat/completions", json!({"model": "m", "messages": [{"role": "user", "content": huge}]})).await;
    assert_eq!(status, 400, "{text}");
    assert!(text.contains("context window"));
}

#[tokio::test(flavor = "multi_thread")]
async fn limits_and_stops() {
    let base = start(MockMode::Scripted, 0).await;
    let off = json!({"type": "disabled"});
    let r = post_json(&base, "/v1/chat/completions", json!({"model": "m", "thinking": off, "max_tokens": 3,
        "messages": [{"role": "user", "content": "Hello"}]})).await;
    assert_eq!(r["choices"][0]["finish_reason"], "length");
    assert_eq!(r["usage"]["completion_tokens"], 3);
    let r = post_json(&base, "/v1/chat/completions", json!({"model": "m", "thinking": off, "stop": [" mock"],
        "messages": [{"role": "user", "content": "Hello"}]})).await;
    assert_eq!(r["choices"][0]["message"]["content"], "Hello from the dsx");
    assert_eq!(r["choices"][0]["finish_reason"], "stop");
    let r = post_json(&base, "/v1/messages", json!({"model": "m", "max_tokens": 100, "thinking": off, "stop_sequences": [" mock"],
        "messages": [{"role": "user", "content": "Hello"}]})).await;
    assert_eq!(r["stop_reason"], "stop_sequence");
    assert_eq!(r["stop_sequence"], " mock");
    // images need the vision weights: rejected up front by a server without them
    let (status, text) = post(&base, "/v1/chat/completions", json!({"model": "m", "messages": [{"role": "user",
        "content": [{"type": "text", "text": "what is this"}, {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}]}]})).await;
    assert_eq!(status, 400);
    assert!(text.contains("vision weights"), "{text}");
}

#[tokio::test(flavor = "multi_thread")]
async fn disconnect_cancels_generation() {
    // echo a ~300-token prompt at 20 ms/token: ~6 s if nobody cancels
    let base = start(MockMode::Echo, 20).await;
    let long: String = (0..250).map(|i| format!("t{i} ")).collect();
    let client = reqwest::Client::new();
    let mut r = client
        .post(format!("{base}/v1/chat/completions"))
        .json(&json!({"model": "m", "stream": true, "messages": [{"role": "user", "content": long}]}))
        .send()
        .await
        .unwrap();
    assert!(r.chunk().await.unwrap().is_some());
    let health = || async { client.get(format!("{base}/health")).send().await.unwrap().json::<Value>().await.unwrap() };
    assert_eq!(health().await["pending"], 1);
    drop(r);
    let t = Instant::now();
    while health().await["pending"] != 0 {
        assert!(t.elapsed() < Duration::from_secs(2), "generation was not cancelled");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let models: Value = client.get(format!("{base}/v1/models")).send().await.unwrap().json().await.unwrap();
    assert_eq!(models["data"][0]["id"], "deepseek-v4.1-flash");
}

// ---- api key

const KEY: &str = "sk-dsx-test-0123456789abcdef";

fn keyed() -> ServerConfig {
    ServerConfig { api_key: Some(KEY.into()), ..ServerConfig::default() }
}

async fn chat_status(base: &str, headers: &[(&str, &str)]) -> (u16, String) {
    let body = json!({"model": "m", "messages": [{"role": "user", "content": "Hello"}]});
    post_h(base, "/v1/chat/completions", body, headers).await
}

#[tokio::test(flavor = "multi_thread")]
async fn api_key_guards_the_api() {
    let base = start_cfg(MockMode::Scripted, 0, keyed()).await;
    let bearer = format!("Bearer {KEY}");

    // no credentials: 401 in the OpenAI envelope, and no generation started
    let (status, text) = chat_status(&base, &[]).await;
    assert_eq!(status, 401, "{text}");
    let e: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(e["error"]["type"], "invalid_request_error");
    assert_eq!(e["error"]["code"], "invalid_api_key");
    let health: Value = reqwest::Client::new().get(format!("{base}/health")).send().await.unwrap().json().await.unwrap();
    assert_eq!(health["pending"], 0);

    // either header, either protocol
    assert_eq!(chat_status(&base, &[("authorization", &bearer)]).await.0, 200);
    assert_eq!(chat_status(&base, &[("x-api-key", KEY)]).await.0, 200);
    let msgs = json!({"model": "m", "max_tokens": 16, "messages": [{"role": "user", "content": "Hello"}]});
    assert_eq!(post_h(&base, "/v1/messages", msgs.clone(), &[("x-api-key", KEY)]).await.0, 200);
    let lower = format!("bearer {KEY}");
    assert_eq!(chat_status(&base, &[("authorization", &lower)]).await.0, 200);

    // wrong key, a prefix of the key, and a bare key without the scheme are all rejected
    assert_eq!(chat_status(&base, &[("authorization", "Bearer nope")]).await.0, 401);
    assert_eq!(chat_status(&base, &[("x-api-key", &KEY[..8])]).await.0, 401);
    assert_eq!(chat_status(&base, &[("authorization", KEY)]).await.0, 401);
    assert_eq!(chat_status(&base, &[("x-api-key", "")]).await.0, 401);

    // Anthropic's error envelope on /v1/messages
    let (status, text) = post(&base, "/v1/messages", msgs).await;
    assert_eq!(status, 401);
    let e: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(e["type"], "error");
    assert_eq!(e["error"]["type"], "authentication_error");

    // the other API surfaces are guarded too
    for (path, body) in [
        ("/v1/responses", json!({"model": "m", "input": "hi"})),
        ("/v1/messages/count_tokens", json!({"model": "m", "messages": [{"role": "user", "content": "hi"}]})),
        ("/v1/streams/lookup", json!({})),
        ("/tools", json!({})),
    ] {
        let (status, text) = post(&base, path, body).await;
        assert_eq!(status, 401, "{path}: {text}");
    }
    for path in ["/v1/models", "/props", "/slots", "/tools"] {
        assert_eq!(get(&base, path).await, 401, "{path}");
    }

    // health stays open (tunnel health checks) and the UI shell loads so its key field is reachable
    assert_eq!(get(&base, "/health").await, 200);
    assert_eq!(get(&base, "/v1/health").await, 200);
    assert_eq!(get(&base, "/").await, 200);
    assert_eq!(get(&base, "/_app/immutable/entry/start.js").await, 404, "assets need no key, missing ones are just missing");
}

#[tokio::test(flavor = "multi_thread")]
async fn no_api_key_leaves_the_api_open() {
    // the default: no key configured, every request is served (as before the flag existed)
    let base = start(MockMode::Scripted, 0).await;
    assert_eq!(chat_status(&base, &[]).await.0, 200);
    assert_eq!(chat_status(&base, &[("authorization", "Bearer whatever")]).await.0, 200);
    // an empty --api-key is not a lock with no key, it is no lock at all
    let base = start_cfg(MockMode::Scripted, 0, ServerConfig { api_key: Some("  ".into()), ..ServerConfig::default() }).await;
    assert_eq!(chat_status(&base, &[]).await.0, 200);
}

#[tokio::test(flavor = "multi_thread")]
async fn large_request_bodies_are_accepted() {
    // base64 images make multi-MB requests; axum's default 2 MB body limit used to reject them with 413
    let base = start(MockMode::Scripted, 0).await;
    let big = "a ".repeat(4 * 1024 * 1024); // 8 MB of text
    let body = json!({"model": "m", "max_tokens": 4, "messages": [{"role": "user", "content": big}]});
    let (status, text) = post(&base, "/v1/chat/completions", body).await;
    assert_ne!(status, 413, "{}", &text[..text.len().min(200)]);
}
