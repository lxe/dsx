//! The API key lock: one key per server (`--api-key`), checked before every route that can run the model
//! or report on it.
//!
//! Clients present the key either as `Authorization: Bearer <key>` (the OpenAI convention, and what the
//! embedded web UI sends once its API key setting is filled in) or as `x-api-key: <key>` (the Anthropic
//! convention), so the one key works with both protocols. `/health`, `/v1/health` and the web UI's static
//! files stay open: a browser cannot put a header on a `<script src>`, and a tunnel's health check should
//! not have to hold a secret.
//!
//! This is a lock on the door, not accounts: one key, no scopes, no expiry, no revocation list, no
//! per-client accounting. It keeps strangers off a tunneled port. Terminate TLS in front of the server
//! (Cloudflare Tunnel, Caddy, nginx): the key travels as a plain header.

use std::net::SocketAddr;
use std::sync::Arc;

use axum::extract::{ConnectInfo, Request, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use serde_json::json;

use super::AppState;

/// Where a client may put the key: `x-api-key` (Anthropic) first, then `Authorization: Bearer` (OpenAI).
const X_API_KEY: &str = "x-api-key";
const AUTHORIZATION: &str = "authorization";

/// Middleware for the API routes (see `router`): pass, or 401 in the protocol's own error envelope.
///
/// With `api_key: None` every request passes, unchanged from before the flag existed.
pub async fn require_key(State(st): State<Arc<AppState>>, req: Request, next: Next) -> Response {
    let Some(key) = st.cfg.api_key.as_deref() else { return next.run(req).await };
    let presented = presented_key(req.headers());
    if presented.is_some_and(|p| eq(p.as_bytes(), key.as_bytes())) {
        return next.run(req).await;
    }
    // log who and what was missing, never the key itself (or the one presented: it may be a typo of it)
    let peer = req.extensions().get::<ConnectInfo<SocketAddr>>().map(|c| c.0);
    let client = req
        .headers()
        .get("cf-connecting-ip")
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned)
        .or_else(|| peer.map(|p| p.to_string()))
        .unwrap_or_else(|| "-".into());
    tracing::warn!(
        path = %req.uri().path(),
        client = %client,
        failure = if presented.is_some() { "wrong key" } else { "no key" },
        "rejected request: api key"
    );
    deny(super::proto_for(req.uri().path()))
}

/// The key a request presents, if any.
fn presented_key(h: &HeaderMap) -> Option<&str> {
    let header = |name: &str| h.get(name).and_then(|v| v.to_str().ok()).map(str::trim).filter(|s| !s.is_empty());
    if let Some(k) = header(X_API_KEY) {
        return Some(k);
    }
    let v = h.get(AUTHORIZATION)?.to_str().ok()?.trim();
    let (scheme, key) = v.split_once(' ')?;
    scheme.eq_ignore_ascii_case("bearer").then(|| key.trim())
}

/// `a == b` without an early exit on the first differing byte, so a rejection takes as long whatever
/// prefix of the key was guessed. Lengths are compared first; the key's length is not the secret.
fn eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let diff = a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y));
    std::hint::black_box(diff) == 0
}

/// 401 in the envelope the caller parses: OpenAI clients read `error.code`, Anthropic clients `error.type`.
fn deny(proto: super::Proto) -> Response {
    const MESSAGE: &str = "invalid or missing api key";
    let body = match proto {
        super::Proto::OpenAi => {
            json!({"error": {"message": MESSAGE, "type": "invalid_request_error", "code": "invalid_api_key"}})
        }
        super::Proto::Anthropic => json!({"type": "error", "error": {"type": "authentication_error", "message": MESSAGE}}),
    };
    let mut res = (StatusCode::UNAUTHORIZED, axum::Json(body)).into_response();
    res.headers_mut().insert(header::WWW_AUTHENTICATE, HeaderValue::from_static("Bearer"));
    res
}

#[cfg(test)]
mod tests {
    use super::*;

    fn headers(pairs: &[(&'static str, &str)]) -> HeaderMap {
        let mut h = HeaderMap::new();
        for (k, v) in pairs {
            h.append(*k, HeaderValue::from_str(v).unwrap());
        }
        h
    }

    #[test]
    fn reads_the_key_from_either_header() {
        assert_eq!(presented_key(&headers(&[("x-api-key", "sk-abc")])), Some("sk-abc"));
        assert_eq!(presented_key(&headers(&[("authorization", "Bearer sk-abc")])), Some("sk-abc"));
        assert_eq!(presented_key(&headers(&[("authorization", "bearer sk-abc")])), Some("sk-abc"));
        assert_eq!(presented_key(&headers(&[("authorization", "BEARER  sk-abc ")])), Some("sk-abc"));
        // x-api-key wins when both are there
        assert_eq!(presented_key(&headers(&[("x-api-key", "one"), ("authorization", "Bearer two")])), Some("one"));
        // not a key: another scheme, a bare key, nothing
        assert_eq!(presented_key(&headers(&[("authorization", "Basic sk-abc")])), None);
        assert_eq!(presented_key(&headers(&[("authorization", "sk-abc")])), None);
        assert_eq!(presented_key(&headers(&[("authorization", "Bearer ")])), None);
        assert_eq!(presented_key(&headers(&[("x-api-key", "  ")])), None);
        assert_eq!(presented_key(&headers(&[])), None);
    }

    #[test]
    fn compares_without_an_early_exit() {
        assert!(eq(b"sk-abc", b"sk-abc"));
        assert!(!eq(b"sk-abc", b"sk-abd"));
        assert!(!eq(b"sk-ab", b"sk-abc"));
        assert!(!eq(b"", b"sk-abc"));
        assert!(eq(b"", b""));
        // a prefix of the key is not the key
        assert!(!eq(b"sk-ab", b"sk-abc"));
    }
}
