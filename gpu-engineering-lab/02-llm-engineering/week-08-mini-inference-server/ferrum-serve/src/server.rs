//! Day 3 — axum front-end: OpenAI-compatible route, SSE streaming,
//! cancellation, graceful shutdown.
//!
//! API (OpenAI-compatible — `bench/loadtest.py` and any OpenAI SDK client
//! work unchanged):
//!
//!   POST /v1/completions
//!     body: {model?, prompt, max_tokens=128, temperature=0.0, stop=[], stream=true}
//!     stream=true  -> text/event-stream, OpenAI chunk shape:
//!                       data: {"choices":[{"text":"<chunk>","index":0}]}\n\n
//!                     terminated by:  data: [DONE]\n\n
//!     stream=false -> {"choices":[{"text":"<full text>","index":0}]}
//!   GET /health -> {"status":"ok"}
//!   GET /stats  -> engine::Stats as JSON (watch blocks recycle during the
//!                  Day-3 smoke test — this is how you SEE continuous batching)
//!
//! Stretch: POST /v1/chat/completions (apply the model's chat template, same
//! streaming plumbing).
//!
//! GRACEFUL SHUTDOWN is a first-class deliverable (week 11 rolls this server
//! in K8s, where SIGTERM-then-SIGKILL is the pod lifecycle): on SIGTERM or
//! ctrl-c, stop ACCEPTING requests, let in-flight streams drain, then exit 0.

use axum::extract::State;
use axum::response::Response;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;

use crate::engine::EngineHandle;

#[derive(Debug, Deserialize)]
pub struct CompletionBody {
    #[serde(default)]
    pub model: Option<String>, // accepted-and-ignored: one model per process
    pub prompt: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default)]
    pub temperature: f64,
    #[serde(default)]
    pub stop: Vec<String>,
    #[serde(default = "default_stream")]
    pub stream: bool,
}

fn default_max_tokens() -> usize {
    128
}

fn default_stream() -> bool {
    true
}

pub fn build_router(engine: EngineHandle) -> Router {
    Router::new()
        .route("/v1/completions", post(completions))
        .route("/health", get(health))
        .route("/stats", get(stats))
        .with_state(engine)
}

async fn health() -> &'static str {
    "{\"status\":\"ok\"}"
}

/// TODO(Day 3):
///   1. Build an mpsc channel, send EngineCommand::Generate to the engine.
///   2. stream=true: wrap the receiver in
///      tokio_stream::wrappers::ReceiverStream, map StreamEvents to
///      axum::response::sse::Event payloads in the OpenAI chunk shape above,
///      end with a literal "[DONE]" event, return Sse::new(stream) as a
///      Response.
///      CANCELLATION comes for free with ownership: when the client
///      disconnects, axum drops the stream, the Receiver drops, and the
///      engine's try_send fails -> it cancels the request and frees blocks.
///      Verify with ctrl-C'd curls while watching /stats. No is_disconnected
///      polling — this is the Rust-shaped version of that logic.
///   3. stream=false: drain the receiver into a String, return the JSON body.
async fn completions(
    State(engine): State<EngineHandle>,
    Json(body): Json<CompletionBody>,
) -> Response {
    let _ = (engine, body);
    todo!("Day 3: implement completions")
}

/// TODO(Day 3): oneshot to the engine (EngineCommand::Stats), await the
/// reply, Json(stats) into a Response.
async fn stats(State(engine): State<EngineHandle>) -> Response {
    let _ = engine;
    todo!("Day 3: implement stats")
}

/// Resolves when SIGTERM (unix) or ctrl-c arrives.
///
/// TODO(Day 3): tokio::select! over tokio::signal::ctrl_c() and
/// tokio::signal::unix::signal(SignalKind::terminate()). Log which one fired.
pub async fn shutdown_signal() {
    todo!("Day 3: implement shutdown_signal")
}

/// Bind + serve with graceful drain.
///
/// TODO(Day 3):
///   let listener = tokio::net::TcpListener::bind(addr).await?;
///   axum::serve(listener, build_router(engine))
///       .with_graceful_shutdown(shutdown_signal())
///       .await?;
/// Then (after serve returns) give in-flight engine work a bounded drain
/// window before exiting. Test: start a long stream, kill -TERM the pid —
/// the stream must finish (or hit the drain deadline), the process exit 0.
pub async fn serve(addr: std::net::SocketAddr, engine: EngineHandle) -> anyhow::Result<()> {
    let _ = (addr, engine);
    todo!("Day 3: implement serve")
}
