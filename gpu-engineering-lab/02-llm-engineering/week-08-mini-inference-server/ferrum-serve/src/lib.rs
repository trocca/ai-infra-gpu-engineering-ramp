//! ferrum-serve — a miniature vLLM in Rust.
//!
//! Module map (data flows top to bottom):
//!
//! ```text
//!   server.rs     axum: HTTP + SSE streaming, cancellation, SIGTERM
//!      |
//!   engine.rs     model runner: Candle forward passes, sampling, KV writes
//!      |
//!   scheduler.rs  continuous batching: who prefills / decodes this iteration
//!      |
//!   blocks.rs     paged KV bookkeeping: block tables, alloc/free, ref-counts
//! ```
//!
//! `blocks` and `scheduler` are pure logic (no candle, no tokio) — they carry
//! the COMPLETE test suites in `tests/` and are where the red→green work
//! starts. `engine` and `server` are integration skeletons.

pub mod blocks;
pub mod engine;
pub mod scheduler;
pub mod server;
