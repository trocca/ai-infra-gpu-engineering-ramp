//! ferrum-serve binary. COMPLETE except the final wiring line (Day 3).
//!
//!   cargo run --release --features cuda -- --port 8000
//!   cargo run --release --features cuda -- --model-id meta-llama/Llama-3.2-1B-Instruct

use clap::Parser;
use ferrum_serve::engine::EngineConfig;

#[derive(Parser, Debug)]
#[command(name = "ferrum-serve", about = "Mini LLM inference server in Rust")]
struct Args {
    /// HF model id (Qwen2.5-1.5B-Instruct is ungated; Llama-3.2-1B is gated)
    #[arg(long, default_value = "Qwen/Qwen2.5-1.5B-Instruct")]
    model_id: String,

    #[arg(long, default_value = "main")]
    revision: String,

    #[arg(long, default_value = "0.0.0.0")]
    host: String,

    #[arg(long, default_value_t = 8000)]
    port: u16,

    /// Tokens per KV block
    #[arg(long, default_value_t = 16)]
    block_size: usize,

    /// Total KV blocks (blocks * block_size = max cached tokens)
    #[arg(long, default_value_t = 4096)]
    num_blocks: usize,

    #[arg(long, default_value_t = 16)]
    max_running: usize,

    #[arg(long, default_value_t = 2048)]
    max_prefill_tokens: usize,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "ferrum_serve=info,info".into()),
        )
        .init();

    let args = Args::parse();
    let cfg = EngineConfig {
        model_id: args.model_id,
        revision: args.revision,
        block_size: args.block_size,
        num_blocks: args.num_blocks,
        max_running: args.max_running,
        max_prefill_tokens: args.max_prefill_tokens,
        ..EngineConfig::default()
    };
    let addr: std::net::SocketAddr = format!("{}:{}", args.host, args.port).parse()?;
    tracing::info!(?addr, model = %cfg.model_id, "starting ferrum-serve");

    // TODO(Day 3): the two lines that make it a server —
    //   let engine = ferrum_serve::engine::spawn_engine(cfg)?;
    //   ferrum_serve::server::serve(addr, engine).await
    let _ = (cfg, addr);
    todo!("Day 3: spawn engine + serve (see comment above)")
}
