//! Days 1 & 3 — the engine: Candle model runner + paged KV tensors + the loop.
//!
//! Threading model (decide it once, here): Candle forward passes are
//! synchronous CPU/GPU work — they must NOT run on the tokio reactor. The
//! engine therefore runs on its OWN dedicated std thread with a
//! `tokio::sync::mpsc` command channel in and per-request token channels out.
//! The axum handlers only ever touch channels — that is the whole
//! server/engine interface, and it is what makes cancellation and shutdown
//! clean (drop the channel, the engine notices).
//!
//! Day 1 scoping note (read this before designing): KV blocks are integrated
//! by GATHERING each sequence's blocks into a contiguous (seq_len, kv_heads,
//! head_dim) tensor per layer per step, then running ordinary attention.
//! This teaches the memory management honestly without writing a custom
//! kernel. TRUE paged attention (the kernel indexes the block table directly)
//! is the stretch goal — and the honest gap vs real vLLM that MUST be stated
//! in the benchmark writeup.

use candle_core::{DType, Device, Tensor};
use tokio::sync::{mpsc, oneshot};

use crate::blocks::SeqId;

/// Runtime configuration (filled from CLI flags in main.rs).
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// HF model id. Default `Qwen/Qwen2.5-1.5B-Instruct` (ungated — downloads
    /// with no token). Alternative: `meta-llama/Llama-3.2-1B-Instruct` is
    /// GATED — accept the license on the model page + `huggingface-cli login`.
    pub model_id: String,
    pub revision: String,
    pub block_size: usize,
    /// num_blocks * block_size = max cached tokens. Sizing this against what
    /// the weights leave free on the 24 GB card is a real Day-1 decision.
    pub num_blocks: usize,
    pub max_running: usize,
    pub max_prefill_tokens: usize,
    pub dtype: DType,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            model_id: "Qwen/Qwen2.5-1.5B-Instruct".to_string(),
            revision: "main".to_string(),
            block_size: 16,
            num_blocks: 4096,
            max_running: 16,
            max_prefill_tokens: 2048,
            dtype: DType::BF16,
        }
    }
}

/// What the server sends the engine.
pub enum EngineCommand {
    Generate {
        prompt: String,
        max_tokens: usize,
        temperature: f64,
        stop: Vec<String>,
        /// Engine pushes decoded text chunks here; closing the channel (or
        /// sending `StreamEvent::Done`) ends the client's SSE stream.
        events: mpsc::Sender<StreamEvent>,
    },
    Cancel(SeqId),
    Stats(oneshot::Sender<Stats>),
}

#[derive(Debug)]
pub enum StreamEvent {
    /// The request's SeqId, sent once so the server can cancel later.
    Started(SeqId),
    Token(String),
    Done,
    Error(String),
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct Stats {
    pub running: usize,
    pub waiting: usize,
    pub free_block_frac: f64,
}

/// Cheap clonable handle the axum handlers hold.
#[derive(Clone)]
pub struct EngineHandle {
    pub cmd_tx: mpsc::Sender<EngineCommand>,
}

/// Everything loaded from the Hub: weights, tokenizer, model config.
pub struct LoadedModel {
    pub device: Device,
    // TODO(Day 1): fields — the candle_transformers qwen2 model struct, the
    // tokenizers::Tokenizer, eos token id(s), layer/head/dim counts you need
    // for KV sizing. Keep it a plain struct; no trait gymnastics needed.
}

/// Day 1, first half: "Candle hello" — prove the stack end to end BEFORE any
/// batching exists.
///
/// TODO(Day 1):
///   1. hf_hub::api::sync::Api -> repo(model_id, revision) -> download
///      "config.json", "tokenizer.json", and the safetensors listed in
///      "model.safetensors.index.json" (or the single "model.safetensors").
///   2. candle_nn::VarBuilder::from_mmaped_safetensors(&paths, dtype, &device)
///   3. Build the model with candle_transformers::models::qwen2 (its Config
///      derives Deserialize — serde_json the config.json into it).
///   4. Device: candle_core::Device::new_cuda(0) under `--features cuda`,
///      Device::Cpu otherwise (CPU decode of a 1.5B is slow but WORKS — fine
///      for the hello).
pub fn load_model(cfg: &EngineConfig) -> anyhow::Result<LoadedModel> {
    let _ = cfg;
    todo!("Day 1: implement load_model")
}

/// Day 1, second half: single-prompt greedy decode, no batching, model's own
/// internal KV cache. This output is the ORACLE for the week's acceptance
/// test: at temperature 0, the batched engine must reproduce it token for
/// token, at any concurrency.
pub fn generate_single(
    model: &mut LoadedModel,
    prompt: &str,
    max_tokens: usize,
) -> anyhow::Result<String> {
    let _ = (model, prompt, max_tokens);
    todo!("Day 1: implement generate_single (greedy, single sequence)")
}

/// The paged KV storage that BlockManager ids index into.
///
/// Layout per layer: k/v tensors of shape
/// (num_blocks, block_size, num_kv_heads, head_dim).
pub struct PagedKvCache {
    pub k: Vec<Tensor>, // one per layer
    pub v: Vec<Tensor>,
}

impl PagedKvCache {
    /// TODO(Day 3): allocate zeros on the model's device/dtype. Log the GB.
    pub fn new(
        cfg: &EngineConfig,
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        device: &Device,
    ) -> anyhow::Result<Self> {
        let _ = (cfg, num_layers, num_kv_heads, head_dim, device);
        todo!("Day 3: implement PagedKvCache::new")
    }

    /// Scatter `k`/`v` for T_new tokens into the blocks covering logical
    /// positions [start_pos, start_pos + T_new). Position p lives at
    /// (table[p / block_size], p % block_size).
    ///
    /// TODO(Day 3): build flat index tensors once per call and use
    /// Tensor::index_add / slice assignment — a per-token loop of tiny CUDA
    /// copies will dominate your ITL (you will SEE this in the load test).
    pub fn write(
        &mut self,
        layer: usize,
        table: &[u32],
        start_pos: usize,
        k: &Tensor,
        v: &Tensor,
    ) -> anyhow::Result<()> {
        let _ = (layer, table, start_pos, k, v);
        todo!("Day 3: implement PagedKvCache::write")
    }

    /// Gather a sequence's KV back to contiguous (num_tokens, kv_heads, dim):
    /// index_select the table's blocks, reshape, slice to num_tokens.
    /// (This gather is the "not real paged attention" concession — see the
    /// module docs and say so in the writeup.)
    pub fn gather(
        &self,
        layer: usize,
        table: &[u32],
        num_tokens: usize,
    ) -> anyhow::Result<(Tensor, Tensor)> {
        let _ = (layer, table, num_tokens);
        todo!("Day 3: implement PagedKvCache::gather")
    }
}

/// Spawn the engine on its own thread; return the handle the server holds.
///
/// TODO(Day 3): std::thread::spawn a loop that:
///   1. Drains pending EngineCommands (non-blocking `try_recv` when busy,
///      blocking `blocking_recv` when idle — the idle path is what keeps a
///      quiet server at 0% CPU):
///        Generate -> tokenize, build a scheduler Request, stash the events
///                    sender in a HashMap<SeqId, Sender<StreamEvent>>
///        Cancel   -> scheduler.cancel(id)
///        Stats    -> reply on the oneshot
///   2. If scheduler.has_work(): out = scheduler.schedule();
///        - prefill: full-prompt forward per request (write KV via block
///          table), sample the first token
///        - decode: ONE batched forward over all decode ids' single tokens
///          (gather KV per sequence; ragged lengths -> per-sequence attention
///          inside the batched step is the simple correct version)
///        - per sampled token: scheduler.on_token_generated, detokenize
///          incrementally, try_send StreamEvent::Token. try_send FAILING
///          (receiver dropped = client disconnected) IS your cancellation
///          signal -> scheduler.cancel + free the entry. Check stop strings /
///          EOS / max_tokens -> scheduler.finish + StreamEvent::Done.
///   3. Acceptance gate: temperature-0 outputs identical to generate_single.
pub fn spawn_engine(cfg: EngineConfig) -> anyhow::Result<EngineHandle> {
    let _ = cfg;
    todo!("Day 3: implement spawn_engine")
}
