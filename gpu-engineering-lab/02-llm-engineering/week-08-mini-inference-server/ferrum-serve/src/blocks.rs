//! Day 1-2 — paged KV cache bookkeeping: the `BlockManager`.
//!
//! Pure logic, no GPU: this module manages block IDs; the engine maps IDs to
//! slices of the real KV tensors. The analogy to run with: this is an OS page
//! table for KV memory. vLLM's insight (PagedAttention, arXiv 2309.06180) is
//! that KV demand is dynamic and unpredictable, exactly like process memory —
//! so manage it like an OS does: fixed-size pages, per-sequence tables,
//! allocation on demand, ref-counts for sharing.
//!
//! CONTRACT — `tests/blocks_test.rs` (COMPLETE, do not edit) pins EXACTLY:
//!
//! - `new(num_blocks, block_size)` — all blocks start free.
//! - `blocks_needed(n)` = ceil(n / block_size).
//! - `allocate(seq, n)` -> `Ok(block_table)`;
//!   `Err(AlreadyAllocated)` on double-allocate, `Err(OutOfBlocks)` when the
//!   pool is short (callers are supposed to `can_allocate` first — erroring
//!   loudly catches scheduler bugs). Fresh blocks have ref-count 1.
//! - `append_slot(seq)` — room for ONE more token; `Ok(Some(new_block))` iff
//!   the tail block was full (a new block was grabbed), else `Ok(None)`;
//!   `Err(OutOfBlocks)` / `Err(UnknownSeq)`.
//! - `free(seq)` — decrement ref-count of each block in the seq's table;
//!   blocks return to the pool at ref-count 0. Idempotent: freeing an
//!   unknown/already-freed seq is a no-op.
//! - `fork(parent, child)` — child SHARES the parent's table (ref-count += 1
//!   on every block) and token count. This is the prefix-caching primitive:
//!   N requests with the same system prompt hold the same physical blocks.
//!   (Appending to a forked sequence needs copy-on-write — STRETCH; the tests
//!   only exercise fork + free semantics.)
//! - `block_table(seq)` / `num_tokens(seq)` -> `Option<_>` accessors,
//!   `ref_count(block)` for the sharing tests.

pub type SeqId = u64;
pub type BlockId = u32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockError {
    OutOfBlocks,
    AlreadyAllocated(SeqId),
    UnknownSeq(SeqId),
}

impl std::fmt::Display for BlockError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for BlockError {}

pub struct BlockManager {
    num_blocks: usize,
    block_size: usize,
    // TODO(Day 1 design / Day 2 impl): state you will need —
    //   free list:    Vec<BlockId> (pop/push — order doesn't matter)
    //   ref_counts:   Vec<u32>, indexed by BlockId (0 = free)
    //   block tables: HashMap<SeqId, Vec<BlockId>>
    //   token counts: HashMap<SeqId, usize>  (to know when the tail fills)
    // Rust note (first big program!): interior state behind &mut self only —
    // no RefCell/Mutex needed here; the OWNER of BlockManager (the Scheduler)
    // serializes access for you. That's the borrow checker doing concurrency
    // design.
}

impl BlockManager {
    pub fn new(num_blocks: usize, block_size: usize) -> Self {
        assert!(num_blocks > 0 && block_size > 0);
        let _ = (num_blocks, block_size);
        todo!("Day 2: implement BlockManager::new")
    }

    pub fn num_blocks(&self) -> usize {
        self.num_blocks
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn num_free_blocks(&self) -> usize {
        todo!("Day 2: implement num_free_blocks")
    }

    /// ceil(num_tokens / block_size). The internal-fragmentation story: a
    /// 17-token prompt with block_size 16 wastes 15 slots of its 2nd block —
    /// bounded waste, unlike reserving max_seq_len contiguously per request.
    pub fn blocks_needed(&self, num_tokens: usize) -> usize {
        let _ = num_tokens;
        todo!("Day 2: implement blocks_needed")
    }

    pub fn can_allocate(&self, num_tokens: usize) -> bool {
        let _ = num_tokens;
        todo!("Day 2: implement can_allocate")
    }

    /// Prefill-time allocation: grab `blocks_needed(num_tokens)` fresh blocks
    /// (each at ref-count 1), record table + token count, return the table.
    pub fn allocate(&mut self, seq: SeqId, num_tokens: usize) -> Result<Vec<BlockId>, BlockError> {
        let _ = (seq, num_tokens);
        todo!("Day 2: implement allocate")
    }

    /// Decode-time: make room for ONE more token.
    ///
    /// TODO(Day 2): if the previous token count is a multiple of block_size
    /// the tail block is full -> pop a fresh block (rc = 1), push onto the
    /// table, return Ok(Some(id)); else Ok(None). Increment the count either
    /// way. Errors BEFORE mutating anything (no half-applied state).
    pub fn append_slot(&mut self, seq: SeqId) -> Result<Option<BlockId>, BlockError> {
        let _ = seq;
        todo!("Day 2: implement append_slot")
    }

    /// Decrement ref-counts; blocks with rc 0 rejoin the free list. Drop the
    /// seq's table + count. Idempotent (finished and cancelled paths may both
    /// call it — the server's no-leak invariant depends on this).
    pub fn free(&mut self, seq: SeqId) {
        let _ = seq;
        todo!("Day 2: implement free")
    }

    /// Prefix sharing: `child` gets a CLONE of `parent`'s table and token
    /// count; every shared block's ref-count += 1. No new physical blocks.
    pub fn fork(&mut self, parent: SeqId, child: SeqId) -> Result<(), BlockError> {
        let _ = (parent, child);
        todo!("Day 2: implement fork")
    }

    pub fn block_table(&self, seq: SeqId) -> Option<&[BlockId]> {
        let _ = seq;
        todo!("Day 2: implement block_table")
    }

    pub fn num_tokens(&self, seq: SeqId) -> Option<usize> {
        let _ = seq;
        todo!("Day 2: implement num_tokens")
    }

    pub fn ref_count(&self, block: BlockId) -> u32 {
        let _ = block;
        todo!("Day 2: implement ref_count")
    }
}
