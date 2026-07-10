//! COMPLETE test harness — do not edit. `cargo test` runs this CPU-only.
//!
//! Red→green: every test panics with "not yet implemented" until you replace
//! the `todo!()`s in src/blocks.rs. The suite IS the spec — read a failing
//! test before writing code.

use ferrum_serve::blocks::{BlockError, BlockManager};

const NUM_BLOCKS: usize = 8; // tiny pool so exhaustion is easy to hit
const BLOCK_SIZE: usize = 4;

fn bm() -> BlockManager {
    BlockManager::new(NUM_BLOCKS, BLOCK_SIZE)
}

#[test]
fn starts_with_all_blocks_free() {
    let m = bm();
    assert_eq!(m.num_free_blocks(), NUM_BLOCKS);
}

#[test]
fn blocks_needed_is_ceil_division() {
    let m = bm();
    assert_eq!(m.blocks_needed(1), 1);
    assert_eq!(m.blocks_needed(4), 1);
    assert_eq!(m.blocks_needed(5), 2);
    assert_eq!(m.blocks_needed(17), 5);
}

#[test]
fn allocate_returns_table_and_decrements_pool() {
    let mut m = bm();
    let table = m.allocate(1, 10).unwrap(); // 10 tokens -> 3 blocks of 4
    assert_eq!(table.len(), 3);
    let mut sorted = table.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted.len(), 3, "blocks in a table must be distinct");
    assert!(table.iter().all(|&b| (b as usize) < NUM_BLOCKS));
    assert_eq!(m.num_free_blocks(), NUM_BLOCKS - 3);
    assert_eq!(m.block_table(1).unwrap(), table.as_slice());
    assert_eq!(m.num_tokens(1), Some(10));
}

#[test]
fn two_sequences_get_disjoint_blocks() {
    let mut m = bm();
    let t1 = m.allocate(1, 8).unwrap();
    let t2 = m.allocate(2, 8).unwrap();
    assert!(t1.iter().all(|b| !t2.contains(b)));
}

#[test]
fn double_allocate_errors() {
    let mut m = bm();
    m.allocate(1, 4).unwrap();
    assert_eq!(m.allocate(1, 4), Err(BlockError::AlreadyAllocated(1)));
}

#[test]
fn can_allocate_and_exhaustion() {
    let mut m = bm();
    m.allocate(1, 6 * BLOCK_SIZE).unwrap(); // 6 of 8 blocks
    assert!(m.can_allocate(2 * BLOCK_SIZE));
    assert!(!m.can_allocate(3 * BLOCK_SIZE));
    assert_eq!(
        m.allocate(2, 3 * BLOCK_SIZE),
        Err(BlockError::OutOfBlocks)
    );
}

#[test]
fn append_slot_within_block_returns_none() {
    let mut m = bm();
    m.allocate(1, 2).unwrap(); // 1 block holding 2/4 tokens
    let free_before = m.num_free_blocks();
    assert_eq!(m.append_slot(1).unwrap(), None); // token 3 fits
    assert_eq!(m.append_slot(1).unwrap(), None); // token 4 fits (block now full)
    assert_eq!(m.num_free_blocks(), free_before);
    assert_eq!(m.num_tokens(1), Some(4));
}

#[test]
fn append_slot_at_boundary_allocates_new_block() {
    let mut m = bm();
    m.allocate(1, BLOCK_SIZE).unwrap(); // exactly one full block
    let free_before = m.num_free_blocks();
    let new_block = m.append_slot(1).unwrap();
    assert!(new_block.is_some(), "full tail block must trigger a new block");
    assert_eq!(m.num_free_blocks(), free_before - 1);
    assert_eq!(m.block_table(1).unwrap().last(), new_block.as_ref());
    assert_eq!(m.num_tokens(1), Some(BLOCK_SIZE + 1));
}

#[test]
fn append_slot_errors_when_pool_empty() {
    let mut m = bm();
    m.allocate(1, NUM_BLOCKS * BLOCK_SIZE).unwrap(); // whole pool, all full
    assert_eq!(m.append_slot(1), Err(BlockError::OutOfBlocks));
}

#[test]
fn append_slot_unknown_seq_errors() {
    let mut m = bm();
    assert_eq!(m.append_slot(99), Err(BlockError::UnknownSeq(99)));
}

#[test]
fn free_returns_blocks_and_allows_reuse() {
    let mut m = bm();
    m.allocate(1, 12).unwrap();
    m.free(1);
    assert_eq!(m.num_free_blocks(), NUM_BLOCKS, "block leak after free");
    let table = m.allocate(2, NUM_BLOCKS * BLOCK_SIZE).unwrap(); // needs everything
    assert_eq!(table.len(), NUM_BLOCKS);
}

#[test]
fn free_is_idempotent() {
    let mut m = bm();
    m.allocate(1, 4).unwrap();
    m.free(1);
    m.free(1); // second free: no-op
    m.free(42); // unknown seq: no-op
    assert_eq!(m.num_free_blocks(), NUM_BLOCKS);
}

#[test]
fn no_leak_after_churn() {
    // Allocate/append/free interleaved — the pool must come back to 100%.
    // This is the invariant the server's cancellation path depends on.
    let mut m = bm();
    for _ in 0..5 {
        m.allocate(1, 5).unwrap();
        m.allocate(2, 3).unwrap();
        for _ in 0..=BLOCK_SIZE {
            m.append_slot(2).unwrap();
        }
        m.free(1);
        m.allocate(3, 8).unwrap();
        m.free(3);
        m.free(2);
    }
    assert_eq!(m.num_free_blocks(), NUM_BLOCKS);
}

// ------------------------------------------------------- ref-counts / fork

#[test]
fn fresh_blocks_have_refcount_one() {
    let mut m = bm();
    let table = m.allocate(1, 8).unwrap();
    for &b in &table {
        assert_eq!(m.ref_count(b), 1);
    }
}

#[test]
fn fork_shares_blocks_and_increments_refcounts() {
    let mut m = bm();
    let table = m.allocate(1, 8).unwrap(); // 2 blocks
    let free_before = m.num_free_blocks();
    m.fork(1, 2).unwrap();
    assert_eq!(m.num_free_blocks(), free_before, "fork must not allocate");
    assert_eq!(m.block_table(2).unwrap(), table.as_slice());
    assert_eq!(m.num_tokens(2), Some(8));
    for &b in &table {
        assert_eq!(m.ref_count(b), 2);
    }
}

#[test]
fn fork_unknown_parent_errors() {
    let mut m = bm();
    assert_eq!(m.fork(7, 8), Err(BlockError::UnknownSeq(7)));
}

#[test]
fn free_parent_keeps_shared_blocks_alive() {
    let mut m = bm();
    let table = m.allocate(1, 8).unwrap(); // 2 blocks
    m.fork(1, 2).unwrap();
    m.free(1);
    assert_eq!(
        m.num_free_blocks(),
        NUM_BLOCKS - 2,
        "shared blocks must survive until every holder frees them"
    );
    for &b in &table {
        assert_eq!(m.ref_count(b), 1);
    }
    m.free(2);
    assert_eq!(m.num_free_blocks(), NUM_BLOCKS, "block leak after last free");
}
