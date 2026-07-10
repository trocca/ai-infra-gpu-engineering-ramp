//! COMPLETE test harness — do not edit. `cargo test` runs this CPU-only.
//!
//! Specifies the continuous-batching contract: FIFO admission under three
//! budgets (slots, prefill tokens, KV blocks), iteration-level join/leave,
//! and the no-block-leak invariant. Red until src/scheduler.rs is
//! implemented (which itself needs src/blocks.rs green first).

use ferrum_serve::blocks::BlockManager;
use ferrum_serve::scheduler::{Request, RequestState, Scheduler};

const NUM_BLOCKS: usize = 8; // 32 KV slots total
const BLOCK_SIZE: usize = 4;

fn sched_with(num_blocks: usize, max_running: usize, max_prefill_tokens: usize) -> Scheduler {
    Scheduler::new(
        BlockManager::new(num_blocks, BLOCK_SIZE),
        max_running,
        max_prefill_tokens,
    )
}

fn sched() -> Scheduler {
    sched_with(NUM_BLOCKS, 16, 2048)
}

fn req(id: u64, prompt_len: usize, max_new: usize) -> Request {
    Request::new(id, vec![7u32; prompt_len], max_new)
}

// ---------------------------------------------------------------- admission

#[test]
fn fifo_admission() {
    let mut s = sched();
    for id in 1..=3 {
        s.add_request(req(id, 6, 8));
    }
    let out = s.schedule();
    assert_eq!(out.prefill, vec![1, 2, 3]);
    assert!(out.decode.is_empty());
    for id in 1..=3 {
        assert_eq!(s.request(id).unwrap().state, RequestState::Running);
    }
}

#[test]
fn max_running_respected() {
    let mut s = sched_with(NUM_BLOCKS, 2, 2048);
    for id in 1..=3 {
        s.add_request(req(id, 6, 8));
    }
    let out = s.schedule();
    assert_eq!(out.prefill, vec![1, 2]);
    assert_eq!(s.num_waiting(), 1);
    assert_eq!(s.num_running(), 2);
}

#[test]
fn prefill_token_budget() {
    let mut s = sched_with(NUM_BLOCKS, 16, 10);
    s.add_request(req(1, 6, 8));
    s.add_request(req(2, 6, 8));
    let out1 = s.schedule();
    assert_eq!(out1.prefill, vec![1], "6 + 6 > 10: request 2 must wait one iteration");
    let out2 = s.schedule();
    assert_eq!(out2.prefill, vec![2]);
    assert_eq!(out2.decode, vec![1]);
}

#[test]
fn block_capacity_gates_admission_fcfs() {
    // If the HEAD of the queue doesn't fit, nothing behind it is admitted
    // either (no skipping -> no starvation of large requests).
    let mut s = sched(); // 8 blocks
    s.add_request(req(1, 6, 8)); // 2 blocks
    s.add_request(req(2, 26, 8)); // 7 blocks: won't fit after request 1
    s.add_request(req(3, 2, 8)); // would fit, but is behind request 2
    let out = s.schedule();
    assert_eq!(out.prefill, vec![1]);
    assert_eq!(s.num_waiting(), 2);
}

// ------------------------------------------------------ continuous batching

#[test]
fn running_requests_move_to_decode() {
    let mut s = sched();
    s.add_request(req(1, 6, 8));
    let out1 = s.schedule();
    assert_eq!(out1.prefill, vec![1]);
    let out2 = s.schedule();
    assert!(out2.prefill.is_empty());
    assert_eq!(out2.decode, vec![1]);
}

#[test]
fn join_mid_flight() {
    // The Orca property: a request arriving while others decode is admitted
    // at the NEXT iteration, not after the "batch" drains.
    let mut s = sched();
    s.add_request(req(1, 6, 8));
    s.schedule(); // 1 prefills
    s.schedule(); // 1 decodes
    s.add_request(req(2, 6, 8)); // arrives mid-flight
    let out = s.schedule();
    assert_eq!(out.prefill, vec![2]);
    assert_eq!(out.decode, vec![1]);
}

#[test]
fn leave_mid_flight_frees_capacity_immediately() {
    // Finishing a request frees its blocks so a waiting request can be
    // admitted at the very next iteration.
    let mut s = sched();
    s.add_request(req(1, 18, 8)); // 5 blocks of the 8-block pool
    s.schedule();
    s.add_request(req(2, 18, 8)); // 5 blocks -> cannot coexist with request 1
    s.finish(1);
    assert_eq!(s.request(1).unwrap().state, RequestState::Finished);
    let out = s.schedule();
    assert_eq!(out.prefill, vec![2]);
    assert!(
        !out.decode.contains(&1),
        "finished request kept decoding"
    );
}

#[test]
fn decode_reserves_one_slot_per_iteration() {
    let mut s = sched();
    s.add_request(req(1, 6, 8));
    s.schedule(); // prefill
    let base = s.block_manager().num_tokens(1).unwrap();
    for i in 1..=3usize {
        s.schedule(); // decode iterations
        assert_eq!(
            s.block_manager().num_tokens(1),
            Some(base + i),
            "each schedule() must reserve exactly one KV slot per running request"
        );
    }
}

#[test]
fn on_token_generated_appends() {
    let mut s = sched();
    s.add_request(req(1, 6, 3));
    s.schedule();
    s.on_token_generated(1, 101);
    s.on_token_generated(1, 102);
    assert_eq!(s.request(1).unwrap().output_token_ids, vec![101, 102]);
}

// -------------------------------------------------------------------- cleanup

#[test]
fn cancel_waiting_request() {
    let mut s = sched();
    s.add_request(req(1, 6, 8));
    s.cancel(1);
    assert_eq!(s.num_waiting(), 0);
    let out = s.schedule();
    assert!(out.prefill.is_empty() && out.decode.is_empty());
    assert_eq!(s.block_manager().num_free_blocks(), NUM_BLOCKS);
}

#[test]
fn cancel_running_request_frees_blocks() {
    let mut s = sched();
    s.add_request(req(1, 6, 8));
    s.schedule();
    assert!(s.block_manager().num_free_blocks() < NUM_BLOCKS);
    s.cancel(1);
    assert_eq!(s.request(1).unwrap().state, RequestState::Cancelled);
    assert_eq!(
        s.block_manager().num_free_blocks(),
        NUM_BLOCKS,
        "cancelled request leaked blocks"
    );
    let out = s.schedule();
    assert!(out.decode.is_empty());
}

#[test]
fn no_leak_after_full_workload() {
    // Churn: admissions, decodes, finishes, a cancellation — the pool ends at
    // 100% and the scheduler ends idle. The server's health depends on this.
    let mut s = sched_with(16, 16, 2048); // roomy pool: 3 live seqs x 3 blocks
    for id in 1..=4 {
        s.add_request(req(id, 5, 4));
    }
    s.cancel(2);
    let mut steps = 0;
    while s.has_work() && steps < 100 {
        let out = s.schedule();
        let ids: Vec<u64> = out.prefill.iter().chain(out.decode.iter()).copied().collect();
        for id in ids {
            s.on_token_generated(id, 7);
            let done = {
                let r = s.request(id).unwrap();
                r.num_output_tokens() >= r.max_new_tokens
            };
            if done {
                s.finish(id);
            }
        }
        steps += 1;
    }
    assert!(steps < 100, "scheduler never drained");
    assert_eq!(s.block_manager().num_free_blocks(), 16, "block leak after workload");
    assert_eq!(s.num_running(), 0);
    assert_eq!(s.num_waiting(), 0);
    for id in [1u64, 3, 4] {
        assert_eq!(s.request(id).unwrap().state, RequestState::Finished);
    }
}
