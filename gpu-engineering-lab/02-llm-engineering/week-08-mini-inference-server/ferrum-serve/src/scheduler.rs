//! Day 2 — continuous-batching scheduler (the Orca idea: schedule at
//! ITERATION granularity, not request granularity).
//!
//! Pure logic, no GPU. The Scheduler OWNS the BlockManager — a deliberate
//! ownership choice: exactly one mutator, no locks, and the engine reads
//! stats through `block_manager()`. Requests are owned by the scheduler too;
//! `schedule()` hands out SeqIds and callers look up state via `request()`.
//!
//! CONTRACT — `tests/scheduler_test.rs` (COMPLETE, do not edit) pins EXACTLY:
//!
//! `schedule()` is called once per engine iteration and, in this order:
//!   1. DECODE: for every Running request, reserve KV room for the token this
//!      iteration will produce (`block_manager.append_slot`) and push its id
//!      into `decode` (FIFO by admission order).
//!   2. ADMIT (prefill): pop Waiting requests FIFO while ALL hold:
//!        - running_count + admitted_now < max_running
//!        - sum of admitted PROMPT lengths this call <= max_prefill_tokens
//!        - block_manager.can_allocate(prompt_len + 1)   [+1 reserves the
//!          first generated token's slot — the convention used throughout]
//!      Allocate (prompt_len + 1) slots, mark Running, push into `prefill`.
//!      FCFS: if the HEAD of the queue does not fit, STOP admitting — no
//!      skipping. Skipping starves large requests forever.
//!
//! `finish(id)`: state = Finished, remove from running, FREE ITS BLOCKS
//! IMMEDIATELY — that line IS continuous batching: the freed capacity is
//! admittable at the very next iteration, no batch boundary.
//!
//! `cancel(id)`: Waiting -> dequeued; Running -> blocks freed; state =
//! Cancelled either way. Unknown id: no-op.
//!
//! Finished/Cancelled requests STAY in the requests map (the tests — and the
//! server — read their final state via `request(id)` afterwards).
//!
//! BlockManager allocations are keyed by the request's SeqId (the tests read
//! `block_manager().num_tokens(id)` directly).

use crate::blocks::{BlockManager, SeqId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestState {
    Waiting,
    Running,
    Finished,
    Cancelled,
}

#[derive(Debug, Clone)]
pub struct Request {
    pub id: SeqId,
    pub prompt_token_ids: Vec<u32>,
    pub max_new_tokens: usize,
    pub temperature: f64,
    pub state: RequestState,
    pub output_token_ids: Vec<u32>,
}

impl Request {
    pub fn new(id: SeqId, prompt_token_ids: Vec<u32>, max_new_tokens: usize) -> Self {
        Self {
            id,
            prompt_token_ids,
            max_new_tokens,
            temperature: 0.0,
            state: RequestState::Waiting,
            output_token_ids: Vec::new(),
        }
    }

    pub fn num_prompt_tokens(&self) -> usize {
        self.prompt_token_ids.len()
    }

    pub fn num_output_tokens(&self) -> usize {
        self.output_token_ids.len()
    }

    pub fn total_tokens(&self) -> usize {
        self.num_prompt_tokens() + self.num_output_tokens()
    }
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct ScheduleOutput {
    pub prefill: Vec<SeqId>,
    pub decode: Vec<SeqId>,
}

pub struct Scheduler {
    // TODO(Day 2): state you will need —
    //   block_manager: BlockManager        (owned — see module docs)
    //   waiting:  VecDeque<SeqId>          (FIFO admission queue)
    //   running:  Vec<SeqId>               (admission order)
    //   requests: HashMap<SeqId, Request>  (single owner of Request structs)
    //   max_running / max_prefill_tokens
    // Rust note: you cannot hold `&mut self.block_manager` while iterating
    // `&self.running` if both borrows come through methods on self — iterate
    // over a clone of the id list, or split borrows via separate fields.
    // This fight with the borrow checker is the lesson; have it on purpose.
}

impl Scheduler {
    pub fn new(block_manager: BlockManager, max_running: usize, max_prefill_tokens: usize) -> Self {
        let _ = (block_manager, max_running, max_prefill_tokens);
        todo!("Day 2: implement Scheduler::new")
    }

    pub fn add_request(&mut self, req: Request) {
        let _ = req;
        todo!("Day 2: implement add_request")
    }

    pub fn has_work(&self) -> bool {
        todo!("Day 2: implement has_work")
    }

    /// One engine iteration's worth of work. The module docs are the spec.
    ///
    /// Preemption (evicting a running seq when `append_slot` fails) is the
    /// production answer — out of scope here; a panic/expect is acceptable,
    /// just size `max_running` so it cannot trigger.
    pub fn schedule(&mut self) -> ScheduleOutput {
        todo!("Day 2: implement schedule")
    }

    /// The engine calls this after sampling; it only records the token.
    /// Deciding to stop (EOS / max_new_tokens / stop strings) is the ENGINE's
    /// job, which then calls `finish`.
    pub fn on_token_generated(&mut self, id: SeqId, token: u32) {
        let _ = (id, token);
        todo!("Day 2: implement on_token_generated")
    }

    pub fn finish(&mut self, id: SeqId) {
        let _ = id;
        todo!("Day 2: implement finish")
    }

    pub fn cancel(&mut self, id: SeqId) {
        let _ = id;
        todo!("Day 2: implement cancel")
    }

    pub fn num_running(&self) -> usize {
        todo!("Day 2: implement num_running")
    }

    pub fn num_waiting(&self) -> usize {
        todo!("Day 2: implement num_waiting")
    }

    pub fn request(&self, id: SeqId) -> Option<&Request> {
        let _ = id;
        todo!("Day 2: implement request")
    }

    pub fn block_manager(&self) -> &BlockManager {
        todo!("Day 2: implement block_manager")
    }
}
