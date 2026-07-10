# Ch 17 — Async and Await 🆕 (new in the 3rd edition)

📖 https://doc.rust-lang.org/book/ch17-00-async-await.html

Concurrency for **I/O-bound** work: many concurrent operations on few threads. The book uses the `trpl` crate (a thin teaching wrapper re-exporting tokio + futures pieces).

## Futures and async/await

```rust
async fn page_title(url: &str) -> Option<String> {
    let response = trpl::get(url).await;          // .await = yield until ready
    let text = response.text().await;
    Html::parse(&text)
        .select_first("title")
        .map(|t| t.inner_html())
}
// an `async fn` RETURNS a Future — the body doesn't run until awaited
```

- A `Future` is **lazy** and inert: it does nothing until polled by a runtime.
- `.await` is postfix → chains nicely: `get(url).await.text().await`.
- Each await point is where the task can pause and the runtime can run other tasks (cooperative scheduling).
- `main` can't be async in plain Rust — a **runtime** (executor) must drive the top-level future:

```rust
fn main() {
    trpl::run(async {                 // real world: #[tokio::main]
        ...
    });
}
```

## Concurrency with async

```rust
// spawn a task (like thread::spawn, but a task, not an OS thread)
let handle = trpl::spawn_task(async { ... });
handle.await.unwrap();

// join: run futures concurrently, wait for ALL (single task, no spawn needed)
let (a, b) = trpl::join(fut1, fut2).await;
trpl::join_all(vec_of_futures).await;      // homogeneous collection
trpl::join!(fut1, fut2, fut3);             // macro, mixed types

// race: first to finish wins, the rest are DROPPED (cancelled)
trpl::race(fut1, fut2).await;

// async channels
let (tx, mut rx) = trpl::channel();
tx.send(val).unwrap();
while let Some(msg) = rx.recv().await { ... }

// yield control voluntarily (long CPU stretches starve other tasks)
trpl::yield_now().await;
```

- **Cancellation = dropping a future.** Anything not yet completed just stops at its last await point. This is how `race`/timeouts work.

## Streams — async iterators

```rust
use trpl::StreamExt;                        // brings next() etc.

let mut stream = trpl::stream_from_iter(iter);
while let Some(value) = stream.next().await { ... }

// stream adapters mirror Iterator: map, filter, throttle, timeout, merge...
let merged = stream1.merge(stream2).throttle(Duration::from_millis(100));
```

## Under the hood (17.5)

- `async` blocks compile to **state machines** implementing `Future`; each `.await` is a state transition.
- `poll(...) -> Poll<T>` (`Ready(T)` / `Pending`); the runtime re-polls when woken by a `Waker`.
- `Pin<Box<T>>` / pinning: self-referential state machines must not move in memory once polled — why some APIs demand `Pin`, and why `join_all` needs `Box::pin(fut)` for mixed types.
- Traits involved: `Future`, `Stream`, plus `Unpin` (marker: safe to move even when "pinned" — most types are Unpin).

## Threads vs async

| | OS threads (Ch 16) | async tasks |
|---|---|---|
| Best for | CPU-bound, blocking work | I/O-bound, many concurrent waits |
| Cost | ~MB stack each, OS scheduling | ~bytes per task, millions possible |
| Preemption | preemptive | cooperative (must hit an await) |

They compose: runtimes are multithreaded; you can `spawn_blocking` CPU work from async code.

## Gotchas

- Blocking calls (`std::thread::sleep`, sync file I/O) inside async code **stall the whole executor thread** — use async equivalents.
- Futures do nothing unawaited — an unused `async fn` call is a silent no-op (compiler warns).
- In production code, replace `trpl` with **tokio** (`#[tokio::main]`, `tokio::spawn`, `tokio::sync::mpsc`) — the concepts map one-to-one.
