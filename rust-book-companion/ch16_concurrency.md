# Ch 16 — Fearless Concurrency ⭐

📖 https://doc.rust-lang.org/book/ch16-00-fearless-concurrency.html

Ownership + type system turn data races into **compile errors**. 1:1 OS threads in std (async is Ch 17).

## Spawning threads

```rust
use std::thread;

let handle = thread::spawn(|| {
    for i in 1..10 { println!("spawned: {i}"); }
});
handle.join().unwrap();               // block until it finishes

// Closures that use outer data MUST take ownership:
let v = vec![1, 2, 3];
let handle = thread::spawn(move || println!("{v:?}"));   // `move` required —
// a borrow could outlive `v`; the compiler refuses without move
```

- Main thread exiting kills all spawned threads — `join` what you need finished.

## Message passing — channels

> "Do not communicate by sharing memory; share memory by communicating."

```rust
use std::sync::mpsc;                   // multiple producer, single consumer

let (tx, rx) = mpsc::channel();
thread::spawn(move || {
    tx.send(String::from("hi")).unwrap();   // send MOVES the value —
});                                          // no use-after-send bugs
let received = rx.recv().unwrap();     // blocking; try_recv() = non-blocking

for received in rx { ... }             // rx is an iterator; ends when all tx dropped

let tx2 = tx.clone();                  // multiple producers
```

## Shared state — `Mutex<T>` and `Arc<T>`

```rust
use std::sync::{Arc, Mutex};

let counter = Arc::new(Mutex::new(0));         // Arc = atomic Rc (thread-safe)
let mut handles = vec![];

for _ in 0..10 {
    let counter = Arc::clone(&counter);
    handles.push(thread::spawn(move || {
        let mut num = counter.lock().unwrap(); // returns MutexGuard (smart pointer)
        *num += 1;
    }));                                       // guard dropped → lock released
}
for h in handles { h.join().unwrap(); }
println!("Result: {}", *counter.lock().unwrap());   // 10
```

- You **cannot forget to lock**: the data lives inside the mutex; the only access is through `lock()`.
- `lock()` returns `Err` if another thread panicked holding the lock (poisoning).
- `Mutex` provides interior mutability, like `RefCell` — and like `Rc`-cycles, deadlocks are still possible.
- Also in std: `RwLock` (many readers or one writer), atomics (`AtomicUsize`, ...).

## `Send` and `Sync` — the marker traits that make it "fearless"

| Trait | Meaning |
|---|---|
| `Send` | ownership can be **transferred** to another thread |
| `Sync` | `&T` can be **shared** between threads (T is Sync ⇔ &T is Send) |

- Auto-derived: a type is Send/Sync if all its fields are. Almost everything is; `Rc`, `RefCell`, raw pointers are not.
- This is why `thread::spawn` rejects an `Rc` at **compile time**: `Rc<T>: !Send`. Swap in `Arc` and it compiles.
- Implementing Send/Sync manually is `unsafe` — you're vouching for invariants the compiler can't see.

## Choosing

- Transfer of data / pipeline stages → **channels**.
- Shared counters, caches, shared config → **`Arc<Mutex<T>>`** (or `RwLock`).
- Simple flags/counters → **atomics**.
- Massive parallel compute → rayon (ecosystem); I/O-bound many-connection work → async (Ch 17).
