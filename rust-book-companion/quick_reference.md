# Rust Quick Reference for the NVIDIA Ramp

[<- Rust companion index](README.md)

## Week-Relevant Chapter Map

| Need | Read |
|------|------|
| First Rust syntax | [ch03_common_concepts.md](ch03_common_concepts.md) |
| Ownership and borrowing | [ch04_ownership.md](ch04_ownership.md) |
| Structs and state | [ch05_structs.md](ch05_structs.md) |
| Errors and `Result` | [ch09_error_handling.md](ch09_error_handling.md) |
| Traits and lifetimes | [ch10_generics_traits_lifetimes.md](ch10_generics_traits_lifetimes.md) |
| Tests | [ch11_testing.md](ch11_testing.md) |
| Iterators and closures | [ch13_iterators_closures.md](ch13_iterators_closures.md) |
| Smart pointers and shared state | [ch15_smart_pointers.md](ch15_smart_pointers.md) |
| Concurrency | [ch16_concurrency.md](ch16_concurrency.md) |
| Async serving | [ch17_async_await.md](ch17_async_await.md) |
| Unsafe and FFI | [ch20_advanced_features.md](ch20_advanced_features.md) |
| Server architecture | [ch21_final_project_web_server.md](ch21_final_project_web_server.md) |

## Ownership Cheat Sheet

```rust
let x = String::from("gpu"); // x owns the allocation
let y = x;                   // ownership moves to y
// x is no longer valid

let s = String::from("cuda");
let r = &s;                  // shared borrow
println!("{r}");
```

Borrowing rules:

- many shared references, or one mutable reference;
- references must not outlive the value they point to;
- mutation must be explicit.

## Error Handling

Use `Result<T, E>` for recoverable errors:

```rust
fn load_config(path: &str) -> Result<String, std::io::Error> {
    std::fs::read_to_string(path)
}
```

Use `?` to return early on error:

```rust
let text = std::fs::read_to_string(path)?;
```

## Shared State

- `Box<T>`: heap allocation with one owner.
- `Rc<T>`: shared ownership in single-threaded code.
- `Arc<T>`: shared ownership across threads/tasks.
- `Mutex<T>`: interior mutability with locking.

For async GPU serving, expect `Arc<Engine>` and channels more often than global mutable state.

## Traits

Traits define behavior:

```rust
trait Scheduler {
    fn admit(&mut self, request: Request) -> Decision;
}
```

Use traits when you need interchangeable policies, test doubles, or a clean boundary.

## Unsafe Boundary Rule

Write the invariant before the unsafe block:

```rust
// SAFETY: device_ptr is valid for len elements and outlives the returned slice.
unsafe { std::slice::from_raw_parts(device_ptr, len) }
```

If you cannot state the invariant, you are not ready to write the unsafe block.

## Async Serving Reminders

- `async fn` returns a future.
- `.await` yields control while waiting.
- Do not block the runtime with long CPU/GPU orchestration work.
- Use bounded queues when overload should backpressure instead of silently grow.

## Cargo Commands

```text
cargo check
cargo test
cargo clippy -- -D warnings
cargo fmt
cargo run --release
```

Use the repo's `Makefile` when present; it encodes the week contract.
