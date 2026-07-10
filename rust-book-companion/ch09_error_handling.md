# Ch 9 — Error Handling

📖 https://doc.rust-lang.org/book/ch09-00-error-handling.html

Rust splits errors into **unrecoverable** (`panic!`) and **recoverable** (`Result<T, E>`). No exceptions.

## `panic!` — unrecoverable

```rust
panic!("crash and burn");
let v = vec![1, 2, 3];
v[99];                       // also panics (bounds check)
```

- Panics unwind the stack (running destructors) by default; `panic = "abort"` in Cargo.toml for smaller binaries.
- `RUST_BACKTRACE=1 cargo run` for a backtrace.
- Panic when a bug/invariant violation makes continuing meaningless; use `Result` for expected failures.

## `Result<T, E>` — recoverable

```rust
enum Result<T, E> { Ok(T), Err(E) }

let file = match File::open("hello.txt") {
    Ok(f) => f,
    Err(e) => match e.kind() {
        ErrorKind::NotFound => File::create("hello.txt").expect("create failed"),
        other => panic!("open failed: {other:?}"),
    },
};
```

### Shortcuts

```rust
let f = File::open("hello.txt").unwrap();               // panic on Err
let f = File::open("hello.txt").expect("hello.txt should exist");  // panic w/ message (preferred)
let f = File::open("x").unwrap_or_else(|e| ...);        // closure fallback (Ch 13)
```

## `?` — the error-propagation operator

```rust
fn read_username() -> Result<String, io::Error> {
    let mut s = String::new();
    File::open("hello.txt")?.read_to_string(&mut s)?;   // Err → early-return it
    Ok(s)
}
// even shorter: fs::read_to_string("hello.txt")
```

- `?` on `Err(e)` returns `Err(From::from(e))` — auto-converts error types via the `From` trait. This is how `Box<dyn Error>` and custom error enums absorb everything.
- `?` also works on `Option` (propagates `None`), but you can't mix `Result` and `Option` in one function without converting (`ok_or`, `.ok()`).
- `main` can return a `Result`:

```rust
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let f = File::open("hello.txt")?;
    Ok(())
}
```

## To panic or not

- **Panic / expect**: examples, prototypes, tests, and cases where you have more information than the compiler (`"127.0.0.1".parse().expect("hardcoded IP is valid")`).
- **Result**: anything a caller might reasonably want to handle — I/O, parsing, network.
- Encode invariants in **types** so bad states can't be constructed (e.g., a `Guess` struct whose `new(value)` validates 1..=100 once).

## Gotchas

- `?` only works in functions whose return type supports it (`Result`, `Option`, or other `FromResidual` types) — using it in a `()` main is a compile error.
- `unwrap()` in library code is a smell; prefer `expect("why this can't fail")` so panics are self-documenting.
- Ecosystem (beyond the book): `thiserror` for library error enums, `anyhow` for application code.
