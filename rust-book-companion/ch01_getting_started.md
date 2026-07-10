# Ch 1 — Getting Started

📖 https://doc.rust-lang.org/book/ch01-00-getting-started.html

## Installation

```sh
# Linux/macOS/WSL2
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
# Windows: download rustup-init.exe from https://rustup.rs
```

- `rustup` manages toolchains: `rustup update`, `rustup self uninstall`, `rustup doc` (offline docs)
- Check versions: `rustc --version`, `cargo --version`

## Hello, World

```rust
fn main() {
    println!("Hello, world!");
}
```

- `main` is always the entry point.
- `println!` is a **macro** (the `!` marks macros; they differ from functions).
- Compile + run manually: `rustc main.rs && ./main` — but real projects use Cargo.

## Cargo essentials

```sh
cargo new hello_cargo      # new project (binary crate, git repo, Cargo.toml)
cargo build                # compile → target/debug/
cargo run                  # compile + run
cargo check                # type-check without producing a binary (fast; use constantly)
cargo build --release      # optimized build → target/release/
```

`Cargo.toml`:

```toml
[package]
name = "hello_cargo"
version = "0.1.0"
edition = "2024"        # 3rd edition of the book targets the 2024 edition

[dependencies]
```

## Gotchas / notes

- `cargo check` is much faster than `cargo build` — the standard inner-loop command.
- `Cargo.lock` pins exact dependency versions; commit it for binaries.
- Rust style: 4-space indent, `snake_case` names, `rustfmt` (`cargo fmt`) settles all formatting arguments.
