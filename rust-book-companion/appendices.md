# Appendices A-E - Rust Reference Notes

[<- Rust companion index](README.md)

## Appendix A - Keywords

Rust keywords are reserved words with language meaning, such as `fn`, `let`, `mut`, `struct`, `enum`, `trait`, `impl`, `match`, `async`, `await`, `unsafe`, and `dyn`.

For this ramp, the load-bearing ones are:

- `mut`: mutation must be explicit;
- `match`: exhaustive branching;
- `trait` / `impl`: shared behavior and implementations;
- `async` / `await`: asynchronous execution;
- `unsafe`: manually upheld invariants;
- `dyn`: dynamic dispatch through trait objects.

## Appendix B - Operators and Symbols

Common symbols:

- `&T`: shared reference;
- `&mut T`: mutable reference;
- `*const T` / `*mut T`: raw pointers;
- `::`: namespace path;
- `?`: propagate `Result` or `Option`;
- `!`: macro call when after an identifier, never type in type position;
- `|x|`: closure parameter list;
- `..`: range or struct update syntax.

## Appendix C - Derivable Traits

Useful derives:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
```

For systems work:

- derive `Debug` on most config/state structs;
- derive `Clone` deliberately, not as a reflex;
- use `Copy` only for small plain values where implicit duplication is harmless;
- derive `PartialEq` for test comparisons.

## Appendix D - Useful Development Tools

Core commands:

```text
cargo check
cargo test
cargo clippy
cargo fmt
cargo doc --open
rustup doc --book
```

For the GPU/Rust weeks, also know where the pinned toolchain lives: `rust-toolchain.toml`.

## Appendix E - Editions

Rust editions let the language evolve without breaking old code. The 2024 edition is current for the third edition of the book, but many crates still use 2021. Match the repo's `Cargo.toml` and toolchain pin before changing editions.

## Quick Checks

- What does `?` do in a function returning `Result`?
- Why is `Copy` not the same thing as `Clone`?
- What is the difference between `&mut T` and `*mut T`?
- Why should you avoid changing a crate edition casually?
