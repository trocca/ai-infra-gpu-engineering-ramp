# Ch 14 — More About Cargo and Crates.io

📖 https://doc.rust-lang.org/book/ch14-00-more-about-cargo.html

## Profiles

```toml
[profile.dev]
opt-level = 0        # default: fast compile, slow code

[profile.release]    # cargo build --release
opt-level = 3        # default: slow compile, fast code
```

(Other useful knobs: `debug`, `lto`, `codegen-units`, `panic = "abort"`.)

## Documentation

```rust
//! # My Crate                       ← inner doc: documents the ENCLOSING item
//! Top-of-crate overview (lib.rs)   (crate/module level)

/// Adds one to the number given.    ← outer doc: documents the NEXT item
///
/// # Examples
///
/// ```
/// let answer = my_crate::add_one(5);
/// assert_eq!(6, answer);           // ← this runs as a DOC TEST in `cargo test`
/// ```
pub fn add_one(x: i32) -> i32 { x + 1 }
```

- `cargo doc --open` builds and opens HTML docs.
- Common doc sections: `# Examples`, `# Panics`, `# Errors`, `# Safety`.
- Doc tests are the killer feature: examples that stop compiling fail CI.

## Re-exports for a friendly API

```rust
// users type art::PrimaryColor instead of art::kinds::PrimaryColor
pub use self::kinds::PrimaryColor;
pub use self::utils::mix;
```

Internal module structure ≠ public API; `pub use` flattens the surface and shows up on the docs front page.

## Publishing to crates.io

```sh
cargo login <token>          # token from crates.io account
cargo publish                # name must be unique; version per SemVer
cargo yank --vers 1.0.1      # block NEW deps on a bad version (doesn't delete!)
cargo yank --vers 1.0.1 --undo
```

Cargo.toml needs `name`, `version`, `edition`, `description`, `license` before publishing. **Publishing is permanent** — versions can't be deleted, only yanked.

## Workspaces — multi-crate repos

```toml
# root Cargo.toml (no [package])
[workspace]
resolver = "3"                    # matches edition 2024
members = ["adder", "add_one"]
```

- One shared `target/` and one `Cargo.lock` at the root → consistent deps, shared build cache.
- Path deps between members: `add_one = { path = "../add_one" }`.
- `cargo build` builds all; `cargo run -p adder`, `cargo test -p add_one` target one member.

## Installing binaries & extending cargo

```sh
cargo install ripgrep        # install a binary crate to ~/.cargo/bin
cargo something              # any binary named cargo-something works as a subcommand
```

## Gotchas

- Workspace members must still be listed even though they're subdirectories — cargo doesn't auto-discover.
- Doc tests only run for **library** crates.
- Yanking doesn't remove code from existing lockfiles — never publish secrets.
