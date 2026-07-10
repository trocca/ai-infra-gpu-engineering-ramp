# Ch 7 — Managing Growing Projects: Packages, Crates, and Modules

📖 https://doc.rust-lang.org/book/ch07-00-managing-growing-projects-with-packages-crates-and-modules.html

## The hierarchy

| Term | Meaning |
|---|---|
| **Package** | A Cargo.toml + one or more crates. `cargo new` makes one. |
| **Crate** | The compilation unit. **Binary** crate (has `main`) or **library** crate. A package: ≤1 library crate, any number of binaries. |
| **Crate root** | `src/main.rs` (binary) / `src/lib.rs` (library) — the file the compiler starts from. |
| **Module** | Namespacing + privacy boundary *within* a crate. |
| **Path** | How you name an item: `crate::garden::vegetables::Asparagus`. |

## Module declaration & file layout

```rust
// src/lib.rs (crate root)
mod front_of_house;          // loads from src/front_of_house.rs
                             //         or src/front_of_house/mod.rs (older style)

// src/front_of_house.rs
pub mod hosting;             // loads from src/front_of_house/hosting.rs

// src/front_of_house/hosting.rs
pub fn add_to_waitlist() {}
```

- `mod x;` is a **declaration** (compiler goes to find the file) — not an "include" you put in every file, and not an import.

## Paths and privacy

```rust
crate::front_of_house::hosting::add_to_waitlist();   // absolute path
front_of_house::hosting::add_to_waitlist();          // relative path
super::deliver_order();                              // parent module
```

- **Everything is private by default.** `pub` exposes an item to parent modules.
- Child modules can see ancestors' items; parents can't see into children without `pub`.
- `pub struct` needs `pub` on each field individually; `pub enum` makes all variants public.

## `use` — bring paths into scope

```rust
use crate::front_of_house::hosting;      // idiom: import the PARENT for functions
hosting::add_to_waitlist();              // → clear it's not local

use std::collections::HashMap;           // idiom: import the item itself for types

use std::fmt::Result;
use std::io::Result as IoResult;         // `as` to disambiguate

pub use crate::front_of_house::hosting;  // re-export: external code can use your::hosting

use std::{cmp::Ordering, io};            // nested imports
use std::io::{self, Write};              // self = std::io itself + Write
use std::collections::*;                 // glob — mainly for tests/preludes
```

## Gotchas

- `use` only creates a shortcut in the scope where it appears — inner modules need their own `use` (or `super::`).
- The module tree is built from the crate root, **not** the filesystem — a file becomes a module only when some `mod` declaration references it.
- In a package with both `src/main.rs` and `src/lib.rs`, the binary uses the library like any external crate: `use my_package_name::thing;`.
