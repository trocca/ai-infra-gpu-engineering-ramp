# Further Reading — After the Rust Book

[<- Rust companion index](README.md)

Where to go once the 21 chapters are done, roughly in order of usefulness for the ramp.

## Rust for Rustaceans — Jon Gjengset (No Starch Press, 2021) 📕 paid

The canonical "book two." Written explicitly as the next step after The Rust Programming Language — assumes you know the syntax and teaches how experienced Rust developers actually design code.

- **Not a community book**: commercial and copyrighted, no free online version. Buy: https://nostarch.com/rust-rustaceans (ebook is DRM-free from No Starch, so it goes straight onto a Kindle).
- **What it covers** (public ToC): memory layout in depth, API design (the `#[non_exhaustive]`/sealed-trait/builder toolbox), error handling design, project structure, testing beyond the basics, macros, async internals, `unsafe` correctly, concurrency (data + lock-free), FFI, `no_std`, and the ecosystem.
- **Highest-value chapters for GPU/systems work**: Ch 1–2 (memory, types), Ch 9 (unsafe), Ch 10 (concurrency), Ch 11 (FFI).
- Note: written pre-2024-edition (Rust ~2021), so cross-check `unsafe extern` / `&raw` syntax against [ch20_advanced_features.md](ch20_advanced_features.md).
- Same author, **free** companion material: the *Crust of Rust* YouTube series (deep dives on lifetimes, atomics, channels, async) — https://www.youtube.com/@jonhoo

## Free / community books (open like TRPL)

| Resource | What | Link |
|---|---|---|
| Rustonomicon | The official unsafe-Rust book — pairs with Ch 20 | https://doc.rust-lang.org/nomicon/ |
| Rust Atomics and Locks — Mara Bos | Low-level concurrency (atomics, memory ordering, building locks); O'Reilly book, **free online** | https://marabos.nl/atomics/ |
| Rust by Example | Runnable snippets per topic | https://doc.rust-lang.org/rust-by-example/ |
| Async Book | Official async deep dive beyond Ch 17 | https://rust-lang.github.io/async-book/ |
| Rust Reference | The language spec | https://doc.rust-lang.org/reference/ |
| Rust API Guidelines | How std-quality APIs are designed | https://rust-lang.github.io/api-guidelines/ |
| Rustlings | Exercise track that mirrors TRPL chapters | https://github.com/rust-lang/rustlings |
| The Performance Book | Profiling and optimizing Rust | https://nnethercote.github.io/perf-book/ |

## Suggested sequence for the ramp

1. Finish TRPL (this companion) → 2. *Rust Atomics and Locks* alongside Ch 16 work → 3. *Rust for Rustaceans* for API/unsafe/FFI maturity → 4. Rustonomicon only when actually writing nontrivial `unsafe`.
