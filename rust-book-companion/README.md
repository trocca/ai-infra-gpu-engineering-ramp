# The Rust Programming Language (3rd Edition) — Companion Reference

Companion notes for **"The Rust Programming Language"** by Steve Klabnik, Carol Nichols, and Chris Krycho — the official Rust book, freely available online.

- **Read online (always current):** https://doc.rust-lang.org/book/
- **Interactive version (with quizzes):** https://rust-book.cs.brown.edu/
- **Local copy (ships with Rust):** run `rustup doc --book`
- **Print:** No Starch Press, 3rd Edition (covers the Rust 2024 edition)

## What's new in the 3rd edition

- Updated for the **Rust 2024 edition** (`edition = "2024"` in Cargo.toml)
- New **Chapter 17: Async and Await** (futures, streams, `async`/`await`)
- Chapters after 16 are renumbered vs. the 2nd edition (OOP is now Ch 18, Patterns Ch 19, Advanced Ch 20, Final Project Ch 21)
- Refreshed examples throughout (newer `rand`, updated compiler output, current idioms)

## How to use this folder

Each `chNN_*.md` file is a condensed reference for one chapter: key concepts, essential syntax, common pitfalls, and a link to the full chapter. These are notes, not a replacement — when something is unclear, follow the link and read the real thing.

| File | Chapter |
|------|---------|
| [ch01_getting_started.md](ch01_getting_started.md) | 1. Getting Started |
| [ch02_guessing_game.md](ch02_guessing_game.md) | 2. Programming a Guessing Game |
| [ch03_common_concepts.md](ch03_common_concepts.md) | 3. Common Programming Concepts |
| [ch04_ownership.md](ch04_ownership.md) | 4. Understanding Ownership |
| [ch05_structs.md](ch05_structs.md) | 5. Using Structs |
| [ch06_enums_matching.md](ch06_enums_matching.md) | 6. Enums and Pattern Matching |
| [ch07_packages_crates_modules.md](ch07_packages_crates_modules.md) | 7. Packages, Crates, and Modules |
| [ch08_collections.md](ch08_collections.md) | 8. Common Collections |
| [ch09_error_handling.md](ch09_error_handling.md) | 9. Error Handling |
| [ch10_generics_traits_lifetimes.md](ch10_generics_traits_lifetimes.md) | 10. Generics, Traits, and Lifetimes |
| [ch11_testing.md](ch11_testing.md) | 11. Writing Automated Tests |
| [ch12_io_project_minigrep.md](ch12_io_project_minigrep.md) | 12. An I/O Project: minigrep |
| [ch13_iterators_closures.md](ch13_iterators_closures.md) | 13. Iterators and Closures |
| [ch14_cargo_crates_io.md](ch14_cargo_crates_io.md) | 14. More About Cargo and Crates.io |
| [ch15_smart_pointers.md](ch15_smart_pointers.md) | 15. Smart Pointers |
| [ch16_concurrency.md](ch16_concurrency.md) | 16. Fearless Concurrency |
| [ch17_async_await.md](ch17_async_await.md) | 17. Async and Await *(new in 3rd ed.)* |
| [ch18_oop_features.md](ch18_oop_features.md) | 18. Object-Oriented Programming Features |
| [ch19_patterns_matching.md](ch19_patterns_matching.md) | 19. Patterns and Matching |
| [ch20_advanced_features.md](ch20_advanced_features.md) | 20. Advanced Features |
| [ch21_final_project_web_server.md](ch21_final_project_web_server.md) | 21. Final Project: Multithreaded Web Server |
| [appendices.md](appendices.md) | Appendices A–E |
| [quick_reference.md](quick_reference.md) | Cross-chapter cheatsheet |
| [further_reading.md](further_reading.md) | After the book: Rust for Rustaceans & free community books |

## Suggested reading order for systems/GPU work

For the NVIDIA ramp (CUDA/FFI-adjacent Rust), the load-bearing chapters are **4 (ownership)**, **10 (traits/lifetimes)**, **15 (smart pointers)**, **16 (concurrency)**, and **20 (unsafe, FFI)**. Chapters 1–3 can be skimmed if you already know a systems language.
