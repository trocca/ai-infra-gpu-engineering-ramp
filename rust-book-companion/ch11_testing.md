# Ch 11 — Writing Automated Tests

📖 https://doc.rust-lang.org/book/ch11-00-testing.html

## Anatomy of a test

```rust
#[cfg(test)]                     // compile this module only for `cargo test`
mod tests {
    use super::*;                // bring parent module's items in

    #[test]
    fn it_adds_two() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }

    #[test]
    fn greeting_contains_name() {
        let result = greeting("Carol");
        assert!(
            result.contains("Carol"),
            "Greeting did not contain name, value was `{result}`"   // custom message
        );
    }

    #[test]
    #[should_panic(expected = "between 1 and 100")]   // expected = substring match
    fn rejects_out_of_range() {
        Guess::new(200);
    }

    #[test]
    fn using_result() -> Result<(), String> {          // tests can return Result
        if add(2, 2) == 4 { Ok(()) } else { Err(String::from("2+2 != 4")) }
        // enables `?` inside tests; can't combine with #[should_panic]
    }

    #[test]
    #[ignore]                    // skip unless `cargo test -- --ignored`
    fn expensive_test() { ... }
}
```

- Assertions: `assert!(bool)`, `assert_eq!(a, b)`, `assert_ne!(a, b)` — the eq/ne macros print both values on failure (types need `PartialEq + Debug`; just `#[derive(PartialEq, Debug)]`).
- A test fails when it panics.

## Running tests

```sh
cargo test                        # all tests, in parallel threads
cargo test add                    # only tests whose name contains "add"
cargo test -- --test-threads=1    # serial (shared state / env vars)
cargo test -- --show-output       # show stdout of PASSING tests too
cargo test -- --ignored           # only ignored tests
cargo test -- --include-ignored   # everything
```

(Args before `--` go to cargo, after `--` go to the test binary.)

## Test organization

| Kind | Where | Can test |
|---|---|---|
| **Unit tests** | same file, `#[cfg(test)] mod tests` | private functions too (child module sees parent) |
| **Integration tests** | `tests/` dir, each file = its own crate | only the public API, `use your_crate::...` |
| **Doc tests** | code blocks in `///` comments | your examples stay correct (Ch 14) |

```text
project/
├── src/lib.rs               # unit tests inside
└── tests/
    ├── integration_test.rs   # a test crate
    └── common/mod.rs         # shared helpers — mod.rs style avoids it
                              # being treated as a test file itself
```

- Integration tests only work against a **library** crate — keep `main.rs` thin, logic in `lib.rs` (the Ch 12 pattern).

## Gotchas

- Tests run in parallel by default — don't share files/env vars between tests, or serialize.
- Printed output from passing tests is captured (hidden) by default.
- `#[cfg(test)]` code is entirely absent from normal builds — zero cost in release binaries.
