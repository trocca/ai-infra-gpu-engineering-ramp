# Ch 12 — An I/O Project: Building `minigrep`

📖 https://doc.rust-lang.org/book/ch12-00-an-io-project.html

Builds a tiny `grep`: `cargo run -- searchstring example.txt`. The chapter is really about **project structure and refactoring discipline**.

## Key APIs

```rust
use std::env;
use std::fs;
use std::process;

let args: Vec<String> = env::args().collect();     // args[0] = binary name
let contents = fs::read_to_string(file_path)?;     // whole file → String
let ignore_case = env::var("IGNORE_CASE").is_ok(); // env vars
eprintln!("Problem: {e}");                         // errors → stderr, not stdout
process::exit(1);
```

## The architecture pattern (memorize this)

**Split into `main.rs` + `lib.rs`.** `main` only: parse config, call `lib::run`, handle errors. All logic + tests live in the library.

```rust
// src/main.rs
fn main() {
    let config = Config::build(env::args()).unwrap_or_else(|err| {
        eprintln!("Problem parsing arguments: {err}");
        process::exit(1);
    });
    if let Err(e) = minigrep::run(config) {
        eprintln!("Application error: {e}");
        process::exit(1);
    }
}

// src/lib.rs
pub struct Config { pub query: String, pub file_path: String, pub ignore_case: bool }

impl Config {
    pub fn build(mut args: impl Iterator<Item = String>) -> Result<Config, &'static str> {
        args.next();                                        // skip program name
        let query = args.next().ok_or("Didn't get a query string")?;
        let file_path = args.next().ok_or("Didn't get a file path")?;
        let ignore_case = env::var("IGNORE_CASE").is_ok();
        Ok(Config { query, file_path, ignore_case })
    }
}

pub fn run(config: Config) -> Result<(), Box<dyn Error>> {
    let contents = fs::read_to_string(config.file_path)?;
    let results = if config.ignore_case {
        search_case_insensitive(&config.query, &contents)
    } else {
        search(&config.query, &contents)
    };
    for line in results { println!("{line}"); }
    Ok(())
}

pub fn search<'a>(query: &str, contents: &'a str) -> Vec<&'a str> {
    contents.lines().filter(|line| line.contains(query)).collect()
}
```

## Lessons the chapter teaches

- **Separation of concerns for binaries**: `main.rs` = thin shell; `lib.rs` = testable logic.
- **Test-driven development**: write the failing `search` test first, then implement.
- Lifetime `'a` in `search`: the returned slices borrow from `contents`, not `query` — the annotation states that.
- `Box<dyn Error>` as an easy "any error" return type; `?` everywhere inside.
- `unwrap_or_else` + `process::exit` for clean CLI error UX; `eprintln!` so `cargo run > out.txt` keeps errors visible.
- Ch 13 revision: replace `clone()`-based parsing with the iterator version shown above.

## Gotchas

- `env::args()` panics on invalid Unicode args (use `args_os` if you care).
- Case-insensitive compare via `to_lowercase()` is approximate for full Unicode — fine here.
