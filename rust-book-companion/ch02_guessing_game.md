# Ch 2 — Programming a Guessing Game

📖 https://doc.rust-lang.org/book/ch02-00-programming-a-guessing-game.html

A whirlwind tour project: read input, generate a random number, compare, loop. Introduces concepts that later chapters formalize.

## The complete program (condensed)

```rust
use std::cmp::Ordering;
use std::io;
use rand::Rng;

fn main() {
    let secret_number = rand::rng().random_range(1..=100);

    loop {
        println!("Input your guess:");
        let mut guess = String::new();
        io::stdin()
            .read_line(&mut guess)
            .expect("Failed to read line");

        let guess: u32 = match guess.trim().parse() {
            Ok(num) => num,
            Err(_) => continue,        // ignore non-numbers, ask again
        };

        match guess.cmp(&secret_number) {
            Ordering::Less => println!("Too small!"),
            Ordering::Greater => println!("Too big!"),
            Ordering::Equal => {
                println!("You win!");
                break;
            }
        }
    }
}
```

## Concepts introduced (formalized later)

| Concept | Here | Deep dive |
|---|---|---|
| `let` / `let mut` | variables immutable by default | Ch 3 |
| `String::new()` | growable UTF-8 string | Ch 8 |
| `&mut guess` | mutable reference | Ch 4 |
| `Result` + `.expect()` | recoverable errors; `.expect` panics on `Err` | Ch 9 |
| `match` + `Ordering` | exhaustive pattern matching | Ch 6, 19 |
| **Shadowing** | re-`let` the same name (`guess: u32` shadows the `String`) | Ch 3 |
| `parse()` | string → number, returns `Result`; needs a type annotation | Ch 9 |
| External crates | add `rand = "0.9"` under `[dependencies]` | Ch 14 |

## Gotchas

- `read_line` **appends** to the string and includes the trailing newline — hence `.trim()`.
- `parse()` can't infer the target type by itself; annotate (`let guess: u32`) or turbofish (`parse::<u32>()`).
- Range syntax `1..=100` is inclusive on both ends; `1..100` excludes 100.
- `rand` API (0.9+): `rand::rng().random_range(...)` — older editions of the book used `thread_rng().gen_range(...)`.
