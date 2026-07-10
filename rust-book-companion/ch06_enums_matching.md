# Ch 6 — Enums and Pattern Matching

📖 https://doc.rust-lang.org/book/ch06-00-enums.html

## Enums — variants can carry data

```rust
enum Message {
    Quit,                       // no data
    Move { x: i32, y: i32 },    // named fields (like a struct)
    Write(String),              // one value
    ChangeColor(i32, i32, i32), // tuple of values
}

impl Message {                  // enums have methods too
    fn call(&self) { ... }
}
```

## `Option<T>` — Rust has no null

```rust
enum Option<T> { Some(T), None }     // in the prelude; Some/None used directly

let some_number = Some(5);
let absent: Option<i32> = None;      // None needs a type annotation
```

- `Option<T>` and `T` are different types — the compiler **forces** you to handle the `None` case before using the value. This eliminates the entire class of null-pointer bugs.

## `match` — exhaustive pattern matching

```rust
fn plus_one(x: Option<i32>) -> Option<i32> {
    match x {
        None => None,
        Some(i) => Some(i + 1),      // `i` binds the inner value
    }
}

match dice_roll {
    3 => do_thing(),
    7 => do_other(),
    other => move_player(other),     // catch-all that binds
    // or:  _ => reroll(),           // catch-all that ignores the value
    // or:  _ => (),                 // do nothing
}
```

- Matches are **exhaustive**: forget a variant and it won't compile. This is the payoff of enums.
- Arms are checked top-down; put catch-alls last.

## `if let` / `let else` — one-pattern sugar

```rust
if let Some(max) = config_max {
    println!("max is {max}");
} else {
    // optional else = the `_` arm
}

// let-else: bind or diverge (return/break/panic) — great for early exits
let Some(value) = maybe_value else {
    return Err("missing".into());
};
// `value` usable from here on
```

## Gotchas

- `match` arms must all return the same type.
- Matching moves non-`Copy` data out of the matched value; match on a reference (`match &val`) or use `ref` bindings to borrow instead.
- Prefer `match` when you need exhaustiveness; `if let` trades that away for brevity.
- Standard tools: `opt.unwrap_or(default)`, `opt.map(|v| ...)`, `opt.ok_or(err)` — often cleaner than explicit `match` (Ch 9, 13).
