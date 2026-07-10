# Ch 3 — Common Programming Concepts

📖 https://doc.rust-lang.org/book/ch03-00-common-programming-concepts.html

## Variables and mutability

```rust
let x = 5;            // immutable by default
let mut y = 5;        // mutable
const MAX_POINTS: u32 = 100_000;   // const: type required, compile-time value, ALL_CAPS

let x = x + 1;        // shadowing: new binding, can change type
let spaces = "   ";
let spaces = spaces.len();   // legal — shadowing; `mut` could NOT change type
```

## Scalar types

| Type | Notes |
|---|---|
| `i8..i128`, `isize` | signed ints; `isize` = pointer-sized |
| `u8..u128`, `usize` | unsigned; `usize` for indexing/lengths |
| `i32` | default integer type |
| `f32`, `f64` | floats; `f64` is default |
| `bool` | `true` / `false` |
| `char` | 4-byte Unicode scalar, single quotes: `'z'`, `'😻'` |

- Literals: `98_222`, `0xff`, `0o77`, `0b1111_0000`, `b'A'` (byte), `57u8` (suffixed)
- **Overflow:** panics in debug builds; wraps (two's complement) in release. Explicit control: `wrapping_add`, `checked_add`, `overflowing_add`, `saturating_add`.
- Integer division truncates toward zero: `-5 / 3 == -1`.

## Compound types

```rust
let tup: (i32, f64, u8) = (500, 6.4, 1);
let (x, y, z) = tup;          // destructuring
let five_hundred = tup.0;     // index access
// () is the "unit" — empty tuple, returned by expressions that return nothing

let a: [i32; 5] = [1, 2, 3, 4, 5];   // fixed length, on the stack
let zeros = [0; 5];                   // [value; count]
let first = a[0];                     // out-of-bounds → runtime panic (checked)
```

## Functions

```rust
fn plus_one(x: i32) -> i32 {
    x + 1        // expression, no semicolon → return value
}                // adding a `;` makes it a statement → returns (), type error
```

- Parameter types are always required; return type after `->`.
- **Statements vs expressions**: blocks `{ ... }` are expressions; their last expression (no `;`) is the block's value.

## Control flow

```rust
let n = if cond { 5 } else { 6 };   // if is an expression; branches must be same type
// condition must be `bool` — no truthiness, `if x` where x: i32 won't compile

loop { break value; }                // loop can return a value via break
'outer: loop { loop { break 'outer; } }   // loop labels

while cond { ... }
for element in a { ... }             // preferred for collections
for i in (1..4).rev() { ... }        // ranges + iterator adapters
```

## Gotchas

- No implicit numeric conversions — `let x: u64 = some_u32;` fails; use `as` or `From`/`TryFrom`.
- `if` branches must produce the same type when used as an expression.
- Array indexing is bounds-checked at runtime (panic, not UB) — one of Rust's memory-safety guarantees.
