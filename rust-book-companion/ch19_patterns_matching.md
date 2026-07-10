# Ch 19 — Patterns and Matching

📖 https://doc.rust-lang.org/book/ch19-00-patterns.html

(Was Ch 18 in the 2nd edition.)

## Everywhere patterns appear

```rust
match value { PATTERN => expr, ... }         // must be exhaustive
if let PATTERN = value { }  else { }         // single pattern, may miss cases
while let Some(x) = stack.pop() { }          // loop while it matches
for (i, v) in v.iter().enumerate() { }       // for-loop binding is a pattern
let (x, y, z) = tuple;                       // let IS a pattern
let Point { x, y } = point;
fn print_coords(&(x, y): &(i32, i32)) { }    // function params are patterns
let Some(v) = maybe else { return };         // let-else
```

- **Irrefutable** patterns always match (`let x`, `let (a, b)`); **refutable** ones can fail (`Some(x)`). `let`/`for`/params require irrefutable; `if let`/`while let`/`let else` want refutable.

## Pattern syntax toolbox

```rust
match x {
    1 | 2 => ...,                     // or
    3..=5 => ...,                     // inclusive range (numbers, chars)
    'a'..='j' => ...,
    _ => ...,
}

// Destructuring
let Point { x, y } = p;               // shorthand when names match
let Point { x: a, y: 0 } = p;         // rename + literal constraint
match p {
    Point { x, y: 0 } => "on x axis",
    Point { x: 0, y } => "on y axis",
    Point { x, y } => "elsewhere",
}
match msg {                            // enums, incl. nested
    Message::Move { x, y } => ...,
    Message::ChangeColor(Color::Hsv(h, s, v)) => ...,
}
let ((feet, inches), Point { x, y }) = ((3, 10), Point { x: 3, y: -10 });

// Ignoring
_           // ignore value (doesn't bind, doesn't move)
_name       // bind but silence unused warning (DOES move/borrow!)
..          // "the rest": Point { x, .. }   (first, .., last)

// Bindings and guards
Message::Hello { id: id_var @ 3..=7 } => ...,   // @ = test AND bind
Some(x) if x % 2 == 0 => ...,                    // match guard (extra if)
4 | 5 | 6 if y => ...,                           // guard applies to the whole or-chain
```

## Match ergonomics (why you rarely write `ref`)

Matching on a **reference** automatically binds by reference:

```rust
let opt: &Option<String> = &maybe_name;
match opt {
    Some(name) => ...,      // name: &String — no move, no `ref` keyword needed
    None => ...,
}
```

## Gotchas

- `Some(y)` inside a match **shadows** an outer `y` — the pattern variable is new, not a comparison with the outer one. To compare against a variable, use a match guard: `Some(n) if n == y`.
- `_x` still binds (moves!) the value; bare `_` doesn't.
- Match guards are not counted for exhaustiveness — you'll still need a catch-all.
- `..` must be unambiguous: `(.., mid, ..)` is a compile error.
