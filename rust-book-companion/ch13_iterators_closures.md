# Ch 13 — Functional Features: Iterators and Closures

📖 https://doc.rust-lang.org/book/ch13-00-functional-features.html

## Closures

```rust
let expensive = |num: u32| -> u32 { num + 1 };   // full annotation (rarely needed)
let add_one = |x| x + 1;                          // types inferred from first use

let list = vec![1, 2, 3];
let borrows = || println!("{list:?}");            // captures by shared borrow
let mut mutates = || list.push(4);                // captures by &mut
let owns = move || println!("{list:?}");          // `move` forces taking ownership
                                                  // (needed for threads / returning closures)
```

### The three closure traits (compiler picks automatically)

| Trait | Body does | Callable |
|---|---|---|
| `FnOnce` | may consume captures | once (all closures are at least FnOnce) |
| `FnMut` | mutates captures | many times |
| `Fn` | only reads (or captures nothing) | many times, concurrently |

- APIs: `unwrap_or_else(f)` takes `FnOnce`; `sort_by_key(f)` takes `FnMut` (called per element).
- `move` affects **how it captures**, not which trait it implements.

## Iterators

```rust
pub trait Iterator {
    type Item;                                    // associated type (Ch 20)
    fn next(&mut self) -> Option<Self::Item>;     // the only required method
}
```

- Iterators are **lazy** — nothing happens until a consumer runs.

```rust
let v1 = vec![1, 2, 3];
v1.iter()          // Iterator<Item = &T>      — borrow
v1.iter_mut()      // Iterator<Item = &mut T>  — mutable borrow
v1.into_iter()     // Iterator<Item = T>       — take ownership
```

### Adapters (lazy) → Consumers (drive the iteration)

```rust
let total: i32 = v1.iter().sum();                          // consumer
let doubled: Vec<i32> = v1.iter().map(|x| x * 2).collect(); // adapter + consumer
let evens: Vec<_> = v1.into_iter().filter(|x| x % 2 == 0).collect();

// Everyday adapters: map, filter, filter_map, enumerate, zip, take, skip,
//                    rev, chain, flat_map, peekable
// Everyday consumers: collect, sum, count, fold, for_each, any, all,
//                     find, position, max_by_key, last
```

- `collect()` needs a target type: `let v: Vec<i32> = ...collect();` or `collect::<Vec<i32>>()`.
- `collect()` can also build `String`, `HashMap<K,V>` (from pairs), even `Result<Vec<T>, E>` from an iterator of `Result`s.

## Performance

Iterators are a **zero-cost abstraction** — they compile to the same code as (often better than) hand-written loops, unlocking unrolling and SIMD. The book benchmarks iterator vs loop versions of `search`: identical. Prefer iterator chains; they eliminate index-arithmetic bugs.

## Gotchas

- Adapter chains do nothing without a consumer (`.map(...)` alone → warning: unused iterator).
- `iter()` gives `&T` — inside `filter` closures you often see `|&x|` or `|x| **x` patterns; let the compiler errors guide the deref.
- `for x in collection` implicitly calls `into_iter()` — loop over `&collection` to keep it usable afterward.
