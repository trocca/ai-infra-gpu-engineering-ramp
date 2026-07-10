# Ch 8 — Common Collections

📖 https://doc.rust-lang.org/book/ch08-00-common-collections.html

## `Vec<T>` — growable array

```rust
let mut v: Vec<i32> = Vec::new();
let v = vec![1, 2, 3];                 // macro with initial values
v.push(4);

let third: &i32 = &v[2];               // panics if out of bounds
let third: Option<&i32> = v.get(2);    // None if out of bounds — choose per situation

for i in &v { println!("{i}"); }       // iterate by reference
for i in &mut v { *i += 50; }          // mutate in place (deref with *)

// Heterogeneous data: wrap in an enum
enum Cell { Int(i32), Float(f64), Text(String) }
let row = vec![Cell::Int(3), Cell::Text(String::from("blue"))];
```

- ⚠️ Classic borrow error: `let first = &v[0]; v.push(6); println!("{first}");` fails — `push` may reallocate, invalidating the reference. The borrow checker catches it.

## `String` — growable UTF-8 text

```rust
let mut s = String::from("foo");
s.push_str("bar");                     // append &str
s.push('!');                           // append char
let s3 = s1 + &s2;                     // + moves s1, borrows s2
let s = format!("{s1}-{s2}-{s3}");     // no ownership taken; usually preferable
```

**Strings are not indexable**: `s[0]` doesn't compile. UTF-8 chars are 1–4 bytes, so byte index ≠ character.

```rust
for c in "Зд".chars() { }              // characters (Unicode scalar values)
for b in "Зд".bytes() { }              // raw bytes
let slice = &hello[0..4];              // byte-range slicing works but PANICS
                                       // if the range splits a character
```

## `HashMap<K, V>`

```rust
use std::collections::HashMap;

let mut scores = HashMap::new();
scores.insert(String::from("Blue"), 10);        // owned keys/values are MOVED in

let score = scores.get("Blue").copied().unwrap_or(0);

for (key, value) in &scores { }

scores.insert(String::from("Blue"), 25);        // insert overwrites

scores.entry(String::from("Yellow")).or_insert(50);   // only if absent

// Update based on old value:
let count = map.entry(word).or_insert(0);
*count += 1;
```

- Default hasher is SipHash — DoS-resistant, not the fastest. Swap hashers via `BuildHasher` if profiling says so.

## Gotchas

- Inserting into a `HashMap`/`Vec` moves non-`Copy` values; the original binding is invalid after.
- `String` vs `&str`: own with `String`, borrow with `&str`; APIs should accept `&str`.
- Also in `std::collections`: `VecDeque`, `BTreeMap` (sorted keys), `HashSet`, `BTreeSet`, `BinaryHeap`.
