# Ch 4 — Understanding Ownership ⭐ (the most important chapter)

📖 https://doc.rust-lang.org/book/ch04-00-understanding-ownership.html

## The three ownership rules

1. Each value has exactly **one owner**.
2. There can only be one owner at a time.
3. When the owner goes out of scope, the value is **dropped** (destructor runs, memory freed).

## Moves vs copies

```rust
let s1 = String::from("hello");
let s2 = s1;              // MOVE: s1 is now invalid (heap data not copied)
// println!("{s1}");      // ❌ compile error: value borrowed after move

let x = 5;
let y = x;                // COPY: i32 is `Copy`, both remain valid

let s3 = s2.clone();      // deep copy, explicit and visible
```

- Types that are `Copy`: all integers, floats, `bool`, `char`, tuples of `Copy` types, shared references `&T`. Anything with heap data / a `Drop` impl is not.
- Passing a value to a function moves it (or copies if `Copy`); returning moves ownership out.

## References and borrowing

```rust
fn calculate_length(s: &String) -> usize { s.len() }   // borrow, don't take ownership
let len = calculate_length(&s);

fn change(s: &mut String) { s.push_str(", world"); }   // mutable borrow
change(&mut s);
```

**The borrowing rules** (enforced at compile time):

> At any given time you may have **either** any number of immutable references (`&T`) **or** exactly one mutable reference (`&mut T`) — never both. References must never outlive the data they point to.

```rust
let mut s = String::from("hi");
let r1 = &s;
let r2 = &s;          // ✅ many shared borrows
// let r3 = &mut s;   // ❌ can't mutably borrow while shared borrows are live
println!("{r1} {r2}");
let r3 = &mut s;      // ✅ OK — r1/r2's last use was above (non-lexical lifetimes)
```

- Dangling references are impossible: returning `&local_var` from a function is a compile error.

## Slices

```rust
let s = String::from("hello world");
let hello: &str = &s[0..5];       // string slice — borrowed view
let word = s.split_whitespace().next().unwrap();

fn first_word(s: &str) -> &str { ... }   // take &str, not &String — works for both

let a = [1, 2, 3, 4, 5];
let slice: &[i32] = &a[1..3];     // array/vec slice
```

- `&str` is the type of string literals and string slices; prefer `&str` parameters over `&String`.
- Slices are borrows, so the borrow rules apply — you can't `s.clear()` while a slice of `s` is live.

## Mental model (stack vs heap)

- A `String` is a stack struct `{ ptr, len, capacity }` pointing to heap bytes. Move = copy the stack struct + invalidate the source. No garbage collector, no manual `free` — `Drop` at scope end, deterministically.

## Gotchas

- "Value borrowed after move" and "cannot borrow as mutable because it is also borrowed as immutable" are the two errors you'll fight most — both are this chapter.
- Iterating `for x in v` **moves** the vector; use `for x in &v` or `&mut v`.
- The borrow checker tracks borrows to their **last use**, not end of scope (NLL) — code that looks illegal may compile fine.
