# Ch 10 — Generic Types, Traits, and Lifetimes ⭐

📖 https://doc.rust-lang.org/book/ch10-00-generics.html

## Generics

```rust
fn largest<T: PartialOrd>(list: &[T]) -> &T {
    let mut largest = &list[0];
    for item in list {
        if item > largest { largest = item; }
    }
    largest
}

struct Point<T> { x: T, y: T }
impl<T> Point<T> {                      // methods for any T
    fn x(&self) -> &T { &self.x }
}
impl Point<f64> {                       // methods only for Point<f64>
    fn distance_from_origin(&self) -> f64 { (self.x.powi(2) + self.y.powi(2)).sqrt() }
}

enum Option<T> { Some(T), None }        // std types are generic too
```

- **Monomorphization**: the compiler stamps out a concrete copy per type used → zero runtime cost (like C++ templates, but type-checked at definition).

## Traits — shared behavior (like interfaces)

```rust
pub trait Summary {
    fn summarize_author(&self) -> String;          // required
    fn summarize(&self) -> String {                // default impl, can call required
        format!("(Read more from {}...)", self.summarize_author())
    }
}

impl Summary for Tweet {
    fn summarize_author(&self) -> String { format!("@{}", self.username) }
}
```

- **Orphan rule (coherence)**: you can `impl Trait for Type` only if the trait or the type is local to your crate — prevents conflicting impls across crates.

### Trait bounds

```rust
pub fn notify(item: &impl Summary) { ... }                 // impl-Trait sugar
pub fn notify<T: Summary>(item: &T) { ... }                // equivalent generic form
fn f<T: Summary + Display>(t: &T) { ... }                  // multiple bounds
fn g<T, U>(t: &T, u: &U) -> i32                            // where-clause form
where
    T: Display + Clone,
    U: Clone + Debug,
{ ... }

fn returns_summarizable() -> impl Summary { ... }          // return opaque type
                                                           // (single concrete type only)

impl<T: Display> ToString for T { ... }                    // blanket impl (std does this)
impl<T: Display + PartialOrd> Pair<T> {                    // conditional methods
    fn cmp_display(&self) { ... }
}
```

## Lifetimes — validating references

Lifetimes don't change how long anything lives; they **describe relationships** between reference lifetimes so the compiler can verify no reference outlives its data.

```rust
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() { x } else { y }
}
// "returned reference is valid as long as BOTH inputs are valid"
// (concretely: 'a = the smaller of the two input lifetimes)

struct ImportantExcerpt<'a> {          // struct holding a reference
    part: &'a str,                     // instance can't outlive what `part` points to
}

let s: &'static str = "literal";       // 'static = whole program (string literals)
```

### Elision rules (why most functions need no annotations)

1. Each reference parameter gets its own lifetime.
2. If there is exactly **one input lifetime**, it's assigned to all outputs.
3. If a method has `&self`/`&mut self`, **self's lifetime** goes to all outputs.

If the rules can't determine the output lifetime, you must annotate (that's the `longest` case: two inputs, no self).

## All together

```rust
fn longest_with_announcement<'a, T: Display>(x: &'a str, y: &'a str, ann: T) -> &'a str {
    println!("Announcement! {ann}");
    if x.len() > y.len() { x } else { y }
}
```

## Gotchas

- Lifetime errors usually mean the *design* wants owned data — clone or restructure before fighting annotations.
- `'a` on a struct means "this struct borrows"; if it spreads everywhere, consider owning (`String`) instead.
- `impl Trait` in return position ≠ dynamic dispatch — it's still one concrete type; for "either type A or B" you need `Box<dyn Trait>` (Ch 18).
