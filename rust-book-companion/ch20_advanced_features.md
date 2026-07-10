# Chapter 20 - Advanced Features

[<- Rust companion index](README.md)

## Why this chapter matters here

For the NVIDIA ramp, this is the Rust chapter that keeps kernel hosts, FFI, and low-level performance work honest. You do not need to write large amounts of `unsafe`, but you do need to recognize where Rust's guarantees stop and where you must manually preserve them.

## Unsafe Rust

`unsafe` unlocks five extra abilities:

- dereference raw pointers;
- call unsafe functions or methods;
- access or modify mutable statics;
- implement unsafe traits;
- access fields of unions.

`unsafe` does not turn off the borrow checker everywhere. It creates a small region where you promise the compiler that invariants are still true.

Good rule for this repo: keep unsafe blocks tiny, wrap them in safe APIs, and document the invariant.

Example invariant:

```rust
// SAFETY: ptr comes from a CUDA allocation that is valid for len f32 values.
unsafe {
    let slice = std::slice::from_raw_parts(ptr, len);
}
```

## Advanced Traits

Associated types name placeholder types inside a trait:

```rust
trait Iterator {
    type Item;
    fn next(&mut self) -> Option<Self::Item>;
}
```

Default generic type parameters make operator overloading ergonomic:

```rust
trait Add<Rhs = Self> {
    type Output;
    fn add(self, rhs: Rhs) -> Self::Output;
}
```

Use fully qualified syntax when method names collide:

```rust
<Type as Trait>::method(&value)
```

## Advanced Types

Type aliases improve readability but do not create new types:

```rust
type Result<T> = std::result::Result<T, MyError>;
```

The never type `!` means "this never returns." It appears in panics, infinite loops, and process exits.

Dynamically sized types like `str` and slices must live behind a pointer such as `&str`, `Box<str>`, or `&[T]`.

## Advanced Functions and Closures

Function pointers use `fn`, while closures use one of the `Fn`, `FnMut`, or `FnOnce` traits. In GPU-serving Rust, this matters when you pass callbacks, handlers, or scheduling policies around.

## Macros

Macros generate code before the compiler type-checks the expanded result. Declarative macros use `macro_rules!`; procedural macros operate on token streams. You do not need macro fluency for the ramp, but you should recognize them in crates like `tokio`, `axum`, and tracing.

## Quick Checks

- What invariant must hold when creating a slice from a raw pointer?
- Why should unsafe code be wrapped in a safe function?
- What is the difference between an associated type and a generic type parameter?
- When do you need fully qualified syntax?
