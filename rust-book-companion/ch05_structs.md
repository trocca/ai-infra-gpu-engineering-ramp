# Ch 5 — Using Structs

📖 https://doc.rust-lang.org/book/ch05-00-structs.html

## Defining and instantiating

```rust
struct User {
    active: bool,
    username: String,
    email: String,
    sign_in_count: u64,
}

let mut user1 = User {
    active: true,
    username: String::from("someone"),
    email: String::from("someone@example.com"),
    sign_in_count: 1,
};
user1.email = String::from("new@example.com");   // whole instance must be mut

// Field init shorthand (param name == field name)
fn build_user(email: String, username: String) -> User {
    User { active: true, username, email, sign_in_count: 1 }
}

// Struct update syntax — remaining fields from another instance
let user2 = User { email: String::from("a@b.com"), ..user1 };
// ⚠️ ..user1 MOVES non-Copy fields (username here) — user1 partially invalid after
```

## Tuple structs & unit structs

```rust
struct Point(i32, i32, i32);        // tuple struct — named type, unnamed fields
struct Meters(f64);                 // "newtype" pattern for type safety
struct AlwaysEqual;                 // unit struct — no data, useful for traits
```

## Methods and associated functions

```rust
#[derive(Debug)]                    // enables {:?} / {:#?} printing, dbg!()
struct Rectangle { width: u32, height: u32 }

impl Rectangle {
    fn area(&self) -> u32 {                      // &self = borrow immutably
        self.width * self.height
    }
    fn grow(&mut self, by: u32) { self.width += by; }   // &mut self
    fn consume(self) -> u32 { self.width }              // self = take ownership (rare)

    fn square(size: u32) -> Self {               // associated function (no self)
        Self { width: size, height: size }       // called as Rectangle::square(3)
    }
}
```

- `Self` = alias for the impl'd type. Multiple `impl` blocks per type are allowed.
- **Automatic referencing/dereferencing**: `rect.area()` works whether you have `Rectangle`, `&Rectangle`, or `Box<Rectangle>` — no `->` operator needed.

## Debug printing

```rust
println!("{rect:?}");    // Debug, one line
println!("{rect:#?}");   // Debug, pretty multi-line
dbg!(&rect);             // prints file:line + value to stderr, returns the value
```

## Gotchas

- Structs holding references need lifetime parameters (Ch 10) — start by owning your data (`String`, not `&str`).
- You can't mark individual fields `mut`; mutability is per-binding.
- `..other` struct-update must come last and takes no trailing comma issues — and it moves fields, it's not a deep copy.
