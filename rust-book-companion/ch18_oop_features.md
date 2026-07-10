# Ch 18 — Object-Oriented Programming Features

📖 https://doc.rust-lang.org/book/ch18-00-oop.html

(Was Ch 17 in the 2nd edition.)

## How Rust maps to OOP

- **Encapsulation**: ✅ `pub` vs private on structs/fields/methods.
- **Objects = data + methods**: ✅ structs/enums + `impl`.
- **Inheritance**: ❌ none. Reuse via default trait methods; polymorphism via generics (static) or trait objects (dynamic). Composition over inheritance.

## Trait objects — `dyn Trait`

```rust
pub trait Draw {
    fn draw(&self);
}

pub struct Screen {
    pub components: Vec<Box<dyn Draw>>,     // ANY type implementing Draw —
}                                            // types can differ per element

impl Screen {
    pub fn run(&self) {
        for component in self.components.iter() {
            component.draw();                // dynamic dispatch via vtable
        }
    }
}

let screen = Screen {
    components: vec![Box::new(SelectBox {...}), Box::new(Button {...})],
};
```

### Generics vs trait objects

| | `Vec<T> where T: Draw` | `Vec<Box<dyn Draw>>` |
|---|---|---|
| Types in one collection | all the same | mixed |
| Dispatch | static (monomorphized) | dynamic (vtable) |
| Cost | zero, but code bloat | pointer indirection, no inlining |
| Open to new types at runtime | no | yes |

- **Dyn compatibility** (formerly "object safety"): a trait can be a trait object only if its methods don't return `Self` and have no generic type parameters (roughly). `Clone` is not dyn-compatible, for example.

## Typestate: encoding states in the type system

The chapter's blog-post example, done two ways:

```rust
// 1) Classic OOP style: one struct, internal state as Box<dyn State>,
//    delegating methods. Invalid ops are silently ignored at runtime.
let mut post = Post::new();
post.add_text("...");
post.request_review();
post.approve();
assert_eq!("...", post.content());

// 2) Rust-idiomatic: STATES AS TYPES — invalid transitions don't compile
let post: DraftPost = Post::new();
let post: PendingReviewPost = post.request_review();  // consumes the draft
let post: Post = post.approve();                      // only Post has .content()
// draft.content()  ❌ method doesn't exist on DraftPost — compile error
```

Takeaway: **make invalid states unrepresentable**. Transitions that consume `self` and return a new type move whole classes of bugs from runtime to compile time.

## Gotchas

- `dyn Trait` is unsized — always behind a pointer: `Box<dyn T>`, `&dyn T`, `Rc<dyn T>`.
- Don't reach for trait objects by default; generics are the common case. Use `dyn` when you genuinely need heterogeneous collections or runtime plugin-style extension.
- Missing inheritance is intentional — default methods + composition + traits cover the legitimate uses.
