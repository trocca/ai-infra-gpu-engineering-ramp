# Ch 15 — Smart Pointers ⭐

📖 https://doc.rust-lang.org/book/ch15-00-smart-pointers.html

Structs that act like pointers (via `Deref`) with extra powers (via `Drop`). `String` and `Vec` already are smart pointers.

## `Box<T>` — heap allocation

```rust
let b = Box::new(5);                 // value on the heap, box on the stack

enum List {                          // recursive types need indirection:
    Cons(i32, Box<List>),            // Box gives the type a known size
    Nil,
}
```

Use when: recursive types, large data you want to move cheaply, or trait objects (`Box<dyn Trait>`).

## `Deref` — make `*` and auto-coercion work

```rust
impl<T> Deref for MyBox<T> {
    type Target = T;
    fn deref(&self) -> &T { &self.0 }
}
// *my_box  desugars to  *(my_box.deref())
```

**Deref coercion**: `&MyBox<String>` → `&String` → `&str` automatically at call sites — why `hello(&boxed_name)` works for `fn hello(name: &str)`. Also `&mut T → &T` (never the reverse). `DerefMut` for `*x = ...`.

## `Drop` — code on cleanup

```rust
impl Drop for CustomSmartPointer {
    fn drop(&mut self) { println!("Dropping `{}`!", self.data); }
}
// values dropped in REVERSE declaration order at scope end
// early drop: std::mem::drop(value)  — you cannot call .drop() directly
```

## `Rc<T>` — reference counting (single-threaded, shared ownership)

```rust
use std::rc::Rc;
let a = Rc::new(Cons(5, Rc::new(Nil)));
let b = Cons(3, Rc::clone(&a));      // bumps refcount — cheap, not a deep clone
let c = Cons(4, Rc::clone(&a));
Rc::strong_count(&a);                // → 3
```

- Immutable access only — many owners with mutation would violate borrow rules.
- Not thread-safe (use `Arc` across threads, Ch 16).

## `RefCell<T>` — interior mutability

Borrow rules enforced **at runtime** instead of compile time — panics on violation instead of failing to compile.

```rust
use std::cell::RefCell;
let cell = RefCell::new(vec![1, 2]);
cell.borrow_mut().push(3);           // → RefMut<T>  (like &mut T)
let v = cell.borrow();               // → Ref<T>     (like &T)
// two simultaneous borrow_mut() → PANIC at runtime
```

- Lets you mutate through a shared reference (`&self`) — e.g., a mock object recording calls, or caches.
- **The combo**: `Rc<RefCell<T>>` = multiple owners of mutable data (single-threaded).

| | `Box<T>` | `Rc<T>` | `RefCell<T>` |
|---|---|---|---|
| Owners | one | many | one |
| Borrows checked | compile time | compile time | **runtime** |
| Mutation | yes (if mut) | no | yes, via `&self` |

## Reference cycles and `Weak<T>`

`Rc` cycles never hit refcount 0 → **memory leak** (safe, but a leak). Break cycles with `Weak<T>`:

```rust
struct Node {
    parent: RefCell<Weak<Node>>,       // child → parent: weak (no ownership)
    children: RefCell<Vec<Rc<Node>>>,  // parent → child: strong (owns)
}
let parent = leaf.parent.borrow().upgrade();   // Weak → Option<Rc<T>>
// Rc::downgrade(&rc) → Weak; weak_count doesn't keep data alive
```

Rule of thumb: ownership edges = `Rc`, back-edges = `Weak`.

## Gotchas

- `Rc::clone(&x)` by convention (not `x.clone()`) to signal "refcount bump, not deep copy".
- `RefCell` moves borrow errors from compile time to **runtime panics** — use sparingly, keep borrow scopes tiny.
- Need this across threads? `Rc<RefCell<T>>` → `Arc<Mutex<T>>` (Ch 16).
