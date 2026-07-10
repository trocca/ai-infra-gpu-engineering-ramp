# Chapter 21 - Final Project: Multithreaded Web Server

[<- Rust companion index](README.md)

## Why this chapter matters here

This chapter is the bridge from Rust syntax to systems shape: sockets, request parsing, worker pools, message passing, and graceful shutdown. Week 8 uses a production web stack (`axum` + `tokio`), but the underlying ideas are the same.

## Core Architecture

A minimal server has:

- a listener that accepts TCP connections;
- a request parser;
- a response writer;
- concurrency so one slow request does not block every other request;
- shutdown behavior that lets in-flight work finish cleanly.

## Thread Pool Mental Model

A thread pool owns a fixed number of worker threads. New jobs are sent through a channel. Workers loop:

```text
receive job -> run job -> wait for next job
```

This separates accepting work from executing work. In serving systems, that separation becomes request admission, queueing, scheduling, and execution.

## Ownership Pattern

The server owns the workers. Each worker owns its thread handle. The channel sender is cloned into the server path, while workers share the receiver through synchronization.

That pattern shows up again in async form:

- `tokio::spawn` owns a task;
- channels connect producers and consumers;
- shared state uses `Arc`, sometimes with `Mutex` or lock-free structures;
- graceful shutdown needs explicit signaling.

## Failure Modes to Notice

- Unbounded queues hide overload until latency explodes.
- Dropping a sender can signal workers to stop.
- Joining threads matters if you want clean shutdown.
- Blocking CPU work inside an async runtime can starve unrelated requests.

## GPU-Serving Translation

In week 8, the "job" is not serving a static HTML file. It is a generation request. The same design questions remain:

- Where does the request wait?
- Who owns the model engine?
- How do streaming responses leave the system?
- What happens when the queue is full?
- How do you shut down without losing in-flight requests?

## Quick Checks

- Why does a fixed-size worker pool protect a server?
- What is the difference between queueing work and executing work?
- How does graceful shutdown differ from killing the process?
- Why is blocking work dangerous in an async server?
