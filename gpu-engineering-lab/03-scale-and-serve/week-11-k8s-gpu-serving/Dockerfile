# Dockerfile for the week-08 ferrum-serve Rust engine — SKELETON.
# The layer-ordering and image-size decisions are the learning payload.
# This is where the Rust bet pays off visibly: a PyTorch serving image is
# 6–10 GB and cold-starts in tens of seconds; your target here is a
# < 500 MB image whose process is serving traffic in < 1 s (weights load
# time aside). Measure BOTH and put the table in the README.
#
# Targets: final image < 500 MB, rebuild after a src-only change < 60 s
# (dependency layers must cache).

# ---- builder -----------------------------------------------------------
# Needs the CUDA *toolkit* (nvcc/NVRTC headers) because candle's cuda
# feature compiles kernels at build time. -devel image, big — that's fine,
# it never ships.
FROM nvidia/cuda:12.9.0-devel-ubuntu24.04 AS builder

# TODO(you): install rustup + stable toolchain in ONE layer.

WORKDIR /app

# ---- deps layer (must cache!) ------------------------------------------
# TODO(you): copy ONLY Cargo.toml + Cargo.lock first, create a dummy
# src/main.rs, `cargo build --release`, then delete the dummy. This is the
# Rust equivalent of the requirements.txt-before-source trick: if you COPY
# the whole repo first, every source edit recompiles candle + all deps
# (~10+ min). Same interview question, Rust edition.
# COPY Cargo.toml Cargo.lock ./
# RUN ...dummy-build trick...

# ---- app layer -----------------------------------------------------------
# TODO(you): copy the real src/, touch main.rs, cargo build --release.
# COPY src/ ./src/
# RUN cargo build --release

# ---- runtime -------------------------------------------------------------
# The binary is static-ish but NOT self-contained: it dlopens CUDA runtime
# libs (libcublas, libcurand…). Two defensible options — pick one, justify:
#   nvidia/cuda:12.9.0-runtime-ubuntu24.04            (~2 GB, easy)
#   ubuntu:24.04 + COPY the exact .so files you need   (< 500 MB, the punchline)
# libcuda.so itself comes from the HOST driver, injected by the NVIDIA
# container toolkit via runtimeClassName — never bake it into an image.
FROM ubuntu:24.04 AS runtime

# TODO(you): copy the binary + the minimal set of CUDA .so files from the
# builder (ldd the binary to find them). Document each one you ship.
# COPY --from=builder /app/target/release/ferrum-serve /usr/local/bin/
# COPY --from=builder /usr/local/cuda/lib64/libcublas.so.12 /usr/local/cuda/lib64/ ...

# Model weights: do NOT bake multi-GB weights into the image.
# TODO(you): decide + document — download at startup (initContainer or
# entrypoint) vs hostPath/PVC mount. Either is defensible; say why.

# TODO(you): non-root user (runAsNonRoot in the Deployment will require it).

EXPOSE 8000

# Healthcheck endpoint must exist in the server (week-08 /healthz).
# K8s probes live in deployment.yaml; a local `docker run` smoke test
# should still work standalone:
# HEALTHCHECK --interval=30s --start-period=60s CMD curl -sf http://localhost:8000/healthz || exit 1

# A single Rust binary handles signals itself (tokio + graceful shutdown) —
# no tini needed IF your axum server listens for SIGTERM. Verify that in
# week 08; it's what makes terminationGracePeriodSeconds actually work.
# ENTRYPOINT ["/usr/local/bin/ferrum-serve", "--port", "8000"]
