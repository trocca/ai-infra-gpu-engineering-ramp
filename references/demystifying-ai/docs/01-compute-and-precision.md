# 1. Compute: FLOPS and exaFLOPS

[<- Demystifying AI visual primer](../README.md) · [Reference shelf](../../README.md)

## FLOP versus FLOPS

A **FLOP** is one floating-point operation.

**FLOPS** measures how many floating-point operations a system can perform per second.

\[
1\ \text{exaFLOPS}=10^{18}\ \text{floating-point operations per second}
\]

That is one billion billion operations every second.

## Precision changes the meaning of a throughput number

A processor can have very different peak throughput depending on the format:

- FP64
- FP32
- FP16
- BF16
- FP8
- FP4

Therefore, “10 exaFLOPS” is incomplete unless the precision and assumptions are specified.

## FP8

FP8 stores each value in 8 bits. Common layouts include:

- **E4M3**: more precision, smaller numeric range
- **E5M2**: wider range, less precision

FP8 reduces:

- memory consumption
- memory bandwidth
- data movement
- energy per operation

## FP4

FP4 stores each value in only 4 bits.

It provides much stronger compression but usually requires:

- scale factors
- grouping
- calibration
- quantization-aware techniques
- higher-precision accumulation
- special handling of sensitive layers

A model may use:

```text
Weights:       FP4
Activations:   FP8
Accumulation:  FP16 or FP32
```

The format used for storage, multiplication, and accumulation need not be the same.
