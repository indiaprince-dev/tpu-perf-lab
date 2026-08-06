# Glossary

Terms as used on this site.

## Performance

### Arithmetic intensity
FLOPs performed per byte of memory traffic, $I = \text{FLOPs} / \text{bytes}$.
Compared against [machine balance](#machine-balance) to classify a kernel as
memory-bound or compute-bound. See [the roofline model](../concepts/roofline.md).

### Machine balance
Peak throughput divided by memory bandwidth, in FLOP per byte. The arithmetic
intensity a kernel must exceed to be compute-bound on a given chip.

### MFU
**Model FLOPs Utilisation.** Achieved throughput as a fraction of the chip's
peak, computed from the model's *useful* FLOPs — the arithmetic the model
definition requires, excluding recomputation from rematerialisation.
Large-model training commonly lands between 20% and 50%.

### HFU
**Hardware FLOPs Utilisation.** As MFU, but counting every FLOP actually
executed, including recomputation. HFU is always ≥ MFU; the gap is the cost of
[rematerialisation](#rematerialisation).

### Roofline
A bound on attainable performance, $\min(P_{\text{peak}}, BW \times I)$,
plotted against arithmetic intensity. A bound rather than a prediction; kernels can
fall below it for reasons the model omits.

### Compute-bound / memory-bound
Whether a kernel's limiting resource is the arithmetic units or the memory
system. Determined by comparing $I$ against machine balance.

## Hardware

### Systolic array
A grid of multiply-accumulate cells through which operands flow, each cell
passing its result to its neighbour. Eliminates per-operation register and
cache access, which is why it is far more area- and power-efficient than a
general-purpose unit for dense matmul.

### HBM
**High Bandwidth Memory.** The off-chip memory attached to the accelerator.
Large, and slow relative to the arithmetic units. The usual bottleneck.

### VMEM
On-chip memory on TPU, software-managed rather than a hardware cache. Kernel
authors and the compiler decide explicitly what resides there and when, which
is why TPU kernel code is more explicit about tiling than typical GPU code.

### bf16
**Brain float 16.** Sixteen-bit float with the same exponent range as fp32 but
fewer mantissa bits. Trades precision for range, which suits neural network
training, and is the format TPU matrix units are built around.

## Software

### JAX
Numerical computing library with composable transformations (`jit`, `grad`,
`vmap`, `shard_map`). The primary interface to TPUs used here.

### XLA / OpenXLA
The compiler that lowers high-level array operations to accelerator
instructions. Responsible for fusion, layout assignment, and scheduling.

### HLO
**High Level Operations.** XLA's intermediate representation. Dumping it shows
what the compiler actually fused, as opposed to what one hoped it would.

### Pallas
A JAX extension for writing custom accelerator kernels **in Python**, with
explicit control over block sizes and the memory hierarchy. Removes the need
to write C++ to author a TPU kernel.

### Fusion
Combining several operations into one kernel so intermediates stay in fast
memory instead of round-tripping through HBM. The most effective single
optimisation for memory-bound sequences.

### XProf
Profiler for TPU and GPU workloads. Produces traces, kernel-level timing, and
a roofline view.

### Sharding
Splitting arrays and computation across devices — by data, by tensor
dimension, or in combination. Introduces collective communication whose cost
is neither compute nor local memory traffic.

### Rematerialisation
Also *gradient checkpointing*. Discarding intermediate activations during the
forward pass and recomputing them during the backward pass, trading extra
compute for reduced peak memory. Widens the gap between [MFU](#mfu) and
[HFU](#hfu).

### Collective
A communication operation across devices — all-reduce, all-gather,
reduce-scatter. Frequently the limiting factor at scale.
