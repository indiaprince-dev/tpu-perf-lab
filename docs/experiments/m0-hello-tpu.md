# M0 · Hello TPU

## Question

Does the environment work, and what fraction of the chip's peak throughput
does a single large matrix multiplication reach?

The second question is the substantive one. The remainder of this repository
addresses the gap it exposes.

## Method

A square bf16 matrix multiplication at $N = 4096$, timed under the standard
[measurement conventions](../reference/measurement.md): three untimed warm-up
calls, then the median of twenty timed repetitions, each followed by
`jax.block_until_ready`.

$N = 4096$ is chosen to be compute-bound on current TPU generations.
Arithmetic intensity is approximately $2N/3s \approx 1365$ FLOP/byte in bf16,
several times typical machine balance. A chip that does not approach peak at
this shape will not do so at any other.

Derived quantities:

$$
\text{FLOPs} = 2N^3 \qquad
\text{bytes} \ge (3N^2)\,s \qquad
I = \frac{\text{FLOPs}}{\text{bytes}}
$$

```python title="mini/m0_hello_tpu.py"
@jax.jit
def matmul(a, b):
    return a @ b

key_a, key_b = jax.random.split(prng_key(0))
a = jax.random.normal(key_a, (N, N), dtype=DTYPE)
b = jax.random.normal(key_b, (N, N), dtype=DTYPE)

timing = benchmark(matmul, a, b, warmup=3, reps=20)
```

## Results

Colab, JAX 0.7.2. Arithmetic intensity is 1365 FLOP/byte in every case; only
the hardware changes.

| Device | Time (median) | sd | Achieved | Reference peak | Fraction |
|---|---|---|---|---|---|
| CPU (Colab) | 1825.7 ms | 179.3 (9.8%) | 0.075 TFLOP/s | not published | — |
| NVIDIA Tesla T4 | 36.6 ms | 6.2 (17%) | 3.76 TFLOP/s | 8.1 TFLOP/s fp32 | 46.9% |
| **TPU v5e-1** | **0.909 ms** | **0.007 (0.8%)** | **151.3 TFLOP/s** | 197 TFLOP/s bf16 | **76.8%** |

On v5e: intensity 1365 against machine balance 229, so the shape sits a factor
of six inside the compute-bound region. Measured traffic is 110.8 GB/s against
859 GB/s of bandwidth, 13% utilised, which corroborates it.

Raw record: `results/m0_hello_tpu.json`

!!! warning "The T4 figure is not a bf16 measurement"

    Turing tensor cores cover fp16, int8 and int4. There is no bf16 path;
    bf16 arrived with Ampere. The kernel therefore fell back to the CUDA cores,
    and 3.76 TFLOP/s is **46.9% of the 8.1 TFLOP/s fp32 rate** and **5.8% of
    the 65 TFLOP/s fp16 tensor-core figure**. Reporting it against the latter
    would attribute a hardware gap to the kernel.

## Interpretation

**Choosing a dtype for the target architecture does not make it portable.**
bf16 is the format TPU matrix units are built around, and on Turing it has no
hardware path at all. The same source, unchanged, measures the matrix unit on
one device and an emulation on another. Any cross-architecture comparison has
to establish dtype support before it compares numbers.

**76.8% of peak on a single unfused matmul.** The shortfall is not bandwidth:
at 13% of HBM utilised and six times above machine balance, nothing here is
waiting on memory. The remaining 23% is pipeline fill and drain, MXU issue
efficiency, and per-call overhead on a kernel that runs for under a
millisecond. N = 4096 divides evenly by the 128×128 MXU, so tile padding is not
a factor. Whether the fraction improves with size is [M2](index.md).

**The device ordering is 2000x, and most of it is not raw hardware.** v5e is
2009x the CPU and 40x the T4. The T4 gap is inflated because bf16 ran emulated
there; against the T4's native fp16 tensor rate of 65 TFLOP/s, v5e's 197 peak
is only 3x. Most of the measured 40x is a dtype support difference, not a
throughput difference.

**Measurement stability differs by an order of magnitude.** Standard deviation
as a fraction of the median: v5e 0.8%, CPU 9.8%, T4 17%. The TPU runtime gives
a dedicated chip while Colab GPU and CPU are contended. On v5e the minimum
(0.904 ms) and median (0.909 ms) are within 0.6% of each other, so the
[minimum-to-median gap](../reference/measurement.md) that flags contention is
effectively absent.

!!! bug "Fixed after this run"

    The v5e run printed `no published bfloat16 peak` before falling through to
    the correct comparison. JAX reports the dtype as `bfloat16` while the
    datasheet key is `bf16`, so the support check failed on spelling and the
    JSON record stored nulls for peak, fraction and regime. `normalize_dtype`
    now maps between the two. The 76.8% figure was correct; the message and the
    record were not.

## What this does not show

One shape at one size. The result says nothing about where the memory-bound
crossover falls, how the figure varies with dtype, or what changes once the
operation is embedded in a larger graph subject to fusion. Those are M2, M9,
and M7 respectively.

## Reproduce

```bash
pip install -r requirements.txt
python mini/m0_hello_tpu.py
```

On Colab: *Runtime → Change runtime type → TPU*. JAX is preinstalled, so the
install step can be skipped.

## Next

**M2 — matmul MFU sweep.** One shape gives one point. Sweeping $N$ produces the
curve, and the crossover it reveals can be compared against the machine balance
[M1](m1-spec-sheet.md) predicts for this chip.
