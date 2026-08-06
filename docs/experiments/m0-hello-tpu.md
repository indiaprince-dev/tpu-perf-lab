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
| CPU (Colab) | 1825.7 ms | 179.3 | 0.075 TFLOP/s | not published | — |
| NVIDIA Tesla T4 | 36.6 ms | 6.2 | 3.76 TFLOP/s | 8.1 TFLOP/s fp32 | **46.9%** |
| TPU v5e-1 | pending | | | 197 TFLOP/s bf16 | |

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

**The CPU-to-T4 ratio is 50x** (1825.7 ms against 36.6 ms) even with the GPU
running an emulated path. On a dtype the T4 executes natively the gap would be
considerably wider.

**Run-to-run spread is large on shared infrastructure.** The T4 standard
deviation is 6.2 ms against a 36.6 ms median, 17%. Colab hardware is
multi-tenant, and the minimum of 35.9 ms is the better estimate of uncontended
performance. This is the case the
[measurement conventions](../reference/measurement.md) anticipate.

Remaining once the TPU run lands:

- Does the achieved figure approach peak, given that intensity of 1365 is well
  above v5e machine balance of 229?
- How does the v5e minimum-to-median gap compare with the T4's 17%?

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
