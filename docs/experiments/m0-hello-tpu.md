# M0 · Hello TPU

## Question

Does the environment work, and what fraction of the chip's peak throughput
does a single large matrix multiply actually reach?

The second half is the point. Everything else in this repository is an attempt
to explain the gap this experiment opens.

## Method

A square bf16 matmul at $N = 4096$, timed through the standard
[measurement conventions](../reference/measurement.md): three untimed warm-up
calls, then the median of twenty timed repetitions with
`jax.block_until_ready` on each.

$N = 4096$ is chosen to be comfortably compute-bound on current TPU
generations — arithmetic intensity is around $2N/3s \approx 1365$ FLOP/byte in
bf16, several times typical machine balance. If a chip cannot approach peak
*here*, it will not anywhere.

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

!!! info "Pending"

    Awaiting first run. Results land here with the environment record and the
    fraction-of-peak calculation once M1 establishes the peak figure.

| Field | Value |
|---|---|
| Device | — |
| Time (median) | — |
| Achieved | — TFLOP/s |
| Peak (from M1) | — TFLOP/s |
| **Fraction of peak** | **—** |
| Arithmetic intensity | — FLOP/byte |

Raw record: `results/m0_hello_tpu.json`

## Interpretation

To be written against the measurement. The questions it should answer:

- Is the achieved figure plausible for a compute-bound shape, or is something
  structurally wrong (wrong dtype dispatched, host transfer in the loop,
  running on CPU)?
- How large is the minimum/median gap? A wide spread points at a contended
  host rather than at the kernel.
- Does the bf16 result look like it is using the matrix unit, or falling back
  to the vector unit?

## What this does not show

A single shape at a single size. It says nothing about where the memory-bound
crossover falls, how the number moves with dtype, or what happens once the
operation is part of a larger graph where fusion decisions matter. Those are
M2, M9, and M7 respectively.

## Reproduce

```bash
pip install -r requirements.txt
python mini/m0_hello_tpu.py
```

On Colab: *Runtime → Change runtime type → TPU*. JAX is preinstalled, so the
install step can be skipped.

## Next

**M1 — TPU specification table.** The fraction-of-peak cell above cannot be
filled without a trustworthy peak figure, and vendor numbers vary by dtype and
by what they assume about sparsity. Collecting them properly is the next
experiment for exactly that reason.
