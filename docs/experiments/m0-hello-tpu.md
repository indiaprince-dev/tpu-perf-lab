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

!!! info "Pending first run"

    [M1](m1-spec-sheet.md) has supplied the peak figures, so the script now
    resolves the chip against `tpuperf/specs.py` and reports fraction of peak
    and regime directly. The table below fills in from that output.

| Field | Value |
|---|---|
| Device | — |
| Time (median) | — |
| Achieved | — TFLOP/s |
| Peak, bf16 ([M1](m1-spec-sheet.md)) | — TFLOP/s |
| **Fraction of peak** | **—** |
| Arithmetic intensity | — FLOP/byte |
| Machine balance | — FLOP/byte |
| Regime | — |

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
