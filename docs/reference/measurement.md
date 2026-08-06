# Measurement conventions

Naive JAX benchmarks are wrong in three predictable ways. Every number on this
site is collected through [`tpuperf/bench.py`][bench], which addresses each.

[bench]: https://github.com/indiaprince-dev/tpu-perf-lab/blob/main/tpuperf/bench.py

## 1. Dispatch is asynchronous

JAX returns a future, not a result. A call completes on the host long before
the accelerator has finished.

```python
start = time.perf_counter()
y = matmul(a, b)              # returns immediately
elapsed = time.perf_counter() - start   # measures dispatch, not compute
```

The fix is to wait on the result:

```python
jax.block_until_ready(fn(*args))
```

`jax.block_until_ready` handles pytrees, so it works for functions returning
tuples or dicts, unlike calling `.block_until_ready()` on a single array.

## 2. `jit` compiles on first call

The first invocation of a jitted function includes XLA compilation, which can
be orders of magnitude slower than steady state. Warm-up runs are untimed, and
`warmup >= 1` is enforced rather than defaulted.

Compilation time is itself worth measuring, since it affects both iteration
speed and deployment, but it is a separate quantity and is reported separately
where relevant.

## 3. Single-shot timing is noisy

Host scheduling, clock behaviour, and neighbour traffic on shared hardware all
add variance. Results are the **median** over repeated runs, reported with the
minimum and standard deviation so the spread is visible.

The minimum is reported because it is the closest available estimate of
uncontended performance. A large gap between minimum and median indicates a
contended measurement environment.

## Inputs are random, not constant

```python
a = jax.random.normal(key, (n, n), dtype=jnp.bfloat16)   # correct
a = jnp.ones((n, n), dtype=jnp.bfloat16)                 # wrong
```

Constant inputs allow XLA to fold the operation away entirely, after which the
benchmark measures nothing at an implausible rate.

## Reported memory traffic is a lower bound

For an $(m,k) \times (k,n)$ matmul this repository counts each input read once
and the output written once:

$$
\text{bytes} = (mk + kn + mn) \times s
$$

Real kernels move more when tiles do not fit in on-chip memory and operands
are re-read. Two consequences:

- Reported bandwidth is a **lower bound** on actual HBM traffic.
- Reported arithmetic intensity is an **upper bound**, so the roofline
  prediction derived from it is optimistic.

The asymmetry is deliberate. When a measurement falls below its predicted
roofline, the first hypothesis is that the traffic estimate was too generous,
which is testable.

## Environment is recorded with every result

Each JSON record carries the JAX version, backend, device kind, device count,
and HBM limit. A TFLOP/s figure without the chip it came from is not a result.

```json
{
  "experiment": "m0_hello_tpu",
  "env": {
    "jax_version": "...",
    "device_kind": "...",
    "device_count": 1,
    "hbm_limit_gb": 0.0
  },
  "config": {"n": 4096, "dtype": "bfloat16"},
  "timing_s": {"median": 0.0, "minimum": 0.0, "stdev": 0.0, "reps": 20},
  "derived": {"achieved_tflops": 0.0}
}
```

## Uncontrolled variables

These bound the precision of every result on this site:

- **No clock control.** Thermal and power behaviour is not held fixed between
  runs.
- **Shared infrastructure.** Colab and multi-tenant Cloud TPU hosts are subject
  to neighbour effects. The minimum-to-median gap is the only available
  indicator.
- **Single host.** Measurements involving inter-chip collectives (M8 onward)
  will require separate conventions for attributing communication time.
