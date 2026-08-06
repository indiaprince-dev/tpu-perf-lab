"""Timing utilities for TPU and GPU benchmarking.

Naive JAX benchmarks are wrong in three ways, each of which this module
addresses:

1. Dispatch is asynchronous. `f(x)` returns a future immediately, so timing
   without `block_until_ready` measures dispatch overhead rather than compute.
2. `jit` compiles on the first call. That invocation includes XLA compilation,
   which can be orders of magnitude slower than steady state.
3. Single-shot timing is noisy. Host scheduling, clock behaviour, and
   neighbour traffic all contribute variance, so the median over repeated runs
   is reported.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, asdict
from typing import Any, Callable

import jax


@dataclass(frozen=True)
class Timing:
    """Result of a benchmark run. Times are in seconds."""

    median: float
    minimum: float
    stdev: float
    reps: int

    def tflops(self, flops: float) -> float:
        """Achieved TFLOP/s given the operation's FLOP count."""
        return flops / self.median / 1e12

    def gbytes_per_s(self, num_bytes: float) -> float:
        """Achieved GB/s given the operation's memory traffic in bytes."""
        return num_bytes / self.median / 1e9

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def benchmark(
    fn: Callable[..., Any],
    *args: Any,
    warmup: int = 3,
    reps: int = 20,
    **kwargs: Any,
) -> Timing:
    """Time `fn(*args, **kwargs)`, excluding compilation.

    Args:
        fn: Callable to time. Usually a `jax.jit`-wrapped function.
        warmup: Untimed calls run first. Must be >= 1 so compilation is
            excluded from the measurement.
        reps: Timed repetitions. The median is reported.

    Returns:
        A `Timing` with median / minimum / stdev in seconds.
    """
    if warmup < 1:
        raise ValueError("warmup must be >= 1 so JIT compilation is excluded")
    if reps < 1:
        raise ValueError("reps must be >= 1")

    for _ in range(warmup):
        jax.block_until_ready(fn(*args, **kwargs))

    samples: list[float] = []
    for _ in range(reps):
        start = time.perf_counter()
        jax.block_until_ready(fn(*args, **kwargs))
        samples.append(time.perf_counter() - start)

    return Timing(
        median=statistics.median(samples),
        minimum=min(samples),
        stdev=statistics.stdev(samples) if len(samples) > 1 else 0.0,
        reps=reps,
    )


def matmul_flops(m: int, k: int, n: int) -> float:
    """FLOP count for an (m, k) @ (k, n) matmul.

    One multiply and one add per inner-product term, so 2 * m * k * n.
    """
    return 2.0 * m * k * n


def matmul_bytes(m: int, k: int, n: int, itemsize: int) -> float:
    """Minimum HBM traffic for an (m, k) @ (k, n) matmul.

    Counts each input read once and the output written once. Real kernels move
    more when tiles do not fit in on-chip memory, so this is a lower bound and
    therefore the appropriate denominator for arithmetic intensity.
    """
    return float((m * k + k * n + m * n) * itemsize)
