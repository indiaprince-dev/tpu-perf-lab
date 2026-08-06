"""M0: Hello TPU.

Confirms the environment and produces the first measurement: what fraction of
the chip's peak throughput a single large matrix multiplication reaches.

A datasheet quotes peak FLOP/s; real kernels rarely approach it. Accounting
for that difference is the subject of the remaining experiments.

Run:
    python mini/m0_hello_tpu.py

On Colab, select Runtime > Change runtime type > TPU first.
"""

from __future__ import annotations

import json
import pathlib
import sys

import jax
import jax.numpy as jnp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tpuperf.specs import normalize_dtype  # noqa: E402
from tpuperf import (  # noqa: E402
    benchmark,
    device_info,
    format_device_info,
    lookup,
    matmul_bytes,
    matmul_flops,
    prng_key,
)

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"

# 4096 is large enough to be compute-bound on current TPUs and small enough to
# fit in HBM. M2 sweeps this dimension to locate the memory-bound to
# compute-bound crossover.
N = 4096
DTYPE = jnp.bfloat16  # the format TPU matrix units are designed around


@jax.jit
def matmul(a: jax.Array, b: jax.Array) -> jax.Array:
    return a @ b


def main() -> None:
    info = device_info()
    print(format_device_info(info))
    print()

    if info["backend"] == "cpu":
        print("WARNING: running on CPU. Numbers below are not TPU numbers.")
        print("In Colab: Runtime > Change runtime type > TPU.\n")

    # Random inputs, not ones — constant inputs let XLA fold the operation
    # away and the benchmark would measure nothing.
    key_a, key_b = jax.random.split(prng_key(0))
    a = jax.random.normal(key_a, (N, N), dtype=DTYPE)
    b = jax.random.normal(key_b, (N, N), dtype=DTYPE)

    # Verify the result actually lands on the accelerator before timing.
    out = matmul(a, b)
    print(f"Result       {out.shape} {out.dtype} on {out.devices()}")

    timing = benchmark(matmul, a, b, warmup=3, reps=20)

    flops = matmul_flops(N, N, N)
    hbm_bytes = matmul_bytes(N, N, N, jnp.dtype(DTYPE).itemsize)
    intensity = flops / hbm_bytes

    spec = lookup(info["device_kind"])

    print()
    print(f"matmul       {N}x{N} @ {N}x{N}, {jnp.dtype(DTYPE).name}")
    print(f"  time       {timing.median * 1e3:.3f} ms "
          f"(min {timing.minimum * 1e3:.3f}, sd {timing.stdev * 1e3:.3f}, "
          f"n={timing.reps})")
    print(f"  achieved   {timing.tflops(flops):.1f} TFLOP/s")
    print(f"  traffic    {timing.gbytes_per_s(hbm_bytes):.1f} GB/s "
          f"(lower bound)")
    print(f"  intensity  {intensity:.0f} FLOP/byte")

    dtype_name = normalize_dtype(jnp.dtype(DTYPE).name)
    fraction = balance = regime = None

    print()
    if spec is None:
        print(f"No published peak on file for {info['device_kind']!r}.")
        print("Add it to tpuperf/specs.py before interpreting the figures above.")
    elif not spec.supports(dtype_name):
        # The device has no hardware path for this dtype, so the measurement
        # describes an emulation rather than the chip. Compare against a dtype
        # it does support instead of reporting a meaningless fraction of peak.
        ref = spec.reference_dtype
        ref_peak = spec.peak(ref)
        print(f"{spec.name}: no published {dtype_name} peak")
        print(f"  {spec.notes}")
        print(f"  vs {ref} peak {ref_peak:.1f} TFLOP/s: "
              f"{timing.tflops(flops) / ref_peak * 100:.1f}%")
    else:
        peak = spec.peak(dtype_name)
        balance = spec.machine_balance(dtype_name)
        fraction = timing.tflops(flops) / peak
        regime = "compute-bound" if intensity > balance else "memory-bound"
        print(f"against {spec.name} (M1)")
        print(f"  peak       {peak:.0f} TFLOP/s {dtype_name}")
        print(f"  achieved   {fraction * 100:.1f}% of peak")
        print(f"  balance    {balance:.0f} FLOP/byte "
              f"-> this shape is {regime}")

    RESULTS.mkdir(exist_ok=True)
    payload = {
        "experiment": "m0_hello_tpu",
        "env": info,
        "config": {"n": N, "dtype": jnp.dtype(DTYPE).name},
        "timing_s": timing.as_dict(),
        "derived": {
            "achieved_tflops": timing.tflops(flops),
            "memory_gb_per_s_lower_bound": timing.gbytes_per_s(hbm_bytes),
            "arithmetic_intensity_flop_per_byte": intensity,
            "device_spec": spec.name if spec else None,
            "dtype_natively_supported": (
                spec.supports(dtype_name) if spec else None
            ),
            "peak_tflops": spec.peak(dtype_name) if spec else None,
            "fraction_of_peak": fraction,
            "machine_balance": balance,
            "regime": regime,
        },
    }
    path = RESULTS / "m0_hello_tpu.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nWrote {path.relative_to(RESULTS.parent)}")


if __name__ == "__main__":
    main()
