"""Accelerator environment capture.

Every measurement in this repo is meaningless without knowing which chip
produced it. Each experiment records this alongside its results.
"""

from __future__ import annotations

from typing import Any

import jax


def prng_key(seed: int = 0):
    """PRNG key that works across JAX versions.

    `jax.random.key` is the typed-key API; older releases only have
    `PRNGKey`. Benchmarks use random data rather than `ones` so XLA cannot
    constant-fold the operation being measured.
    """
    factory = getattr(jax.random, "key", None) or jax.random.PRNGKey
    return factory(seed)


def device_info() -> dict[str, Any]:
    """Snapshot of the current accelerator setup."""
    devices = jax.devices()
    first = devices[0]

    info: dict[str, Any] = {
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
        "device_kind": first.device_kind,
        "platform": first.platform,
        "device_count": jax.device_count(),
        "local_device_count": jax.local_device_count(),
        "process_count": jax.process_count(),
    }

    # memory_stats is available on TPU and GPU; CPU returns None.
    stats = getattr(first, "memory_stats", lambda: None)()
    if stats:
        limit = stats.get("bytes_limit")
        in_use = stats.get("bytes_in_use")
        if limit is not None:
            info["hbm_limit_gb"] = round(limit / 1e9, 2)
        if in_use is not None:
            info["hbm_in_use_gb"] = round(in_use / 1e9, 3)

    return info


def format_device_info(info: dict[str, Any]) -> str:
    """Human-readable one-block summary."""
    lines = [
        f"JAX            {info['jax_version']}",
        f"Backend        {info['backend']}",
        f"Device         {info['device_kind']} ({info['platform']})",
        f"Devices        {info['device_count']} total, "
        f"{info['local_device_count']} local, "
        f"{info['process_count']} process(es)",
    ]
    if "hbm_limit_gb" in info:
        lines.append(f"HBM per chip   {info['hbm_limit_gb']} GB")
    return "\n".join(lines)
