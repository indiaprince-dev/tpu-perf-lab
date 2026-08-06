"""Shared helpers for TPU performance experiments.

Attributes are resolved lazily so that jax-free modules stay importable on
machines without an accelerator. `tpuperf.specs` holds published figures and
`tpuperf.plot` draws them, and both are needed by the roofline calculator,
which should run anywhere.
"""

from typing import TYPE_CHECKING

__all__ = [
    "Timing",
    "benchmark",
    "matmul_bytes",
    "matmul_flops",
    "device_info",
    "format_device_info",
    "prng_key",
    "DeviceSpec",
    "SPECS",
    "lookup",
]

_LAZY = {
    "Timing": "tpuperf.bench",
    "benchmark": "tpuperf.bench",
    "matmul_bytes": "tpuperf.bench",
    "matmul_flops": "tpuperf.bench",
    "device_info": "tpuperf.env",
    "format_device_info": "tpuperf.env",
    "prng_key": "tpuperf.env",
    "DeviceSpec": "tpuperf.specs",
    "SPECS": "tpuperf.specs",
    "lookup": "tpuperf.specs",
}


def __getattr__(name: str):
    module_path = _LAZY.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_path), name)


def __dir__():
    return sorted(__all__)


if TYPE_CHECKING:  # pragma: no cover
    from tpuperf.bench import Timing, benchmark, matmul_bytes, matmul_flops
    from tpuperf.env import device_info, format_device_info, prng_key
    from tpuperf.specs import SPECS, DeviceSpec, lookup
