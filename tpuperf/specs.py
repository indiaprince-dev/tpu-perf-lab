"""Published TPU specifications, per chip.

Figures are transcribed from the official Cloud TPU documentation as of
2026-08. Each entry records the unit as written in the source, because the
documentation is inconsistent between GB and GiB across generations and the
difference propagates into every roofline calculation.

Sources:
    v4   https://docs.cloud.google.com/tpu/docs/v4
    v5e  https://docs.cloud.google.com/tpu/docs/v5e
    v5p  https://docs.cloud.google.com/tpu/docs/v5p
    v6e  https://docs.cloud.google.com/tpu/docs/v6e
    arch https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

GIB = 1 << 30
GB = 10**9


@dataclass(frozen=True)
class ChipSpec:
    """Per-chip published figures for one TPU generation."""

    name: str
    bf16_tflops: float
    hbm_bandwidth_bytes_per_s: float
    hbm_capacity_bytes: float
    tensorcores: int
    mxu_dim: int
    pod_chips: int
    bandwidth_as_documented: str
    capacity_as_documented: str
    low_precision_tops: Optional[float] = None
    low_precision_format: Optional[str] = None
    ici_bandwidth_gbps: Optional[float] = None
    source: str = ""

    @property
    def machine_balance(self) -> float:
        """Arithmetic intensity, in FLOP per byte, required to be compute-bound."""
        return self.bf16_tflops * 1e12 / self.hbm_bandwidth_bytes_per_s

    @property
    def low_precision_speedup(self) -> Optional[float]:
        """Throughput ratio of the low-precision format to bf16, if published."""
        if self.low_precision_tops is None:
            return None
        return self.low_precision_tops / self.bf16_tflops


# Documented bandwidth units differ by generation: v5e is published in GiB/s,
# the others in GB/s. Both are converted to bytes/s here so machine balance is
# comparable across rows.
SPECS: dict[str, ChipSpec] = {
    "v4": ChipSpec(
        name="TPU v4",
        bf16_tflops=275,
        low_precision_tops=275,
        low_precision_format="int8",
        hbm_bandwidth_bytes_per_s=1200 * GB,
        hbm_capacity_bytes=32 * GIB,
        tensorcores=2,
        mxu_dim=128,
        pod_chips=4096,
        bandwidth_as_documented="1200 GBps",
        capacity_as_documented="32 GiB",
        source="https://docs.cloud.google.com/tpu/docs/v4",
    ),
    "v5e": ChipSpec(
        name="TPU v5e",
        bf16_tflops=197,
        low_precision_tops=393,
        low_precision_format="int8",
        hbm_bandwidth_bytes_per_s=800 * GIB,
        hbm_capacity_bytes=16 * GB,
        tensorcores=1,
        mxu_dim=128,
        pod_chips=256,
        bandwidth_as_documented="800 GiBps",
        capacity_as_documented="16 GB",
        ici_bandwidth_gbps=400,
        source="https://docs.cloud.google.com/tpu/docs/v5e",
    ),
    "v5p": ChipSpec(
        name="TPU v5p",
        bf16_tflops=459,
        low_precision_tops=459,
        low_precision_format="fp8",
        hbm_bandwidth_bytes_per_s=2765 * GB,
        hbm_capacity_bytes=95 * GIB,
        tensorcores=2,
        mxu_dim=128,
        pod_chips=8960,
        bandwidth_as_documented="2765 GBps",
        capacity_as_documented="95 GiB",
        ici_bandwidth_gbps=1200,
        source="https://docs.cloud.google.com/tpu/docs/v5p",
    ),
    "v6e": ChipSpec(
        name="TPU v6e (Trillium)",
        bf16_tflops=918,
        low_precision_tops=1836,
        low_precision_format="int8",
        hbm_bandwidth_bytes_per_s=1638 * GB,
        hbm_capacity_bytes=32 * GB,
        tensorcores=1,
        mxu_dim=256,
        pod_chips=256,
        bandwidth_as_documented="1638 GBps",
        capacity_as_documented="32 GB",
        ici_bandwidth_gbps=800,
        source="https://docs.cloud.google.com/tpu/docs/v6e",
    ),
}


def lookup(device_kind: str) -> Optional[ChipSpec]:
    """Match a JAX `device_kind` string to a published specification.

    `device_kind` reports values such as "TPU v5 lite" or "TPU v6e", so the
    match is on substrings rather than equality. Returns None when the device
    is not recognised, which callers should treat as "peak is unknown" rather
    than substituting a guess.
    """
    k = device_kind.lower().replace("-", " ")
    if "v6" in k:
        return SPECS["v6e"]
    if "v5" in k and ("lite" in k or "5e" in k):
        return SPECS["v5e"]
    if "v5" in k:
        return SPECS["v5p"]
    if "v4" in k:
        return SPECS["v4"]
    return None
