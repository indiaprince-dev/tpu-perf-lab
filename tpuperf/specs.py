"""Published accelerator specifications, per chip.

Figures are transcribed from vendor documentation as of 2026-08. Each entry
records the unit as written in the source, because the documentation is
inconsistent between GB and GiB across TPU generations and the difference
propagates into every roofline calculation.

Peak throughput is stored per dtype. A dtype absent from `peak_tflops` is not
natively supported by that device, which matters: bf16 on a Turing GPU is
emulated rather than executed on tensor cores, and the measured result reflects
that rather than any property of the kernel.

Sources:
    v4   https://docs.cloud.google.com/tpu/docs/v4
    v5e  https://docs.cloud.google.com/tpu/docs/v5e
    v5p  https://docs.cloud.google.com/tpu/docs/v5p
    v6e  https://docs.cloud.google.com/tpu/docs/v6e
    arch https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm
    T4   https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet.pdf
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

GIB = 1 << 30
GB = 10**9


@dataclass(frozen=True)
class DeviceSpec:
    """Per-chip published figures for one accelerator."""

    name: str
    vendor: str
    peak_tflops: dict[str, float]
    memory_bandwidth_bytes_per_s: float
    memory_capacity_bytes: float
    bandwidth_as_documented: str
    capacity_as_documented: str
    memory_kind: str = "HBM"
    reference_dtype: str = "bf16"
    tensorcores: Optional[int] = None
    mxu_dim: Optional[int] = None
    pod_chips: Optional[int] = None
    ici_bandwidth_gbps: Optional[float] = None
    notes: str = ""
    source: str = ""
    _aliases: tuple[str, ...] = field(default=(), repr=False)

    def supports(self, dtype: str) -> bool:
        """Whether the device has a published peak for this dtype.

        Accepts either the datasheet spelling ("bf16") or the JAX one
        ("bfloat16").
        """
        return normalize_dtype(dtype) in self.peak_tflops

    def peak(self, dtype: str) -> Optional[float]:
        """Published peak in TFLOP/s (or TOPS) for a dtype, if any."""
        return self.peak_tflops.get(normalize_dtype(dtype))

    def machine_balance(self, dtype: str = "bf16") -> Optional[float]:
        """Arithmetic intensity, FLOP per byte, needed to be compute-bound.

        Returns None when the dtype has no published peak, since a balance
        computed from a fallback path would describe the emulation rather than
        the hardware.
        """
        p = self.peak(dtype)
        if p is None:
            return None
        return p * 1e12 / self.memory_bandwidth_bytes_per_s

    @property
    def best_float_dtype(self) -> str:
        """Highest-throughput floating-point dtype with a published peak.

        Integer formats are excluded: a TOPS figure is not comparable with a
        FLOP/s measurement, so using int4 as a reference would overstate the
        denominator by an order of magnitude.
        """
        floats = {d: v for d, v in self.peak_tflops.items() if not d.startswith("int")}
        return max(floats, key=floats.__getitem__)


# Documented bandwidth units differ by generation: v5e is published in GiB/s,
# the others in GB/s. All are converted to bytes/s here so machine balance is
# comparable across rows.
SPECS: dict[str, DeviceSpec] = {
    "v4": DeviceSpec(
        name="TPU v4",
        vendor="Google",
        peak_tflops={"bf16": 275, "int8": 275},
        memory_bandwidth_bytes_per_s=1200 * GB,
        memory_capacity_bytes=32 * GIB,
        bandwidth_as_documented="1200 GBps",
        capacity_as_documented="32 GiB",
        tensorcores=2,
        mxu_dim=128,
        pod_chips=4096,
        notes="int8 is published at the same rate as bf16; no quantisation gain.",
        source="https://docs.cloud.google.com/tpu/docs/v4",
        _aliases=("v4",),
    ),
    "v5e": DeviceSpec(
        name="TPU v5e",
        vendor="Google",
        peak_tflops={"bf16": 197, "int8": 393},
        memory_bandwidth_bytes_per_s=800 * GIB,
        memory_capacity_bytes=16 * GB,
        bandwidth_as_documented="800 GiBps",
        capacity_as_documented="16 GB",
        tensorcores=1,
        mxu_dim=128,
        pod_chips=256,
        ici_bandwidth_gbps=400,
        notes="Bandwidth is published in GiBps; the GBps reading gives 246 balance.",
        source="https://docs.cloud.google.com/tpu/docs/v5e",
        _aliases=("v5 lite", "v5e", "v5litepod"),
    ),
    "v5p": DeviceSpec(
        name="TPU v5p",
        vendor="Google",
        peak_tflops={"bf16": 459, "fp8": 459},
        memory_bandwidth_bytes_per_s=2765 * GB,
        memory_capacity_bytes=95 * GIB,
        bandwidth_as_documented="2765 GBps",
        capacity_as_documented="95 GiB",
        tensorcores=2,
        mxu_dim=128,
        pod_chips=8960,
        ici_bandwidth_gbps=1200,
        notes="Publishes fp8 rather than int8, at parity with bf16.",
        source="https://docs.cloud.google.com/tpu/docs/v5p",
        _aliases=("v5p",),
    ),
    "v6e": DeviceSpec(
        name="TPU v6e (Trillium)",
        vendor="Google",
        peak_tflops={"bf16": 918, "int8": 1836},
        memory_bandwidth_bytes_per_s=1638 * GB,
        memory_capacity_bytes=32 * GB,
        bandwidth_as_documented="1638 GBps",
        capacity_as_documented="32 GB",
        tensorcores=1,
        mxu_dim=256,
        pod_chips=256,
        ici_bandwidth_gbps=800,
        notes="MXU widened to 256x256; balance rose to 560, the highest here.",
        source="https://docs.cloud.google.com/tpu/docs/v6e",
        _aliases=("v6e", "v6"),
    ),
    "t4": DeviceSpec(
        name="NVIDIA Tesla T4",
        vendor="NVIDIA",
        # Turing tensor cores cover fp16, int8 and int4. There is no bf16 path:
        # bf16 arrived with Ampere, so bf16 work here is emulated on CUDA cores.
        peak_tflops={"fp32": 8.1, "fp16": 65, "int8": 130, "int4": 260},
        memory_bandwidth_bytes_per_s=320 * GB,
        memory_capacity_bytes=16 * GB,
        bandwidth_as_documented="320 GB/s",
        capacity_as_documented="16 GB",
        memory_kind="GDDR6",
        # bf16 falls back to the CUDA cores, so fp32 is the rate to compare
        # against rather than the 65 TFLOP/s fp16 tensor-core headline.
        reference_dtype="fp32",
        notes=(
            "Turing. No native bf16: tensor cores support fp16, int8 and int4 "
            "only. bf16 kernels fall back and should be compared against the "
            "8.1 TFLOP/s fp32 rate, not the 65 TFLOP/s fp16 tensor rate."
        ),
        source="https://www.nvidia.com/en-us/data-center/tesla-t4/",
        _aliases=("tesla t4", "t4"),
    ),
}


# JAX reports dtype names such as "bfloat16" and "float32"; vendor datasheets
# use "bf16" and "fp32". Mapping between them has to happen somewhere, and
# getting it wrong reads as "this device has no path for that dtype".
_DTYPE_ALIASES = {
    "bfloat16": "bf16",
    "float16": "fp16",
    "half": "fp16",
    "float32": "fp32",
    "float": "fp32",
    "float64": "fp64",
    "double": "fp64",
    "float8_e4m3fn": "fp8",
    "float8_e4m3b11fnuz": "fp8",
    "float8_e5m2": "fp8",
}


def normalize_dtype(name: str) -> str:
    """Map a JAX or NumPy dtype name onto the key used in `peak_tflops`."""
    key = str(name).lower()
    return _DTYPE_ALIASES.get(key, key)


def lookup(device_kind: str) -> Optional[DeviceSpec]:
    """Match a JAX `device_kind` string to a published specification.

    `device_kind` reports values such as "TPU v5 lite" or "Tesla T4", so the
    match is on aliases rather than equality. Returns None when the device is
    unknown, which callers should treat as "peak is unknown" rather than
    substituting a guess. CPU deliberately has no entry: peak varies by host.
    """
    k = device_kind.lower().replace("-", " ").strip()
    for spec in SPECS.values():
        if any(alias in k for alias in spec._aliases):
            return spec
    return None
