# TPU specifications

Per-chip figures transcribed from the official Cloud TPU documentation,
August 2026. Every roofline calculation on this site depends on these numbers.
The collection process and its findings are [M1](../experiments/m1-spec-sheet.md).

## Published figures

| Generation | Peak bf16 | Low precision | HBM capacity | HBM bandwidth | TensorCores | MXU | Pod | Source |
|---|---|---|---|---|---|---|---|---|
| TPU v4 | 275 TFLOP/s | 275 TOPS int8 | 32 GiB | 1200 GBps | 2 | 128×128 | 4096 | [docs](https://docs.cloud.google.com/tpu/docs/v4) |
| TPU v5e | 197 TFLOP/s | 393 TOPS int8 | 16 GB | 800 GiBps | 1 | 128×128 | 256 | [docs](https://docs.cloud.google.com/tpu/docs/v5e) |
| TPU v5p | 459 TFLOP/s | 459 TFLOP/s fp8 | 95 GiB | 2765 GBps | 2 | 128×128 | 8960 | [docs](https://docs.cloud.google.com/tpu/docs/v5p) |
| TPU v6e | 918 TFLOP/s | 1836 TOPS int8 | 32 GB | 1638 GBps | 1 | 256×256 | 256 | [docs](https://docs.cloud.google.com/tpu/docs/v6e) |

Capacity and bandwidth are reproduced in the units used by the source. The
documentation mixes GB and GiB between generations; see
[units](#a-note-on-units) below.

Interchip interconnect bandwidth, bidirectional, per chip: v5e 400 GBps,
v5p 1200 GBps, v6e 800 GBps. The v4 page gives six interconnect links per chip
without a per-chip bandwidth figure.

TPU v7 (Ironwood) uses the 256×256 MXU according to the architecture page, but
a per-chip specification page was not reachable at the time of collection.

## Derived: machine balance

Machine balance is peak throughput divided by memory bandwidth. It is the
arithmetic intensity a kernel must exceed to be compute-bound.

| Generation | Machine balance | Regime threshold |
|---|---|---|
| TPU v5p | **166** FLOP/byte | easiest to keep the matrix units fed |
| TPU v4 | 229 FLOP/byte | |
| TPU v5e | 229 FLOP/byte | (246 under the GB/s reading) |
| TPU v6e | **560** FLOP/byte | hardest |

**The threshold varies by a factor of 3.4 across generations.** A kernel with
intensity 300 FLOP/byte is compute-bound on v5p, v4, and v5e, and memory-bound
on v6e. Roofline conclusions do not transfer between chips.

## A note on units

The source documentation is not internally consistent:

- v5e bandwidth is published as **800 GiBps**; v4, v5p, and v6e use **GBps**.
- v4 and v5p capacity is published in **GiB**; v5e and v6e in **GB**.

For v5e the reading changes machine balance from **229.3** (GiB/s, as written)
to **246.2** (GB/s), a difference of 7%. That is small enough to be invisible
in a plot and large enough to matter when a measurement lands near the ridge.

This repository stores the figure as documented and converts to bytes/s at the
point of use. `tpuperf/specs.py` records the original string in
`bandwidth_as_documented` and `capacity_as_documented` so the conversion stays
auditable.

## Limitations

**These are theoretical peaks.** Vendor peak throughput assumes ideal issue
rates, and published HBM bandwidth is the interface maximum rather than what a
streaming kernel achieves. Measured bandwidth is lower, sometimes considerably.
Machine balance computed from theoretical figures is therefore an upper bound
on the intensity actually required.

Measuring achievable bandwidth to sit beside the theoretical figure is
outstanding work.

## Programmatic access

```python
from tpuperf.specs import SPECS, lookup

SPECS["v6e"].machine_balance("bf16")   # 560.4
lookup("TPU v5 lite").peak("bf16")     # 197.0
lookup("Tesla T4").supports("bf16")    # False
```

Peak is stored per dtype. An absent dtype means the device has no hardware
path for it, which is distinct from the device being unknown. `lookup` returns
None for unrecognised devices, and CPU has no entry because peak varies by host.

`tpuperf.specs` imports without JAX so the roofline calculator runs on any
machine.

## Non-TPU devices

Comparison hardware is included so the same script can run across
architectures. The dtype a device supports is a hardware property and does not
transfer.

| Device | Reference | Peak | Memory | Balance | Note |
|---|---|---|---|---|---|
| NVIDIA Tesla T4 | fp32 | 8.1 TFLOP/s | 320 GB/s GDDR6 | 25 | **No bf16 path.** Turing tensor cores cover fp16 (65), int8 (130), int4 (260). bf16 arrived with Ampere |

A bf16 kernel on a T4 does not fail; it falls back to the CUDA cores. The
result is a valid measurement of the fallback and says nothing about the
kernel, which is why `supports()` exists.
