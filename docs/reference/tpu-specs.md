# TPU specifications

Peak throughput, memory bandwidth, and on-chip memory capacity per generation.
Every roofline calculation on this site depends on these figures, which is why
collecting them is [M1](../experiments/index.md), the first substantive
experiment.

!!! warning "Not yet collected"

    This table is populated by M1 from primary sources. It is left empty rather
    than filled with figures quoted second-hand.

## Why this is harder than looking it up

Vendor peak figures are not directly comparable without unpacking what they
assume:

- **dtype.** A chip quoted at *X* TFLOP/s in int8 may be at *X/2* in bf16 and
  far lower in fp32. Comparisons must fix the dtype.
- **Sparsity.** Some published figures assume structured sparsity that dense
  matmul does not benefit from.
- **Chip versus board versus pod.** A "v5e" figure may refer to one chip, or
  to a host with several, or to a full pod slice. Per-chip normalisation is
  required before any of it means anything.
- **Bandwidth definition.** HBM bandwidth is usually theoretical peak.
  Achievable bandwidth on a streaming benchmark is lower, and that is the
  number a roofline should arguably use.

M1 records both the quoted figure and its source, and where feasible a
measured bandwidth to sit beside the theoretical one.

## Table

| Generation | Peak (bf16) | HBM capacity | HBM bandwidth | On-chip memory | Source |
|---|---|---|---|---|---|
| v4 | — | — | — | — | — |
| v5e | — | — | — | — | — |
| v5p | — | — | — | — | — |
| v6e | — | — | — | — | — |

## Derived

| Generation | Machine balance (FLOP/byte) |
|---|---|
| — | — |

Machine balance is peak divided by bandwidth — the arithmetic intensity a
kernel must exceed to be compute-bound. See
[the roofline model](../concepts/roofline.md).
