# M1 · TPU specification table

## Question

What are the peak throughput, memory bandwidth, and on-chip characteristics of
each TPU generation, and what arithmetic intensity does each require for a
kernel to be compute-bound?

[M0](m0-hello-tpu.md) produces an achieved TFLOP/s figure that means nothing
without a peak to divide by. This experiment supplies it.

## Method

Per-chip figures transcribed from the version-specific Cloud TPU documentation
pages, one source per generation. Units are recorded as written rather than
normalised at transcription time, then converted to bytes per second at the
point of calculation.

Machine balance is derived:

$$
\text{balance} = \frac{P_{\text{peak}}}{BW}
$$

No measurement is involved. Every figure is a published maximum.

```bash
python mini/m1_spec_table.py
```

## Results

Full table: [TPU specifications](../reference/tpu-specs.md).

| Generation | Peak bf16 | HBM bandwidth | **Machine balance** | MXU |
|---|---|---|---|---|
| TPU v4 | 275 TFLOP/s | 1200 GBps | 229 | 128×128 |
| TPU v5e | 197 TFLOP/s | 800 GiBps | 229 | 128×128 |
| TPU v5p | 459 TFLOP/s | 2765 GBps | **166** | 128×128 |
| TPU v6e | 918 TFLOP/s | 1638 GBps | **560** | 256×256 |

Raw record: `results/m1_spec_table.json`

## Interpretation

**Machine balance varies by a factor of 3.4 across generations.** A kernel at
300 FLOP/byte is compute-bound on v5p, v4, and v5e, and memory-bound on v6e.
Roofline conclusions are per-chip and do not transfer.

**The gap widened sharply at v6e.** Against v5e, compute increased 4.66× while
bandwidth increased 1.91×, so the intensity required to stay compute-bound rose
2.44×. Successive generations are not uniformly easier to use; this one demands
more of the kernel author.

**v5p is bandwidth-rich and v6e is not.** v6e has twice the compute of v5p and
**59% of its bandwidth** (1638 against 2765 GBps), alongside a third of the HBM
capacity (32 GB against 95 GiB). The two chips target different problems, and
the same code will behave differently on each.

**Low-precision speedup is not universal.** v5e and v6e publish 2× throughput
for int8. v4 publishes the *same* figure for int8 as for bf16, and v5p
publishes fp8 rather than int8, also at parity with bf16. Quantisation is worth
a measured 2× on two of these four generations and nothing on the other two.

**The MXU changed shape at v6e**, from 128×128 to 256×256 multiply-accumulators
in the systolic array, while dropping from four MXUs per TensorCore to two. Net
multiply-accumulators per TensorCore therefore doubled; the remaining 2.3× of
the observed 4.66× compute increase comes from elsewhere.

### A methodological finding

The source documentation mixes GB and GiB between generations. v5e bandwidth is
published in **GiBps** while the others use **GBps**, and capacity units differ
in the opposite direction. For v5e the reading changes machine balance from
229.3 to 246.2, a 7% difference.

That is invisible on a log-log roofline plot and material when a measurement
lands near the ridge. It is also the kind of error that silently survives into
every downstream calculation, which is the argument for transcribing units
verbatim and converting late.

## What this does not show

**Nothing here was measured.** These are theoretical peaks. Published HBM
bandwidth is an interface maximum; a streaming kernel achieves less. Peak
throughput assumes ideal issue rates. Machine balance computed this way is an
upper bound on the intensity actually required, so a kernel may become
compute-bound sooner than the table suggests.

The table also omits on-chip memory capacity (VMEM, SMEM, CMEM), which the
public documentation does not give per generation. Those sizes govern tiling
decisions directly and their absence will constrain M11 and M12.

## Next

**M2 — matmul MFU sweep.** With a peak figure available, the achieved TFLOP/s
from M0 becomes a fraction of peak, and sweeping $N$ locates the empirical
crossover to compare against the 229 or 560 predicted here.
