# The roofline model

A chip has two ceilings: how fast it can compute, and how fast it can move
data. Any kernel is limited by one of them. The roofline model says which, and
by how much, from two numbers you can look up and one you can calculate.

## Machine balance

Take a TPU generation with roughly 200 TFLOP/s of bf16 throughput and 820 GB/s
of HBM bandwidth. The ratio has units of FLOP per byte:

$$
\text{machine balance} = \frac{200 \times 10^{12}\ \text{FLOP/s}}
{820 \times 10^{9}\ \text{byte/s}} \approx 244\ \frac{\text{FLOP}}{\text{byte}}
$$

Read it as a requirement. In the time it takes to fetch one byte from HBM, the
chip can perform about 244 floating-point operations. **A kernel that does not
reuse each fetched byte at least 244 times will leave the matrix units idle**,
no matter how well it is written.

!!! warning "Use real numbers"

    The figures above are illustrative. Peak throughput and bandwidth differ
    substantially across TPU generations, and vendor peak figures often assume
    a specific dtype and sparsity assumption. Collecting the actual values is
    [M1](../experiments/index.md), and it comes first for exactly this reason.

## Arithmetic intensity

The kernel-side counterpart is arithmetic intensity: FLOPs performed per byte
of memory traffic.

$$
I = \frac{\text{FLOPs}}{\text{bytes moved}}
$$

Comparing $I$ against machine balance classifies the kernel:

- $I <$ balance → **memory-bound**. The chip is waiting on HBM.
- $I >$ balance → **compute-bound**. The matrix units are the limit.

Attainable performance is the minimum of the two ceilings:

$$
P_{\text{attainable}} = \min\left(P_{\text{peak}},\ BW \times I\right)
$$

Plotted on log-log axes, that is a diagonal line rising with bandwidth, then a
flat ceiling at peak — the shape the model is named for.

<figure markdown>
<svg viewBox="0 0 720 420" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Roofline model: attainable performance versus arithmetic intensity on log-log axes" style="max-width:100%;height:auto;color:inherit">
  <defs>
    <clipPath id="plot"><rect x="90" y="40" width="590" height="310"/></clipPath>
  </defs>

  <!-- regions -->
  <g clip-path="url(#plot)">
    <rect x="90" y="40" width="400" height="310" fill="#e08a1e" opacity="0.05"/>
    <rect x="490" y="40" width="190" height="310" fill="#2f9e8f" opacity="0.06"/>
  </g>

  <!-- grid -->
  <g stroke="currentColor" opacity="0.14" stroke-width="1">
    <line x1="208" y1="40" x2="208" y2="350"/>
    <line x1="326" y1="40" x2="326" y2="350"/>
    <line x1="444" y1="40" x2="444" y2="350"/>
    <line x1="562" y1="40" x2="562" y2="350"/>
    <line x1="90" y1="272.5" x2="680" y2="272.5"/>
    <line x1="90" y1="195" x2="680" y2="195"/>
    <line x1="90" y1="117.5" x2="680" y2="117.5"/>
  </g>

  <!-- axes -->
  <g stroke="currentColor" opacity="0.5" stroke-width="1.4" fill="none">
    <line x1="90" y1="350" x2="680" y2="350"/>
    <line x1="90" y1="40" x2="90" y2="350"/>
  </g>

  <!-- ridge marker -->
  <line x1="490" y1="94" x2="490" y2="350" stroke="currentColor" opacity="0.35" stroke-width="1.2" stroke-dasharray="4 4"/>

  <!-- roofline -->
  <polyline points="100,350 490,94 680,94" fill="none" stroke="#e08a1e" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>

  <!-- measured points -->
  <g fill="#2f9e8f">
    <circle cx="101" cy="349" r="5"/>
    <circle cx="472" cy="106" r="5"/>
    <circle cx="649" cy="94" r="5"/>
  </g>

  <!-- labels -->
  <g fill="currentColor" font-family="ui-sans-serif,-apple-system,Segoe UI,sans-serif" font-size="12.5">
    <g opacity="0.62" text-anchor="middle">
      <text x="90" y="370">0.1</text>
      <text x="208" y="370">1</text>
      <text x="326" y="370">10</text>
      <text x="444" y="370">100</text>
      <text x="562" y="370">1K</text>
      <text x="680" y="370">10K</text>
    </g>
    <g opacity="0.62" text-anchor="end">
      <text x="80" y="354">0.1</text>
      <text x="80" y="277">1</text>
      <text x="80" y="199">10</text>
      <text x="80" y="122">100</text>
      <text x="80" y="44">1000</text>
    </g>
    <text x="385" y="396" text-anchor="middle" opacity="0.8" font-size="13">Arithmetic intensity — FLOP / byte</text>
    <text x="20" y="195" text-anchor="middle" opacity="0.8" font-size="13" transform="rotate(-90 20 195)">Attainable TFLOP/s</text>
  </g>

  <g font-family="ui-sans-serif,-apple-system,Segoe UI,sans-serif" font-size="12.5">
    <text x="600" y="82" fill="#e08a1e" text-anchor="middle" font-weight="600">peak compute</text>
    <text x="215" y="250" fill="#e08a1e" font-weight="600" transform="rotate(-32 215 250)">bandwidth limit</text>
    <text x="490" y="386" fill="currentColor" opacity="0.72" text-anchor="middle" font-size="11.5">machine balance ≈ 244</text>
    <g fill="currentColor" opacity="0.78" font-size="11.5">
      <text x="112" y="336">elementwise · I = 0.125</text>
      <text x="462" y="128" text-anchor="end">matmul N=256 · I ≈ 171</text>
      <text x="640" y="78" text-anchor="end">matmul N=8192 · I ≈ 5461</text>
    </g>
    <text x="240" y="120" fill="currentColor" opacity="0.5" font-size="11.5">memory-bound</text>
    <text x="560" y="300" fill="currentColor" opacity="0.5" font-size="11.5">compute-bound</text>
  </g>
</svg>
<figcaption>Schematic roofline. Illustrative figures — 200 TFLOP/s peak, 820 GB/s bandwidth.</figcaption>
</figure>

## Worked examples

### Elementwise addition

`y = x + 1` in fp32 reads four bytes, writes four bytes, and performs one
operation:

$$
I = \frac{1}{8} = 0.125\ \text{FLOP/byte}
$$

That is roughly 1/2000 of machine balance. This kernel will reach a fraction
of a percent of peak, and there is nothing to be done about it in isolation —
the only fix is to stop running it in isolation. Fusing it into an adjacent
operation removes the round trip to HBM entirely, which is why fusion is the
first thing any ML compiler tries.

### Matrix multiplication

For an $N \times N$ matmul: $2N^3$ FLOPs, and at minimum $3N^2$ elements of
traffic (two inputs read, one output written).

$$
I = \frac{2N^3}{3N^2 \cdot s} = \frac{2N}{3s}
$$

where $s$ is the element size in bytes. Intensity grows **linearly with $N$** —
so the same operation sits on either side of the ridge depending only on size.

| $N$ | $I$ (fp32) | Regime |
|---|---|---|
| 256 | ≈ 171 | memory-bound |
| 1024 | ≈ 683 | compute-bound |
| 8192 | ≈ 5461 | compute-bound, comfortably |

This is the single most useful intuition the model provides: **small matmuls
are not slow because they are small. They are slow because they are
memory-bound.** Batching them, or fusing them, moves them right along the
x-axis.

Finding where the crossover actually falls on real hardware is
[M2](../experiments/index.md).

## What the model deliberately ignores

Roofline is a bound, not a prediction. A kernel can land well below its
roofline for reasons the model does not represent:

- **Traffic that was not counted.** The $3N^2$ figure assumes each input is
  read exactly once. If tiles do not fit in on-chip memory they are re-read,
  and real intensity is lower than the paper calculation.
- **Failure to overlap.** The model assumes transfer and compute proceed
  concurrently. A kernel that fetches, then computes, then fetches achieves
  neither ceiling.
- **Collective communication.** In multi-chip settings, time spent in
  all-reduce is neither compute nor local HBM traffic.
- **Pipeline bubbles and idle gaps.** Visible in a profiler; invisible to the
  model.

Each of those is a separate experiment ([M6](../experiments/index.md) onward),
and each exists because a roofline calculation disagreed with a measurement.
That disagreement is the useful part.

## Related

- [MFU](../reference/glossary.md#mfu) — the same idea expressed as a single
  utilisation percentage
- [Measurement conventions](../reference/measurement.md) — how these numbers
  are collected without lying to yourself
