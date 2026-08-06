# P1 — Fused Attention Kernel in Pallas

Can a hand-written kernel beat what XLA produces for attention, and if so,
which hardware property explains the difference?

Attention is the right target: it is the dominant cost in transformer
workloads, and its arithmetic intensity depends on sequence length in a way
that puts it on both sides of the roofline as the shape changes.

## Approach

1. Baseline the pure-JAX implementation and capture an XProf trace.
2. Roofline analysis — compute arithmetic intensity, classify the regime.
3. Write the Pallas kernel: block specs, HBM → VMEM tiling, overlapping
   transfer with compute.
4. Verify numerics against the reference implementation before timing anything.
5. Sweep block sizes to produce a performance heat map.
6. Use XProf kernel profiling to explain *why* the best configuration wins.

## Status

Not started. Prerequisites are M11 and M12 — the Pallas syntax and the
reduction-shaped tiling problem are both easier to learn on smaller kernels.

If the attention kernel proves too large a first step, the fallback target is
a quantised (int8) matmul, which exercises the same tiling and pipelining
concepts with a simpler data flow.

## Note on outcomes

A kernel that fails to beat XLA is still a result, provided the roofline
analysis explains the failure. The compiler is good; understanding precisely
where it is already optimal is as informative as beating it.
