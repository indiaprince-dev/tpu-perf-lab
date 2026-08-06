# Experiments

Each experiment answers one question, is small enough to complete in a sitting,
and writes a JSON record to `results/` alongside its figures.

The sequence is constrained rather than arbitrary. Arithmetic intensity cannot
be computed without the chip's bandwidth and peak figures, so the specification
table precedes the sweep, and the sweep precedes any kernel work.

---

## Tier 0 · Ground truth

| | Experiment | Question | Status |
|---|---|---|---|
| **M0** | [Hello TPU](m0-hello-tpu.md) | Does the environment work, and what does one large matmul achieve? | code written |
| **M1** | [TPU specification table](m1-spec-sheet.md) | What are the peak FLOP/s, HBM bandwidth, and on-chip figures per generation? | **complete** |
| M2 | matmul MFU sweep | Where does the memory-bound / compute-bound crossover actually fall? | planned |
| M3 | CPU vs GPU vs TPU | How does one operation behave across three architectures? | planned |

## Tier 1 · Measurement and modelling

| | Experiment | Question | Status |
|---|---|---|---|
| M4 | Roofline calculator | Can attainable performance be predicted before running the kernel? | planned |
| M5 | Transformer FLOP/byte accounting | Does hand analysis of a layer match measurement? | planned |
| M6 | Finding idle gaps in a trace | Where does time go that is neither compute nor transfer? | planned |
| M7 | Reading HLO dumps | What does XLA actually fuse? | planned |

## Tier 2 · Parallelism and precision

| | Experiment | Question | Status |
|---|---|---|---|
| M8 | Sharding strategies | What does the collective communication cost? | planned |
| M9 | bf16 vs fp32 | What is the real speed / accuracy trade? | planned |
| M10 | Rematerialisation policies | Where is the memory–compute Pareto frontier? | planned |

## Tier 3 · Kernels

| | Experiment | Question | Status |
|---|---|---|---|
| M11 | First Pallas kernel | How do block specs and the memory hierarchy work? | planned |
| M12 | Pallas LayerNorm | How do reductions change the tiling problem? | planned |
| **P1** | Fused attention kernel | Can a hand-written kernel beat the compiler, and why? | planned |

---

## Page format

Every experiment page follows the same structure, which keeps results
comparable and the reasoning auditable.

**Question.** One sentence, falsifiable.
**Method.** What was run, on what hardware, with which parameters.
**Results.** Numbers and figures, with the raw JSON linked.
**Interpretation.** What the numbers mean, including what they do not show.
**Reproduce.** The exact command.

!!! note "Negative results"

    An experiment that fails to show what was expected is published on the same
    terms as one that succeeds, provided the analysis accounts for the outcome.
    A hand-written kernel that loses to XLA is informative: it locates where the
    compiler is already optimal.
