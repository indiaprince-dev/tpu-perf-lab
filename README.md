# TPU Performance Lab

**[Documentation](https://indiaprince-dev.github.io/tpu-perf-lab/)**

Experiments in measuring and closing the gap between a TPU's peak throughput
and the throughput real kernels achieve.

A chip's datasheet quotes peak FLOP/s. Large model training commonly sustains
20–50% of it. This repository is a systematic attempt to account for the
difference, beginning with first measurements and building toward custom
kernels.

Each experiment is a script that records its environment alongside its numbers,
and each result is written up with its method, its interpretation, and what it
does not show.

---

## Premise

Compute throughput has outpaced memory bandwidth for several decades. On a
current TPU the ratio is a few hundred FLOPs per byte of HBM traffic, so a
kernel must reuse every byte it fetches hundreds of times to keep the matrix
units occupied. Kernels that do not are memory-bound, and additional FLOP/s on
the next chip does not help them.

Determining which regime a kernel occupies, and moving it, is the subject of
these experiments.

---

## Structure

```
docs/             documentation site — concepts, experiment write-ups, reference
tpuperf/          measurement helpers (timing, environment capture, plotting)
mini/             M0–M12: small, self-contained experiments
p1-pallas-attention/   fused attention kernel in Pallas
results/          JSON artifacts and figures emitted by each experiment
```

Every number in this repo comes from a script in `mini/` or a project
directory, and every script writes a JSON record to `results/`.

### Documentation

The site is the primary artifact; the scripts are how its numbers are
produced. Built with MkDocs Material and deployed to GitHub Pages on push.

```bash
pip install -r requirements-docs.txt
mkdocs serve
```

---

## Experiments

### Tier 0 — Establishing the ground truth

| | Experiment | Question it answers |
|---|---|---|
| **M0** | [Hello TPU](mini/m0_hello_tpu.py) | Does the environment work, and what does one large matmul actually achieve? |
| **M1** | [Specification table](mini/m1_spec_table.py) | What are the peak FLOP/s, HBM bandwidth, and on-chip figures per generation? |
| M2 | matmul MFU sweep | Where is the memory-bound / compute-bound crossover? |
| M3 | CPU vs GPU vs TPU | How does the same operation behave across architectures? |

### Tier 1 — Measurement and modelling

| | Experiment | Question it answers |
|---|---|---|
| M4 | Roofline calculator | Can I predict achieved performance before running the kernel? |
| M5 | Transformer FLOP/byte accounting | Does hand analysis of a transformer layer match measurement? |
| M6 | XProf trace: finding idle gaps | Where does the time go that is neither compute nor transfer? |
| M7 | Reading HLO dumps | What does XLA actually fuse? |

### Tier 2 — Parallelism and precision

| | Experiment | Question it answers |
|---|---|---|
| M8 | Sharding strategies | How much does the collective communication cost? |
| M9 | bf16 vs fp32 | What is the real speed / accuracy trade? |
| M10 | Rematerialisation policies | Where is the memory–compute Pareto frontier? |

### Tier 3 — Kernels

| | Experiment | Question it answers |
|---|---|---|
| M11 | First Pallas kernel (elementwise) | How do block specs and the memory hierarchy work? |
| M12 | Pallas LayerNorm | How do reductions change the tiling problem? |
| **P1** | [Fused attention kernel](p1-pallas-attention/README.md) | Can a hand-written kernel beat the compiler, and why? |

---

## Running

Requires a TPU. The simplest option is Google Colab: select
*Runtime → Change runtime type → TPU*; JAX comes preinstalled.

```bash
pip install -r requirements.txt
python mini/m0_hello_tpu.py
```

Scripts run standalone and add the repo root to `sys.path`, so no install step
is needed for `tpuperf`.

---

## Measurement conventions

Naive JAX benchmarks are wrong in three predictable ways, each addressed by
[`tpuperf/bench.py`](tpuperf/bench.py):

- **Dispatch is asynchronous.** A call returns a future, so timing without
  `block_until_ready` measures dispatch overhead.
- **`jit` compiles on the first call.** Warm-up runs are untimed.
- **Single-shot timing is noisy.** Results are the median over repeated runs,
  reported with minimum and standard deviation.

Inputs are random rather than constant, so XLA cannot fold away the operation
under test.

Reported HBM traffic is a lower bound: each input read once, each output
written once. Real kernels move more when tiles spill, which makes this the
appropriate denominator for arithmetic intensity.

---

## Results

Populated as experiments land. Each entry links to the script that produced it
and the JSON record in `results/`.

| Experiment | Result |
|---|---|
| [M0](https://indiaprince-dev.github.io/tpu-perf-lab/experiments/m0-hello-tpu/) | A 4096³ bf16 matmul reaches **76.8% of peak on TPU v5e** (151.3 of 197 TFLOP/s). The same source measures an emulated fallback on a Turing GPU, which has no bf16 path. |
| [M1](https://indiaprince-dev.github.io/tpu-perf-lab/experiments/m1-spec-sheet/) | Machine balance spans 166 (v5p) to 560 (v6e) FLOP/byte, a factor of 3.4. Roofline conclusions do not transfer between generations. |
