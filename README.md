# TPU Performance Lab

**📖 [Read the documentation](https://example.github.io/tpu-performance-lab/)**

Experiments in measuring and closing the gap between a TPU's peak throughput
and what real kernels actually achieve.

A chip's spec sheet quotes peak FLOP/s. Large model training commonly runs at
20–50% of that. This repo is a systematic attempt to understand where the rest
goes — starting from first measurements and building toward custom kernels.

Everything here is reproducible: each experiment is a script that records its
environment alongside its numbers, and each result is written up with its
method, its interpretation, and what it does *not* show.

---

## Why this exists

Compute has outpaced memory bandwidth for decades. On a current TPU the ratio
is roughly a few hundred FLOPs per byte of HBM traffic, which means a kernel
must reuse every byte it fetches hundreds of times to keep the matrix units
busy. Kernels that do not are memory-bound, and no amount of extra FLOP/s on
the next chip will help them.

Finding out which regime a given kernel is in — and moving it — is the work.

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
| M1 | TPU generation spec sheet | What are the real peak FLOP/s, HBM bandwidth, and on-chip memory figures? |
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
| **P1** | [Fused attention kernel](p1-pallas-attention/PLAN.md) | Can a hand-written kernel beat the compiler, and why? |

---

## Running

Requires a TPU. The quickest path is Google Colab — set
*Runtime → Change runtime type → TPU*; JAX comes preinstalled.

```bash
pip install -r requirements.txt
python mini/m0_hello_tpu.py
```

Scripts run standalone and add the repo root to `sys.path`, so no install step
is needed for `tpuperf`.

---

## Measurement conventions

Naive JAX benchmarks are wrong in three predictable ways, and
[`tpuperf/bench.py`](tpuperf/bench.py) handles all three:

- **Dispatch is asynchronous.** A call returns a future; without
  `block_until_ready` you are timing dispatch overhead.
- **`jit` compiles on first call.** Warm-up runs are untimed.
- **Single-shot timing is noisy.** Results are the median over repeated runs,
  reported with minimum and standard deviation.

Inputs are random rather than constant, so XLA cannot fold away the operation
under test.

Reported HBM traffic is a *lower bound*: each input read once, each output
written once. Real kernels move more when tiles spill. That makes it the right
denominator for arithmetic intensity and a useful upper bound on efficiency.

---

## Results

Populated as experiments land. Each entry links to the script that produced it
and the JSON record in `results/`.

| Experiment | Chip | Headline number |
|---|---|---|
| M0 | — | pending |
