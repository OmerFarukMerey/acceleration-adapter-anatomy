# Anatomy of Acceleration Adapters

What do different distillation methods change inside a diffusion model, and is it the same
thing? Few-step distillation is often shipped as a low-rank weight change on a base model,
such as LCM-LoRA. This repo takes those changes apart in weight space and compares them
across methods.

It is a term project for Generative Models for Multimedia at METU. The proposal is in
[`research_ideas/proposal.md`](research_ideas/proposal.md) and the results so far are in
[`docs/RESULTS.md`](docs/RESULTS.md).

## Current direction

**Status: first study done on Stable Diffusion 1.5 at small scale. Quality metrics, larger
models and training have not started.**

The study compares nine public acceleration adapters from five independent methods (LCM,
TCD, PCM, Hyper-SD, Flash Diffusion), with three style adapters as controls. It uses 4
steps, 16 prompts and one seed.

- **Each adapter is close to one direction per layer.** The strongest of 64 directions
  holds 93% to 98% of the energy and keeps the full adapter's prompt alignment.
- **Independent methods partly agree on that direction.** Cosine 0.29 to 0.76 between
  methods, against 0.02 for random directions and 0.02 to 0.07 for style adapters.
- **The direction acts almost like a constant.** One fixed vector added to each layer's
  output, the same for every prompt and step, keeps prompt alignment within 0.003 of the
  full adapter in four of five methods.

| Adapter | Energy in top direction | CLIP: no adapter | Top direction only | One constant per layer | Full adapter |
|---|---:|---:|---:|---:|---:|
| LCM-LoRA | 94% | 0.218 | 0.321 | 0.317 | 0.320 |
| TCD | 98% | 0.260 | 0.328 | 0.330 | 0.330 |
| PCM | 97% | 0.272 | 0.325 | 0.325 | 0.322 |
| Hyper-SD 4-step | 96% | 0.272 | 0.316 | 0.319 | 0.322 |
| Flash Diffusion | 93% | 0.218 | 0.337 | 0.313 | 0.335 |

![Pairwise agreement of top output directions between twelve adapters](results/study/s1_agreement.png)

![TCD adapter. Columns: no adapter, one constant per layer, one constant per layer and step, coefficient averaged over positions, top direction only, full adapter](results/study/s3_mechanism_tcd.png)

*TCD adapter at 4 steps. Columns: no adapter, one constant per layer, one constant per
layer and step, coefficient averaged over positions, top direction only, full adapter.*

These are prompt-alignment and closeness measures on 16 prompts, not image quality. The
next steps are quality metrics on thousands of prompts, SDXL and a transformer model, and
training the shared direction directly. A decision point is set for 6 December 2026.

| Read | For |
|---|---|
| [`research_ideas/proposal.md`](research_ideas/proposal.md) | Questions, hypotheses, plan, division of work and timeline |
| [`docs/RESULTS.md`](docs/RESULTS.md) | Every result so far, with tables, images and limits |
| [`research_ideas/plan-explained.md`](research_ideas/plan-explained.md) | The plan in plain language |
| [`docs/LITERATURE_REVIEW.md`](docs/LITERATURE_REVIEW.md) | Background and related work |
| [`research_ideas/study-guide.md`](research_ideas/study-guide.md) | Reading list and reading order |

## Reproduce

Tested on an Apple M4 Pro with 24 GB, on MPS at batch size 1.

```bash
python3.13 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/pytest

.venv/bin/python scripts/study/s1_weights.py     # weight comparison; no image generation
.venv/bin/python scripts/study/s2_function.py    # rank-1 and swap tests
.venv/bin/python scripts/study/s3_mechanism.py   # constant-shift and guidance tests
```

The scripts need Stable Diffusion 1.5 and download about 2 GB of adapters on first use.
The full list of scripts and their run time is at the end of
[`docs/RESULTS.md`](docs/RESULTS.md).

## Layout

```
research_ideas/              Proposal, plain-language plan and study guide
docs/RESULTS.md              Results of the first study
docs/LITERATURE_REVIEW.md    Background and related work
scripts/study/               The cross-method study (adapters.py holds the adapter list)
results/study/               Its outputs: JSON tables and image sheets
scripts/pilots/              Earlier pilots, including the first LCM-LoRA dissection
results/pilots/              Their outputs
src/wspd/metrics.py          LPIPS, CLIP and spectral metrics
docs/EXPERIMENT_1.md         The project's earlier experiment, kept as a record
runs/                        Caches and generated images (gitignored)
```

## Earlier work

The repo began with a different experiment, on LLM-driven weight editing. Its write-up,
code and results are kept in [`docs/EXPERIMENT_1.md`](docs/EXPERIMENT_1.md).
