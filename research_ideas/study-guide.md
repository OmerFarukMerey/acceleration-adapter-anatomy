# Study guide: what distillation adapters agree on

What to read for the project, in what order, and what to look for in each paper. Written
11 November 2026 and rewritten 24 November 2026 for the current plan
([proposal.md](proposal.md)) and the current results
([`docs/RESULTS.md`](../docs/RESULTS.md)). The same papers are discussed in context in
[`docs/LITERATURE_REVIEW.md`](../docs/LITERATURE_REVIEW.md).

**How far to trust the citations.** Every arXiv number here was checked against the arXiv
listing on 24 November 2026 and opens the paper named. The descriptions are less certain:
most come from abstracts, search summaries or memory, and a few from an automated reader.
The "Known from" columns say which. No paper had been read in full by the author when
this was written.

## The project in five lines

1. **Rank 1.** Each few-step adapter is close to one direction per layer (results,
   sections 1 and 7).
2. **Agreement.** Independent methods partly agree on that direction, on four base models
   and in function (sections 2, 5, 6 and 11).
3. **A constant.** The direction acts almost like one fixed vector added to each layer's
   output (sections 4, 7 and 9).
4. **Use.** With the direction fixed, about a thousand trained strengths give few-step
   sampling (section 8).
5. **Origin.** The gradient of a plain few-step loss at the base weights already points
   along the shared direction (sections 12 and 13).

Reading has two jobs: to find out which of these is already known, and to learn how each
adapter was trained.

## First pass: eight readings

About two days. For each paper write three lines: what it claims, what it leaves open,
and which sentence of the proposal it affects. Statements of the form "this paper does
not do X" matter most for positioning and are the least reliable in our notes, so check
those first.

| # | Read | Question to answer | Affects |
|---|---|---|---|
| 1 | LoRA, [2106.09685](https://arxiv.org/abs/2106.09685), section 7 only | How do they compare the subspaces of two adapters? Our cosine measure descends from it | H1, H2 |
| 2 | LCM, [2310.04378](https://arxiv.org/abs/2310.04378), the loss; then LCM-LoRA, [2311.05556](https://arxiv.org/abs/2311.05556) | How exactly are the consistency loss and the guidance scale `w` set up? Section 13 takes the gradient of this loss | H4, section 13 |
| 3 | WaDi, [2603.08258](https://arxiv.org/abs/2603.08258), in full | Does it compare two distillation methods anywhere? What is the lowest rank it tries? Does it mention a per-layer constant? | Positioning of H1 and H2 |
| 4 | Intruder dimensions, [2410.21228](https://arxiv.org/abs/2410.21228) | Could our top direction be an intruder dimension? What does it say about full fine-tuning (compare section 6)? | H1 |
| 5 | LoRA-GA, [2407.05000](https://arxiv.org/abs/2407.05000), and LoRA-One, [2502.01235](https://arxiv.org/abs/2502.01235) | Do they show that a trained adapter's direction equals the first gradient's, and for which models? Anything on diffusion, or on several losses sharing a direction? | H4, sections 12 and 13 |
| 6 | MTEO, [2603.22375](https://arxiv.org/abs/2603.22375) | What exactly is trained, how many numbers, and what does it score at 4 steps on SD 1.5? It is the baseline named in experiment E4 | H3, H4 |
| 7 | BitFit, [2106.10199](https://arxiv.org/abs/2106.10199), and DiffFit, [2304.06648](https://arxiv.org/abs/2304.06648), skim | How is "one fixed vector per layer" different from bias tuning? | H3 |
| 8 | Method sections of TCD, [2402.19159](https://arxiv.org/abs/2402.19159); PCM, [2405.18407](https://arxiv.org/abs/2405.18407); Hyper-SD, [2404.13686](https://arxiv.org/abs/2404.13686); Flash Diffusion, [2406.02347](https://arxiv.org/abs/2406.02347) | For each: teacher, training data, LoRA rank, target layers, guidance scale, losses. Fill one table | H2, and the shared-training-data risk |

- **Reading 5 carries the most risk.** Both papers are known from abstracts only, and the
  one-page proposal already states that the direction can be read from the base model.
  "This is what LoRA theory predicts" is the most likely objection.
- **Reading 8 gives the first answer to the shared-data question.** The results list
  "shared training data is not ruled out" as a limit. The papers say what each method
  trained on.

## Background

Read these first only if a point is shaky.

- **Sampling.** DDPM ([2006.11239](https://arxiv.org/abs/2006.11239)), then DDIM
  ([2010.02502](https://arxiv.org/abs/2010.02502)), the sampler used for PCM, Hyper-SD
  and the training pilot.
- **Latent diffusion and guidance.** Stable Diffusion denoises in a compressed latent
  space ([2112.10752](https://arxiv.org/abs/2112.10752)). Classifier-free guidance runs
  the network twice per step ([2207.12598](https://arxiv.org/abs/2207.12598)); the
  adapters are run without it.
- **The metrics the results use.** CLIP ([2103.00020](https://arxiv.org/abs/2103.00020)),
  LPIPS ([1801.03924](https://arxiv.org/abs/1801.03924)), FID
  ([1706.08500](https://arxiv.org/abs/1706.08500)) in its clean-FID form
  ([2104.11222](https://arxiv.org/abs/2104.11222)), CMMD
  ([2401.09603](https://arxiv.org/abs/2401.09603)) and PickScore
  ([2305.01569](https://arxiv.org/abs/2305.01569)). Know what each one cannot see.
- **Linear algebra.** The Eckart-Young theorem (why keeping the top singular directions
  is the best low-rank approximation) and principal angles between subspaces.

## Second pass, by topic

"Known from": **reader** = read in full or in main text by an automated reader,
**search** = abstract, summary or snippet from a web search, **memory** = description
written from memory.

### The methods behind the adapters

| Paper | Why | Known from |
|---|---|---|
| Progressive distillation, [2202.00512](https://arxiv.org/abs/2202.00512); consistency models, [2303.01469](https://arxiv.org/abs/2303.01469) | How a student learns to skip steps | memory |
| Guidance distillation, Meng et al., [2210.03142](https://arxiv.org/abs/2210.03142) | Guidance built into the student; needed for the guidance question | memory |
| DMD, [2311.18828](https://arxiv.org/abs/2311.18828); DMD2, [2405.14867](https://arxiv.org/abs/2405.14867) | Distribution matching. DMD2 is a method in the SDXL test and a full fine-tune in section 6 | memory |
| ADD / SDXL-Turbo, [2311.17042](https://arxiv.org/abs/2311.17042); SDXL-Lightning, [2402.13929](https://arxiv.org/abs/2402.13929) | Adversarial losses. Their gradient direction is the untested case of section 13 | memory |
| TDD, [2409.01347](https://arxiv.org/abs/2409.01347) | A method in the SDXL and FLUX comparisons | memory |
| Rectified flow, [2209.03003](https://arxiv.org/abs/2209.03003); InstaFlow, [2309.06380](https://arxiv.org/abs/2309.06380); PeRFlow, [2405.07510](https://arxiv.org/abs/2405.07510) | Flow straightening. The 2-rectified-flow model is the weakest match in section 6 | memory; PeRFlow search |
| SDXL, [2307.01952](https://arxiv.org/abs/2307.01952); SD3, [2403.03206](https://arxiv.org/abs/2403.03206) | The other base models | memory |

### LoRA and what its spectra mean

| Paper | Why | Known from |
|---|---|---|
| Intrinsic dimension, Aghajanyan et al., [2012.13255](https://arxiv.org/abs/2012.13255) | The general claim that fine-tuning moves in few directions | memory |
| Task arithmetic, Ilharco et al., [2212.04089](https://arxiv.org/abs/2212.04089) | The vocabulary of weight changes as vectors; LCM-LoRA's "acceleration vector" comes from it | memory |
| Weight-space geometry of offline reasoning training, [2606.23740](https://arxiv.org/abs/2606.23740) | Reports that the output side of a LoRA agrees across seeds and the input side does not. Compare our output and input sides | search |
| Compress then Serve, [2407.00066](https://arxiv.org/abs/2407.00066); 1LoRA (Quercia et al., WACV 2026) | Truncating adapters and rank-one adapters are prior art; cite them as such | search |

### Weights of distilled models

| Paper | Why | Known from |
|---|---|---|
| CASA, [2605.01929](https://arxiv.org/abs/2605.01929) | Distillation and LoRA rotate singular subspaces and leave singular values nearly unchanged | search |
| DART, [2609.20051](https://arxiv.org/abs/2609.20051) | Singular values and subspace rotation when reusing LoRAs on few-step video models | reader |
| Gandikota and Bau, [2503.10637](https://arxiv.org/abs/2503.10637) | What distillation changes in a model's behaviour | search |
| Zheng and Yang, [2405.20750](https://arxiv.org/abs/2405.20750) | One-step distillation with most convolutions frozen. Compare with section 10, where convolutions matter most | search |
| B-LoRA, [2403.14572](https://arxiv.org/abs/2403.14572) | The precedent for block-level analysis | search |
| FreeU, [2309.11497](https://arxiv.org/abs/2309.11497) | Backbone against skip features; context for "the up blocks carry it" | memory |

### Constants and few trained numbers

| Paper | Why | Known from |
|---|---|---|
| ReSeed, [2610.37110](https://arxiv.org/abs/2610.37110) | Adds a constant bias per block to a distilled student, for diversity | reader |
| Epsilon Scaling, [2308.15321](https://arxiv.org/abs/2308.15321); Time-Shift Sampler, [2305.15583](https://arxiv.org/abs/2305.15583) | Step-dependent corrections in the sampler; related to a constant per step | search |

### Shared structure across separately trained networks

| Paper | Why | Known from |
|---|---|---|
| EigenLoRAx, [2502.04700](https://arxiv.org/abs/2502.04700); the Universal Weight Subspace Hypothesis, [2512.05117](https://arxiv.org/abs/2512.05117); Share, [2602.06043](https://arxiv.org/abs/2602.06043) | The competing general claim that LoRAs share a subspace (see below) | search; the second reader |
| SVCCA, [1706.05806](https://arxiv.org/abs/1706.05806); CKA, [1905.00414](https://arxiv.org/abs/1905.00414) | Similarity measures and how they mislead without baselines | memory |
| Platonic Representation Hypothesis, [2405.07987](https://arxiv.org/abs/2405.07987); Git Re-Basin, [2209.04836](https://arxiv.org/abs/2209.04836) | Convergence of separately trained models; why a shared frozen base makes our comparison easier | memory |

### Evaluation

| Paper | Why | Known from |
|---|---|---|
| HPSv2, [2306.09341](https://arxiv.org/abs/2306.09341); ImageReward, [2304.05977](https://arxiv.org/abs/2304.05977) | Other preference scores; a second one would back up PickScore | memory |
| Guidance Matters, [2602.22570](https://arxiv.org/abs/2602.22570); AGD, [2503.07274](https://arxiv.org/abs/2503.07274) | Few-step evaluation is confounded by built-in guidance | search |

### How the shared-subspace papers relate to this project

EigenLoRAx takes many trained LoRAs for one base model, runs a principal-component
analysis across them, and finds a small subspace they share. The Universal Weight Subspace
Hypothesis turns that into a general claim.

- **"Adapters share a subspace" is therefore not a headline on its own.** The project's
  claim is narrower: adapters that differ by distillation objective share a direction that
  style adapters do not.
- **Their prediction for our control held only in part.** Style adapters are concentrated
  too (90% to 95% of the energy in the top direction), but they do not share output
  directions with each other or with the acceleration adapters (0.02 to 0.07).
- **It offers a method.** Its extraction of shared directions is an alternative to our
  consensus direction, and training only coefficients on shared directions is close to the
  pilot of section 8.

## What reading cannot supply

- **Scale.** FID on 5,000 captions with several seeds, function tests on SD3 and FLUX, and
  the gradients of adversarial and distribution-matching losses need a GPU.
- **The shared-data question in full.** The papers say what each method trained on. Only
  training our own adapters on separate data would settle it.
- **Writing.** Read two recent analysis papers from the target venue for structure. The
  intruder-dimensions paper is a good model.

## Getting to know the code

1. Run `.venv/bin/python scripts/study/s1_weights.py`. It compares weights only and
   generates no image. Read `scripts/study/adapters.py` until `factorize` and the
   bench's `install` are clear.
2. Read `scripts/study/s14_gradient.py` next to section 12 of the results. It is the
   shortest route from the base model to the shared direction.
3. The other scripts and their order are listed at the end of
   [`docs/RESULTS.md`](../docs/RESULTS.md).

## How the idea got here

- **11 November.** A pilot on LCM-LoRA alone said: effectively rank 4, mostly in the
  decoder, time path idle. That version of the idea is kept in
  [01-anatomy-of-acceleration-adapters.md](01-anatomy-of-acceleration-adapters.md).
- **15 and 16 November.** The first study on nine adapters sharpened rank 4 to rank 1 and
  found the agreement between methods and the constant.
- **19 to 21 November.** Other base models, full fine-tunes, quality metrics, the transfer
  test, the training pilot and the gradient direction (sections 5 to 13).
