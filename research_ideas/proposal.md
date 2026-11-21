# Project proposal: What do distillation methods agree on?

**Few-step ability in diffusion models as one shared, nearly constant shift per layer**

| | |
|---|---|
| Author | Ömer Faruk Merey (individual project) |
| Course | Generative Models for Multimedia, METU |
| Date | 16 November 2026 (third version; earlier versions 11 and 15 November) |
| Status | Draft for discussion with the course advisor |

Supporting documents: the [study results](../docs/RESULTS.md) behind every number here,
a [plain-language walk-through](plan-explained.md), and the
[literature review](../docs/LITERATURE_REVIEW.md).

## Summary

Diffusion models can be distilled to sample in 4 steps, not 50, and the result is often
shipped as a small weight change (a LoRA adapter) added to the base model. Several
distillation methods exist, built on different training objectives. We ask what they have
in common inside the model.

A first study on nine public adapters for Stable Diffusion 1.5, from five independent
methods, gives three results:

1. **Every adapter is close to one direction per layer.** One of 64 directions holds 93%
   to 98% of each adapter's energy and keeps its prompt alignment.
2. **The methods partly agree on that direction.** Between independent methods, the
   directions have a cosine of 0.29 to 0.76. Random directions give 0.02, and style
   adapters give 0.02 to 0.07.
3. **The direction acts almost like a constant.** Replacing each layer's change with one
   fixed vector added to its output, the same for every prompt and step, keeps the full
   adapter's prompt alignment in four of the five methods.

These are small-scale results: 16 prompts, one seed, one base model, and no image-quality
metric yet. The proposal is to test them properly, explain them, and use them.

## Motivation

Step distillation is the main route to fast diffusion sampling. LCM, TCD, PCM, Hyper-SD,
Flash Diffusion, SDXL-Lightning and DMD2 each train a student to do in a few steps what
the teacher does in many. Their papers compare image quality. We found none that compares
what the trained students changed.

Whether they changed the same thing matters:

- **Understanding.** If methods with different objectives converge on one change, that
  change is a property of the base model, not of the method.
- **Cost.** Distillation takes many GPU-hours. If the change is one known direction per
  layer, most of that training may be unnecessary.
- **Practice.** People already combine, shrink and re-scale these adapters by trial and
  error. A measured account would replace folklore.

## Background

- **Few-step sampling.** A diffusion model turns noise into an image in steps. With few
  steps each jump is large and a base model gives blurry, off-prompt images.
- **Distillation.** A student copy of the model is trained to match the teacher's
  many-step result in a few steps.
- **LoRA.** The student's weight change `ΔW` for a layer is stored as a product of two
  thin matrices, so its rank is at most `r` (64 for most adapters here).
- **Singular value decomposition.** `ΔW` splits into `r` independent directions, each
  with an output side `u`, an input side `v` and a strength `s`. A rank-1 change adds
  `s · u · (v · x)` to the layer's output for an input `x`.

## Research questions

**Main question.** Is few-step ability one shared change to a diffusion model, whatever
distillation method produced it; and if so, what is that change and can it be obtained
without full distillation?

1. **Universality.** Do adapters from different methods, step counts and base models put
   their effect in the same direction per layer?
2. **Mechanism.** What does that direction do? Two candidates from the first study: a
   nearly constant shift of each layer's output, and built-in guidance.
3. **Use.** Can few-step sampling be had from the shared direction with only a few
   numbers to fit, or from a per-layer constant, at a fraction of the training cost?

## Hypotheses

Each can fail. "First study" is the 16-prompt result; "open" means untested.

| | Hypothesis | Supported if | Status |
|---|---|---|---|
| U1 | Adapters are near rank 1 | Rank 1 keeps CLIP within 0.01 of the full adapter, and holds on a quality metric | First study: 5 of 5 methods on CLIP. Quality open |
| U2 | Output directions agree across independent methods | Cosine far above the random and style-adapter references | First study: 0.29 to 0.76 against 0.02 to 0.07 |
| U3 | The agreement holds on other base models | Same test on SDXL and one transformer model | Open; needs a GPU |
| M1 | The direction acts as a constant per layer | A fixed vector per layer keeps prompt alignment and quality | First study: CLIP within 0.003 of full in 4 of 5 methods. Quality open |
| M2 | The effect is mostly built-in guidance | The adapter's change to the output lines up with the guidance direction | First study: partly. Cosine 0.2 to 0.6 at the first step, falling later |
| P1 | The shared direction is usable with few fitted numbers | The other methods' direction, re-scaled, matches the host adapter on quality | First study: with one global scale, CLIP reaches the host's level on 3 of 4 hosts. Quality open |
| P2 | A per-layer constant can be trained directly | Training only one vector per layer reaches the quality of a full adapter | Open; needs a GPU |

**What would end the main claim:** U2 or M1 failing on a quality metric with thousands of
prompts, or style adapters showing the same agreement on other base models.

## Results so far

Full tables, images and limits are in [`docs/RESULTS.md`](../docs/RESULTS.md).

| Adapter | Method family | Energy in top direction | Rank-1 recovery | CLIP: none / rank 1 / full |
|---|---|---:|---:|---|
| LCM-LoRA | Consistency | 94% | 0.87 | 0.218 / 0.321 / 0.320 |
| TCD | Trajectory consistency | 98% | 0.93 | 0.260 / 0.328 / 0.330 |
| PCM | Phased consistency, adversarial | 97% | 0.83 | 0.272 / 0.325 / 0.322 |
| Hyper-SD 4-step | Segmented consistency, adversarial, preference | 96% | 0.62 | 0.272 / 0.316 / 0.322 |
| Flash Diffusion | Distribution matching, adversarial | 93% | 0.79 | 0.218 / 0.337 / 0.335 |

- **Agreement is specific to acceleration.** Three style adapters agree with the
  acceleration adapters, and with each other, no more than random directions do.
- **Agreement is not inherited.** Adapters from different methods share no starting
  point: their weak directions overlap exactly as much as unrelated subspaces do.
- **Swapping needs a scale.** One method's direction at another's strength gives
  over-contrasted or soft images. With one global scale factor, the direction agreed by
  the other methods reaches the host's prompt alignment on three of four hosts.
- **The constant is prompt-independent.** One fixed vector per layer, estimated on half
  of the prompts, works equally well on the other half.

## Method

### A. Universality

1. **Scale up the first study** on SD 1.5: 5,000 COCO captions, 3 seeds, all nine
   adapters, with clean-FID, CLIP score and one preference metric.
2. **Repeat on SDXL** (LCM-LoRA, SDXL-Lightning, DMD2, Hyper-SD, PCM) and on one
   transformer model with public step-distillation adapters, such as FLUX.
3. **Location.** Extend the block and layer-type ablation, done so far only on LCM-LoRA,
   to every adapter. On a transformer, report by block depth.
4. **Step count.** Hyper-SD's 1, 2, 4 and 8-step adapters share a starting point and
   agree at 0.85 to 0.99. Test whether one direction with a step-dependent strength
   covers all four.

### B. Mechanism

1. **Constant shift.** Repeat the constant-per-layer test at scale. Separate what the
   constant does from what the input-dependent remainder does, by layer type and step.
2. **Where the constant comes from.** Test whether the per-layer vector can be predicted
   from the base model alone, for example from the difference between its activations on
   a 50-step path and on a 4-step path.
3. **Guidance.** Compare with an adapter trained only to remove guidance, and with the
   base model run with guidance, to separate step-skipping from built-in guidance.
4. **Relation to the time path.** A constant per layer and step is close to what MTEO
   trains (per-layer time embeddings). Test whether the two are the same change.

### C. Use

1. **Fit scales only.** Fix the shared direction and fit one strength per layer (278
   numbers), then one per block, then one in total. Compare with the full adapters.
2. **Train the constant.** Train one vector per layer with a standard distillation
   objective, and compare cost and quality with a rank-64 adapter.
3. **Transfer.** Apply a direction or constant found on the base model to fine-tuned
   versions of it, as LCM-LoRA is used in practice.

### Controls used throughout

- **Random directions** of the same size, for every truncation, swap and cosine.
- **Style adapters** on the same base model, to separate "distillation" from "LoRA".
- **Same-method references**: Hyper-SD at different step counts, and two PCM versions.
- **A starting-point check** on every new pair of adapters before calling it independent.
- **Held-out prompts** for anything estimated from data, and 3 seeds with intervals.
- **A guided baseline** (base model with guidance at the same step count).

## Order of work

The project is done by one person, in two tracks:

| Track | Covers | First task |
|---|---|---|
| Universality | Scale-up, SDXL, transformer model, quality metrics | Build the 5,000-caption evaluation and rerun the SD 1.5 study on it |
| Mechanism and use | Constant shift, guidance, scale fitting, training | Rerun the constant-shift test by layer type; set up scale fitting |

The first week goes to reading WaDi, the intruder-dimensions paper and MTEO, and to fixing
the thresholds above with the advisor.

## Timeline

Assumes the term ends in mid-January 2027.

| Weeks | Dates | Work | Output |
|---|---|---|---|
| 1 | 16-22 Nov | Reading; reproduce the first study; fix thresholds | Fixed hypotheses |
| 2-3 | 23 Nov - 6 Dec | A1 (evaluation at scale); B1 (constant shift by layer and step) | Quality table for SD 1.5 |
| 4-5 | 7-20 Dec | A3, A4; B2, B3; C1 | Mechanism results; scale-fitting result |
| 6 | 21-27 Dec | A2 on SDXL if a GPU is available | Second base model |
| 7-8 | 28 Dec - 12 Jan | Write the term report | Report and presentation |
| After | Jan - Apr | Transformer model; C2 and C3; paper draft | Submission |

**Decision point, 6 December.** If rank 1, cross-method agreement or the constant shift
fails on the quality metrics at scale, the claim is narrowed to what survives and the
paper plan is reviewed with the advisor.

## Resources

- **Available:** an Apple M4 Pro laptop with 24 GB. The first study ran on it in about
  two hours. One arm of 16 images takes under a minute.
- **Needed:** a GPU with 24 GB or more for 5,000-caption evaluation with several arms and
  seeds, for SDXL and a transformer model, and for any training.
- **Software:** PyTorch, diffusers, and the code in `scripts/study/`. All models and
  adapters are public; Flash Diffusion's licence is non-commercial.

## Risks

| Risk | Likelihood | Response |
|---|---|---|
| Results weaken on a quality metric with thousands of prompts | Medium | Decision point; report what survives |
| The constant shift is already known under another name | Medium | Search before writing: bias-only tuning (BitFit, DiffFit), MTEO, exposure-bias corrections |
| Agreement comes from shared training data, not from the task | Medium | Cannot be ruled out with public adapters; state it, and test with our own training in C2 |
| The effect is mostly built-in guidance | Medium | B3; the first study says partly |
| Results on SD 1.5 look dated | High | SDXL and a transformer model |
| "LoRA ranks can be cut" is seen as known | High | It is known. The claims are agreement, mechanism and use |
| No GPU access | Unknown | SD 1.5 work at reduced scale still makes a complete term report |

## Expected outcomes

- **Minimum (term report):** the SD 1.5 study at scale with quality metrics, the
  mechanism tests, and scale fitting.
- **Target (conference paper):** the above on SDXL and a transformer model, plus a
  directly trained per-layer constant. Candidate venues are ICCV 2027 and NeurIPS 2027;
  their deadlines are not yet announced.

## Related work

Covered at length in the [literature review](../docs/LITERATURE_REVIEW.md). The closest
papers, none yet read in full by us:

- **WaDi** ([2603.08258](https://arxiv.org/abs/2603.08258)) compares fully distilled
  checkpoints with their teachers; no adapters, no cross-method comparison.
- **MTEO** ([2603.22375](https://arxiv.org/abs/2603.22375)) trains per-layer time
  embeddings for few-step sampling, which a per-layer constant may explain.
- **The Universal Weight Subspace Hypothesis**
  ([2512.05117](https://arxiv.org/abs/2512.05117)) claims shared subspaces across style
  adapters. In our data, style adapters do not share output directions.
- **Intruder dimensions** ([2410.21228](https://arxiv.org/abs/2410.21228)): LoRA adds new
  dominant directions. Ours are new in this sense, and still agree across methods.
- **CASA** ([2605.01929](https://arxiv.org/abs/2605.01929)) studies why style adapters
  fail on step-distilled video models, from singular-value spectra.

Searches on 15 and 16 November found no paper that compares acceleration adapters across
methods, reports rank 1 for them, or reduces them to a per-layer constant. That is a
handful of searches, not proof.

## Changes from the second version

- The prerequisite and the main hypothesis of the second version were tested and held at
  small scale, so the plan moves on to explaining and using the result.
- Two questions are added: mechanism and use.
- The adapter set grew from one to nine, with three style adapters as controls.

## Relation to the original course proposal

The original proposal asked whether an LLM could find and edit the denoiser weights behind
a few-step failure. That was tested first; the write-up is in
[`docs/EXPERIMENT_1.md`](../docs/EXPERIMENT_1.md). The LLM agents rediscovered published
sampler corrections and a blind search matched them, so the advisor's doubt was justified.
This proposal keeps the weight-space question and drops the LLM.
