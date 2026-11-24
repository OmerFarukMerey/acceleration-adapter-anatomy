# Project proposal: What do distillation methods agree on?

**Few-step ability in diffusion models as one shared, nearly constant shift per layer**

| | |
|---|---|
| Author | Ömer Faruk Merey (individual project) |
| Course | Generative Models for Multimedia, METU |
| Date | 24 November 2026 (fourth version; earlier versions 11, 15 and 16 November) |
| Status | Long version of the [one-page course proposal](proposal-one-page.pdf), which was due on 22 November 2026. Same hypotheses, with the plan, timeline and risks |

Supporting documents: the [study results](../docs/RESULTS.md) behind every number here,
a [plain-language walk-through](plan-explained.md), the
[literature review](../docs/LITERATURE_REVIEW.md) and the [study guide](study-guide.md).

## Summary

Diffusion models can be distilled to sample in 4 steps, not 50, and the result is often
shipped as a small weight change (a LoRA adapter) added to the base model. Several
distillation methods exist, built on different training objectives. We ask what they have
in common inside the model.

Studies on public adapters, run on one laptop from 15 to 21 November, give five results:

1. **Every adapter is close to one direction per layer.** One of 64 directions holds 93%
   to 98% of each adapter's energy. Keeping only that one costs at most 0.05 PickScore
   points on four of five methods.
2. **Independent methods partly agree on that direction.** Cosine 0.29 to 0.76 on Stable
   Diffusion 1.5, against 0.02 to 0.07 for random directions and style adapters. The same
   holds in weights on SDXL, SD3 Medium and FLUX.1-dev.
3. **The agreement is functional.** An adapter built only from the other four methods
   matches the host's own adapter on three of five methods and keeps about 70% of its gain
   on two.
4. **The direction acts almost like a constant.** One fixed vector added to each layer's
   output, the same for every prompt and step, costs 0.01 to 0.15 PickScore points on four
   of five methods, where removing the adapter costs 2.8 to 3.4.
5. **The direction can be found and used without the adapters.** The gradient of a plain
   few-step loss at the base weights agrees with it at 0.44 (random 0.03). Training 1,112
   strengths on that gradient direction raises PickScore from 18.77 to 20.75; public
   adapters score 21.3 to 21.9.

**Scale.** Weights on four base models. Function on SD 1.5 (500 COCO captions) and SDXL
(100 captions), one seed each. Not done: FID on 5,000 captions with several seeds,
function tests on SD3 and FLUX, and the gradients of adversarial and distribution-matching
losses. These need a GPU.

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
2. **Mechanism.** What does that direction do, and where does it come from?
3. **Use.** Can few-step sampling be had from the shared direction with only a few
   numbers to fit, at a fraction of the training cost?

## Hypotheses

The four hypotheses of the one-page proposal. Each can fail. In earlier versions they
were called U1 to U3 (now H1 and H2), M1 (H3), and P1 and P2 (H4).

| | Hypothesis | Supported if | Status on 24 November |
|---|---|---|---|
| H1 | Each adapter's weight change is close to rank 1 in every layer | Keeping only the top direction keeps the full adapter's score on a quality metric | Held on SD 1.5 at 500 captions: rank 1 costs at most 0.05 PickScore points on four of five methods and 0.20 on Flash Diffusion, FID unchanged. On SDXL (100 captions) it holds for three of four methods; Hyper-SD loses 0.23. Fully fine-tuned models are not rank 1 |
| H2 | Adapters from independent methods agree on that direction far more than random directions or style LoRAs do | Cosine far above both references on several base models; a direction taken from other methods works on held-out captions | Weights: held on SD 1.5, SDXL, SD3 Medium and FLUX.1-dev. Function: held on three of five hosts, about 70% of the gain on the other two (100 held-out captions, SD 1.5 only) |
| H3 | The change acts mostly as one fixed vector added to each layer's output, independent of the prompt | A fixed vector per layer keeps the full adapter's quality | Held for four of five methods on SD 1.5 (500 captions) and four of four on SDXL (100 captions). Flash Diffusion loses 0.7 points |
| H4 | With the direction known, few-step sampling can be reached by training only a strength or a constant per layer on a frozen base model | Trained strengths approach the public adapters and beat the same budget on a random direction | Pilot positive: PickScore 18.77 to 20.90 with 1,112 strengths, 19.76 on a random direction. With the direction from a loss gradient: 20.75. The public adapters score 21.3 to 21.9 |

**A side question: how much of the effect is built-in guidance?** Partly. The adapters'
change to the output lines up with the guidance direction at the first step (cosine 0.20
to 0.61) and less later (0.10 to 0.18). The base model with guidance at 4 steps scores 0.8
to 1.4 PickScore points below the adapters.

**What would end the main claim.** H2 falling to the control level, or H3 giving an FID
far from the full adapter's, on 5,000 captions with several seeds; style adapters showing
the same agreement in a larger control set; or the function tests failing on SD3 and
FLUX. The numeric thresholds are to be agreed with the advisor before the 5,000-caption
run.

## Results so far

Full tables, images and limits are in [`docs/RESULTS.md`](../docs/RESULTS.md).

| Adapter (SD 1.5) | Method family | Energy in top direction | PickScore: no adapter / top direction / one constant per layer / full |
|---|---|---:|---|
| LCM-LoRA | Consistency | 94% | 17.85 / 21.19 / 21.06 / 21.21 |
| TCD | Trajectory consistency | 98% | 18.28 / 21.30 / 21.28 / 21.32 |
| PCM | Phased consistency, adversarial | 97% | 18.71 / 21.52 / 21.47 / 21.49 |
| Hyper-SD 4-step | Segmented consistency, adversarial, preference | 96% | 18.71 / 21.74 / 21.74 / 21.79 |
| Flash Diffusion | Distribution matching, adversarial | 93% | 17.85 / 21.30 / 20.79 / 21.50 |

4 steps without guidance, 500 COCO captions, each method with its own sampler.

| Result | Scale | Section |
|---|---|---|
| Agreement between independent methods on SD 1.5: 0.29 to 0.76, mean 0.47; controls 0.02 to 0.07 | Weights, 10 pairs | 2 |
| Agreement is not inherited from a shared starting point | Weights, all pairs | 2 |
| Agreement on SDXL 0.14 to 0.68 (8 methods), SD3 Medium 0.15 to 0.30, FLUX.1-dev 0.20 to 0.49; controls at or below 0.10 | Weights | 5 |
| Fully fine-tuned few-step models are not rank 1 (top direction 4% to 14% of the energy) but share the adapters' top direction at 0.12 to 0.38 | Weights, 3 models | 6 |
| On SDXL, one constant per layer stays within 0.15 PickScore points of the full adapter on four of four methods | 100 captions | 9 |
| The up blocks carry the acceleration: removing them loses 81% to 98% of the gain; convolutions matter most | 50 captions, 4 methods | 10 |
| An adapter built from the other four methods: within 0.1 points of the host's own or above on LCM-LoRA, TCD and Flash; 73% and 68% of the gain on PCM and Hyper-SD | 100 held-out captions | 11 |
| 1,112 trained strengths on the shared direction: PickScore 18.77 to 20.90; random direction 19.76; a free vector with 1.26 million numbers 20.51 | 200 training captions, 100 for scoring | 8 |
| The gradient of a plain few-step loss agrees with the shared direction at 0.44; strengths trained on it reach 20.75 | 200 captions | 12 |
| The gradient of LCM's consistency loss agrees with the plain loss's at 0.67 and with the shared direction at 0.40 | 400 photos | 13 |

## Method

Each item says what is done and what is open.

### A. Universality

1. **Scale on SD 1.5.** Done at 500 COCO captions and one seed, with PickScore, CLIP,
   CMMD, clean-FID and KID. Open: 5,000 captions and 3 seeds, the scale at which FID can
   be read against published numbers.
2. **Other base models.** Weights done on SDXL, SD3 Medium and FLUX.1-dev. Function done
   on SDXL for four methods. Open: function tests on SD3 and FLUX, and style-adapter
   controls on those two.
3. **Location.** Done by ablation on four methods. Open: by block depth on a transformer.
4. **Step count.** Open. Hyper-SD's 1, 2, 4 and 8-step adapters share a starting point
   and agree at 0.85 to 0.99. Test whether one direction with a step-dependent strength
   covers all four.
5. **Full fine-tunes.** Done for DMD2, LCM DreamShaper and 2-rectified flow.

### B. Mechanism

1. **Constant shift.** Done at 500 captions, with one constant per layer and with one per
   layer and step. Open: the same by layer type, and why Flash Diffusion is the exception.
2. **Where the direction comes from.** Done for two losses: the gradient at the base
   weights of a plain teacher-matching loss and of LCM's consistency loss. Open: the
   adversarial and distribution-matching losses, which is where Hyper-SD, Flash Diffusion
   and DMD2 differ; and why the adapters agree with each other (up to 0.76) more than with
   either gradient (0.40 and 0.44).
3. **Guidance.** Done: the comparison with the guidance direction, and a guided baseline
   at 4 steps. Open: an adapter trained only to remove guidance.
4. **Relation to the time path.** Open. A constant per layer and step is close to what
   MTEO trains (per-layer time embeddings). Test whether the two are the same change.

### C. Use

1. **Train strengths only.** Pilot done with 1,112 strengths (per layer and step) and
   with 278 scales (per layer), on the adapters' direction and on the gradient direction.
   Open: longer training, tuned learning rates, 3 seeds, one strength per block.
2. **Train the constant.** Pilot done: a free vector per layer and step reaches 20.51,
   below the 1,112 strengths. Open: plain gradient descent in place of Adam, which
   section 12 suggests is why the vector missed the direction; a full distillation loss.
3. **Transfer.** Open. Apply a direction or constant found on the base model to
   fine-tuned versions of it, as LCM-LoRA is used in practice.

### Controls used throughout

- **Random directions** of the same size, for every truncation, swap and cosine.
- **Style adapters** on the same base model, to separate "distillation" from "LoRA".
- **Same-method references**: Hyper-SD at different step counts, and two PCM versions.
- **A starting-point check** on every new pair of adapters before calling it independent.
- **Held-out captions** for anything estimated from data, with paired bootstrap intervals.
  One seed so far; 3 seeds at scale.
- **A guided baseline** (base model with guidance at the same step count).

## Order of work

The project is done by one person. What is left, by what it needs:

| Needs | Work |
|---|---|
| Reading | The eight readings in the [study guide](study-guide.md), then a revised literature review |
| The laptop | Step count (A4); constant by layer type (B1); the free vector with plain gradient descent (C2); transfer to a fine-tuned SD 1.5 model (C3) |
| A GPU | FID on 5,000 captions with 3 seeds (A1); function tests on SD3 and FLUX (A2); gradients of adversarial and distribution-matching losses (B2); longer training runs (C1, C2) |

## Timeline

Assumes the term ends in mid-January 2027.

| Dates | Work | Output | State |
|---|---|---|---|
| 16-22 Nov | One-page course proposal; results sections 5 to 13 | Proposal; results at 100 to 500 captions | Done |
| 23 Nov - 6 Dec | First reading pass; arrange GPU access; laptop tests A4 and B1 | Reading notes; revised literature review | Current |
| 7-20 Dec | With a GPU: A1 and A2. Without: C2 and C3 on the laptop | FID table at scale, or the laptop results | |
| 21-27 Dec | B2 under other losses; B3 | Mechanism results | |
| 28 Dec - 12 Jan | Write the term report | Report and presentation | |
| Jan - Apr | Remaining GPU work; paper draft | Submission | |

**Decision point, 6 December.** The claim is reviewed with the advisor on what exists
then. If no GPU is available, the 500-caption results stand in for the 5,000-caption
ones and the report says so. If a result fails at scale, the claim is narrowed to what
survives.

## Resources

- **Available:** an Apple M4 Pro laptop with 24 GB. Every result so far ran on it, one
  job at a time.
- **Needed:** a GPU with 24 GB or more, for the work listed under "A GPU" above.
- **Data:** MS-COCO 2014 validation captions and photos.
- **Software:** PyTorch, diffusers, clean-fid, and the code in `scripts/study/`. All
  models and adapters are public; Flash Diffusion's licence is non-commercial.

## Risks

| Risk | Likelihood | State and response |
|---|---|---|
| Results weaken on FID at 5,000 captions and several seeds | Medium | Held at 500 captions on PickScore, CMMD and FID. Decision point; report what survives |
| Samples are small: one seed, 100 to 500 captions | High | Intervals are reported; the GPU run fixes it |
| The constant shift is already known under another name | Medium | MTEO and ReSeed were read by an automated reader and do not report it for existing adapters. BitFit and DiffFit are in the reading plan |
| The gradient result is what LoRA theory predicts | Medium | LoRA-GA and LoRA-One say a LoRA follows its first gradient. The new part is that different distillation losses share that direction: shown for two losses, open for adversarial and distribution-matching ones |
| Agreement comes from shared training data, not from the task | Medium | Cannot be ruled out with public adapters. The gradient at the base weights uses no adapter and agrees with the shared direction at 0.44, which overlap in the adapters' training images cannot explain; the rest of the agreement could still come from data. Record each method's training data from its paper |
| The effect is mostly built-in guidance | Medium | Partly, at the first step. B3 |
| Results on SD 1.5 look dated | Medium | Weights on SDXL, SD3 and FLUX; function on SDXL. Function on a transformer model is open |
| "LoRA ranks can be cut" is seen as known | High | It is known, including rank-1 resizes on model hubs. The claims are agreement, mechanism and use |
| No GPU access | Unknown | The laptop results already make a complete term report |

## Expected outcomes

- **Minimum (term report):** the results in `docs/RESULTS.md`, the laptop tests listed
  above, and related work written from papers read in full.
- **Target (conference paper):** FID at scale with seeds, function tests on a transformer
  model, the gradient direction under other losses, and trained strengths that close the
  gap to the public adapters. Candidate venues are ICCV 2027 and NeurIPS 2027; neither
  has announced its deadline (dates checked on 24 November are in the
  [folder README](README.md)).

## Related work

Covered at length in the [literature review](../docs/LITERATURE_REVIEW.md). None of these
had been read in full by the author on 24 November; the reading plan is in the
[study guide](study-guide.md).

- **WaDi** ([2603.08258](https://arxiv.org/abs/2603.08258)) compares fully distilled
  checkpoints with their teachers; no adapters, no cross-method comparison.
- **MTEO** ([2603.22375](https://arxiv.org/abs/2603.22375)) trains per-layer time
  embeddings for few-step sampling on a frozen model, which a per-layer constant may
  explain.
- **ReSeed** ([2610.37110](https://arxiv.org/abs/2610.37110)) adds a constant bias per
  block to a distilled student, for diversity and not for acceleration.
- **LoRA-GA** ([2407.05000](https://arxiv.org/abs/2407.05000)) and **LoRA-One**
  ([2502.01235](https://arxiv.org/abs/2502.01235)): a LoRA follows the top directions of
  its first gradient. Language models, one task at a time.
- **The Universal Weight Subspace Hypothesis**
  ([2512.05117](https://arxiv.org/abs/2512.05117)) claims shared subspaces across style
  adapters. In our data, style adapters do not share output directions.
- **Intruder dimensions** ([2410.21228](https://arxiv.org/abs/2410.21228)): LoRA adds new
  dominant directions. Ours are new in this sense, and still agree across methods.
- **CASA** ([2605.01929](https://arxiv.org/abs/2605.01929)) studies why style adapters
  fail on step-distilled video models, from singular-value spectra.
- **BitFit** ([2106.10199](https://arxiv.org/abs/2106.10199)) and **DiffFit**
  ([2304.06648](https://arxiv.org/abs/2304.06648)) tune bias terms only, for adaptation
  to new data.

Searches on 15, 16, 19 and 21 November found no paper that compares acceleration adapters
across methods or reduces them to a per-layer constant. Rank-1 versions of such adapters
circulate on model hubs without measurement. That is a handful of searches, not proof.

## Changes from the third version

- The hypotheses are renamed H1 to H4, to match the one-page proposal.
- Every hypothesis and plan item carries its state after sections 5 to 13 of the results.
- The origin of the direction is answered differently from the plan. The third version
  proposed to predict the constant from the base model's activations on a 50-step and a
  4-step path. That was not tried; the loss gradient at the base weights was used.
- The project is individual. The timeline is rewritten from 24 November.

## Changes from the second version

- The prerequisite and the main hypothesis of the second version were tested and held at
  small scale, so the plan moves on to explaining and using the result.
- Two questions are added: mechanism and use.
- The adapter set grew from one to nine, with three style adapters as controls.

## Relation to the original course proposal

The original proposal asked whether an LLM could find and edit the denoiser weights behind
a few-step failure. That was tested first. The LLM agents rediscovered published sampler
corrections and a blind search matched them, so the advisor's doubt was justified. This
proposal keeps the weight-space question and drops the LLM. That experiment's write-up,
code and results were removed from the repository on 24 November 2026 and remain in its
git history.
