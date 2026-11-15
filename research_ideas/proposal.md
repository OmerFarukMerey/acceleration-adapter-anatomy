# Project proposal: Do distillation methods find the same direction?

**A weight-space comparison of few-step acceleration adapters for diffusion models**

| | |
|---|---|
| Author | Ömer Faruk Merey |
| Course | Generative Models for Multimedia, METU |
| Date | 15 November 2026 (second version; the first was dated 11 November) |
| Status | Draft for discussion with the course advisor |

A plain-language walk-through of this plan is in [plan-explained.md](plan-explained.md).
The background and related work are covered at more length in the
[literature review](../docs/LITERATURE_REVIEW.md).

## Summary

Diffusion models can be distilled to sample in 4 steps, not 50, and the result is often
shipped as a small weight delta (a LoRA) added to the base model. Several distillation
methods exist, with different training objectives. I propose to test whether they end up
making the same change to the model.

A pilot makes this testable. In LCM-LoRA for Stable Diffusion 1.5, a single direction per
layer carries the few-step ability: keeping one of 64 directions preserves 87% of the
adapter's effect, and the other 63 together do nothing measurable. If other adapters
are also close to one direction per layer, comparing two methods reduces to comparing
one vector per layer.

## Motivation

Step distillation is the main route to fast diffusion sampling. LCM, TCD, Hyper-SD,
SDXL-Lightning and DMD2 each train a student to do in a few steps what the teacher does in
many. Their papers compare image quality. None compares what the trained students changed.

Whether they changed the same thing matters:

- **Understanding.** If methods with different objectives converge on one direction, that
  direction is a property of the base model, not of the method.
- **Cost.** A shared direction could be reused: found once, then transferred or fine-tuned
  with very few parameters.
- **Practice.** People already combine and shrink these adapters by trial and error. A
  measured answer would replace folklore.

## Background

- **Few-step sampling.** A diffusion model turns noise into an image in steps. With few
  steps each jump is large and a base model gives blurry, off-prompt images.
- **Distillation.** A student copy of the model is trained to match the teacher's
  many-step result in a few steps.
- **LoRA.** The student's weight change `ΔW` for a layer is stored as a product of two
  thin matrices, so its rank is at most `r` (64 for LCM-LoRA).
- **Singular value decomposition.** `ΔW` splits into `r` independent directions, each
  with an output side `u`, an input side `v` and a strength `s`. Keeping the strongest
  `k` is the best rank-`k` approximation.

## Research questions

**Main question (RQ3 in the first version).** Do acceleration adapters trained with
different distillation methods, on the same base model, change the model in the same
direction?

Supporting questions:

1. **Rank.** Is each adapter close to one direction per layer, as LCM-LoRA is?
2. **Location.** Which blocks and layer types carry the ability, and is that the same
   across adapters?
3. **Use.** Can a single shared direction per layer, taken from one adapter or averaged
   over several, give few-step sampling with a small fraction of the parameters?

## Hypotheses

Each is written so that it can fail. The thresholds are first guesses, to be fixed with
the advisor before the comparison is run.

| | Hypothesis | Supported if | Rejected if |
|---|---|---|---|
| P | Prerequisite: other adapters are also near rank 1 | Rank 1 keeps prompt alignment (CLIP within 0.01 of full) for at least 2 of 3 further adapters | Most adapters need many directions |
| H1 | **Output directions agree across methods** | The top output direction of two adapters has a cosine far above the random baseline in most layers that carry energy | Cosines match the random baseline |
| H2 | **The agreement is functional** | One adapter's top direction, scaled by the other's strength, still gives few-step sampling | Swapped directions fail |
| H3 | Location agrees across adapters | The decoder holds most of the energy in at least 3 of 4 adapters | No consistent pattern |
| H4 | A shared direction is usable | An average of top directions from several adapters keeps prompt alignment | The average fails where each adapter works |

**If P fails,** the main question remains but needs a subspace comparison (principal
angles on the top-`k` output subspaces) in place of a single cosine.

## Preliminary results

All on Stable Diffusion 1.5 with LCM-LoRA (278 layers, rank 64), 4 steps, no guidance, 16
prompts, one seed per prompt. "Recovery" is 1 minus the LPIPS distance to the full
adapter's same-seed image, divided by the distance of the base model with no adapter.

**Rank and controls** (15 November)

| Directions kept per layer | Share of adapter energy | Recovery | CLIP |
|---|---:|---:|---:|
| None (base model at 4 steps) | 0% | 0.00 | 0.218 |
| All 64 | 100% | 1.00 | 0.320 |
| Top 4 | 97% | 0.96 | 0.320 |
| Top 2 | 96% | 0.92 | 0.320 |
| **Top 1** | **94%** | **0.87** | **0.321** |
| Directions 5 to 8 | 1% | 0.00 | 0.217 |
| Directions 5 to 64 | 3% | 0.01 | 0.215 |
| 4 random directions (two draws) | 8% | 0.01 - 0.02 | 0.217 - 0.218 |
| Top 1, decoder only | 77% | 0.73 | 0.322 |

**Location** (10 November)

| Part kept | Recovery | | Layer type | Energy share | Kept alone | Removed |
|---|---:|---|---|---:|---:|---:|
| Decoder (up) blocks | 0.75 | | ResNet convolutions | 40% | 0.15 | 0.39 |
| Encoder (down) blocks | 0.01 | | Feed-forward | 38% | 0.04 | 0.59 |
| Middle block | 0.00 | | Attention projections | 7% | 0.04 | 0.76 |
| | | | Self-attention | 7% | 0.02 | 0.82 |
| | | | Cross-attention | 6% | 0.02 | 0.75 |
| | | | Time-embedding projections | 1% | 0.00 | 0.94 |

What this shows:

- **One direction per layer carries the adapter.** The remaining 63 directions are not
  merely weaker; alone they produce the same blur as no adapter.
- **It is not redundancy.** Four random directions do nothing.
- **A very small version works.** One direction in decoder layers only is 0.9% of the
  adapter's parameters and keeps full prompt alignment, with a different composition.

Limits: one adapter, 16 prompts, one seed, and a metric that measures closeness to the
full adapter, not image quality. The "no adapter" baseline uses the LCM sampler without
guidance, so the adapter's measured effect mixes step-skipping with built-in guidance.
Code: `scripts/pilots/c_lora_anatomy.py` and `scripts/pilots/c2_lora_controls.py`.

## Method

### Step 1: Check the prerequisite on more adapters

Run the rank and location analysis on TCD-LoRA and the Hyper-SD LoRAs for SD 1.5. Hyper-SD
ships 1, 2, 4 and 8-step versions, which also gives a same-method comparison.

Before comparing, record for each adapter its rank, scale, and which layers it touches.
Adapters that skip a layer type cannot be compared on it.

### Step 2: Compare directions (the main experiment)

For each layer that two adapters share:

- **Output-side agreement.** Absolute cosine between the two top output directions `u`.
  Singular vectors have arbitrary sign, so the absolute value is used.
- **Input-side agreement.** The same for the input directions `v`. Prior work suggests
  this side depends on the random initialization, so low values here are expected.
- **Baselines.** Random unit vectors of the same size; a style LoRA for the same base
  model; and Hyper-SD versions against each other as the same-method reference.

Report agreement per layer, weighted by energy, and by block and layer type.

### Step 3: Test function

Weight-space agreement can mislead, so each claim gets a functional test:

- **Swap.** Install adapter A's top direction with adapter B's strength, and the reverse.
- **Project.** Keep only the part of A that lies along B's top direction.
- **Average.** Install the mean of the top directions of several adapters.

### Step 4: Measure quality

Evaluate the full, rank-1, swapped and averaged adapters on at least 5,000 COCO captions
with clean-FID, CLIP score and one preference metric, with 3 seeds. Add a baseline with
guidance and a standard sampler, to separate step-skipping from built-in guidance.

### Step 5: Scale up (needs a GPU)

Repeat steps 1 to 3 on SDXL (LCM-LoRA, SDXL-Lightning, DMD2, Hyper-SD) and on one
transformer model with public acceleration LoRAs, such as FLUX. A transformer has no
encoder and decoder, so location is reported by block depth.

### Step 6 (stretch): Train the direction

Train a rank-1 adapter directly and compare it with the truncated one.

## Controls

- **Random directions** of the same count, for every truncation and every cosine.
- **A non-acceleration LoRA** on the same base model, to separate "distillation" from
  "LoRA in general".
- **Same method, different setting** (Hyper-SD step counts) as the upper reference.
- **Seeds:** 3 per prompt, with intervals.

## Timeline

Assumes the term ends in mid-January 2027.

| Weeks | Dates | Work | Output |
|---|---|---|---|
| 1 | 15-22 Nov | Read LoRA, LCM-LoRA, WaDi, the intruder-dimensions paper. Fix thresholds | Fixed hypotheses |
| 2 | 23 Nov - 1 Dec | Step 1 on TCD and Hyper-SD | Prerequisite answered |
| 3-4 | 2-15 Dec | Steps 2 and 3 on SD 1.5 | Main result |
| 5-6 | 16-29 Dec | Step 4; step 5 on SDXL if a GPU is available | Quality table |
| 7-8 | 30 Dec - 12 Jan | Write the term report | Report and presentation |
| After | Feb - Apr | Step 5 on a transformer model; step 6 | Submission draft |

**Decision point, 1 December.** If the other adapters are not near rank 1, the comparison
switches to subspaces and the paper plan is reviewed with the advisor.

## Resources

- **Available:** an Apple M4 Pro laptop with 24 GB. Steps 1 to 3 on SD 1.5 fit; one arm of
  16 images takes under a minute.
- **Needed:** a GPU with 24 GB or more for SDXL, a transformer model, and step 4 at scale.
- **Software:** PyTorch, diffusers, and the harness in this repository. All models and
  adapters are public.

## Risks

| Risk | Likelihood | Response |
|---|---|---|
| Other adapters are not near rank 1 | Unknown | Decision point; fall back to subspace comparison |
| Directions do not agree across methods | Unknown | A clean negative result is still a term report, and says methods find different solutions |
| Agreement is trivial, because every LoRA on this model shares that direction | Medium | The style-LoRA control tests exactly this |
| Rank 1 is an artifact of how LCM-LoRA was trained | Medium | Step 1, and the intruder-dimensions literature |
| "LoRA ranks can be cut" is seen as known | High | It is known. The claim rests on the cross-method comparison, not on truncation |
| Results on SD 1.5 and SDXL look dated | High | Step 5 on a transformer model |
| No GPU access | Unknown | Steps 1 to 3 on SD 1.5 still make a complete term report |

## Expected outcomes

- **Minimum (term report):** rank and location for three SD 1.5 adapters, the direction
  comparison with controls, and functional tests.
- **Target (workshop paper):** the above with a quality table and SDXL.
- **Stretch (conference paper):** a transformer model and a directly trained rank-1
  adapter. Candidate venues are ICCV 2027 and NeurIPS 2027; deadlines are not announced.

## Related work

Entries marked * were read in full or in large part by an automated reader on 15 November
2026; I have not yet read them myself. The rest are known from abstracts or summaries.

- **Weights of distilled models.** WaDi* (Wang et al., CVPR 2026,
  [2603.08258](https://arxiv.org/abs/2603.08258)) compares full distilled checkpoints with
  their teachers and finds direction changes about 22 times larger than norm changes. It
  does not study LoRA adapters, break results down by block or layer type, or compare
  methods. Its own one-step adapter works best at rank 256, which contrasts with the
  rank-1 result here.
- **Shared subspaces.** EigenLoRAx ([2502.04700](https://arxiv.org/abs/2502.04700)) and
  the Universal Weight Subspace Hypothesis* ([2512.05117](https://arxiv.org/abs/2512.05117))
  find shared directions across many LoRAs. The diffusion part covers SDXL style LoRAs
  only, with no acceleration adapters and no random-subspace control.
- **What LoRA directions mean.** Shuttleworth et al.
  ([2410.21228](https://arxiv.org/abs/2410.21228)) show LoRA adds dominant "intruder"
  directions that full fine-tuning does not.
- **Low-rank acceleration adapters in practice.** Step-distillation LoRAs for the Wan
  video models are published at ranks from 4 to 256.
- **Other routes to few steps.** MTEO* ([2603.22375](https://arxiv.org/abs/2603.22375))
  trains only per-layer time embeddings. Here the time layers carry 1% of the energy.
- **Block-level LoRA analysis.** B-LoRA ([2403.14572](https://arxiv.org/abs/2403.14572)).
- **Distillation methods.** LCM-LoRA ([2311.05556](https://arxiv.org/abs/2311.05556)); TCD
  ([2402.19159](https://arxiv.org/abs/2402.19159)); Hyper-SD
  ([2404.13686](https://arxiv.org/abs/2404.13686)); DMD2
  ([2405.14867](https://arxiv.org/abs/2405.14867)); SDXL-Lightning
  ([2402.13929](https://arxiv.org/abs/2402.13929)).

## Changes from the first version

- The main question is now the cross-method comparison; rank and location support it.
- The pilot result is stronger: rank 1, with random-direction and tail controls.
- The adapter list is corrected. A search found a DMD2 LoRA for SDXL only, so DMD2 moves
  to the SDXL step.
- The comparison uses output-side directions and functional tests, because input-side
  directions of LoRAs are reported to depend on initialization.

## Relation to the original course proposal

The original proposal asked whether an LLM could find and edit the denoiser weights behind
a few-step failure. I tested that first; the write-up is in
[`docs/EXPERIMENT_1.md`](../docs/EXPERIMENT_1.md). The LLM agents rediscovered published
sampler corrections and a blind search matched them, so the advisor's doubt was justified.
This proposal keeps the weight-space question and drops the LLM.
