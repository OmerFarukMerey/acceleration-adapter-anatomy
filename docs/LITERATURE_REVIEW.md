# Literature review

**What do distillation methods agree on?** Background and related work for the
[project proposal](../research_ideas/proposal.md). Written 15 November 2026; updated 16
November with what the [first study](RESULTS.md) found.

The project asks whether few-step "acceleration" adapters trained with different
distillation methods make the same change to a diffusion model, what that change is, and
whether it can be had without full distillation. This review covers
the work that question rests on, the work closest to it, and what is still open. A
plain-language introduction to the same ideas is in
[plan-explained.md](../research_ideas/plan-explained.md); a reading order is in the
[study guide](../research_ideas/study-guide.md).

## How far to trust this review

No paper here has been read in full by a person on this project yet. Each citation
carries a mark for how it is known:

| Mark | Meaning |
|---|---|
| **[A]** | Read in full or in large part by an automated reader on 15 November 2026 |
| **[R]** | Read in full or in main text by an automated reader on 19 November 2026 |
| **[S]** | Confirmed by a web search (abstract, summary or snippet) on 10-15 November 2026 |
| **[M]** | Written from memory. The arXiv number and the description are unverified |

Open each paper before citing it. Claims about what a paper does **not** do are the least
reliable kind and matter most for positioning, so check those first.

## 1. Why few steps is hard

A diffusion model generates an image by removing noise in many small steps (DDPM,
[2006.11239](https://arxiv.org/abs/2006.11239) [M]). DDIM
([2010.02502](https://arxiv.org/abs/2010.02502) [M]) showed the same trained model can be
sampled deterministically with fewer steps, by treating sampling as solving an ordinary
differential equation. Stable Diffusion runs this process in a compressed latent space
([2112.10752](https://arxiv.org/abs/2112.10752) [M]).

Two facts make few steps hard:

- **Each step is a straight-line approximation of a curved path.** With 4 steps the
  approximation error is large, and an unmodified model gives blurry, off-prompt images.
  This is the "no adapter" row in our pilots (CLIP 0.218 against 0.320 with the adapter).
- **Guidance doubles the cost.** Classifier-free guidance
  ([2207.12598](https://arxiv.org/abs/2207.12598) [M]) runs the network twice per step and
  extrapolates between the two outputs. Text-to-image models depend on it for prompt
  alignment.

The second fact matters for this project. Most acceleration adapters are trained to
include the effect of guidance, so they are run without it. An adapter's weight change
therefore mixes two things: skipping steps, and built-in guidance. Section 3 returns to
this.

## 2. Faster sampling without retraining

One line of work keeps the weights and changes the sampler.

- **Better solvers.** DPM-Solver ([2206.00927](https://arxiv.org/abs/2206.00927) [M]) and
  its successors use higher-order updates and reach good quality at 10 to 20 steps. Below
  that they degrade.
- **Corrections for accumulated error.** Epsilon Scaling
  ([2308.15321](https://arxiv.org/abs/2308.15321) [S]) scales the predicted noise down
  slightly; the Time-Shift Sampler ([2305.15583](https://arxiv.org/abs/2305.15583) [S])
  feeds the network a shifted timestep. Both address "exposure bias": at sampling time the
  network sees inputs that differ from its training inputs.
- **Network feature re-weighting.** FreeU ([2309.11497](https://arxiv.org/abs/2309.11497)
  [M]) rescales the UNet's backbone and skip features at inference. It is useful here as
  evidence that the decoder half of the UNet controls detail and denoising strength.

**Relation to this project.** Our [first experiment](EXPERIMENT_1.md) let LLM agents edit
the weights of a small model. They converged on edits that reproduce the second group
above, and the same correction applied in the sampler gave nearly identical images. That
result is why the project moved from sampler-like edits to distillation, which changes
what the network computes and reaches 1 to 4 steps.

## 3. Step distillation

Distillation trains a student network to do in a few steps what the teacher does in many.
The methods differ in what the student is asked to match.

| Family | Idea | Methods |
|---|---|---|
| Progressive | Each student step matches two teacher steps; repeat to halve again | Progressive distillation, [2202.00512](https://arxiv.org/abs/2202.00512) [M] |
| Consistency (trajectory) | Any point on a sampling path must map to the same final image | Consistency models, [2303.01469](https://arxiv.org/abs/2303.01469) [M]; LCM, [2310.04378](https://arxiv.org/abs/2310.04378) [M]; TCD, [2402.19159](https://arxiv.org/abs/2402.19159) [M]; PCM, [2405.18407](https://arxiv.org/abs/2405.18407) [M] |
| Distribution matching | The set of student images must match the set of teacher images; no per-path target | DMD, [2311.18828](https://arxiv.org/abs/2311.18828) [M]; DMD2, [2405.14867](https://arxiv.org/abs/2405.14867) [M] |
| Adversarial | A discriminator judges student outputs | ADD / SDXL-Turbo, [2311.17042](https://arxiv.org/abs/2311.17042) [M]; SDXL-Lightning, [2402.13929](https://arxiv.org/abs/2402.13929) [M] |
| Hybrid | Consistency on path segments, plus adversarial and human-preference losses | Hyper-SD, [2404.13686](https://arxiv.org/abs/2404.13686) [M] |
| Flow straightening | Make the sampling path piecewise straight, so few steps suffice | PeRFlow, [2405.07510](https://arxiv.org/abs/2405.07510) [S]; Flash Diffusion, [2406.02347](https://arxiv.org/abs/2406.02347) [M] |

Three points from this literature shape the project:

- **The objectives are different in kind.** Consistency methods supervise each sampling
  path. Distribution-matching and adversarial methods supervise only the output
  distribution. There is no prior reason to expect the same weight change from both. That
  is what makes agreement, if found, a finding.
- **The papers compare images, not weights.** Each reports FID, CLIP score or a
  preference score against earlier methods. A search found none that compares what the
  trained students changed [S].
- **Guidance is usually distilled in.** LCM trains the student to reproduce the guided
  teacher, and most later methods do the same. Guidance distillation was first studied
  separately (Meng et al., [2210.03142](https://arxiv.org/abs/2210.03142) [M]). Two recent
  papers argue that few-step evaluation is confounded by this: Guidance Matters
  ([2602.22570](https://arxiv.org/abs/2602.22570) [S]) and AGD
  ([2503.07274](https://arxiv.org/abs/2503.07274) [S]). Our pilot baseline has this
  confound, and the proposal adds a guided baseline to separate the two effects.

## 4. LoRA and the low-rank structure of fine-tuning

**LoRA** (Hu et al., [2106.09685](https://arxiv.org/abs/2106.09685) [M]) freezes a
pretrained weight matrix `W` and learns its change as a product of two thin matrices,
`ΔW = B·A`, of rank at most `r`. It was motivated by an earlier finding that fine-tuning
succeeds inside a very low-dimensional subspace of the weights (Aghajanyan et al.,
[2012.13255](https://arxiv.org/abs/2012.13255) [M]).

Section 7 of the LoRA paper is the direct ancestor of our analysis [M]. It takes the
singular value decomposition of trained adapters and reports that the top singular
directions are shared between adapters of different rank and between random seeds, while
the remaining directions are not. Our pilot finds the same shape, more extreme: one
direction per layer carries LCM-LoRA, and the other 63 do nothing measurable.

Later work adds three cautions:

- **Dominant LoRA directions may be artifacts of LoRA.** Shuttleworth et al.
  ([2410.21228](https://arxiv.org/abs/2410.21228) [S]) show that LoRA introduces
  high-ranking singular directions that are nearly orthogonal to those of the pretrained
  weights, and that full fine-tuning does not. They call these "intruder dimensions". Our
  rank-1 direction may be one. If so, "methods agree" could mean "LoRA training produces
  similar intruders", not "distillation has one solution". Comparing against a
  non-acceleration LoRA tests this.
- **The two sides of a LoRA behave differently.** A 2026 preprint
  ([2606.23740](https://arxiv.org/abs/2606.23740) [S]) reports that across random seeds
  the output-side directions agree almost perfectly (0.99) and the input-side directions
  do not (0.07). Our first study compares both sides. Across independent acceleration
  methods both agree (output 0.29 to 0.76, input 0.33 to 0.69). Across style adapters
  only the input side shows any agreement (0.14 to 0.43), so the output side is the one
  that separates acceleration from other tasks.
- **Truncating a LoRA is known.** Compress then Serve
  ([2407.00066](https://arxiv.org/abs/2407.00066) [S]) compresses adapters by per-adapter
  SVD and by a shared basis. Step-distillation LoRAs for the Wan video models are
  published at ranks from 4 to 256 [S]. The project cannot claim rank reduction as a
  contribution; it uses rank 1 as a tool.
- **A LoRA is expected to follow its first gradient.** LoRA-GA
  ([2407.05000](https://arxiv.org/abs/2407.05000), NeurIPS 2024) initialises the adapter
  from the singular vectors of the full gradient at the pretrained weights. LoRA-One
  ([2502.01235](https://arxiv.org/abs/2502.01235), ICML 2025) proves, for a linear model,
  that LoRA's factors align with the top singular subspace of the one-step full gradient.
  Both are known here from a web search on 21 November 2026 (abstract level). They are about
  language models and one task at a time. For this project they turn section 12 of the
  results into an explanation: if every method's loss has a similar top gradient direction
  at the base weights, their adapters should agree. We measured this for a plain loss
  (0.44 with the shared direction) and for LCM's consistency loss (0.40); the two gradients
  agree with each other at 0.67. The adapters agree with each other more than with either
  gradient (up to 0.76), so the first gradient is part of the explanation, not all of it.

## 5. Acceleration as an adapter

**LCM-LoRA** ([2311.05556](https://arxiv.org/abs/2311.05556) [M]) showed that consistency
distillation can be trained as a LoRA instead of a full model. The authors call the
result an "acceleration vector", and show it can be added to fine-tuned versions of the
same base model and combined with style LoRAs. The vocabulary comes from **task
arithmetic** (Ilharco et al., [2212.04089](https://arxiv.org/abs/2212.04089) [M]), where
weight differences are treated as vectors that can be added, negated and scaled.

Most later methods also release LoRA versions. For Stable Diffusion 1.5 a search
confirmed LCM-LoRA, TCD-LoRA and Hyper-SD LoRAs at 1, 2, 4 and 8 steps; a DMD2 LoRA was
found for SDXL only [S]. PeRFlow releases its result as delta weights that transfer
across pipelines [S].

This is what makes the project possible on a laptop. The few-step ability of each method
exists as a small public file for the same base model, so methods can be compared in
weight space without any training.

## 6. What is known about the weights of distilled models

This is the closest prior work.

**WaDi** (Wang et al., CVPR 2026, [2603.08258](https://arxiv.org/abs/2603.08258) [A]) is
the nearest paper. It compares fully distilled checkpoints with their teachers (DMD2
against SD 1.5, and a distilled PixArt-α) and decomposes the weight change into norm and
direction. Direction changes by about 2.2% and norm by about 0.1%, a ratio near 22. It
also reports that keeping 30% of the rank of the change recovers 93% of the effect, and
proposes a rotation-based adapter (LoRaD) for one-step distillation that works best at
rank 256.

| | WaDi | This project |
|---|---|---|
| Object studied | Full distilled checkpoints | Published LoRA adapters |
| Methods compared | One per base model | Five on one base model |
| Breakdown by block or layer type | No | Yes |
| Rank result | 30% of rank for 93% | 1 of 64 directions keeps prompt alignment, in five methods |
| Best rank for its own adapter | 256 | Not yet trained |

The rank results point in different directions, and the difference needs an explanation.
Candidates: full fine-tuning against LoRA training (the intruder-dimension effect), one
step against four steps, and a quality metric against our closeness metric.

**MTEO** ([2603.22375](https://arxiv.org/abs/2603.22375) [A]) gets few-step sampling by
training only per-step, per-layer time embeddings on a frozen backbone. It contains no
adapter analysis. It sits in tension with our pilot, where the time-embedding projections
hold 1% of LCM-LoRA's energy and recover nothing alone. Both can be true: the time path
may be a sufficient route that LoRA distillation does not take.

The first study suggests a closer link. In all five methods, the adapter's main direction
can be replaced by a fixed vector added to each layer's output, and a vector per layer and
step does better still. A learned time embedding per layer is one way to supply exactly
that: a step-dependent constant. Whether MTEO's embeddings and our constants are the same
change is an open question in the proposal.

**B-LoRA** (Frenkel et al., ECCV 2024, [2403.14572](https://arxiv.org/abs/2403.14572)
[S]) is the precedent for block-level analysis. It finds that two specific SDXL blocks
are enough to capture style and content separately. It studies personalisation, not
acceleration.

**DART** ([2609.20051](https://arxiv.org/abs/2609.20051) [S]) studies singular values and
subspace rotation when style LoRAs are reused on step-distilled video models. Known from
a snippet only; the overlap looks partial.

**Gandikota and Bau** ([2503.10637](https://arxiv.org/abs/2503.10637) [S]) is related
interpretability work on distilled diffusion models. Known from a search result only;
read it for what distillation changes in model behaviour.

A targeted search found no paper that ablates which layers matter inside a few-step
distillation LoRA [S]. That is a small number of searches, not proof.

### Changes that do not depend on the input

The first study found that the adapters act almost like a constant added to each layer.
Three lines of work are close to that and must be read before claiming it:

- **Bias-only tuning.** BitFit (Ben Zaken et al., [2106.10199](https://arxiv.org/abs/2106.10199)
  [M]) fine-tunes only bias terms in language models. DiffFit (Xie et al.,
  [2304.06648](https://arxiv.org/abs/2304.06648) [M]) extends it to diffusion
  transformers with added scale factors. Both are for adapting to new data, not for
  few-step sampling; a search found neither applied to step distillation [S].
- **Rank-one adapters.** 1LoRA (Quercia et al., WACV 2026 [S]) proposes a rank-one
  adapter and reports it competitive with LoRA. Known from a supplement snippet only.
- **Exposure-bias corrections** (section 2) shift or rescale the network's prediction by
  amounts that depend only on the step. Our constants live inside the network, per layer,
  but the idea is related.

Read in full on 19 November 2026, with what each leaves open:

- **MTEO** ([2603.22375](https://arxiv.org/abs/2603.22375) [R]) keeps the backbone frozen
  and learns a time embedding per step and per layer for few-step sampling (SD 1.5 FID
  14.93 and 12.92 at 3 and 4 steps, against 36.04 and 24.02 for DDIM). It is the closest
  work to our training pilot. It does not analyse existing adapters and gives no shared
  direction.
- **ReSeed** ([2610.37110](https://arxiv.org/abs/2610.37110), 27 October 2026 [R]) adds a
  constant bias per block to an already distilled student to restore diversity. The
  constant is an add-on for diversity, not the acceleration itself.
- **WaDi** ([2603.08258](https://arxiv.org/abs/2603.08258) [R], main text) compares DMD2
  students with their teachers and finds the change low-rank ("30% of the rank recovers
  93%"). The lowest rank tried is 64, and no two distillation methods are compared.
- **Community rank-1 resizes.** Rank-1 versions of LCM, Turbo and Lightning adapters have
  circulated on model hubs since December 2023, without measurement. So "rank 1 is enough"
  is known in practice; what is new is measuring it across methods and what the one
  direction is.
- **The intruder-dimensions paper** ([2410.21228](https://arxiv.org/abs/2410.21228) [R]),
  the Universal Weight Subspace Hypothesis ([2512.05117](https://arxiv.org/abs/2512.05117)
  [R]) and DART ([2609.20051](https://arxiv.org/abs/2609.20051) [R]) do not cover few-step
  adapters.

**CASA** (Wang et al., ICML 2026, [2605.01929](https://arxiv.org/abs/2605.01929) [S])
studies why style adapters fail on step-distilled video models. It reports that
distillation and LoRA both leave the singular values of the weights nearly unchanged and
act by rotating singular subspaces. That fits WaDi's finding and ours (new directions,
small in norm). It does not compare distillation methods with each other.

## 7. Do separately trained networks find the same solution?

The main question is an instance of an older one.

- **Shared LoRA subspaces.** EigenLoRAx (Kaushik et al., CVPR 2025,
  [2502.04700](https://arxiv.org/abs/2502.04700) [S]) runs a principal-component analysis
  across many LoRAs for one base model, finds a small shared subspace, and trains new
  tasks as coefficients in it. The Universal Weight Subspace Hypothesis
  ([2512.05117](https://arxiv.org/abs/2512.05117) [A]) extends this to a general claim
  that networks trained on different tasks converge to a shared low-rank subspace. Its
  diffusion evidence is about 300 SDXL style LoRAs, 16 components, evaluated with CLIP
  score, with no acceleration adapters and no random-subspace control. "Share"
  ([2602.06043](https://arxiv.org/abs/2602.06043) [S]) applies the idea to continual
  learning.
- **What our data says.** The three style adapters in the first study do not share
  output directions with each other (0.02 to 0.03, the random value), including two
  trained on the same painter. That is three adapters against the paper's hundreds, and
  top directions against its 16-component subspaces, so it does not contradict the paper;
  it does show that the strong agreement among acceleration adapters is not something
  every LoRA on this model has.
- **Consequence for this project.** "Adapters share a subspace" is already claimed in
  general. Our claim has to be narrower and better controlled: adapters that differ by
  *distillation objective*, compared against random directions and against a style LoRA,
  with a functional test. If acceleration adapters agree with each other no more than
  with a style LoRA, the shared structure belongs to the base model or to LoRA, not to
  distillation.
- **Representation similarity.** SVCCA ([1706.05806](https://arxiv.org/abs/1706.05806)
  [M]) and CKA ([1905.00414](https://arxiv.org/abs/1905.00414) [M]) compare what networks
  compute, and document how similarity scores mislead without baselines. The Platonic
  Representation Hypothesis ([2405.07987](https://arxiv.org/abs/2405.07987) [M]) argues
  that different models converge on similar representations.
- **Permutation symmetry.** Git Re-Basin ([2209.04836](https://arxiv.org/abs/2209.04836)
  [M]) shows that independently trained networks often differ only by a reordering of
  hidden units. This does not apply directly here, which is an advantage of the setting:
  all adapters are deltas on the *same* frozen base model, so their coordinates are
  already aligned and can be compared directly.

## 8. Tools for comparing weight changes

- **Singular value decomposition.** By the Eckart-Young theorem, keeping the top `k`
  singular directions is the best rank-`k` approximation of a matrix. This justifies
  truncation as the way to ask "how many directions matter".
- **Cosine between top directions.** Valid when each adapter is close to rank 1. Singular
  vectors have arbitrary sign, so the absolute value is used.
- **Principal angles between subspaces.** The general tool when more than one direction
  matters. The LoRA paper's subspace-similarity score is a normalised version [M]. This
  is the fallback if other adapters are not near rank 1.
- **Baselines.** In high dimensions two random unit vectors have a cosine near zero, with
  a spread of about one over the square root of the dimension. Every agreement number
  needs this reference, and a same-method reference (Hyper-SD at different step counts)
  for the upper end.
- **Functional tests.** Similar weights do not guarantee similar behaviour, and
  dissimilar weights do not rule it out. Swapping, projecting and averaging directions,
  then generating images, is the check.

## 9. Evaluation

| Metric | What it measures | Limit | Source |
|---|---|---|---|
| LPIPS | Perceptual distance between two images | Needs a reference image; measures closeness, not quality | [1801.03924](https://arxiv.org/abs/1801.03924) [M] |
| CLIP score | Agreement between image and prompt | Insensitive to detail and artifacts | [2103.00020](https://arxiv.org/abs/2103.00020) [M] |
| FID | Distance between sets of generated and real images | Depends on resizing; noisy below tens of thousands of images | [1706.08500](https://arxiv.org/abs/1706.08500) [M] |
| clean-FID | FID with the resizing problems fixed | Same sample-size limits | [2104.11222](https://arxiv.org/abs/2104.11222) [M] |
| CMMD | Alternative to FID using CLIP features | Newer, fewer published comparisons | [2401.09603](https://arxiv.org/abs/2401.09603) [M] |
| HPSv2, ImageReward, PickScore | Predicted human preference | Learned from specific user populations | [2306.09341](https://arxiv.org/abs/2306.09341), [2304.05977](https://arxiv.org/abs/2304.05977), [2305.01569](https://arxiv.org/abs/2305.01569) [M] |

Our pilots use LPIPS distance to the full adapter's image and CLIP score. That shows a
truncated adapter behaves like the full one. It does not show the images are good. The
proposal adds clean-FID and one preference metric on at least 5,000 prompts.

## 10. The gap

Putting the sections together, with what the first study adds:

| Question | Status in the literature | This project (see `docs/RESULTS.md`) |
|---|---|---|
| Can few-step ability be stored as a low-rank adapter? | Yes: LCM-LoRA and successors | |
| Can LoRAs be truncated? | Yes: known in research and in community practice | Down to rank 1, in five methods; community rank-1 resizes exist without measurement |
| How do distilled weights differ from the teacher's? | WaDi, for full checkpoints, one method per model | New directions, not the base model's dominant ones; full fine-tunes are not rank 1 but share the adapters' top direction (0.12 to 0.38) |
| Which blocks and layer types carry acceleration? | No paper found | Decoder holds 80% to 94% of energy in nine adapters |
| Do many LoRAs share a subspace? | Claimed for style LoRAs, without a random control | Style adapters: no shared output direction |
| **Do adapters from different distillation methods share directions?** | **No paper found** | **Yes, partly: 0.29 to 0.76 against 0.02 random; also on SDXL, SD3 Medium and FLUX.1-dev** |
| **Is the agreement specific to distillation, and functional?** | **No paper found** | **Specific: yes. Functional: an adapter built from the other four methods, with one scale chosen on other prompts, matches the host's own on three of five methods and keeps about 70% of the gain on two** |
| **What does the shared direction do?** | **No paper found; MTEO, ReSeed and bias tuning are adjacent** | **Close to a constant per layer; partly guidance; a strength per layer on it can be trained (pilot)** |
| **Where does the shared direction come from?** | **LoRA-GA and LoRA-One: a LoRA follows the top directions of its first gradient (language models, one task)** | **The gradients of a plain few-step loss and of LCM's consistency loss at the base weights agree with it at 0.44 and 0.40, and with each other at 0.67; strengths trained on the gradient direction work with no adapter** |

The project's contribution, if it survives larger tests, is the last four rows.

Results from this review that could undercut it, and the control for each:

1. Agreement that is generic to LoRAs on this base model (sections 4 and 7): the
   style-adapter control. Passed in the first study.
2. Agreement inherited from a shared starting point: the weak-direction check. Passed for
   all pairs of different methods.
3. A rank-1 structure that is an artifact of LoRA training (section 4): comparison with
   fully distilled checkpoints, as in WaDi. Done for three students: concentration in one
   direction is specific to the LoRAs, the direction itself is not.
4. An effect that is mostly built-in guidance (section 3): partly true at the first step;
   a guided baseline at scale is still needed.
5. A constant-shift result that is already known (section 6): MTEO and ReSeed are the
   nearest and were read in full; neither reports it for existing adapters.
6. Agreement that comes from shared training data: cannot be settled with public adapters.
7. A gradient result that LoRA theory already predicts (section 4): the new part would be
   that different distillation losses share the top gradient direction. Shown for a plain
   loss and for LCM's consistency loss (section 13 of the results); the adversarial and
   distribution-matching losses are not tested.

## What to read first

1. LoRA, section 7, then LCM-LoRA.
2. WaDi in full. It decides the positioning.
3. The intruder-dimensions paper.
4. The Universal Weight Subspace Hypothesis and EigenLoRAx.
5. MTEO, BitFit and DiffFit, for the constant-shift result.
6. TCD, PCM, Hyper-SD and Flash Diffusion, for how each adapter was trained.
