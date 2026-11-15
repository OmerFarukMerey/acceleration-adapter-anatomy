# Study guide: anatomy of acceleration adapters

A deeper explanation of [the idea](01-anatomy-of-acceleration-adapters.md): the background
you need, what the pilot showed, what the paper would claim, and what to read. Written
2026-11-11.

**Check every citation before you use it.** The arXiv numbers for older papers were written
from memory. The 2026 papers came from an automated literature search and were not read by
the author of this guide. Open each link first.

## Background

Learn these first if any are shaky.

- **How sampling works.** A diffusion model starts from noise and denoises in steps. Fewer
  steps means bigger jumps and more error. Read DDPM
  ([2006.11239](https://arxiv.org/abs/2006.11239)), then DDIM
  ([2010.02502](https://arxiv.org/abs/2010.02502)), which is the sampler this repo uses.
- **Latent diffusion and guidance.** Stable Diffusion denoises in a compressed latent space
  ([2112.10752](https://arxiv.org/abs/2112.10752)), and classifier-free guidance runs the
  network twice per step ([2207.12598](https://arxiv.org/abs/2207.12598)). That is why
  "6 steps" costs 12 network evaluations here.
- **The metrics.** LPIPS ([1801.03924](https://arxiv.org/abs/1801.03924)), FID
  ([1706.08500](https://arxiv.org/abs/1706.08500)) and CLIP
  ([2103.00020](https://arxiv.org/abs/2103.00020)). Know what each one cannot see.

## The idea

### Where it comes from

There are two ways to make sampling faster. One changes the sampler and leaves the model
alone, which is what the agents in this repo ended up doing. The other is **distillation**:
train a student model to do in 4 steps what the teacher does in 50.

The key fact for this idea is how distilled models are shipped. Fine-tuning changes every
weight matrix `W` to `W + ΔW`. LoRA forces `ΔW` to be a product of two thin matrices,
`B·A`, with a small rank `r`. LCM-LoRA showed that the whole 4-step ability can be stored
this way, as a 135 MB file you add to Stable Diffusion 1.5. Its authors called it an
"acceleration vector".

So the few-step ability is a concrete object: 278 small matrix pairs. You can open it and
measure it without running the model. That makes it a weight-space project that does not
need an LLM.

### What the pilot did

1. **Measured the delta.** For each of the 278 modules, compute `ΔW` and its singular
   values. A singular value decomposition splits a matrix into independent directions
   sorted by strength. If a few directions hold most of the energy, the matrix is
   effectively lower rank than its nominal 64.
2. **Truncated it.** Keep only the top 4 directions per module and generate images. They
   were 96% as close to the full adapter's images as possible, and looked identical.
3. **Ablated it.** Keep one part of the adapter and zero the rest, then do the reverse.
   "Only X works" means X is sufficient; "everything except X fails" means X is necessary.

| Finding | Evidence | Why it matters |
|---|---|---|
| Effectively rank 4 or lower | The median module needs 2 directions for 90% of its energy | The adapter is about 16 times bigger than it needs to be |
| The decoder carries it | Up blocks hold 82% of the energy and alone recover 75% | Distillation mostly rewrites the half of the UNet that draws detail |
| The time path is idle | Time layers alone recover 0%; removing them costs 6% | Conflicts with MTEO, which trains only time embeddings |
| No single module type suffices | The best single type (convolutions) recovers 15% | The ability is spread across types |

Code: `scripts/pilots/c_lora_anatomy.py`. Output: `results/pilots/c_lora_anatomy.json`.

### What the paper would claim

"Few-step ability, as learned by distillation, lives in a very low-dimensional,
decoder-heavy subspace, and this is true across distillation methods."

The second half is untested, and the paper depends on it. If Hyper-SD, TCD and DMD2
adapters show the same structure, or share the same directions, you have a finding about
distillation itself. If only LCM-LoRA does, you have a footnote.

The open questions, in the order to answer them:

1. **Does it replicate?** Run the same dissection on other adapters and on SDXL.
2. **Do methods agree?** Measure the overlap between the top directions of two adapters
   for the same base model.
3. **Does quality survive?** The pilot measured closeness to the full adapter, not quality.
   You need FID and a preference metric at rank 4.
4. **Can you use it?** Train a rank-4, decoder-only adapter directly and compare its cost.

### What to read

| Topic | Paper | Why |
|---|---|---|
| Low-rank adaptation | LoRA, Hu et al., [2106.09685](https://arxiv.org/abs/2106.09685) | Read first. Its argument for why low rank works is your starting hypothesis |
| Fine-tuning moves in few directions | Aghajanyan et al., [2012.13255](https://arxiv.org/abs/2012.13255) | The general version of your claim |
| Weight deltas as vectors | Task arithmetic, Ilharco et al., [2212.04089](https://arxiv.org/abs/2212.04089) | The vocabulary for comparing adapters |
| Distillation | Progressive distillation, [2202.00512](https://arxiv.org/abs/2202.00512); consistency models, [2303.01469](https://arxiv.org/abs/2303.01469) | How a student learns to skip steps |
| The adapter in the pilot | LCM, [2310.04378](https://arxiv.org/abs/2310.04378); LCM-LoRA, [2311.05556](https://arxiv.org/abs/2311.05556) | What you dissected |
| Other adapters to dissect | DMD, [2311.18828](https://arxiv.org/abs/2311.18828); DMD2, [2405.14867](https://arxiv.org/abs/2405.14867); Hyper-SD, [2404.13686](https://arxiv.org/abs/2404.13686); TCD, [2402.19159](https://arxiv.org/abs/2402.19159); SDXL-Lightning, [2402.13929](https://arxiv.org/abs/2402.13929) | Step 1 of the plan |
| Linear algebra | Eckart–Young theorem; principal angles between subspaces | Why SVD truncation is optimal, and how to compare two adapters |
| Closest competitors | WaDi, [2603.08258](https://arxiv.org/abs/2603.08258); MTEO, [2603.22375](https://arxiv.org/abs/2603.22375) | Read before anything else. They decide how you position the paper. Not read by the author of this guide |
| Shared subspaces across adapters | EigenLoRAx, Kaushik et al., CVPR 2025, [2502.04700](https://arxiv.org/abs/2502.04700); The Universal Weight Subspace Hypothesis, Kaushik et al., [2512.05117](https://arxiv.org/abs/2512.05117) | A method for the "do methods agree?" question, and a competing claim. Known from search summaries only, not read by the author of this guide |
| Follow-up to the subspace papers | Shared LoRA Subspaces for almost Strict Continual Learning ("Share"), Kaushik et al., [2602.06043](https://arxiv.org/abs/2602.06043) | Learns one shared low-rank subspace meant to replace many task-specific LoRAs. Lower priority. Known from a search summary only |
| Secondary summary | [Review of the Universal Weight Subspace Hypothesis](https://www.themoonlight.io/en/review/the-universal-weight-subspace-hypothesis) | Says the paper's SDXL LoRA subspace kept image quality. Not a citable source: use it to decide what to check in the paper itself |

### How EigenLoRAx relates to this project

EigenLoRAx takes many trained LoRA adapters for one base model, runs a principal-component
analysis across them, and finds a small subspace they share. A new task then trains only
coefficients on those shared directions. Code: `github.com/toshi2k2/EigenLoRA`.

- **It gives you a method.** Its procedure for extracting shared directions can be reused
  for the "do methods agree?" question, next to principal angles.
- **It predicts your control.** If ordinary LoRAs already share a low-rank subspace, the
  style-LoRA control should show low rank too. Then "LCM-LoRA is rank 4" alone says little.
- **It suggests a use.** Extract a shared acceleration subspace from several adapters and
  train only coefficients in it, as an alternative to training a rank-4 adapter directly.
- **Its follow-up competes.** The Universal Weight Subspace Hypothesis claims that networks
  trained on different tasks converge to a shared low-rank subspace. "Adapters share a
  subspace" is therefore not a headline claim on its own.

What remains specific to this project: adapters that differ by distillation objective and
not by task, the block and module-type breakdown, and whether the acceleration subspace
differs from the generic one that style LoRAs share.

## Gaps found on review (15 November 2026)

A check of this guide against what a paper on this topic needs. Rows marked "search" were
confirmed to exist by a web search on this date, from abstracts or summaries. Rows marked
"memory" were written from memory and their arXiv numbers are unverified. None were read
in full.

### Reading that was missing

| Gap | Paper | Why you need it | Source |
|---|---|---|---|
| What LoRA spectra mean | LoRA vs Full Fine-tuning: An Illusion of Equivalence, Shuttleworth et al., [2410.21228](https://arxiv.org/abs/2410.21228) | Shows LoRA adds "intruder" singular directions that full fine-tuning does not. Your top-4 directions may be these. Reviewers will expect you to engage with it | search |
| Block-level LoRA analysis in diffusion | B-LoRA, Frenkel et al., ECCV 2024, [2403.14572](https://arxiv.org/abs/2403.14572) | Finds two SDXL blocks suffice for style and content. The closest precedent for "which blocks carry it" | search |
| Compressing adapters | Compress then Serve, [2407.00066](https://arxiv.org/abs/2407.00066) | Per-adapter SVD truncation and a shared-basis method. Truncation alone is prior art; cite it as such | search |
| Acceleration as a weight delta | PeRFlow, NeurIPS 2024, [2405.07510](https://arxiv.org/abs/2405.07510) | Ships few-step ability as delta weights that transfer across pipelines. A fifth adapter family to dissect | search |
| Spectral view of few-step LoRAs | DART, [2609.20051](https://arxiv.org/abs/2609.20051) | Studies singular values and subspace rotation when reusing LoRAs on few-step video models. Check for overlap | search (snippet) |
| More adapter families | Phased Consistency Models, [2405.18407](https://arxiv.org/abs/2405.18407); Flash Diffusion, [2406.02347](https://arxiv.org/abs/2406.02347); SDXL-Turbo / ADD, [2311.17042](https://arxiv.org/abs/2311.17042) | More methods make the cross-method claim stronger | memory |
| SDXL itself | SDXL, [2307.01952](https://arxiv.org/abs/2307.01952) | Needed before dissecting SDXL adapters | memory |
| UNet roles | FreeU, [2309.11497](https://arxiv.org/abs/2309.11497) | Backbone against skip features; context for "the decoder carries it" | memory |
| Comparing subspaces | SVCCA, [1706.05806](https://arxiv.org/abs/1706.05806); CKA, [1905.00414](https://arxiv.org/abs/1905.00414); section 7 of the LoRA paper | Standard similarity measures and their pitfalls. The LoRA paper defines the subspace-similarity score you would reuse | memory |
| FID done correctly | clean-FID, Parmar et al., [2104.11222](https://arxiv.org/abs/2104.11222); CMMD, Jayasumana et al., [2401.09603](https://arxiv.org/abs/2401.09603) | FID changes with resizing and is weak at 5,000 images. Use clean-FID and report CMMD too | memory |
| Preference metrics | HPSv2, [2306.09341](https://arxiv.org/abs/2306.09341); ImageReward, [2304.05977](https://arxiv.org/abs/2304.05977); PickScore, [2305.01569](https://arxiv.org/abs/2305.01569) | Few-step papers now report at least one | memory |
| Few-step evaluation pitfalls | Guidance Matters, [2602.22570](https://arxiv.org/abs/2602.22570) | Argues FID and guidance choices mislead for few-step models | search (snippet) |
| Metric tooling | [EvalScope text-to-image evaluation guide](https://evalscope.readthedocs.io/en/latest/user_guides/aigc/t2i.html) | A practical overview of PickScore, HPSv2 and ImageReward with typical score ranges. A how-to, not a citable paper | search |

### What a reading list cannot supply

- **Results.** The paper needs the replication on other adapters, the controls and the
  quality table. One pilot on one adapter is all that exists.
- **The time-path caveat.** LCM-LoRA's finding that time layers are idle may depend on how
  that adapter was trained. Check which modules each adapter targets before comparing.
- **A random-subspace baseline** for every overlap number, and at least 3 seeds.
- **Full readings of WaDi, MTEO and EigenLoRAx.** Positioning depends on details that
  abstracts do not give.
- **A GPU** for SDXL and for training a compact adapter.
- **Writing.** Read two recent analysis papers from the target venue for structure. The
  intruder-dimensions paper above is a good model.

A targeted search on this date found no paper that ablates which layers matter inside a
few-step distillation LoRA. That is one search, not proof the question is open.

## Where to start

A first week, in order. Each step has a concrete output.

1. **Read** LoRA (sections 4 and 7), LCM-LoRA, and the intruder-dimensions paper.
   Output: half a page on what a singular direction of a LoRA delta means.
2. **Rerun the pilot**: `.venv/bin/python scripts/pilots/c_lora_anatomy.py`. Read the
   script until you can explain every line of `factorize` and `install`.
   Output: the same table as in the proposal, from your own run.
3. **Read** WaDi, MTEO and EigenLoRAx in full. Output: one paragraph each on what they
   claim and what they leave open.
4. **List the adapters.** For Hyper-SD, TCD, DMD2 and PeRFlow on SD 1.5, find the weight
   file, its rank, and which modules it targets. Output: one table. If an adapter does
   not touch a module type, say so before comparing it with LCM-LoRA.
5. **Fix the thresholds** in the proposal's hypotheses with your advisor.
6. **Run the dissection on one more adapter.** This is the first real test of whether
   the finding is general.

## Suggested reading order

1. DDIM, then classifier-free guidance.
2. LoRA, then LCM and LCM-LoRA.
3. WaDi and MTEO, to see how close the competition is.
4. EigenLoRAx, then the Universal Weight Subspace Hypothesis.
5. Task arithmetic and the intrinsic-dimension paper.
6. The other distillation papers, as you reach each adapter.
7. Before running anything: the intruder-dimensions paper and B-LoRA.
8. Before the quality table: clean-FID, CMMD and one preference metric.
