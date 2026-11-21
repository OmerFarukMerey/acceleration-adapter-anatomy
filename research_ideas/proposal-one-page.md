# What do distillation methods agree on?

**Few-step sampling in diffusion models as a shared, nearly constant weight shift**

Ömer Faruk Merey · Generative Models for Multimedia, METU · November 2026

## 1. Scientific question

Diffusion models can be distilled to sample in 4 steps instead of 50, and the result is
often released as a small LoRA adapter on the base model. Many methods do this with
different training objectives (LCM, TCD, PCM, Hyper-SD, Flash Diffusion). Their papers
compare image quality; we found none that compares what they changed in the weights.

**Is few-step ability one shared, simple change to the base model, whichever method found it?**
Three sub-questions: do independent methods find the same weight direction (universality),
what does that change do (mechanism), and can it be obtained without full distillation (use)?

## 2. Hypotheses

- **H1.** Each adapter's weight change is close to rank 1 in every layer.
- **H2.** Adapters from independent methods agree on that direction far more than random
  directions or style LoRAs do.
- **H3.** The change acts mostly as one fixed vector added to each layer's output,
  independent of the prompt.
- **H4.** With the direction known, few-step sampling can be reached by training only a
  strength or a constant per layer on a frozen base model.

Closest work: WaDi ([arXiv:2603.08258](https://arxiv.org/abs/2603.08258)) compares one
distilled model with its own teacher; MTEO ([arXiv:2603.22375](https://arxiv.org/abs/2603.22375))
trains per-step embeddings from scratch on a frozen model. Neither compares methods with each other.

## 3. Experiment plan

| | Tests | Experiment |
|---|---|---|
| E1 | H1, H2 | SVD of every layer's change; cosine between the top directions of all adapter pairs. Controls: random directions, mismatched layers, style LoRAs. Also on SDXL and a transformer model. |
| E2 | H1, H2 | 4-step sampling with only the rank-1 part, only the remainder, a direction taken from another method, and the direction the other methods agree on. Metrics: FID, CLIP score, 3 seeds. |
| E3 | H3 | Replace each layer's change with a fixed vector estimated on separate prompts; test on held-out prompts. Compare the adapter's effect with classifier-free guidance. |
| E4 | H4 | Train only per-layer strengths or constants on the frozen base; compare with the full adapter and with MTEO at equal step counts. |

H2 fails if agreement drops to the control level at scale; H3 fails if FID with the fixed
vector is far from the full adapter's.

## 4. Dataset

- **Objects of study:** nine public few-step LoRAs for Stable Diffusion 1.5 from five
  methods, plus three style LoRAs as controls, at pinned revisions. They were trained
  independently by different groups on the same base model, so agreement between them
  cannot come from shared training.
- **Evaluation:** 5,000 MS-COCO 2014 validation captions, with their images as the FID
  reference. This is the benchmark used by the distillation papers we compare with, so
  our numbers can be read against theirs.
- **Development:** 16 separate prompts with fixed seeds. E4 needs captions only; the base
  model supplies the targets.

## 5. Preliminary results

Stable Diffusion 1.5, 4 steps, 500 COCO captions (PickScore, FID):

- **H1:** the top direction holds 93–98% of each adapter's energy; rank 1 alone costs at
  most 0.05 PickScore on four of five methods (0.20 on the fifth), FID unchanged.
- **H2:** top directions of independent methods have cosine 0.29–0.76 (mean 0.47), against
  0.02–0.07 for random directions and style LoRAs. Weights only: SDXL 0.14–0.68
  (8 methods), SD3 0.15–0.30, FLUX 0.20–0.49; controls ≤ 0.10. In function
  (100 held-out captions), an adapter built only from the other four methods is within 0.1
  PickScore of the host's own on three of five methods and keeps about 70% of its gain on two.
- **H3:** one fixed vector per layer costs 0.01–0.15 PickScore on four of five methods
  (removing the adapter: 2.8–3.4), FID not worse; the fifth loses 0.7. On SDXL
  (100 captions) it stays within 0.15 on four of four methods.
- **H4 (pilot, 100 captions):** 1,112 trained strengths on the shared direction raise
  PickScore from 18.77 to 20.90 (random direction 19.76; public adapters 21.3–21.9). The
  direction can be read from the base model alone: the gradient of a plain distillation
  loss agrees with it at cosine 0.44 (random 0.03) and gives 20.75, with no adapter used.

FID on 5,000 captions and function tests on transformer models are not done yet.\
Code and results: <https://github.com/OmerFarukMerey/acceleration-adapter-anatomy>
