# What do distillation methods agree on?

**Few-step sampling in diffusion models as a shared, nearly constant weight shift**

Ömer Faruk Merey, Osman Semih Tiryaki · Generative Models for Multimedia, METU · November 2026

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
| E1 | H1, H2 | SVD of every layer's change; cosine between the top directions of all adapter pairs. Controls: random directions, mismatched layers, style LoRAs. Repeat on SDXL and one transformer model if a GPU is available. |
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
- **Development:** a separate pool of 16 prompts with fixed seeds, kept apart from the
  evaluation set. E4 needs captions only; the base model supplies the training targets.

## 5. Preliminary results

Stable Diffusion 1.5, 4 steps, 16 prompts, one seed:

- **H1:** the top direction holds 93–98% of each adapter's energy; the rank-1 part alone
  recovers 0.62–0.93 of the adapter's effect.
- **H2:** top directions of independent methods have cosine 0.29–0.76 (mean 0.47), against
  0.02–0.03 for random directions and 0.02–0.07 for style LoRAs.
- **H3:** a fixed vector per layer and step recovers 0.55–0.86 of the effect, with CLIP
  score within 0.02 of the full adapter.

No image-quality metric has been run yet, and only one base model is covered.
Code and results: <https://github.com/OmerFarukMerey/acceleration-adapter-anatomy>
