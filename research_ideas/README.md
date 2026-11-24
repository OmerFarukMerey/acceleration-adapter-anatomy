# Research direction: what acceleration adapters have in common

The term project follows one idea. This folder holds the plan; the results are in
[`../docs/RESULTS.md`](../docs/RESULTS.md). Last updated 24 November 2026.

| File | What it is |
|---|---|
| [proposal-one-page.pdf](proposal-one-page.pdf) | The one-page course proposal, due 22 November 2026. Source: [proposal-one-page.md](proposal-one-page.md) |
| [proposal.md](proposal.md) | The long proposal (fourth version): questions, hypotheses with their current state, plan, timeline, risks |
| [plan-explained.md](plan-explained.md) | The same plan and findings in plain language, for a newcomer to generative AI |
| [study-guide.md](study-guide.md) | What to read, in what order, and what to look for |
| [gpu-plan.md](gpu-plan.md) | What to run once a GPU is available: setup, order, and the code to write first |
| [../docs/LITERATURE_REVIEW.md](../docs/LITERATURE_REVIEW.md) | Background and related work |
| [pilots.md](pilots.md) | The LCM-LoRA pilot that started the project, and its controls |
| [01-anatomy-of-acceleration-adapters.md](01-anatomy-of-acceleration-adapters.md) | The idea as first written on 11 November, kept as a record |

## The idea in one paragraph

Few-step distillation is shipped as a low-rank weight change on a base model, such as
LCM-LoRA. In public adapters from independent methods, one direction per layer carries
the effect, the methods partly agree on that direction on four base models, and the
direction acts almost like a constant added to each layer's output. An adapter built only
from other methods works, a thousand trained strengths on the shared direction give
few-step sampling, and the direction can be read from a loss gradient on the base model
alone. These reduced forms also work on fine-tuned versions of the base model. The project
asks whether few-step ability is one shared, simple change to the
model, what that change is, and whether it can be obtained without full distillation.

## Deadlines

Checked by web search on 24 November 2026. Confirm each on the conference's own site
before planning around it.

| Venue | Full paper | Source |
|---|---|---|
| ICML 2027 (Rio de Janeiro) | 22 January 2027; abstracts 14 January | icml.cc, as reported by the search |
| SIGGRAPH 2027 (Anaheim, 8-12 August) | Not announced. In 2026 it was 22 January | An earlier note here gave 28 January 2027 as official; this search could not confirm it |
| ICCV 2027 (Hong Kong, 2-8 October) | Not announced. In 2025 it was 7 March | Estimate from the previous cycle |
| NeurIPS 2027 (Europe) | Not announced. In 2026 it was 6 May | Estimate from the previous cycle |

## Caveats

- No paper had been read in full by the author on 24 November. Several were read by an
  automated reader, the rest are known from abstracts and searches. The reading plan is
  in the study guide.
- Results are at small scale: 16 prompts for the first study, 50 to 500 COCO captions for
  the later ones, one seed. The start-token test of section 15 was designed after its
  first results and is not yet repeated on fresh captions. FID on 5,000 captions and function tests on SD3 and FLUX are
  not done.
- The project's first experiment, on LLM-driven weight editing, and the pilots of four
  directions dropped on 2026-11-11 were removed from the repository on 24 November 2026.
  They remain in its git history.
