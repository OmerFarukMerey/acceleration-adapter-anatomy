# Research direction: what acceleration adapters have in common

The term project follows one idea. This folder holds the plan; the results are in
[`../docs/RESULTS.md`](../docs/RESULTS.md).

| File | What it is |
|---|---|
| [proposal.md](proposal.md) | The project proposal (third version): questions, hypotheses, plan, order of work, timeline |
| [plan-explained.md](plan-explained.md) | The same plan in plain language, for a newcomer to generative AI |
| [study-guide.md](study-guide.md) | Reading list and reading order |
| [../docs/LITERATURE_REVIEW.md](../docs/LITERATURE_REVIEW.md) | Background and related work |
| [pilots.md](pilots.md) | How the earlier pilot experiments were run |
| [01-anatomy-of-acceleration-adapters.md](01-anatomy-of-acceleration-adapters.md) | The idea as first written on 11 November, kept as a record |

## The idea in one paragraph

Few-step distillation is shipped as a low-rank weight change on a base model, such as
LCM-LoRA. In nine public adapters from five independent methods, one direction per layer
carries the effect, the methods partly agree on that direction, and the direction acts
almost like a constant added to each layer's output. The project asks whether few-step
ability is one shared, simple change to the model, what that change is, and whether it
can be obtained without full distillation.

## How this follows from the first experiment

The first experiment in this repo asked whether an LLM could edit denoiser weights to fix
few-step sampling. It could, but only by rediscovering known sampler corrections, and a
blind search did as well. The weight-space question that survived is a different one: not
"can an LLM read weights" but "what does a trained few-step delta look like". That needs no
LLM and no claim the course advisor doubted.

## Deadlines

From a literature search on 2026-11-10. Only SIGGRAPH is official; the rest are estimates
from earlier years.

| Venue | Full paper |
|---|---|
| SIGGRAPH 2027 | 28 Jan 2027 |
| ICML 2027 | late Jan to early Feb 2027 |
| ICCV 2027 | first week of Mar 2027 |
| NeurIPS 2027 | early to mid May 2027 |

## Caveats

- Papers dated 2025-2026 come from automated literature searches and have not been read
  in full by us. Read them before writing related work.
- The first study used 16 prompts, one seed and one base model, and measured prompt
  alignment and closeness to the full adapter's image, not image quality.
- Four other directions were considered and dropped on 2026-11-11. Their pilot scripts and
  outputs are still in `scripts/pilots/` and `results/pilots/`, described in pilots.md.
