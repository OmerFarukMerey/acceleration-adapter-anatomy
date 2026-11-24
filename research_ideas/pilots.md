# The LCM-LoRA pilot

The pilot that started the project, and its controls. Both take one adapter, LCM-LoRA for
Stable Diffusion 1.5, apart in weight space. The cross-method study that followed is in
[`docs/RESULTS.md`](../docs/RESULTS.md), with code in `scripts/study/`.

Run on an Apple M4 Pro (24 GB, MPS): pilot C on 10 November 2026, its controls by 15
November. 4 steps, no guidance, the 16 development prompts, one seed.

| Pilot | Asks | Script | Output |
|---|---|---|---|
| C adapter anatomy | Which parts of the adapter carry the few-step ability: one layer type alone, everything but that type, one block group alone, and the whole adapter cut to a lower rank | `scripts/pilots/c_lora_anatomy.py` | `results/pilots/c_lora_anatomy.json`, `.png` |
| C2 controls | Is the low-rank result about the top directions, or would any few directions do: random directions, the directions truncation throws away, ranks 1 and 2, a decoder-only version | `scripts/pilots/c2_lora_controls.py` | `results/pilots/c2_lora_controls.json`, `.png` |

Run either with `.venv/bin/python scripts/pilots/<name>.py`. Pilot C downloads a 135 MB
adapter on first use.

## How to read the numbers

- **Recovery:** 1 minus the LPIPS distance to the full adapter's same-seed image, divided
  by the distance of the base model with no adapter. 1 means identical to the full
  adapter, 0 means no better than no adapter. Brackets are 95% paired bootstrap intervals
  over prompts.
- **CLIP:** similarity between the image and its prompt.
- **Texture:** mean absolute difference from the full adapter's image in the 1/16 to 1/4
  cycles-per-pixel bands, in dB.

## What came of it

- The pilot's own reading was "effectively rank 4, mostly in the decoder, time path idle".
  It is kept as first written in
  [01-anatomy-of-acceleration-adapters.md](01-anatomy-of-acceleration-adapters.md).
- The controls and the first study sharpened rank 4 to rank 1 (results, section 1).
- The location result was repeated on four methods with a quality score (section 10).

## Limits

- Both pilots measure closeness to the full adapter's image, not image quality, on 16
  prompts and one adapter.
- Neither was repeated with other seeds.

The study scripts `s2_function.py`, `s3_mechanism.py` and `s6_strength.py` import the
scoring and image-sheet helpers from `c_lora_anatomy.py`.
