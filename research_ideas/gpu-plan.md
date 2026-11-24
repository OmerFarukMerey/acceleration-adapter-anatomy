# GPU plan

What to run once a GPU is available, in order, with what has to be set up and written
first. Written on 24 November 2026, when every laptop item of the
[proposal](proposal.md) was done ([results](../docs/RESULTS.md), sections 1 to 16).

The items are the proposal's: A1 and A2 (universality), B2 (mechanism), C1 and C2 (use).
A1 comes first because it is the test that can end the main claim.

## Before the first run

1. **Code.** Clone the repository and install with `pip install -e '.[dev,study]'`. Run
   `pytest` (11 tests, no model download). The scripts pick CUDA on their own when there is
   no Apple GPU; `WSP_DEVICE` overrides the choice and `WSP_RUNS_DIR` moves `runs/` to
   another disk.
2. **Gated models.** SD3 Medium and FLUX.1-dev need their licences accepted on Hugging
   Face and a login on the machine (`huggingface-cli login`). SD 1.5, SDXL and every
   adapter are open.
3. **COCO.** The scripts expect `runs/study/coco/annotations/captions_val2014.json` and
   the photos in `runs/study/coco/val2014/`. The laptop holds only the first 1,000 photos
   of the fixed order. For 5,000 captions take the whole set:
   `http://images.cocodataset.org/annotations/annotations_trainval2014.zip` and
   `http://images.cocodataset.org/zips/val2014.zip` (about 6 GB).
4. **Small files from the laptop.** `runs/` is not in git. Everything in it can be
   rebuilt, but copying these keeps the GPU runs on the same objects as the laptop
   results: `runs/study/constants_*.pt` (the fitted constants), `trained_*.pt` (the trained
   strengths and vectors) and `factors_gradient.pt` (the loss-gradient direction), about
   0.2 GB together. The adapters' factor files are rebuilt from the downloaded adapters on
   first use.
5. **Check against the laptop.** Run section 7 on the first 100 captions and compare with
   the laptop's scores. Initial noise is drawn on the CPU, so the images should be close
   to identical.

   ```bash
   python scripts/study/s7_quality.py 100 50
   python scripts/study/s7_score.py --no-fid
   ```

   `s7_score.py` writes `results/study/s7_quality.json`, the 500-caption table in the
   repository. Keep a copy, or change the output name first (see the code list below).
6. **Disk.** One 512 px PNG is about 0.42 MB. The run in step 1 below writes about 130,000
   images per seed, about 55 GB. Score each seed, keep the cached per-image features under
   `runs/study/quality_features/`, then delete its images.

## Decide before step 1

The proposal leaves the thresholds open ("What would end the main claim"): how far rank 1
and the constant may fall from the full adapter on FID and PickScore at 5,000 captions
before H1 and H3 count as failed. Fix them with the advisor and write them into the
proposal before the run, not after.

## Work, in order

### 1. SD 1.5 at 5,000 captions and 3 seeds (A1)

The arms of `s7_quality.py`, at ten times the captions, with FID that can be read against
published 5,000-image numbers.

Code to write first:

- **A seed option.** The seed is the COCO image id everywhere (`s7_quality.py`,
  `s12_ablation.py: sample`). Add an offset, and give each seed its own image folder,
  feature prefix and results file, so that nothing overwrites the 500-caption results.
- **The two-constant arm.** Section 15's `<host>_const2` lives in `s17_layer_types.py`.
  Add it to the arms of `s7_quality.py`. This also repeats that test on captions that
  played no part in designing it (all but the first 100), which section 15 lists as open.
- **Optional: batches.** The sampler makes one image per call. Batching is only worth it
  if the first 100 captions show the run is too slow.

Then: generate, score with FID (`s7_score.py` without `--no-fid`), and report mean and
spread over the 3 seeds. `<host>_const_step` can be left at one seed.

Output: section 7 of the results at scale; the state of H1 and H3 against the thresholds.

### 2. Function tests on SD3 Medium and FLUX.1-dev (A2)

So far these two models are compared in weights only (section 5).

- A new script on the pattern of `s11_sdxl.py`: no adapter, full adapter, rank 1, one
  constant per layer, 100 COCO captions first. The adapter files are listed in
  `s8_other_models.py` (SD3: Hyper-SD, PCM, Flash Diffusion; FLUX: Hyper-SD, Turbo-Alpha,
  TDD). The work is mapping adapter keys onto the transformer's layers and choosing each
  release's sampler settings.
- Use section 15's two constants wherever a layer reads the prompt. On a transformer the
  text tokens pass through every block, so check first which layers that means.
- Style-adapter controls on these two models, in weights (`s8_other_models.py`).
- FLUX.1-dev does not fit in 24 GB at full precision without offloading.

Output: H1 and H3 in function on a transformer model; section 9 extended.

### 3. Longer training (C1, C2)

`s10_train_shift.py` trains on 200 captions for 2 epochs with one seed, and the strengths
end 0.2 to 0.8 PickScore points below the public adapters.

- More training captions (`N_TRAIN`), more epochs (`S10_EPOCHS`), 3 seeds (the script
  fixes `torch.manual_seed(0)`).
- The same variants as section 8, each with its learning rate searched: `strength_lr`,
  `random_lr`, `gradient_lr`, `bias_adam`, `bias_row`.
- One strength per block, in place of one per layer.
- Score on more than 100 captions.

Question to answer: does the gap to the public adapters close with more training, and do
the 1,112 strengths stay level with the free vector?

### 4. Gradients of other losses (B2)

Sections 12 and 13 take the gradient at the base weights for a plain teacher-matching
loss and for LCM's consistency loss. Hyper-SD, Flash Diffusion and DMD2 differ from these
in their adversarial and distribution-matching terms.

- New code on the pattern of `s15_loss_gradient.py`. Both losses need a second network (a
  discriminator, or a model of the student's own outputs), and at the base weights that
  network is untrained, so the first decision is what "the gradient at the start" means
  for them. This is the least defined item; do it last and read the methods' papers first.

### 5. Small repeats

Cheap on a GPU, listed as open in the results:

- The two constants of section 15 on the two fine-tunes (`s18_finetunes.py`) and on SDXL
  (`s11_sdxl.py`).
- The direction that independent methods share, at 2 and 8 steps (`s13_transfer.py` with
  the step counts of `s16_steps.py`).
- The transfer test of section 11 on more than 100 captions.
- More fine-tunes in section 16, and fine-tunes of SDXL.

## After each step

Update the results and every document that quotes them in the same pass: `docs/RESULTS.md`,
`proposal.md`, `plan-explained.md`, the two READMEs and the literature review's table of
the gap. The decision point in the proposal is 6 December.
