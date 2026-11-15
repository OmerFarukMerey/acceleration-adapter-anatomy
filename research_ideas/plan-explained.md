# The plan, explained from the start

A plain-language companion to [proposal.md](proposal.md), written for someone new to
generative AI. It explains what we are going to do, why, and what each result would mean.
Numbers are from the pilots in this repository.

## 1. The problem: diffusion models are slow

A diffusion model makes an image by starting from pure noise and cleaning it up a little
at a time. Each clean-up pass is one **step**, and each step runs a large neural network.
Stable Diffusion normally uses 25 to 50 steps.

If you simply ask for 4 steps, each step has to remove far too much noise at once. The
result is a blurry image that ignores most of the prompt. You can see this in the left
columns of `results/pilots/c2_lora_controls.png`.

## 2. The fix: distillation

**Distillation** trains a copy of the model, the *student*, to produce in 4 steps what
the original, the *teacher*, produces in 50. The student is the same network with
slightly different numbers inside it.

Several research groups have their own recipe for this training. The ones we care about
are LCM, TCD, Hyper-SD, SDXL-Lightning and DMD2. They use different training objectives:
some ask the student to land where the teacher would land, some add a critic network
that judges realism, some match the overall distribution of images.

## 3. What a weight is, and what an adapter is

A neural network is a long chain of **layers**. Each layer is mostly a big table of
numbers called a **weight matrix**. Training means nudging those numbers.

After distillation, every layer's table has changed a little. Call the change for one
layer `ΔW` ("delta W"): the new table minus the old one.

Storing a full new model is wasteful, so the change is usually shipped separately as an
**adapter**. The common format is **LoRA**. It stores each `ΔW` in a compressed form that
can only express a limited number of independent patterns. That limit is the **rank**.
LCM-LoRA uses rank 64 in each of 278 layers and is a 135 MB file. You download it, add it
to Stable Diffusion, and the model can now sample in 4 steps.

## 4. Directions: the one piece of maths you need

A layer takes a list of numbers in and gives a list of numbers out. A change `ΔW` can
always be broken into simple pieces, each of which says:

> "When the input looks like pattern `v`, push the output along pattern `u`, with strength `s`."

Each such piece is a **direction**. A rank-64 change has 64 of them, sorted from strongest
to weakest. The tool that finds them is the **singular value decomposition** (SVD).

An analogy: a sound engineer's change to a recording might be described as 64 separate
knob movements. If one knob does almost all the work, the change is really "turn up the
bass", however many knobs the mixing desk has.

## 5. What we found so far

We opened LCM-LoRA, split every layer's change into its 64 directions, and tested what
happens when we keep only some of them.

| What we kept in each layer | Result |
|---|---|
| All 64 directions | Sharp, on-prompt images (the normal adapter) |
| Only the strongest 1 | Still sharp and on-prompt; 87% as close to the normal result as possible |
| Only the other 63 | Blurry, as if there were no adapter at all |
| 4 directions picked at random | Blurry |

So in this adapter, the ability to sample in 4 steps sits in **one direction per layer**.
The strongest direction holds 94% of the change's energy.

We also found that the **decoder** half of the network matters most. Stable Diffusion's
network first compresses the image (encoder) and then rebuilds it (decoder). Keeping the
adapter only in the decoder still works reasonably; keeping it only in the encoder does
nothing.

## 6. The question we want to answer

> Do different distillation methods find the same direction?

Take one layer. LCM-LoRA changes it mainly along one direction. TCD's adapter also changes
it. Is TCD's main direction the same one, or a different one?

There are two possible answers, and both are interesting:

- **Same direction.** Then the direction is a property of Stable Diffusion itself, a kind
  of built-in "few-step switch" that every method ends up finding. It could be found once
  and reused.
- **Different directions.** Then there are several distinct ways to make a model fast,
  and the methods really do differ under the hood, not only in image quality.

Nobody appears to have checked this. The closest paper we found studies full distilled
models, not adapters, and does not compare methods.

## 7. What we will do, step by step

### Step 1. Check that other adapters also have one main direction

Repeat the test from section 5 on TCD and Hyper-SD adapters for the same base model. If
they need many directions, the comparison gets harder (see section 9).

### Step 2. Compare the directions

For each layer, take the main output pattern `u` from two adapters and measure the angle
between them. The standard measure is the **cosine**: 1 means identical, 0 means
unrelated.

Two details matter here:

- **We need a yardstick.** Two random patterns in a space with over a thousand
  dimensions have a cosine near 0. So we compare against random patterns, and against a
  LoRA that does something unrelated to speed, such as an art style.
- **We compare the output side.** Earlier research suggests the input pattern `v` of a
  LoRA depends on the random numbers training started from, while the output pattern `u`
  is stable. So `u` is the meaningful one to compare.

### Step 3. Check that the agreement is real

Numbers in weight tables can look similar without behaving the same. So we test behaviour
directly:

- **Swap:** put TCD's direction into the place of LCM's and see whether 4-step sampling
  still works.
- **Average:** blend the directions of several adapters into one and test it.

### Step 4. Measure image quality properly

So far we measured how close an image is to the full adapter's image. A paper needs
quality scores on thousands of prompts:

- **FID**: how close the set of generated images is to real photos, statistically.
- **CLIP score**: how well an image matches its prompt.
- **A preference score**: a model trained on human votes about which image looks better.

### Step 5. Repeat on bigger, newer models

Stable Diffusion 1.5 is small and old, which is why it runs on a laptop. Reviewers will
want to see the result on SDXL and on a newer transformer-based model such as FLUX. This
needs a proper GPU.

## 8. Why this is a weight-space project

Most research on fast sampling looks at the images. We look at the numbers inside the
model and ask what changed. That is what "weight space" means. It is the same theme the
project started with, without the claim that an LLM can read weights.

## 9. What could go wrong

| Risk | What it would mean |
|---|---|
| Other adapters are not one direction per layer | LCM-LoRA is a special case. We would compare groups of directions, which is harder to interpret |
| Every LoRA on this model shares that direction, including style LoRAs | The direction is about LoRA training, not about speed. Our style-LoRA yardstick is there to catch this |
| The directions differ between methods | Still a result: methods find different solutions |
| No access to a GPU | The laptop part still makes a complete term project, but probably not a conference paper |

The first risk is tested first, by 1 December, because everything else depends on it.

## 10. Glossary

| Term | Meaning |
|---|---|
| Diffusion model | A model that makes images by removing noise step by step |
| Step | One pass of the network during image generation |
| Teacher / student | The original slow model / the copy trained to be fast |
| Distillation | Training the student to imitate the teacher in fewer steps |
| Weight matrix | A layer's table of learned numbers |
| `ΔW` | The change to a weight matrix after training |
| Adapter, LoRA | A small file holding `ΔW` in compressed form |
| Rank | How many independent directions a change contains |
| Direction | One "if input looks like `v`, push output along `u`" piece of a change |
| SVD | The method that splits a change into directions, strongest first |
| Energy | The squared strength of a direction; shares of energy add up to 100% |
| Cosine | A measure of the angle between two patterns: 1 same, 0 unrelated |
| UNet, encoder, decoder | Stable Diffusion's network, its compressing half and its rebuilding half |
| Guidance | A trick that runs the network twice per step to follow the prompt more closely |
| LPIPS | A measure of how different two images look to a neural network |
| Recovery | Our pilot's measure: how much of the full adapter's effect a part reproduces |
