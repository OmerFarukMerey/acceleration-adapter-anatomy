# The plan, explained from the start

A plain-language companion to [proposal.md](proposal.md), written for someone new to
generative AI. It explains what we are going to do, why, and what each result would mean.
Numbers are from the first study in this repository; the full tables are in
[`docs/RESULTS.md`](../docs/RESULTS.md).

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
are LCM, TCD, PCM, Hyper-SD and Flash Diffusion. They use different training objectives:
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

> "The more the input looks like pattern `v`, the harder you push the output along pattern `u`. The overall strength is `s`."

Each such piece is a **direction**. A rank-64 change has 64 of them, sorted from strongest
to weakest. The tool that finds them is the **singular value decomposition** (SVD).

An analogy: a sound engineer's change to a recording might be described as 64 separate
knob movements. If one knob does almost all the work, the change is really "turn up the
bass", however many knobs the mixing desk has.

## 5. What we found: one direction per layer

We opened nine acceleration adapters from five research groups, split every layer's
change into its directions, and tested what happens when we keep only some of them.

| What we kept in each layer | Result, for every method |
|---|---|
| All directions | Sharp, on-prompt images (the normal adapter) |
| Only the strongest 1 | Still sharp and on-prompt |
| Everything except the strongest 1 | Blurry, as if there were no adapter at all |
| 1 direction picked at random | Blurry |

So in each of these adapters, the ability to sample in 4 steps sits in **one direction per
layer**. That direction holds 93% to 98% of the change's energy.

We also found that the **decoder** half of the network matters most. Stable Diffusion's
network first compresses the image (encoder) and then rebuilds it (decoder). In every
adapter, 80% to 94% of the change is in the decoder.

## 6. What we found: the methods partly agree

Take one layer. LCM's adapter changes it mainly along one direction. TCD's adapter does
too. Are the two directions the same?

We measure this with the **cosine**: 1 means identical, 0 means unrelated. In a space
with over a thousand dimensions, two unrelated directions give about 0.02.

| Which adapters we compared | Cosine |
|---|---|
| Two acceleration adapters from different research groups | 0.29 to 0.76 |
| An acceleration adapter and an art-style adapter | 0.02 to 0.07 |
| Two art-style adapters | 0.02 to 0.03 |
| Two random directions | about 0.02 |

So groups that trained with different recipes, independently, pushed the model in
similar directions. Adapters that do something else (an art style) did not. The picture
`results/study/s1_agreement.png` shows every pair.

We checked one thing that could have faked this: that one adapter was built on top of
another. It was not. Training leaves a fingerprint of its random starting point, and
adapters from different groups have unrelated fingerprints.

The agreement is partial, not perfect. Putting one method's direction straight into
another's place gives images that are too contrasty or too soft. Turn one "volume knob"
and they come right: the direction was good, only its strength was off.

## 7. What we found: the direction is almost a constant push

Recall what a direction does: the more the input looks like pattern `v`, the harder it
pushes the output along `u`. We measured how much "looks like `v`" actually varies while
the model is making images. It barely does. It is nearly the same number for every
prompt, every part of the image and every step.

So we tried the simplest version: forget the input, and add the *same* fixed push to
each layer's output every time. That still gives images that match the prompt: as well
as the real adapter in four of the five methods, and somewhat less well in the fifth. Look at the second column of `results/study/s3_mechanism_tcd.png`.

In plain terms, a large part of what distillation teaches the model is "shift each
layer's output by a fixed amount". That is a much simpler thing than an adapter with
millions of numbers.

We also tested a common suspicion, that these adapters are only **guidance** baked into
the model. They partly are, at the first step, and less so afterwards.

## 8. The question now

> Is few-step ability one shared, simple change to the model, whatever method found it?
> If so, what is that change, and can we get it without the expensive training?

It has three parts, and each is a block of work:

- **Is it universal?** We tested one base model with 16 prompts. Does it hold on thousands
  of prompts, on proper quality scores, and on bigger models?
- **What is it?** Why is a constant push enough? Where does that push come from? How
  much of it is guidance?
- **Can we use it?** If the direction is known in advance, training only has to find its
  strength: a few hundred numbers, not millions.

## 9. What we will do, step by step

### Part A. Check that it is universal (person A)

1. **Measure image quality properly.** So far we measured how close an image is to the
   full adapter's image, and whether it matches the prompt. A paper needs quality scores
   on thousands of prompts:
   - **FID**: how close the set of generated images is to real photos, statistically.
   - **CLIP score**: how well an image matches its prompt.
   - **A preference score**: a model trained on human votes about which image looks better.
2. **Repeat on bigger, newer models.** Stable Diffusion 1.5 is small and old, which is
   why it runs on a laptop. Reviewers will want SDXL and a newer transformer-based model
   such as FLUX. This needs a proper GPU.

### Part B. Explain it (person B)

1. **Take the constant push apart.** Which layers need it? Does it need to differ from
   step to step?
2. **Predict it.** Try to work out the push from the original model alone, without
   opening any adapter. If that works, no distillation is needed to find it.
3. **Separate out guidance.** Compare against the original model run with guidance.

### Part C. Use it (both)

1. **Fit only the strength.** Keep the shared direction fixed and find the best strength
   per layer. Compare against the real adapters.
2. **Train the constant push directly.** Needs a GPU.

## 10. Why this matters, and what could go wrong

Most research on fast sampling looks at the images. We look at the numbers inside the
model and ask what changed. That is what "weight space" means. It is the same theme the
project started with, without the claim that an LLM can read weights.

| Risk | What it would mean |
|---|---|
| The results fade on thousands of prompts or on quality scores | The first study was too small to trust. We keep whatever survives |
| It only holds on Stable Diffusion 1.5 | An interesting fact about one old model, not a general finding |
| Someone has already shown the constant-push result | We would confirm and extend it, not discover it. We must search carefully first |
| The groups agree because they trained on the same images | The shared direction would say something about the data, not about speed. Hard to rule out without training our own |
| No access to a GPU | The laptop part still makes a complete term project, but probably not a conference paper |

We decide on 6 December, after the quality scores are in, how much of the claim stands.

## 11. Glossary

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
| Recovery | Our measure of how much of the full adapter's effect a part reproduces: 1 is all, 0 is none |
| CLIP score | How well an image matches its prompt, judged by a neural network |
| Constant shift | The same fixed push added to a layer's output every time, whatever the input |
| Style adapter | A LoRA that teaches an art style; our example of a LoRA unrelated to speed |
