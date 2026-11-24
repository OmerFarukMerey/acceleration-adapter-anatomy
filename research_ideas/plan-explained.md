# The plan, explained from the start

A plain-language companion to [proposal.md](proposal.md), written for someone new to
generative AI. It explains what the project asks, what has been found so far, what is
left, and what each result would mean. Updated 24 November 2026. The numbers are from the
studies in this repository; the full tables are in
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

We then checked this with a quality score on 500 photo captions. **PickScore** is a model
trained on people's votes about which image they prefer. Adding an adapter raises it by
about 3 points. Keeping only the strongest direction costs at most 0.05 of those points
in four of the five methods, and 0.2 in the fifth.

The **decoder** half of the network matters most. Stable Diffusion's network first
compresses the image (encoder) and then rebuilds it (decoder). In every adapter, 80% to
94% of the change is in the decoder. In the four methods where we removed the decoder's
part of the adapter, 81% to 98% of its benefit was gone.

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

**It holds on bigger models.** We repeated the comparison on three larger base models:
SDXL, SD3 Medium and FLUX. Independent methods agree there too (0.14 to 0.68), and the
unrelated comparisons stay at or below 0.10. See `results/study/s8_models.png`.

**An adapter built from the others works.** The agreement is partial, not perfect.
Putting one method's direction straight into another's place gives images that are too
contrasty or too soft. Turn one "volume knob" and they come right: the direction was
good, only its strength was off. So for each method we built a replacement adapter that
uses nothing from that method: the direction and strength come from the other four, and
one volume setting is chosen on separate prompts. On 100 new captions it is as good as
the method's own adapter for three of the five methods, and keeps about 70% of the
benefit for the other two. See `results/study/s13_transfer.jpg`.

**Different step counts are the same change.** One group (Hyper-SD) publishes separate
adapters for 1, 2, 4 and 8 steps. For 2, 4 and 8 steps they turn out to be one direction
with a different volume setting, and even the setting hardly matters. The 1-step adapter
is a different change. See `results/study/s16_steps.jpg`.

## 7. What we found: the direction is almost a constant push

Recall what a direction does: the more the input looks like pattern `v`, the harder it
pushes the output along `u`. We measured how much "looks like `v`" actually varies while
the model is making images. It barely does. It is nearly the same number for every
prompt, every part of the image and every step.

So we tried the simplest version: forget the input, and add the *same* fixed push to
each layer's output every time. On 500 captions that costs 0.01 to 0.15 PickScore points
in four of the five methods, out of the 3 points an adapter is worth. The fifth method,
Flash Diffusion, loses 0.7. On the larger SDXL model it works for all four methods we
tested. Look at the second column of `results/study/s3_mechanism_tcd.png`.

**Why the fifth method was different.** We split the layers by kind and tried the fixed
push in one kind at a time. In most kinds it costs nothing. The loss came from the few
layers that read the prompt and not the image. The prompt reaches the model as 77 slots:
a start marker, the words, then filler. The start marker is identical for every prompt,
and we measured that the model gives it about 80% of its attention. Our single fixed push
was the average over all 77 slots, mostly filler, so it was the wrong number for the slot
that counts. Flash Diffusion puts over half of its change in these layers; the other
methods put about 5% there. With two fixed pushes in those layers, one for the start
marker and one for the rest, Flash Diffusion's loss is more than halved, and for the
other four methods the fixed push is as good as the full adapter. It is still the same for
every prompt. See `results/study/s17_flash.jpg`.

In plain terms, a large part of what distillation teaches the model is "shift each
layer's output by a fixed amount". That is a much simpler thing than an adapter with
millions of numbers.

We also tested a common suspicion, that these adapters are only **guidance** baked into
the model. They partly are, at the first step, and less so afterwards.

## 8. What we found: the direction can be trained, and found without any adapter

**Training only the strength.** If the direction is known, training does not have to
find it. We froze the whole model, fixed the shared direction, and trained only how
strongly to push along it: 1,112 numbers, where LCM-LoRA has 67 million. In 4 steps
PickScore went from 18.8 to 21.1. The public adapters score 21.3 to 21.9. The same
training on a random direction reached 19.7.

**Does knowing the direction beat training freely?** No, and at first we thought it did.
We also trained a completely free push, 1.26 million numbers with no direction given. It
first scored 20.5, below the 1,112 strengths. That was our mistake: its training speed
(the learning rate) was set too high. With the speed chosen by a short search it reaches
21.0, level with the strengths. So the direction does not give better images. It gives the
same images with a thousand times fewer numbers to train.

**Finding the direction from the original model.** In that test the direction came from
the public adapters, so someone still had to do the expensive distillation first. Can the
original model tell us the direction on its own?

Training works by computing a **gradient**: for every weight, which way to nudge it so
that the error gets smaller. We wrote down a simple few-step training error and computed
its gradient on the original model, before any training. The strongest direction of that
gradient agrees with the direction the five methods share: cosine 0.44, against 0.03 for
a random direction. Training the 1,112 strengths on it gives 20.95, almost the same as
with the adapters' own direction, and no adapter was used anywhere.

We repeated this with the actual training error of one method (LCM). Its gradient points
the same way.

So part of the reason the methods agree is simple: at the starting point, their training
errors all pull the model the same way. It is not the whole reason. The adapters agree
with each other more than with either gradient.

**It carries over to customised models.** People use these adapters on customised
versions of the model, tuned for a look such as photo-realism. We put our reduced
versions on two popular ones (DreamShaper 8 and Realistic Vision 5.1) without re-fitting
anything: the single direction, the fixed push, the adapter built from the other methods
and the trained strengths. All of them work there about as well as on the original model.
See `results/study/s18_finetune_ds8.jpg`.

## 9. The question now

> Is few-step ability one shared, simple change to the model, whatever method found it?
> If so, what is that change, and can we get it without the expensive training?

It has three parts, and each is a block of work:

- **Is it universal?** It holds on four base models when we compare weights, and on two
  when we generate images. Does it hold on thousands of captions and on the newest models?
- **What is it?** A constant push per layer, pointing where the first training step
  would go. Why is that enough, and how much of it is guidance?
- **Can we use it?** A thousand trained numbers get most of the way. Can they get all
  the way?

## 10. What is done and what is left

### Part A. Check that it is universal

1. **Measure image quality properly.** Done on 500 captions with three kinds of score:
   - **FID** and **CMMD**: how close the set of generated images is to real photos.
   - **CLIP score**: how well an image matches its prompt.
   - **PickScore**: which image people would prefer.

   Left: 5,000 captions with several random seeds, which is what published papers report.
   This needs a proper GPU.
2. **Repeat on bigger, newer models.** Done for the weights on SDXL, SD3 and FLUX, and
   with generated images on SDXL. Left: generated images on SD3 and FLUX. Needs a GPU.
3. **Different step counts.** Done for the one method that publishes four of them. Left:
   whether the direction shared between methods also works at 2 and 8 steps.

### Part B. Explain it

1. **Take the constant push apart.** Done: one push per layer is as good as one per
   layer and step; which kinds of layer take it; and why one method was an exception
   (section 7). Left: repeat the two-push version on new captions and on SDXL.
2. **Find where it comes from.** Done for two training errors (section 8). Left: the
   training errors that use a critic network, which is where the methods differ most.
3. **Separate out guidance.** Partly done. Left: compare with an adapter trained only to
   replace guidance.

### Part C. Use it

1. **Fit only the strength.** A small test is done (section 8). Left: train longer and
   see whether the gap to the public adapters closes.
2. **Train the constant push directly.** A small test is done. With a well-chosen training
   speed it does as well as fitting the strength, with a thousand times more numbers.
   Left: one of the methods' real training errors.
3. **Carry it over.** Done on two customised versions of the model (section 8). Left: more
   of them, and customised versions of SDXL.

## 11. Why this matters, and what could go wrong

Most research on fast sampling looks at the images. We look at the numbers inside the
model and ask what changed. That is what "weight space" means. It is the same theme the
project started with, without the claim that an LLM can read weights.

| Risk | What it would mean |
|---|---|
| The results fade on thousands of captions | The tests so far were too small to trust. We keep whatever survives |
| It only works in images on Stable Diffusion 1.5 and SDXL | A finding about older models. The weights of newer models agree, but that is weaker evidence |
| Someone has already shown the constant-push result | We would confirm and extend it, not discover it. The closest papers are on the reading list |
| The gradient result is already expected | Theory says a LoRA follows its first gradient. What would be new is that different training recipes share that gradient |
| The groups agree because they trained on the same images | The shared direction would say something about the data, not about speed. Hard to rule out without training our own |
| No access to a GPU | The laptop results still make a complete term project, but probably not a conference paper |

We decide on 6 December how much of the claim stands.

## 12. Glossary

| Term | Meaning |
|---|---|
| Diffusion model | A model that makes images by removing noise step by step |
| Step | One pass of the network during image generation |
| Teacher / student | The original slow model / the copy trained to be fast |
| Distillation | Training the student to imitate the teacher in fewer steps |
| Weight matrix | A layer's table of learned numbers |
| `ΔW` | The change to a weight matrix after training |
| Adapter, LoRA | A small file holding `ΔW` in compressed form |
| Fully fine-tuned model | A distilled model shipped whole, not as an adapter |
| Rank | How many independent directions a change contains |
| Direction | One "if input looks like `v`, push output along `u`" piece of a change |
| SVD | The method that splits a change into directions, strongest first |
| Energy | The squared strength of a direction; shares of energy add up to 100% |
| Cosine | A measure of the angle between two patterns: 1 same, 0 unrelated |
| Gradient | For every weight, the nudge that reduces the training error fastest |
| UNet, encoder, decoder | Stable Diffusion's network, its compressing half and its rebuilding half |
| Guidance | A trick that runs the network twice per step to follow the prompt more closely |
| LPIPS | A measure of how different two images look to a neural network |
| Recovery | Our measure of how much of the full adapter's effect a part reproduces: 1 is all, 0 is none |
| CLIP score | How well an image matches its prompt, judged by a neural network |
| PickScore | A score trained on people's choices between images; higher is better |
| FID, CMMD | Distances between a set of generated images and real photos; lower is better |
| Held-out captions | Captions that played no part in choosing any setting |
| Constant shift | The same fixed push added to a layer's output every time, whatever the input |
| Style adapter | A LoRA that teaches an art style; our example of a LoRA unrelated to speed |
