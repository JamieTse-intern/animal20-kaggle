# Findings

Reusable conclusions, each one backed by a leaderboard measurement rather than a hunch.
Where a conclusion was later overturned, both versions are kept and the correction is marked.

---

## 1. Match the test-time degradation before touching the architecture

The first model validated at 98.9% and scored **0.740**. The training images were clean; the
test images are 64x64 JPEG at quality 75 with 4:2:0 chroma subsampling.

Replicating that exact pipeline — squash-crop, LANCZOS resize to 64px, re-encode with the
**test set's own quantisation tables** — took the score to 0.859 and was worth more than every
architecture change that followed combined.

**Transferable form:** a validation set drawn from the training distribution cannot see a
domain shift that lives in the *encoding*. Look at the bytes of the test data, not only at the
labels.

---

## 2. The pretraining recipe is the lever that keeps paying

Measured at the same stage with the same data and the same recipe, only the pretrained weights
changing:

| pretraining | base-stage public score |
|---|---|
| CLIP contrastive (LAION-2B) | plateaued near 0.965 |
| FCMAE masked autoencoding (ImageNet-22k) | 0.96350 |
| DINOv3 self-supervised, on a ConvNeXt trunk | 0.96264 |
| DINOv3 self-supervised, on a **ViT** trunk | 0.97556 |
| DINOv2 self-supervised (LVD-142M), ViT-L | 0.97363 |
| SigLIP 2, language-supervised, ViT-L | **0.94268** |
| Web-SSL, the DINOv2 recipe on uncurated web data | **0.94236** |

Two things separate out cleanly:

- **The architecture change is confirmed**: DINOv3 on ConvNeXt 0.96264 to DINOv3 on ViT 0.97556
  is +1.29pt with the prior held fixed.
- **Curation matters as much as the objective.** Web-SSL is the DINOv2 *recipe* on MC-2B web
  images instead of the curated LVD-142M, and it loses 3.1pt. The curated data is part of what
  made DINOv2 work here, not only the self-supervised objective.

The mechanism is specific and the arithmetic closes to 0.01pt. On 772 held-out renditions,
DINOv2 scores 93.78% against FCMAE's 89.90%. The test set is roughly 21-27% renditions, so
3.11% x 2,336 images is about 73 images, or +0.62pt. The observed leaderboard gap was +0.61pt.
**DINOv2's entire advantage is the non-photographic stratum.**

---

## 3. Capacity pays, but only attached to a prior that already transfers

The project recorded "capacity is not the bottleneck, data is" after ConvNeXt-V2-Huge (658M)
scored 0.95869. **That conclusion was too broad and was overturned.**

Inside the ViT + DINO family, 303M to 840M gave **+0.394pt at the same stage with the same
data and the same recipe** — a leaderboard signal (k=183, win-rate 60.1%).

It did not pay in the FCMAE/ConvNeXt family. The bottleneck was never capacity in the abstract;
it was capacity attached to a prior that transfers to this domain.

**Bounded on the other side too.** At 1.14B the same techniques failed on this hardware: 9.81 GB
steady state with gradients released and no EMA, because at depth 40 the gradient-checkpoint
boundaries are themselves multiple GB. That is a hardware boundary, not a claim that the larger
model would have been worse.

---

## 4. The two-bar rule for ensemble members

A member helps a blend only if it clears **both** bars:

- **strength** — within roughly 0.2pt of the anchor. The cutoff is bracketed by measurement:
  a member 0.14pt weaker **helped**; one 0.42pt weaker **hurt**.
- **decorrelation** — enough disagreement to contribute a different view.

It is a two-dimensional condition, not "diversity beats strength". Strong-but-similar hurts
(0.99% disagreement, −0.085pt strength: hurt). Different-but-weak hurts (2.29% disagreement,
−0.99pt strength: hurt).

**A later correction to this rule.** Raw disagreement rate turned out to be the wrong variable:
a member at 1.50% disagreement produced a signal where one at 1.61% was a wash. The operative
quantity is **complementarity across error strata** — one model strong on renditions, the other
on photos — and raw disagreement was only a proxy that held while every member shared one prior.

**The strength clause has no escape through shape.** A focal-loss member had the most favourable
*shape* any member ever had here — mean confidence 0.6588 against an anchor at 0.89, far blunter
than the model that first demonstrated the rule — and every blend containing it lost by 0.25 to
0.50pt, because it was 0.95pt weaker than the anchor. **Strength and shape are separate
conditions and shape cannot substitute for strength.**

---

## 5. Blend construction: match sharpness, then average

In a plain probability average the sharper model dominates the `argmax` regardless of merit.
Three independent routes undo that bias and land within noise of each other:

- up-weighting the blunter member (optimum bracketed by a clean 7-point unimodal curve),
- geometric (logit) averaging,
- **solving a temperature per member so every member has the same mean top-1 confidence, then
  averaging with equal weight.**

The third is what both final submissions use, for a reason beyond accuracy: **it has no
parameter fitted on the leaderboard**, so it cannot be overfitting the public split.

---

## 6. Distillation moves answers, not ability

One round of Noisy-Student self-distillation on pseudo-labelled test images was worth +0.59 to
+1.22pt to every lineage. Three limits were measured, each the hard way:

- **Round 2 collapses.** Retraining a model on its own raw predictions produced the documented
  feedback-collapse signature. `pseudo_rounds = 1`.
- **The student cannot be inside the teacher.** Distilling from a blend pulls the student toward
  *every member of that blend*, which both homogenises it and risks learning its own errors back.
- **Capability does not transfer, only decisions do.** A ConvNeXt student of a DINOv2 teacher
  agreed with its teacher on 99% of *test* rows — it had been trained on the teacher's labels
  for those images — and still collapsed back to FCMAE-level accuracy on renditions it had never
  seen, 4.15pt below its teacher. **A capability living in the pretrained prior cannot be
  distilled into an architecture that lacks it.**

**And the gain is conditional, not a law.** The strongest base the project produced gained only
+0.064pt from the same stage — one ninth of the typical gain — because it had no headroom left.
Three advance warnings all said so and were under-weighted: the reserve fell 2.46pt, mean
confidence fell from 0.8580 to 0.8097, and the best epoch was 1 of 16.

**Yet that erosion is what made it the best blend partner.** It became the bluntest model in the
project, and the blend of the sharpest member with the bluntest — after temperature matching —
is the best result here. A model can be made worse standalone and better in an ensemble by the
same operation.

---

## 7. Inference-side findings

- **Restoring a ViT to its native patch grid helps.** A ViT-H+ with a native 256px position grid
  was being inferred at 192px (144 tokens instead of 256), with the position embedding
  interpolated. Moving its TTA scales to 192/224/256 raised held-out rendition accuracy from
  95.21% to 95.98% (6 fixed, 0 broken, p = 0.031) and lowered neither held-out stratum.
- **It is also nearly invisible.** Inside a two-model blend that change moves **7 of 11,681
  rows**, because the unchanged member outvotes it on the marginal ones. A real, locally
  significant, and unmeasurable effect.
- **Training at the native grid is a different intervention and it loses.** At 256px the model
  sees 256 tokens of a 64x64 source upsampled 4x — the extra tokens carry interpolated pixels,
  not information. Measured twice, on two architectures.
- **Wider TTA by rotation was an exact tie.** Six views were already sufficient.

---

## 8. Post-processing is exhausted, and three of four cost nothing to rule out

| idea | verdict |
|---|---|
| Power averaging on top of geometric | **provably a no-op** — a positive power is monotone so the argmax cannot change. Confirmed to five decimals |
| Rank averaging | −3.4pt. Rank-transforming per class destroys the within-image comparability argmax needs |
| Class-prior rebalancing | tested four times, −0.5 to −1.1pt every time |
| Graph label propagation | 3 to 4 rows broken per row fixed, monotone in the mixing weight. Rejected for 5 GPU-minutes of measurement on labelled held-out data, instead of a submission |

The label-propagation failure was predictable from the graph before running it: low-confidence
rows sit in a sparse, anchor-poor part of the manifold — 29.5% of their neighbours are
high-confidence anchors against 77% for confident rows — so the propagation source is weakest
exactly where help is needed.

---

## 9. Two training details that cost real points

**Flat learning rates beat layer-wise decay on a self-supervised trunk.** Overriding the
project's flat 2e-5 with a layer-decay ladder off a 1e-4 top cost **1.95pt of held-out rendition
accuracy** against an otherwise identical control. The higher rate was destroying pretrained
features that then had to be relearned; the loss fell *faster* at the lower rate. The intuition
that a raw SSL trunk needs a bigger step is wrong at this data scale.

**Label smoothing silently breaks confidence thresholds.** With smoothing 0.1 over 20 classes,
the maximum attainable confidence is `1 - 0.1 + 0.1/20 = 0.905`. FixMatch's canonical 0.95
threshold therefore masks 100% of samples and the method degenerates into plain supervised
training without erroring. Any threshold on a smoothed softmax has to be set against that cap.
