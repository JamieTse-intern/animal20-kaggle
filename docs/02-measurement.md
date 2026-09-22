# The measurement scorecard

Five local evaluation instruments were built over this project. **All five failed.** This is the
record of how, because the pattern is more transferable than any of the models.

---

## The scorecard

| instrument | what it was | how it failed |
|---|---|---|
| **Validation accuracy** | 1,419 competition images, stratified, frozen on disk | Saturated at ~99% by day four. At one point it recorded **0.9922 for the model that scored lowest on the leaderboard of its whole family** (0.96926) — not merely uninformative, anti-predictive by 0.42pt |
| **Sketch-val** | accuracy on an XDoG edge transform of the validation images | **Circular.** The intervention it was used to judge trained on the same operator. Reinstating that augmentation moved sketch-val +15pt and a genuinely independent rendition check +0.15pt, which is noise |
| **The rendition reserve** | 772 real external renditions, reserved on disk with a fingerprint before any training draw could see them | Valid in one regime and one only (below) |
| **A distribution-matched estimator** | photos and renditions scored separately on all held-out data, then weighted 80/20 to match the test mix. SE 0.20pt, better than the leaderboard's 0.25pt | Ranked the **worst** model first — and won on **both** strata, so no re-weighting could fix it |
| **A photo holdout** | 3,124 photos from a different public dataset, pushed through the same degradation | Could not rank these models at all. Model spread 0.42pt, every paired p at least 0.31. Its one "significant" result (p = 0.004) was simply absent on the leaderboard |

---

## The rendition reserve, in detail

The most useful of the five, and the one whose limits were mapped most precisely.

**Where it works — same-stage direction, 7 for 7.** Comparing two models at the same training
stage, with identical test-set exposure, its *direction* has never been wrong on a gap that was
statistically significant.

**Where it inverts — across the distillation boundary.** It rated one model 4.4pt worse
(p < 0.001); the leaderboard rated it 0.85pt better. This is mechanical, not bad luck: the
reserve is 100% renditions, the test set is roughly a quarter renditions, and the distillation
stage trains on pseudo-labelled *test* images, which are mostly photographs. **A falling reserve
accuracy during distillation is the signature of that stage working.**

**Where it is now permanently closed — ranking blends. Four attempts, four failures:**

| attempt | reserve said | leaderboard said |
|---|---|---|
| 1 | one blend ahead by reserve | reverse |
| 2 | 95.73% beats 95.34% | 0.98217 loses to 0.98313 |
| 3 | **95.85%, the highest ever measured** | 0.98164, well below a blend at 93.91% |
| 4 | 96.11% > 95.08% > 94.56% | 0.97833 < 0.97929 < 0.98089 — **Spearman −0.95, a near-perfect inversion** |

**Magnitudes never convert.** Across the pairs measured, the reserve-to-leaderboard ratio spans
roughly 1:1 to 6:1. Converting a reserve gap into a predicted leaderboard number was tried once
and was wrong by a factor of four. Use the direction; never the magnitude.

---

## The case that closes the question

Everything above can be explained away as "the proxy covers the wrong stratum". This one cannot.

A model trained with focal loss, **identical to its control in every respect except the loss
function**:

| | focal | control | direction |
|---|---|---|---|
| validation photos (1,419 held out) | **0.9958** | 0.9951 | focal better — the best base model in the project |
| rendition reserve (772 held out) | **94.04%** | 92.88% | focal better by 1.17pt |
| **Kaggle public (9,345 images)** | **0.97385** | **0.97556** | **focal worse by 0.171pt** |

**Better on every held-out set this project owns. Worse on the only one that counts.**

Every earlier failure was an instrument wrong about one stratum, or inverting across one
boundary. This one is wrong on both strata simultaneously and in the same direction. It closes
the question of whether a better local proxy could have been built: **not from held-out data
drawn from these sources.**

---

## What replaced local evaluation

### The paired win-rate test

Comparing two submissions on their full scores wastes the information that they agree on almost
everything. Instead, compare them **only on the rows where they disagree**:

```
k        = rows where the two submissions differ         (all 11,681 rows)
rows     = the public portion of those, ~0.8k            (the public split is 9,345 of 11,681)
net      = (score_A - score_B) * 9345                    (net images won, on the public split)
win-rate = (rows + net) / (2 * rows)

SIGNAL requires  k >= 60  AND  win-rate >= 60%   (or <= 40% for significantly worse)
```

Equivalently, a signal needs `delta >= 0.16 * k / 9345`.

**A change that moves many rows must win by proportionally more, because it had proportionally
more chances.** At k = 60 a signal needs +0.10pt; at k = 172 it needs +0.29pt. Two results that
cleared their delta failed their win-rate for exactly this reason.

**Most differences are ties, and saying so is the finding.** Of 70 submissions, a handful ever
produced a signal. The rest measured noise, and the discipline of labelling them ties is what
kept 0.36pt of spread across 24 submissions from being read as progress.

**A tie is never used as ground truth.** One leaderboard tie was once used to calibrate a metric
and to explain a mechanism; three conclusions were built on it and all three had to be withdrawn.

### Pre-registration

Before any run started, the log recorded: the model, the exact recipe, the gate that would close
the line, the screen that would decide whether to spend a submission, and **an explicit
prediction with a probability**. Afterwards the prediction was scored.

The predictions that were **wrong** are the most useful entries in the log:

| prediction | outcome | what it taught |
|---|---|---|
| "EVA-02's base clears its gate" | missed by 0.085pt | The line was continued as a labelled post-hoc override and produced the best result in the project. Gates are decisions, not laws |
| "ViT-H+ pseudo1 has the widest upside — every lineage gained +0.59 to +1.22pt" | +0.064pt, one ninth of the typical gain | Lineage gain is **conditional on the base having headroom**, not a constant |
| "Native-grid *training* is the most promising of three levers" (40%) | the worst of the three | An inference-side result does not transfer to training when the extra capacity is filled with interpolated pixels |
| "Focal loss will not help" (15%) | correct on the leaderboard | but for the wrong reason — it was reasoning about *which images* focal would help, when what focal actually changed was *how confident the model is everywhere* |
| "Native-grid TTA will be a wash on both strata" (25%) | won both strata, significantly on one | The concern (256px outside the fine-tuned range) was real but smaller than the cost of interpolating the position embedding |

### Reserve the instrument before optimising against it

Three separate bugs came from letting training-side logic decide what counted as held out:

1. A random-seed change silently moved the train/validation split, so an external diagnostic
   scored a checkpoint on images it had trained on — 86% overlap — and **flipped a conclusion**.
2. A class-balanced sampling change consumed the *entire* external pool for three classes, so
   the only non-circular metric went to zero held-out images on exactly the classes the change
   targeted.
3. An augmentation trained on the same operator its metric measured with.

The fix is structural, not procedural: the held-out set is written to disk with a fingerprint of
the source pool **before any training draw can see it**, and the training code raises if that
fingerprint stops matching. Never derive an evaluation split by mirroring training-selection
logic in a second script — the two drift, and the drift is silent.

---

## The honest summary

For anything touching pseudo-labelling — which here is the entire pipeline — **"measure locally
first, then commit GPU" does not work.** Such experiments have to be gated on the leaderboard at
0.25pt standard error, which means only effects of roughly 0.5pt or larger are detectable at all.

That, more than any modelling idea, was the binding constraint on this project.
