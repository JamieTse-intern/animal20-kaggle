# Animal-20: a 64x64 image-classification competition, and what it taught about measurement

A 20-class animal/object classifier for a Kaggle-hosted university competition.
**Final public score 0.98335**, from 0.740 at the first attempt.

The models are ordinary — fine-tuned Vision Transformers. The part worth reading is
**how decisions were made**, because every local evaluation metric this project built
eventually failed, and the record of those failures is more transferable than the score.

| | |
|---|---|
| Task | 20 classes, 9,466 labelled training images, 11,681 unlabelled test images at **64x64** JPEG q75 |
| Final | **0.98335** public leaderboard, 2 submissions selected from 70 |
| Hardware | one RTX 3080, **10 GB** — the binding constraint on everything |
| Scale | 109 training runs, 70 scored submissions, ~30 GPU-hours in the final generation |

## The arc

```
0.740   first model             domain mismatch: training images were clean, test images are
                                64px JPEG q75. Replicating the exact degradation was worth
                                more than every architecture change that followed.
0.947   + external renditions   ~21-27% of the test set is paintings/drawings/sculpture.
0.973   ConvNeXt-V2 (FCMAE)     masked-autoencoder pretraining beat CLIP contrastive.
0.980   DINOv2 / DINOv3 ViT-L   self-supervised, Meta-curated pretraining. The single
                                biggest lever in the whole project.
0.983   ViT-H+/16 at 840M       capacity, once three memory techniques made it fit in 10 GB.
0.98335 final                   + native-grid test-time augmentation.
```

## What actually moved the score

Ranked by measured leaderboard effect, not by how much work each took.

| lever | effect | note |
|---|---|---|
| Matching the test degradation exactly | +0.119 | squash-crop, LANCZOS to 64px, re-encoded with the **test set's own JPEG quantisation tables** |
| External rendition data | +0.025 | DomainNet *painting* + ImageNet-R, pushed through that same degradation and filtered on radial-spectrum distance |
| Changing the pretraining recipe | +0.006 to +0.013 | the only lever that paid **repeatedly**: FCMAE, then DINOv2, then DINOv3 |
| One round of Noisy-Student self-distillation | +0.006 to +0.012 | pseudo-labelled test images from a model-generated teacher |
| Capacity, *inside a recipe that already works* | +0.004 | 303M to 840M. The same bet in the ConvNeXt family had failed at 658M |
| Probability blending of two decorrelated models | +0.001 to +0.004 | temperature-matched, equal weight, **no parameter fitted on the leaderboard** |

## What did not work, and why it is worth knowing

Every one of these was run to completion and scored, not abandoned on a hunch.

| | result |
|---|---|
| Training at higher resolution (256px) | tested twice, on a ConvNeXt **and** on a ViT — lost both times. The sources are 64x64, so the extra tokens carry interpolated pixels, not information |
| Focal loss | better on **both** held-out sets and **worse** on the leaderboard (see below) |
| Cross-stage weight soup | lands between its ingredients; averaging weights dilutes exactly the test-distribution adaptation that makes the stronger one good |
| FixMatch | highest validation accuracy in the project (0.9922) and the lowest leaderboard score of its family (0.96926) |
| Language-supervised backbones (SigLIP 2, Web-SSL) | 2.7 and 2.8 points below the same recipe on DINO weights |
| Label propagation on a kNN feature graph | 3 to 4 rows broken for every one fixed; killed for 5 GPU-minutes of measurement instead of a submission |
| Rank averaging / power averaging / class-prior rebalancing | 3.4 points lost, an exact no-op, and 0.5 to 1.1 points lost respectively |

## The finding this project is actually about

**Every local evaluation metric built here eventually failed, and they failed in ways that
were only visible from the leaderboard.**

| instrument | how it failed |
|---|---|
| Validation accuracy, 1,419 held-out images | saturated at ~99%, and once ranked the *worst* model first |
| A synthetic sketch metric | circular — it measured the same transform it trained on |
| A reserved set of 772 real renditions | valid for same-stage direction (**7-for-7** on significant gaps), **inverts** across the distillation boundary, and failed **four times out of four** at ranking blends — the last time with Spearman −0.95, a near-perfect inversion |
| A distribution-matched estimator built to fix all of the above | ranked the worst model first, on **both** strata |
| A 3,124-image photo holdout from a different public dataset | could not rank the models at all; every paired p was at least 0.31 |

The sharpest single case, and the one that closes the question:

> A focal-loss model, identical to its control except for the loss function, scored
> **better on validation photos** (0.9958 vs 0.9951 — the best base model in the project)
> **and better on held-out renditions** (94.04% vs 92.88%, +1.17pt) — and **worse on the
> leaderboard** (0.97385 vs 0.97556).
>
> Better on every held-out set the project owns. Worse on the only one that counts.

The mechanism is not mysterious: the distillation stage trains on pseudo-labelled **test**
images, and no held-out proxy contains those images. But the consequence is what matters —
**a better local proxy could not have been built from data drawn from these sources.**

## How decisions were made instead

With no local metric to trust, every decision ran through a **paired win-rate test on the
leaderboard**: compare two submissions only on the rows where they disagree, and call it a
signal only when they differ on at least 60 rows *and* the winner takes at least 60% of the
differing public rows. Most differences are ties. Saying so was usually the finding.

Two practices did the real work.

**Pre-registration.** Before a run started, the model, the recipe, the gate, the screen and an
explicit **prediction with a probability** went into the log. Afterwards the prediction was
scored — including, repeatedly, as wrong. `results/leaderboard.csv` carries all 70. A sample
of the predictions that failed, and what each one taught:

- *"EVA-02's base will pass its gate"* — missed by 0.085pt. The line was continued anyway as a
  labelled post-hoc override, and became the best result in the project.
- *"ViT-H+ pseudo1 has the widest upside"* — smallest distillation gain of any lineage, by 9x.
  Its base was already too strong to have headroom left.
- *"Native-grid training is the most promising of three levers"* — the worst of the three.
- *"Focal loss will not help"* — right about the leaderboard, wrong about the mechanism, and
  wrong about which channel it acts through.

**Reserving the measurement instrument before optimising against it.** Three separate bugs came
from letting training-side logic decide what counted as held out. The held-out rendition set is
now written to disk with a fingerprint *before* any training draw can see it, and the training
code raises if that fingerprint stops matching.

## Fitting an 840M-parameter model in 10 GB

Full fine-tuning of DINOv3 ViT-H+/16 needs about 18 GB: parameters 3.13, gradients 3.13, Adam
moments 6.26, fp32 EMA 3.13. Almost none of it is activations, so a smaller batch does not help.
Three changes, none of which touches the model, the data or the objective, bring the measured
peak to **8.3 GB at the same batch size**:

| change | saves |
|---|---|
| 8-bit Adam moments (`bitsandbytes.AdamW8bit`), master weights still fp32 | 4.7 GB |
| the EMA copy held in system RAM, swapped in only to validate | 3.13 GB |
| the EMA updated every 4 steps at `decay**4`, so its horizon in steps is unchanged | speed, not memory |

Master weights deliberately stay fp32: at lr 2e-5 on weights of magnitude ~0.02 the relative
update is 1e-3 while bf16 resolves about 8e-3, so every update would be rounded away and the
run would silently learn nothing.

The same stack was then measured on a 1.14B model and **failed** — 9.81 GB steady with
gradients released and no EMA at all, because at depth 40 the gradient-checkpoint boundaries
are themselves multiple GB. Counting only parameters and optimiser state is not a feasibility
test; activations scale with depth x width x tokens and there they are the binding term.

## Academic integrity

One line of work in this project — hand-reviewing low-confidence test predictions and
overriding labels — **violated the competition's rule against manually labelling the test set,
and was retracted in full**, including a teacher probability file that had corrections baked
into it and had been used to seed pseudo-labels for retraining. Those submissions are struck
from the record, that tooling is not published in this repository, and every result reported
here descends from a lineage verified clean.

The verification is mechanical rather than asserted: every submission CSV is the plain argmax
of its own probability array; every teacher reconstructs from its stated members; and
re-running inference from the saved checkpoints reproduces the cached arrays on 300/300
sampled test images. See [`docs/03-integrity.md`](docs/03-integrity.md).

## Repository layout

```
src/train/       the two training stages, the FixMatch variant, a weight-format converter
src/data/        external-data construction, the test-degradation pipeline, the held-out reserve
src/blend/       temperature-matched blending and every pre-registered blend screen
src/eval/        integrity audits, checkpoint re-verification, the held-out evaluators
src/analysis/    the diagnostics and the negative results
results/         109 runs, 70 scored submissions, 19 per-epoch training logs, 36 probability arrays
docs/            findings, the full experiment log, integrity, the measurement scorecard
reproduce/       rebuilds both final submissions from the saved arrays and verifies them row-for-row
```

## Reproducing

The competition data and the third-party rendition datasets are not redistributed here. What
*is* here is every model's own output — the `(11681, 20)` probability array each run produced —
so the final submissions can be rebuilt and checked without a GPU:

```bash
python reproduce/reproduce.py     # "both picks reproduce exactly", 0 rows differ
```

## Licence

Code is MIT. Trained checkpoints (8.5 GB) are not in this repository. The competition data
belongs to the unit that set the assignment and is not redistributed.
