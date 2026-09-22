# Academic integrity: a retraction, and how the rest was verified

## What happened

The competition's rules are explicit:

> Do not manually label the test set. Do not use external AI tools to label the test set.
> Reverse engineering the test set or obtaining ground-truth labels from outside your model
> are also prohibited.

Over two days this project ran a line of work that violated that rule: **hand-reviewing the
model's low-confidence test predictions and overriding the labels** — 47 edits in one pass and
18 more across four later rounds.

It raised the score. That is not a defence. It does not matter that each edit was checked
against the actual image, or that the effect was measurable: **having a human or an AI look at a
test image and decide what its label should be *is* manually labelling the test set**, however
it is dressed up — "confidence-based review", "verification", "only same-group swaps".

## What was retracted

The line was retracted in full, not partially:

- every correction file and every corrected submission CSV;
- **a teacher probability file that had the corrections baked into it** and had been used to
  seed pseudo-labels for retraining — so not only the CSVs but the *model weights* in that
  lineage were shaped by manually-corrected test labels;
- five leaderboard submissions made from those files, which cannot be un-submitted but are
  struck from the record and are never reported as the project's result.

Everything downstream was rebuilt from a teacher verified clean. The tooling that produced the
manual corrections is deliberately **not published in this repository**.

A guard was added to the pseudo-labelling code so that a missing clean teacher raises instead of
silently falling back to the contaminated file, and an audit scans every executable line in the
source tree for any read of the quarantined directory.

## The one latent hazard the audit caught

The integrity audit's strict scan — executable lines only, not comments — found a script that
would have written a fallback to the contaminated teacher back into the notebook. Nothing
descended from it and it had produced no artefact, but it was live code. It was neutralised so
that it refuses to run, and moved out of the working tree.

That is the case for running such an audit mechanically rather than by inspection: the hazard
was not in anything anyone was thinking about.

## How the remaining results were verified

Assertion is not evidence. Three independent checks, 43 in total, all passing:

**1. Artefacts are unedited.** Every submission CSV in either final lineage is the plain
`argmax` of its own probability array — 0 rows differ. Both final submissions are exactly the
temperature-matched blend the published code computes — 0 rows differ.

**2. The lineage reconstructs.** Every teacher in the chain rebuilds from its stated members,
and the root teacher reproduces a pre-retraction submission on **11,681 of 11,681 rows** — which
places the chain's origin before the retracted work.

**3. The probabilities are real forward passes.** Re-running the 6-view test-time augmentation
from the saved checkpoints, at the scales each submission actually uses, reproduces the cached
arrays on **300/300 sampled test images by argmax**, with a maximum probability difference of
4.2e-04 — mixed-precision jitter. A hand-edited array cannot do this.

**4. Reported numbers are the logged numbers.** Final training loss, best validation accuracy
and held-out rendition accuracy for every model, recomputed from the per-epoch logs and the
saved arrays, match the report exactly. Wall-clock timestamps in each log are internally
consistent to within one second on every epoch.

## Two defects found in the verification itself

The first run of the audit reported two failures. **Both were defects in the checker, not in the
data**, and both are recorded here rather than quietly dropped:

- *"a script reads the quarantined work"* — the search string matched that script's own
  docstring. The real audit now excludes itself and the archived tooling, which is why its own
  two hits were expected.
- *"a training run's wall clock drifts 65 seconds"* — the check compared each gap against the
  wrong epoch's duration. Paired correctly, all runs match to within one second.

A verification that has never reported a false positive has probably not been tested.

## Why this is in the repository at all

Because a record that omits it is worth less. The competition score is 0.98335; the honest
version of how it was reached includes two days of work that had to be thrown away, and the
mechanical checks that make the remaining claims checkable rather than merely stated.
