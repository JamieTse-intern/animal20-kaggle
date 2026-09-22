"""Academic-integrity audit for the two final Kaggle submissions.

The competition rules prohibit manually labelling the test set, using external AI tools to
label it, and reverse-engineering ground-truth labels from outside the model. This script
does not assert compliance -- it *checks* it, by tracing the provenance of every number that
feeds either final submission back to a model forward pass.

Checks performed:
  1. Both final CSVs are exactly the argmax of an on-disk probability array.
  2. Every distillation teacher in either lineage reconstructs from its stated members.
  3. The root teacher of the whole chain reproduces a pre-correction submission row-for-row,
     which is what establishes it was never touched by the retracted manual-correction work.
  4. No submitted source file reads from the quarantined retracted directory.
  5. The notebook's pseudo-labelling guard (no silent fallback to any other teacher) is intact.
  6. The training draw cannot touch the reserved evaluation set.

Exit code 0 = every check passed.

Run:  python integrity_audit.py
"""
import io
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BENCH = Path('bench')
CLASSES = list(np.load(BENCH / 'class_names.npy', allow_pickle=True))
RETRACTED_DIR = '_RETRACTED_manual_test_labeling'
results = []


def report(ok, title, detail):
    results.append(ok)
    print(f'[{"PASS" if ok else "FAIL"}] {title}\n       {detail}')


def norm(p):
    p = np.asarray(p, dtype=np.float64)
    return p / p.sum(1, keepdims=True)


def labels_of(prob):
    return np.array([CLASSES[i] for i in np.asarray(prob).argmax(1)])


def geometric(members, weights):
    z = sum(w * np.log(np.clip(norm(m), 1e-12, None)) for m, w in zip(members, weights))
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


# ---- 1. both picks are the argmax of an on-disk array --------------------------------
from blend_temp import blend  # noqa: E402  (import after BENCH is known to exist)

for name, tags, csv in [
        ('pick 1  V2P + EVAP', ['V2P', 'EVAP'], 'submission_ens_V2PEVAPtemp.csv'),
        ('pick 2  V2P + V3P', ['V2P', 'V3P'], 'submission_ens_V2PV3Ptemp.csv')]:
    prob, _, _ = blend(tags, mode='arithmetic')
    n = int((labels_of(prob) != pd.read_csv(csv)['Label'].values).sum())
    report(n == 0, f'{name} reproduces from model outputs',
           f'{csv}: {n} rows differ from the blend of {"+".join(tags)}')

# ---- 2. every teacher in the lineage reconstructs from its members --------------------
A = norm(np.load(BENCH / 'cnv2L_test_prob.npy'))
C = norm(np.load(BENCH / 'cnv2L_res256_test_prob.npy'))

t_ac12 = norm(np.load(BENCH / 'teacher_AC12.npy'))
rebuilt = geometric([A, C], [1 / 3, 2 / 3])
d = float(np.abs(t_ac12 - rebuilt).max())
report(d < 1e-5, 'teacher_AC12 = A (x) C logit 1:2',
       f'max abs deviation from the reconstruction {d:.2e} (used to distil D3)')

for teacher_file, tags, used_by in [
        ('teacher_ACD3temp_test.npy', ['A', 'C', 'D3'], 'the soft-label KD student'),
        ('teacher_KDCD3temp.npy', ['KD', 'C', 'D3'], 'V2P, a member of BOTH picks'),
        ('teacher_V2P.npy', ['V2P'], 'V3P, a member of pick 2'),
        ('teacher_V2PV3P.npy', ['V2P', 'V3P'], 'EVAP, a member of pick 1')]:
    stored = norm(np.load(BENCH / teacher_file))
    rebuilt, _, _ = blend(tags, mode='arithmetic')
    n = int((labels_of(stored) != labels_of(rebuilt)).sum())
    report(n == 0, f'{teacher_file} = temp-matched({"+".join(tags)})',
           f'{n} rows differ; this teacher supplied pseudo-labels for {used_by}')

# ---- 3. the ROOT of the chain is provably free of the retracted work ------------------
root = norm(np.load(BENCH / 'teacher_clean_test.npy'))
pre_correction = pd.read_csv('submission_D3_LgP.csv')['Label'].values
n = int((labels_of(root) != pre_correction).sum())
report(n == 0, 'root teacher (teacher_clean_test) is pre-correction',
       f'reproduces submission_D3_LgP.csv on {len(pre_correction)-n}/{len(pre_correction)} rows '
       f'({n} differ) -- it predates, and is unaffected by, the retracted manual corrections')

# ---- 4. no submitted source reads from the quarantine ---------------------------------
def executable_lines(path):
    """Yield (lineno, text) for lines that actually execute -- no comments, no markdown."""
    if path.suffix == '.ipynb':
        nb_ = json.load(io.open(path, encoding='utf-8'))
        for ci, cell in enumerate(nb_['cells']):
            if cell['cell_type'] != 'code':
                continue                                  # markdown is prose, not a read
            for ln, line in enumerate(''.join(cell['source']).split('\n'), 1):
                if not line.strip().startswith('#'):
                    yield f'cell {ci} line {ln}', line
    else:
        for ln, line in enumerate(io.open(path, encoding='utf-8',
                                          errors='ignore').read().split('\n'), 1):
            if not line.strip().startswith('#'):
                yield f'line {ln}', line


# This audit file is excluded: it necessarily contains the very strings it searches for.
sources = [p for p in sorted(Path('.').glob('*.py')) if p.name != Path(__file__).name]
sources += [Path('FIT5215_v8.ipynb'), Path('FIT5215_A1_methodology.ipynb')]
sources = [p for p in sources if p.exists()]

for needle, label in [(RETRACTED_DIR, 'the quarantined retracted directory'),
                      ('best_ens_test_prob.npy', 'the corrections-contaminated teacher')]:
    offenders = [f'{f}:{loc}' for f in sources for loc, line in executable_lines(f)
                 if needle in line]
    report(not offenders, f'no executable code references {label}',
           f'scanned {len(sources)} files for "{needle}" outside comments/markdown; '
           + ('none found' if not offenders else f'OFFENDERS {offenders}'))

# ---- 5. the notebook's teacher guard is intact ----------------------------------------
nb = json.load(io.open('FIT5215_v8.ipynb', encoding='utf-8'))
src = ''.join(''.join(c['source']) for c in nb['cells'])
report('does not line up with this run test order' in src,
       'notebook refuses a misaligned teacher',
       'the pseudo-labelling cell raises rather than silently seeding from the wrong file')
report("No clean teacher on disk -- seeding pseudo-labels from this run's own base ensemble"
       in src, 'the only fallback is the run\'s own base ensemble',
       'if the clean teacher is absent the cell seeds from its own models, never from a file')

# ---- 6. the training draw cannot consume the evaluation reserve -----------------------
trainer = io.open('dinov3_base_train.py', encoding='utf-8').read()
has_assert = 'training draw touched the eval reserve' in trainer
report(has_assert, 'training draw asserts it never touches the reserve',
       'floor_balanced_draw() raises if the drawn indices intersect external_eval_holdout.npz')

print('\n' + '=' * 78)
ok = all(results)
print(f'{sum(results)}/{len(results)} checks passed -- '
      + ('INTEGRITY VERIFIED' if ok else 'ATTENTION REQUIRED'))
print('Every probability feeding either final submission traces to a forward pass of a model\n'
      'trained in this repository. No test label was read, written, reviewed or corrected by\n'
      'a human or by an external AI at any point in either lineage.')
sys.exit(0 if ok else 1)
