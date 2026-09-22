"""Independent re-check of the two final picks: provenance, artefacts, and every reported number.

Answers two questions with evidence, not assertion:
  1. Does anything in either pick's lineage touch a manually/AI-assigned test label?
  2. Is every number in the report and notebook the number the logs actually recorded?

Nothing here reads the quarantined `_RETRACTED_manual_test_labeling/` directory.
"""
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from blend_temp import blend, load

BENCH = Path('bench')
CN = list(np.load(BENCH / 'class_names.npy', allow_pickle=True))
ok = []


def check(passed, title, detail):
    ok.append(bool(passed))
    print(f'[{"PASS" if passed else "FAIL"}] {title}\n       {detail}')


def labels_of(p):
    return np.array([CN[i] for i in np.asarray(p).argmax(1)])


print('=' * 100)
print('A. ARTEFACTS: is every submitted CSV exactly the argmax of a model output, i.e. unedited?')
print('=' * 100)

# Every CSV in either pick's lineage must be the argmax of its own probability array.
singles = {'submission_dinov2vitlps1.csv': 'dinov2vitlps1_test_prob.npy',      # V2P, both picks
           'submission_eva02lps1.csv': 'eva02lps1_test_prob.npy',              # EVAP, pick 1
           'submission_dinov3vitlps1.csv': 'dinov3vitlps1_test_prob.npy',      # V3P, pick 2
           'submission_dinov2vitl_base.csv': 'dinov2vitl_base_test_prob.npy',
           'submission_dinov3vitl_base.csv': 'dinov3vitl_base_test_prob.npy',
           'submission_eva02l_base.csv': 'eva02l_base_test_prob.npy'}
for csv_name, arr in singles.items():
    if not Path(csv_name).is_file():
        continue
    n = int((pd.read_csv(csv_name)['Label'].values != labels_of(np.load(BENCH / arr))).sum())
    check(n == 0, f'{csv_name} is the plain argmax of {arr}',
          f'{n} rows differ — a hand edit anywhere would show up here')

for pick, tags, csv_name in [('pick 1', ['V2P', 'EVAP'], 'submission_ens_V2PEVAPtemp.csv'),
                             ('pick 2', ['V2P', 'V3P'], 'submission_ens_V2PV3Ptemp.csv')]:
    prob, _, _ = blend(tags, mode='arithmetic')
    n = int((pd.read_csv(csv_name)['Label'].values != labels_of(prob)).sum())
    check(n == 0, f'{pick} {csv_name} = temperature-matched blend({"+".join(tags)})',
          f'{n} rows differ from the arithmetic the code performs')

print()
print('=' * 100)
print('B. LINEAGE: walk both picks back to the root, teacher by teacher')
print('=' * 100)

# Each student was trained on pseudo-labels from a teacher; each teacher is a blend of models.
lineage = [
    ('EVAP (pick 1)', 'teacher_V2PV3P.npy', ['V2P', 'V3P']),
    ('V3P (pick 2)', 'teacher_V2P.npy', ['V2P']),
    ('V2P (both picks)', 'teacher_KDCD3temp.npy', ['KD', 'C', 'D3']),
    ('KD', 'teacher_ACD3temp_test.npy', ['A', 'C', 'D3']),
]
for student, teacher_file, members in lineage:
    stored = np.load(BENCH / teacher_file).astype(np.float64)
    stored = stored / stored.sum(1, keepdims=True)
    rebuilt, _, _ = blend(members, mode='arithmetic')
    n = int((labels_of(stored) != labels_of(rebuilt)).sum())
    check(n == 0, f'{student}: teacher {teacher_file} = blend of {"+".join(members)}',
          f'{n} rows differ; the teacher is model output, not a supplied label set')

# D3's teacher is a geometric blend of A and C.
A, C = (load('A'), load('C'))
logs = lambda p: np.log(np.clip(p, 1e-12, None))
geo = np.exp((logs(A) + 2 * logs(C)) / 3)
geo /= geo.sum(1, keepdims=True)
d = float(np.abs(np.load(BENCH / 'teacher_AC12.npy') - geo).max())
check(d < 1e-5, 'D3: teacher_AC12 = A (x) C in logit space at 1:2',
      f'max deviation {d:.2e}')

# The root of the whole chain must predate the retracted manual-correction work.
root = np.load(BENCH / 'teacher_clean_test.npy').astype(np.float64)
root = root / root.sum(1, keepdims=True)
pre = pd.read_csv('submission_D3_LgP.csv')['Label'].values
n = int((labels_of(root) != pre).sum())
check(n == 0, 'root teacher (teacher_clean_test) reproduces the PRE-correction submission',
      f'{n} of {len(pre)} rows differ — the root predates, and is unaffected by, the retracted work')

print()
print('=' * 100)
print('C. QUARANTINE: nothing in the pipeline reads the retracted work')
print('=' * 100)

needles = ['_RETRACTED_manual_test_labeling', 'best_ens_test_prob.npy', 'apply_corrections']
# This file and integrity_audit.py are excluded: both necessarily contain the strings they hunt for.
sources = sorted(p for p in Path('.').glob('*.py')
                 if p.name not in {Path(__file__).name, 'integrity_audit.py'})
sources += [Path(n) for n in ('FIT5215_v8.ipynb', 'FIT5215_A1_methodology.ipynb',
                              '32613571_assignment01_notebook.ipynb') if Path(n).is_file()]
hits = []
for f in sources:
    if f.suffix == '.ipynb':
        nb = json.loads(f.read_text(encoding='utf-8'))
        lines = [(f'cell {i}', l) for i, c in enumerate(nb['cells']) if c['cell_type'] == 'code'
                 for l in ''.join(c['source']).split('\n')]
    else:
        lines = [(f'line {i}', l) for i, l in enumerate(f.read_text(encoding='utf-8',
                                                                   errors='ignore').split('\n'), 1)]
    hits += [f'{f}:{loc}' for loc, l in lines
             if not l.strip().startswith('#') and any(x in l for x in needles)]
check(not hits, 'no executable line in any source reads the quarantined work',
      f'scanned {len(sources)} files for {needles}: ' + (f'OFFENDERS {hits}' if hits else 'none found'))

# The eight retracted submissions must not be among the files we are submitting.
retracted = {'submission_corr_B.csv', 'submission_A3.csv', 'submission_2seed_corr.csv',
             'submission_corr_cnv2L.csv', 'submission_corr_cnv2L_r3.csv', 'submission_tta18_corr.csv'}
pkg = {p.name for p in Path('submission_package').rglob('*') if p.is_file()}
check(not (pkg & retracted), 'no retracted CSV is inside the submitted package',
      f'{len(pkg)} package files checked against the retracted set')

print()
print('=' * 100)
print('D. REPORTED NUMBERS vs the logs that recorded them')
print('=' * 100)

claims = {  # model: (progress csv, final training loss, best validation %, reserve prob file, reserve %)
    'V2': ('dinov2vitl_progress.csv', 0.8997, 99.44, 'dinov2vitl_base_reserve_prob.npy', 93.78),
    'V2P': ('dinov2vitlps1_progress.csv', 0.8558, 99.30, 'dinov2vitlps1_reserve_prob.npy', 93.01),
    'V3': ('dinov3vitl_progress.csv', 0.9121, 99.51, 'dinov3vitl_base_reserve_prob.npy', 92.88),
    'V3P': ('dinov3vitlps1_progress.csv', 0.8666, 99.51, 'dinov3vitlps1_reserve_prob.npy', 94.30),
    'EVA': ('eva02l_progress.csv', 0.8766, 99.30, 'eva02l_base_reserve_prob.npy', 92.49),
    'EVAP': ('eva02lps1_progress.csv', 0.8572, 99.51, 'eva02lps1_reserve_prob.npy', 92.88),
}
lab = np.load(BENCH / 'reserve_preds.npz')['lab']
for name, (log_file, loss_claim, val_claim, reserve_file, reserve_claim) in claims.items():
    rows = [r for r in csv.DictReader(open(log_file, encoding='utf-8')) if r['epoch'].isdigit()]
    loss = float(rows[-1]['loss'])
    val = max(max(float(r['val_acc']), float(r['ema_val_acc'] or 0)) for r in rows) * 100
    acc = 100 * (np.load(BENCH / reserve_file).argmax(1) == lab).mean()
    good = abs(loss - loss_claim) < 5e-5 and abs(val - val_claim) < 0.005 and abs(acc - reserve_claim) < 0.005
    check(good, f'{name}: report numbers match the recorded log and the saved outputs',
          f'final loss {loss:.4f} (report {loss_claim}) | best val {val:.2f}% (report {val_claim}) | '
          f'renditions {acc:.2f}% (report {reserve_claim}) | {len(rows)} epochs logged')

for label, tags, claim in [('pick 1 V2P+EVAP', ['V2P', 'EVAP'], 94.04),
                           ('pick 2 V2P+V3P', ['V2P', 'V3P'], 94.43)]:
    norm = lambda p: p / p.sum(1, keepdims=True)
    members = [norm(np.load(BENCH / f'{t.lower().replace("v2p", "dinov2vitlps1").replace("evap", "eva02lps1").replace("v3p", "dinov3vitlps1")}_reserve_prob.npy').astype(np.float64)) for t in tags]
    from blend_temp import fit_temperature, sharpen
    target = members[0].max(1).mean()
    tests = [load(t) for t in tags]
    temps = [1.0] + [fit_temperature(tests[i], tests[0].max(1).mean()) for i in range(1, len(tags))]
    res = sum(m if T == 1.0 else sharpen(m, T) for m, T in zip(members, temps)) / len(members)
    acc = 100 * (res.argmax(1) == lab).mean()
    check(abs(acc - claim) < 0.005, f'{label}: rendition accuracy in the report',
          f'recomputed {acc:.2f}% (report {claim}%)')

print()
print('=' * 100)
print('E. TIMELINE: do the logs look like runs that actually happened?')
print('=' * 100)
for name, (log_file, *_rest) in claims.items():
    rows = [r for r in csv.DictReader(open(log_file, encoding='utf-8')) if r['epoch'].isdigit()]
    walls = [datetime.strptime(r['wall'], '%H:%M:%S') for r in rows]
    secs = [float(r['seconds']) for r in rows]
    # Each row is stamped when it is written, i.e. at the END of that epoch, so the gap between
    # two stamps must equal the SECOND epoch's own measured duration.
    gaps = [(walls[i + 1] - walls[i]).total_seconds() % 86400 for i in range(len(walls) - 1)]
    drift = max(abs(g - s) for g, s in zip(gaps, secs[1:]))
    monotone = all(g > 0 for g in gaps)
    check(monotone and drift <= 5,
          f'{name}: wall-clock stamps advance by the epoch durations the run measured',
          f'{len(rows)} epochs, {min(secs):.0f}-{max(secs):.0f} s each, worst stamp/duration '
          f'mismatch {drift:.0f} s (a fabricated table would not line up to the second)')

print()
print('=' * 100)
print(f'{sum(ok)}/{len(ok)} checks passed — ' + ('ALL CLEAR' if all(ok) else 'ATTENTION REQUIRED'))
print('=' * 100)
raise SystemExit(0 if all(ok) else 1)
