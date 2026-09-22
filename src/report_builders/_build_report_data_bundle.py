"""Assemble report_bundle/ -- the notebook plus the small data folder it needs to run standalone.

The competition's own image arrays are 350 MB and are not ours to redistribute, so the bundle
ships: every probability array the notebook actually computes with, the recorded training logs,
the two submitted CSVs, a 20-image sample for the illustration, and a metadata file holding the
dataset shapes the notebook prints.
"""
import csv
import json
import shutil
from pathlib import Path

import numpy as np

BENCH = Path('bench')
OUT = Path('report_bundle')
DATA = OUT / 'data'
for sub in ('bench', 'logs', 'submissions', 'samples'):
    d = DATA / sub
    if d.is_dir():                       # start clean: stale files from an earlier pick survived
        for f in d.iterdir():            # several rebuilds and shipped alongside the real ones
            if f.is_file():
                f.unlink()
    d.mkdir(parents=True, exist_ok=True)

# ---- probability arrays the notebook actually reads -------------------------------------
test_probs = ['dinov2vitlps1', 'eva02lps1', 'dinov3vitlps1', 'dinov3vithpps1',
              'dinov3vithp_base', 'dinov3vithpps1B', 'dinov3vithp_baseB']        # the picks' members
reserve_probs = ['dinov2vitl_base', 'dinov2vitlps1', 'dinov3vitl_base',
                 'dinov3vitlps1', 'eva02l_base', 'eva02lps1',
                 'dinov3vithp_base', 'dinov3vithpps1',
                 'dinov3vithpps1B', 'dinov3vithp_baseB']                        # the §7 results table
for tag in test_probs:
    shutil.copy2(BENCH / f'{tag}_test_prob.npy', DATA / 'bench' / f'{tag}_test_prob.npy')
for tag in reserve_probs:
    shutil.copy2(BENCH / f'{tag}_reserve_prob.npy', DATA / 'bench' / f'{tag}_reserve_prob.npy')
for small in ('class_names.npy', 'reserve_preds.npz'):
    shutil.copy2(BENCH / small, DATA / 'bench' / small)

# ---- recorded training logs -------------------------------------------------------------
RUNS = ['dinov2vitl', 'dinov2vitlps1', 'dinov3vitl', 'dinov3vitlps1', 'eva02l', 'eva02lps1',
        'dinov3vithp', 'dinov3vithpps1']
for tag in RUNS:
    shutil.copy2(f'{tag}_progress.csv', DATA / 'logs' / f'{tag}_progress.csv')
for name in ('dinov2vitl_ps1.log', 'eva02l_ps1.log', 'dinov3vithp_base.log', 'dinov3vithp_ps1.log'):
    lines = Path(name).read_text(encoding='utf-8', errors='ignore').splitlines()
    keep = [l for l in lines if 'warn' not in l.lower()][-40:]
    (DATA / 'logs' / name).write_text('\n'.join(keep) + '\n', encoding='utf-8')

# ---- the two submitted CSVs -------------------------------------------------------------
for name in ('submission_ens_EVAPHP256temp.csv', 'submission_ens_V3PHPbase256temp.csv'):
    shutil.copy2(name, DATA / 'submissions' / name)

# ---- a 20-image sample for the illustration, and the dataset facts the notebook prints ---
train_images = np.load(BENCH / 'train_images.npy', mmap_mode='r')
test_images = np.load(BENCH / 'test_images.npy', mmap_mode='r')
ext_images = np.load(BENCH / 'external_images.npy', mmap_mode='r')
targets = np.load(BENCH / 'targets.npy')
ext_labels = np.load(BENCH / 'external_labels.npy')
val_idx = np.load(BENCH / 'validation_indices.npy')
reserve = np.load(BENCH / 'external_eval_holdout.npz')['indices']

rng = np.random.default_rng(0)
photo_idx = [int(rng.choice(np.flatnonzero(targets == c))) for c in range(10)]
rend_idx = [int(rng.choice(np.flatnonzero(ext_labels == c))) for c in range(10)]
np.savez_compressed(DATA / 'samples' / 'sample_images.npz',
                    photos=np.asarray(train_images[photo_idx]),
                    renditions=np.asarray(ext_images[rend_idx]),
                    classes=np.arange(10))

meta = {
    'train_images_shape': list(train_images.shape),
    'train_images_dtype': str(train_images.dtype),
    'test_images_shape': list(test_images.shape),
    'external_images_shape': list(ext_images.shape),
    'n_train_split': int(len(targets) - len(val_idx)),
    'n_validation': int(len(val_idx)),
    'n_reserved_renditions': int(len(reserve)),
    'per_class_counts': np.bincount(targets, minlength=20).tolist(),
    'note': ('Shapes and counts recorded from the full arrays in the project repository. '
             'train_images.npy is the competition JPEGs (original resolution, 617 MB in '
             'Animals_Dataset.zip) squashed to 64x64 with LANCZOS, the resolution the test set is '
             'delivered at. Those arrays (350 MB) and the external rendition pool (103 MB) are not '
             'redistributed in this bundle; samples/sample_images.npz holds 20 images for the '
             'illustration.'),
}
(DATA / 'samples' / 'dataset_metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')

total = sum(p.stat().st_size for p in DATA.rglob('*') if p.is_file())
n = sum(1 for p in DATA.rglob('*') if p.is_file())
print(f'data/ bundle: {n} files, {total/2**20:.1f} MB')
for p in sorted(DATA.rglob('*')):
    if p.is_file():
        print(f'   {p.relative_to(OUT).as_posix():58s} {p.stat().st_size/1024:8.0f} KB')
