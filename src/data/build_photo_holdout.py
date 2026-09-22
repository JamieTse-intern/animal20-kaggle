"""Build a labelled, NON-saturated PHOTO holdout for evaluation only.

Why it exists: the competition validation set (1,419 photos) sits at ~99% and cannot rank
models; the rendition reserve (772 images) measures only renditions. When two models differ
in opposite directions across the photo and rendition strata -- DINOv3 ViT-L base vs DINOv2
ViT-L base did -- the reserve predicts the wrong sign. This set measures the photo stratum.

Source: DomainNet `real`, OFFICIAL TEST SPLIT, fetched by `real_test.txt` path. Nothing in
this project has ever trained on DomainNet `real`.

Design decisions, fixed before any model was evaluated on this set:
  * exact-name classes only (17). The looser training mappings (rifle->handguns,
    swan->ducks, shark->fishes, ...) are excluded: they would score models against class
    boundaries the competition test set may not share. chickens, seals and handguns have
    no exact DomainNet class and are therefore not covered.
  * degraded with prepare_external.degrade -- imported, not reimplemented -- using the
    test set's own JPEG quantisation tables.
  * NO spectrum filtering: keeping only test-like images would bias toward easy cases.
  * near-duplicates of the competition TRAINING images are removed (the training set's
    source is unknown). Holdout images are deliberately NEVER compared against test
    images: matching test images to labelled external photos would be obtaining labels
    from outside the model, which the assignment prohibits.
  * written outside bench/ with a fingerprint; no trainer reads this directory.

Run:  python build_photo_holdout.py
"""
import csv
import hashlib
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from prepare_external import (CLASS_INDEX, CLASS_NAMES, degrade, laplacian_variance,
                              radial_spectrum, read_encoder_fingerprint)

OUT_DIR = Path('photo_holdout')
OUT_FILE = OUT_DIR / 'photo_holdout.npz'
DUP_COSINE = 0.98          # 16x16 grey, mean-centred; near-identical images score ~0.99+

assert 'bench' not in OUT_FILE.parts, 'the holdout must never live where trainers read'


def thumbprint(batch_nchw):
    """Mean-centred, L2-normalised 16x16 greyscale vectors for near-duplicate search."""
    grey = batch_nchw.astype(np.float32).mean(axis=1)                  # (N,64,64)
    small = grey.reshape(len(grey), 16, 4, 16, 4).mean(axis=(2, 4))    # (N,16,16)
    v = small.reshape(len(small), -1)
    v -= v.mean(axis=1, keepdims=True)
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-6)


def main():
    rows = list(csv.DictReader(open(OUT_DIR / 'manifest.csv', encoding='utf-8')))
    if not rows:
        sys.exit('manifest.csv is empty -- run the fetch first')
    qtables, subsampling = read_encoder_fingerprint(Path('test_set/test_dataset'))

    images, labels, paths, sha1s, failures = [], [], [], [], 0
    for r in rows:
        f = OUT_DIR / 'raw' / r['our_class'] / Path(r['zip_path']).name
        try:
            with Image.open(f) as handle:
                images.append(degrade(handle, qtables, subsampling))
        except Exception:
            failures += 1
            continue
        labels.append(CLASS_INDEX[r['our_class']])
        paths.append(r['zip_path'])
        sha1s.append(r['sha1'])
    images = np.ascontiguousarray(np.stack(images).transpose(0, 3, 1, 2))   # NCHW, like bench
    labels = np.asarray(labels, dtype=np.int64)
    print(f'degraded {len(images)} photos, {failures} failed')

    # ---- near-duplicates of the competition TRAINING images (never the test set) --------
    train = np.load('bench/train_images.npy', mmap_mode='r')
    h, t = thumbprint(images), thumbprint(np.asarray(train))
    best = np.zeros(len(h), dtype=np.float32)
    for i in range(0, len(h), 512):
        best[i:i + 512] = (h[i:i + 512] @ t.T).max(axis=1)
    dup = best >= DUP_COSINE
    print(f'near-duplicates of training images (cosine >= {DUP_COSINE}): {int(dup.sum())} '
          f'-> removed; max similarity median {np.median(best):.3f}')
    images, labels = images[~dup], labels[~dup]
    paths = [p for p, d in zip(paths, dup) if not d]
    sha1s = [s for s, d in zip(sha1s, dup) if not d]

    # ---- domain self-check: aggregate pixel statistics, same check as the external data --
    nhwc = images.transpose(0, 2, 3, 1)
    test_ref = np.ascontiguousarray(np.load('bench/test_images.npy', mmap_mode='r')[:2000]
                                    ).transpose(0, 2, 3, 1)
    val_idx = np.load('bench/validation_indices.npy')
    val_ref = np.ascontiguousarray(np.asarray(train)[val_idx]).transpose(0, 2, 3, 1)
    ref_spec = radial_spectrum(test_ref)
    print('\ndomain self-check vs the test set (aggregate statistics only)')
    print(f'  {"set":24s} {"lapvar":>8} {"spectrum dist":>14}')
    print(f'  {"test set":24s} {laplacian_variance(test_ref):8.4f} {0:14.4f}')
    for name, batch in [('photo holdout (new)', nhwc[:2000]), ('validation photos', val_ref)]:
        d = float(np.abs(radial_spectrum(batch) - ref_spec).mean())
        print(f'  {name:24s} {laplacian_variance(batch):8.4f} {d:14.4f}')

    # ---- fingerprint + save ------------------------------------------------------------
    digest = hashlib.sha1()
    digest.update(images.tobytes())
    digest.update(labels.tobytes())
    covered = sorted({CLASS_NAMES[i] for i in labels})
    np.savez_compressed(OUT_FILE, images=images, labels=labels, zip_paths=np.array(paths),
                        sha1=np.array(sha1s), fingerprint=digest.hexdigest(),
                        classes_covered=np.array(covered))
    counts = Counter(CLASS_NAMES[i] for i in labels)
    print(f'\nwrote {OUT_FILE}: {len(labels)} images, {len(covered)} classes, '
          f'fingerprint {digest.hexdigest()[:12]}')
    print('  ' + '  '.join(f'{c}:{counts[c]}' for c in covered))
    print('  not covered:', sorted(set(CLASS_NAMES) - set(covered)))

    # ---- contact sheet of HOLDOUT images for a label/quality sanity check -----------------
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        picks = np.random.default_rng(0).choice(len(images), size=min(48, len(images)),
                                                replace=False)
        fig, axes = plt.subplots(6, 8, figsize=(16, 12.5))
        for ax, k in zip(axes.ravel(), picks):
            ax.imshow(nhwc[k])
            ax.set_title(CLASS_NAMES[labels[k]], fontsize=9)
            ax.axis('off')
        fig.suptitle('photo holdout after test-pipeline degradation (DomainNet real, test split)')
        fig.tight_layout()
        fig.savefig(OUT_DIR / 'contact.png', dpi=80)
        print(f'contact sheet -> {OUT_DIR / "contact.png"}')
    except Exception as error:
        print('contact sheet failed (data unaffected):', error)
    return 0


if __name__ == '__main__':
    sys.exit(main())
