"""Single source of truth for the held-out real-rendition evaluation set.

Loads the PERMANENT reserve written by _make_eval_holdout.py. Never re-derives a split by
mirroring the notebook's training-draw logic -- that coupling is what broke twice before
(a SEED change silently moved the validation split; the floor-balanced draw silently ate the
whole pool of the classes under test). Reserve first, load the same file everywhere.
"""
import hashlib
import numpy as np

CN = ['birds','bottles','breads','butterflies','cakes','cats','chickens','cows','dogs','ducks',
      'elephants','fishes','handguns','horses','lions','lipsticks','seals','snakes','spiders','vases']


def _fingerprint(labels):
    h = hashlib.sha256()
    h.update(np.asarray(labels, dtype=np.int64).tobytes())
    return h.hexdigest()[:16]


def load_eval_set(images_path='bench/external_images.npy', labels_path='bench/external_labels.npy'):
    """Returns (images, labels) of the reserved evaluation set, or raises if it is stale."""
    images = np.load(images_path)
    labels = np.load(labels_path)
    reserve = np.load('bench/external_eval_holdout.npz')
    if int(reserve['pool_size']) != len(labels):
        raise RuntimeError(
            f'reserve was built against a pool of {int(reserve["pool_size"])} but the pool is now '
            f'{len(labels)} -- external data was rebuilt; re-run _make_eval_holdout.py')
    if str(reserve['fingerprint']) != _fingerprint(labels):
        raise RuntimeError('external label fingerprint changed since the reserve was built -- '
                           're-run _make_eval_holdout.py and re-evaluate anything cached')
    idx = reserve['indices']
    return images[idx], labels[idx]


if __name__ == '__main__':
    img, lab = load_eval_set()
    counts = np.bincount(lab, minlength=20)
    print(f'reserved evaluation set: {len(lab)} images')
    for c in np.argsort(counts):
        note = '   <-- too thin to resolve small effects' if counts[c] < 20 else ''
        print(f'  {CN[c]:12s} {counts[c]:4d}{note}')
