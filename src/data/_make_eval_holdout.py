"""Reserve a permanent held-out evaluation set from the external rendition pool, BEFORE any
training draw ever touches it.

Why this file exists: the floor-balanced training draw consumed the ENTIRE pool of the very
classes it was designed to help (bottles/cows/seals hit 0 held-out images, chickens 23), so
the one non-circular metric this project has for rendition generalisation went blind on
exactly the classes under test. Reserve first, train on what's left -- never the reverse.

Writes bench/external_eval_holdout.npz holding the reserved indices plus a fingerprint of the
pool they were drawn from, so any consumer can detect staleness instead of silently scoring
against the wrong images (the same failure mode as the SEED/validation-split bug).
"""
import hashlib
import numpy as np
import torch

CN = ['birds','bottles','breads','butterflies','cakes','cats','chickens','cows','dogs','ducks',
      'elephants','fishes','handguns','horses','lions','lipsticks','seals','snakes','spiders','vases']
RESERVE_CAP = 40          # never reserve more than this per class
RESERVE_FRACTION = 0.25   # ...nor more than this share of a thin class's pool
SEED = 20260912


def pool_fingerprint(labels):
    h = hashlib.sha256()
    h.update(np.asarray(labels, dtype=np.int64).tobytes())
    return h.hexdigest()[:16]


def build(labels):
    labels = np.asarray(labels)
    picker = torch.Generator().manual_seed(SEED)
    reserved = []
    for c in range(len(CN)):
        pool = np.flatnonzero(labels == c)
        if len(pool) == 0:
            continue
        n = min(RESERVE_CAP, int(round(RESERVE_FRACTION * len(pool))))
        if n == 0:
            continue
        order = pool[torch.randperm(len(pool), generator=picker).numpy()]
        reserved.append(order[:n])
    return np.sort(np.concatenate(reserved))


if __name__ == '__main__':
    labels = np.load('bench/external_labels.npy')
    reserved = build(labels)
    fp = pool_fingerprint(labels)
    np.savez('bench/external_eval_holdout.npz',
             indices=reserved, pool_size=len(labels), fingerprint=fp)

    pool_counts = np.bincount(labels, minlength=20)
    res_counts = np.bincount(labels[reserved], minlength=20)
    print(f'pool {len(labels)} images, fingerprint {fp}')
    print(f'reserved {len(reserved)} images for permanent evaluation; '
          f'{len(labels) - len(reserved)} remain available for training\n')
    print(f'{"class":12s} {"pool":>6s} {"reserved":>9s} {"trainable":>10s}')
    for c in np.argsort(pool_counts):
        print(f'{CN[c]:12s} {pool_counts[c]:6d} {res_counts[c]:9d} {pool_counts[c]-res_counts[c]:10d}')
    thin = [CN[c] for c in range(20) if 0 < res_counts[c] < 20]
    if thin:
        print(f'\nNOTE: reserved set is under 20 images for {thin} -- report those per-class')
        print('      numbers with an explicit caveat, they cannot resolve small effects.')
