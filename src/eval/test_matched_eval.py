"""Distribution-matched local evaluation: the instrument this project never had.

Every local metric here has failed in a different way, and all for the same root reason --
none of them matched the test distribution:
  - validation (100% photos): saturated since 09-10; the 0.97342 and 0.97257 models both
    score 98.94%.
  - sketch-val (100% synthetic XDoG): circular whenever the intervention touches the same
    operator (+15pt local, +0.15pt real).
  - reserve renditions (100% renditions): inverts across the base/pseudo boundary AND for
    single-vs-blend, because the test set is only ~20% renditions.

The test set is ~80% photos / ~20% renditions. So estimate each stratum separately on ALL
available held-out data and combine with the test weights, rather than subsampling to a mixed
set (subsampling to 80/20 gives n=1,773 and SE 0.41pt -- worse than the leaderboard;
the weighted estimator gives SE ~0.20pt -- better).

    photos     : 1,419 validation images (never trained on, any run)
    renditions : 5,339 external images -- 772 reserved + 4,567 that the floor-balanced
                 training draw never selected. Both are held out; only the reserve was
                 held out *by design*.

CONTAMINATION WARNING, enforced in code: checkpoints trained before 2026-09-12 used the OLD
external pool and DID see part of these images (seed4242 saw 27.6% of the reserve). Pass
`pre_0912=True` for those models and the rendition stratum is restricted to the subset that
predates nothing -- see clean_rendition_mask().
"""
import hashlib
import numpy as np
import torch

PHOTO_WEIGHT = 0.80          # test set is ~80% photographic, ~20% rendition
RENDITION_WEIGHT = 0.20
CN = ['birds', 'bottles', 'breads', 'butterflies', 'cakes', 'cats', 'chickens', 'cows',
      'dogs', 'ducks', 'elephants', 'fishes', 'handguns', 'horses', 'lions', 'lipsticks',
      'seals', 'snakes', 'spiders', 'vases']


def _md5(a):
    return hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()


def held_out_rendition_indices(ratio=0.30, real_count=8047, seed=1234):
    """External images the training draw never selected: the reserve plus the unused remainder.

    Mirrors the notebook's floor-balanced draw exactly. If that sampling code changes, this
    must change with it -- which is why the assertion below is worth keeping.
    """
    labels = np.load('bench/external_labels.npy')
    reserved = np.load('bench/external_eval_holdout.npz')['indices']
    available = np.ones(len(labels), dtype=bool)
    available[reserved] = False
    wanted = min(int(round(real_count * ratio / (1.0 - ratio))), int(available.sum()))
    picker = torch.Generator().manual_seed(seed)
    floor = wanted // len(CN)
    chosen, spare = [], []
    for class_index in range(len(CN)):
        pool = np.flatnonzero((labels == class_index) & available)
        if len(pool) == 0:
            continue
        order = pool[torch.randperm(len(pool), generator=picker).numpy()]
        chosen.append(order[:floor])
        spare.append(order[floor:])
    chosen = np.concatenate(chosen)
    spare = np.concatenate(spare)
    if len(chosen) < wanted and len(spare):
        extra = spare[torch.randperm(len(spare), generator=picker).numpy()][:wanted - len(chosen)]
        chosen = np.concatenate([chosen, extra])
    trained = np.zeros(len(labels), dtype=bool)
    trained[chosen] = True
    return np.flatnonzero(~trained)


def clean_rendition_mask(indices):
    """True for images no PRE-2026-09-12 checkpoint trained on (old pool, uniform draw)."""
    new_images = np.load('bench/external_images.npy')
    old_images = np.load('bench_pre_audit_backup/external_images.npy')
    wanted = int(round(8047 * 0.25 / 0.75))
    picker = torch.Generator().manual_seed(1234)
    old_train = {_md5(old_images[i])
                 for i in torch.randperm(len(old_images), generator=picker)[:wanted].numpy()}
    return np.array([_md5(new_images[i]) not in old_train for i in indices])


def score(photo_pred, photo_true, rend_pred, rend_true):
    """Weighted accuracy plus its standard error, in points."""
    a_p = float(np.mean(photo_pred == photo_true))
    a_r = float(np.mean(rend_pred == rend_true))
    acc = PHOTO_WEIGHT * a_p + RENDITION_WEIGHT * a_r
    var = (PHOTO_WEIGHT ** 2 * a_p * (1 - a_p) / len(photo_true)
           + RENDITION_WEIGHT ** 2 * a_r * (1 - a_r) / len(rend_true))
    return dict(matched=acc, photo=a_p, rendition=a_r, se_pt=float(np.sqrt(var) * 100),
                n_photo=len(photo_true), n_rend=len(rend_true))


if __name__ == '__main__':
    idx = held_out_rendition_indices()
    clean = clean_rendition_mask(idx)
    labels = np.load('bench/external_labels.npy')
    print(f'held-out renditions: {len(idx)}  '
          f'({int((~np.isin(idx, np.load("bench/external_eval_holdout.npz")["indices"])).sum())} '
          f'unused + {len(idx) - int((~np.isin(idx, np.load("bench/external_eval_holdout.npz")["indices"])).sum())} reserved)')
    print(f'of those, clean for PRE-09-12 checkpoints: {int(clean.sum())} '
          f'({clean.mean():.1%}) -- use this subset when scoring model A or seed4242')
    counts = np.bincount(labels[idx], minlength=20)
    thin = [(CN[c], int(counts[c])) for c in range(20) if counts[c] < 40]
    print(f'per-class held-out renditions: min {counts.min()} max {counts.max()}'
          + (f'; under 40: {thin}' if thin else ''))
    print(f'\nweights: {PHOTO_WEIGHT:.0%} photo / {RENDITION_WEIGHT:.0%} rendition '
          f'(the ~20% rendition share is a project estimate, not a measured figure -- '
          f'if it is wrong the weighting is wrong with it)')
