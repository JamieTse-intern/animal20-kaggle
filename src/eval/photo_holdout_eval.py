"""Evaluate the four final ViT models, and both picks, on the PHOTO holdout.

Pre-registered before any result was seen:
  * accuracy = argmax over all 20 classes (restricting to the 17 covered classes would
    inflate it -- predicting "chickens" for a duck photo must count as wrong);
  * blends use temperatures fitted on the TEST probabilities, exactly as the submitted
    construction does; nothing is refitted on this holdout;
  * THE CALIBRATION TEST: the leaderboard puts DINOv3 ViT-L base (0.97556) above DINOv2
    ViT-L base (0.97363) while the rendition reserve put it 0.90pt BELOW, implying V3T is
    ~0.47pt better on photos. This holdout must rank V3T base above V2 base on photos. If
    it ranks V2 higher, it has failed its only available calibration check.
  * cross-stage comparisons (base vs pseudo1) are NOT valid on any held-out set in this
    project -- the pseudo stage adapts to test images no proxy contains. Only same-stage
    pairs and blends are read.

Run:  python photo_holdout_eval.py
"""
import hashlib
import math
import sys
import time
from pathlib import Path

import numpy as np
import timm
import torch

from blend_temp import fit_temperature, sharpen
from dinov3_base_train import DEV, NormalizedModel, predict_tta

CLASS_NAMES = ['birds', 'bottles', 'breads', 'butterflies', 'cakes', 'cats', 'chickens', 'cows',
               'dogs', 'ducks', 'elephants', 'fishes', 'handguns', 'horses', 'lions',
               'lipsticks', 'seals', 'snakes', 'spiders', 'vases']
HOLDOUT = Path('photo_holdout/photo_holdout.npz')
MODELS = {  # tag: (checkpoint, public LB when submitted alone, stage)
    'dinov2vitl_base': ('checkpoints_backup/dinov2vitl_base_20260914_091501.pt', 0.97363, 'base'),
    'dinov3vitl_base': ('checkpoints_backup/dinov3vitl_base_20260914_164608.pt', 0.97556, 'base'),
    'dinov2vitlps1':   ('checkpoints_backup/dinov2vitlps1_20260914_111544.pt', 0.97950, 'pseudo1'),
    'dinov3vitlps1':   ('checkpoints_backup/dinov3vitlps1_20260914_184748.pt', None, 'pseudo1'),
}
PICKS = {'pick 1 (V2P+V3P)': ('dinov2vitlps1', 'dinov3vitlps1', 0.98132),
         'pick 2 (V2P+V3T)': ('dinov2vitlps1', 'dinov3vitl_base', 0.98089)}


def norm(p):
    p = np.asarray(p, dtype=np.float64)
    return p / p.sum(1, keepdims=True)


def mcnemar(right_a, right_b):
    """Exact two-sided McNemar: b = A right & B wrong, c = B right & A wrong."""
    b, c = int((right_a & ~right_b).sum()), int((right_b & ~right_a).sum())
    n = b + c
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)
    return b, c, p


def photo_probabilities(tag, images, fingerprint):
    """TTA inference on the holdout, cached against the holdout fingerprint."""
    cache = Path(f'photo_holdout/{tag}_photo_prob.npz')
    if cache.is_file():
        z = np.load(cache)
        if str(z['fingerprint']) == fingerprint:
            return z['prob']
    path, _, _ = MODELS[tag]
    t0 = time.time()
    ck = torch.load(path, map_location='cpu', weights_only=False)
    bb = timm.create_model(ck['backbone'], pretrained=False, num_classes=20,
                           dynamic_img_size=True, global_pool='avg')
    model = NormalizedModel(bb, bb.pretrained_cfg['mean'], bb.pretrained_cfg['std'])
    model.load_state_dict(ck['model_state_dict'], strict=True)
    model = model.to(DEV).to(memory_format=torch.channels_last)
    prob = predict_tta(model, images, np.arange(len(images)), ck['cfg']['tta_scales'])
    np.savez(cache, prob=prob.astype(np.float32), fingerprint=fingerprint)
    print(f'  inferred {tag} on {len(images)} photos ({time.time() - t0:.0f}s)')
    del model, bb, ck
    torch.cuda.empty_cache()
    return prob


def main():
    z = np.load(HOLDOUT)
    images_np, labels = z['images'], z['labels']
    digest = hashlib.sha1()
    digest.update(images_np.tobytes())
    digest.update(labels.tobytes())
    fingerprint = str(z['fingerprint'])
    if digest.hexdigest() != fingerprint:
        sys.exit('photo holdout fingerprint mismatch -- the file changed since it was built')
    covered = sorted(set(labels.tolist()))
    images = torch.from_numpy(images_np)
    print(f'photo holdout: {len(labels)} images, {len(covered)} classes, '
          f'fingerprint {fingerprint[:12]}')

    photo = {t: norm(photo_probabilities(t, images, fingerprint)) for t in MODELS}
    reserve_lab = np.load('bench/reserve_preds.npz')['lab']
    rend = {t: norm(np.load(f'bench/{t}_reserve_prob.npy')) for t in MODELS}
    test = {t: norm(np.load(f'bench/{t}_test_prob.npy')) for t in MODELS}

    def macro(pred):
        return float(np.mean([(pred[labels == c] == c).mean() for c in covered]))

    # ---- 1. every model ----------------------------------------------------------------
    print(f'\n{"model":18s} {"stage":8s} {"photo":>7} {"photo macro":>11} {"renditions":>10} {"public LB":>9}')
    for t, (_, lb, stage) in MODELS.items():
        pred = photo[t].argmax(1)
        print(f'{t:18s} {stage:8s} {100 * (pred == labels).mean():6.2f}% {100 * macro(pred):10.2f}% '
              f'{100 * (rend[t].argmax(1) == reserve_lab).mean():9.2f}% '
              f'{("%.5f" % lb) if lb else "   --":>9}')

    # ---- 2. the pre-registered calibration test --------------------------------------------
    print('\nCALIBRATION TEST (pre-registered): does the photo holdout rank V3T base above V2 base?')
    r3 = photo['dinov3vitl_base'].argmax(1) == labels
    r2 = photo['dinov2vitl_base'].argmax(1) == labels
    b, c, p = mcnemar(r3, r2)
    gap_photo = 100 * (r3.mean() - r2.mean())
    gap_rend = 100 * ((rend['dinov3vitl_base'].argmax(1) == reserve_lab).mean()
                      - (rend['dinov2vitl_base'].argmax(1) == reserve_lab).mean())
    print(f'  photo    V3T - V2 = {gap_photo:+.2f}pt   (V3T right/V2 wrong {b}, reverse {c}, '
          f'exact McNemar p = {p:.3f})')
    print(f'  rendition V3T - V2 = {gap_rend:+.2f}pt   (the reserve alone called the wrong sign)')
    print(f'  80/20 stratified  = {0.8 * gap_photo + 0.2 * gap_rend:+.2f}pt   '
          f'(leaderboard: +0.193pt, i.e. V3T ahead)')
    verdict = 'PASS' if gap_photo > 0 else 'FAIL'
    print(f'  -> {verdict}: the holdout {"agrees" if gap_photo > 0 else "disagrees"} with the '
          f'leaderboard on the photo stratum'
          + ('' if p < 0.05 else ' (direction only -- not significant on its own)'))

    # ---- 3. the other same-stage pair (no leaderboard reference: V3P was never submitted) --
    r3p = photo['dinov3vitlps1'].argmax(1) == labels
    r2p = photo['dinov2vitlps1'].argmax(1) == labels
    b, c, p = mcnemar(r3p, r2p)
    print(f'\npseudo1 pair on photos: V3P - V2P = {100 * (r3p.mean() - r2p.mean()):+.2f}pt '
          f'(V3P right/V2P wrong {b}, reverse {c}, p = {p:.3f})')

    # ---- 4. both picks, with the SUBMITTED temperatures ----------------------------------------
    print(f'\n{"blend":18s} {"photo":>7} {"vs best member":>13} {"renditions":>10} {"vs best":>8} {"public LB":>9}')
    for name, (ref, other, lb) in PICKS.items():
        T = fit_temperature(test[other], test[ref].max(1).mean())
        bp = (photo[ref] + sharpen(photo[other], T)) / 2
        br = (rend[ref] + sharpen(rend[other], T)) / 2
        pa = 100 * (bp.argmax(1) == labels).mean()
        ra = 100 * (br.argmax(1) == reserve_lab).mean()
        best_p = max(100 * (photo[m].argmax(1) == labels).mean() for m in (ref, other))
        best_r = max(100 * (rend[m].argmax(1) == reserve_lab).mean() for m in (ref, other))
        print(f'{name:18s} {pa:6.2f}% {pa - best_p:+12.2f}pt {ra:9.2f}% {ra - best_r:+7.2f}pt {lb:9.5f}')

    # ---- 5. where pick 1 still fails on photos ----------------------------------------------
    T = fit_temperature(test['dinov3vitlps1'], test['dinov2vitlps1'].max(1).mean())
    pred = ((photo['dinov2vitlps1'] + sharpen(photo['dinov3vitlps1'], T)) / 2).argmax(1)
    print('\npick 1 per-class photo accuracy (weakest first)')
    per = []
    for cls in covered:
        m = labels == cls
        wrong = pred[m & (pred != cls)]
        top = np.bincount(wrong, minlength=20).argmax() if len(wrong) else None
        per.append((100 * (pred[m] == cls).mean(), CLASS_NAMES[cls], int(m.sum()),
                    f'{CLASS_NAMES[top]} x{int((wrong == top).sum())}' if top is not None else '-'))
    for acc, name, n, top in sorted(per)[:8]:
        print(f'  {name:12s} n={n:3d}  {acc:5.1f}%   most common error: {top}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
