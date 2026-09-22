"""LP-FT vs the baseline DINOv3 ViT-L base -- the same-stage local gate.

Pre-registered decision rule, written before the run:
  GO   : rendition-reserve accuracy improves over the baseline with exact McNemar p < 0.05,
         AND the photo holdout does not get significantly worse (a drop with p < 0.05 vetoes).
  WEAK : the reserve improves but p >= 0.05 -- inconclusive locally; only the leaderboard can
         arbitrate, and the paired test will likely call a small change a tie.
  STOP : the reserve does not improve, or photos drop significantly.

Why the reserve can judge this: a base-vs-base comparison at the same stage is the one regime
where the reserve has been right on every comparison that was a real leaderboard signal.
Caveat: one training run per arm, so training randomness is NOT inside the p-value.

Run:  python compare_lpft.py
"""
import glob
import hashlib
import sys

import numpy as np
import pandas as pd
import torch

import photo_holdout_eval as ev

BASE, NEW = 'dinov3vitl_base', 'dinov3vitl_lpft_base'


def main():
    runs = []
    for f in glob.glob('checkpoints_backup/dinov3vitl_lpft_base_*.pt'):
        ck = torch.load(f, map_location='cpu', mmap=True, weights_only=False)
        runs.append((ck['val_acc'], ck['cfg'].get('lp_epochs'), ck['best_epoch'], f))
        del ck
    real = [r for r in runs if r[0] > 0.9 and r[1] == 2]
    if len(real) != 1:
        sys.exit(f'expected exactly one real LP-FT checkpoint (val > 0.9, lp_epochs 2): {runs}')
    val, lp, best_epoch, path = real[0]
    print(f'LP-FT checkpoint {path}\n  lp_epochs {lp}, best fine-tune epoch {best_epoch}, '
          f'ema val {val:.4f}  (baseline: best epoch 4, val 0.9951)')
    ev.MODELS[NEW] = (path, None, 'base')

    # ---- rendition reserve: the deciding instrument ----------------------------------------
    lab = np.load('bench/reserve_preds.npz')['lab']
    rb = np.load(f'bench/{BASE}_reserve_prob.npy').argmax(1) == lab
    rn = np.load('bench/dinov3vitl_lpft_base_reserve_prob.npy').argmax(1) == lab
    b, c, p_r = ev.mcnemar(rn, rb)
    gain_r = 100 * (rn.mean() - rb.mean())
    print(f'\nrendition reserve (772)  baseline {100 * rb.mean():.2f}%  LP-FT {100 * rn.mean():.2f}%  '
          f'delta {gain_r:+.2f}pt  (LP-FT right/base wrong {b}, reverse {c}, p = {p_r:.4f})')

    # ---- photo holdout: the guardrail ------------------------------------------------------
    z = np.load(ev.HOLDOUT)
    digest = hashlib.sha1()
    digest.update(z['images'].tobytes())
    digest.update(z['labels'].tobytes())
    fp = str(z['fingerprint'])
    assert digest.hexdigest() == fp, 'photo holdout changed since it was built'
    images, plab = torch.from_numpy(z['images']), z['labels']
    pb = ev.photo_probabilities(BASE, images, fp).argmax(1) == plab
    pn = ev.photo_probabilities(NEW, images, fp).argmax(1) == plab
    b2, c2, p_p = ev.mcnemar(pn, pb)
    gain_p = 100 * (pn.mean() - pb.mean())
    print(f'photo holdout ({len(plab)})  baseline {100 * pb.mean():.2f}%  LP-FT {100 * pn.mean():.2f}%  '
          f'delta {gain_p:+.2f}pt  (LP-FT right/base wrong {b2}, reverse {c2}, p = {p_p:.4f})')

    # ---- how far the test predictions moved (for a possible leaderboard check) -------------
    cn = list(np.load('bench/class_names.npy', allow_pickle=True))
    tn = np.load('bench/dinov3vitl_lpft_base_test_prob.npy').argmax(1)
    tb = np.load(f'bench/{BASE}_test_prob.npy').argmax(1)
    pick1 = pd.read_csv('submission_ens_V2PV3Ptemp.csv')['Label'].values
    print(f'\ntest rows changed vs baseline V3T base: {int((tn != tb).sum())}; '
          f'vs pick 1: {int((np.array([cn[i] for i in tn]) != pick1).sum())}')

    # ---- the pre-registered verdict ------------------------------------------------------------
    photo_veto = gain_p < 0 and p_p < 0.05
    if gain_r > 0 and p_r < 0.05 and not photo_veto:
        verdict = 'GO -- clear rendition gain, photos not significantly worse'
    elif photo_veto:
        verdict = 'STOP -- photos got significantly worse'
    elif gain_r > 0:
        verdict = 'WEAK -- rendition gain not significant; only the leaderboard can arbitrate'
    else:
        verdict = 'STOP -- no rendition gain'
    print(f'\nVERDICT: {verdict}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
