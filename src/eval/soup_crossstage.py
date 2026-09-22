"""Cross-stage model soup: average the WEIGHTS of the V3 base and V3 pseudo1 checkpoints.

Pre-registered in CLAUDE.md 2026-09-22. Both were initialised from the same DINOv3 pretrained
checkpoint, which is the soup precondition; the relative weight distance is reported first.
"""
import os, sys
os.environ.update(DINO_BACKBONE='vit_large_patch16_dinov3.lvd1689m', DINO_SIZE='192',
                  DINO_TTA='160,192,224', DINO_POOL='avg', DINO_GRAD_CKPT='1',
                  DINO_LR='2e-5', DINO_LAYER_DECAY='1.0', DINO_TTA_CHUNK='32')
import numpy as np, torch
from dinov3_base_train import BENCH, CN, build_model, predict_tta
from _heldout_split import load_eval_set

A = 'checkpoints_backup/dinov3vitl_base_20260914_164608.pt'
B = 'checkpoints_backup/dinov3vitlps1_20260914_184748.pt'
sa = torch.load(A, map_location='cpu', weights_only=False)['model_state_dict']
sb = torch.load(B, map_location='cpu', weights_only=False)['model_state_dict']
assert sa.keys() == sb.keys()

num = den = 0.0
for k in sa:
    if sa[k].dtype.is_floating_point:
        num += float((sa[k].double() - sb[k].double()).pow(2).sum())
        den += float(sa[k].double().pow(2).sum())
print(f'mean relative weight distance base<->pseudo1: {(num/den)**0.5:.4f}'
      f'   (ConvNeXt-era soup ingredients were 0.017 apart, i.e. one basin)')

soup = {k: ((sa[k].double() + sb[k].double()) / 2).to(sa[k].dtype)
         if sa[k].dtype.is_floating_point else sa[k] for k in sa}

model, _ = build_model()
model.load_state_dict(soup)
lab = np.load(BENCH / 'reserve_preds.npz')['lab']
img, rlab = load_eval_set()
img = torch.from_numpy(img) if isinstance(img, np.ndarray) else img
rp = predict_tta(model, img, np.arange(len(rlab)), (160, 192, 224))
acc = 100 * (rp.argmax(1) == rlab).mean()
print(f'SOUP rendition reserve {acc:.2f}%   (V3 base 92.88%, V3P 94.30%)')
np.save(BENCH / 'v3soup_reserve_prob.npy', rp)
if acc > 94.30:
    test_images = torch.from_numpy(np.load(BENCH / 'test_images.npy'))
    tp = predict_tta(model, test_images, np.arange(len(test_images)), (160, 192, 224))
    np.save(BENCH / 'v3soup_test_prob.npy', tp)
    import pandas as pd
    pd.DataFrame({'ID': np.arange(len(tp)), 'Label': [CN[i] for i in tp.argmax(1)]}).to_csv(
        'submission_v3soup.csv', index=False)
    print('soup beat V3P on the reserve -> submission_v3soup.csv written')
else:
    print('soup did NOT beat V3P on the reserve -> no test inference, no CSV (screen rule)')
