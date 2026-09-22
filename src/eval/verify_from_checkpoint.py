"""Re-run inference from a saved checkpoint and compare with the cached probability array.

The strongest available check that a submission's probabilities are a model forward pass and
were never edited: rebuild the model, load the weights that were saved during that run, run the
same 6-view TTA on a sample of test images, and compare against bench/<tag>_test_prob.npy.

    python verify_from_checkpoint.py eva02lps1 <checkpoint.pt> <backbone> <size> <tta> [n]
"""
import os
import sys

tag, ckpt_path, backbone, size, tta = sys.argv[1:6]
n_sample = int(sys.argv[6]) if len(sys.argv) > 6 else 300

os.environ.update(DINO_BACKBONE=backbone, DINO_SIZE=size, DINO_TTA=tta,
                  DINO_GRAD_CKPT='1', DINO_BATCH='20', DINO_LR='2e-5', DINO_LAYER_DECAY='1.0')
if 'dinov' in backbone:
    os.environ['DINO_POOL'] = 'avg'

import numpy as np
import torch

from dinov3_base_train import BENCH, CFG, build_model, predict_tta

ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
print(f'checkpoint      {ckpt_path}')
print(f'  backbone      {ckpt["backbone"]}')
print(f'  best epoch    {ckpt["best_epoch"]}   recorded val {ckpt["val_acc"]:.4f}')

model, _ = build_model()
missing, unexpected = model.load_state_dict(ckpt['model_state_dict'], strict=False)
assert not [k for k in missing if 'scale' not in k and 'shift' not in k], missing
assert not unexpected, unexpected

test_images = torch.from_numpy(np.load(BENCH / 'test_images.npy'))
rng = np.random.default_rng(2026)
idx = np.sort(rng.choice(len(test_images), n_sample, replace=False))
fresh = predict_tta(model, test_images, idx, CFG['tta_scales'])

cached = np.load(BENCH / f'{tag}_test_prob.npy')[idx]
same_argmax = int((fresh.argmax(1) == cached.argmax(1)).sum())
print(f'  re-ran TTA on  {n_sample} random test images at scales {CFG["tta_scales"]}')
print(f'  argmax agrees  {same_argmax}/{n_sample}')
print(f'  max |delta p|  {np.abs(fresh - cached).max():.2e}   mean {np.abs(fresh - cached).mean():.2e}')
print('  VERDICT:', 'the cached probabilities ARE this checkpoint\'s forward pass'
      if same_argmax == n_sample else 'MISMATCH — investigate')
