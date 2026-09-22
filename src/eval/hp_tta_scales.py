"""Per-scale TTA probabilities for the two ViT-H+ checkpoints on BOTH held-out strata.

Inference only, fixed checkpoints -- the regime where local evaluation is still valid.
Computes each scale once, then scores the pre-registered TTA subsets arithmetically.
"""
import os, sys
os.environ.update(DINO_BACKBONE='vit_huge_plus_patch16_dinov3.lvd1689m', DINO_SIZE='192',
                  DINO_POOL='avg', DINO_GRAD_CKPT='1', DINO_LR='2e-5', DINO_LAYER_DECAY='1.0',
                  DINO_WEIGHTS_FILE='hplus_weights/model.safetensors', DINO_TTA_CHUNK='16')
import numpy as np, torch
from dinov3_base_train import BENCH, build_model, predict_tta
from _heldout_split import load_eval_set

SCALES = (160, 192, 224, 256)
RUNS = [('dinov3vithpps1', 'checkpoints_for_submission/pick1_HP_dinov3_vit_huge_plus_distilled.pt'),
        ('dinov3vithp_base', 'checkpoints_for_submission/pick2_HPbase_dinov3_vit_huge_plus_supervised.pt')]

train_images = torch.from_numpy(np.load(BENCH / 'train_images.npy'))   # eval_tf needs a tensor, not a memmap
targets = np.load(BENCH / 'targets.npy')
val_idx = np.load(BENCH / 'validation_indices.npy')
res_img, res_lab = load_eval_set()
res_img = torch.from_numpy(res_img) if isinstance(res_img, np.ndarray) else res_img

for tag, ckpt in RUNS:
    model, _ = build_model()
    sd = torch.load(ckpt, map_location='cpu', weights_only=False)['model_state_dict']
    model.load_state_dict(sd)
    for name, images, idx in (('val', train_images, val_idx),
                              ('reserve', res_img, np.arange(len(res_lab)))):
        for s in SCALES:
            out = BENCH / f'_tta_{tag}_{name}_{s}.npy'
            if out.exists():
                continue
            np.save(out, predict_tta(model, images, idx, (s,)))
            print(f'  {tag} {name} {s}px done', flush=True)
    del model, sd
    torch.cuda.empty_cache()
print('per-scale probabilities cached')
