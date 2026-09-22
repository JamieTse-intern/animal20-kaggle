"""Recompute ViT-H+ test probabilities under TTA set B (192,224,256)."""
import os
os.environ.update(DINO_BACKBONE='vit_huge_plus_patch16_dinov3.lvd1689m', DINO_SIZE='192',
                  DINO_POOL='avg', DINO_GRAD_CKPT='1', DINO_LR='2e-5', DINO_LAYER_DECAY='1.0',
                  DINO_WEIGHTS_FILE='hplus_weights/model.safetensors', DINO_TTA_CHUNK='16')
import numpy as np, torch
from dinov3_base_train import BENCH, build_model, predict_tta
SCALES = (192, 224, 256)
RUNS = [('dinov3vithpps1', 'checkpoints_for_submission/pick1_HP_dinov3_vit_huge_plus_distilled.pt'),
        ('dinov3vithp_base', 'checkpoints_for_submission/pick2_HPbase_dinov3_vit_huge_plus_supervised.pt')]
test_images = torch.from_numpy(np.load(BENCH / 'test_images.npy'))
idx = np.arange(len(test_images))
for tag, ckpt in RUNS:
    out = BENCH / f'{tag}_tta256_test_prob.npy'
    if out.exists():
        print(f'{tag} already done'); continue
    model, _ = build_model()
    model.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=False)['model_state_dict'])
    np.save(out, predict_tta(model, test_images, idx, SCALES))
    print(f'{tag}: {out} written', flush=True)
    del model; torch.cuda.empty_cache()
print('test probabilities under TTA set B cached')
