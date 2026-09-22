"""H2 test: does the model attend to the animal/object itself, or partly to background
context (shortcut learning)? Grad-CAM on labeled VALIDATION images only -- ground truth is
already known from the training split, so this never touches or infers a test-set label.
"""
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import timm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

BENCH = Path('bench')
CN = ['birds','bottles','breads','butterflies','cakes','cats','chickens','cows','dogs','ducks',
      'elephants','fishes','handguns','horses','lions','lipsticks','seals','snakes','spiders','vases']
MEAN = torch.tensor([0.485, 0.456, 0.406])
STD = torch.tensor([0.229, 0.224, 0.225])
CKPT = 'checkpoints_backup/timm_convnextv2_large.fcmae_ft_in22k_in1k_base_20260911_204734.pt'
BACKBONE = 'convnextv2_large.fcmae_ft_in22k_in1k'
dev = 'cuda'


class NormalizedModel(nn.Module):
    def __init__(self, backbone, mean, std):
        super().__init__()
        mean_t, std_t = torch.tensor(mean), torch.tensor(std)
        self.register_buffer('scale', (STD / std_t).view(1, 3, 1, 1))
        self.register_buffer('shift', ((MEAN - mean_t) / std_t).view(1, 3, 1, 1))
        self.backbone = backbone

    def forward_features(self, x):
        return self.backbone.forward_features(x * self.scale + self.shift)

    def forward_head(self, feat):
        return self.backbone.forward_head(feat)


_M = MEAN.view(1, 3, 1, 1).to(dev)
_S = STD.view(1, 3, 1, 1).to(dev)


def load_model():
    bb = timm.create_model(BACKBONE, pretrained=False, num_classes=20)
    cfg = bb.default_cfg
    m = NormalizedModel(bb, cfg['mean'], cfg['std'])
    sd = torch.load(CKPT, map_location='cpu')['model_state_dict']
    m.load_state_dict(sd)
    return m.eval().to(dev)


def gradcam(model, img_u8_64, target_class, model_in=192):
    x0 = torch.from_numpy(np.ascontiguousarray(img_u8_64[None])).to(dev).float().div_(255.)
    x0 = (x0 - _M) / _S
    x = F.interpolate(x0, size=(model_in, model_in), mode='bicubic', align_corners=False)
    x.requires_grad_(False)
    feat = model.forward_features(x)          # (1, C, h, w)
    feat.retain_grad()
    out = model.forward_head(feat)
    score = out[0, target_class]
    model.backbone.zero_grad(set_to_none=True)
    score.backward()
    grads = feat.grad[0]                       # (C, h, w)
    weights = grads.mean(dim=(1, 2))            # (C,)
    cam = F.relu((weights[:, None, None] * feat[0].detach()).sum(0))
    cam = cam / (cam.max() + 1e-8)
    cam = F.interpolate(cam[None, None], size=(64, 64), mode='bilinear', align_corners=False)[0, 0]
    return cam.detach().cpu().numpy()


def main():
    train_u8 = np.load(BENCH / 'train_images.npy')
    vi = np.load(BENCH / 'validation_indices.npy')
    targets = np.load(BENCH / 'targets.npy')
    val_u8 = train_u8[vi]
    y = targets[vi]

    model = load_model()

    # sample a few images per confusion-cluster class (correct predictions -- ground truth
    # known -- to see whether the model's OWN attention lands on the animal or drifts to
    # background/context even when it gets the answer right)
    rng = np.random.RandomState(0)
    classes_to_check = ['ducks', 'chickens', 'birds', 'cats', 'dogs', 'bottles', 'vases',
                         'breads', 'cakes']
    picks = []
    for cname in classes_to_check:
        cidx = CN.index(cname)
        candidates = np.flatnonzero(y == cidx)
        picks.append(rng.choice(candidates))

    ncols = 3
    nrows = int(np.ceil(len(picks) / ncols))
    fig, axes = plt.subplots(nrows, ncols * 2, figsize=(ncols * 2 * 2.2, nrows * 2.6))
    for i, idx in enumerate(picks):
        r, c = divmod(i, ncols)
        img = np.transpose(val_u8[idx], (1, 2, 0))
        cam = gradcam(model, val_u8[idx], int(y[idx]))
        ax_img = axes[r, c * 2]
        ax_cam = axes[r, c * 2 + 1]
        ax_img.imshow(img); ax_img.set_title(f'{CN[y[idx]]} (#{idx})', fontsize=9); ax_img.axis('off')
        ax_cam.imshow(img); ax_cam.imshow(cam, cmap='jet', alpha=0.5); ax_cam.set_title('Grad-CAM', fontsize=9); ax_cam.axis('off')
    for i in range(len(picks), nrows * ncols):
        r, c = divmod(i, ncols)
        axes[r, c * 2].axis('off'); axes[r, c * 2 + 1].axis('off')
    plt.tight_layout()
    plt.savefig('gradcam_validation.png', dpi=120)
    print('wrote gradcam_validation.png')


if __name__ == '__main__':
    main()
