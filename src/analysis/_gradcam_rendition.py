"""Grad-CAM on real held-out RENDITION images (not validation photos), one per confusion
cluster, on the clean SEED=2026 checkpoint: does attention stay on the subject, or does it
drift to background/stylistic texture once the image is a painting/illustration rather than
a photo?
"""
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import timm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CN = ['birds','bottles','breads','butterflies','cakes','cats','chickens','cows','dogs','ducks',
      'elephants','fishes','handguns','horses','lions','lipsticks','seals','snakes','spiders','vases']
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
BACKBONE = 'convnextv2_large.fcmae_ft_in22k_in1k'
CKPT = 'checkpoints_backup/timm_convnextv2_large.fcmae_ft_in22k_in1k_base_20260911_204734.pt'
dev = 'cuda'


class NormalizedModel(nn.Module):
    def __init__(self, backbone, mean, std):
        super().__init__()
        mean_t, std_t = torch.tensor(mean), torch.tensor(std)
        self.register_buffer('scale', (torch.tensor(STD) / std_t).view(1, 3, 1, 1))
        self.register_buffer('shift', ((torch.tensor(MEAN) - mean_t) / std_t).view(1, 3, 1, 1))
        self.backbone = backbone

    def forward_features(self, x):
        return self.backbone.forward_features(x * self.scale + self.shift)

    def forward_head(self, feat):
        return self.backbone.forward_head(feat)


mean_t = torch.tensor(MEAN).view(1, 3, 1, 1).to(dev)
std_t = torch.tensor(STD).view(1, 3, 1, 1).to(dev)


def load_model():
    bb = timm.create_model(BACKBONE, pretrained=False, num_classes=20)
    cfg = bb.default_cfg
    m = NormalizedModel(bb, cfg['mean'], cfg['std'])
    sd = torch.load(CKPT, map_location='cpu')['model_state_dict']
    m.load_state_dict(sd)
    return m.eval().to(dev)


def gradcam(model, img_u8_64, target_class, model_in=192):
    raw = torch.from_numpy(np.ascontiguousarray(img_u8_64[None])).to(dev).float() / 255.0
    normed = (raw - mean_t) / std_t
    x = F.interpolate(normed, size=(model_in, model_in), mode='bicubic', align_corners=False)
    feat = model.forward_features(x)
    feat.retain_grad()
    out = model.forward_head(feat)
    score = out[0, target_class]
    model.backbone.zero_grad(set_to_none=True)
    score.backward()
    grads = feat.grad[0]
    weights = grads.mean(dim=(1, 2))
    cam = F.relu((weights[:, None, None] * feat[0].detach()).sum(0))
    cam = cam / (cam.max() + 1e-8)
    cam = F.interpolate(cam[None, None], size=(64, 64), mode='bilinear', align_corners=False)[0, 0]
    return cam.detach().cpu().numpy()


def main():
    ext_img = np.load('bench/external_images.npy')
    ext_lab = np.load('bench/external_labels.npy')
    real_count, ratio = 8047, 0.25
    wanted = min(int(round(real_count * ratio / (1.0 - ratio))), len(ext_img))
    picker = torch.Generator().manual_seed(1234)
    keep = torch.randperm(len(ext_img), generator=picker)[:wanted].numpy()
    held_out_mask = np.ones(len(ext_img), dtype=bool)
    held_out_mask[keep] = False
    ho_img, ho_lab = ext_img[held_out_mask], ext_lab[held_out_mask]

    model = load_model()
    rng = np.random.RandomState(1)
    classes_to_check = ['ducks', 'chickens', 'cows', 'elephants', 'cats', 'dogs',
                         'bottles', 'breads', 'snakes']
    picks = []
    for cname in classes_to_check:
        cidx = CN.index(cname)
        cands = np.flatnonzero(ho_lab == cidx)
        picks.append((cname, rng.choice(cands)))

    ncols = 3
    nrows = int(np.ceil(len(picks) / ncols))
    fig, axes = plt.subplots(nrows, ncols * 2, figsize=(ncols * 2 * 2.2, nrows * 2.6))
    for i, (cname, idx) in enumerate(picks):
        r, c = divmod(i, ncols)
        img = np.transpose(ho_img[idx], (1, 2, 0))
        cidx = CN.index(cname)
        cam = gradcam(model, ho_img[idx], cidx)
        with torch.inference_mode():
            raw = torch.from_numpy(ho_img[idx][None]).to(dev).float() / 255.0
            normed = (raw - mean_t) / std_t
            x = F.interpolate(normed, size=(192, 192), mode='bicubic', align_corners=False)
            with torch.amp.autocast('cuda', enabled=True):
                p = torch.softmax(model.forward_head(model.forward_features(x)).float(), 1)[0]
        pred_top = CN[p.argmax().item()]
        correct = 'OK' if pred_top == cname else f'-> {pred_top}'
        axes[r, c*2].imshow(img); axes[r, c*2].set_title(f'{cname} [{correct}]', fontsize=9); axes[r, c*2].axis('off')
        axes[r, c*2+1].imshow(img); axes[r, c*2+1].imshow(cam, cmap='jet', alpha=0.5)
        axes[r, c*2+1].set_title('Grad-CAM', fontsize=9); axes[r, c*2+1].axis('off')
    for i in range(len(picks), nrows*ncols):
        r, c = divmod(i, ncols)
        axes[r, c*2].axis('off'); axes[r, c*2+1].axis('off')
    plt.tight_layout()
    plt.savefig('gradcam_rendition.png', dpi=120)
    print('wrote gradcam_rendition.png')


if __name__ == '__main__':
    main()
