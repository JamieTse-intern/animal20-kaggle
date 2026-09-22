"""Independent, non-circular check for MVT1: the external rendition pool has 8,437 real
DomainNet-painting/ImageNet-R images, but training only samples a fixed subset of 2,682 of
them (torch.Generator().manual_seed(1234), independent of SEED, so the SAME subset every
run). The other 5,755 are real renditions never seen by either checkpoint -- evaluate both
the sketch_prob=0.0 and sketch_prob=0.10 base-stage checkpoints on this held-out set to see
whether the sketch-val gain reflects genuine rendition-domain generalization or just learning
the XDoG operator (sketch-val's own synthetic transform is identical to what sketch_prob
trains on, which the notebook's own comments already flag as a circularity risk).
"""
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import timm
from pathlib import Path

CN = ['birds','bottles','breads','butterflies','cakes','cats','chickens','cows','dogs','ducks',
      'elephants','fishes','handguns','horses','lions','lipsticks','seals','snakes','spiders','vases']
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
BACKBONE = 'convnextv2_large.fcmae_ft_in22k_in1k'
dev = 'cuda'


class NormalizedModel(nn.Module):
    def __init__(self, backbone, mean, std):
        super().__init__()
        mean_t, std_t = torch.tensor(mean), torch.tensor(std)
        self.register_buffer('scale', (torch.tensor(STD) / std_t).view(1, 3, 1, 1))
        self.register_buffer('shift', ((torch.tensor(MEAN) - mean_t) / std_t).view(1, 3, 1, 1))
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x * self.scale + self.shift)


def load_model(ckpt_path):
    bb = timm.create_model(BACKBONE, pretrained=False, num_classes=20)
    cfg = bb.default_cfg
    m = NormalizedModel(bb, cfg['mean'], cfg['std'])
    sd = torch.load(ckpt_path, map_location='cpu')['model_state_dict']
    m.load_state_dict(sd)
    return m.eval().to(dev)


mean_t = torch.tensor(MEAN).view(1, 3, 1, 1).to(dev)
std_t = torch.tensor(STD).view(1, 3, 1, 1).to(dev)


@torch.inference_mode()
def predict(model, images_u8, scales=(160, 192, 224), chunk=48):
    out = []
    for i in range(0, len(images_u8), chunk):
        raw = torch.from_numpy(np.ascontiguousarray(images_u8[i:i+chunk])).to(dev).float() / 255.0
        normed = (raw - mean_t) / std_t
        acc = None
        for size in scales:
            resized = F.interpolate(normed, size=(size, size), mode='bicubic', align_corners=False)
            for view in (resized, torch.flip(resized, dims=[3])):
                with torch.amp.autocast('cuda', enabled=True):
                    p = torch.softmax(model(view).float(), dim=1)
                acc = p if acc is None else acc + p
        out.append((acc / 6).cpu())
    return torch.cat(out).numpy()


def main():
    ext_img = np.load('bench/external_images.npy')
    ext_lab = np.load('bench/external_labels.npy')
    real_count = 8047  # len(train_indices) at validation_fraction=0.15, matches the run logs
    ratio = 0.25
    wanted = min(int(round(real_count * ratio / (1.0 - ratio))), len(ext_img))
    picker = torch.Generator().manual_seed(1234)
    keep = torch.randperm(len(ext_img), generator=picker)[:wanted].numpy()
    held_out_mask = np.ones(len(ext_img), dtype=bool)
    held_out_mask[keep] = False
    held_out_img = ext_img[held_out_mask]
    held_out_lab = ext_lab[held_out_mask]
    print(f'external pool {len(ext_img)}, trained-on subset {len(keep)}, '
          f'held-out (never seen in training) {len(held_out_img)}')

    for label, ckpt in [
        ('baseline sketch_prob=0.0', 'checkpoints_backup/timm_convnextv2_large.fcmae_ft_in22k_in1k_base_20260911_204734.pt'),
        ('treatment sketch_prob=0.10', 'timm_convnextv2_large.fcmae_ft_in22k_in1k_base.pt'),
    ]:
        model = load_model(ckpt)
        p = predict(model, held_out_img)
        acc = (p.argmax(1) == held_out_lab).mean()
        print(f'{label:28s}  real-rendition held-out accuracy: {acc:.4f}  (n={len(held_out_lab)})')
        del model
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
