"""Does the REAL held-out-rendition accuracy (~85.5%, non-circular metric) move with input
resolution the way validation did? Same degrade-then-upsample method, same clean SEED=2026
checkpoint, but scored against the 5,755 real DomainNet/ImageNet-R images never used in
training -- the actual test-relevant distribution, not synthetic validation photos.
"""
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import timm

CN = ['birds','bottles','breads','butterflies','cakes','cats','chickens','cows','dogs','ducks',
      'elephants','fishes','handguns','horses','lions','lipsticks','seals','snakes','spiders','vases']
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
BACKBONE = 'convnextv2_large.fcmae_ft_in22k_in1k'
CKPT = 'checkpoints_backup/timm_convnextv2_large.fcmae_ft_in22k_in1k_base_20260911_204734.pt'
dev = 'cuda'

CLUSTERS = [{'breads','cakes'}, {'bottles','vases','lipsticks'}, {'chickens','ducks','birds'},
            {'cats','dogs'}, {'cows','elephants','horses'}, {'snakes','spiders','butterflies'}]
def cluster_of(name):
    for c in CLUSTERS:
        if name in c:
            return c
    return None


class NormalizedModel(nn.Module):
    def __init__(self, backbone, mean, std):
        super().__init__()
        mean_t, std_t = torch.tensor(mean), torch.tensor(std)
        self.register_buffer('scale', (torch.tensor(STD) / std_t).view(1, 3, 1, 1))
        self.register_buffer('shift', ((torch.tensor(MEAN) - mean_t) / std_t).view(1, 3, 1, 1))
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x * self.scale + self.shift)


def load_model():
    bb = timm.create_model(BACKBONE, pretrained=False, num_classes=20)
    cfg = bb.default_cfg
    m = NormalizedModel(bb, cfg['mean'], cfg['std'])
    sd = torch.load(CKPT, map_location='cpu')['model_state_dict']
    m.load_state_dict(sd)
    return m.eval().to(dev)


mean_t = torch.tensor(MEAN).view(1, 3, 1, 1).to(dev)
std_t = torch.tensor(STD).view(1, 3, 1, 1).to(dev)


@torch.inference_mode()
def predict(model, images_u8, degrade_to=None, model_in=192, chunk=48):
    out = []
    for i in range(0, len(images_u8), chunk):
        raw = torch.from_numpy(np.ascontiguousarray(images_u8[i:i+chunk])).to(dev).float() / 255.0
        if degrade_to is not None and degrade_to < 64:
            raw = F.interpolate(raw, size=(degrade_to, degrade_to), mode='bicubic', align_corners=False)
            raw = F.interpolate(raw, size=(64, 64), mode='bicubic', align_corners=False).clamp(0, 1)
        normed = (raw - mean_t) / std_t
        resized = F.interpolate(normed, size=(model_in, model_in), mode='bicubic', align_corners=False)
        with torch.amp.autocast('cuda', enabled=True):
            p = torch.softmax(model(resized).float(), dim=1)
        out.append(p.cpu())
    return torch.cat(out).numpy()


def main():
    ext_img = np.load('bench/external_images.npy')
    ext_lab = np.load('bench/external_labels.npy')
    real_count = 8047
    ratio = 0.25
    wanted = min(int(round(real_count * ratio / (1.0 - ratio))), len(ext_img))
    picker = torch.Generator().manual_seed(1234)
    keep = torch.randperm(len(ext_img), generator=picker)[:wanted].numpy()
    held_out_mask = np.ones(len(ext_img), dtype=bool)
    held_out_mask[keep] = False
    held_out_img = ext_img[held_out_mask]
    held_out_lab = ext_lab[held_out_mask]
    in_cluster = np.array([cluster_of(CN[t]) is not None for t in held_out_lab])

    model = load_model()
    print(f'real held-out rendition set: n={len(held_out_lab)}')
    for degrade_to in [None, 48, 40, 32, 24, 16]:
        p = predict(model, held_out_img, degrade_to=degrade_to)
        pred = p.argmax(1)
        acc = (pred == held_out_lab).mean()
        acc_c = (pred[in_cluster] == held_out_lab[in_cluster]).mean()
        acc_o = (pred[~in_cluster] == held_out_lab[~in_cluster]).mean()
        label = 'native 64px' if degrade_to is None else f'degraded to {degrade_to}px'
        print(f'{label:18s}  overall {acc:.4f}   cluster {acc_c:.4f} (n={in_cluster.sum()})'
              f'   other {acc_o:.4f} (n={(~in_cluster).sum()})')


if __name__ == '__main__':
    main()
