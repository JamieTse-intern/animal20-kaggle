"""H1 test: is validation accuracy bottlenecked by genuine information loss from the 64px
source resolution, concentrated in the semantic clusters the model's own softmax shows it
confusing (breads/cakes, bottles/vases/lipsticks, chickens/ducks/birds, cats/dogs,
cows/elephants/horses, snakes/spiders/butterflies)? Simulate a lower effective resolution by
downsampling validation images further before the model's normal upsample-to-192 pipeline,
and see whether accuracy degrades faster inside those clusters than outside them.
Labeled validation data only -- no test-set image is inspected or labeled.
"""
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import timm
from pathlib import Path

BENCH = Path('bench')
CN = ['birds','bottles','breads','butterflies','cakes','cats','chickens','cows','dogs','ducks',
      'elephants','fishes','handguns','horses','lions','lipsticks','seals','snakes','spiders','vases']
MEAN = torch.tensor([0.485, 0.456, 0.406])
STD = torch.tensor([0.229, 0.224, 0.225])
CKPT = 'checkpoints_backup/timm_convnextv2_large.fcmae_ft_in22k_in1k_base_20260911_204734.pt'
BACKBONE = 'convnextv2_large.fcmae_ft_in22k_in1k'
dev = 'cuda'

CLUSTERS = [
    {'breads', 'cakes'},
    {'bottles', 'vases', 'lipsticks'},
    {'chickens', 'ducks', 'birds'},
    {'cats', 'dogs'},
    {'cows', 'elephants', 'horses'},
    {'snakes', 'spiders', 'butterflies'},
]
def cluster_of(name):
    for c in CLUSTERS:
        if name in c:
            return c
    return None


class NormalizedModel(nn.Module):
    def __init__(self, backbone, mean, std):
        super().__init__()
        mean_t, std_t = torch.tensor(mean), torch.tensor(std)
        self.register_buffer('scale', (STD / std_t).view(1, 3, 1, 1))
        self.register_buffer('shift', ((MEAN - mean_t) / std_t).view(1, 3, 1, 1))
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x * self.scale + self.shift)


_M = MEAN.view(1, 3, 1, 1).to(dev)
_S = STD.view(1, 3, 1, 1).to(dev)


def load_model():
    bb = timm.create_model(BACKBONE, pretrained=False, num_classes=20)
    cfg = bb.default_cfg
    m = NormalizedModel(bb, cfg['mean'], cfg['std'])
    sd = torch.load(CKPT, map_location='cpu')['model_state_dict']
    m.load_state_dict(sd)
    return m.eval().to(dev).to(memory_format=torch.channels_last)


@torch.inference_mode()
def predict(model, images_u8, degrade_to=None, model_in=192, chunk=64):
    """images_u8: (N,3,64,64) uint8. Optionally downsample to degrade_to then back to 64
    (simulating a lower-information source) before the normal upsample to model_in."""
    out = []
    for i in range(0, len(images_u8), chunk):
        x0 = torch.from_numpy(np.ascontiguousarray(images_u8[i:i+chunk])).to(dev).float().div_(255.)
        if degrade_to is not None and degrade_to < 64:
            x0 = F.interpolate(x0, size=(degrade_to, degrade_to), mode='bicubic', align_corners=False)
            x0 = F.interpolate(x0, size=(64, 64), mode='bicubic', align_corners=False)
            x0 = x0.clamp(0, 1)
        x0 = (x0 - _M) / _S
        xs = F.interpolate(x0, size=(model_in, model_in), mode='bicubic', align_corners=False)
        xs = xs.contiguous(memory_format=torch.channels_last)
        with torch.autocast('cuda', dtype=torch.float16):
            p = torch.softmax(model(xs).float(), dim=1)
        out.append(p.cpu())
    return torch.cat(out).numpy()


def main():
    train_u8 = np.load(BENCH / 'train_images.npy')
    vi = np.load(BENCH / 'validation_indices.npy')
    targets = np.load(BENCH / 'targets.npy')
    val_u8 = train_u8[vi]
    y = targets[vi]

    model = load_model()

    for degrade_to in [None, 48, 40, 32, 24, 16]:
        p = predict(model, val_u8, degrade_to=degrade_to)
        pred = p.argmax(1)
        acc = (pred == y).mean()
        # accuracy restricted to images whose TRUE class belongs to one of the confusion
        # clusters identified from the test set's own softmax, vs. everything else
        in_cluster = np.array([cluster_of(CN[t]) is not None for t in y])
        acc_cluster = (pred[in_cluster] == y[in_cluster]).mean()
        acc_other = (pred[~in_cluster] == y[~in_cluster]).mean()
        label = 'native 64px' if degrade_to is None else f'degraded to {degrade_to}px'
        print(f'{label:18s}  overall {acc:.4f}   cluster-classes {acc_cluster:.4f} (n={in_cluster.sum()})'
              f'   other-classes {acc_other:.4f} (n={(~in_cluster).sum()})')


if __name__ == '__main__':
    main()
