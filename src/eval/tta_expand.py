"""Expanded TTA on the best single model (cnv2-large pseudo1, 0.97342).

Inference only, no training. The current pipeline uses 3 scales x hflip = 6 views.
This tries more scales and light rotations. Validation is saturated so we cannot
*tune* TTA on it -- we just widen the averaging (monotonically helps up to a point)
and report the validation number as a sanity check, then write the test CSV.
"""
from __future__ import annotations
import time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import timm
import pandas as pd
from pathlib import Path

BENCH = Path('bench')
CN = ['birds','bottles','breads','butterflies','cakes','cats','chickens','cows','dogs','ducks',
      'elephants','fishes','handguns','horses','lions','lipsticks','seals','snakes','spiders','vases']
MEAN = torch.tensor([0.485, 0.456, 0.406])   # (3,) like the notebook
STD = torch.tensor([0.229, 0.224, 0.225])
CKPT = 'timm_convnextv2_large.fcmae_ft_in22k_in1k_pseudo1.pt'
BACKBONE = 'convnextv2_large.fcmae_ft_in22k_in1k'
dev = 'cuda'


class NormalizedModel(nn.Module):
    def __init__(self, backbone, mean, std):
        super().__init__()
        mean_t, std_t = torch.tensor(mean), torch.tensor(std)
        self.register_buffer('scale', (STD / std_t).view(1, 3, 1, 1))
        self.register_buffer('shift', ((MEAN - mean_t) / std_t).view(1, 3, 1, 1))
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x * self.scale + self.shift)


_M = MEAN.view(1, 3, 1, 1)
_S = STD.view(1, 3, 1, 1)


def load_model():
    bb = timm.create_model(BACKBONE, pretrained=False, num_classes=20)
    cfg = bb.default_cfg
    m = NormalizedModel(bb, cfg['mean'], cfg['std'])
    sd = torch.load(CKPT, map_location='cpu')['model_state_dict']
    m.load_state_dict(sd)
    return m.eval().to(dev).to(memory_format=torch.channels_last)


@torch.inference_mode()
def predict(model, images_u8, scales, rotations=(0,), chunk=48):
    """images_u8: (N,3,64,64) uint8. ImageNet-normalise, upsample per scale, hflip + rot TTA."""
    out = []
    t0 = time.time()
    for i in range(0, len(images_u8), chunk):
        x0 = torch.from_numpy(np.ascontiguousarray(images_u8[i:i + chunk])).to(dev).float().div_(255.)
        x0 = (x0 - _M.to(dev)) / _S.to(dev)
        acc = None
        n = 0
        for s in scales:
            xs = F.interpolate(x0, size=(s, s), mode='bicubic', align_corners=False)
            xs = xs.contiguous(memory_format=torch.channels_last)
            for rot in rotations:
                xr = torch.rot90(xs, rot // 90, dims=(2, 3)) if rot in (90, 180, 270) else xs
                if rot not in (0, 90, 180, 270):  # arbitrary small angle
                    theta = torch.tensor([[[np.cos(np.radians(rot)), -np.sin(np.radians(rot)), 0],
                                           [np.sin(np.radians(rot)), np.cos(np.radians(rot)), 0]]],
                                         dtype=torch.float32, device=dev).repeat(xs.size(0), 1, 1)
                    grid = F.affine_grid(theta, xs.size(), align_corners=False)
                    xr = F.grid_sample(xs, grid, align_corners=False, padding_mode='reflection')
                for view in (xr, torch.flip(xr, dims=[3])):
                    with torch.autocast('cuda', dtype=torch.float16):
                        p = torch.softmax(model(view).float(), dim=1)
                    acc = p if acc is None else acc + p
                    n += 1
        out.append((acc / n).float().cpu())
        if i % (chunk * 20) == 0:
            print(f'  {min(i + chunk, len(images_u8))}/{len(images_u8)}  {(i + chunk) / max(time.time() - t0, 1e-6):.0f}/s', flush=True)
    return torch.cat(out).numpy()


def val_acc(p):
    tg = np.load(BENCH / 'targets.npy')
    vi = np.load(BENCH / 'validation_indices.npy')
    return (p.argmax(1) == tg[vi]).mean()


def main():
    train_u8 = np.load(BENCH / 'train_images.npy')
    test_u8 = np.load(BENCH / 'test_images.npy')
    vi = np.load(BENCH / 'validation_indices.npy')
    val_u8 = train_u8[vi]

    model = load_model()
    baseline = np.load('C_cnv2L/timm_convnextv2_large.fcmae_ft_in22k_in1k_pseudo1_run.npz')
    bl_order = baseline['test_key_order'].astype(int)
    bl_test = np.empty_like(baseline['test_probabilities'])
    bl_test[bl_order] = baseline['test_probabilities']
    print(f'baseline (6-view TTA) val {val_acc(baseline["validation_probabilities"]):.4f}')

    configs = {
        '6view_repro':    dict(scales=(160, 192, 224), rotations=(0,)),
        '12view_scales':  dict(scales=(160, 176, 192, 208, 224, 240), rotations=(0,)),
        '10view_wide':    dict(scales=(144, 168, 192, 224, 256), rotations=(0,)),
        '18view_scale_rot': dict(scales=(160, 192, 224), rotations=(0, -8, 8)),
        '14view_hi':      dict(scales=(192, 224, 256, 288), rotations=(0,)),
    }
    results = {}
    for name, cfg in configs.items():
        pv = predict(model, val_u8, **cfg)
        results[name] = (val_acc(pv), cfg)
        print(f'{name:20s} val {val_acc(pv):.4f}   scales={cfg["scales"]} rot={cfg["rotations"]}')

    # pick the widest config that is not worse than the 6-view repro on validation
    repro = results['6view_repro'][0]
    ok = {k: v for k, v in results.items() if v[0] >= repro - 0.002 and k != '6view_repro'}
    best = max(ok, key=lambda k: len(ok[k][1]['scales']) * len(ok[k][1]['rotations'])) if ok else '6view_repro'
    print(f'\nchosen: {best}  (val {results.get(best, (repro,))[0]:.4f})')
    cfg = configs[best]

    pt = predict(model, test_u8, **cfg)
    out = pd.DataFrame({'ID': np.arange(11681), 'Label': [CN[i] for i in pt.argmax(1)]})
    base_csv = pd.read_csv('submission_CL_L_only.csv').sort_values('ID').reset_index(drop=True)
    changed = int((out['Label'].values != base_csv['Label'].values).sum())
    out.to_csv('submission_ttaX_cnv2L.csv', index=False)
    np.save('bench/ttaX_cnv2L_test.npy', pt)
    from collections import Counter
    tbl = Counter((base_csv['Label'].values[i], out['Label'].values[i])
                  for i in np.flatnonzero(out['Label'].values != base_csv['Label'].values))
    print(f'\nsubmission_ttaX_cnv2L.csv  ({best})  vs 0.97342: {changed} rows changed')
    for (a, b), n in tbl.most_common(10):
        print(f'  {a:9s} -> {b:9s}  {n}')


if __name__ == '__main__':
    main()
