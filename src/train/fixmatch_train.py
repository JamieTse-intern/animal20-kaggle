"""FixMatch on the test set — the one semi-supervised family this project has never used.

Everything so far is offline hard pseudo-labelling: an external teacher produces one argmax
per test image, thresholded once, and the student fits it. There is no consistency
regularisation anywhere in the pipeline. FixMatch replaces that with:

    weak view  -> model's own prediction -> hard label, kept only if confident  (no grad)
    strong view -> trained to match that label                                  (grad)

so the labels update every step and the objective is consistency across augmentations.

Design decisions specific to this project, each with its reason:

* threshold 0.90, not the paper's 0.95. label_smoothing=0.1 over 20 classes caps the
  attainable confidence at 1-0.1+0.1/20 = 0.905, so a 0.95 threshold masks EVERYTHING and
  FixMatch silently degenerates to plain supervised training. Measured on the teacher: 0%
  of test images reach 0.95, 63.7% reach 0.90 -- which lands in the 50-90% mask rate the
  paper reports.
* warm start from the 192px base checkpoint (val 99.44%). Vanilla FixMatch bootstraps from
  a weak model; here the on-the-fly labels are only useful if the model is already good.
* mu=1, labeled batch 10. VRAM-probed: 8.10 GB peak (12/mu=1 is 9.09, too tight alongside
  the EMA copy). The labeled:unlabeled data ratio is ~1:1 anyway (11,496 vs 11,681).
* the weak view runs under no_grad -- it only produces targets, so a FixMatch step costs
  ~2x a normal step, not 3x.

Writes fixmatch_progress.csv every epoch (nbconvert buffers stdout; a long run must be
observable from outside -- this project lost a 221-minute run's curve to that once).
"""
from __future__ import annotations

import csv
import math
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

BENCH = Path('bench')
CN = ['birds', 'bottles', 'breads', 'butterflies', 'cakes', 'cats', 'chickens', 'cows',
      'dogs', 'ducks', 'elephants', 'fishes', 'handguns', 'horses', 'lions', 'lipsticks',
      'seals', 'snakes', 'spiders', 'vases']
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
NUM_CLASSES = 20
DEV = 'cuda'

CFG = dict(
    backbone='convnextv2_large.fcmae_ft_in22k_in1k',
    warm_start='timm_convnextv2_large.fcmae_ft_in22k_in1k_base.pt',
    image_size=192, tta_scales=(160, 192, 224),
    labeled_batch=10, mu=1, epochs=6,
    threshold=0.90,          # see module docstring: 0.95 is unreachable under label smoothing
    lambda_u=1.0,
    backbone_lr=2e-5, head_lr=1e-3, weight_decay=0.05, warmup_epochs=1,
    label_smoothing=0.1, grad_clip=1.0, ema_decay=0.999,
    external_ratio=0.30, real_count=8047, seed=2026,
)


# --------------------------------------------------------------------------- model
class NormalizedModel(nn.Module):
    """Re-normalise an ImageNet-normalised batch into the backbone's own statistics."""

    def __init__(self, backbone, mean, std):
        super().__init__()
        mean_t, std_t = torch.tensor(mean), torch.tensor(std)
        self.register_buffer('scale', (torch.tensor(STD) / std_t).view(1, 3, 1, 1))
        self.register_buffer('shift', ((torch.tensor(MEAN) - mean_t) / std_t).view(1, 3, 1, 1))
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x * self.scale + self.shift)


class ModelEMA:
    def __init__(self, model, decay):
        import copy
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model, step):
        d = min(self.decay, (1.0 + step) / (10.0 + step))
        for e, m in zip(self.module.state_dict().values(), model.state_dict().values()):
            if e.dtype.is_floating_point:
                e.mul_(d).add_(m.detach(), alpha=1.0 - d)
            else:
                e.copy_(m)


def build_model():
    bb = timm.create_model(CFG['backbone'], pretrained=False, num_classes=NUM_CLASSES)
    cfg = bb.default_cfg
    model = NormalizedModel(bb, cfg['mean'], cfg['std'])
    state = torch.load(CFG['warm_start'], map_location='cpu')['model_state_dict']
    model.load_state_dict(state)
    head = list(bb.get_classifier().parameters())
    return model.to(DEV).to(memory_format=torch.channels_last), head


# --------------------------------------------------------------------------- data
strong_tf = v2.Compose([
    v2.RandomHorizontalFlip(),
    v2.TrivialAugmentWide(),
    v2.RandomResizedCrop(64, scale=(0.60, 1.0), ratio=(0.75, 1.333), antialias=True),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(MEAN, STD),
    v2.RandomErasing(p=0.25, value='random'),
])
# FixMatch's weak view is deliberately mild: flip plus a small crop jitter, nothing that
# changes semantics, because its only job is to produce a trustworthy pseudo-label.
weak_tf = v2.Compose([
    v2.RandomHorizontalFlip(),
    v2.RandomResizedCrop(64, scale=(0.90, 1.0), ratio=(0.90, 1.111), antialias=True),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(MEAN, STD),
])
eval_tf = v2.Compose([v2.ToDtype(torch.float32, scale=True), v2.Normalize(MEAN, STD)])


class LabeledSet(Dataset):
    def __init__(self, images, labels, indices, transform):
        self.images, self.labels = images, np.asarray(labels)
        self.indices, self.transform = np.asarray(indices), transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        j = int(self.indices[i])
        return self.transform(self.images[j]), int(self.labels[j])


class TwoViewSet(Dataset):
    """Unlabeled test images, returned as (weak, strong) of the SAME source image."""

    def __init__(self, images):
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = self.images[i]
        return weak_tf(img), strong_tf(img)


class EvalSet(Dataset):
    def __init__(self, images, indices):
        self.images, self.indices = images, np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        j = int(self.indices[i])
        return eval_tf(self.images[j]), j


def floor_balanced_draw(labels, ratio, real_count, seed=1234):
    """Same draw the notebook uses, with the evaluation reserve excluded."""
    reserved = np.load(BENCH / 'external_eval_holdout.npz')['indices']
    available = np.ones(len(labels), dtype=bool)
    available[reserved] = False
    wanted = min(int(round(real_count * ratio / (1.0 - ratio))), int(available.sum()))
    picker = torch.Generator().manual_seed(seed)
    floor = wanted // NUM_CLASSES
    chosen, spare = [], []
    for c in range(NUM_CLASSES):
        pool = np.flatnonzero((labels == c) & available)
        if len(pool) == 0:
            continue
        order = pool[torch.randperm(len(pool), generator=picker).numpy()]
        chosen.append(order[:floor])
        spare.append(order[floor:])
    chosen, spare = np.concatenate(chosen), np.concatenate(spare)
    if len(chosen) < wanted and len(spare):
        extra = spare[torch.randperm(len(spare), generator=picker).numpy()][:wanted - len(chosen)]
        chosen = np.concatenate([chosen, extra])
    return np.sort(chosen)


# --------------------------------------------------------------------------- train
def to_gpu(x, size):
    x = x.to(DEV, torch.float32, non_blocking=True)
    if x.shape[-1] != size:
        x = F.interpolate(x, size=(size, size), mode='bicubic', align_corners=False)
    return x.contiguous(memory_format=torch.channels_last)


def one_hot_smooth(labels, smoothing):
    off = smoothing / NUM_CLASSES
    t = torch.full((labels.size(0), NUM_CLASSES), off, device=labels.device)
    return t.scatter_(1, labels.unsqueeze(1), 1.0 - smoothing + off)


@torch.inference_mode()
def evaluate(model, loader, targets, size):
    model.eval()
    correct = total = 0
    for images, idx in loader:
        x = to_gpu(images, size)
        with torch.amp.autocast('cuda'):
            pred = model(x).argmax(1).cpu().numpy()
        correct += int((pred == targets[idx.numpy()]).sum())
        total += len(pred)
    return correct / max(total, 1)


@torch.inference_mode()
def predict_tta(model, images, indices, size_list, chunk=32):
    model.eval()
    out = []
    for i in range(0, len(indices), chunk):
        batch = torch.stack([eval_tf(images[int(j)]) for j in indices[i:i + chunk]])
        acc = None
        for s in size_list:
            x = to_gpu(batch, s)
            for view in (x, torch.flip(x, dims=[3])):
                with torch.amp.autocast('cuda'):
                    p = torch.softmax(model(view).float(), 1)
                acc = p if acc is None else acc + p
        out.append((acc / (2 * len(size_list))).cpu())
    return torch.cat(out).numpy()


def main():
    torch.manual_seed(CFG['seed'])
    np.random.seed(CFG['seed'])
    train_images = torch.from_numpy(np.load(BENCH / 'train_images.npy'))
    test_images = torch.from_numpy(np.load(BENCH / 'test_images.npy'))
    targets = np.load(BENCH / 'targets.npy')
    val_idx = np.load(BENCH / 'validation_indices.npy')
    train_idx = np.setdiff1d(np.arange(len(targets)), val_idx)
    ext_images = torch.from_numpy(np.load(BENCH / 'external_images.npy'))
    ext_labels = np.load(BENCH / 'external_labels.npy')
    keep = floor_balanced_draw(ext_labels, CFG['external_ratio'], CFG['real_count'])

    labeled = torch.utils.data.ConcatDataset([
        LabeledSet(train_images, targets, train_idx, strong_tf),
        LabeledSet(ext_images, ext_labels, keep, strong_tf),
    ])
    unlabeled = TwoViewSet(test_images)
    print(f'labeled {len(labeled):,} (= {len(train_idx):,} real + {len(keep):,} external)  '
          f'unlabeled {len(unlabeled):,}')

    B, mu = CFG['labeled_batch'], CFG['mu']
    l_loader = DataLoader(labeled, batch_size=B, shuffle=True, num_workers=4,
                          drop_last=True, pin_memory=True, persistent_workers=True)
    u_loader = DataLoader(unlabeled, batch_size=B * mu, shuffle=True, num_workers=4,
                          drop_last=True, pin_memory=True, persistent_workers=True)
    v_loader = DataLoader(EvalSet(train_images, val_idx), batch_size=64, num_workers=2)

    model, head = build_model()
    head_ids = {id(p) for p in head}
    groups = [
        {'params': [p for n, p in model.named_parameters()
                    if id(p) not in head_ids and p.ndim > 1],
         'lr': CFG['backbone_lr'], 'weight_decay': CFG['weight_decay']},
        {'params': [p for n, p in model.named_parameters()
                    if id(p) not in head_ids and p.ndim <= 1],
         'lr': CFG['backbone_lr'], 'weight_decay': 0.0},
        {'params': [p for p in head if p.ndim > 1],
         'lr': CFG['head_lr'], 'weight_decay': CFG['weight_decay']},
        {'params': [p for p in head if p.ndim <= 1],
         'lr': CFG['head_lr'], 'weight_decay': 0.0},
    ]
    opt = torch.optim.AdamW([g for g in groups if g['params']])
    steps_per_epoch = min(len(l_loader), len(u_loader))
    # FIXMATCH_MAX_STEPS caps steps/epoch for smoke tests -- a multi-hour run is a bad place
    # to discover the loop does not run or the confidence mask is empty.
    import os
    if os.environ.get('FIXMATCH_MAX_STEPS'):
        steps_per_epoch = min(steps_per_epoch, int(os.environ['FIXMATCH_MAX_STEPS']))
        print(f'SMOKE TEST: capped at {steps_per_epoch} steps/epoch')
    total_steps = steps_per_epoch * CFG['epochs']
    warmup = max(1, int(CFG['warmup_epochs'] * steps_per_epoch))

    def lr_factor(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    scaler = torch.amp.GradScaler('cuda')
    ema = ModelEMA(model, CFG['ema_decay'])
    size = CFG['image_size']

    log = Path('fixmatch_progress.csv')
    with open(log, 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(['wall', 'epoch', 'sup_loss', 'unsup_loss',
                                'mask_rate', 'pseudo_agree', 'val_acc', 'seconds'])
    best = (-1.0, None)
    for epoch in range(1, CFG['epochs'] + 1):
        model.train()
        started = time.time()
        u_iter = iter(u_loader)
        sup_sum = uns_sum = mask_sum = agree_sum = 0.0
        for step, (xl, yl) in enumerate(l_loader):
            if step >= steps_per_epoch:
                break
            try:
                xw, xs = next(u_iter)
            except StopIteration:
                u_iter = iter(u_loader)
                xw, xs = next(u_iter)
            xl = to_gpu(xl, size)
            yl = yl.to(DEV, torch.long, non_blocking=True)

            # weak view: targets only, so no graph is built and memory stays ~2x not 3x
            with torch.no_grad(), torch.amp.autocast('cuda'):
                probs = torch.softmax(ema.module(to_gpu(xw, size)).float(), 1)
            conf, pseudo = probs.max(1)
            mask = (conf >= CFG['threshold']).float()

            with torch.amp.autocast('cuda'):
                logits_l = model(xl)
                logits_s = model(to_gpu(xs, size))
                sup = -(one_hot_smooth(yl, CFG['label_smoothing'])
                        * torch.log_softmax(logits_l.float(), 1)).sum(1).mean()
                # unsupervised term: hard pseudo-label, masked by confidence. No smoothing --
                # the target is already a guess, softening it further removes what signal
                # the confidence gate just selected for.
                uns = (F.cross_entropy(logits_s.float(), pseudo, reduction='none')
                       * mask).mean()
                loss = sup + CFG['lambda_u'] * uns

            scaler.scale(loss).backward()
            if CFG['grad_clip']:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), CFG['grad_clip'])
            before = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            if scaler.get_scale() >= before:
                sched.step()
            ema.update(model, epoch * steps_per_epoch + step)

            sup_sum += float(sup)
            uns_sum += float(uns)
            mask_sum += float(mask.mean())
            agree_sum += float(((logits_s.argmax(1) == pseudo).float() * mask).sum()
                               / mask.sum().clamp(min=1))

        n = max(step, 1)
        val = evaluate(ema.module, v_loader, targets, size)
        row = [time.strftime('%H:%M:%S'), epoch, round(sup_sum / n, 4), round(uns_sum / n, 4),
               round(mask_sum / n, 4), round(agree_sum / n, 4), round(val, 4),
               round(time.time() - started)]
        with open(log, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(row)
        print(f'  epoch {epoch}/{CFG["epochs"]}  sup {row[2]}  unsup {row[3]}  '
              f'mask {row[4]:.1%}  val {val:.4f}  {row[7]}s', flush=True)
        if val > best[0]:
            best = (val, {k: v.detach().cpu().clone()
                          for k, v in ema.module.state_dict().items()})

    ema.module.load_state_dict(best[1])
    stamp = time.strftime('%Y%m%d_%H%M%S')
    torch.save({'model_state_dict': best[1], 'class_names': CN, 'backbone': CFG['backbone'],
                'cfg': CFG}, f'checkpoints_backup/fixmatch_{stamp}.pt')
    val_prob = predict_tta(ema.module, train_images, val_idx, CFG['tta_scales'])
    test_prob = predict_tta(ema.module, test_images, np.arange(len(test_images)),
                            CFG['tta_scales'])
    np.save('bench/fixmatch_test_prob.npy', test_prob)
    acc = float((val_prob.argmax(1) == targets[val_idx]).mean())
    print(f'best val {best[0]:.4f} | TTA val {acc:.4f} | saved bench/fixmatch_test_prob.npy')
    import pandas as pd
    pd.DataFrame({'ID': np.arange(len(test_prob)),
                  'Label': [CN[i] for i in test_prob.argmax(1)]}).to_csv(
        'submission_fixmatch.csv', index=False)
    print('submission_fixmatch.csv written')


if __name__ == '__main__':
    main()
