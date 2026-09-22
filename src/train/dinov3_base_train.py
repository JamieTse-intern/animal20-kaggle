"""Task 2 (NEXT_ROUND_PROMPT §4): a new pretraining prior, base stage only.

Every ensemble member this project has is a ConvNeXt-V2-Large with the FCMAE prior, and the
disagreement rate is locked near 2% by that shared prior. §12 lists exactly one untried lever
with a plausible >=0.5pt ceiling: a genuinely different pretraining objective.

First choice per §4.1: `convnext_large.dinov3_lvd1689m` -- SAME architecture (so the batch
budget and the 192px recipe carry over unchanged) with a completely different prior: DINOv3
self-distillation on LVD-1689M, no labels, no language supervision, versus FCMAE masked
autoencoding then IN-22k/IN-1k supervised fine-tuning.

Gate (§4.2): base stage only, 192px, produce a CSV. Reference is the only base-stage
leaderboard number on record, D = 256px base = 0.96350, so go requires >= 0.96150.
Clean-reserve accuracy is computed alongside as a same-stage direction check only -- §6 says
it cannot gate anything, and a base-vs-base comparison is the one case where it is 3-for-3.

RECIPE PARITY. Everything here mirrors the notebook's base stage so the comparison is against
the prior and nothing else: the same degraded uint8 arrays, RandomJpeg(0.3, 55-95),
TrivialAugmentWide, RandomResizedCrop(64, 0.60-1.0), RandomErasing(0.25), label smoothing 0.1,
MixUp(0.2)/CutMix(1.0) on half the batches with soft targets, floor-balanced external draw at
ratio 0.30 with the evaluation reserve excluded, EMA 0.999, grad clip 1.0, 16 epochs, warmup 2,
cosine, AMP, TTA 160/192/224 + hflip, checkpoint on best validation.

ONE DELIBERATE DEVIATION, stated up front. The project standard is a flat backbone_lr of 2e-5,
chosen because the CLIP/in12k/FCMAE checkpoints "arrive strong and a 2e-4 step damages them".
Those are all SUPERVISED-fine-tuned checkpoints. DINOv3 is a raw self-supervised trunk that has
never seen a label, and 2e-5 flat would very likely underfit it -- closing this line for a
tuning reason rather than a prior reason, which is the expensive error here given §4 calls this
the only bet with a >=0.5pt ceiling. So the trunk uses layer-wise LR decay 0.75 off a 1e-4 top,
which is the standard DINO fine-tune recipe and is also what §4.1 specifies for the DINOv2
fallback. Deepest stages therefore see ~1e-4 and the stem ~2e-5 -- the protection the flat
2e-5 was for, applied where it belongs. One value, no sweep.
"""
import copy
import csv
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision.transforms import v2

BENCH = Path('bench')
CN = ['birds', 'bottles', 'breads', 'butterflies', 'cakes', 'cats', 'chickens', 'cows',
      'dogs', 'ducks', 'elephants', 'fishes', 'handguns', 'horses', 'lions', 'lipsticks',
      'seals', 'snakes', 'spiders', 'vases']
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
NUM_CLASSES = 20
DEV = 'cuda'
TAG = os.environ.get('DINO_TAG', 'dinov3cnl')

# DINO_BEST_TO_DISK keeps the best-epoch snapshot on disk instead of in system RAM. Only needed
# for the 840M ViT-H+, where a snapshot is 3.13 GB and this machine has 8 GB free; default off
# leaves every earlier run's behaviour untouched.
BEST_TO_DISK = bool(int(os.environ.get('DINO_BEST_TO_DISK', 0)))
BEST_PATH = f'_best_{TAG}.pt'
# DINO_FOCAL_GAMMA > 0 switches the loss to its focal form. 0 = the plain soft-target
# cross-entropy every run before 2026-09-22 used.
FOCAL_GAMMA = float(os.environ.get('DINO_FOCAL_GAMMA', 0.0))

CFG = dict(
    backbone=os.environ.get('DINO_BACKBONE', 'convnext_large.dinov3_lvd1689m'),
    # DINO_SIZE / DINO_TTA exist for patch-based trunks: a ViT-L/14 needs every side length
    # divisible by 14, so the ConvNeXt defaults (192, 160/192/224) become 196 and 168/196/224.
    # Defaults are unchanged so the DINOv3 ConvNeXt runs stay reproducible from this file.
    image_size=int(os.environ.get('DINO_SIZE', 192)),
    tta_scales=tuple(int(s) for s in os.environ.get('DINO_TTA', '160,192,224').split(',')),
    batch_size=int(os.environ.get('DINO_BATCH', 20)),
    epochs=int(os.environ.get('DINO_EPOCHS', 16)), warmup_epochs=2,
    # DINO_LR / DINO_LAYER_DECAY exist so the flat-2e-5 control (layer_decay=1.0) is a
    # command-line change, not a code edit -- see the 2026-09-13 note in CLAUDE.md.
    backbone_lr=float(os.environ.get('DINO_LR', 1e-4)),
    layer_decay=float(os.environ.get('DINO_LAYER_DECAY', 0.75)),
    head_lr=1e-3, weight_decay=0.05,
    label_smoothing=0.1, grad_clip=1.0, ema_decay=0.999,
    mix_prob=0.5, mixup_alpha=0.2, cutmix_alpha=1.0, random_erasing=0.25,
    jpeg_aug_prob=0.3, jpeg_aug_quality=(55, 95),
    external_ratio=0.30, real_count=8047, seed=2026,
    # LP-FT: train only the classifier on a frozen trunk for this many epochs before the normal
    # fine-tune, so a random head's large early gradients cannot distort pretrained features.
    # 0 = off, which is exactly the behaviour of every run before 2026-09-15.
    lp_epochs=int(os.environ.get('DINO_LP_EPOCHS', 0)),
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
    """Shadow weights, optionally held in system RAM and updated every few steps.

    Defaults (offload=False, every=1) are byte-for-byte the original behaviour, so every run
    before 2026-09-21 reproduces unchanged. A 840M-parameter trunk needs 3.13 GB for an fp32
    shadow copy, which this 10 GB card does not have alongside parameters, gradients and
    optimiser state -- `offload` moves it to CPU. The GPU->CPU copy then costs 712 ms/step,
    more than the training step itself, so `every` coarsens it; `decay` is raised to the same
    power, which leaves the averaging horizon measured in steps unchanged.
    """

    def __init__(self, model, decay, offload=False, every=1):
        self.module = copy.deepcopy(model).eval()
        self.offload, self.every, self.decay = offload, max(1, every), decay
        if offload:
            self.module.cpu()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model, step):
        if step % self.every:
            return
        d = min(self.decay, (1.0 + step) / (10.0 + step)) ** self.every
        for e, m in zip(self.module.state_dict().values(), model.state_dict().values()):
            if e.dtype.is_floating_point:
                e.mul_(d).add_(m.detach().to(e.device), alpha=1.0 - d)
            else:
                e.copy_(m)

    _backup = None

    def for_eval(self, model=None):
        """The shadow weights, ready to evaluate. Always pair with .park(model).

        When offloaded, this SWAPS them into the live model instead of moving a second copy
        onto the GPU: at 840M parameters that copy is 3.13 GB, and holding it beside the
        parameters and the optimiser state is exactly what pushed this run past 10 GB and into
        the WDDM spill that stalls the GPU at 0% utilisation.
        """
        if not self.offload:
            return self.module
        assert model is not None, 'offloaded EMA needs the live model to swap into'
        self._backup = {k: v.detach().to('cpu', copy=True)
                        for k, v in model.state_dict().items()}
        model.load_state_dict(self.module.state_dict())
        return model

    def park(self, model=None):
        if self.offload and self._backup is not None:
            model.load_state_dict(self._backup)
            self._backup = None


class FusedBackward:
    """One optimiser per parameter, stepped from a post-accumulate-grad hook.

    Each gradient is used and freed the instant it is produced, so the fp32 gradient buffer for
    the whole model never exists at once. On the 1.14B ViT-g that buffer is 4.23 GB -- the
    difference between 6.4 GB resident and 9.7 GB, which is where this card starts spilling to
    system RAM and step time swings 5x (measured 1310 / 2647 / 7327 ms at three batch sizes).

    Two costs, and they are recipe changes, not free:
      * bf16 autocast, no GradScaler. A per-parameter step cannot honour a global inf/nan check,
        because early parameters are already updated before a later inf is seen. bf16 carries
        fp32's exponent range so no loss scaling is needed.
      * per-parameter gradient clipping replaces the global norm. At the same threshold this is
        a weaker constraint (each tensor's norm is far below the norm over all of them).
    """

    def __init__(self, groups, opt_cls, clip):
        self.clip, self.opts, self.base_lr, self.handles = clip, {}, {}, []
        for g in groups:
            for prm in g['params']:
                self.opts[prm] = opt_cls([{k: v for k, v in g.items() if k != 'params'}
                                          | {'params': [prm]}])
                self.base_lr[prm] = g['lr']
                self.handles.append(prm.register_post_accumulate_grad_hook(self._step_one))

    def _step_one(self, prm):
        if self.clip:
            nn.utils.clip_grad_norm_([prm], self.clip)
        self.opts[prm].step()
        self.opts[prm].zero_grad(set_to_none=True)

    def set_lr_factor(self, factor):
        for prm, opt in self.opts.items():
            opt.param_groups[0]['lr'] = self.base_lr[prm] * factor


def build_model():
    kwargs = {}
    # Web-SSL is published only in transformers format; webssl_timm renames it into timm.
    if any(k in CFG['backbone'] for k in ('vit', 'eva', 'deit', 'webssl')):
        kwargs['dynamic_img_size'] = True      # needed for the 160/192/224 TTA on a patch grid
        # §4.1 specifies mean patch-token pooling for the DINOv2 trunk. timm's ViT default is
        # 'token' (the CLS vector); 'avg' averages the patch tokens and drops the prefix
        # tokens (CLS + 4 registers) via num_prefix_tokens, which is what is wanted here.
        if os.environ.get('DINO_POOL'):
            kwargs['global_pool'] = os.environ['DINO_POOL']
    from webssl_timm import create_webssl, is_webssl
    if is_webssl(CFG['backbone']):
        bb = create_webssl(CFG['backbone'], NUM_CLASSES, **kwargs)
    else:
        # DINO_WEIGHTS_FILE loads the checkpoint from a local safetensors file instead of the
        # Hugging Face cache. The 3.36 GB ViT-H+ download stalled twice through the hub client
        # (the documented xet failure mode), so it was fetched with curl -C - instead.
        if os.environ.get('DINO_WEIGHTS_FILE'):
            kwargs['pretrained_cfg_overlay'] = dict(file=os.environ['DINO_WEIGHTS_FILE'])
            print(f"loading weights from {os.environ['DINO_WEIGHTS_FILE']}")
        bb = timm.create_model(CFG['backbone'], pretrained=True, num_classes=NUM_CLASSES, **kwargs)
    if int(os.environ.get('DINO_GRAD_CKPT', 0)):
        bb.set_grad_checkpointing(True)        # ViT-L/14 at 196px does not fit 10GB without it
        print('gradient checkpointing ON')
    cfg = bb.pretrained_cfg
    print(f"backbone {CFG['backbone']}  mean {cfg['mean']}  std {cfg['std']}  "
          f"native {cfg['input_size']}  params {sum(p.numel() for p in bb.parameters())/1e6:.1f}M"
          f"  pool {getattr(bb, 'global_pool', '?')}"
          f"  prefix_tokens {getattr(bb, 'num_prefix_tokens', '-')}")
    model = NormalizedModel(bb, cfg['mean'], cfg['std'])
    return model.to(DEV).to(memory_format=torch.channels_last), bb


def param_groups(model, bb):
    """Layer-wise LR decay over the trunk, flat head_lr on the classifier.

    COARSE grouping deliberately. ConvNeXt's fine matcher yields 38 depths, and 0.75^37 = 3.6e-5
    would put the stem at 2e-9 -- frozen, not decayed. Coarse gives 6 groups (stem, 4 stages,
    head-norm), so 0.75 lands the stem at 2.4e-5 and the last stage at 7.5e-5: the bottom sits
    almost exactly on the project's proven flat 2e-5 while the top gets the 5x it needs to adapt
    a never-supervised trunk. That is the whole point of the deviation, and it stays bounded by
    the value the project already trusts.
    """
    head_ids = {id(p) for p in bb.get_classifier().parameters()}
    matcher = bb.group_matcher(coarse=True)
    layer_ids = timm.models._manipulate.group_parameters(bb, matcher, return_values=False)
    layer_ids = dict(layer_ids) if not isinstance(layer_ids, dict) else layer_ids
    depth_of = {}
    for depth, names in sorted(layer_ids.items()):
        for n in names:
            depth_of[n] = depth
    max_depth = max(depth_of.values()) if depth_of else 0
    buckets = {}
    for name, p in bb.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in head_ids:
            lr = CFG['head_lr']
        else:
            d = depth_of.get(name, max_depth)
            lr = CFG['backbone_lr'] * (CFG['layer_decay'] ** (max_depth - d))
        wd = CFG['weight_decay'] if p.ndim > 1 else 0.0
        buckets.setdefault((round(lr, 10), wd), []).append(p)
    groups = [{'params': ps, 'lr': lr, 'weight_decay': wd} for (lr, wd), ps in buckets.items()]
    lrs = sorted({g['lr'] for g in groups})
    print(f'{len(groups)} param groups, {max_depth + 1} trunk depths, '
          f'lr range {lrs[0]:.2e} .. {lrs[-1]:.2e}')
    return groups


# --------------------------------------------------------------------------- data
class RandomJpeg(nn.Module):
    """Re-encode a uint8 CHW image at a random JPEG quality (notebook parity)."""

    def __init__(self, probability, quality_range):
        super().__init__()
        self.probability, self.quality_range = probability, tuple(quality_range)

    def forward(self, image):
        if self.probability <= 0 or random.random() > self.probability:
            return image
        q = random.randint(*self.quality_range)
        return torchvision.io.decode_jpeg(torchvision.io.encode_jpeg(image, quality=q))


train_tf = v2.Compose([
    RandomJpeg(CFG['jpeg_aug_prob'], CFG['jpeg_aug_quality']),
    v2.RandomHorizontalFlip(),
    v2.TrivialAugmentWide(),
    v2.RandomResizedCrop(64, scale=(0.60, 1.0), ratio=(0.75, 1.333), antialias=True),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(MEAN, STD),
    v2.RandomErasing(p=CFG['random_erasing'], value='random'),
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


class EvalSet(Dataset):
    def __init__(self, images, indices):
        self.images, self.indices = images, np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        j = int(self.indices[i])
        return eval_tf(self.images[j]), j


def floor_balanced_draw(labels, ratio, real_count, seed=1234):
    """The notebook's draw, with the evaluation reserve excluded first (CLAUDE.md rule)."""
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
    chosen = np.sort(chosen)
    assert not np.intersect1d(chosen, reserved).size, 'training draw touched the eval reserve'
    return chosen


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


def random_box(height, width, lam):
    ratio = math.sqrt(max(0.0, 1.0 - lam))
    cut_h, cut_w = int(height * ratio), int(width * ratio)
    cy, cx = random.randrange(height), random.randrange(width)
    y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, height)
    x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, width)
    return y1, y2, x1, x2


def mix_batch(images, targets):
    """MixUp or CutMix on soft targets -- notebook parity (mix_prob 0.5, then 50/50)."""
    if CFG['mix_prob'] <= 0 or random.random() > CFG['mix_prob']:
        return images, targets
    perm = torch.randperm(images.size(0), device=images.device)
    if random.random() < 0.5:
        lam = float(np.random.beta(CFG['mixup_alpha'], CFG['mixup_alpha']))
        images = lam * images + (1.0 - lam) * images[perm]
    else:
        lam = float(np.random.beta(CFG['cutmix_alpha'], CFG['cutmix_alpha']))
        y1, y2, x1, x2 = random_box(images.size(2), images.size(3), lam)
        images = images.clone()
        images[:, :, y1:y2, x1:x2] = images[perm][:, :, y1:y2, x1:x2]
        lam = 1.0 - ((y2 - y1) * (x2 - x1) / (images.size(2) * images.size(3)))
    return images, lam * targets + (1.0 - lam) * targets[perm]


@torch.inference_mode()
def evaluate(model, loader, targets, size):
    model.eval()
    correct = total = 0
    for images, idx in loader:
        with torch.amp.autocast('cuda'):
            pred = model(to_gpu(images, size)).argmax(1).cpu().numpy()
        correct += int((pred == targets[idx.numpy()]).sum())
        total += len(pred)
    return correct / max(total, 1)


@torch.inference_mode()
def predict_tta(model, images, indices, size_list, chunk=None):
    chunk = chunk or int(os.environ.get('DINO_TTA_CHUNK', 32))
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
    random.seed(CFG['seed'])

    train_images = torch.from_numpy(np.load(BENCH / 'train_images.npy'))
    test_images = torch.from_numpy(np.load(BENCH / 'test_images.npy'))
    targets = np.load(BENCH / 'targets.npy')
    val_idx = np.load(BENCH / 'validation_indices.npy')
    train_idx = np.setdiff1d(np.arange(len(targets)), val_idx)
    ext_images = torch.from_numpy(np.load(BENCH / 'external_images.npy'))
    ext_labels = np.load(BENCH / 'external_labels.npy')
    keep = floor_balanced_draw(ext_labels, CFG['external_ratio'], CFG['real_count'])

    labeled = ConcatDataset([
        LabeledSet(train_images, targets, train_idx, train_tf),
        LabeledSet(ext_images, ext_labels, keep, train_tf),
    ])
    print(f'labeled {len(labeled):,} (= {len(train_idx):,} real + {len(keep):,} external)  '
          f'val {len(val_idx):,}')

    loader = DataLoader(labeled, batch_size=CFG['batch_size'], shuffle=True, num_workers=4,
                        drop_last=True, pin_memory=True, persistent_workers=True)
    v_loader = DataLoader(EvalSet(train_images, val_idx),
                          batch_size=int(os.environ.get('DINO_EVAL_BATCH', 64)), num_workers=2)

    model, bb = build_model()
    fused = None
    if int(os.environ.get('DINO_FUSED_BWD', 0)):
        import bitsandbytes as bnb
        groups = param_groups(model, bb)
        fused = FusedBackward(groups, bnb.optim.AdamW8bit, CFG['grad_clip'])
        opt = next(iter(fused.opts.values()))          # only so the LambdaLR below has a target
        print(f'fused backward: {len(fused.opts)} per-parameter AdamW8bit optimisers, '
              f'bf16 autocast, per-parameter clip {CFG["grad_clip"]}')
    elif int(os.environ.get('DINO_OPT8BIT', 0)):
        # 8-bit Adam moments: 2 bytes/parameter instead of 8. On the 840M ViT-H+ that is the
        # difference between 13.0 GB and 8.3 GB of optimiser+parameter state. Master weights
        # stay fp32 -- at lr 2e-5 a bf16 master would round every update away.
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(param_groups(model, bb))
        print('optimiser: bitsandbytes AdamW8bit')
    else:
        opt = torch.optim.AdamW(param_groups(model, bb))
    steps_per_epoch = len(loader)
    if os.environ.get('DINO_MAX_STEPS'):
        steps_per_epoch = min(steps_per_epoch, int(os.environ['DINO_MAX_STEPS']))
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
    ema = ModelEMA(model, CFG['ema_decay'],
                   offload=bool(int(os.environ.get('DINO_EMA_CPU', 0))),
                   every=int(os.environ.get('DINO_EMA_EVERY', 1)))
    if ema.offload or ema.every > 1:
        print(f'EMA: offload={ema.offload} every={ema.every} '
              f'effective decay {CFG["ema_decay"] ** ema.every:.4f}')
    size = CFG['image_size']

    log = Path(f'{TAG}_progress.csv')
    with open(log, 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(['wall', 'epoch', 'loss', 'val_acc', 'ema_val_acc', 'seconds'])
    if CFG['lp_epochs']:
        # Linear probe: only the classifier trains; the trunk stays exactly at its pretrained
        # values. Same data, augmentation, loss and head_lr as the fine-tune that follows.
        head_params = list(bb.get_classifier().parameters())
        for p in model.parameters():
            p.requires_grad_(False)
        for p in head_params:
            p.requires_grad_(True)
        lp_opt = torch.optim.AdamW(
            [{'params': [p for p in head_params if p.ndim > 1], 'weight_decay': CFG['weight_decay']},
             {'params': [p for p in head_params if p.ndim <= 1], 'weight_decay': 0.0}],
            lr=CFG['head_lr'])
        lp_scaler = torch.amp.GradScaler('cuda')
        print(f"LP stage: {CFG['lp_epochs']} epochs, trunk frozen, "
              f"{sum(p.numel() for p in head_params):,} head params at lr {CFG['head_lr']:.0e}")
        for lp_epoch in range(1, CFG['lp_epochs'] + 1):
            model.train()
            started, loss_sum, step = time.time(), 0.0, 0
            for step, (images, labels) in enumerate(loader, 1):
                if step > steps_per_epoch:
                    step -= 1
                    break
                x = to_gpu(images, size)
                y = one_hot_smooth(labels.to(DEV, torch.long, non_blocking=True),
                                   CFG['label_smoothing'])
                x, y = mix_batch(x, y)
                with torch.amp.autocast('cuda'):
                    loss = -(y * torch.log_softmax(model(x).float(), 1)).sum(1).mean()
                lp_scaler.scale(loss).backward()
                if CFG['grad_clip']:
                    lp_scaler.unscale_(lp_opt)
                    nn.utils.clip_grad_norm_(head_params, CFG['grad_clip'])
                lp_scaler.step(lp_opt)
                lp_scaler.update()
                lp_opt.zero_grad(set_to_none=True)
                loss_sum += float(loss.detach())
            acc = evaluate(model, v_loader, targets, size)
            row = [time.strftime('%H:%M:%S'), f'LP{lp_epoch}', round(loss_sum / max(step, 1), 4),
                   round(acc, 4), '', round(time.time() - started)]
            with open(log, 'a', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow(row)
            print(f"  LP epoch {lp_epoch}/{CFG['lp_epochs']}  loss {row[2]}  val {acc:.4f}  "
                  f"{row[5]}s", flush=True)
        for p in model.parameters():
            p.requires_grad_(True)
        # The fine-tune must differ from the baseline in ONE thing only -- where the head starts.
        # So the EMA restarts from the probed model, as the baseline's starts from the raw one.
        ema = ModelEMA(model, CFG['ema_decay'])
        del lp_opt, lp_scaler

    best = (-1.0, None, -1)
    global_step = 0
    for epoch in range(1, CFG['epochs'] + 1):
        model.train()
        started, loss_sum, step = time.time(), 0.0, 0
        for step, (images, labels) in enumerate(loader, 1):
            if step > steps_per_epoch:
                step -= 1
                break
            x = to_gpu(images, size)
            y = one_hot_smooth(labels.to(DEV, torch.long, non_blocking=True),
                               CFG['label_smoothing'])
            x, y = mix_batch(x, y)
            if fused is not None:
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    loss = -(y * torch.log_softmax(model(x).float(), 1)).sum(1).mean()
                loss.backward()                    # the hooks clip, step and free each gradient
                global_step += 1
                fused.set_lr_factor(lr_factor(global_step))
            else:
                with torch.amp.autocast('cuda'):
                    logits = model(x).float()
                    logp = torch.log_softmax(logits, 1)
                    if FOCAL_GAMMA:
                        # Focal form of the soft-target cross-entropy: each class term is
                        # down-weighted by (1 - p_c)^gamma, so confident-and-correct rows
                        # contribute less and the low-confidence tail contributes more.
                        loss = -(((1.0 - logp.exp()) ** FOCAL_GAMMA) * y * logp).sum(1).mean()
                    else:
                        loss = -(y * logp).sum(1).mean()
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
            loss_sum += float(loss.detach())

        raw = evaluate(model, v_loader, targets, size)
        eva = evaluate(ema.for_eval(model), v_loader, targets, size)
        ema.park(model)
        row = [time.strftime('%H:%M:%S'), epoch, round(loss_sum / max(step, 1), 4),
               round(raw, 4), round(eva, 4), round(time.time() - started)]
        with open(log, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(row)
        print(f'  epoch {epoch}/{CFG["epochs"]}  loss {row[2]}  val {raw:.4f}  '
              f'ema {eva:.4f}  {row[5]}s', flush=True)
        pick, which = (eva, ema.module) if eva >= raw else (raw, model)
        if pick > best[0]:
            if BEST_TO_DISK:
                # A 840M-parameter snapshot is 3.13 GB of system RAM, and swapping `best` holds
                # two of them for an instant. With 8 GB free next to a 3.13 GB CPU-resident EMA
                # that pages to disk, which is far more expensive than one torch.save.
                torch.save({k: v.detach().cpu() for k, v in which.state_dict().items()},
                           BEST_PATH)
                best = (pick, None, epoch)
            else:
                best = (pick, {k: v.detach().cpu().clone() for k, v in which.state_dict().items()},
                        epoch)

    print(f'best epoch {best[2]}/{CFG["epochs"]}  val {best[0]:.4f}')
    # Collapse signal (§4.5): best epoch <= 2 with validation falling afterwards.
    best_state = best[1] if best[1] is not None else torch.load(BEST_PATH, map_location='cpu')
    del ema                                  # release the 3.13 GB shadow copy
    model.load_state_dict(best_state)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    Path('checkpoints_backup').mkdir(exist_ok=True)
    torch.save({'model_state_dict': best_state, 'class_names': CN, 'backbone': CFG['backbone'],
                'best_epoch': best[2], 'val_acc': best[0], 'cfg': CFG},
               f'checkpoints_backup/{TAG}_base_{stamp}.pt')

    val_prob = predict_tta(model, train_images, val_idx, CFG['tta_scales'])
    tta_val = float((val_prob.argmax(1) == targets[val_idx]).mean())
    # DINO_SMOKE shortens the 11,681-image inference pass so a dry run finishes in a minute;
    # it makes the CSV meaningless, which is why it also renames the outputs.
    n_test = int(os.environ.get('DINO_SMOKE', 0)) or len(test_images)
    test_prob = predict_tta(model, test_images, np.arange(n_test), CFG['tta_scales'])
    if n_test != len(test_images):
        print(f'SMOKE: predicted {n_test} of {len(test_images)} test images -- outputs are junk')
        np.save(BENCH / f'_smoke_{TAG}_test_prob.npy', test_prob)
        print(f'TTA val {tta_val:.4f} | mean conf {test_prob.max(1).mean():.4f} | smoke OK')
        return
    np.save(BENCH / f'{TAG}_base_test_prob.npy', test_prob)

    # same-stage direction check only -- §6: this cannot gate anything
    try:
        from _heldout_split import load_eval_set
        img, lab = load_eval_set()
        rp = predict_tta(model, torch.from_numpy(img) if isinstance(img, np.ndarray) else img,
                         np.arange(len(lab)), CFG['tta_scales'])
        np.save(BENCH / f'{TAG}_base_reserve_prob.npy', rp)
        reserve = float((rp.argmax(1) == lab).mean())
    except Exception as exc:                       # never let a diagnostic kill the run
        reserve = float('nan')
        print(f'reserve check skipped: {type(exc).__name__}: {exc}')

    import pandas as pd
    sub = pd.DataFrame({'ID': np.arange(len(test_prob)),
                        'Label': [CN[i] for i in test_prob.argmax(1)]})
    assert len(sub) == 11681 and set(sub['Label']) <= set(CN)
    sub.to_csv(f'submission_{TAG}_base.csv', index=False)
    print(f'TTA val {tta_val:.4f} | clean-reserve {reserve:.4f} | '
          f'mean test conf {test_prob.max(1).mean():.4f}')
    print(f'submission_{TAG}_base.csv written | bench/{TAG}_base_test_prob.npy saved')


if __name__ == '__main__':
    main()
