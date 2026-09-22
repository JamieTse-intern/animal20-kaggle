"""Generate 32613571_assignment01_notebook.ipynb -- the notebook submitted with the report.

It must show traces of training/fine-tuning, so it replays the recorded per-epoch logs of every
run behind the two final submissions, and rebuilds both submitted CSVs from the saved model
outputs and verifies them row-for-row. The training code itself is included verbatim but guarded
(`RUN_TRAINING = False`): the eight runs took ~28 GPU-hours on one RTX 3080.
"""
import io
import json

cells = []


def md(t):
    cells.append({'cell_type': 'markdown', 'metadata': {},
                  'source': t.strip('\n').splitlines(keepends=True)})


def code(t):
    cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None,
                  'outputs': [], 'source': t.strip('\n').splitlines(keepends=True)})


# ---------------------------------------------------------------- 0. header
md(r"""
# FIT3181/5215 Assignment 1 — 20-class animal image classification

**Student ID:** 32613571  **Name:** Hao Xie  **Kaggle display name:** 32613571_Hao Xie

| final submission | public LB | what it is |
|---|---|---|
| `submission_ens_EVAPHP256temp.csv` | **0.98335** | EVA-02 Large distilled + DINOv3 ViT-H+/16 distilled |
| `submission_ens_V3PHPbase256temp.csv` | **0.98335** | DINOv3 ViT-L/16 distilled + DINOv3 ViT-H+/16 supervised |

Both are temperature-matched equal-weight probability blends of two fine-tuned Vision
Transformers. No parameter in either was fitted on the leaderboard.

**Reading the file names.** `ens` = ensemble; `temp` = the members' sharpness was temperature-
matched before averaging (§6); `256` = the ViT-H+ member was inferred on its native 256px patch
grid (§6); the middle is the member tags below, where a trailing **P** means that model's
self-distilled stage.

| tag | backbone | stage |
|---|---|---|
| `V2` / `V2P` | `vit_large_patch14_reg4_dinov2.lvd142m` — DINOv2 ViT-L/14, LVD-142M | base / + self-distillation |
| `V3` / `V3P` | `vit_large_patch16_dinov3.lvd1689m` — DINOv3 ViT-L/16, LVD-1689M | base / + self-distillation |
| `EVA` / `EVAP` | `eva02_large_patch14_448.mim_m38m_ft_in22k_in1k` — EVA-02 Large, Merged-38M | base / + self-distillation |
| `HP` (`HP base` / `HP`) | `vit_huge_plus_patch16_dinov3.lvd1689m` — DINOv3 ViT-H+/16, LVD-1689M, 840.5M | base / + self-distillation |

**Three different things are called "distillation" here; only the third is ours.**

1. *DINO* itself stands for self-**di**stillation with **no** labels — the self-supervised
   objective Meta pretrained these trunks with, where a student matches a momentum teacher's
   output on different crops of the same image. It involves no labels and no test data.
2. The released DINOv2/DINOv3 **ViT-L checkpoints are themselves distilled** from the much
   larger models of their families, which is part of why they are strong at this size.
3. **"+ self-distillation" in this notebook means only our own stage 2** (§4): a fresh student
   is trained on the labelled data plus pseudo-labelled *test* images whose labels come from an
   earlier model of ours. Senses 1 and 2 are properties of the downloaded weights; sense 3 is
   the only thing this project performed.

**What this notebook shows**

1. Environment and data (§1–§2).
2. The model and the exact training code that produced every checkpoint (§3–§4).
3. **Traces of the eight fine-tuning runs** — the full per-epoch loss/validation logs recorded
   while they ran, plus the console output of the runs (§5).
4. Inference, blending, and a row-for-row rebuild of both submitted CSVs from the saved model
   outputs (§6), and the results table (§7).

**Running it.** Unzip the submission and run this notebook from the folder it sits in — the
`data/` folder beside it holds everything it reads: the saved per-model probability arrays, the
recorded training logs, the two submitted CSVs, and a 20-image sample. Only `numpy`, `pandas`,
`torch` and `matplotlib` are required (`timm` is optional and only used to instantiate the four
architectures in §3). The competition's own image arrays (350 MB) are not redistributed here, so
§2 reports the dataset shapes recorded from them.

Training is **not re-executed here**: the eight runs took about 28 GPU-hours on one RTX 3080, so
the code cell that performs them is guarded by `RUN_TRAINING = False`. Everything else in this
notebook runs in about a minute and its outputs below are real.

**Trained models.** The four fine-tuned checkpoints behind the two submissions (8.52 GB) are too
large for this archive and are shared at
<https://drive.google.com/drive/folders/1gMk58k2UlutXSzoQ_jx7WEHFjQSARvDY?usp=sharing>.
Re-running the 6-view TTA from each checkpoint — **at the scales that submission actually uses**,
including 192/224/256 for the two ViT-H+ members — reproduces the exact array blended below:
300/300 sampled test images agree on argmax for all four, max probability difference 4.2e-04
(mixed-precision jitter).

**Academic integrity.** No test label was ever assigned, reviewed, corrected or verified by a
human or by an external AI. Every probability used here is a forward pass of a model trained by
this code; pseudo-labelling uses only the permitted route — labels generated by our own models.
""")

# ---------------------------------------------------------------- 1. setup
md('## 1. Environment')

code(r'''
import json, math, sys, csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch

print('python     ', sys.version.split()[0])
print('torch      ', torch.__version__)
print('numpy      ', np.__version__, '| pandas', pd.__version__)
try:
    import timm, torchvision
    print('timm       ', timm.__version__, '| torchvision', torchvision.__version__)
except ImportError:
    print('timm/torchvision not installed (needed only for training)')
print('CUDA       ', torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')

# This notebook runs either inside the project repository or from the submitted bundle, which
# ships the data it needs in data/ (the competition images themselves are not redistributed).
STANDALONE = Path('data').is_dir()
BENCH = Path('data/bench') if STANDALONE else Path('bench')
LOGS = Path('data/logs') if STANDALONE else Path('.')
SUBS = Path('data/submissions') if STANDALONE else Path('.')
SAMPLES = Path('data/samples') if STANDALONE else Path('bench')
print('mode       ', 'standalone bundle (data/)' if STANDALONE else 'project repository')

CLASSES = list(np.load(BENCH / 'class_names.npy', allow_pickle=True))
print('classes    ', CLASSES)
''')

# ---------------------------------------------------------------- 2. data
md(r"""
## 2. Data

**Competition data.** 9,466 labelled JPEGs in 20 class folders (`Animals_Dataset.zip`, 617 MB)
and 11,681 unlabelled test images (`test_set.zip`, 64x64 JPEG q75 4:2:0). The labelled images
arrive at their **original resolution** — mostly 300-640px, a minority already 64x64 — so the
first preprocessing step squashes every one of them to 64x64 with LANCZOS resampling, matching
the resolution the test set is delivered at. `bench/train_images.npy` is that (9466, 3, 64, 64)
uint8 array. The split is stratified per class at `SEED = 2026`, 15% held out for
validation, and frozen on disk (`bench/validation_indices.npy`) so every run in the project is
scored on exactly the same images.

**The problem the data poses.** The training set is essentially all photographs, while a
sizeable minority of the test set is *renditions* — paintings, drawings, sculptures, line art.
The competition does not state that fraction and it cannot be measured directly without
labelling test images, which is not permitted. It can be backed out, though: solving
`test = w x (rendition accuracy) + (1 - w) x (photo accuracy)` with each submission's measured
rendition accuracy, validation accuracy and public score gives **w = 21-27%** (21.0-24.0% from
submission 1, 24.0-27.1% from submission 2, whose members are different models). Whatever is in
that stratum, this domain gap is the single largest error source.

**External training data.** DomainNet *painting* and ImageNet-R images for the same 20 classes,
pushed through the exact test degradation (squash-crop, LANCZOS to 64px, re-encoded with the
test set's own JPEG quantisation tables) so they match the test spectrum. They are mixed into
every batch by a class-balanced draw at ratio 0.30.

**Reserved evaluation set.** 772 external rendition images were reserved on disk *before* any
training draw could see them, and the training draw asserts it never touches them. This is the
only non-circular rendition metric in the project.

**Normalisation.** Images are kept as uint8 64x64 and normalised with ImageNet statistics
(mean 0.485/0.456/0.406, std 0.229/0.224/0.225); each backbone's own normalisation is applied
inside the model by a fixed re-normalising layer, so one pipeline feeds all four backbones.
""")

code(r'''
meta_path = SAMPLES / 'dataset_metadata.json'
if meta_path.is_file():                       # standalone bundle: facts recorded from the arrays
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    counts = np.array(meta['per_class_counts'])
else:                                         # inside the project repository: read them live
    targets = np.load(BENCH / 'targets.npy')
    val_idx = np.load(BENCH / 'validation_indices.npy')
    meta = {
        'train_images_shape': list(np.load(BENCH / 'train_images.npy', mmap_mode='r').shape),
        'train_images_dtype': 'uint8',
        'test_images_shape': list(np.load(BENCH / 'test_images.npy', mmap_mode='r').shape),
        'external_images_shape': list(np.load(BENCH / 'external_images.npy', mmap_mode='r').shape),
        'n_train_split': int(len(targets) - len(val_idx)),
        'n_validation': int(len(val_idx)),
        'n_reserved_renditions': int(len(np.load(BENCH / 'external_eval_holdout.npz')['indices'])),
        'note': 'read live from the full arrays',
    }
    counts = np.bincount(targets, minlength=20)

print(f"train images      {tuple(meta['train_images_shape'])}  dtype {meta['train_images_dtype']}")
print(f"  training split  {meta['n_train_split']:,}")
print(f"  validation      {meta['n_validation']:,}  (stratified 15%, frozen at SEED=2026)")
print(f"test images       {tuple(meta['test_images_shape'])}")
print(f"external pool     {tuple(meta['external_images_shape'])}  of which "
      f"{meta['n_reserved_renditions']:,} are the reserved rendition evaluation set "
      f"(never trained on)")
print('\nlabelled images per class:')
print(pd.DataFrame({'class': CLASSES, 'train+val': counts}).to_string(index=False))
print('\n' + meta['note'])
''')

code(r'''
# The 64x64 sources, and what the external rendition data looks like after the test degradation.
import matplotlib.pyplot as plt

sample_path = SAMPLES / 'sample_images.npz'
if sample_path.is_file():
    s = np.load(sample_path)
    photos, renditions = s['photos'], s['renditions']
else:                                          # inside the repository: draw from the full arrays
    rng = np.random.default_rng(0)
    tr = np.load(BENCH / 'train_images.npy', mmap_mode='r')
    ex = np.load(BENCH / 'external_images.npy', mmap_mode='r')
    ex_lab = np.load(BENCH / 'external_labels.npy')
    photos = np.asarray([tr[int(rng.choice(np.flatnonzero(targets == c)))] for c in range(10)])
    renditions = np.asarray([ex[int(rng.choice(np.flatnonzero(ex_lab == c)))] for c in range(10)])

fig, axes = plt.subplots(2, 10, figsize=(11, 2.6))
for col in range(10):
    axes[0, col].imshow(photos[col].transpose(1, 2, 0))
    axes[0, col].set_title(CLASSES[col], fontsize=7, color='#52514e')
    axes[1, col].imshow(renditions[col].transpose(1, 2, 0))
for ax in axes.ravel():
    ax.set_xticks([]); ax.set_yticks([])
axes[0, 0].set_ylabel('competition\n(photos)', fontsize=7, color='#52514e')
axes[1, 0].set_ylabel('external\n(renditions)', fontsize=7, color='#52514e')
fig.suptitle('Training images: competition photographs (top) and degraded external renditions (bottom)',
             fontsize=9, color='#0b0b0b')
fig.tight_layout()
plt.show()
''')

# ---------------------------------------------------------------- 3. architecture
md(r"""
## 3. Model architecture

Every final model is a **pretrained Vision Transformer fine-tuned end-to-end** with a fresh
20-way linear head. Four backbones were trained across three pretraining recipes. Submission 1
blends two *recipes* (EVA-02 and DINOv3); submission 2 blends two *DINOv3* models that differ in
capacity and in training stage — ViT-L self-distilled and ViT-H+ supervised only.

| Architecture | Additional notes |
|---|---|
| `vit_large_patch14_reg4_dinov2.lvd142m` (**V2**) | DINOv2 self-supervised ViT-L/14, 4 register tokens, 304M parameters. Input 196px (divisible by 14). |
| `vit_large_patch16_dinov3.lvd1689m` (**V3**) | DINOv3 self-supervised ViT-L/16, 303M parameters. Input 192px. |
| `eva02_large_patch14_448.mim_m38m_ft_in22k_in1k` (**EVA**) | EVA-02 Large, masked image modelling on Merged-38M then ImageNet-22k/1k supervised fine-tuning, 304M parameters. Input 196px. |
| `vit_huge_plus_patch16_dinov3.lvd1689m` (**HP**) | DINOv3 self-supervised ViT-H+/16, **840.5M parameters** — 2.8x the others, 32 blocks, width 1280. Input 192px. |
| Pooling | mean over patch tokens (`global_pool='avg'`), class/register tokens dropped |
| Classifier head | a single `Linear(width, 20)` — 1024 for the ViT-L models, 1280 for ViT-H+ — randomly initialised (the pretrained 1000-way head is discarded) |
| Normalisation layer | fixed buffers that re-normalise ImageNet-normalised input into each backbone's own mean/std |
| EMA | exponential moving average of all weights, decay 0.999; inference uses the EMA weights |

The cell below builds each backbone from `timm` (architecture only, `pretrained=False`, so it
downloads nothing) and reports its real parameter count and head.
""")

code(r'''
import torch.nn as nn

class NormalizedModel(nn.Module):
    """Re-normalise an ImageNet-normalised batch into the backbone's own statistics."""
    MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    def __init__(self, backbone, mean, std):
        super().__init__()
        mean_t, std_t = torch.tensor(mean), torch.tensor(std)
        self.register_buffer('scale', (torch.tensor(self.STD) / std_t).view(1, 3, 1, 1))
        self.register_buffer('shift', ((torch.tensor(self.MEAN) - mean_t) / std_t).view(1, 3, 1, 1))
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x * self.scale + self.shift)


BACKBONES = {
    'V2  DINOv2 ViT-L/14': ('vit_large_patch14_reg4_dinov2.lvd142m', 196),
    'V3  DINOv3 ViT-L/16': ('vit_large_patch16_dinov3.lvd1689m', 192),
    'EVA EVA-02 Large':    ('eva02_large_patch14_448.mim_m38m_ft_in22k_in1k', 196),
    'HP  DINOv3 ViT-H+/16': ('vit_huge_plus_patch16_dinov3.lvd1689m', 192),
}

try:
    import timm
except ImportError:                 # timm is needed only to instantiate the architectures
    timm = None
    print('timm is not installed, so the architectures cannot be instantiated here.')
    print('Recorded sizes: DINOv2 ViT-L/14 304.4M, DINOv3 ViT-L/16 303.1M, EVA-02 L 304.1M, '
          'DINOv3 ViT-H+/16 840.5M parameters; all with mean patch-token pooling and a fresh '
          'Linear(width, 20) head.')

for name, (arch, size) in (BACKBONES.items() if timm else []):
    bb = timm.create_model(arch, pretrained=False, num_classes=20,
                           dynamic_img_size=True, global_pool='avg')
    total = sum(p.numel() for p in bb.parameters())
    head = bb.get_classifier()
    print(f'{name:22s} {arch}')
    print(f'{"":22s} {total/1e6:6.1f}M parameters | input {size}px | pool '
          f'{getattr(bb, "global_pool", "?")} | prefix tokens {getattr(bb, "num_prefix_tokens", "-")}')
    print(f'{"":22s} head: {head.__class__.__name__}{tuple(head.weight.shape)} '
          f'(trainable, randomly initialised)')
    del bb
''')

# ---------------------------------------------------------------- 4. training code
md(r"""
## 4. How each model was trained

Two stages per backbone, 16 epochs each.

**Stage 1 — supervised fine-tuning ("base").** The labelled training split plus the external
rendition images. Augmentation: `RandomJpeg(p=0.3, quality 55–95)`, horizontal flip,
TrivialAugmentWide, `RandomResizedCrop(64, scale 0.60–1.0)`, `RandomErasing(0.25)`, and
MixUp(0.2)/CutMix(1.0) on half the batches with soft targets. Label smoothing 0.1, AdamW,
2 warmup epochs then cosine decay, AMP with gradient checkpointing, EMA 0.999, batch size 20.
The trunk uses a **flat learning rate of 2e-5** and the head 1e-3.

**Stage 2 — self-distillation (Noisy Student).** A *fresh* student, re-initialised from
pretrained weights, trained on labelled + external + **pseudo-labelled test images**. The
pseudo-labels come from an external teacher — the best blend available at that point, never the
student's own predictions. Selection is `argmax` with teacher confidence >= 0.7, and pseudo rows
carry loss weight 0.5 against 1.0 for real rows. Exactly one round per model. This is the route
the assignment permits: labels generated by our own trained models.

Teacher chain, each generation taught by the previous generation's best blend:

    0.96499 -> 0.97566 -> 0.97673 -> 0.97716 -> 0.97950 -> 0.98132 -> 0.98196 (teacher of HP)

**Fitting an 840M-parameter model in 10 GB.** The largest backbone, DINOv3 ViT-H+/16, has 2.8x
the parameters of the others. Fine-tuning all of them the ordinary way needs about 18 GB, and
the card has 10 GB — but almost none of that is activations, so a smaller batch does not help.
The cost is parameters (3.13 GB), gradients (3.13 GB), the Adam moments (6.26 GB) and an fp32
EMA copy (3.13 GB). Three changes, none of which touches the model, the data or the objective,
bring it to a measured **8.3 GB at the same batch size of 20**:

| change | saves |
|---|---|
| 8-bit Adam moments (`bitsandbytes.AdamW8bit`); master weights stay fp32 | 4.7 GB |
| the EMA copy held in system RAM and swapped in only to validate | 3.13 GB |
| the EMA updated every 4 steps at `decay**4`, so its horizon in steps is unchanged | — (speed) |

Master weights are deliberately **not** stored in bf16: at lr 2e-5 on weights of magnitude
~0.02 the relative update is 1e-3 while bf16 resolves ~8e-3, so every update would be rounded
away and the run would silently learn nothing.

The next cell contains the real training step, the EMA, and the 6-view TTA used at inference.
It is guarded so the notebook stays runnable.
""")

code(r'''
RUN_TRAINING = False     # the eight runs below took ~28 GPU-hours on one RTX 3080

class ModelEMA:
    """Exponential moving average of the weights; inference uses these."""

    def __init__(self, model, decay=0.999):
        import copy
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model, step):
        d = min(self.decay, (1.0 + step) / (10.0 + step))
        for e, m in zip(self.module.state_dict().values(), model.state_dict().values()):
            e.mul_(d).add_(m.detach(), alpha=1.0 - d) if e.dtype.is_floating_point else e.copy_(m)


def one_hot_smooth(labels, smoothing=0.1, num_classes=20):
    off = smoothing / num_classes
    t = torch.full((labels.size(0), num_classes), off, device=labels.device)
    return t.scatter_(1, labels.unsqueeze(1), 1.0 - smoothing + off)


def train_one_epoch(model, ema, loader, opt, sched, scaler, image_size, epoch, steps_per_epoch):
    """One epoch of the real training loop (dinov3_base_train.py / dinov3_pseudo_train.py).

    `weights` is 1.0 for labelled and external rows and 0.5 for pseudo-labelled test rows, so
    all three sources share one collated batch.
    """
    model.train()
    loss_sum = 0.0
    for step, batch in enumerate(loader, 1):
        images, labels = batch[0], batch[1]
        weights = batch[2].to('cuda') if len(batch) > 2 else None
        x = to_gpu(images, image_size)
        y = one_hot_smooth(labels.to('cuda', torch.long, non_blocking=True))
        x, y = mix_batch(x, y)                      # MixUp / CutMix on soft targets
        with torch.amp.autocast('cuda'):
            per_sample = -(y * torch.log_softmax(model(x).float(), 1)).sum(1)
            loss = ((per_sample * weights).sum() / weights.sum().clamp(min=1e-8)
                    if weights is not None else per_sample.mean())
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        before = scaler.get_scale()
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        if scaler.get_scale() >= before:
            sched.step()
        ema.update(model, epoch * steps_per_epoch + step)
        loss_sum += float(loss.detach())
    return loss_sum / max(step, 1)


@torch.inference_mode()
def predict_tta(model, images, indices, scales, chunk=32):
    """6-view test-time augmentation: three scales x horizontal flip, averaged as probabilities."""
    model.eval()
    out = []
    for i in range(0, len(indices), chunk):
        batch = torch.stack([eval_transform(images[int(j)]) for j in indices[i:i + chunk]])
        acc = None
        for s in scales:
            x = to_gpu(batch, s)
            for view in (x, torch.flip(x, dims=[3])):
                with torch.amp.autocast('cuda'):
                    p = torch.softmax(model(view).float(), 1)
                acc = p if acc is None else acc + p
        out.append((acc / (2 * len(scales))).cpu())
    return torch.cat(out).numpy()


print('Training code defined. RUN_TRAINING =', RUN_TRAINING)
print('The eight runs were launched exactly like this (full source: dinov3_base_train.py,')
print('dinov3_pseudo_train.py; every option is an environment variable):\n')
print("""DINO_TAG=dinov2vitl    DINO_BACKBONE=vit_large_patch14_reg4_dinov2.lvd142m \\
  DINO_SIZE=196 DINO_TTA=168,196,224 DINO_POOL=avg DINO_GRAD_CKPT=1 DINO_BATCH=20 \\
  DINO_LR=2e-5 DINO_LAYER_DECAY=1.0 python dinov3_base_train.py

DINO_TAG=dinov2vitlps1 DINO_BACKBONE=vit_large_patch14_reg4_dinov2.lvd142m \\
  DINO_SIZE=196 DINO_TTA=168,196,224 DINO_POOL=avg DINO_GRAD_CKPT=1 DINO_BATCH=20 \\
  DINO_TEACHER=bench/teacher_KDCD3temp.npy python dinov3_pseudo_train.py

DINO_TAG=dinov3vitl    DINO_BACKBONE=vit_large_patch16_dinov3.lvd1689m \\
  DINO_POOL=avg DINO_GRAD_CKPT=1 DINO_BATCH=20 DINO_LR=2e-5 DINO_LAYER_DECAY=1.0 \\
  python dinov3_base_train.py

DINO_TAG=dinov3vitlps1 DINO_BACKBONE=vit_large_patch16_dinov3.lvd1689m \\
  DINO_POOL=avg DINO_GRAD_CKPT=1 DINO_BATCH=20 \\
  DINO_TEACHER=bench/teacher_V2P.npy python dinov3_pseudo_train.py

DINO_TAG=eva02l        DINO_BACKBONE=eva02_large_patch14_448.mim_m38m_ft_in22k_in1k \\
  DINO_SIZE=196 DINO_TTA=168,196,224 DINO_GRAD_CKPT=1 DINO_BATCH=20 \\
  DINO_LR=2e-5 DINO_LAYER_DECAY=1.0 python dinov3_base_train.py

DINO_TAG=eva02lps1     DINO_BACKBONE=eva02_large_patch14_448.mim_m38m_ft_in22k_in1k \\
  DINO_SIZE=196 DINO_TTA=168,196,224 DINO_GRAD_CKPT=1 DINO_BATCH=20 \\
  DINO_TEACHER=bench/teacher_V2PV3P.npy python dinov3_pseudo_train.py

# DINOv3 ViT-H+/16, 840.5M parameters. The four extra switches are the 10 GB memory path
# described in section 4; they change no training number, only where the state is held.
DINO_TAG=dinov3vithp   DINO_BACKBONE=vit_huge_plus_patch16_dinov3.lvd1689m \
  DINO_POOL=avg DINO_GRAD_CKPT=1 DINO_BATCH=20 DINO_LR=2e-5 DINO_LAYER_DECAY=1.0 \
  DINO_OPT8BIT=1 DINO_EMA_CPU=1 DINO_EMA_EVERY=4 DINO_BEST_TO_DISK=1 \
  python dinov3_base_train.py

DINO_TAG=dinov3vithpps1 DINO_BACKBONE=vit_huge_plus_patch16_dinov3.lvd1689m \
  DINO_POOL=avg DINO_GRAD_CKPT=1 DINO_BATCH=20 \
  DINO_OPT8BIT=1 DINO_EMA_CPU=1 DINO_EMA_EVERY=4 DINO_BEST_TO_DISK=1 \
  DINO_TEACHER=bench/teacher_V2PEVAP.npy python dinov3_pseudo_train.py""")
''')

# ---------------------------------------------------------------- 5. training traces
md(r"""
## 5. Traces of the eight fine-tuning runs

Each run wrote one row per epoch while it trained (`<tag>_progress.csv`: wall clock, training
loss, validation accuracy of the raw weights, validation accuracy of the EMA weights, seconds).
Those files are replayed below exactly as recorded — they are the training traces.
""")

code(r'''
RUNS = [
    ('dinov2vitl',     'V2   DINOv2 ViT-L/14 — base'),
    ('dinov2vitlps1',  'V2P  DINOv2 ViT-L/14 — distilled'),
    ('dinov3vitl',     'V3   DINOv3 ViT-L/16 — base'),
    ('dinov3vitlps1',  'V3P  DINOv3 ViT-L/16 — distilled'),
    ('eva02l',         'EVA  EVA-02 Large — base'),
    ('eva02lps1',      'EVAP EVA-02 Large — distilled'),
    ('dinov3vithp',    'HP   DINOv3 ViT-H+/16 — base'),
    ('dinov3vithpps1', 'HP   DINOv3 ViT-H+/16 — distilled'),
]

logs = {}
for tag, label in RUNS:
    df = pd.read_csv(LOGS / f'{tag}_progress.csv')
    df = df[pd.to_numeric(df['epoch'], errors='coerce').notna()].copy()
    df['epoch'] = df['epoch'].astype(int)
    logs[label] = df

for label, df in logs.items():
    best = df.loc[df[['val_acc', 'ema_val_acc']].max(axis=1).idxmax()]
    print(f'\n=== {label} '.ljust(78, '='))
    print(df[['wall', 'epoch', 'loss', 'val_acc', 'ema_val_acc', 'seconds']].to_string(index=False))
    print(f'  best epoch {int(best.epoch)}/{len(df)}  '
          f'validation {max(best.val_acc, best.ema_val_acc)*100:.2f}%  '
          f'| total {df.seconds.sum()/60:.0f} min at {df.seconds.mean():.0f} s/epoch')
''')

code(r'''
# Training curves. Two charts, one measure each (never two scales on one axis).
# Colours: validated categorical palette, assigned in fixed order; the per-epoch tables
# printed above are the table view, so identity never rests on colour alone.
import matplotlib.pyplot as plt

SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e1e0d9'

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
for ax, (col, title, ylab) in zip(axes, [
        ('loss', 'Training loss', 'loss (soft-target cross-entropy)'),
        ('ema_val_acc', 'Validation accuracy (EMA weights)', 'accuracy')]):
    for (label, df), colour in zip(logs.items(), SERIES):
        y = df[col].astype(float)
        ax.plot(df['epoch'], y, color=colour, linewidth=2, solid_capstyle='round',
                label=label, zorder=3)
    ax.set_title(title, fontsize=11, color=INK, pad=10, loc='left')
    ax.set_xlabel('epoch', fontsize=9, color=MUTED)
    ax.set_ylabel(ylab, fontsize=9, color=MUTED)
    ax.set_xlim(0.6, 16.4)
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color('#c3c2b7')
    ax.tick_params(colors=MUTED, labelsize=8)

axes[1].set_ylim(0.984, 0.998)
axes[1].annotate('validation saturates here — it cannot rank these models (§7)',
                 xy=(0.02, 0.04), xycoords='axes fraction', fontsize=8, color=MUTED)
handles, labels_ = axes[0].get_legend_handles_labels()
fig.legend(handles, labels_, frameon=False, fontsize=8, labelcolor=MUTED,
           loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.02))
fig.suptitle('Fine-tuning traces of the eight runs behind the two submissions',
             fontsize=12, color=INK, x=0.02, ha='left')
fig.tight_layout(rect=(0, 0.08, 1, 1))
plt.show()

print('Validation is close to its ceiling for every run (98.2–99.5%), which is why model')
print('selection in this project was never made on validation alone — see §7.')
''')

md(r"""
The distilled runs start from a lower loss than their own base runs because the pseudo-labelled
test images are easier targets than the augmented labelled data, and their validation accuracy
is flat: **validation saturates at ~99% and cannot separate these models.** What separates them
is the held-out rendition set and the leaderboard (§7).

Below are the tails of the actual console logs of the two runs that make up the first
submission — the raw traces as printed while training.
""")

code(r'''
for tag, label in [('eva02l_ps1', 'EVAP EVA-02 Large distilled'),
                   ('dinov3vithp_base', 'HP   DINOv3 ViT-H+/16 base'),
                   ('dinov3vithp_ps1', 'HP   DINOv3 ViT-H+/16 distilled')]:
    text = (LOGS / f'{tag}.log').read_text(encoding='utf-8', errors='ignore').splitlines()
    keep = [l for l in text if l.strip() and 'warn' not in l.lower() and 'Warning' not in l]
    print(f'\n=== {label}  ({tag}.log) '.ljust(96, '='))
    print('\n'.join(keep[-26:]))
''')

# ---------------------------------------------------------------- 6. blending
md(r"""
## 6. Inference, blending, and rebuilding the two submissions

Inference uses **6-view TTA** (three scales x horizontal flip, averaged in probability space)
with the EMA weights, and each run saved its `(11681, 20)` probability array. The scales are
160/192/224px (168/196/224 for the patch-14 models, so every side stays divisible by 14) —
**except for the two ViT-H+ members, which use 192/224/256px.**

That exception is the last change made to these submissions and it is worth stating plainly.
ViT-H+ has a **native 256px position grid** (16x16 = 256 patch tokens); run at 192px it sees
144, and `dynamic_img_size` interpolates the position embedding to fit. Moving its scales up to
192/224/256 restores the grid it was pretrained on. Measured on both held-out strata before any
submission — this is inference-only post-processing on a fixed checkpoint, so local evaluation
is valid here — it raised the supervised model's rendition accuracy from 95.21% to **95.98%**
(6 images fixed, 0 broken, McNemar p = 0.031) and the distilled model's from 92.75% to 93.26%,
while lowering neither model's validation accuracy. Inside a two-model blend it moves only
**7 of 11,681 rows**, because the other member is unchanged and outvotes it on the marginal
ones: a real, locally significant effect that is almost invisible where it has to count.

Two models are combined by **matching their sharpness and then averaging with equal weight**: a
model with higher mean top-1 confidence dominates the `argmax` of a plain average regardless of
whether it is more often right, so for each member a temperature `T` is solved by bisection such
that `softmax(log p / T)` has the same mean top-1 confidence as the reference member. Nothing in
this construction is fitted on the leaderboard.

The cell below rebuilds both submitted CSVs from the saved probabilities and compares them, row
for row, with the files that were actually submitted to Kaggle.
""")

code(r'''
EPS = 1e-12

def normalise(p):
    p = np.asarray(p, dtype=np.float64)
    return p / p.sum(axis=1, keepdims=True)

def sharpen(p, T):
    z = np.log(np.clip(p, EPS, None)) / T
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

def fit_temperature(p, target_confidence, low=0.05, high=20.0, iterations=200):
    """Mean top-1 confidence is monotone decreasing in T, so bisect."""
    for _ in range(iterations):
        mid = 0.5 * (low + high)
        if sharpen(p, mid).max(axis=1).mean() > target_confidence:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)

def temperature_matched_blend(members):
    members = [normalise(p) for p in members]
    target = members[0].max(axis=1).mean()
    matched, temps = [], []
    for i, p in enumerate(members):
        T = 1.0 if i == 0 else fit_temperature(p, target)
        temps.append(T)
        matched.append(p if T == 1.0 else sharpen(p, T))
    return sum(matched) / len(matched), temps

PICKS = {
    'pick 1': ('submission_ens_EVAPHP256temp.csv', ['eva02lps1', 'dinov3vithpps1B'], 0.98335),
    'pick 2': ('submission_ens_V3PHPbase256temp.csv', ['dinov3vitlps1', 'dinov3vithp_baseB'], 0.98335),
}

all_match = True
for name, (filename, tags, public) in PICKS.items():
    members = [normalise(np.load(BENCH / f'{t}_test_prob.npy')) for t in tags]
    blended, temps = temperature_matched_blend(members)
    labels = np.array([CLASSES[i] for i in blended.argmax(1)])

    rebuilt = pd.DataFrame({'ID': np.arange(len(labels)), 'Label': labels})
    assert rebuilt['ID'].tolist() == list(range(11681)) and set(rebuilt['Label']) <= set(CLASSES)
    submitted = pd.read_csv(SUBS / filename)['Label'].values
    differing = int((labels != submitted).sum())
    all_match &= differing == 0

    print(f'{name}: {filename}   public LB {public:.5f}')
    for tag, T, p in zip(tags, temps, members):
        print(f'    {tag:16s} T = {T:.4f}   mean top-1 confidence '
              f'{p.max(1).mean():.4f} -> {blended.max(1).mean():.4f} after blending')
    print(f'    rebuilt from the saved model outputs: {differing} rows differ from the '
          f'submitted file  {"EXACT MATCH" if differing == 0 else "MISMATCH"}\n')

print('RESULT:', 'both submissions reproduce exactly' if all_match else 'MISMATCH')
''')

# ---------------------------------------------------------------- 7. results
md(r"""
## 7. Results

`validation` is the best epoch's EMA accuracy on the 1,419 held-out competition images;
`renditions` is accuracy on the 772 reserved external rendition images (never trained on);
`public LB` is the Kaggle public score on 80% of the test set.
""")

code(r'''
lab = np.load(BENCH / 'reserve_preds.npz')['lab']

def reserve_accuracy(tag):
    p = np.load(BENCH / f'{tag}_reserve_prob.npy')
    return 100 * (p.argmax(1) == lab).mean()

rows = [
    ('V2   DINOv2 ViT-L/14 base',      'dinov2vitl_base',  'dinov2vitl',     0.97363),
    ('V2P  DINOv2 ViT-L/14 distilled', 'dinov2vitlps1',    'dinov2vitlps1',  0.97950),
    ('V3   DINOv3 ViT-L/16 base',      'dinov3vitl_base',  'dinov3vitl',     0.97556),
    ('V3P  DINOv3 ViT-L/16 distilled', 'dinov3vitlps1',    'dinov3vitlps1',  0.98196),
    ('EVA  EVA-02 Large base',         'eva02l_base',      'eva02l',         0.96915),
    ('EVAP EVA-02 Large distilled',    'eva02lps1',        'eva02lps1',      0.98132),
    ('HP   DINOv3 ViT-H+/16 base',     'dinov3vithp_base', 'dinov3vithp',    0.97950),
    ('HP   DINOv3 ViT-H+/16 distilled','dinov3vithpps1',   'dinov3vithpps1', 0.98014),
    ('HP   base, native-grid TTA',     'dinov3vithp_baseB','dinov3vithp',    None),
    ('HP   distilled, native-grid TTA','dinov3vithpps1B',  'dinov3vithpps1', None),
]

table = []
for label, prob_tag, log_tag, lb in rows:
    df = logs[[k for k in logs if k.startswith(label.split()[0])
               and ('distilled' in k) == ('distilled' in label)][0]]
    best = df[['val_acc', 'ema_val_acc']].max(axis=1).max() * 100
    table.append({'model': label, 'validation %': round(best, 2),
                  'renditions %': round(reserve_accuracy(prob_tag), 2),
                  'public LB': '—' if lb is None else f'{lb:.5f}'})

blends = [('pick 1  EVAP + HP', ['eva02lps1', 'dinov3vithpps1B'], 0.98335),
          ('pick 2  V3P + HP base', ['dinov3vitlps1', 'dinov3vithp_baseB'], 0.98335)]
for label, tags, lb in blends:
    # Use the temperatures solved on the TEST probabilities in §6 -- those define the blend that
    # was actually submitted. Re-solving them on the 772 reserve rows would score a different,
    # never-submitted blend (it moves these numbers by one to two images).
    _, temps = temperature_matched_blend(
        [normalise(np.load(BENCH / f'{t}_test_prob.npy')) for t in tags])
    members = [normalise(np.load(BENCH / f'{t}_reserve_prob.npy')) for t in tags]
    blended = sum(p if T == 1.0 else sharpen(p, T)
                  for p, T in zip(members, temps)) / len(members)
    table.append({'model': label, 'validation %': None,
                  'renditions %': round(100 * (blended.argmax(1) == lab).mean(), 2),
                  'public LB': f'{lb:.5f}'})

print(pd.DataFrame(table).to_string(index=False, na_rep='—'))
''')

md(r"""
**Reading the table.**

* Validation is saturated: every model sits between 98% and 99.5% and the ordering there does
  not match the leaderboard, so no decision in this project was made on validation alone.
* The reserved renditions are where the models actually differ, and both blends score **above
  both of their own members** there — the signature that the two models are right on different
  images, which is why blending them pays.
* Self-distillation is worth +0.59pt (DINOv2) to +1.22pt (EVA-02) over the same backbone's
  base model — **except for ViT-H+, where it added only +0.064pt.** That base was already the
  strongest the project produced (0.97950, and the best rendition accuracy of any model at
  95.21%), so the pseudo stage had little to add and measurably eroded it: renditions fell to
  92.75% and mean confidence from 0.8580 to 0.8097. A lineage's distillation gain is not a
  constant; it is conditional on the base having headroom.
* **That erosion is exactly what made it the best blend partner.** At mean confidence 0.8097
  it is the bluntest model here and EVA-02 distilled is the sharpest (0.9013); after temperature
  matching, the blend of the two is the project's best result, and it beats both of its own
  members and both earlier blends on the leaderboard.
* Capacity paid here and had not before. A 658M ConvNeXt-V2 tried earlier scored 0.95869, but
  inside this ViT + DINO family the 840.5M model beat its own 303M counterpart at the same
  stage by +0.394pt with the same data and recipe.
* **Submission 2 deliberately contains no distilled ViT-H+ and one model that never saw a test
  image.** Its members are DINOv3 ViT-L distilled and the ViT-H+ *supervised* model, which was
  trained on labelled and external data only. The two submissions therefore fail differently:
  if the pseudo-labelling route has a systematic weakness, submission 2 carries only half of it.

**Limitations.** Validation cannot rank these models; the reserved rendition set covers only
the rendition stratum (21-27% of the test set, and backed out from accuracies rather than
measured directly), and it has no valid reading across the base-to-distilled boundary at all.
The ViT-H+ runs also carry two uncontrolled differences forced by the 10 GB budget — an 8-bit
optimiser and a 4x-coarsened EMA — so "capacity pays" is stated with those attached rather than
isolated. Remaining gains are smaller than one submission can measure.
""")

nb = {'cells': cells,
      'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python',
                                  'name': 'python3'},
                   'language_info': {'name': 'python', 'version': '3.11'}},
      'nbformat': 4, 'nbformat_minor': 5}

with io.open('32613571_assignment01_notebook.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f'wrote 32613571_assignment01_notebook.ipynb  ({len(cells)} cells: '
      f'{sum(c["cell_type"] == "markdown" for c in cells)} markdown, '
      f'{sum(c["cell_type"] == "code" for c in cells)} code)')
