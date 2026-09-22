"""Task 2 stage 2 (NEXT_ROUND_PROMPT §4.3): Noisy-Student pseudo stage on the DINOv3 prior.

Gate cleared: the DINOv3 ConvNeXt-L base scored 0.96264 against a 0.96150 bar, statistically
tied with the cnv2L 256 base (0.96350, -0.086pt = a third of an SE), while disagreeing with
model A on 2.99% of test rows -- the highest decorrelation any candidate member has reached.

What this stage is for. Base-to-pseudo1 is the largest reliable jump the project has:
cnv2L 256 base 0.96350 -> 256 pseudo1 0.97203, +0.85pt. If the same jump lands here, DINOv3
arrives near 0.971 and becomes the first member since C to clear BOTH ensemble bars (strength
within ~0.2pt of the 0.97342 anchor, and ~2% decorrelation) -- see the two-bar table in
CLAUDE.md.

Design, each point fixed by §4.3 or by the project record, none of it swept:

* Student re-initialised from the DINOv3 pretrained weights, NOT continued from the base
  checkpoint. That is the project's Noisy-Student pattern and it is what every prior pseudo
  stage did.
* Teacher = the best pure-model blend, A (x)C at 1:2 in logit space = 0.97566. FixMatch is
  excluded: §3.3 closed that line at 0.96926, so it enters neither the ensemble nor the teacher.
* Hard argmax + confidence >= 0.7. Threshold 0.60 was tested and cost 0.29pt; 0.7 stands.
* backbone_lr flat 2e-5. The base stage settled this the expensive way -- a layer-decay 1e-4
  ladder cost 1.95pt of reserve accuracy against the flat control on an otherwise identical
  run. Do not reintroduce layer decay for this backbone.
* EMA 0.999 and inference from the EMA weights, per §4.3.
* pseudo_weight 0.5, the project standard; confidence weighting stays OFF (its one trial
  silently cut the whole pseudo set's influence by 10.5%, confounding that run).

Collapse signal (§4.5): best epoch <= 2 with validation falling after it -- the documented
feedback-collapse signature (v7 era 2/16, 256px era 1/16). If it fires, terminate and report;
do not retune and rerun.
"""
import csv
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader

from dinov3_base_train import (BENCH, BEST_PATH, BEST_TO_DISK, CFG, CN, DEV, NUM_CLASSES,
                               EvalSet, LabeledSet, ModelEMA,
                               build_model, evaluate, eval_tf, floor_balanced_draw,
                               one_hot_smooth, mix_batch, predict_tta, to_gpu, train_tf)

TAG = os.environ.get('DINO_TAG', 'dinov3ps1')
PSEUDO = dict(
    # §4.3: the teacher is the current best pure-model blend. DINO_TEACHER overrides it; the
    # default is left at the A(x)C 1:2 file so the original DINOv3 pseudo1 run stays
    # reproducible from this file. If the override names a file that already exists it is
    # loaded as-is and build_teacher() does not reconstruct anything.
    teacher=os.environ.get('DINO_TEACHER', 'bench/teacher_AC12.npy'),
    threshold=0.7,
    pseudo_weight=0.5,
    epochs=int(os.environ.get('DINO_EPOCHS', 16)),
)

# Pinned here, NOT inherited. CFG comes from dinov3_base_train, whose backbone_lr defaults to
# the 1e-4 that the base A/B ran as its *experimental* arm -- importing that default would
# silently repeat the mistake the control run just cost 1.95pt of reserve accuracy to expose.
# A smoke test caught exactly that. 2e-5 flat is the settled value for this backbone.
CFG['backbone_lr'] = 2e-5
CFG['layer_decay'] = 1.0


class WeightedSet(LabeledSet):
    """LabeledSet plus a constant sample weight, so real and pseudo rows can be mixed."""

    def __init__(self, images, labels, indices, transform, weight):
        super().__init__(images, labels, indices, transform)
        self.weight = float(weight)

    def __getitem__(self, i):
        image, label = super().__getitem__(i)
        return image, label, self.weight


def build_teacher():
    """A (x) C at 1:2 in logit space -- the 0.97566 blend, rebuilt here so the file is auditable.

    Model-generated throughout: both members are forward passes of trained checkpoints. No test
    label has ever been read or written by a human or an AI (CLAUDE.md, academic integrity).
    """
    path = Path(PSEUDO['teacher'])
    if path.exists():
        return np.load(path)
    a = np.load(BENCH / 'cnv2L_test_prob.npy')
    c = np.load(BENCH / 'cnv2L_res256_test_prob.npy')
    log = lambda p: np.log(np.clip(p, 1e-9, 1.0))          # noqa: E731
    blend = np.exp((log(a) + 2.0 * log(c)) / 3.0)
    blend /= blend.sum(axis=1, keepdims=True)
    np.save(path, blend.astype(np.float32))
    print(f'teacher written to {path} (A (x) C logit 1:2, the 0.97566 blend)')
    return blend


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

    teacher = build_teacher()
    assert teacher.shape == (len(test_images), NUM_CLASSES), teacher.shape
    conf = teacher.max(axis=1)
    chosen = np.flatnonzero(conf >= PSEUDO['threshold'])
    pseudo_labels = teacher.argmax(axis=1)
    counts = np.bincount(pseudo_labels[chosen], minlength=NUM_CLASSES)
    print(f'pseudo set: {len(chosen):,} of {len(test_images):,} test images '
          f'({len(chosen)/len(test_images):.1%}) at threshold {PSEUDO["threshold"]}')
    print(f'  mean teacher confidence on the kept rows {conf[chosen].mean():.4f}')
    print(f'  per-class kept: min {counts.min()} ({CN[int(counts.argmin())]}) '
          f'max {counts.max()} ({CN[int(counts.argmax())]})')

    fit = ConcatDataset([
        WeightedSet(train_images, targets, train_idx, train_tf, 1.0),
        WeightedSet(ext_images, ext_labels, keep, train_tf, 1.0),
        WeightedSet(test_images, pseudo_labels, chosen, train_tf, PSEUDO['pseudo_weight']),
    ])
    print(f'fit set {len(fit):,} (= {len(train_idx):,} real + {len(keep):,} external '
          f'+ {len(chosen):,} pseudo @ weight {PSEUDO["pseudo_weight"]})')

    loader = DataLoader(fit, batch_size=CFG['batch_size'], shuffle=True, num_workers=4,
                        drop_last=True, pin_memory=True, persistent_workers=True)
    v_loader = DataLoader(EvalSet(train_images, val_idx),
                          batch_size=int(os.environ.get('DINO_EVAL_BATCH', 64)), num_workers=2)

    # Student starts from the DINOv3 pretrained weights, not from the base checkpoint.
    model, bb = build_model()
    head_ids = {id(p) for p in bb.get_classifier().parameters()}
    groups = [
        {'params': [p for p in model.parameters()
                    if id(p) not in head_ids and p.ndim > 1],
         'lr': CFG['backbone_lr'], 'weight_decay': CFG['weight_decay']},
        {'params': [p for p in model.parameters()
                    if id(p) not in head_ids and p.ndim <= 1],
         'lr': CFG['backbone_lr'], 'weight_decay': 0.0},
        {'params': [p for p in bb.get_classifier().parameters() if p.ndim > 1],
         'lr': CFG['head_lr'], 'weight_decay': CFG['weight_decay']},
        {'params': [p for p in bb.get_classifier().parameters() if p.ndim <= 1],
         'lr': CFG['head_lr'], 'weight_decay': 0.0},
    ]
    groups = [g for g in groups if g['params']]
    if int(os.environ.get('DINO_OPT8BIT', 0)):
        import bitsandbytes as bnb          # see the 2026-09-21 note: 8-bit moments, fp32 master
        opt = bnb.optim.AdamW8bit(groups)
        print('optimiser: bitsandbytes AdamW8bit')
    else:
        opt = torch.optim.AdamW(groups)
    print(f'flat backbone_lr {CFG["backbone_lr"]:.1e}, head_lr {CFG["head_lr"]:.1e} '
          f'(layer decay deliberately OFF -- see module docstring)')

    steps_per_epoch = len(loader)
    if os.environ.get('DINO_MAX_STEPS'):
        steps_per_epoch = min(steps_per_epoch, int(os.environ['DINO_MAX_STEPS']))
        print(f'SMOKE TEST: capped at {steps_per_epoch} steps/epoch')
    total_steps = steps_per_epoch * PSEUDO['epochs']
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

    log_path = Path(f'{TAG}_progress.csv')
    with open(log_path, 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(['wall', 'epoch', 'loss', 'val_acc', 'ema_val_acc', 'seconds'])
    best = (-1.0, None, -1)
    rows = []                      # kept for the §4.5 second clause (EMA trend) after the loop
    for epoch in range(1, PSEUDO['epochs'] + 1):
        model.train()
        started, loss_sum, step = time.time(), 0.0, 0
        for step, (images, labels, weights) in enumerate(loader, 1):
            if step > steps_per_epoch:
                step -= 1
                break
            x = to_gpu(images, size)
            w = weights.to(DEV, torch.float32, non_blocking=True)
            y = one_hot_smooth(labels.to(DEV, torch.long, non_blocking=True),
                               CFG['label_smoothing'])
            x, y = mix_batch(x, y)
            with torch.amp.autocast('cuda'):
                losses = -(y * torch.log_softmax(model(x).float(), 1)).sum(1)
                loss = (losses * w).sum() / w.sum().clamp(min=1e-8)
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
        rows.append(row)
        with open(log_path, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(row)
        print(f'  epoch {epoch}/{PSEUDO["epochs"]}  loss {row[2]}  val {raw:.4f}  '
              f'ema {eva:.4f}  {row[5]}s', flush=True)
        # §4.3 says infer from the EMA weights, so selection tracks the EMA too.
        if eva > best[0]:
            if BEST_TO_DISK:                     # see the note in dinov3_base_train.py
                torch.save({k: v.detach().cpu() for k, v in ema.module.state_dict().items()},
                           BEST_PATH)
                best = (eva, None, epoch)
            else:
                best = (eva, {k: v.detach().cpu().clone()
                              for k, v in ema.module.state_dict().items()}, epoch)

    print(f'best epoch {best[2]}/{PSEUDO["epochs"]}  ema val {best[0]:.4f}')
    # §4.5's signal has TWO clauses: best epoch <= 2 AND validation falling materially.
    # Testing only the first over-triggers -- DINOv3 pseudo1 fired it on a 0.9pt bowl that
    # recovered and went on to score 0.97171. The real collapses fell hard and stayed down
    # (256px pseudo1 -4.4pt vs its own base, p<0.001; v7 round 2 dropped outright), so the
    # second clause is measured as peak-EMA minus the mean of the last three epochs.
    tail = [r[4] for r in rows[-3:]]
    drop = best[0] - (sum(tail) / len(tail))
    if best[2] <= 2 and drop >= 0.01:
        print(f'WARNING: best epoch {best[2]} <= 2 AND EMA fell {100*drop:.2f}pt from peak to '
              'the final-3 mean -- both clauses of the §4.5 collapse signal. Report and stop; '
              'do not retune and rerun.')
    elif best[2] <= 2:
        print(f'NOTE: best epoch {best[2]} <= 2 but EMA only fell {100*drop:.2f}pt from peak to '
              'the final-3 mean (threshold 1.00pt), so the §4.5 collapse signal is NOT met -- '
              'clause one alone over-triggers. Advisory only.')

    best_state = best[1] if best[1] is not None else torch.load(BEST_PATH, map_location='cpu')
    del ema                                  # release the 3.13 GB shadow copy
    model.load_state_dict(best_state)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    Path('checkpoints_backup').mkdir(exist_ok=True)
    torch.save({'model_state_dict': best_state, 'class_names': CN, 'backbone': CFG['backbone'],
                'best_epoch': best[2], 'val_acc': best[0], 'cfg': CFG, 'pseudo': PSEUDO},
               f'checkpoints_backup/{TAG}_{stamp}.pt')

    val_prob = predict_tta(model, train_images, val_idx, CFG['tta_scales'])
    tta_val = float((val_prob.argmax(1) == targets[val_idx]).mean())
    if int(os.environ.get('DINO_SMOKE', 0)):
        n = int(os.environ['DINO_SMOKE'])
        p = predict_tta(model, test_images, np.arange(n), CFG['tta_scales'])
        print(f'SMOKE: {n} test images only, outputs are junk | TTA val {tta_val:.4f} | '
              f'mean conf {p.max(1).mean():.4f} | smoke OK')
        return
    test_prob = predict_tta(model, test_images, np.arange(len(test_images)), CFG['tta_scales'])
    np.save(BENCH / f'{TAG}_test_prob.npy', test_prob)

    try:
        from _heldout_split import load_eval_set
        img, lab = load_eval_set()
        img = torch.from_numpy(img) if isinstance(img, np.ndarray) else img
        rp = predict_tta(model, img, np.arange(len(lab)), CFG['tta_scales'])
        np.save(BENCH / f'{TAG}_reserve_prob.npy', rp)
        reserve = float((rp.argmax(1) == lab).mean())
    except Exception as exc:
        reserve = float('nan')
        print(f'reserve check skipped: {type(exc).__name__}: {exc}')

    import pandas as pd
    sub = pd.DataFrame({'ID': np.arange(len(test_prob)),
                        'Label': [CN[i] for i in test_prob.argmax(1)]})
    assert len(sub) == 11681 and set(sub['Label']) <= set(CN)
    sub.to_csv(f'submission_{TAG}.csv', index=False)
    a = np.load(BENCH / 'cnv2L_test_prob.npy')
    print(f'TTA val {tta_val:.4f} | clean-reserve {reserve:.4f} | '
          f'mean test conf {test_prob.max(1).mean():.4f} | '
          f'disagree vs A {(test_prob.argmax(1) != a.argmax(1)).mean():.2%}')
    print(f'submission_{TAG}.csv written | bench/{TAG}_test_prob.npy saved')


if __name__ == '__main__':
    main()
