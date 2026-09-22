"""Functional check of the Web-SSL -> timm weight conversion, before any training.

`transformers` is not installed, so the converted model cannot be compared against a reference
forward pass. Tensor shapes do not catch the errors that matter (q/k/v order, SwiGLU gate order).
A frozen-feature k-NN classifier does: a correctly loaded self-supervised trunk separates these
20 classes well, and a mis-wired one does not.

Three frozen trunks, identical protocol (196px, mean patch-token pooling + final norm, cosine
20-NN with the DINO temperature-weighted vote, real training images -> validation images):
    Web-SSL, converted as in webssl_timm.py
    Web-SSL, SwiGLU gate order flipped         (negative control)
    DINOv2 ViT-L/14 reg4 via timm              (reference: a known-good load of the same recipe)

Rule, written before running: the conversion passes if Web-SSL beats the flipped control by at
least 10pt AND lands within 10pt of the DINOv2 reference. Labelled training/validation images only.
"""
import sys

import numpy as np
import timm
import torch
import torch.nn.functional as F

from dinov3_base_train import BENCH, eval_tf, to_gpu
from webssl_timm import create_webssl

REPO = 'facebook/webssl-dino300m-full2b-224'
SIZE, K, TEMP = 196, 20, 0.07


@torch.inference_mode()
def features(bb, images, indices, chunk=64):
    bb = bb.cuda().eval()
    out = []
    for i in range(0, len(indices), chunk):
        x = torch.stack([eval_tf(images[int(j)]) for j in indices[i:i + chunk]])
        with torch.amp.autocast('cuda'):
            f = bb(to_gpu(x, SIZE).contiguous()).float()
        out.append(F.normalize(f, dim=1))
    bb.cpu()
    return torch.cat(out)


def knn_accuracy(train_f, train_y, val_f, val_y):
    sims = val_f @ train_f.T
    top, idx = sims.topk(K, dim=1)
    votes = torch.zeros(len(val_f), 20, device=val_f.device)
    votes.scatter_add_(1, train_y[idx], (top / TEMP).exp())
    return 100 * (votes.argmax(1) == val_y).float().mean().item()


def main():
    images = torch.from_numpy(np.load(BENCH / 'train_images.npy'))
    targets = np.load(BENCH / 'targets.npy')
    val_idx = np.load(BENCH / 'validation_indices.npy')
    train_idx = np.setdiff1d(np.arange(len(targets)), val_idx)
    ty = torch.from_numpy(targets[train_idx]).cuda()
    vy = torch.from_numpy(targets[val_idx]).cuda()

    trunks = {
        'Web-SSL (converted)': lambda: create_webssl(REPO, 0),
        'Web-SSL gate flipped (control)': lambda: create_webssl(REPO, 0, gate_last=True),
        # fc_norm=False keeps the pretrained final norm where the checkpoint has it (before
        # pooling). With fc_norm on, num_classes=0 makes timm load strictly and it refuses the
        # checkpoint's `norm` keys; the reference only needs a correctly loaded trunk.
        'DINOv2 ViT-L reg4 (reference)': lambda: timm.create_model(
            'vit_large_patch14_reg4_dinov2.lvd142m', pretrained=True, num_classes=0,
            global_pool='avg', fc_norm=False, dynamic_img_size=True),
    }
    acc = {}
    for name, make in trunks.items():
        bb = make()
        acc[name] = knn_accuracy(features(bb, images, train_idx), ty,
                                 features(bb, images, val_idx), vy)
        print(f'{name:32s} frozen 20-NN val {acc[name]:.2f}%', flush=True)
        del bb
        torch.cuda.empty_cache()

    ours, flipped, ref = acc.values()
    ok = ours - flipped >= 10 and ref - ours <= 10
    print(f'margin over flipped control {ours - flipped:+.2f}pt, gap to DINOv2 {ours - ref:+.2f}pt '
          f'-> conversion {"PASSES" if ok else "FAILS"}')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
