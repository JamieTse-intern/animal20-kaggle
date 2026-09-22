"""Before training: confirm timm loads every pretrained EVA-02 tensor except the 1000-class head.

The DINOv2 member silently lost its pretrained final norm to a non-strict load (CLAUDE.md,
2026-09-15). timm always loads non-strictly when num_classes differs from the checkpoint, so this
compares the checkpoint file against the built model tensor by tensor. Model weights only.
"""
import sys

import timm
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

NAME = 'eva02_large_patch14_448.mim_m38m_ft_in22k_in1k'
HEAD = {'head.weight', 'head.bias'}


def main():
    ck = load_file(hf_hub_download('timm/' + NAME, 'model.safetensors'))
    model = timm.create_model(NAME, pretrained=True, num_classes=20, dynamic_img_size=True)
    sd = model.state_dict()
    dropped = sorted(set(ck) - set(sd))
    fresh = sorted(set(sd) - set(ck))
    shared = [k for k in ck if k in sd and k not in HEAD]
    differ = [k for k in shared
              if ck[k].shape != sd[k].shape or not torch.equal(ck[k].float(), sd[k].float())]
    print(f'checkpoint tensors {len(ck)}, model tensors {len(sd)}, identical {len(shared) - len(differ)}')
    print(f'in checkpoint but not in model: {dropped}')
    print(f'in model but not in checkpoint: {fresh}')
    print(f'loaded but different: {differ[:10]}{" ..." if len(differ) > 10 else ""}')
    print(f"fc_norm present and loaded: {'fc_norm.weight' in shared and 'fc_norm.weight' not in differ}")
    ok = set(dropped) <= HEAD and set(fresh) <= HEAD and not differ
    print(f'load check {"PASSES" if ok else "FAILS"}')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
