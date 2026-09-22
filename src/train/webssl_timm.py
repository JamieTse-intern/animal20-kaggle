"""Load Meta's Web-SSL DINO ViT-300M checkpoints into a timm VisionTransformer.

Web-SSL ("Scaling Language-Free Visual Representation Learning", Fan et al. 2025) is the DINOv2
self-supervised recipe trained on 2B MetaCLIP web images, with no language supervision. Meta
publishes it only in Hugging Face `transformers` Dinov2Model format, which is not installed here.
The architecture is nonetheless one timm already implements, so the weights are renamed into timm
and the model trains through exactly the code path the DINOv2 ViT-L runs used (dynamic image
size for the 168/196/224 TTA, mean patch-token pooling, gradient checkpointing).

Architecture, from the checkpoint's config.json and its tensor shapes (the model card's
"1536 width, 40 depth" is a copy-paste error -- 1.21 GB is 303.7M float32 parameters):
    ViT-L/14, width 1024, depth 24, 16 heads, CLS token, no register tokens, LayerScale,
    224px position grid (16x16 + CLS = 257), SwiGLU MLP packed 1024 -> 5472 -> 2736 -> 1024.

HF's Dinov2SwiGLUFFN computes `silu(x1) * x2` on the two halves of weights_in, which is timm's
GluMlp with gate_last=False (timm's SwiGLUPacked). webssl_convert_check.py confirms it
functionally against the opposite gate order.

The pretrained final LayerNorm is DISCARDED by default and `fc_norm` (applied after pooling) starts
at weight 1 / bias 0. That is what actually happened to the DINOv2 ViT-L member: with
global_pool='avg' timm builds `fc_norm` instead of `norm`, and with num_classes=20 it loads the
checkpoint non-strictly, so the checkpoint's `norm` keys were silently dropped. Verified on the
saved dinov2vitl_base checkpoint: fc_norm.weight mean 0.9987, std 0.0007, correlation 0.07 with
the pretrained norm (std 0.59). Matching that keeps the comparison with V2 to the pretrained
weights and nothing else. `pretrained_final_norm=True` loads it into fc_norm instead.

Model weights only: nothing here reads the test set.
"""
from functools import partial

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from timm.layers import GluMlp
from timm.models.vision_transformer import VisionTransformer

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def is_webssl(name):
    return name.startswith('facebook/webssl-')


def convert_hf_dinov2(hf, depth=24):
    """Rename a transformers Dinov2Model state dict (SwiGLU, no registers) to timm names."""
    # 6 top-level tensors + 18 per block + the masked-modelling token, which only the
    # pretraining objective uses. Anything else means this is not the architecture assumed here.
    assert len(hf) == 6 + 18 * depth + 1 and 'embeddings.mask_token' in hf, len(hf)
    out = {
        'cls_token': hf['embeddings.cls_token'],
        'pos_embed': hf['embeddings.position_embeddings'],
        'patch_embed.proj.weight': hf['embeddings.patch_embeddings.projection.weight'],
        'patch_embed.proj.bias': hf['embeddings.patch_embeddings.projection.bias'],
        'fc_norm.weight': hf['layernorm.weight'],
        'fc_norm.bias': hf['layernorm.bias'],
    }
    for i in range(depth):
        h, t = f'encoder.layer.{i}.', f'blocks.{i}.'
        att = h + 'attention.attention.'
        # timm's Attention splits one qkv projection as [q, k, v] along the output dimension.
        for kind in ('weight', 'bias'):
            out[t + 'attn.qkv.' + kind] = torch.cat(
                [hf[att + n + '.' + kind] for n in ('query', 'key', 'value')])
            out[t + 'attn.proj.' + kind] = hf[h + 'attention.output.dense.' + kind]
            out[t + 'norm1.' + kind] = hf[h + 'norm1.' + kind]
            out[t + 'norm2.' + kind] = hf[h + 'norm2.' + kind]
            out[t + 'mlp.fc1.' + kind] = hf[h + 'mlp.weights_in.' + kind]
            out[t + 'mlp.fc2.' + kind] = hf[h + 'mlp.weights_out.' + kind]
        out[t + 'ls1.gamma'] = hf[h + 'layer_scale1.lambda1']
        out[t + 'ls2.gamma'] = hf[h + 'layer_scale2.lambda1']
    return out


def create_webssl(repo_id, num_classes, global_pool='avg', dynamic_img_size=True,
                  gate_last=False, pretrained_final_norm=False):
    """Build the timm model and load the converted weights.

    `gate_last` exists only for the conversion check: the opposite gate order must give clearly
    worse frozen features, otherwise the check cannot tell a right conversion from a wrong one.
    """
    assert global_pool == 'avg', 'fc_norm handling below assumes mean pooling'
    bb = VisionTransformer(
        img_size=224, patch_size=14, embed_dim=1024, depth=24, num_heads=16, qkv_bias=True,
        init_values=1.0, mlp_ratio=5472 / 1024,
        mlp_layer=partial(GluMlp, act_layer=nn.SiLU, gate_last=gate_last), act_layer=nn.SiLU,
        class_token=True, reg_tokens=0, global_pool=global_pool, num_classes=num_classes,
        dynamic_img_size=dynamic_img_size)
    state = convert_hf_dinov2(load_file(hf_hub_download(repo_id, 'model.safetensors')))
    fresh = {'head.weight', 'head.bias'} if num_classes else set()
    if not pretrained_final_norm:
        del state['fc_norm.weight'], state['fc_norm.bias']
        fresh |= {'fc_norm.weight', 'fc_norm.bias'}
    missing, unexpected = bb.load_state_dict(state, strict=False)
    assert set(missing) == fresh and not unexpected, (missing, unexpected)
    bb.pretrained_cfg = dict(mean=IMAGENET_MEAN, std=IMAGENET_STD, input_size=(3, 224, 224),
                             hf_hub_id=repo_id, license='cc-by-nc-4.0')
    return bb
