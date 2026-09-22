"""
STEP 2 — 冻结特征提取。约 10 分钟 GPU，只跑一次。

为什么值得做：
  现在每试一个想法要重训 25 分钟起，一晚上试 2 次。
  把 backbone 冻住、只提一次特征存到磁盘之后，
  线性探针 / kNN / 集成 / 配比搜索全都在 CPU 上几秒钟跑完。
  一晚上能试 100 次。

而且冻结特征顺带解决了我们发现的一个问题：
  CLIP 在几亿张网络图上预训练，本来就见过大量插画和素描；
  用 9,466 张照片去微调它，等于把那部分能力覆盖掉。
  实测：CLIP 微调后单模型 0.916，比没微调过 CLIP 的 tiny(0.922) 还差。
  冻结就不会破坏它。

用法：
    python step2_extract_features.py                       # 默认 CLIP ConvNeXt
    python step2_extract_features.py --model <timm名字>     # 换 backbone
    python step2_extract_features.py --list                # 看有哪些 CLIP 模型可用

输入：bench/*.npy（由 step0_dump_arrays.py 生成）
输出：bench/feat_<模型简称>_{train,test,val_sketch}.npy
"""

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

BENCH = Path('bench')
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@torch.no_grad()
def extract(images: np.ndarray, model, size: int, batch_size: int,
            device: torch.device, mean, std, label: str) -> np.ndarray:
    """(N,3,64,64) uint8  ->  (N,D) float32，每行已 L2 归一化。

    预处理必须跟训练时一致：GPU 上 bicubic 升采样到 size，然后按 backbone
    自己的 mean/std 归一化。timm 的不同 checkpoint 用不同的统计量
    （CLIP 系用的不是 ImageNet 的那一组），所以 mean/std 从模型配置读，不写死。
    """
    mean_t = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, 3, 1, 1)
    use_amp = device.type == 'cuda'
    chunks = []
    started = time.time()

    for start in range(0, len(images), batch_size):
        batch = torch.from_numpy(np.ascontiguousarray(images[start:start + batch_size]))
        batch = batch.to(device, non_blocking=True).float().div_(255.)
        batch = F.interpolate(batch, size=size, mode='bicubic', align_corners=False)
        batch = (batch - mean_t) / std_t
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            features = model(batch)
        # 归一化成单位长度：之后点积就直接是余弦相似度，kNN 和线性探针都受益
        chunks.append(F.normalize(features.float(), dim=1).cpu())

        done = min(start + batch_size, len(images))
        if start % (batch_size * 20) == 0 or done == len(images):
            elapsed = time.time() - started
            rate = done / max(elapsed, 1e-6)
            print(f'  {label}: {done:>6,}/{len(images):,}  '
                  f'{rate:5.0f} img/s  剩余 ~{(len(images)-done)/max(rate,1e-6):4.0f}s',
                  flush=True)

    return torch.cat(chunks).numpy().astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='convnext_base.clip_laion2b_augreg_ft_in12k_in1k')
    parser.add_argument('--size', type=int, default=192,
                        help='跟训练时的 image_size 一致')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='冻结推理没有梯度，比训练能开大；10GB 显存 base 模型给 32')
    parser.add_argument('--extra', default=None,
                        help='额外的 uint8 数组，例如外部数据 bench/external_images.npy')
    parser.add_argument('--list', action='store_true', help='列出可用的 CLIP backbone 后退出')
    arguments = parser.parse_args()

    import timm

    if arguments.list:
        print('可用的 CLIP 预训练 backbone：')
        for name in timm.list_models('*clip*', pretrained=True):
            if any(k in name for k in ('convnext', 'vit_base', 'vit_large')):
                print(' ', name)
        return 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}   model: {arguments.model}')

    # num_classes=0 砍掉分类头，模型直接输出池化后的特征向量
    model = timm.create_model(arguments.model, pretrained=True, num_classes=0)
    model.eval().to(device)

    # 从模型自己的配置读预处理统计量，不要假设是 ImageNet 那一组
    config = getattr(model, 'pretrained_cfg', {}) or {}
    mean = config.get('mean', IMAGENET_MEAN)
    std = config.get('std', IMAGENET_STD)
    print(f'预处理 mean={tuple(round(v,3) for v in mean)} '
          f'std={tuple(round(v,3) for v in std)}')

    jobs = [
        ('train',      BENCH / 'train_images.npy'),
        ('test',       BENCH / 'test_images.npy'),
        ('val_sketch', BENCH / 'val_sketch_images.npy'),
    ]
    if arguments.extra:
        jobs.append(('extra', Path(arguments.extra)))

    missing = [str(p) for _, p in jobs if not p.exists()]
    if missing:
        print('\n缺少输入文件：')
        for path in missing:
            print(' ', path)
        print('\n先在 notebook 里跑 step0_dump_arrays.py 那一格。')
        return 1

    tag = arguments.model.split('.')[0].replace('/', '_')
    for label, path in jobs:
        images = np.load(path, mmap_mode='r')
        if images.ndim != 4 or images.shape[1] != 3:
            print(f'{path} 形状是 {images.shape}，期望 (N,3,64,64)')
            return 1
        print(f'\n{label}: {images.shape}')
        features = extract(images, model, arguments.size, arguments.batch_size,
                           device, mean, std, label)
        output = BENCH / f'feat_{tag}_{label}.npy'
        np.save(output, features)
        print(f'  -> {output}  {features.shape}')

    del model
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    print(f'\n完成。下一步：python step3_probe.py --tag {tag}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
