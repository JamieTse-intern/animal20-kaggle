"""
prepare_external.py — 把外部图片转成跟测试集同一个域的训练数据。

输入：一个目录，里面是 <外部类名>/*.jpg 的子文件夹结构
      （DomainNet、ImageNet-Sketch、自己爬的，任何来源都行）
      加一个 JSON 映射：外部类名 -> 你的 20 个类之一

输出：bench/external_images.npy   uint8 (N,3,64,64)，跟 train_images 完全同格式
      bench/external_labels.npy   int64 (N,)
      bench/external_contact.png  抽样拼图，必须肉眼看一眼
      bench/external_report.txt   每类数量 + 域匹配自检

为什么必须走这个脚本而不是直接喂原图：
  测试集是 64x64、JPEG 质量 75、4:2:0 子采样。外部图是几百像素的干净 PNG。
  v1 版本 val 98.9% 而 Kaggle 只有 0.74，就是因为训练图干净、测试图有压缩痕迹。
  这个脚本复刻 notebook 里那条退化管线（squash + LANCZOS -> 64px -> 测试集自己的
  量化表重新编码），并在最后用拉普拉斯方差和径向频谱验证两边真的对上了。

用法：
    python prepare_external.py --source-dir D:\\data\\domainnet\\sketch ^
                               --mapping bench\\mapping.json ^
                               --per-class 400
    python prepare_external.py --source-dir ... --mapping ... --append   # 追加第二个域
"""

import argparse
import io
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, JpegImagePlugin

BENCH = Path('bench')
IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
CLASS_NAMES = [
    'birds', 'bottles', 'breads', 'butterflies', 'cakes', 'cats', 'chickens',
    'cows', 'dogs', 'ducks', 'elephants', 'fishes', 'handguns', 'horses',
    'lions', 'lipsticks', 'seals', 'snakes', 'spiders', 'vases',
]
CLASS_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}

_GREY = np.array([0.299, 0.587, 0.114], np.float32)


# --------------------------------------------------------------------------
# 退化管线：跟 notebook cell 10 的 degrade() 逐行等价
# --------------------------------------------------------------------------
def read_encoder_fingerprint(test_dir: Path):
    """从一张真实的测试图上读出量化表和子采样模式。

    不写死数字，直接从文件读 —— 这样脚本和 notebook 永远不可能分叉。
    """
    candidates = sorted(p for p in test_dir.rglob('*') if p.suffix.lower() in {'.jpg', '.jpeg'})
    if not candidates:
        raise SystemExit(f'{test_dir} 里没有 JPEG，无法读取编码器指纹')
    with Image.open(candidates[0]) as probe:
        probe.load()
        qtables = probe.quantization
        subsampling = JpegImagePlugin.get_sampling(probe)
    standard_luma = np.array([16, 11, 10, 16, 24, 40, 51, 61], dtype=float)
    scale = float(np.median(np.asarray(qtables[0][:8], dtype=float) / standard_luma)) * 100
    print(f'编码器指纹（读自 {candidates[0].name}）: '
          f'JPEG 质量 ~{(200 - scale) / 2:.0f}, 子采样码 {subsampling} (2 = 4:2:0), '
          f'{len(qtables)} 张量化表')
    return qtables, subsampling


def degrade(image: Image.Image, qtables, subsampling) -> np.ndarray:
    """squash 到方形 -> LANCZOS 缩到 64 -> 用测试集的量化表重新编码。

    v5 的核搜索测出 ('squash', 'PIL:LANCZOS') 是最贴近测试集的组合
    （lapvar 0.0398 对 0.0402，频谱距离 0.0213），所以这里写死这一对。
    squash 指直接拉成正方形，不裁剪 —— 裁剪会丢掉可能是主体的部分。
    """
    rgb = image.convert('RGB')
    resized = np.asarray(rgb.resize((64, 64), Image.LANCZOS), dtype=np.uint8).copy()
    buffer = io.BytesIO()
    Image.fromarray(resized).save(buffer, format='JPEG',
                                  qtables=qtables, subsampling=subsampling)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        decoded.load()
        return np.asarray(decoded.convert('RGB'), dtype=np.uint8).copy()


# --------------------------------------------------------------------------
# 域匹配自检
# --------------------------------------------------------------------------
def laplacian_variance(batch: np.ndarray) -> float:
    """拉普拉斯方差，衡量锐度。batch 是 (N,64,64,3) uint8。

    两个域如果这个数差很多，说明一边比另一边糊，模型会学到这个差别当捷径。
    """
    grey = (batch.astype(np.float32) / 255.0) @ _GREY
    lap = (-4 * grey[:, 1:-1, 1:-1] + grey[:, :-2, 1:-1] + grey[:, 2:, 1:-1]
           + grey[:, 1:-1, :-2] + grey[:, 1:-1, 2:])
    return float(lap.reshape(len(batch), -1).var(axis=1).mean())


def radial_spectrum(batch: np.ndarray) -> np.ndarray:
    """径向平均的对数功率谱。比 lapvar 更细：能看出高频是不是被压掉了。"""
    grey = (batch.astype(np.float32) / 255.0) @ _GREY
    grey = grey - grey.mean((1, 2), keepdims=True)
    power = np.abs(np.fft.fftshift(np.fft.fft2(grey), axes=(1, 2))) ** 2
    size = grey.shape[1]
    centre = size // 2
    rows, columns = np.mgrid[:size, :size]
    radius = np.sqrt((rows - centre) ** 2 + (columns - centre) ** 2).astype(int)
    profile = np.array([power[:, radius == k].mean() for k in range(1, centre)])
    profile = profile / profile[:8].mean()
    return np.log10(profile + 1e-12)


# --------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-dir', required=True,
                        help='外部数据根目录，下面是 <外部类名>/*.jpg')
    parser.add_argument('--mapping', required=True,
                        help='JSON: {"外部类名": "你的类名", ...}')
    parser.add_argument('--test-dir', default='data/test',
                        help='真实测试图所在目录，用来读编码器指纹')
    parser.add_argument('--reference', default='bench/test_images.npy',
                        help='用来做域匹配对比的测试图数组')
    parser.add_argument('--per-class', type=int, default=400,
                        help='每个目标类最多取多少张，防止某一类淹没其他类')
    parser.add_argument('--append', action='store_true',
                        help='追加到已有的 external_*.npy，用于合并多个域')
    parser.add_argument('--seed', type=int, default=1234)
    arguments = parser.parse_args()

    BENCH.mkdir(exist_ok=True)
    random.seed(arguments.seed)

    mapping = json.loads(Path(arguments.mapping).read_text(encoding='utf-8'))
    unknown = sorted({v for v in mapping.values()} - set(CLASS_NAMES))
    if unknown:
        raise SystemExit(f'映射里出现了不认识的目标类: {unknown}\n合法取值: {CLASS_NAMES}')

    qtables, subsampling = read_encoder_fingerprint(Path(arguments.test_dir))

    # --- 按目标类收集候选文件，然后每类下采样到 per-class ---
    source = Path(arguments.source_dir)
    buckets: dict[str, list[Path]] = defaultdict(list)
    for external_name, target in mapping.items():
        directory = source / external_name
        if not directory.is_dir():
            print(f'  跳过 {external_name}: 目录不存在')
            continue
        files = [p for p in directory.rglob('*') if p.suffix.lower() in IMG_EXT]
        buckets[target].extend(files)
        print(f'  {external_name:>22s} -> {target:<12s} {len(files):>6,} 张')

    if not buckets:
        raise SystemExit('一个文件都没找到，检查 --source-dir 和映射里的类名拼写')

    selected: list[tuple[Path, int]] = []
    for target, files in buckets.items():
        random.shuffle(files)
        keep = files[:arguments.per_class]
        selected.extend((p, CLASS_INDEX[target]) for p in keep)
    random.shuffle(selected)

    print(f'\n准备退化 {len(selected):,} 张 ...')

    # --- 退化 ---
    images = np.zeros((len(selected), 64, 64, 3), dtype=np.uint8)
    labels = np.zeros(len(selected), dtype=np.int64)
    kept = 0
    failures = 0
    for path, label in selected:
        try:
            with Image.open(path) as handle:
                images[kept] = degrade(handle, qtables, subsampling)
        except Exception:
            failures += 1
            continue
        labels[kept] = label
        kept += 1
        if kept % 500 == 0:
            print(f'  {kept:,}/{len(selected):,}', flush=True)
    images, labels = images[:kept], labels[:kept]
    print(f'完成 {kept:,} 张，失败 {failures} 张')

    # --- 域匹配自检：这是整个脚本最重要的输出 ---
    lines = []
    reference_path = Path(arguments.reference)
    if reference_path.exists():
        reference = np.load(reference_path, mmap_mode='r')[:2000]
        reference = np.ascontiguousarray(reference).transpose(0, 2, 3, 1)  # -> NHWC
        sample = images[:2000]
        lap_ext, lap_ref = laplacian_variance(sample), laplacian_variance(reference)
        spectrum_distance = float(np.abs(radial_spectrum(sample)
                                         - radial_spectrum(reference)).mean())
        lines += [
            '=== 域匹配自检 ===',
            f'拉普拉斯方差  外部 {lap_ext:.4f}   测试集 {lap_ref:.4f}   '
            f'相对差 {abs(lap_ext-lap_ref)/max(lap_ref,1e-9)*100:.1f}%',
            f'径向频谱距离  {spectrum_distance:.4f}',
            '',
            'v5 在原始训练集上量到的基准: lapvar 0.0398 vs 0.0402, 频谱距离 0.0213。',
            '外部数据要达到同一量级才算对齐。相对差超过 25% 或频谱距离超过 0.05,',
            '说明这批图跟测试集还不是一个域, 直接拿去训练会重演 v1 的 0.74。',
        ]
    else:
        lines.append(f'找不到 {reference_path}，跳过域匹配自检（先跑 step0_dump_arrays）')

    # --- 每类数量 ---
    counts = Counter(labels.tolist())
    lines += ['', '=== 每类数量 ===']
    for i, name in enumerate(CLASS_NAMES):
        lines.append(f'  {name:<14s} {counts.get(i, 0):>6,}'
                     + ('   <- 没有外部数据' if counts.get(i, 0) == 0 else ''))

    # --- 合并已有的 ---
    image_path, label_path = BENCH / 'external_images.npy', BENCH / 'external_labels.npy'
    if arguments.append and image_path.exists():
        old_labels = np.load(label_path)
        # 磁盘上存的是 (N,3,64,64)，而 images 此刻还是 (N,64,64,3)。
        # 先把旧的转回 NHWC 再拼，最后统一转成 NCHW 存盘。
        old_images = np.load(image_path).transpose(0, 2, 3, 1)
        images = np.concatenate([old_images, images])
        labels = np.concatenate([old_labels, labels])
        lines.append(f'\n追加模式: 与已有 {len(old_images):,} 张合并 -> {len(images):,} 张')

    # 存成 (N,3,64,64)，跟 train_images 完全同格式
    np.save(image_path, np.ascontiguousarray(images.transpose(0, 3, 1, 2)))
    np.save(label_path, labels)

    report = '\n'.join(lines)
    (BENCH / 'external_report.txt').write_text(report, encoding='utf-8')
    print('\n' + report)

    # --- 抽样拼图：必须肉眼看 ---
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        picks = np.random.default_rng(0).choice(len(images), size=min(40, len(images)),
                                                replace=False)
        figure, axes = plt.subplots(5, 8, figsize=(16, 11))
        for axis, index in zip(axes.ravel(), picks):
            axis.imshow(images[index])
            axis.set_title(CLASS_NAMES[labels[index]], fontsize=8)
            axis.axis('off')
        for axis in axes.ravel()[len(picks):]:
            axis.axis('off')
        figure.suptitle('external data after degradation', fontsize=14)
        figure.tight_layout()
        figure.savefig(BENCH / 'external_contact.png', dpi=90)
        print(f'\n抽样拼图 -> {BENCH / "external_contact.png"}')
        print('务必打开看一眼：标签对不对，画质跟测试图像不像。')
    except Exception as error:
        print(f'拼图生成失败（不影响数据）: {error}')

    print(f'\n写出 {image_path}  {np.load(image_path, mmap_mode="r").shape}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
