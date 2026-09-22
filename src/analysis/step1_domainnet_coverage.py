"""
STEP 1 — DomainNet 覆盖率检查。不下载任何图片，只下载几个几百 KB 的清单文件。

问题：DomainNet 有 345 个类，你的比赛有 20 个类。到底能对上几个？
      每个类各有多少张渲染图？

这一步的产出是一个数字（覆盖率），它决定后面整个计划的形状：
  - 18/20 以上  -> 直接走 DomainNet
  - 10~17/20    -> DomainNet 打底，缺的类另外爬
  - 10 以下     -> 换数据源

用法：
    python step1_domainnet_coverage.py
    python step1_domainnet_coverage.py --local-dir D:\\data\\domainnet   # 已经下好了

不下载图片，所以跑这一步【不构成】使用外部数据，先查清楚再决定也来得及。
"""

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from urllib.request import urlopen

# 你的 20 个类
TARGET_CLASSES = [
    'birds', 'bottles', 'breads', 'butterflies', 'cakes', 'cats', 'chickens',
    'cows', 'dogs', 'ducks', 'elephants', 'fishes', 'handguns', 'horses',
    'lions', 'lipsticks', 'seals', 'snakes', 'spiders', 'vases',
]

# 手工同义词表。DomainNet 用单数、有时用别的词。
# 这里【只是候选】，脚本会把所有匹配到的 DomainNet 类名打印出来让你核对。
# 宁可多列几个候选，也不要漏 —— 最后由你肉眼确认。
SYNONYMS = {
    'birds':       ['bird', 'parrot', 'owl', 'penguin', 'flamingo', 'swan', 'pigeon', 'peanut'],
    'bottles':     ['wine_bottle', 'bottlecap', 'water_bottle', 'bottle'],
    'breads':      ['bread', 'sandwich', 'bagel', 'pretzel', 'hamburger', 'hot_dog'],
    'butterflies': ['butterfly', 'moth'],
    'cakes':       ['cake', 'birthday_cake', 'cookie', 'donut', 'cupcake', 'pie'],
    'cats':        ['cat', 'lion', 'tiger'],          # lion 单列，见下
    'chickens':    ['chicken', 'rooster', 'hen', 'duck'],
    'cows':        ['cow', 'bull', 'zebra', 'horse'],
    'dogs':        ['dog', 'puppy'],
    'ducks':       ['duck', 'goose', 'swan'],
    'elephants':   ['elephant'],
    'fishes':      ['fish', 'shark', 'dolphin', 'whale', 'goldfish'],
    'handguns':    ['gun', 'pistol', 'rifle', 'handgun'],
    'horses':      ['horse', 'zebra'],
    'lions':       ['lion'],
    'lipsticks':   ['lipstick'],
    'seals':       ['sea_turtle', 'dolphin', 'whale', 'penguin', 'walrus', 'seal'],
    'snakes':      ['snake'],
    'spiders':     ['spider', 'scorpion'],
    'vases':       ['vase', 'flower', 'pot'],
}

DOMAINS = ['sketch', 'clipart', 'painting']       # quickdraw 太抽象，先不要
BASE_URLS = [
    'http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/{d}_train.txt',
    'http://csr.bu.edu/ftp/visda/2019/multi-source/txt/{d}_train.txt',
    'http://ai.bu.edu/M3SDA/txt/{d}_train.txt',
]


def fetch_listing(domain: str) -> list[str] | None:
    """下载一个域的清单文件。每行形如  sketch/dog/sketch_123_000001.jpg 0"""
    for template in BASE_URLS:
        url = template.format(d=domain)
        try:
            with urlopen(url, timeout=30) as response:
                text = response.read().decode('utf-8', errors='replace')
            if text.count('\n') > 100:
                print(f'  {domain}: 从 {url} 取到 {text.count(chr(10)):,} 行')
                return text.splitlines()
        except Exception as error:
            print(f'  {domain}: {url} 失败 ({type(error).__name__})')
    return None


def listing_from_local(root: Path, domain: str) -> list[str] | None:
    """本地已经解压好的话，直接遍历目录，不用下载。"""
    directory = root / domain
    if not directory.is_dir():
        return None
    lines = []
    for class_dir in sorted(p for p in directory.iterdir() if p.is_dir()):
        for image in class_dir.iterdir():
            if image.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
                lines.append(f'{domain}/{class_dir.name}/{image.name} 0')
    print(f'  {domain}: 本地目录 {len(lines):,} 张')
    return lines


def class_counts(lines: list[str]) -> Counter:
    """从清单里数出每个类有多少张图。路径的第二段就是类名。"""
    counts = Counter()
    for line in lines:
        path = line.split()[0] if line.strip() else ''
        parts = path.split('/')
        if len(parts) >= 3:
            counts[parts[1]] += 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--local-dir', default=None,
                        help='已解压的 DomainNet 根目录；不给就从网上取清单')
    parser.add_argument('--min-images', type=int, default=100,
                        help='一个类至少要有这么多张才算“有货”')
    arguments = parser.parse_args()

    root = Path(arguments.local_dir) if arguments.local_dir else None

    print('=== 取各域清单 ===')
    per_domain: dict[str, Counter] = {}
    for domain in DOMAINS:
        lines = listing_from_local(root, domain) if root else None
        if lines is None:
            lines = fetch_listing(domain)
        if lines is None:
            print(f'  {domain}: 拿不到。手动去 http://ai.bu.edu/DomainNet/ 下载后用 --local-dir')
            continue
        per_domain[domain] = class_counts(lines)

    if not per_domain:
        print('\n一个域都没拿到，无法继续。')
        return 1

    all_names = sorted(set().union(*(c.keys() for c in per_domain.values())))
    print(f'\nDomainNet 类别总数: {len(all_names)}')

    # --- 匹配 ---
    print('\n=== 覆盖率 ===')
    print(f"{'比赛类别':<14} {'匹配到的 DomainNet 类别':<44} " +
          ' '.join(f'{d:>9}' for d in per_domain) + '     合计')
    print('-' * 110)

    matched_map: dict[str, list[str]] = {}
    covered = 0
    for target in TARGET_CLASSES:
        candidates = SYNONYMS.get(target, []) + [target, target.rstrip('s')]
        hits = []
        for name in all_names:
            normalised = re.sub(r'[^a-z_]', '', name.lower())
            for candidate in candidates:
                # 完整词匹配，避免 'cat' 命中 'caterpillar'
                if normalised == candidate or normalised.startswith(candidate + '_') \
                        or normalised.endswith('_' + candidate):
                    hits.append(name)
                    break
        hits = sorted(set(hits))
        per_domain_counts = [sum(per_domain[d][h] for h in hits) for d in per_domain]
        total = sum(per_domain_counts)
        if total >= arguments.min_images:
            covered += 1
        flag = ' ' if total >= arguments.min_images else '!'
        matched_map[target] = hits
        shown = ', '.join(hits)[:42] or '(无)'
        print(f'{flag}{target:<13} {shown:<44} ' +
              ' '.join(f'{c:>9,}' for c in per_domain_counts) + f' {total:>9,}')

    print('-' * 110)
    print(f'\n覆盖率: {covered}/{len(TARGET_CLASSES)} 个类有至少 '
          f'{arguments.min_images} 张渲染图')

    # --- 把匹配表写出来，下一步要用，而且必须由你人工改过 ---
    output = Path('bench/domainnet_mapping.py')
    output.parent.mkdir(exist_ok=True)
    with output.open('w', encoding='utf-8') as handle:
        handle.write('# 自动生成的候选映射。\n')
        handle.write('# 【必须人工检查】：同义词表是猜的，一定有错。\n')
        handle.write('# 典型错误：cats 会把 lion/tiger 收进来，cows 会把 horse 收进来，\n')
        handle.write('# 而 lions 和 horses 本身就是独立类别 —— 重复归类会直接制造标签噪声。\n')
        handle.write('MAPPING = {\n')
        for target, hits in matched_map.items():
            handle.write(f'    {target!r}: {hits!r},\n')
        handle.write('}\n')
    print(f'候选映射已写入 {output}')
    print('\n下一步：打开这个文件，手工删掉错误的匹配。')
    print("特别注意 cats/lions、cows/horses、ducks/chickens/birds 这几组互相污染。")
    return 0


if __name__ == '__main__':
    sys.exit(main())
