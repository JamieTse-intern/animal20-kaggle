"""
STEP 3 — 在冻结特征上跑线性探针和 kNN。几秒钟，纯 CPU。

这是证伪实验。三个数字决定后面的方向：

  普通验证准确率   —— 冻结特征够不够强？跟微调的 98.1% 比
  素描验证准确率   —— 渲染鲁棒性还在不在？微调版在这里会很惨
  两者的差距       —— 这就是那 5.88 点缺口在本地的投影

如果冻结探针的【素描准确率】明显高于微调模型，就证实了
「微调破坏了 CLIP 的渲染鲁棒性」，那么后面的路线就该整个转向。

用法：
    python step3_probe.py --tag convnext_base
    python step3_probe.py --tag convnext_base --write-submission
    python step3_probe.py --tag convnext_base --extra-labels bench/external_labels.npy
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

BENCH = Path('bench')


def load(tag: str, extra_labels: str | None):
    """读回 step0 和 step2 的产物，并检查它们互相对得上。"""
    need = {
        'train':      BENCH / f'feat_{tag}_train.npy',
        'test':       BENCH / f'feat_{tag}_test.npy',
        'val_sketch': BENCH / f'feat_{tag}_val_sketch.npy',
    }
    missing = [str(p) for p in need.values() if not p.exists()]
    if missing:
        raise SystemExit('缺少特征文件：\n  ' + '\n  '.join(missing) +
                         f'\n先跑 step2_extract_features.py')

    data = {k: np.load(v) for k, v in need.items()}
    data['targets'] = np.load(BENCH / 'targets.npy')
    data['train_idx'] = np.load(BENCH / 'train_indices.npy')
    data['val_idx'] = np.load(BENCH / 'validation_indices.npy')
    data['test_ids'] = np.load(BENCH / 'test_ids.npy')
    data['class_names'] = list(np.load(BENCH / 'class_names.npy', allow_pickle=True))

    # 外部数据（可选）
    extra_path = BENCH / f'feat_{tag}_extra.npy'
    if extra_path.exists() and extra_labels:
        data['extra'] = np.load(extra_path)
        data['extra_y'] = np.load(extra_labels)
        if len(data['extra']) != len(data['extra_y']):
            raise SystemExit(f"外部特征 {len(data['extra'])} 条，"
                             f"标签 {len(data['extra_y'])} 条，对不上")

    # 一致性检查：这些断言比事后 debug 便宜太多
    assert len(data['train']) == len(data['targets']), '训练特征数 != 标签数'
    assert len(data['val_sketch']) == len(data['val_idx']), '素描特征数 != 验证集大小'
    assert len(data['test']) == len(data['test_ids']), '测试特征数 != 测试 ID 数'
    return data


def knn_predict(train_features, train_labels, query_features,
                num_classes, k=20, temperature=0.07, chunk=2048):
    """加权 kNN。特征已 L2 归一化，所以点积就是余弦相似度。

    权重用 exp(sim/T)：相似度高的邻居话语权指数级更大。
    T=0.07 是自监督学习里的常用值，越小越只听最近的邻居的。
    """
    labels = np.asarray(train_labels)
    out = np.zeros((len(query_features), num_classes), dtype=np.float32)
    for start in range(0, len(query_features), chunk):
        block = query_features[start:start + chunk]
        similarity = block @ train_features.T                   # (chunk, N_train)
        top = np.argpartition(-similarity, kth=k, axis=1)[:, :k]  # 每行最像的 k 个
        top_sim = np.take_along_axis(similarity, top, axis=1)
        weight = np.exp(top_sim / temperature)
        for column in range(k):                                  # 把权重投到各自的类上
            np.add.at(out, (np.arange(len(block)) + start, labels[top[:, column]]),
                      weight[:, column])
    return out / np.clip(out.sum(axis=1, keepdims=True), 1e-12, None)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', required=True, help='step2 打印出来的那个 tag')
    parser.add_argument('--C', type=float, default=1.0, help='逻辑回归正则强度的倒数')
    parser.add_argument('--knn-k', type=int, default=20)
    parser.add_argument('--extra-labels', default=None,
                        help='外部数据的标签 .npy；给了才会把外部数据加进训练')
    parser.add_argument('--extra-weight', type=float, default=1.0,
                        help='外部样本的权重，<1 表示别让它们淹没真实数据')
    parser.add_argument('--write-submission', action='store_true')
    arguments = parser.parse_args()

    data = load(arguments.tag, arguments.extra_labels)
    num_classes = len(data['class_names'])
    train_idx, val_idx = data['train_idx'], data['val_idx']

    X_train = data['train'][train_idx]
    y_train = data['targets'][train_idx]
    X_val = data['train'][val_idx]
    y_val = data['targets'][val_idx]
    X_sketch = data['val_sketch']          # 跟 val_idx 同序，所以标签也是 y_val
    weights = np.ones(len(y_train), dtype=np.float32)

    if 'extra' in data:
        print(f"加入外部数据 {len(data['extra']):,} 条，权重 {arguments.extra_weight}")
        X_train = np.concatenate([X_train, data['extra']])
        y_train = np.concatenate([y_train, data['extra_y']])
        weights = np.concatenate([weights,
                                  np.full(len(data['extra']), arguments.extra_weight,
                                          dtype=np.float32)])

    print(f"训练 {len(X_train):,}  验证 {len(X_val):,}  "
          f"测试 {len(data['test']):,}  特征维度 {X_train.shape[1]}")

    # --- 线性探针 ---
    # 特征已 L2 归一化，但各维尺度仍不同；标准化让 L2 正则对每一维公平。
    print('\n=== 线性探针 ===')
    started = time.time()
    scaler = StandardScaler().fit(X_train)
    probe = LogisticRegression(C=arguments.C, max_iter=2000, n_jobs=-1)
    probe.fit(scaler.transform(X_train), y_train,
              sample_weight=weights if 'extra' in data else None)
    print(f'  训练耗时 {time.time()-started:.1f}s')

    probe_val = probe.predict_proba(scaler.transform(X_val))
    probe_sketch = probe.predict_proba(scaler.transform(X_sketch))
    probe_test = probe.predict_proba(scaler.transform(data['test']))
    val_accuracy = (probe_val.argmax(1) == y_val).mean()
    sketch_accuracy = (probe_sketch.argmax(1) == y_val).mean()
    print(f'  普通验证 {val_accuracy:.2%}   素描验证 {sketch_accuracy:.2%}   '
          f'差距 {(val_accuracy-sketch_accuracy)*100:.1f} 点')

    # --- kNN ---
    print(f'\n=== kNN (k={arguments.knn_k}) ===')
    started = time.time()
    knn_val = knn_predict(X_train, y_train, X_val, num_classes, arguments.knn_k)
    knn_sketch = knn_predict(X_train, y_train, X_sketch, num_classes, arguments.knn_k)
    knn_test = knn_predict(X_train, y_train, data['test'], num_classes, arguments.knn_k)
    knn_val_accuracy = (knn_val.argmax(1) == y_val).mean()
    knn_sketch_accuracy = (knn_sketch.argmax(1) == y_val).mean()
    print(f'  耗时 {time.time()-started:.1f}s')
    print(f'  普通验证 {knn_val_accuracy:.2%}   素描验证 {knn_sketch_accuracy:.2%}   '
          f'差距 {(knn_val_accuracy-knn_sketch_accuracy)*100:.1f} 点')

    # --- 两者平均 ---
    blend_val = (probe_val + knn_val) / 2
    blend_sketch = (probe_sketch + knn_sketch) / 2
    blend_test = (probe_test + knn_test) / 2
    print(f'\n=== 探针+kNN 平均 ===')
    print(f'  普通验证 {(blend_val.argmax(1)==y_val).mean():.2%}   '
          f'素描验证 {(blend_sketch.argmax(1)==y_val).mean():.2%}')

    print('\n=== 对照：微调路线的已知成绩 ===')
    print('  微调 convnext_tiny  普通验证 98.10%   Kaggle 0.92198')
    print('  微调 CLIP base      普通验证 98.87%   Kaggle 0.91600')
    print('  CLIP + tiny 集成                      Kaggle 0.92796')
    print('\n注意：本地验证只有 1,419 张，一个标准误 0.37 个点，')
    print('      所以 0.5 点以内的差别不要当真。要看的是【素描验证】那一列，')
    print('      它测的是渲染鲁棒性，差别会大到验证集分辨得出来。')

    if arguments.write_submission:
        import pandas as pd
        for name, probabilities in [('probe', probe_test), ('knn', knn_test),
                                    ('blend', blend_test)]:
            frame = pd.DataFrame({
                'ID': data['test_ids'].astype(int),
                'Label': [data['class_names'][i] for i in probabilities.argmax(1)],
            }).sort_values('ID')
            assert len(frame) == 11_681, f'行数 {len(frame)}，应为 11,681'
            assert frame['ID'].is_unique, 'ID 有重复'
            path = f'submission_frozen_{arguments.tag}_{name}.csv'
            frame.to_csv(path, index=False)
            print(f'写出 {path}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
