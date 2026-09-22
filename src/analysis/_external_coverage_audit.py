"""Zero-cost data audit (no GPU): per-class external-data coverage (pool size + how many
actually get sampled into training) and a resolution/sharpness profile (Laplacian variance,
a standard blur/detail proxy -- higher = more high-frequency detail retained), for the 6
confusion-cluster classes vs the rest. Answers "is there a real coverage gap to fill" before
committing GPU time to data curation or high-res fine-tuning.
"""
import numpy as np
import cv2
import torch

CN = ['birds','bottles','breads','butterflies','cakes','cats','chickens','cows','dogs','ducks',
      'elephants','fishes','handguns','horses','lions','lipsticks','seals','snakes','spiders','vases']

CLUSTERS = {
    'breads/cakes': {'breads', 'cakes'},
    'bottles/vases/lipsticks': {'bottles', 'vases', 'lipsticks'},
    'chickens/ducks/birds': {'chickens', 'ducks', 'birds'},
    'cats/dogs': {'cats', 'dogs'},
    'cows/elephants/horses': {'cows', 'elephants', 'horses'},
    'snakes/spiders/butterflies': {'snakes', 'spiders', 'butterflies'},
}
def cluster_of(name):
    for cname, members in CLUSTERS.items():
        if name in members:
            return cname
    return 'other (fishes/handguns/lions/seals)'


def laplacian_var(img_u8):
    """img_u8: (3,H,W) uint8. Higher = sharper / more retained high-frequency detail."""
    grey = cv2.cvtColor(np.transpose(img_u8, (1, 2, 0)), cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(grey, cv2.CV_64F).var()


def main():
    targets = np.load('bench/targets.npy')
    ext_img = np.load('bench/external_images.npy')
    ext_lab = np.load('bench/external_labels.npy')
    test_img = np.load('bench/test_images.npy')

    real_count = 8047
    ratio = 0.25
    wanted = min(int(round(real_count * ratio / (1.0 - ratio))), len(ext_img))
    picker = torch.Generator().manual_seed(1234)
    keep = torch.randperm(len(ext_img), generator=picker)[:wanted].numpy()
    sampled_mask = np.zeros(len(ext_img), dtype=bool)
    sampled_mask[keep] = True

    train_counts = np.bincount(targets, minlength=20)
    ext_pool_counts = np.bincount(ext_lab, minlength=20)
    ext_sampled_counts = np.bincount(ext_lab[sampled_mask], minlength=20)

    # sharpness profile: sample up to 150 images per class from each source for speed
    rng = np.random.RandomState(0)

    def mean_sharpness(images, labels, cls, cap=150):
        idx = np.flatnonzero(labels == cls)
        if len(idx) == 0:
            return float('nan'), 0
        if len(idx) > cap:
            idx = rng.choice(idx, cap, replace=False)
        vals = [laplacian_var(images[i]) for i in idx]
        return float(np.mean(vals)), len(idx)

    test_sharp_by_class = {}  # test set has no labels -- overall reference only
    test_idx = rng.choice(len(test_img), 300, replace=False)
    overall_test_sharpness = np.mean([laplacian_var(test_img[i]) for i in test_idx])

    print(f'{"class":13s} {"cluster":28s} {"train":>6s} {"ext_pool":>9s} {"ext_used":>9s} '
          f'{"train_sharp":>12s} {"ext_sharp":>10s}')
    rows = []
    for c in range(20):
        name = CN[c]
        train_sharp, _ = mean_sharpness(np.load('bench/train_images.npy'), targets, c)
        ext_sharp, _ = mean_sharpness(ext_img, ext_lab, c)
        rows.append((name, cluster_of(name), train_counts[c], ext_pool_counts[c],
                     ext_sampled_counts[c], train_sharp, ext_sharp))

    rows.sort(key=lambda r: (r[1], r[0]))
    for name, cl, tc, ep, es, ts, xs in rows:
        flag = '  <-- LOW EXT POOL' if ep < 150 else ('  <-- zero external!' if ep == 0 else '')
        print(f'{name:13s} {cl:28s} {tc:6d} {ep:9d} {es:9d} {ts:12.1f} {xs:10.1f}{flag}')

    print(f'\noverall test-set mean sharpness (reference, 300-sample): {overall_test_sharpness:.1f}')


if __name__ == '__main__':
    main()
