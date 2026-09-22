"""Same-stage comparison of the Web-SSL base against the two DINO ViT-L bases (no submission).

The rendition reserve's valid regime is same-stage base-vs-base direction (5-for-5 on leaderboard
signals); its magnitude does not convert to the leaderboard. Test-set disagreement is reported
for the named risk -- Web-SSL is the DINOv2 recipe on different data -- and is not a gate.
"""
import sys
from math import comb

import numpy as np

CN = list(np.load('bench/class_names.npy', allow_pickle=True))
LAB = np.load('bench/reserve_preds.npz')['lab']
NEW = sys.argv[1] if len(sys.argv) > 1 else 'websslvitl_base'     # e.g. eva02l_base
BASES = {'V2 base (0.97363)': 'dinov2vitl_base', 'V3T base (0.97556)': 'dinov3vitl_base'}


def mcnemar(b, c):
    n = b + c
    tail = sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n if n else 1.0
    return min(1.0, 2 * tail)


def main():
    new_t = np.load(f'bench/{NEW}_test_prob.npy').argmax(1)
    new_r = np.load(f'bench/{NEW}_reserve_prob.npy').argmax(1) == LAB
    print(f'{NEW}: reserve {100 * new_r.mean():.2f}%')
    for name, tag in BASES.items():
        old_t = np.load(f'bench/{tag}_test_prob.npy').argmax(1)
        old_r = np.load(f'bench/{tag}_reserve_prob.npy').argmax(1) == LAB
        b, c = int((new_r & ~old_r).sum()), int((~new_r & old_r).sum())
        k = int((new_t != old_t).sum())
        print(f'  vs {name:20s} reserve {100 * (new_r.mean() - old_r.mean()):+.2f}pt '
              f'({b} vs {c}, p = {mcnemar(b, c):.3g}) | test disagreement k = {k} '
              f'({100 * k / len(new_t):.2f}%)')
    v2, v3 = (np.load(f'bench/{t}_test_prob.npy').argmax(1) for t in BASES.values())
    print(f'  reference: V2 base vs V3T base test disagreement {100 * (v2 != v3).mean():.2f}%')


if __name__ == '__main__':
    main()
