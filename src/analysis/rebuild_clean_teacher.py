r"""rebuild_clean_teacher.py -- rebuild a teacher that predates the 47 manual corrections.

bench/best_ens_v2_test.npy (and its v3 / test_prob copies, all written 09-10 01:00) now
reproduce submission_corr_B.csv exactly, which means the hand-edited labels are baked into
the probability matrix. The pseudo stage seeds from that path, so the next pseudo run would
pull manual labels into training and quietly invalidate the "pure model 0.96499" claim.

v10_distil\ was written 09-09 15:45, hours before the corrections, so it is clean. This
rebuilds the ensemble average from those three files and checks whether it reproduces
submission_D3_LgP.csv row for row.

    python rebuild_clean_teacher.py
    python rebuild_clean_teacher.py --members "v10_distil\\*.npz" --write
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd

CLASS_NAMES = [
    'birds', 'bottles', 'breads', 'butterflies', 'cakes', 'cats',
    'chickens', 'cows', 'dogs', 'ducks', 'elephants', 'fishes',
    'handguns', 'horses', 'lions', 'lipsticks', 'seals', 'snakes',
    'spiders', 'vases',
]
CLASS_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--members', default=os.path.join('v10_distil', '*.npz'))
    parser.add_argument('--target', default='submission_D3_LgP.csv')
    parser.add_argument('--out', default=os.path.join('bench', 'teacher_clean_test.npy'))
    parser.add_argument('--write', action='store_true',
                        help='actually save; without it the script only reports')
    arguments = parser.parse_args()

    paths = sorted(glob.glob(arguments.members))
    if not paths:
        raise SystemExit(f'No files matched {arguments.members!r}')

    stack = []
    for path in paths:
        with np.load(path, allow_pickle=True) as archive:
            probabilities = np.asarray(archive['test_probabilities'], dtype=np.float64)
            if 'test_key_order' in archive:
                order = np.asarray(archive['test_key_order']).astype(np.int64)
                ordered = np.empty_like(probabilities)
                ordered[order] = probabilities          # put rows back in ID order 0..n-1
            else:
                ordered = probabilities
            stack.append(ordered)
        print(f'  loaded {path}  {probabilities.shape}')

    clean = np.mean(stack, axis=0)
    print(f'\nAveraged {len(stack)} members -> {clean.shape}')

    if os.path.exists(arguments.target):
        frame = pd.read_csv(arguments.target).sort_values('ID')
        target = frame['Label'].map(CLASS_INDEX).to_numpy()
        agreement = float((clean.argmax(axis=1) == target).mean())
        differing = int((clean.argmax(axis=1) != target).sum())
        print(f'Agreement with {arguments.target}: {agreement:.4%}  ({differing} rows differ)')
        if differing == 0:
            print('Exact. These three files ARE the 0.96499 pure-model ensemble.')
        else:
            print('Not exact. Report the row count above before changing anything.')
    else:
        print(f'{arguments.target} not found; skipping the check.')

    if arguments.write:
        os.makedirs(os.path.dirname(arguments.out) or '.', exist_ok=True)
        np.save(arguments.out, clean)
        order_path = arguments.out.replace('_test.npy', '_order.npy')
        np.save(order_path, np.arange(len(clean), dtype=np.int64))
        print(f'\nWrote {arguments.out} and {order_path}')
        print('Now point the pseudo-stage seed in CONFIG at the new path.')
    else:
        print('\nNothing written. Re-run with --write once the agreement line looks right.')


if __name__ == '__main__':
    main()
