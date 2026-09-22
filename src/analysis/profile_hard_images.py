r"""profile_hard_images.py -- what kind of image does this ensemble get wrong?

The test set has no labels, so "wrong" is unobservable. Two things are observable:

  1. how far each test image sits from the training distribution
     (bench\_sim_TRAIN.npy, _sim_ImageNet-R.npy, _sim_DomainNet.npy, computed earlier
     in the domain-gap investigation)
  2. how unstable each image's prediction is across submissions

If (1) predicts (2), then "hard" reduces to "far from training data", which is a causal
story worth reporting. If it does not, some other mechanism is at work and the domain-gap
narrative does not fully explain the residual errors.

    python profile_hard_images.py
    python profile_hard_images.py --subs submission_B_A3_cnv2.csv submission_D3_LgP.csv submission_corr_B.csv
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd

BENCH = 'bench'
SIM_FILES = {
    'sim_train': '_sim_TRAIN.npy',
    'sim_inr': '_sim_ImageNet-R.npy',
    'sim_domainnet': '_sim_DomainNet.npy',
    'sim_inr_max': '_test_max_sim_inr.npy',
}


def load_optional(name: str, n_rows: int):
    path = os.path.join(BENCH, name)
    if not os.path.exists(path):
        return None
    array = np.load(path)
    return array if array.shape == (n_rows,) else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--subs', nargs='*', default=None,
                        help='submission CSVs to measure instability across')
    parser.add_argument('--quantiles', type=int, default=5)
    parser.add_argument('--out', default='hard_image_profile.csv')
    arguments = parser.parse_args()

    paths = arguments.subs or sorted(glob.glob('submission*.csv'))
    frames = {}
    for path in paths:
        frame = pd.read_csv(path).sort_values('ID').reset_index(drop=True)
        if 'Label' in frame:
            frames[os.path.basename(path)] = frame
    if not frames:
        raise SystemExit('No submission CSVs found.')
    ids = next(iter(frames.values()))['ID'].to_numpy()
    n_rows = len(ids)
    labels = {name: frame['Label'].to_numpy() for name, frame in frames.items()}
    print(f'{len(frames)} submissions, {n_rows} rows each\n')

    # instability: how many distinct labels does an image receive across submissions
    stacked = np.stack(list(labels.values()))
    distinct = np.array([len(set(column)) for column in stacked.T])
    modal = stacked[0]
    unstable = distinct > 1
    print(f'Images predicted differently by at least one submission: '
          f'{int(unstable.sum())} ({unstable.mean():.2%})')
    for k in range(1, len(frames) + 1):
        count = int((distinct == k).sum())
        if count:
            print(f'  {k} distinct label(s): {count}')

    # -------------------------------------------------------------- domain distance
    table = pd.DataFrame({'ID': ids, 'label': modal, 'distinct': distinct})
    available = []
    for key, filename in SIM_FILES.items():
        array = load_optional(filename, n_rows)
        if array is None:
            print(f'  missing or wrong shape: {os.path.join(BENCH, filename)}')
            continue
        table[key] = array
        available.append(key)
    if not available:
        print('\nNo similarity arrays available; stopping after the instability summary.')
        table.to_csv(arguments.out, index=False)
        return

    print('\n' + '=' * 74)
    print('INSTABILITY BY DOMAIN-DISTANCE QUANTILE')
    print('  If distance drives difficulty, the unstable rate rises monotonically')
    print('  as similarity to the training set falls.')
    print('=' * 74)
    for key in available:
        values = table[key].to_numpy()
        edges = np.quantile(values, np.linspace(0, 1, arguments.quantiles + 1))
        edges[-1] += 1e-9
        bucket = np.digitize(values, edges[1:-1])
        print(f'\n{key}  (low = far from that reference)')
        print(f'{"quantile":<12}{"n":>7}{"mean sim":>11}{"unstable":>11}')
        for q in range(arguments.quantiles):
            mask = bucket == q
            if not mask.any():
                continue
            print(f'Q{q + 1} {"(lowest)" if q == 0 else "":<8}{int(mask.sum()):>7}'
                  f'{values[mask].mean():>11.4f}{unstable[mask].mean():>10.1%}')
        lowest = unstable[bucket == 0].mean()
        highest = unstable[bucket == arguments.quantiles - 1].mean()
        ratio = lowest / max(highest, 1e-9)
        print(f'  lowest / highest quantile ratio: {ratio:.2f}x')

    # --------------------------------------------------------- per class distance
    print('\n' + '=' * 74)
    print('PER-CLASS MEAN DOMAIN DISTANCE AND INSTABILITY')
    print('=' * 74)
    rows = []
    for name in sorted(set(modal)):
        mask = modal == name
        row = {'class': name, 'n': int(mask.sum()),
               'unstable': float(unstable[mask].mean())}
        for key in available:
            row[key] = float(table.loc[mask, key].mean())
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values('unstable', ascending=False)
    print(summary.to_string(index=False,
                            formatters={'unstable': '{:.1%}'.format,
                                        **{k: '{:.4f}'.format for k in available}}))

    table['unstable'] = unstable
    table.to_csv(arguments.out, index=False)
    print(f'\nWrote {arguments.out}. The unstable rows, sorted by sim_train ascending,')
    print('are the images to open and look at.')


if __name__ == '__main__':
    main()
