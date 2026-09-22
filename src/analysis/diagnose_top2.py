"""diagnose_top2.py -- where is the probability mass of the under-predicted classes going?

Answers one question: is `fishes` predicted 187 times because the test set really contains
few fish, or because fish are being confidently read as something else?

Run from C:\\ml\\kaggle (no GPU, a few seconds):

    python diagnose_top2.py                                  # auto-find the probabilities
    python diagnose_top2.py --probs bench/best_ens_v2_test.npy
    python diagnose_top2.py --probs "*_run.npz"              # average every cached run

Reads only a (11681, 20) probability matrix. No IDs, no labels, no data needed.
"""

from __future__ import annotations

import argparse
import glob
import sys

import numpy as np

CLASS_NAMES = [
    'birds', 'bottles', 'breads', 'butterflies', 'cakes', 'cats',
    'chickens', 'cows', 'dogs', 'ducks', 'elephants', 'fishes',
    'handguns', 'horses', 'lions', 'lipsticks', 'seals', 'snakes',
    'spiders', 'vases',
]
NUM_CLASSES = len(CLASS_NAMES)

# Same-semantic-group pairs, i.e. the swaps the manual-correction pass allowed.
SEMANTIC_GROUPS = [
    {'birds', 'ducks', 'chickens'},
    {'cats', 'dogs', 'lions'},
    {'breads', 'cakes'},
    {'bottles', 'vases'},
    {'fishes', 'seals'},
    {'cows', 'horses', 'elephants'},
]

DEFAULT_SOURCES = ['bench/best_ens_v2_test.npy', 'bench/*_test.npy', '*_run.npz']


def _from_npz(path: str) -> np.ndarray:
    with np.load(path, allow_pickle=True) as archive:
        for key in ('test_probabilities', 'test_probs', 'test'):
            if key in archive:
                return np.asarray(archive[key], dtype=np.float64)
    raise KeyError(f'{path}: no test_probabilities array inside')


def load_probabilities(pattern: str | None) -> np.ndarray:
    patterns = [pattern] if pattern else DEFAULT_SOURCES
    for candidate in patterns:
        paths = sorted(glob.glob(candidate))
        if not paths:
            continue
        stack = [_from_npz(p) if p.endswith('.npz') else np.load(p) for p in paths]
        shapes = {a.shape for a in stack}
        if len(shapes) > 1:
            raise SystemExit(f'{candidate}: mismatched shapes {shapes}')
        print(f'Loaded {len(paths)} file(s) matching {candidate}:')
        for p in paths:
            print(f'  {p}')
        probabilities = np.mean(stack, axis=0)
        break
    else:
        raise SystemExit(
            'No probability file found. Point --probs at the saved ensemble, e.g.\n'
            '  python diagnose_top2.py --probs bench/best_ens_v2_test.npy'
        )

    if probabilities.ndim != 2 or probabilities.shape[1] != NUM_CLASSES:
        raise SystemExit(f'Expected (N, 20), got {probabilities.shape}')
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if not np.allclose(row_sums, 1.0, atol=1e-3):
        print('Rows do not sum to 1; normalising.')
        probabilities = probabilities / row_sums
    return probabilities


def same_group(a: str, b: str) -> bool:
    return any({a, b} <= group for group in SEMANTIC_GROUPS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--probs', default=None, help='path or glob for the test probabilities')
    parser.add_argument('--margin', type=float, default=0.20,
                        help='an image is "contested" when p1 - p2 is below this')
    parser.add_argument('--top', type=int, default=6, help='how many classes to detail')
    arguments = parser.parse_args()

    probabilities = load_probabilities(arguments.probs)
    n_images = len(probabilities)
    order = np.argsort(-probabilities, axis=1)
    winner, runner_up = order[:, 0], order[:, 1]
    p1 = probabilities[np.arange(n_images), winner]
    p2 = probabilities[np.arange(n_images), runner_up]
    margin = p1 - p2

    hard_count = np.bincount(winner, minlength=NUM_CLASSES)
    soft_mass = probabilities.sum(axis=0)

    print(f'\n{n_images:,} test images | mean top-1 confidence {p1.mean():.3f} '
          f'| contested (margin < {arguments.margin:g}): '
          f'{int((margin < arguments.margin).sum()):,} '
          f'({(margin < arguments.margin).mean():.1%})\n')

    # ---------------------------------------------------------------- table 1
    # A class whose soft mass exceeds its hard count is being shaved: the model
    # keeps putting it second without ever letting it win.
    print('=' * 78)
    print('CLASS LEDGER  (soft mass = sum of probabilities; ratio > 1 means shaved)')
    print('=' * 78)
    print(f'{"class":<13}{"top-1":>8}{"soft":>9}{"soft/top1":>11}'
          f'{"conf":>8}{"runner-up rate":>16}')
    runner_up_rate = np.bincount(runner_up, minlength=NUM_CLASSES) / n_images
    confidence = np.array([p1[winner == c].mean() if hard_count[c] else np.nan
                           for c in range(NUM_CLASSES)])
    for c in np.argsort(hard_count):
        ratio = soft_mass[c] / max(hard_count[c], 1)
        flag = '  <-- shaved' if ratio > 1.15 else ''
        print(f'{CLASS_NAMES[c]:<13}{hard_count[c]:>8}{soft_mass[c]:>9.0f}{ratio:>11.2f}'
              f'{confidence[c]:>8.3f}{runner_up_rate[c]:>15.1%}{flag}')

    # ---------------------------------------------------------------- table 2
    # Directed leak: images where class c is the runner-up, grouped by who won.
    # This is the "where did the fish go" question, asked directly.
    print('\n' + '=' * 78)
    print(f'MASS FLOW  (for the {arguments.top} least-predicted classes)')
    print('  "c loses to w": images where c is runner-up and w wins,')
    print('  reported as a count and as the probability mass c holds in them.')
    print('=' * 78)
    for c in np.argsort(hard_count)[:arguments.top]:
        mask = runner_up == c
        print(f'\n{CLASS_NAMES[c]}  (top-1 {hard_count[c]}, soft mass {soft_mass[c]:.0f}, '
              f'runner-up on {int(mask.sum())} images)')
        rows = []
        for w in range(NUM_CLASSES):
            selected = mask & (winner == w)
            if not selected.any():
                continue
            rows.append((
                int(selected.sum()),
                float(probabilities[selected, c].sum()),
                int((selected & (margin < arguments.margin)).sum()),
                CLASS_NAMES[w],
            ))
        rows.sort(reverse=True)
        for count, mass, contested, name in rows[:5]:
            tag = 'same group' if same_group(CLASS_NAMES[c], name) else ''
            print(f'   loses to {name:<13} n={count:<6} mass={mass:>7.1f} '
                  f'contested={contested:<6} {tag}')

    # ---------------------------------------------------------------- table 3
    # The contested pairs themselves, unordered, ranked. These are the pairs a
    # targeted batch of external data would have to separate.
    print('\n' + '=' * 78)
    print(f'CONTESTED PAIRS  (margin < {arguments.margin:g}, both directions pooled)')
    print('=' * 78)
    pair_counts: dict[tuple[int, int], int] = {}
    contested_mask = margin < arguments.margin
    for a, b in zip(winner[contested_mask], runner_up[contested_mask]):
        key = (min(a, b), max(a, b))
        pair_counts[key] = pair_counts.get(key, 0) + 1
    ranked = sorted(pair_counts.items(), key=lambda item: -item[1])[:15]
    total_contested = int(contested_mask.sum())
    for (a, b), count in ranked:
        tag = 'same group' if same_group(CLASS_NAMES[a], CLASS_NAMES[b]) else 'CROSS GROUP'
        print(f'{CLASS_NAMES[a]:<13} vs {CLASS_NAMES[b]:<13} {count:>6} images '
              f'({count / total_contested:>5.1%} of contested)  {tag}')

    # ---------------------------------------------------------------- verdict
    print('\n' + '=' * 78)
    print('READING')
    print('=' * 78)
    shaved = [CLASS_NAMES[c] for c in range(NUM_CLASSES)
              if soft_mass[c] / max(hard_count[c], 1) > 1.15]
    if shaved:
        print('Shaved classes (soft mass well above top-1 count):', ', '.join(shaved))
        print('  These lose narrowly and repeatedly. Targeted external renditions for')
        print('  them should move the leaderboard.')
    else:
        print('No class is shaved: every class wins roughly as often as its total')
        print('  probability mass says it should. The skewed counts are then a real')
        print('  prior, targeted data will not help, and the remaining errors are')
        print('  spread thin rather than concentrated in a few pairs.')
    cross = [f'{CLASS_NAMES[a]}/{CLASS_NAMES[b]}' for (a, b), _ in ranked
             if not same_group(CLASS_NAMES[a], CLASS_NAMES[b])][:3]
    if cross:
        print('Contested pairs outside the semantic groups:', ', '.join(cross))
        print('  The manual-correction rule ignored these by construction.')


if __name__ == '__main__':
    sys.exit(main())
