"""diagnose_disagreement.py -- find the confident errors that margin cannot see.

Confidence is saturated (mean top-1 ~0.883 against a label-smoothing ceiling of 0.9) and
only 1.5% of the test set is contested by margin, while ~3.4% of it is wrong. So the
remaining errors are confident ones, and they need a different instrument.

Disagreement between ensemble members is that instrument. For each test image:

    H[mean_m p_m]                 total predictive entropy   (ambiguous OR unknown)
  - mean_m H[p_m]                 average member entropy     (ambiguous only)
  = mutual information            members disagree           (unknown only)

The subtraction is the point. An image every member reads as "60% cat, 40% dog" has high
entropy and zero MI: genuinely ambiguous, nothing to fix. An image where one member says
"98% cat" and another says "98% lion" has the same entropy and high MI: the models are
each confident and they contradict each other, which is what a confident error looks like
from the outside. MI is close to orthogonal to margin, so it reaches images margin ranks
as settled.

Run from C:\\ml\\kaggle (no GPU, seconds):

    python diagnose_disagreement.py
    python diagnose_disagreement.py --members "*_run.npz" --submission submission_corr_B.csv
    python diagnose_disagreement.py --top 500 --out review_top500.csv

Writes a ranked CSV you can open alongside the test images.
"""

from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

CLASS_NAMES = [
    'birds', 'bottles', 'breads', 'butterflies', 'cakes', 'cats',
    'chickens', 'cows', 'dogs', 'ducks', 'elephants', 'fishes',
    'handguns', 'horses', 'lions', 'lipsticks', 'seals', 'snakes',
    'spiders', 'vases',
]
NUM_CLASSES = len(CLASS_NAMES)
EPSILON = 1e-12


def entropy(probabilities: np.ndarray) -> np.ndarray:
    """Shannon entropy in nats, per row."""
    return -(probabilities * np.log(probabilities + EPSILON)).sum(axis=-1)


def infer_stage(path: str) -> str:
    """Guess which training stage a cached run belongs to, from its filename."""
    name = os.path.basename(path).lower()
    for stage in ('distil', 'pseudo2', 'pseudo', 'base'):
        if stage in name:
            return stage
    return 'unknown'


def load_members(pattern: str):
    """Return (probabilities [M, N, C], names, stages, ids)."""
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f'No files matched {pattern!r}. Point --members at the cached runs.')

    stack, names, stages, reference_ids = [], [], [], None
    for path in paths:
        with np.load(path, allow_pickle=True) as archive:
            key = next((k for k in ('test_probabilities', 'test_probs') if k in archive), None)
            if key is None:
                print(f'  skipped {path}: no test probabilities inside')
                continue
            probabilities = np.asarray(archive[key], dtype=np.float64)
            ids = None
            for id_key in ('test_key_order', 'test_ids', 'ids'):
                if id_key in archive:
                    ids = np.asarray(archive[id_key]).astype(np.int64)
                    break
            model_name = str(archive['name']) if 'name' in archive else os.path.basename(path)

        if probabilities.ndim != 2 or probabilities.shape[1] != NUM_CLASSES:
            print(f'  skipped {path}: shape {probabilities.shape}')
            continue
        if reference_ids is None:
            reference_ids = ids
        elif ids is not None and not np.array_equal(ids, reference_ids):
            raise SystemExit(f'{path}: test ID order differs from the first file; cannot compare.')

        stack.append(probabilities)
        names.append(re.sub(r'\.npz$', '', os.path.basename(path)))
        stages.append(infer_stage(path))
        print(f'  {os.path.basename(path):<44} stage={stages[-1]:<8} {model_name[:34]}')

    if len(stack) < 2:
        raise SystemExit('Need at least two members to measure disagreement.')
    shapes = {a.shape for a in stack}
    if len(shapes) > 1:
        raise SystemExit(f'Members have mismatched shapes: {shapes}')

    probabilities = np.stack(stack)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    if reference_ids is None:
        reference_ids = np.arange(probabilities.shape[1], dtype=np.int64)
    return probabilities, names, stages, reference_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--members', default='*_run.npz', help='glob for the cached per-model runs')
    parser.add_argument('--submission', default=None, help='optional CSV to compare labels against')
    parser.add_argument('--top', type=int, default=500, help='how many images to export for review')
    parser.add_argument('--out', default='review_disagreement.csv')
    arguments = parser.parse_args()

    print(f'Members matching {arguments.members!r}:')
    members, names, stages, ids = load_members(arguments.members)
    n_members, n_images, _ = members.shape

    ensemble = members.mean(axis=0)
    ensemble_label = ensemble.argmax(axis=1)
    sorted_probabilities = np.sort(ensemble, axis=1)
    margin = sorted_probabilities[:, -1] - sorted_probabilities[:, -2]
    confidence = sorted_probabilities[:, -1]

    total_entropy = entropy(ensemble)
    member_entropy = entropy(members).mean(axis=0)
    mutual_information = np.clip(total_entropy - member_entropy, 0.0, None)

    member_labels = members.argmax(axis=2)                       # (M, N)
    votes_against = (member_labels != ensemble_label[None, :]).sum(axis=0)

    print(f'\n{n_images:,} images | {n_members} members | mean top-1 {confidence.mean():.3f}')
    print(f'Mutual information: mean {mutual_information.mean():.4f} nats, '
          f'max {mutual_information.max():.4f}, '
          f'zero on {float((mutual_information < 1e-4).mean()):.1%} of images')

    # ---------------------------------------------------------------- votes
    print('\n' + '=' * 74)
    print('VOTE SPLITS  (members disagreeing with the ensemble argmax)')
    print('=' * 74)
    print(f'{"against":>8}{"images":>10}{"share":>9}{"mean margin":>14}{"mean MI":>10}')
    for k in range(n_members):
        mask = votes_against == k
        if not mask.any():
            continue
        print(f'{k:>8}{int(mask.sum()):>10}{mask.mean():>9.1%}'
              f'{margin[mask].mean():>14.3f}{mutual_information[mask].mean():>10.4f}')

    # ------------------------------------------------------------ stage split
    unique_stages = sorted(set(stages) - {'unknown'})
    if len(unique_stages) > 1:
        print('\n' + '=' * 74)
        print('STAGE DISAGREEMENT  (each stage averaged separately, then compared)')
        print('=' * 74)
        stage_labels = {}
        for stage in unique_stages:
            index = [i for i, s in enumerate(stages) if s == stage]
            stage_labels[stage] = members[index].mean(axis=0).argmax(axis=1)
        for i, a in enumerate(unique_stages):
            for b in unique_stages[i + 1:]:
                differs = stage_labels[a] != stage_labels[b]
                print(f'{a:<10} vs {b:<10} differ on {int(differs.sum()):>6} images '
                      f'({differs.mean():.2%})')
        agreed = np.all([stage_labels[s] == ensemble_label for s in unique_stages], axis=0)
        print(f'All stages agree with the ensemble on {agreed.mean():.2%} of images.')

    # -------------------------------------------------- MI vs margin overlap
    print('\n' + '=' * 74)
    print('IS MI SAYING SOMETHING MARGIN DOES NOT?')
    print('=' * 74)
    correlation = np.corrcoef(mutual_information, margin)[0, 1]
    k = arguments.top
    by_mi = set(np.argsort(-mutual_information)[:k].tolist())
    by_margin = set(np.argsort(margin)[:k].tolist())
    overlap = len(by_mi & by_margin)
    print(f'corr(MI, margin) = {correlation:+.3f}   '
          f'(strongly negative would mean MI is just margin again)')
    print(f'Top-{k} by MI and bottom-{k} by margin overlap on {overlap} images '
          f'({overlap / k:.1%}).')
    settled = margin > 0.5
    high_mi = mutual_information >= np.quantile(mutual_information, 1 - k / n_images)
    print(f'Images that margin calls settled (margin > 0.5) but MI ranks in the top {k}: '
          f'{int((settled & high_mi).sum())}')

    # ---------------------------------------------------- class concentration
    print('\n' + '=' * 74)
    print(f'WHERE THE TOP-{k} BY MI LAND  (ensemble label)')
    print('=' * 74)
    selected = np.argsort(-mutual_information)[:k]
    counts = np.bincount(ensemble_label[selected], minlength=NUM_CLASSES)
    base_counts = np.bincount(ensemble_label, minlength=NUM_CLASSES)
    rows = []
    for c in range(NUM_CLASSES):
        if base_counts[c] == 0:
            continue
        rows.append((counts[c] / base_counts[c], counts[c], base_counts[c], CLASS_NAMES[c]))
    rows.sort(reverse=True)
    print(f'{"class":<14}{"in top-k":>10}{"class size":>12}{"rate":>9}   '
          f'(overall rate {k / n_images:.1%})')
    for rate, count, size, name in rows[:10]:
        print(f'{name:<14}{count:>10}{size:>12}{rate:>9.1%}')

    # ------------------------------------------------------- runner-up pairs
    print('\n' + '=' * 74)
    print(f'CONTRADICTED PAIRS  (top-{k} by MI: ensemble label vs the label a member preferred)')
    print('=' * 74)
    pair_counts: dict[tuple[str, str], int] = {}
    for i in selected:
        for m in range(n_members):
            other = member_labels[m, i]
            if other == ensemble_label[i]:
                continue
            key = tuple(sorted((CLASS_NAMES[ensemble_label[i]], CLASS_NAMES[other])))
            pair_counts[key] = pair_counts.get(key, 0) + 1
    for (a, b), count in sorted(pair_counts.items(), key=lambda item: -item[1])[:15]:
        print(f'{a:<14} vs {b:<14}{count:>6}')

    # ------------------------------------------------------------ review CSV
    frame = pd.DataFrame({
        'ID': ids[selected],
        'ensemble_label': [CLASS_NAMES[c] for c in ensemble_label[selected]],
        'confidence': confidence[selected].round(3),
        'margin': margin[selected].round(3),
        'mutual_information': mutual_information[selected].round(4),
        'votes_against': votes_against[selected],
    })
    for m, name in enumerate(names):
        frame[name] = [CLASS_NAMES[c] for c in member_labels[m, selected]]

    if arguments.submission:
        submitted = pd.read_csv(arguments.submission)
        frame = frame.merge(submitted.rename(columns={'Label': 'submitted_label'}),
                            on='ID', how='left')
        differs = (frame['submitted_label'] != frame['ensemble_label']).sum()
        print(f'\n{differs} of the top-{k} already differ from {arguments.submission} '
              f'(manual corrections and stage choice).')

    frame.to_csv(arguments.out, index=False)
    print(f'\nWrote {arguments.out} ({len(frame)} rows), ranked by mutual information.')
    print('Open the corresponding test images and read the top 50 by eye. What you are '
          'looking for is whether they share a property -- a rendition style, a scale, a '
          'background -- that the training data lacks.')


if __name__ == '__main__':
    main()
