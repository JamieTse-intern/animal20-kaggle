"""inventory_caches.py -- work out which cached arrays produced which submission.

The question this answers: is the self-distillation stage (the one that took you from
0.962 to 0.965) still on disk, and under what filename?

Filenames lie, mtimes are suggestive, but the decisive test is arithmetic: average some
subset of the cached probability arrays, take the argmax, and see whether it reproduces a
submission CSV row for row. The subset that reproduces submission_D3_LgP.csv at 100% IS
the 0.965 ensemble, whatever its files are called.

Run from C:\\ml\\kaggle:

    python inventory_caches.py                                    # inventory + per-file match
    python inventory_caches.py --target submission_D3_LgP.csv     # + greedy subset search
"""

from __future__ import annotations

import argparse
import datetime as dt
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
NUM_CLASSES = len(CLASS_NAMES)
CLASS_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}


def walk(root: str) -> list[str]:
    found = []
    for directory, _, files in os.walk(root):
        if any(part in directory for part in ('.git', '.venv', '__pycache__')):
            continue
        for name in files:
            if name.endswith(('.npz', '.npy')):
                found.append(os.path.join(directory, name))
    return sorted(found)


def load_candidate(path: str):
    """Return (probabilities, ids, note) or (None, None, note)."""
    try:
        if path.endswith('.npy'):
            array = np.load(path, allow_pickle=False)
            if array.ndim == 2 and array.shape[1] == NUM_CLASSES:
                return array.astype(np.float64), None, f'array {array.shape}'
            return None, None, f'array {array.shape} (not a probability matrix)'

        with np.load(path, allow_pickle=True) as archive:
            keys = list(archive.keys())
            key = next((k for k in ('test_probabilities', 'test_probs') if k in archive), None)
            if key is None:
                return None, None, 'keys: ' + ', '.join(keys[:6])
            probabilities = np.asarray(archive[key], dtype=np.float64)
            ids = None
            for id_key in ('test_key_order', 'test_ids', 'ids'):
                if id_key in archive:
                    ids = np.asarray(archive[id_key]).astype(np.int64)
                    break
            note = f'{probabilities.shape}'
            if 'validation_accuracy' in archive:
                note += f' val={float(archive["validation_accuracy"]):.4f}'
            if 'name' in archive:
                note += f' {str(archive["name"])[:40]}'
            return probabilities, ids, note
    except Exception as error:                                  # noqa: BLE001
        return None, None, f'unreadable: {type(error).__name__}'


def aligned_labels(probabilities: np.ndarray, ids: np.ndarray | None,
                   n_rows: int) -> np.ndarray | None:
    """argmax labels put back into ID order 0..n-1."""
    if len(probabilities) != n_rows:
        return None
    labels = probabilities.argmax(axis=1)
    if ids is None:
        return labels
    ordered = np.empty(n_rows, dtype=np.int64)
    ordered[ids] = labels
    return ordered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='.')
    parser.add_argument('--submissions', default='submission*.csv')
    parser.add_argument('--target', default=None,
                        help='CSV to reproduce by greedy subset averaging')
    arguments = parser.parse_args()

    submissions = {}
    for path in sorted(glob.glob(os.path.join(arguments.root, arguments.submissions))):
        frame = pd.read_csv(path).sort_values('ID')
        if 'Label' not in frame:
            continue
        submissions[os.path.basename(path)] = frame['Label'].map(CLASS_INDEX).to_numpy()
    n_rows = len(next(iter(submissions.values()))) if submissions else 11681
    print(f'Submissions found: {len(submissions)} | rows: {n_rows}\n')

    print('=' * 100)
    print('CACHE INVENTORY  (newest last)')
    print('=' * 100)
    entries = []
    for path in walk(arguments.root):
        stat = os.stat(path)
        probabilities, ids, note = load_candidate(path)
        entries.append((stat.st_mtime, path, stat.st_size, probabilities, ids, note))
    entries.sort()
    for mtime, path, size, probabilities, ids, note in entries:
        stamp = dt.datetime.fromtimestamp(mtime).strftime('%m-%d %H:%M')
        usable = 'OK ' if probabilities is not None else '   '
        print(f'{stamp}  {size / 1e6:>7.1f} MB  {usable}{os.path.relpath(path, arguments.root)}')
        print(f'{"":>28}{note}')

    usable = [(path, p, i) for _, path, _, p, i, _ in entries if p is not None]
    if not usable or not submissions:
        print('\nNothing further to match.')
        return

    print('\n' + '=' * 100)
    print('AGREEMENT OF EACH FILE WITH EACH SUBMISSION  (100% means this file alone wrote it)')
    print('=' * 100)
    names = list(submissions)
    header = f'{"file":<58}' + ''.join(f'{n[:16]:>18}' for n in names)
    print(header)
    label_cache = {}
    for path, probabilities, ids in usable:
        labels = aligned_labels(probabilities, ids, n_rows)
        if labels is None:
            continue
        label_cache[path] = labels
        row = f'{os.path.relpath(path, arguments.root)[-56:]:<58}'
        for name in names:
            row += f'{(labels == submissions[name]).mean():>17.1%} '
        print(row)

    # group averages ------------------------------------------------------
    def stage_of(path: str) -> str:
        lowered = os.path.basename(path).lower()
        for stage in ('distil', 'pseudo2', 'pseudo1', 'pseudo', 'base'):
            if stage in lowered:
                return stage
        return 'other'

    groups: dict[str, list[str]] = {}
    for path, _, _ in usable:
        groups.setdefault(stage_of(path), []).append(path)
    groups['ALL'] = [path for path, _, _ in usable]

    probability_by_path = {path: p for path, p, _ in usable}
    ids_by_path = {path: i for path, _, i in usable}

    print('\n' + '=' * 100)
    print('AGREEMENT OF GROUP AVERAGES')
    print('=' * 100)
    print(f'{"group (n files)":<58}' + ''.join(f'{n[:16]:>18}' for n in names))
    for group, paths in groups.items():
        shaped = [p for p in paths if len(probability_by_path[p]) == n_rows]
        if len(shaped) < 1:
            continue
        stack = []
        for path in shaped:
            probabilities = probability_by_path[path]
            ids = ids_by_path[path]
            ordered = np.empty_like(probabilities)
            ordered[ids if ids is not None else np.arange(n_rows)] = probabilities
            stack.append(ordered)
        labels = np.mean(stack, axis=0).argmax(axis=1)
        row = f'{group + f" ({len(shaped)})":<58}'
        for name in names:
            row += f'{(labels == submissions[name]).mean():>17.1%} '
        print(row)

    # greedy subset search ------------------------------------------------
    if arguments.target:
        key = os.path.basename(arguments.target)
        if key not in submissions:
            print(f'\n{key} is not among the submissions found.')
            return
        target = submissions[key]
        print('\n' + '=' * 100)
        print(f'GREEDY SUBSET SEARCH FOR {key}')
        print('=' * 100)
        pool = []
        for path, probabilities, ids in usable:
            if len(probabilities) != n_rows:
                continue
            ordered = np.empty_like(probabilities)
            ordered[ids if ids is not None else np.arange(n_rows)] = probabilities
            pool.append((path, ordered))
        chosen, running, best = [], None, 0.0
        for _ in range(len(pool)):
            candidate_best, candidate_path, candidate_sum = best, None, None
            for path, ordered in pool:
                if path in chosen:
                    continue
                total = ordered if running is None else running + ordered
                score = float((total.argmax(axis=1) == target).mean())
                if score > candidate_best + 1e-9:
                    candidate_best, candidate_path, candidate_sum = score, path, total
            if candidate_path is None:
                break
            chosen.append(candidate_path)
            running, best = candidate_sum, candidate_best
            print(f'  + {os.path.relpath(candidate_path, arguments.root):<70} -> {best:.2%}')
        print(f'\nBest reachable agreement: {best:.2%} with {len(chosen)} file(s).')
        if best > 0.999:
            print('Exact. These files are the ensemble that wrote that submission.')
        elif best > 0.97:
            print('Close but not exact: one member is missing from disk, or the submission\n'
                  'also carries the manual corrections. Compare against the pure-model CSV.')
        else:
            print('Not reproducible from what is on disk. The winning stage was not cached,\n'
                  'so re-running it is the only way to get its probabilities back.')


if __name__ == '__main__':
    main()
