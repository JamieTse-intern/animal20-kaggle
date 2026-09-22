"""Report a candidate model, and optionally its pre-registered blend, before any submission.

Usage:
    python candidate_report.py --tag siglip2vitl_base --gate 0.9700
    python candidate_report.py --tag siglip2vitlps1 --blend-with dinov2vitlps1 \
                               --write submission_ens_V2PSGPtemp.csv

File convention (what dinov3_base_train.py / dinov3_pseudo_train.py write):
    bench/<tag>_test_prob.npy, bench/<tag>_reserve_prob.npy, submission_<tag>.csv

Blend rule, pre-registered 2026-09-15 before any SigLIP result existed:
    temperature-matched equal blend with the reference member listed in --blend-with;
    a CSV is written ONLY if k >= 60 vs pick 1 AND the blend's rendition-reserve accuracy
    exceeds both members. The photo holdout is deliberately not consulted.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from blend_temp import fit_temperature, sharpen

PICK1_CSV, PICK1_LB = 'submission_ens_V2PV3Ptemp.csv', 0.98132
REFERENCES = {'pick 1 (V2P+V3P)': PICK1_CSV,
              'V2P  (0.97950)': 'submission_dinov2vitlps1.csv',
              'V3T base (0.97556)': 'submission_dinov3vitl_base.csv'}
CN = list(np.load('bench/class_names.npy', allow_pickle=True))


def norm(p):
    p = np.asarray(p, dtype=np.float64)
    return p / p.sum(1, keepdims=True)


def labels_of(prob):
    return np.array([CN[i] for i in prob.argmax(1)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--gate', type=float, default=None, help='pre-registered public-LB gate')
    ap.add_argument('--blend-with', default=None, help='reference member tag for the blend')
    ap.add_argument('--write', default=None, help='blend CSV path, written only if rules pass')
    a = ap.parse_args()

    prob = norm(np.load(f'bench/{a.tag}_test_prob.npy'))
    csv_path = Path(f'submission_{a.tag}.csv')
    s = pd.read_csv(csv_path)
    fmt_ok = (list(s.columns) == ['ID', 'Label'] and len(s) == 11681
              and s['ID'].tolist() == list(range(11681)) and set(s['Label']) <= set(CN)
              and (s['Label'].values == labels_of(prob)).all())
    print(f'{csv_path}: format {"OK" if fmt_ok else "BROKEN"}; matches its probability array: '
          f'{(s["Label"].values == labels_of(prob)).all()}')
    print(f'mean top-1 confidence {prob.max(1).mean():.4f}')

    mine = labels_of(prob)
    print('\nrows that differ:')
    for name, ref in REFERENCES.items():
        k = int((pd.read_csv(ref)['Label'].values != mine).sum())
        print(f'  vs {name:20s} k = {k:4d} ({100 * k / 11681:.2f}%)')

    moved = pd.read_csv(PICK1_CSV)['Label'].values != mine
    c = prob.max(1)[moved]
    if moved.any():
        print('confidence of the rows that differ from pick 1:')
        for lo, hi in [(0, .5), (.5, .7), (.7, .9), (.9, 1.01)]:
            n = int(((c >= lo) & (c < hi)).sum())
            print(f'    [{lo:.1f},{hi:.1f}): {n:4d} ({100 * n / moved.sum():.1f}%)')

    lab = np.load('bench/reserve_preds.npz')['lab']
    reserve_path = Path(f'bench/{a.tag}_reserve_prob.npy')
    if reserve_path.is_file():
        acc = 100 * (np.load(reserve_path).argmax(1) == lab).mean()
        print(f'\nrendition reserve (772): {acc:.2f}%   '
              f'(V3T base 92.88%, V2 base 93.78%, V2P 93.01%, V3P 94.30%)')
    if a.gate is not None:
        print(f'\nPRE-REGISTERED GATE: public LB must be >= {a.gate:.5f} to continue.')

    if not a.blend_with:
        return 0

    # ---- the pre-registered blend ---------------------------------------------------------
    ref = a.blend_with
    t_ref, t_new = norm(np.load(f'bench/{ref}_test_prob.npy')), prob
    T = fit_temperature(t_new, t_ref.max(1).mean())
    blend_test = (t_ref + sharpen(t_new, T)) / 2
    r_ref = norm(np.load(f'bench/{ref}_reserve_prob.npy'))
    r_new = norm(np.load(reserve_path))
    blend_res = (r_ref + sharpen(r_new, T)) / 2
    acc_ref = 100 * (r_ref.argmax(1) == lab).mean()
    acc_new = 100 * (r_new.argmax(1) == lab).mean()
    acc_bl = 100 * (blend_res.argmax(1) == lab).mean()
    exceeds = acc_bl > max(acc_ref, acc_new)
    bl_labels = labels_of(blend_test)
    k = int((pd.read_csv(PICK1_CSV)['Label'].values != bl_labels).sum())
    need = 0.16 * k / 9345
    print(f'\nBLEND {ref} + {a.tag}, temperature {T:.4f} on {a.tag}')
    print(f'  rendition reserve: {ref} {acc_ref:.2f}%, {a.tag} {acc_new:.2f}%, blend {acc_bl:.2f}% '
          f'-> {"EXCEEDS both" if exceeds else "does NOT exceed both"} '
          f'({acc_bl - max(acc_ref, acc_new):+.2f}pt)')
    print(f'  k vs pick 1 = {k}; a signal over pick 1 needs public >= {PICK1_LB + need:.5f}')
    if k >= 60 and exceeds:
        if a.write:
            pd.DataFrame({'ID': np.arange(11681), 'Label': bl_labels}).to_csv(a.write, index=False)
            print(f'  RULES PASS -> wrote {a.write}')
        else:
            print('  RULES PASS (pass --write to create the CSV)')
    else:
        why = []
        if k < 60:
            why.append(f'k = {k} < 60 (could only ever read as a tie)')
        if not exceeds:
            why.append('blend does not exceed both members on renditions')
        print('  RULES FAIL -> no CSV written: ' + '; '.join(why))
    return 0


if __name__ == '__main__':
    sys.exit(main())
