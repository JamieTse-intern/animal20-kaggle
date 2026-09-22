"""Screen the pre-registered EVAP blends (CLAUDE.md 2026-09-17) without spending a submission.

Construction is blend_temp.blend(..., mode='arithmetic'): temperatures fitted on the test set
against the first member's sharpness, equal weights. The same temperatures are applied to each
member's rendition-reserve probabilities. A CSV is written only if the candidate differs from
the current best on k >= 60 rows AND its reserve accuracy exceeds every one of its members.
"""
import numpy as np
import pandas as pd

from blend_temp import MEMBERS, blend, load, sharpen

BEST_CSV, BEST_LB = 'submission_ens_V2PEVAPtemp.csv', 0.98196
CANDIDATES = [['EVAP', 'V3P'], ['EVAP', 'V3T'], ['V2P', 'V3P', 'EVAP'], ['V2P', 'V3T', 'EVAP'],
              ['V2P', 'V3P', 'V3T', 'EVAP']]
CN = list(np.load('bench/class_names.npy', allow_pickle=True))
LAB = np.load('bench/reserve_preds.npz')['lab']


def reserve(tag):
    p = np.load('bench/' + MEMBERS[tag].replace('_test_prob', '_reserve_prob')).astype(np.float64)
    return p / p.sum(1, keepdims=True)


def labels(prob):
    return np.array([CN[i] for i in prob.argmax(1)])


def build(tags):
    out, temps, _ = blend(tags, mode='arithmetic')
    res = sum(sharpen(reserve(t), T) if T != 1.0 else reserve(t) for t, T in zip(tags, temps)) / len(tags)
    return out, res, temps


def main():
    best = pd.read_csv(BEST_CSV)['Label'].values
    out, _, _ = build(['V2P', 'EVAP'])
    d = int((labels(out) != best).sum())
    print(f'sanity: V2P+EVAP rebuilt vs {BEST_CSV}: {d} rows differ ({"EXACT" if d == 0 else "MISMATCH"})')
    assert d == 0, 'construction does not reproduce the submitted blend'
    picks = {'pick 1 V2PV3P': 'submission_ens_V2PV3Ptemp.csv'}

    for tags in CANDIDATES:
        out, res, temps = build(tags)
        lab = labels(out)
        k = int((lab != best).sum())
        k_old = int((lab != pd.read_csv(picks['pick 1 V2PV3P'])['Label'].values).sum())
        member_acc = {t: 100 * (reserve(t).argmax(1) == LAB).mean() for t in tags}
        acc = 100 * (res.argmax(1) == LAB).mean()
        exceeds = acc > max(member_acc.values())
        name = 'submission_ens_' + ''.join(tags) + 'temp.csv'
        passed = k >= 60 and exceeds
        print(f'\n{" + ".join(tags)}   T {["%.3f" % T for T in temps]}')
        print('  reserve: ' + ', '.join(f'{t} {a:.2f}%' for t, a in member_acc.items())
              + f' -> blend {acc:.2f}% ({"EXCEEDS all" if exceeds else "does NOT exceed all"}, '
              f'{acc - max(member_acc.values()):+.2f}pt)')
        print(f'  k vs current best = {k} (signal needs >= {BEST_LB + 0.16 * k / 9345:.5f}); '
              f'k vs old pick 1 = {k_old}')
        if passed:
            pd.DataFrame({'ID': np.arange(len(lab)), 'Label': lab}).to_csv(name, index=False)
            print(f'  SCREEN PASSES -> wrote {name}')
        else:
            why = ([f'k = {k} < 60'] if k < 60 else []) + ([] if exceeds else ['does not exceed all members'])
            print('  SCREEN FAILS: ' + '; '.join(why))


if __name__ == '__main__':
    main()
