"""Screen the pre-registered V3P2 blends (CLAUDE.md 2026-09-17, new student from the 0.98196
teacher) without spending a submission. Same construction and rule as evap_blend_screen.py."""
import numpy as np
import pandas as pd

from evap_blend_screen import BEST_CSV, BEST_LB, LAB, build, labels, reserve

CANDIDATES = [['V2P', 'EVAP', 'V3P2'], ['V2P', 'V3P2'], ['EVAP', 'V3P2']]


def main():
    best = pd.read_csv(BEST_CSV)['Label'].values
    v3p2 = np.load('bench/dinov3vitlps1b_test_prob.npy').argmax(1)
    teacher = np.load('bench/teacher_V2PEVAP.npy').argmax(1)
    old = np.load('bench/dinov3vitlps1_test_prob.npy').argmax(1)
    print(f'V3P2 alone: disagreement with its teacher {100 * (v3p2 != teacher).mean():.2f}%, '
          f'with V3P {100 * (v3p2 != old).mean():.2f}%, '
          f'reserve {100 * (reserve("V3P2").argmax(1) == LAB).mean():.2f}% '
          f'(V3P {100 * (reserve("V3P").argmax(1) == LAB).mean():.2f}%)')
    for tags in CANDIDATES:
        out, res, temps = build(tags)
        lab = labels(out)
        k = int((lab != best).sum())
        member_acc = {t: 100 * (reserve(t).argmax(1) == LAB).mean() for t in tags}
        acc = 100 * (res.argmax(1) == LAB).mean()
        exceeds = acc > max(member_acc.values())
        print(f'\n{" + ".join(tags)}   T {["%.3f" % T for T in temps]}')
        print('  reserve: ' + ', '.join(f'{t} {a:.2f}%' for t, a in member_acc.items())
              + f' -> blend {acc:.2f}% ({"EXCEEDS all" if exceeds else "does NOT exceed all"}, '
              f'{acc - max(member_acc.values()):+.2f}pt)')
        print(f'  k vs current best = {k} (signal needs >= {BEST_LB + 0.16 * k / 9345:.5f})')
        if k >= 60 and exceeds:
            name = 'submission_ens_' + ''.join(tags) + 'temp.csv'
            pd.DataFrame({'ID': np.arange(len(lab)), 'Label': lab}).to_csv(name, index=False)
            print(f'  SCREEN PASSES -> wrote {name}')
        else:
            why = ([f'k = {k} < 60'] if k < 60 else []) + ([] if exceeds else ['does not exceed all members'])
            print('  SCREEN FAILS: ' + '; '.join(why))


if __name__ == '__main__':
    main()
