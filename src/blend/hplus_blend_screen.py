"""Pre-registered blend screen for the ViT-H+ pseudo1 model (CLAUDE.md 2026-09-21).

Candidates fixed before any of them was computed: V2P+HP, EVAP+HP, V2P+EVAP+HP.
A CSV is written only if the candidate differs from the current best on k >= 60 rows AND its
rendition-reserve accuracy exceeds every one of its members. Nothing here is fitted on the LB.
"""
import numpy as np, pandas as pd

CN = np.load('bench/class_names.npy', allow_pickle=True)
LAB = np.load('bench/reserve_preds.npz')['lab']
BEST = pd.read_csv('submission_ens_V2PEVAPtemp.csv')['Label'].values
MEMBERS = {'V2P': 'dinov2vitlps1', 'EVAP': 'eva02lps1', 'V3P': 'dinov3vitlps1',
           'HP': 'dinov3vithpps1'}
CANDIDATES = [['V2P', 'HP'], ['EVAP', 'HP'], ['V2P', 'EVAP', 'HP']]

nz = lambda p: (p := np.asarray(p, np.float64)) / p.sum(1, keepdims=True)          # noqa: E731

def sharpen(p, T):
    z = np.log(np.clip(p, 1e-12, None)) / T
    z -= z.max(1, keepdims=True); e = np.exp(z)
    return e / e.sum(1, keepdims=True)

def fit_T(p, target, lo=0.05, hi=20.0):
    for _ in range(200):
        m = 0.5 * (lo + hi)
        if sharpen(p, m).max(1).mean() > target: lo = m
        else: hi = m
    return 0.5 * (lo + hi)

def blend(tags, kind):
    ps = [nz(np.load(f'bench/{MEMBERS[t]}_{kind}_prob.npy')) for t in tags]
    target = ps[0].max(1).mean()
    temps = [1.0] + [fit_T(p, target) for p in ps[1:]]
    return sum(p if T == 1.0 else sharpen(p, T) for p, T in zip(ps, temps)) / len(ps), temps

print(f'{"candidate":24s} {"reserve":>8s} {"best member":>12s} {"exceeds":>8s} {"k vs best":>10s}  verdict')
for tags in CANDIDATES:
    # temperatures come from the TEST probabilities -- that is the blend that would be submitted
    test_b, temps = blend(tags, 'test')
    ps_r = [nz(np.load(f'bench/{MEMBERS[t]}_reserve_prob.npy')) for t in tags]
    res_b = sum(p if T == 1.0 else sharpen(p, T) for p, T in zip(ps_r, temps)) / len(ps_r)
    acc = 100 * (res_b.argmax(1) == LAB).mean()
    member_acc = {t: 100 * (np.load(f'bench/{MEMBERS[t]}_reserve_prob.npy').argmax(1) == LAB).mean()
                  for t in tags}
    best_member = max(member_acc.values())
    k = int((CN[test_b.argmax(1)] != BEST).sum())
    exceeds = acc > best_member
    ok = exceeds and k >= 60
    name = ' + '.join(tags)
    print(f'{name:24s} {acc:7.2f}% {best_member:11.2f}% {"yes" if exceeds else "NO":>8s} {k:10d}  '
          f'{"PASS" if ok else "FAIL"}   members: ' +
          ', '.join(f'{t} {a:.2f}' for t, a in member_acc.items()) +
          '  T=' + ','.join(f'{t:.4f}' for t in temps))
    if ok:
        out = f'submission_ens_{"".join(tags)}temp.csv'
        pd.DataFrame({'ID': np.arange(len(test_b)), 'Label': CN[test_b.argmax(1)]}).to_csv(out, index=False)
        print(f'    -> {out} written')
