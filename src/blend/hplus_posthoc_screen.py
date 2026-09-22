"""Post-hoc blend screen around EVAP+HP (0.98324). Criteria fixed in CLAUDE.md before running."""
import numpy as np, pandas as pd

CN = np.load('bench/class_names.npy', allow_pickle=True)
LAB = np.load('bench/reserve_preds.npz')['lab']
BEST = pd.read_csv('submission_ens_EVAPHPtemp.csv')['Label'].values
M = {'V2P': 'dinov2vitlps1', 'EVAP': 'eva02lps1', 'V3P': 'dinov3vitlps1',
     'V3P2': 'dinov3vitlps1b', 'HP': 'dinov3vithpps1'}
CANDIDATES = [['EVAP','HP','V3P'], ['EVAP','HP','V3P2'], ['V2P','EVAP','HP'],
              ['V3P','HP'], ['V3P2','HP'], ['EVAP','HP','V2P','V3P']]

def nz(p):
    p = np.asarray(p, np.float64); return p / p.sum(1, keepdims=True)

def sharpen(p, T):
    z = np.log(np.clip(p, 1e-12, None)) / T
    z -= z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)

def fit_T(p, target, lo=0.05, hi=20.0):
    for _ in range(200):
        m = 0.5*(lo+hi)
        if sharpen(p, m).max(1).mean() > target: lo = m
        else: hi = m
    return 0.5*(lo+hi)

print(f'{"candidate":28s} {"reserve":>8s} {"best mbr":>9s} {"exceeds":>8s} {"k":>5s}  verdict')
for tags in CANDIDATES:
    ps = [nz(np.load(f'bench/{M[t]}_test_prob.npy')) for t in tags]
    target = ps[0].max(1).mean()
    temps = [1.0] + [fit_T(p, target) for p in ps[1:]]
    test_b = sum(p if T == 1.0 else sharpen(p, T) for p, T in zip(ps, temps)) / len(ps)
    pr = [nz(np.load(f'bench/{M[t]}_reserve_prob.npy')) for t in tags]
    res = sum(p if T == 1.0 else sharpen(p, T) for p, T in zip(pr, temps)) / len(pr)
    acc = 100*(res.argmax(1) == LAB).mean()
    macc = {t: 100*(np.load(f'bench/{M[t]}_reserve_prob.npy').argmax(1) == LAB).mean() for t in tags}
    k = int((CN[test_b.argmax(1)] != BEST).sum())
    ok = acc > max(macc.values()) and k >= 60
    print(f'{" + ".join(tags):28s} {acc:7.2f}% {max(macc.values()):8.2f}% '
          f'{"yes" if acc > max(macc.values()) else "NO":>8s} {k:5d}  {"PASS" if ok else "FAIL"}'
          f'   needs >= {0.98324 + 0.16*k/9345:.5f}' if ok else
          f'{" + ".join(tags):28s} {acc:7.2f}% {max(macc.values()):8.2f}% '
          f'{"yes" if acc > max(macc.values()) else "NO":>8s} {k:5d}  FAIL')
    if ok:
        out = f'submission_ens_{"".join(tags)}temp.csv'
        pd.DataFrame({'ID': np.arange(len(test_b)), 'Label': CN[test_b.argmax(1)]}).to_csv(out, index=False)
        print(f'    -> {out} written')
