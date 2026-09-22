import numpy as np, pandas as pd
from pathlib import Path
from torchvision import datasets

R = Path.cwd()
ds = datasets.ImageFolder(R / 'FIT5215_Dataset')
names = [{'butterfiles': 'butterflies'}.get(n, n) for n in ds.classes]
t = np.asarray(ds.targets)
rng = np.random.default_rng(2026)
tr, va = [], []
for c in range(20):
    ci = np.flatnonzero(t == c); rng.shuffle(ci)
    n = max(1, int(round(len(ci) * 0.15)))
    va += ci[:n].tolist(); tr += ci[n:].tolist()
va = np.asarray(sorted(va)); tr = np.asarray(sorted(tr))
y = t[va]
prior = np.bincount(t[tr], minlength=20).astype(float); prior /= prior.sum()

runs = {}
for key, f in [('cnx', 'timm_convnext_tiny.in12k_ft_in1k_base_run.npz'),
               ('eff', 'timm_tf_efficientnetv2_s.in21k_ft_in1k_base_run.npz'),
               ('cnxP', 'timm_convnext_tiny.in12k_ft_in1k_pseudo1_run.npz'),
               ('effP', 'timm_tf_efficientnetv2_s.in21k_ft_in1k_pseudo1_run.npz')]:
    if (R / f).is_file():
        a = np.load(R / f)
        runs[key] = (a['validation_probabilities'], a['test_probabilities'], a['test_key_order'])
        print(key, 'val %.4f' % (runs[key][0].argmax(1) == y).mean())
    else:
        print(key, 'MISSING', f)

def temp(p, T):
    s = np.log(np.clip(p, 1e-12, 1)) / T; s -= s.max(1, keepdims=True)
    e = np.exp(s); return e / e.sum(1, keepdims=True)

def fitT(p):
    best, bn = 1.0, 1e9
    for T in np.linspace(0.4, 4.0, 73):
        n = -np.log(np.clip(temp(p, T)[np.arange(len(y)), y], 1e-12, 1)).mean()
        if n < bn: bn, best = n, float(T)
    return best

def em(p, src, it=200, floor=0.02):
    src = src / src.sum(); lo = floor * src; tgt = src.copy()
    for _ in range(it):
        a = p * (tgt / src); a /= a.sum(1, keepdims=True)
        u = np.maximum(a.mean(0), lo); u /= u.sum()
        if np.abs(u - tgt).max() < 1e-7: return u
        tgt = u
    return tgt

tpl = pd.read_csv(R / 'test_set' / 'my_solution.csv')

def write(tag, keys, fix):
    if any(k not in runs for k in keys):
        print('%-22s skipped' % tag); return
    v = sum(runs[k][0] for k in keys) / len(keys)
    p = sum(runs[k][1] for k in keys) / len(keys)
    o = runs[keys[0]][2]; note = ''
    if fix:
        T = fitT(v); p = temp(p, T); e = em(p, prior)
        p = p * (e / prior); p /= p.sum(1, keepdims=True)
        note = '  T=%.2f shift=%.3f' % (T, np.abs(e - prior).sum() / 2)
    d = pd.DataFrame({'ID': o.astype(int), 'Label': [names[i] for i in p.argmax(1)]})
    s = tpl[['ID']].merge(d, on='ID', how='left', validate='one_to_one')
    assert len(s) == len(tpl) and not s['Label'].isna().any()
    s.to_csv(R / ('submission_%s.csv' % tag), index=False)
    c = s['Label'].value_counts()
    print('%-22s val %.4f  counts %4d-%4d%s' % (tag, (v.argmax(1) == y).mean(), c.min(), c.max(), note))

write('A_cnx',          ['cnx'], False)
write('B_base',         ['cnx', 'eff'], False)
write('C_base_prior',   ['cnx', 'eff'], True)
write('D_cnx_prior',    ['cnx'], True)
write('E_pseudo',       ['cnxP', 'effP'], False)
write('F_pseudo_prior', ['cnxP', 'effP'], True)
write('G_all',          ['cnx', 'eff', 'cnxP', 'effP'], False)
write('H_all_prior',    ['cnx', 'eff', 'cnxP', 'effP'], True)
print('done')

write('I_cnxP', ['cnxP'], False)
write('J_both_cnx', ['cnx', 'cnxP'], False)
