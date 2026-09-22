"""Temperature-matched equal-weight blending (the plan's section 3.2 construction).

No parameter is fitted on the leaderboard: each member's temperature is solved so its
mean top-1 confidence equals the reference member's, then the members are averaged with
equal weight.  Reference member is the first one listed.

Usage:
    python blend_temp.py --members A C D3 --out submission_ens_ACD3temp.csv
    python blend_temp.py --members A C --out /tmp/check.csv --verify submission_ens_ACtemp.csv
"""
import argparse
import os

import numpy as np
import pandas as pd

BENCH = 'bench'
MEMBERS = {
    'A':  'cnv2L_test_prob.npy',            # 192px pseudo1, 0.97342
    'C':  'cnv2L_res256_test_prob.npy',     # 256px pseudo1, 0.97203
    'B':  'cnv2L_trio_bal_test_prob.npy',   # 192px trio_bal030, 0.97257
    'D':  'cnv2L_res256_base_test_prob.npy',  # 256px base, 0.96350
    'E':  'cnv2L_thr060_test_prob.npy',     # thr 0.60 pseudo1, 0.97054
    'FM': 'fixmatch_test_prob.npy',         # FixMatch, 0.96926
    'D3': 'dinov3ps1_test_prob.npy',        # DINOv3 ConvNeXt-L pseudo1, 0.97171
    'D3B': 'dinov3flat_base_test_prob.npy',  # DINOv3 base, 0.96264
    'KD': 'softkd_test_prob.npy',           # soft-label KD student, 0.97524
    'V2': 'dinov2vitl_base_test_prob.npy',  # DINOv2 ViT-L/14 base, 0.97363
    'V2P': 'dinov2vitlps1_test_prob.npy',   # DINOv2 ViT-L/14 pseudo1, 0.97950 -- strongest
    'V3T': 'dinov3vitl_base_test_prob.npy', # DINOv3 ViT-L/16 base, 0.97556
    'V3P': 'dinov3vitlps1_test_prob.npy',   # DINOv3 ViT-L/16 pseudo1
    'EVA': 'eva02l_base_test_prob.npy',     # EVA-02 L base, 0.96915
    'EVAP': 'eva02lps1_test_prob.npy',      # EVA-02 L pseudo1 (teacher V2P+V3P), 0.98132
    'V3P2': 'dinov3vitlps1b_test_prob.npy', # DINOv3 ViT-L/16 pseudo1, teacher V2P+EVAP (0.98196)
}
EPS = 1e-12


def load(tag):
    p = np.load(os.path.join(BENCH, MEMBERS[tag])).astype(np.float64)
    return p / p.sum(1, keepdims=True)


def sharpen(p, T):
    """softmax(log p / T) -- T < 1 sharpens, T > 1 blunts."""
    z = np.log(np.clip(p, EPS, None)) / T
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_temperature(p, target_conf, lo=0.05, hi=20.0, iters=200):
    """Solve for T such that mean top-1 confidence of sharpen(p, T) == target_conf.

    Mean top-1 confidence is monotone decreasing in T, so plain bisection.
    """
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if sharpen(p, mid).max(1).mean() > target_conf:
            lo = mid          # too sharp -> need larger T
        else:
            hi = mid
    return 0.5 * (lo + hi)


def blend(tags, mode='geometric', weights=None, temp_match=True):
    probs = [load(t) for t in tags]
    target = probs[0].max(1).mean()          # reference sharpness = first member's
    temps, matched = [], []
    for t, p in zip(tags, probs):
        T = 1.0 if (not temp_match or len(matched) == 0) else fit_temperature(p, target)
        temps.append(T)
        matched.append(sharpen(p, T) if T != 1.0 else p)
    w = np.ones(len(tags)) if weights is None else np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    if mode == 'geometric':
        z = sum(wi * np.log(np.clip(m, EPS, None)) for wi, m in zip(w, matched))
        z -= z.max(1, keepdims=True)
        e = np.exp(z)
        out = e / e.sum(1, keepdims=True)
    else:
        out = sum(wi * m for wi, m in zip(w, matched))
    return out, temps, matched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--members', nargs='+', required=True)
    ap.add_argument('--weights', nargs='+', type=float, default=None)
    ap.add_argument('--mode', default='geometric', choices=['geometric', 'arithmetic'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--baseline', default='submission_ens_AClogit1to2.csv',
                    help='current pick 1, for the k / disagreement report')
    ap.add_argument('--verify', default=None,
                    help='existing CSV this construction should reproduce row-for-row')
    ap.add_argument('--save-prob', default=None)
    ap.add_argument('--no-temp', action='store_true',
                    help='skip temperature matching (the plain weighted-geometric lineage '
                         'that produced submission_ens_AClogit1to2.csv)')
    args = ap.parse_args()

    classes = list(np.load(os.path.join(BENCH, 'class_names.npy'), allow_pickle=True))
    out, temps, matched = blend(args.members, args.mode, args.weights,
                                temp_match=not args.no_temp)

    print(f'mode {args.mode}  members {args.members}  '
          f'weights {"equal" if args.weights is None else args.weights}')
    for t, T, m in zip(args.members, temps, matched):
        raw = load(t).max(1).mean()
        print(f'  {t:>4}  T {T:7.4f}   conf {raw:.4f} -> {m.max(1).mean():.4f}')
    print(f'  blend mean top-1 confidence {out.max(1).mean():.4f}')

    pred = out.argmax(1)
    df = pd.DataFrame({'ID': np.arange(len(pred)), 'Label': [classes[i] for i in pred]})
    df.to_csv(args.out, index=False)
    print(f'wrote {args.out}')
    if args.save_prob:
        np.save(args.save_prob, out.astype(np.float32))
        print(f'wrote {args.save_prob}')

    if args.verify:
        ref = pd.read_csv(args.verify)
        d = int((ref['Label'].values != df['Label'].values).sum())
        print(f'VERIFY vs {args.verify}: {d} rows differ '
              f'({"EXACT MATCH" if d == 0 else "MISMATCH"})')

    if args.baseline and os.path.exists(args.baseline):
        base = pd.read_csv(args.baseline)
        diff = base['Label'].values != df['Label'].values
        k = int(diff.sum())
        conf = out.max(1)[diff]
        print(f'\nvs {args.baseline}:  k = {k} ({100*k/len(pred):.2f}%)')
        if k:
            bins = [(0, .5), (.5, .7), (.7, .9), (.9, 1.01)]
            for lo, hi in bins:
                n = int(((conf >= lo) & (conf < hi)).sum())
                print(f'    conf [{lo:.1f},{hi:.1f}): {n:4d} ({100*n/k:.1f}%)')
            print(f'    median conf on the k rows {np.median(conf):.3f} '
                  f'vs {np.median(out.max(1)):.3f} overall')

    for t in args.members:
        p = load(t)
        d = float((p.argmax(1) != pred).mean())
        print(f'  disagreement blend vs {t}: {100*d:.2f}%')


if __name__ == '__main__':
    main()
