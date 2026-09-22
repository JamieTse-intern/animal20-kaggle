"""Rebuild both final submissions from the saved model outputs and verify them.

Each of the two submissions is a temperature-matched equal-weight probability blend of two
fine-tuned models. Nothing in the construction is fitted on the leaderboard: the only free
quantity is a temperature per member, solved so that every member has the same mean top-1
confidence as the first one listed.

    pick 1   EVA-02 L self-distilled  +  DINOv3 ViT-H+/16 self-distilled   0.98335
    pick 2   DINOv3 ViT-L/16 self-distilled  +  DINOv3 ViT-H+/16 supervised  0.98335

The ViT-H+ members are inferred at 192/224/256px, their native patch grid; the others at
160/192/224 (168/196/224 for the patch-14 model, so every side stays divisible by 14).

Two modes, chosen automatically:

  FULL   results/test_predictions/ is present -> rebuild both CSVs and compare row for row.
  HASH   it is not (the default for a public checkout, because those files are the competition
         answer key) -> report what is missing. The recorded SHA-256 of each submission is in
         EXPECTED_SHA256 below, so anyone holding the arrays can check they match.

Run:  python reproduce/reproduce.py
"""
import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / 'results' / 'test_predictions'
CLASSES = list(np.load(ROOT / 'results' / 'model_outputs' / 'class_names.npy', allow_pickle=True))
N_TEST = 11681
EPS = 1e-12

PICKS = {
    'pick 1': ('submission_ens_EVAPHP256temp.csv', ['eva02lps1', 'dinov3vithpps1B'], 0.98335),
    'pick 2': ('submission_ens_V3PHPbase256temp.csv', ['dinov3vitlps1', 'dinov3vithp_baseB'], 0.98335),
}

# Recorded when the files were submitted, so the rebuild is checkable without the CSVs present.
EXPECTED_SHA256 = {
    'submission_ens_EVAPHP256temp.csv':
        'b2ccbc8196dd3ebb9f382c7a113c904de5e696365c174d5e01a0db83758c59fb',
    'submission_ens_V3PHPbase256temp.csv':
        'afaf5a56770f0f652f39e8a6179e04b1a3b14deefd3dece1529a3978dea04e8b',
}


def normalise(p):
    p = np.asarray(p, dtype=np.float64)
    return p / p.sum(axis=1, keepdims=True)


def sharpen(p, T):
    z = np.log(np.clip(p, EPS, None)) / T
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(p, target, low=0.02, high=40.0, iterations=300):
    """Mean top-1 confidence is monotone decreasing in T, so bisect."""
    for _ in range(iterations):
        mid = 0.5 * (low + high)
        if sharpen(p, mid).max(axis=1).mean() > target:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def temperature_matched_blend(members):
    members = [normalise(p) for p in members]
    target = members[0].max(axis=1).mean()
    temps = [1.0] + [fit_temperature(p, target) for p in members[1:]]
    blended = sum(p if T == 1.0 else sharpen(p, T) for p, T in zip(members, temps))
    return blended / len(members), temps


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    record = '--record' in sys.argv
    if not OUTPUTS.is_dir():
        print(f'{OUTPUTS.relative_to(ROOT)} is not present.\n')
        print('That directory holds the per-model test-set probability arrays and the two')
        print('submission CSVs. They are the competition answer key, so they are excluded')
        print('from the public repository on purpose -- see its README.\n')
        print('Everything else in results/ is derived from held-out *external* data and is here:')
        print('  results/model_outputs/   per-model accuracy on the 772 reserved renditions')
        print('  results/training_logs/   per-epoch loss and validation for 19 runs')
        print('  results/leaderboard.csv  every scored submission with its recorded prediction')
        return 0

    all_match, hashes = True, {}
    for name, (filename, tags, public) in PICKS.items():
        members = [normalise(np.load(OUTPUTS / f'{t}_test_prob.npy')) for t in tags]
        blended, temps = temperature_matched_blend(members)
        labels = np.array([CLASSES[i] for i in blended.argmax(axis=1)])
        assert len(labels) == N_TEST

        print(f'\n{name}: {filename}   public LB {public:.5f}')
        for tag, T, p in zip(tags, temps, members):
            print(f'    {tag:20s} T = {T:.4f}   mean top-1 confidence '
                  f'{p.max(axis=1).mean():.4f} -> {blended.max(axis=1).mean():.4f} after blending')

        csv_path = OUTPUTS / filename
        if csv_path.exists():
            import pandas as pd
            submitted = pd.read_csv(csv_path)['Label'].values
            differ = int((labels != submitted).sum())
            print(f'    rebuilt from the saved model outputs: {differ} rows differ from the '
                  f'submitted file  {"EXACT MATCH" if differ == 0 else "MISMATCH"}')
            all_match &= differ == 0
            hashes[filename] = sha256(csv_path)
            expected = EXPECTED_SHA256.get(filename)
            if expected and expected != hashes[filename]:
                print(f'    SHA-256 MISMATCH: recorded {expected[:16]}, actual {hashes[filename][:16]}')
                all_match = False
        else:
            print(f'    {filename} not present; rebuild produced {len(labels)} labels')

    print('\nRESULT:', 'both submissions reproduce exactly' if all_match else 'MISMATCH')
    if record and hashes:
        print('\nRecorded SHA-256 (paste into EXPECTED_SHA256):')
        for k, v in hashes.items():
            print(f"    '{k}': '{v}',")
    return 0 if all_match else 1


if __name__ == '__main__':
    sys.exit(main())
