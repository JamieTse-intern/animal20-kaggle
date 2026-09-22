"""Filter bench/external_images.npy on per-image *spectrum* distance to the test
set, not just Laplacian variance.

Line art fails the radial-spectrum check even when its total edge energy is matched,
because its energy sits at the stroke frequency with empty background. This ranks
each external image by how close its own radial spectrum is to the test mean profile
and keeps the closest ones, which is what the aggregate check actually measures.
"""
import sys
import numpy as np

_GREY = np.array([0.299, 0.587, 0.114], np.float32)


def radial_profiles(batch_nhwc, chunk=2000):
    """Per-image radially averaged log power spectrum -> (N, 31)."""
    out = []
    size = batch_nhwc.shape[1]; c = size // 2
    r, cc = np.mgrid[:size, :size]
    radius = np.sqrt((r - c) ** 2 + (cc - c) ** 2).astype(int)
    masks = [radius == k for k in range(1, c)]
    for s in range(0, len(batch_nhwc), chunk):
        b = batch_nhwc[s:s + chunk]
        g = (b.astype(np.float32) / 255.0) @ _GREY
        g = g - g.mean((1, 2), keepdims=True)
        p = np.abs(np.fft.fftshift(np.fft.fft2(g), axes=(1, 2))) ** 2
        prof = np.stack([p[:, m].mean(1) for m in masks], axis=1)
        prof = prof / np.maximum(prof[:, :8].mean(1, keepdims=True), 1e-12)
        out.append(np.log10(prof + 1e-12))
    return np.concatenate(out)


def agg_spectrum(batch_nhwc):
    g = (batch_nhwc.astype(np.float32) / 255.0) @ _GREY
    g = g - g.mean((1, 2), keepdims=True)
    p = np.abs(np.fft.fftshift(np.fft.fft2(g), axes=(1, 2))) ** 2
    size = g.shape[1]; c = size // 2
    r, cc = np.mgrid[:size, :size]
    radius = np.sqrt((r - c) ** 2 + (cc - c) ** 2).astype(int)
    prof = np.array([p[:, radius == k].mean() for k in range(1, c)])
    prof = prof / prof[:8].mean()
    return np.log10(prof + 1e-12)


def lapvar(b):
    g = (b.astype(np.float32) / 255.0) @ _GREY
    l = (-4 * g[:, 1:-1, 1:-1] + g[:, :-2, 1:-1] + g[:, 2:, 1:-1]
         + g[:, 1:-1, :-2] + g[:, 1:-1, 2:])
    return l.reshape(len(b), -1).var(1)


ext = np.load('bench/external_images.npy')
lab = np.load('bench/external_labels.npy')
en = np.ascontiguousarray(ext.transpose(0, 2, 3, 1))
test = np.ascontiguousarray(np.asarray(
    np.load('bench/test_images.npy', mmap_mode='r')[:3000]).transpose(0, 2, 3, 1))

test_prof = agg_spectrum(test)
test_lv = lapvar(test).mean()
ep = radial_profiles(en)
dist = np.abs(ep - test_prof[None, :]).mean(1)      # per-image spectrum distance
print(f'external {len(ext)} images; per-image spectrum distance '
      f'min {dist.min():.3f} median {np.median(dist):.3f} max {dist.max():.3f}')

MIN_PER_CLASS = int(sys.argv[1]) if len(sys.argv) > 1 else 40
best = None
for frac in np.arange(0.05, 1.01, 0.05):
    k = max(1, int(len(dist) * frac))
    keep = np.zeros(len(dist), bool)
    keep[np.argsort(dist)[:k]] = True
    # top up any class that fell below MIN_PER_CLASS with its own closest images
    for c in np.unique(lab):
        have = (lab[keep] == c).sum()
        if have < MIN_PER_CLASS:
            pool = np.flatnonzero((lab == c) & ~keep)
            keep[pool[np.argsort(dist[pool])][:MIN_PER_CLASS - have]] = True
    sub = en[keep]
    sd = float(np.abs(agg_spectrum(sub) - test_prof).mean())
    lv = lapvar(sub).mean()
    rel = abs(lv - test_lv) / test_lv * 100
    ok = sd <= 0.05 and rel <= 25
    print(f'  keep {frac:4.0%} ({keep.sum():5d}) -> spectrum {sd:.4f}  lapvar rel {rel:5.1f}%  {"PASS" if ok else ""}')
    if ok:
        best = (keep.copy(), sd, rel)

if best is None:
    print('\nFAIL: no subset of this source passes the spectrum check. Skip this source.')
    sys.exit(1)
keep, sd, rel = best
np.save('bench/external_images.npy', np.ascontiguousarray(ext[keep]))
np.save('bench/external_labels.npy', lab[keep])
print(f'\nPASS: kept {keep.sum()} of {len(ext)}; spectrum {sd:.4f}, lapvar rel {rel:.1f}%')
print('per-class:', {int(c): int((lab[keep] == c).sum()) for c in np.unique(lab[keep])})
