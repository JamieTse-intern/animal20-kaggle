r"""profile_composition.py -- does subject scale explain the rest of the instability?

Desaturation turned out to be real but small: greyscale images are unstable at 9.6% against
4.5% for colour ones, a 2.1x effect covering only about a quarter of the unstable set. The
eyeball impression of the hardest images was that the subject occupies a small part of a
large, busy scene. This measures that, and two neighbouring properties, on all 21k images.

Metrics, all computed on the 64x64 arrays with numpy only:

  center_frac   fraction of total gradient energy inside the central 32x32 box.
                A subject filling the frame concentrates energy centrally; a small subject
                in a wide scene does not. The central box is 25% of the area, so a value
                near 0.25 means energy is spread uniformly.
  spread        gradient-energy radius of gyration about the image centre, in units of half
                the image width. Higher means structure reaching to the edges.
  border_energy mean gradient magnitude in the outer 8 pixel ring. High means a busy
                background rather than a plain one.
  detail        Laplacian variance over the whole image, the usual sharpness proxy.
  entropy       Shannon entropy of the 32-bin grey histogram, a scene-complexity proxy.

    python profile_composition.py
    python profile_composition.py --profile hard_image_profile.csv --out composition.csv
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

BENCH = 'bench'


def to_float_hwc(images: np.ndarray) -> np.ndarray:
    """(N, 3, 64, 64) or (N, 64, 64, 3), any dtype -> float32 (N, 64, 64, 3) in 0..1."""
    array = images
    if array.shape[1] == 3 and array.ndim == 4:
        array = array.transpose(0, 2, 3, 1)
    array = array.astype(np.float32)
    if array.max() > 1.5:
        array = array / 255.0
    return array


def grey(images: np.ndarray) -> np.ndarray:
    return (0.299 * images[..., 0] + 0.587 * images[..., 1] + 0.114 * images[..., 2])


def gradient_magnitude(g: np.ndarray) -> np.ndarray:
    """Central-difference gradient magnitude, edges padded with zeros."""
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    gx[:, :, 1:-1] = g[:, :, 2:] - g[:, :, :-2]
    gy[:, 1:-1, :] = g[:, 2:, :] - g[:, :-2, :]
    return np.sqrt(gx * gx + gy * gy)


def laplacian_variance(g: np.ndarray) -> np.ndarray:
    lap = np.zeros_like(g)
    lap[:, 1:-1, 1:-1] = (g[:, :-2, 1:-1] + g[:, 2:, 1:-1] +
                          g[:, 1:-1, :-2] + g[:, 1:-1, 2:] - 4 * g[:, 1:-1, 1:-1])
    return lap.reshape(len(g), -1).var(axis=1)


def grey_entropy(g: np.ndarray, bins: int = 32) -> np.ndarray:
    quantised = np.clip((g * bins).astype(np.int32), 0, bins - 1)
    flat = quantised.reshape(len(g), -1)
    out = np.empty(len(g), dtype=np.float32)
    for i in range(len(g)):
        counts = np.bincount(flat[i], minlength=bins).astype(np.float64)
        p = counts / counts.sum()
        p = p[p > 0]
        out[i] = float(-(p * np.log2(p)).sum())
    return out


def saturation(images: np.ndarray) -> np.ndarray:
    high = images.max(axis=3)
    low = images.min(axis=3)
    return np.where(high > 0, (high - low) / np.maximum(high, 1e-6), 0).mean(axis=(1, 2))


def composition_features(images: np.ndarray, chunk: int = 2000) -> pd.DataFrame:
    rows = []
    for start in range(0, len(images), chunk):
        batch = to_float_hwc(images[start:start + chunk])
        g = grey(batch)
        n, h, w = g.shape
        magnitude = gradient_magnitude(g)
        total = magnitude.reshape(n, -1).sum(axis=1) + 1e-8

        q_h, q_w = h // 4, w // 4
        center = magnitude[:, q_h:h - q_h, q_w:w - q_w].reshape(n, -1).sum(axis=1)

        ys = (np.arange(h) - (h - 1) / 2) / ((h - 1) / 2)
        xs = (np.arange(w) - (w - 1) / 2) / ((w - 1) / 2)
        r2 = (ys[:, None] ** 2 + xs[None, :] ** 2)
        spread = np.sqrt((magnitude * r2).reshape(n, -1).sum(axis=1) / total)

        ring = np.ones((h, w), dtype=bool)
        ring[8:h - 8, 8:w - 8] = False
        border = magnitude[:, ring].mean(axis=1)

        rows.append(pd.DataFrame({
            'center_frac': center / total,
            'spread': spread,
            'border_energy': border,
            'detail': laplacian_variance(g),
            'entropy': grey_entropy(g),
            'sat': saturation(batch),
        }))
        print(f'  {min(start + chunk, len(images))} / {len(images)}', end='\r')
    print()
    return pd.concat(rows, ignore_index=True)


def quantile_table(values: np.ndarray, unstable: np.ndarray, name: str, k: int = 5) -> None:
    edges = np.quantile(values, np.linspace(0, 1, k + 1))
    edges[-1] += 1e-9
    bucket = np.digitize(values, edges[1:-1])
    print(f'\n{name}')
    print(f'{"quantile":<10}{"n":>7}{"mean":>10}{"unstable":>11}')
    for q in range(k):
        mask = bucket == q
        if mask.any():
            print(f'Q{q + 1:<9}{int(mask.sum()):>7}{values[mask].mean():>10.4f}'
                  f'{unstable[mask].mean():>10.1%}')
    low, high = unstable[bucket == 0].mean(), unstable[bucket == k - 1].mean()
    direction = 'Q1 higher' if low > high else 'Q5 higher'
    print(f'  ratio {max(low, high) / max(min(low, high), 1e-9):.2f}x  ({direction})')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', default='hard_image_profile.csv')
    parser.add_argument('--out', default='composition.csv')
    arguments = parser.parse_args()

    profile = pd.read_csv(arguments.profile)
    test_ids = np.load(os.path.join(BENCH, 'test_ids.npy'))
    position = {int(t): i for i, t in enumerate(test_ids)}
    index = np.array([position[i] for i in profile['ID']])

    print('Test set:')
    test = composition_features(np.load(os.path.join(BENCH, 'test_images.npy')))
    test = test.iloc[index].reset_index(drop=True)

    print('Training set:')
    train = composition_features(np.load(os.path.join(BENCH, 'train_images.npy')))

    external_path = os.path.join(BENCH, 'external_images.npy')
    external = None
    if os.path.exists(external_path):
        print('External renditions:')
        external = composition_features(np.load(external_path))

    features = ['center_frac', 'spread', 'border_energy', 'detail', 'entropy', 'sat']
    unstable = profile['unstable'].to_numpy().astype(bool)
    sim = profile['sim_train'].to_numpy()
    far = sim < np.quantile(sim, 0.2)

    print('\n' + '=' * 72)
    print('DISTRIBUTION GAP  (test far-quintile and unstable vs the training set)')
    print('=' * 72)
    header = f'{"feature":<15}{"train":>10}{"test":>10}{"test Q1":>10}{"unstable":>10}'
    if external is not None:
        header += f'{"external":>10}'
    print(header)
    for name in features:
        line = (f'{name:<15}{train[name].mean():>10.4f}{test[name].mean():>10.4f}'
                f'{test[name][far].mean():>10.4f}{test[name][unstable].mean():>10.4f}')
        if external is not None:
            line += f'{external[name].mean():>10.4f}'
        print(line)

    print('\n' + '=' * 72)
    print('INSTABILITY BY COMPOSITION QUANTILE  (test set only)')
    print('=' * 72)
    for name in features:
        quantile_table(test[name].to_numpy(), unstable, name)

    # ---------------------------------------------- independence from desaturation
    print('\n' + '=' * 72)
    print('IS SUBJECT SCALE INDEPENDENT OF DESATURATION?')
    print('  Colour images only, so the greyscale effect cannot drive this.')
    print('=' * 72)
    colour = test['sat'].to_numpy() >= 0.10
    print(f'colour images: {int(colour.sum())}, unstable {unstable[colour].mean():.2%}')
    quantile_table(test['center_frac'].to_numpy()[colour], unstable[colour],
                   'center_frac (colour images only)')

    print('\n' + '=' * 72)
    print('CROSS-TAB  greyscale x low center_frac')
    print('=' * 72)
    low_center = test['center_frac'].to_numpy() < np.quantile(test['center_frac'], 0.2)
    grey_flag = ~colour
    for g_value in (False, True):
        for c_value in (False, True):
            mask = (grey_flag == g_value) & (low_center == c_value)
            if mask.any():
                print(f'greyscale={str(g_value):<5} low_center={str(c_value):<5} '
                      f'n={int(mask.sum()):>6}  unstable={unstable[mask].mean():>6.1%}')

    out = profile.copy()
    for name in features:
        out[name] = test[name].to_numpy()
    out.to_csv(arguments.out, index=False)
    print(f'\nWrote {arguments.out}')


if __name__ == '__main__':
    main()
