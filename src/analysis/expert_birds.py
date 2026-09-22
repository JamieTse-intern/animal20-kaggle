"""3-class expert head for birds / ducks / chickens  (agent_brief_bird_expert_head.md)

Preflight decisions (see PREFLIGHT section printed at runtime):
  - No clean v9/v10 convnext_large_mlp .pt survives (STEP A overwrote root; backup dirs hold
    only *_run.npz). -> feature source = timm pretrained convnext_large_mlp.clip_laion2b_augreg_ft_in1k,
    NO fine-tuning. Rationale: its CLIP-LAION2B pretraining is rendition-robust and, being
    un-fine-tuned, it carries zero distilled human labels (the manual corrections cannot leak
    into the boundary we are trying to learn).
  - Inference transform, read from FIT5215_v8.ipynb: uint8 -> /255 -> ImageNet-normalise ->
    bicubic interpolate to 192 -> re-normalise to the backbone's own CLIP mean/std. Because
    bicubic is linear and normalisation is per-channel affine, this equals
    /255 -> bicubic 192 -> normalise(CLIP mean/std), which is what we apply. Single scale 192,
    no flip (brief says penultimate layer at 192px).
  - Fusion base P = clean ensemble rebuilt from the four v9/v10 pseudo1 runs (= D3_LgP, 0.96499).
    NOT best_ens_v2/v3/test_prob (those have 47 hand-edited labels baked in).
"""
from __future__ import annotations
import glob, sys, time
from pathlib import Path
import numpy as np
import torch, torch.nn.functional as F
import timm
import pandas as pd

ROOT = Path('.')
BENCH = ROOT / 'bench'
CN = ['birds','bottles','breads','butterflies','cakes','cats','chickens','cows','dogs','ducks',
      'elephants','fishes','handguns','horses','lions','lipsticks','seals','snakes','spiders','vases']
TRI = [0, 6, 9]                      # birds, chickens, ducks  (20-way indices)
TRI_NAME = {0: 'birds', 6: 'chickens', 9: 'ducks'}
LOCAL = {0: 0, 6: 1, 9: 2}           # 20-way -> local 0/1/2
LOCAL_INV = {0: 0, 1: 6, 2: 9}
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
SIZE = 192
SEED = 20260910
FEAT_MODEL = 'convnext_large_mlp.clip_laion2b_augreg_ft_in1k'

dev = 'cuda' if torch.cuda.is_available() else 'cpu'


# ---------------------------------------------------------------- feature extraction
@torch.no_grad()
def extract(images_u8: np.ndarray, model, mean, std, label: str, bs: int = 64) -> np.ndarray:
    mean_t = torch.tensor(mean, device=dev).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=dev).view(1, 3, 1, 1)
    out, t0 = [], time.time()
    for i in range(0, len(images_u8), bs):
        x = torch.from_numpy(np.ascontiguousarray(images_u8[i:i + bs])).to(dev).float().div_(255.)
        x = F.interpolate(x, size=SIZE, mode='bicubic', align_corners=False)
        x = (x - mean_t) / std_t
        with torch.autocast('cuda', dtype=torch.float16, enabled=dev == 'cuda'):
            f = model(x)
        out.append(F.normalize(f.float(), dim=1).cpu())
        if i % (bs * 20) == 0:
            done = min(i + bs, len(images_u8))
            print(f'  {label}: {done}/{len(images_u8)}  {done / max(time.time() - t0, 1e-6):.0f}/s', flush=True)
    return torch.cat(out).numpy().astype(np.float32)


def build_features():
    need = ['expert_feat_xtrain', 'expert_feat_ytrain', 'expert_feat_xphoto', 'expert_feat_yphoto',
            'expert_feat_xrend', 'expert_feat_yrend', 'expert_feat_xtest',
            'expert_feat_x20train', 'expert_feat_y20train']
    if all((BENCH / f'{n}.npy').exists() for n in need):
        print('features already cached, skipping extraction')
        return
    tg = np.load(BENCH / 'targets.npy')
    tri_idx = np.load(BENCH / 'train_indices.npy')
    val_idx = np.load(BENCH / 'validation_indices.npy')
    train_u8 = np.load(BENCH / 'train_images.npy')
    test_u8 = np.load(BENCH / 'test_images.npy')
    ext_u8 = np.load(BENCH / 'external_images.npy')
    ext_y = np.load(BENCH / 'external_labels.npy')

    # ---- expert train set: {b,c,d} rows of train_indices + 85% of external {b,c,d} ----
    rng = np.random.default_rng(SEED)
    tr_bcd = tri_idx[np.isin(tg[tri_idx], TRI)]
    ext_bcd = np.flatnonzero(np.isin(ext_y, TRI))
    rend_hold = np.concatenate([                       # stratified 15% rendition holdout
        rng.permutation(ext_bcd[ext_y[ext_bcd] == c])[:max(1, int(round(0.15 * (ext_y[ext_bcd] == c).sum())))]
        for c in TRI])
    rend_hold = np.sort(rend_hold)
    ext_train = np.array(sorted(set(ext_bcd) - set(rend_hold)))
    photo_hold = val_idx[np.isin(tg[val_idx], TRI)]

    print(f'expert train: {len(tr_bcd)} photo + {len(ext_train)} rendition = {len(tr_bcd) + len(ext_train)}')
    print(f'photo holdout: {len(photo_hold)}   rendition holdout: {len(rend_hold)}')
    for c in TRI:
        print(f'  {TRI_NAME[c]:9s} train {(tg[tr_bcd] == c).sum() + (ext_y[ext_train] == c).sum():4d} '
              f'| photo-hold {(tg[photo_hold] == c).sum():3d} | rend-hold {(ext_y[rend_hold] == c).sum():3d}')

    model = timm.create_model(FEAT_MODEL, pretrained=True, num_classes=0).eval().to(dev)
    if dev == 'cuda':
        model = model.to(memory_format=torch.channels_last)
    cfg = model.pretrained_cfg
    mean, std = cfg['mean'], cfg['std']
    print(f'feature model {FEAT_MODEL}  mean {tuple(round(x, 3) for x in mean)}')

    Xtr_photo = extract(train_u8[tr_bcd], model, mean, std, 'expert-train-photo')
    Xtr_rend = extract(ext_u8[ext_train], model, mean, std, 'expert-train-rend')
    Xtrain = np.concatenate([Xtr_photo, Xtr_rend])
    ytrain = np.array([LOCAL[c] for c in np.concatenate([tg[tr_bcd], ext_y[ext_train]])])
    Xphoto = extract(train_u8[photo_hold], model, mean, std, 'photo-holdout')
    yphoto = np.array([LOCAL[c] for c in tg[photo_hold]])
    Xrend = extract(ext_u8[rend_hold], model, mean, std, 'rend-holdout')
    yrend = np.array([LOCAL[c] for c in ext_y[rend_hold]])
    Xtest = extract(test_u8, model, mean, std, 'test')

    # ---- 20-way probe training data: full train split + all external (same features) ----
    X20_photo = extract(train_u8[tri_idx], model, mean, std, '20way-train-photo')
    X20_ext = extract(ext_u8, model, mean, std, '20way-train-ext')
    X20 = np.concatenate([X20_photo, X20_ext])
    y20 = np.concatenate([tg[tri_idx], ext_y]).astype(np.int64)

    np.save(BENCH / 'expert_feat_xtrain.npy', Xtrain); np.save(BENCH / 'expert_feat_ytrain.npy', ytrain)
    np.save(BENCH / 'expert_feat_xphoto.npy', Xphoto); np.save(BENCH / 'expert_feat_yphoto.npy', yphoto)
    np.save(BENCH / 'expert_feat_xrend.npy', Xrend); np.save(BENCH / 'expert_feat_yrend.npy', yrend)
    np.save(BENCH / 'expert_feat_xtest.npy', Xtest)
    np.save(BENCH / 'expert_feat_x20train.npy', X20); np.save(BENCH / 'expert_feat_y20train.npy', y20)
    np.save(BENCH / 'expert_feat_rendidx.npy', rend_hold)
    del model
    torch.cuda.empty_cache()


# ---------------------------------------------------------------- linear heads
def train_head(X, y, n_classes, seeds=5, hidden=False, epochs=400, lr=3e-3):
    Xt = torch.tensor(X, dtype=torch.float32, device=dev)
    yt = torch.tensor(y, dtype=torch.long, device=dev)
    cls_w = torch.tensor(
        [len(y) / (n_classes * max(1, (y == c).sum())) for c in range(n_classes)],
        dtype=torch.float32, device=dev)
    lossf = torch.nn.CrossEntropyLoss(weight=cls_w, label_smoothing=0.05)
    nets = []
    for s in range(seeds):
        torch.manual_seed(SEED + s)
        if hidden:
            net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 512), torch.nn.GELU(),
                                      torch.nn.Dropout(0.2), torch.nn.Linear(512, n_classes)).to(dev)
        else:
            net = torch.nn.Linear(X.shape[1], n_classes).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
        for _ in range(epochs):
            opt.zero_grad()
            lossf(net(Xt), yt).backward()
            opt.step()
        net.eval()
        nets.append(net)
    return nets


def predict_mean(nets, X):
    Xt = torch.tensor(X, dtype=torch.float32, device=dev)
    with torch.no_grad():
        p = np.mean([torch.softmax(n(Xt).float(), 1).cpu().numpy() for n in nets], 0)
    return p


# ---------------------------------------------------------------- clean ensemble P
def clean_ensemble_P():
    if (BENCH / 'teacher_clean_test.npy').exists():
        print('using bench/teacher_clean_test.npy')
        return np.load(BENCH / 'teacher_clean_test.npy')
    members = sorted(glob.glob('v10_distil/*pseudo1_run.npz')) + sorted(glob.glob('v9_large/*pseudo1_run.npz'))
    assert len(members) == 4, members
    stack = []
    for p in members:
        z = np.load(p, allow_pickle=False)
        pr = z['test_probabilities'].astype(np.float64)
        order = z['test_key_order'].astype(np.int64)
        ordered = np.empty_like(pr)
        ordered[order] = pr
        stack.append(ordered)
        print(f'  {p}')
    return np.mean(stack, 0)


def main():
    print('=== PREFLIGHT ===')
    print(f'feature source: timm pretrained {FEAT_MODEL} (no fine-tuning) -- clean, rendition-robust')
    print('transform: /255 -> bicubic 192 -> CLIP-normalise  (equiv. to notebook to_device+NormalizedModel)')
    print()
    build_features()

    Xtrain = np.load(BENCH / 'expert_feat_xtrain.npy'); ytrain = np.load(BENCH / 'expert_feat_ytrain.npy')
    Xphoto = np.load(BENCH / 'expert_feat_xphoto.npy'); yphoto = np.load(BENCH / 'expert_feat_yphoto.npy')
    Xrend = np.load(BENCH / 'expert_feat_xrend.npy'); yrend = np.load(BENCH / 'expert_feat_yrend.npy')
    Xtest = np.load(BENCH / 'expert_feat_xtest.npy')
    X20 = np.load(BENCH / 'expert_feat_x20train.npy'); y20 = np.load(BENCH / 'expert_feat_y20train.npy')

    print('\n=== training heads (5 seeds each) ===')
    expert = train_head(Xtrain, ytrain, 3)
    probe20 = train_head(X20, y20, 20)

    def acc3(p3, y):  # p3 already 3-way
        return (p3.argmax(1) == y).mean()

    def acc20restrict(p20, y):
        r = p20[:, TRI]; r = r / r.sum(1, keepdims=True)
        return (r.argmax(1) == y).mean()

    ep_photo = predict_mean(expert, Xphoto)
    ep_rend = predict_mean(expert, Xrend)
    p20_photo = predict_mean(probe20, Xphoto)
    p20_rend = predict_mean(probe20, Xrend)

    # actual 0.965 ensemble on the photo holdout (from npz validation_probabilities)
    members = sorted(glob.glob('v10_distil/*pseudo1_run.npz')) + sorted(glob.glob('v9_large/*pseudo1_run.npz'))
    val_idx = np.load(BENCH / 'validation_indices.npy'); tg = np.load(BENCH / 'targets.npy')
    Pv = np.mean([np.load(m)['validation_probabilities'] for m in members], 0)
    photo_mask = np.isin(tg[val_idx], TRI)
    Pv_photo = Pv[photo_mask]
    y_photo_20 = tg[val_idx][photo_mask]
    r = Pv_photo[:, TRI]; r = r / r.sum(1, keepdims=True)
    ens_photo_acc = (np.array([LOCAL_INV[i] for i in r.argmax(1)]) == y_photo_20).mean()

    print('\n=== RESULTS ===')
    print(f'{"":22s} {"photo holdout":>16s} {"rendition holdout":>18s}   (rendition = decision gate)')
    print(f'{"3-way expert head":22s} {acc3(ep_photo, yphoto):16.1%} {acc3(ep_rend, yrend):18.1%}')
    print(f'{"20-way probe (restr.)":22s} {acc20restrict(p20_photo, yphoto):16.1%} {acc20restrict(p20_rend, yrend):18.1%}')
    print(f'{"actual 0.965 ens (restr)":22s} {ens_photo_acc:16.1%} {"n/a (train imgs)":>18s}')
    print(f'  photo holdout n={len(yphoto)}  rendition holdout n={len(yrend)}')

    margin = acc3(ep_rend, yrend) - acc20restrict(p20_rend, yrend)
    print(f'\nrendition-holdout margin (expert - 20way probe): {margin * 100:+.1f} points')

    # ---------------------------------------------------------------- fusion
    P = clean_ensemble_P()
    assert P.shape == (11681, 20)
    test_ids = np.load(BENCH / 'test_ids.npy').astype(int)
    order = np.argsort(test_ids)  # -> ID order; test_ids should already be 0..11680
    assert np.array_equal(test_ids[order], np.arange(11681))
    # Xtest is in bench/test_images.npy order == test_ids order; align both to ID order
    P_id = P.copy()  # clean_ensemble_P already returns ID order
    Xtest_id = Xtest  # test_images.npy is in test_ids order; check:
    if not np.array_equal(test_ids, np.arange(11681)):
        Xtest_id = Xtest[order]
        P_id = P  # clean_ensemble_P is ID-ordered already

    ep_test = predict_mean(expert, Xtest_id)          # (11681, 3) in ID order

    top2 = np.argsort(-P_id, axis=1)[:, :2]
    gate = np.array([set(t).issubset({0, 6, 9}) for t in top2])
    print(f'\ngate (top-1 AND top-2 both in birds/chickens/ducks): {gate.sum()} images')

    P_new = P_id.copy()
    for i in np.flatnonzero(gate):
        mass = P_id[i, TRI].sum()
        for k, c in enumerate(TRI):
            P_new[i, c] = mass * ep_test[i, k]

    base = pd.read_csv('submission_D3_LgP.csv').sort_values('ID').reset_index(drop=True)
    new_lab = np.array([CN[i] for i in P_new.argmax(1)])
    changed = np.flatnonzero(new_lab != base['Label'].values)
    print(f'\nrows changed vs submission_D3_LgP.csv: {len(changed)}')
    from collections import Counter
    tbl = Counter((base['Label'].values[i], new_lab[i]) for i in changed)
    for (a, b), n in tbl.most_common():
        print(f'  {a:9s} -> {b:9s}  {n}')

    if len(changed) > 400:
        print('\n*** >400 rows changed -- gate too loose or fusion misapplied. NOT writing CSV. ***')
        return
    if margin <= 0.005:
        print('\n*** expert head does NOT beat the 20-way probe on the rendition holdout by a clear margin.')
        print('    NULL RESULT -- recommend NOT spending a Kaggle submission on this.')
        print('    Writing the CSV anyway for the record, but do not submit it.')

    out = pd.DataFrame({'ID': np.arange(11681), 'Label': new_lab})
    assert len(out) == 11681 and set(out['Label']) <= set(CN) and not out['Label'].isna().any()
    out.to_csv('submission_expert_birds.csv', index=False)
    print('\nwrote submission_expert_birds.csv')

    # save head weights
    torch.save([n.state_dict() for n in expert], BENCH / 'expert_birds_heads.pt')
    print('saved bench/expert_birds_heads.pt')


if __name__ == '__main__':
    main()
