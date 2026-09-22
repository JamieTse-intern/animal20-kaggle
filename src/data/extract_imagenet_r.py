"""Extract ImageNet-R renditions for the 20 competition classes into
imagenet_r_ext/<our_class>/*.jpg, ready for prepare_external.py.

ImageNet-R: 30k images, 200 classes, renditions (art/cartoon/graffiti/embroidery/
origami/painting/sculpture/sketch/tattoo/toy/videogame). Covers 17 of our 20
classes including the two DomainNet blind spots: hen->chickens, sea_lion->seals.
"""
import io, glob, os
from collections import Counter
import pyarrow.parquet as pq
from PIL import Image

# ImageNet-R class_name -> our class. Conservative: no cross-contamination with
# our separate classes (lions, horses handled explicitly; big cats excluded from 'cats').
MAP = {
    # birds (not 'duck' - separate class)
    'american_egret': 'birds', 'bald_eagle': 'birds', 'flamingo': 'birds',
    'goldfinch': 'birds', 'hummingbird': 'birds', 'junco': 'birds',
    'king_penguin': 'birds', 'lorikeet': 'birds', 'ostrich': 'birds',
    'peacock': 'birds', 'pelican': 'birds', 'toucan': 'birds', 'vulture': 'birds',
    # ducks / waterfowl
    'duck': 'ducks', 'goose': 'ducks', 'black_swan': 'ducks',
    # bottles
    'wine_bottle': 'bottles',
    # breads
    'bagel': 'breads', 'pretzel': 'breads',
    # butterflies
    'monarch_butterfly': 'butterflies',
    # cats (domestic only)
    'tabby_cat': 'cats',
    # chickens  <- BLIND SPOT
    'hen': 'chickens',
    # dogs
    'afghan_hound': 'dogs', 'basset_hound': 'dogs', 'beagle': 'dogs',
    'bloodhound': 'dogs', 'border_collie': 'dogs', 'boston_terrier': 'dogs',
    'boxer': 'dogs', 'chihuahua': 'dogs', 'chow_chow': 'dogs',
    'cocker_spaniels': 'dogs', 'collie': 'dogs', 'dalmatian': 'dogs',
    'french_bulldog': 'dogs', 'german_shepherd_dog': 'dogs',
    'golden_retriever': 'dogs', 'husky': 'dogs', 'italian_greyhound': 'dogs',
    'labrador_retriever': 'dogs', 'pembroke_welsh_corgi': 'dogs',
    'pomeranian': 'dogs', 'pug': 'dogs', 'rottweiler': 'dogs',
    'saint_bernard': 'dogs', 'scottish_terrier': 'dogs', 'shih_tzu': 'dogs',
    'standard_poodle': 'dogs', 'toy_poodle': 'dogs', 'weimaraner': 'dogs',
    'west_highland_white_terrier': 'dogs', 'whippet': 'dogs',
    'yorkshire_terrier': 'dogs',
    # fishes
    'clown_fish': 'fishes', 'goldfish': 'fishes', 'puffer_fish': 'fishes',
    'great_white_shark': 'fishes', 'hammerhead': 'fishes', 'stingray': 'fishes',
    'eel': 'fishes',
    # handguns
    'revolver': 'handguns',
    # horses (equine) -- 'zebra': 'horses' removed 2026-09-11: audit found
    # zebra images were 48% of the horses external pool, visually very
    # different (stripes) from any real horse photo. horses now has zero
    # ImageNet-R source (no non-zebra equine class exists there); DomainNet
    # 'horse' (521 images) remains the only external source.
    # lions
    'lion': 'lions',
    # lipsticks
    'lipstick': 'lipsticks',
    # seals  <- BLIND SPOT
    'sea_lion': 'seals',
    # snakes
    'cobra': 'snakes',
    # spiders
    'tarantula': 'spiders',
    # vases
    'vase': 'vases',
}

OUT = 'imagenet_r_ext'
os.makedirs(OUT, exist_ok=True)
counts = Counter()
saved = 0
for fn in sorted(glob.glob('imagenet_r/*.parquet')):
    t = pq.read_table(fn)
    for img, name in zip(t['image'].to_pylist(), t['class_name'].to_pylist()):
        our = MAP.get(name)
        if our is None:
            continue
        d = os.path.join(OUT, our)
        os.makedirs(d, exist_ok=True)
        try:
            im = Image.open(io.BytesIO(img['bytes'])).convert('RGB')
        except Exception:
            continue
        im.save(os.path.join(d, f'inr_{saved:06d}.jpg'), quality=95)
        counts[our] += 1
        saved += 1
print(f'saved {saved} images across {len(counts)} classes')
for c in sorted(counts):
    print(f'  {c:14s} {counts[c]}')
missing = sorted({v for v in MAP.values()} ^ set(
    ['birds','bottles','breads','butterflies','cakes','cats','chickens','cows',
     'dogs','ducks','elephants','fishes','handguns','horses','lions','lipsticks',
     'seals','snakes','spiders','vases']))
print('no ImageNet-R data for:', [c for c in missing if c not in counts])
