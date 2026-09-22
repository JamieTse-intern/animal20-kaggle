"""Inference-side upsample kernel: bicubic (never varied since v1) vs alternatives.

The source images were DOWN-sampled to 64px with PIL LANCZOS; inference UP-samples 64->192
with bicubic. Different operations, so not strictly a mismatch, but the up-kernel is a pure
inference knob that has never been tested. Measured on the 559 clean reserve images, where
single-model-vs-single-model is the one comparison the reserve metric gets right (3/3).

Note torch has no lanczos in F.interpolate; 'area' and 'bilinear' are the available
alternatives, plus antialias=True which changes the kernel's behaviour materially.
"""
import sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, timm
sys.path.insert(0,'.')
from _heldout_split import load_eval_set
MEAN=(0.485,0.456,0.406); STD=(0.229,0.224,0.225); dev='cuda'
CK='checkpoints_backup/timm_convnextv2_large.fcmae_ft_in22k_in1k_pseudo1_20260912_055608.pt'  # B, 192px
class NM(nn.Module):
    def __init__(s,bb,m,st):
        super().__init__(); mt,stt=torch.tensor(m),torch.tensor(st)
        s.register_buffer('scale',(torch.tensor(STD)/stt).view(1,3,1,1))
        s.register_buffer('shift',((torch.tensor(MEAN)-mt)/stt).view(1,3,1,1)); s.backbone=bb
    def forward(s,x): return s.backbone(x*s.scale+s.shift)
bb=timm.create_model('convnextv2_large.fcmae_ft_in22k_in1k',pretrained=False,num_classes=20)
m=NM(bb,bb.default_cfg['mean'],bb.default_cfg['std'])
m.load_state_dict(torch.load(CK,map_location='cpu')['model_state_dict']); m=m.eval().to(dev)
mt=torch.tensor(MEAN).view(1,3,1,1).to(dev); st=torch.tensor(STD).view(1,3,1,1).to(dev)

@torch.inference_mode()
def predict(imgs, mode, aa, scales=(160,192,224), chunk=24):
    out=[]
    for i in range(0,len(imgs),chunk):
        raw=torch.from_numpy(np.ascontiguousarray(imgs[i:i+chunk])).to(dev).float()/255.
        n=(raw-mt)/st; acc=None
        for s in scales:
            kw=dict(size=(s,s),mode=mode)
            if mode in ('bicubic','bilinear'): kw.update(align_corners=False, antialias=aa)
            r=F.interpolate(n,**kw)
            for v in (r,torch.flip(r,dims=[3])):
                with torch.amp.autocast('cuda',enabled=True): p=torch.softmax(m(v).float(),1)
                acc=p if acc is None else acc+p
        out.append((acc/6).cpu())
    return torch.cat(out).numpy()

img,lab=load_eval_set(); clean=np.load('bench/reserve_clean_mask.npy')
print(f'model B, 559 clean reserve images. SE at p=.91,n=559 is 1.21pt -- read accordingly.\n')
print(f'{"kernel":22s} {"clean acc":>10s} {"delta":>8s}  rows vs bicubic')
base=None
for mode,aa,name in [('bicubic',False,'bicubic (current)'),('bicubic',True,'bicubic+antialias'),
                     ('bilinear',False,'bilinear'),('bilinear',True,'bilinear+antialias'),
                     ('area',False,'area'),('nearest',False,'nearest')]:
    try: p=predict(img,mode,aa)
    except Exception as e: print(f'{name:22s} FAILED: {str(e)[:40]}'); continue
    a=np.mean(p.argmax(1)[clean]==lab[clean])
    if base is None: base=a; bp=p
    print(f'{name:22s} {a:10.4f} {a-base:+8.4f}  {int((p.argmax(1)!=bp.argmax(1)).sum()):5d}')
