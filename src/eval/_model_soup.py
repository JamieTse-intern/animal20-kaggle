"""Model soup: average the WEIGHTS of same-init fine-tunes, then run one inference pass.

Precondition satisfied: B/C/E/seed4242 were all fine-tuned from the same ImageNet-pretrained
init (mean relative weight distance B-C = 0.017, i.e. still one basin). Averaging parameters
produces a single model whose internal features are the average -- a different object from
averaging 20-d probability outputs, which is all every previous blend did.

Evaluated on the 772 labelled reserve images FIRST (same discipline that killed label
propagation for 5 GPU-minutes instead of a submission), then on test only if it survives.
"""
import sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, timm
sys.path.insert(0,'.')
from _heldout_split import load_eval_set

MEAN=(0.485,0.456,0.406); STD=(0.229,0.224,0.225); dev='cuda'
CK={'B':'checkpoints_backup/timm_convnextv2_large.fcmae_ft_in22k_in1k_pseudo1_20260912_055608.pt',
    'C':'checkpoints_backup/timm_convnextv2_large.fcmae_ft_in22k_in1k_pseudo1_20260912_224350.pt',
    'E':'checkpoints_backup/timm_convnextv2_large.fcmae_ft_in22k_in1k_pseudo1_20260913_102422.pt',
    'S':'seed4242/timm_convnextv2_large.fcmae_ft_in22k_in1k_pseudo1.pt'}

class NM(nn.Module):
    def __init__(s,bb,m,st):
        super().__init__(); mt,stt=torch.tensor(m),torch.tensor(st)
        s.register_buffer('scale',(torch.tensor(STD)/stt).view(1,3,1,1))
        s.register_buffer('shift',((torch.tensor(MEAN)-mt)/stt).view(1,3,1,1)); s.backbone=bb
    def forward(s,x): return s.backbone(x*s.scale+s.shift)

def soup(names):
    sds=[torch.load(CK[n],map_location='cpu')['model_state_dict'] for n in names]
    out={}
    for k in sds[0]:
        if sds[0][k].dtype.is_floating_point:
            out[k]=torch.stack([sd[k].float() for sd in sds]).mean(0).to(sds[0][k].dtype)
        else:
            out[k]=sds[0][k]
    bb=timm.create_model('convnextv2_large.fcmae_ft_in22k_in1k',pretrained=False,num_classes=20)
    m=NM(bb,bb.default_cfg['mean'],bb.default_cfg['std']); m.load_state_dict(out)
    return m.eval().to(dev)

mt=torch.tensor(MEAN).view(1,3,1,1).to(dev); st=torch.tensor(STD).view(1,3,1,1).to(dev)
@torch.inference_mode()
def predict(m,imgs,scales,chunk=24):
    out=[]
    for i in range(0,len(imgs),chunk):
        raw=torch.from_numpy(np.ascontiguousarray(imgs[i:i+chunk])).to(dev).float()/255.
        n=(raw-mt)/st; acc=None
        for s in scales:
            r=F.interpolate(n,size=(s,s),mode='bicubic',align_corners=False)
            for v in (r,torch.flip(r,dims=[3])):
                with torch.amp.autocast('cuda',enabled=True): p=torch.softmax(m(v).float(),1)
                acc=p if acc is None else acc+p
        out.append((acc/6).cpu())
    return torch.cat(out).numpy()

img,lab=load_eval_set()
d=np.load('bench/reserve_preds.npz'); ens=d['ens']/d['ens'].sum(1,keepdims=True)
print(f'RESERVE baseline: A .8990 | C .8523 | AC-logit-1:2 blend {np.mean(ens.argmax(1)==lab):.4f}\n')
print(f'{"soup":10s} {"scales":14s} {"reserve acc":>12s}')
res={}
for names in ['BE','BCE','BES','BCES','BC']:
    m=soup(list(names))
    sc=(160,192,224)
    p=predict(m,img,sc); a=np.mean(p.argmax(1)==lab); res[names]=(a,sc)
    print(f'{"+".join(names):10s} {str(sc):14s} {a:12.4f}')
    del m; torch.cuda.empty_cache()
np.save('bench/soup_reserve_scores.npy', np.array([res[k][0] for k in res]))
print('\n(B=trio_bal030 .97257, C=res256 .97203, E=thr060 .97054, S=seed4242)')
