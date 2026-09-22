"""Screen of blends containing the remaining zero-test-exposure members (V2 base, V3T base)."""
import numpy as np, pandas as pd
CN=np.load('bench/class_names.npy',allow_pickle=True); LAB=np.load('bench/reserve_preds.npz')['lab']
BEST=pd.read_csv('submission_ens_EVAPHPtemp.csv')['Label'].values
M={'V2P':'dinov2vitlps1','EVAP':'eva02lps1','V3P':'dinov3vitlps1','V3P2':'dinov3vitlps1b',
   'HP':'dinov3vithpps1','HPbase':'dinov3vithp_base','V2base':'dinov2vitl_base','V3Tbase':'dinov3vitl_base'}
C=[['EVAP','V3Tbase'],['EVAP','V2base'],['HP','V3Tbase'],['HP','V2base'],
   ['EVAP','HP','V3Tbase'],['V3Tbase','HPbase'],['V2base','HPbase'],['EVAP','HPbase','V2base']]
def nz(p):
    p=np.asarray(p,np.float64); return p/p.sum(1,keepdims=True)
def sh(p,T):
    z=np.log(np.clip(p,1e-12,None))/T; z-=z.max(1,keepdims=True); e=np.exp(z); return e/e.sum(1,keepdims=True)
def fitT(p,t,lo=.05,hi=20.):
    for _ in range(200):
        m=.5*(lo+hi)
        if sh(p,m).max(1).mean()>t: lo=m
        else: hi=m
    return .5*(lo+hi)
print(f'{"candidate":30s} {"reserve":>8s} {"best mbr":>9s} {"exceeds":>8s} {"k":>5s}  verdict')
for tags in C:
    ps=[nz(np.load(f'bench/{M[t]}_test_prob.npy')) for t in tags]
    tgt=ps[0].max(1).mean(); temps=[1.0]+[fitT(p,tgt) for p in ps[1:]]
    tb=sum(p if T==1. else sh(p,T) for p,T in zip(ps,temps))/len(ps)
    pr=[nz(np.load(f'bench/{M[t]}_reserve_prob.npy')) for t in tags]
    rb=sum(p if T==1. else sh(p,T) for p,T in zip(pr,temps))/len(pr)
    acc=100*(rb.argmax(1)==LAB).mean()
    best=max(100*(np.load(f'bench/{M[t]}_reserve_prob.npy').argmax(1)==LAB).mean() for t in tags)
    k=int((CN[tb.argmax(1)]!=BEST).sum()); ok=acc>best and k>=60
    print(f'{" + ".join(tags):30s} {acc:7.2f}% {best:8.2f}% {"yes" if acc>best else "NO":>8s} {k:5d}  '
          f'{"PASS  needs >= %.5f"%(0.98324+0.16*k/9345) if ok else "FAIL"}')
    if ok:
        out=f'submission_ens_{"".join(tags)}temp.csv'
        pd.DataFrame({'ID':np.arange(len(tb)),'Label':CN[tb.argmax(1)]}).to_csv(out,index=False)
        print(f'    -> {out} written')
