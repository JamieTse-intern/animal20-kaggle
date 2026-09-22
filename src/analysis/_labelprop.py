"""Graph label propagation, validated on labelled data BEFORE any submission.

Design constraints chosen up front, because propagation's classic failure mode is corrupting
the 97.5% of rows that are already right:
  - anchors (high confidence) are sources only, never updated
  - only low-confidence rows receive
  - the propagated distribution is BLENDED with the model's own, never replaces it
"""
import numpy as np, sys

def knn(F, K=10):
    idx=np.zeros((len(F),K),np.int32); sim=np.zeros((len(F),K),np.float32)
    for i in range(0,len(F),512):
        s=F[i:i+512]@F.T; s[np.arange(len(s)),np.arange(i,i+len(s))]=-2
        top=np.argpartition(-s,K,axis=1)[:,:K]
        o=np.argsort(-np.take_along_axis(s,top,1),axis=1)
        idx[i:i+512]=np.take_along_axis(top,o,1)
        sim[i:i+512]=np.take_along_axis(np.take_along_axis(s,top,1),o,1)
    return idx,sim

def propagate(prob, F, anchor_thr=0.90, recv_thr=0.70, alpha=0.5, K=10, iters=3, temp=0.1):
    conf=prob.max(1); anchor=conf>=anchor_thr; recv=conf<recv_thr
    idx,sim=knn(F,K)
    w=np.exp((sim-1.0)/temp); w*=anchor[idx]          # only anchors emit
    P=prob.copy()
    for _ in range(iters):
        num=(w[:,:,None]*P[idx]).sum(1); den=w.sum(1,keepdims=True)
        msg=np.where(den>1e-8, num/np.maximum(den,1e-8), prob)
        new=(1-alpha)*prob+alpha*msg
        new/=new.sum(1,keepdims=True)
        P=np.where(recv[:,None], new, prob)           # anchors and mid-conf rows untouched
    return P, recv, (w.sum(1)>1e-8)

if __name__=='__main__':
    F=np.load('bench/reserve_feats.npy'); d=np.load('bench/reserve_preds.npz')
    lab=d['lab']; ens=d['ens']; ens=ens/ens.sum(1,keepdims=True)
    base=np.mean(ens.argmax(1)==lab)
    print(f'RESERVE (772 labelled renditions) baseline acc {base:.4f}')
    print(f'{"alpha":>6s} {"recv":>5s} {"reached":>8s} {"changed":>8s} {"acc":>7s} {"delta":>7s}')
    for alpha in (0.3,0.5,0.7):
        P,recv,reached=propagate(ens,F,alpha=alpha)
        ch=P.argmax(1)!=ens.argmax(1); acc=np.mean(P.argmax(1)==lab)
        print(f'{alpha:6.1f} {int(recv.sum()):5d} {int((recv&reached).sum()):8d} {int(ch.sum()):8d} {acc:7.4f} {acc-base:+7.4f}')
        if ch.sum():
            print(f'       of the {int(ch.sum())} changed rows: {int((P.argmax(1)==lab)[ch].sum())} now RIGHT, '
                  f'{int((ens.argmax(1)==lab)[ch].sum())} were right before')
