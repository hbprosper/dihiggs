import numpy as np, pandas as pd
from scipy.optimize import brentq
Q=np.load('qt.npz'); t=Q['t']; ok=t>0
k=np.rint((Q['mhh']*100-0.5)).astype(int)          # bin index 0..79
var=t*Q['tot']/Q['neff']
def z(f,s,a,sel): return (t[sel]-f[sel])/np.sqrt(var[sel]*(1+a)+s[sel]**2)
def est(f,s,sel,kind):
    g={'moment':lambda a: np.mean(z(f,s,a,sel)**2)-1,
       'mad':lambda a: 1.4826*np.median(np.abs(z(f,s,a,sel)))-1,
       'cov68':lambda a: np.mean(np.abs(z(f,s,a,sel))<1)-0.6827,
       'cov95':lambda a: np.mean(np.abs(z(f,s,a,sel))<2)-0.9545}[kind]
    if (g(0)<=0) if kind in('moment','mad') else (g(0)>=0): return 0.0
    return brentq(g,0,50)
R={'thr':ok&(k<20),'rest':ok&(k>=20)}
ids=Q['ID']; uid=np.unique(ids); rng=np.random.default_rng(1)
print('rows', {r:int(R[r].sum()) for r in R}, 'spectra',len(uid))
for ens in ['FT_boot','FT_seeds']:
    f,s=Q[ens],Q[ens+'_sd']
    print(ens)
    for r,sel in R.items():
        z0=z(f,s,0,sel); line=f'  {r:4s} sd(z,a=0)={z0.std():.3f} madsd={1.4826*np.median(np.abs(z0)):.3f} |z|<2={np.mean(np.abs(z0)<2):.4f}  a:'
        for kind in ['moment','mad','cov68','cov95']:
            a=est(f,s,sel,kind)
            # bootstrap over spectra
            bs=[]
            for b in range(100):
                pick=rng.choice(uid,len(uid)); w=np.isin(ids,pick)  # approx (no multiplicity)
                bs.append(est(f,s,sel&w,kind))
            line+=f' {kind} {a:.3f}±{np.std(bs):.3f}'
        print(line)
# per-bin robust a (FT_boot)
f,s=Q['FT_boot'],Q['FT_boot_sd']
ab=np.array([est(f,s,ok&(k==j),'mad') for j in range(80)])
am=np.array([est(f,s,ok&(k==j),'moment') for j in range(80)])
np.save('abin.npy',np.vstack([ab,am]))
print('per-bin mad a:',np.round(ab,2))
print('per-bin moment a:',np.round(am,2))
