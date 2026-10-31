import numpy as np, pandas as pd
from scipy.optimize import brentq, least_squares
Q=np.load('qt.npz'); t=Q['t']; ok=t>0
k=np.rint((Q['mhh']*100-0.5)).astype(int); var=t*Q['tot']/Q['neff']
f,s=Q['FT_boot'],Q['FT_boot_sd']; r2=(t-f)**2
ids=Q['ID']; uid,inv=np.unique(ids,return_inverse=True)
def a_hat(sel,w,kind):
    def g(a):
        z2=r2[sel]/(var[sel]*(1+a)+s[sel]**2); ww=w[sel]
        return (np.sum(ww*z2)/ww.sum()-1) if kind=='moment' else (np.sum(ww*(z2<4))/ww.sum()-0.9545)
    if (g(0)<=0)==(kind=='moment'): return 0.0
    return brentq(g,0,50)
rng=np.random.default_rng(1); one=np.ones_like(t)
regions={'thr':ok&(k<20),'rest':ok&(k>=20)}
res={}
for kind in ['moment','cov95']:
    for r,sel in regions.items():
        a=a_hat(sel,one,kind)
        bs=[a_hat(sel,np.bincount(rng.integers(0,len(uid),len(uid)),minlength=len(uid))[inv].astype(float),kind) for _ in range(200)]
        res[(kind,r)]=a; print(f'{kind:6s} {r:4s} a = {a:.3f} ± {np.std(bs):.3f}')
# smooth model a(m) = a1 + a0 exp(-x/lam), x = m - 255 GeV, fitted by the moment condition in 8 m-slices? use per-bin moment estimates
am=np.load('abin.npy')[1]; mhh=255+15*(np.arange(80)+0.5); n=np.bincount(k[ok],minlength=80)
fun=lambda p: (p[0]*np.exp(-(mhh-255)/p[1])+p[2]-am)*np.sqrt(n)/np.sqrt(2)  # sd of a_bin ~ sqrt(2/n)*(1+a)
p=least_squares(fun,[0.6,100,0.08],bounds=([0,5,0],[10,2000,1])).x
print('smooth: a0=%.3f lambda=%.0f GeV a1=%.3f'%tuple(p))
np.savez('afit.npz',a_thr=res[('moment','thr')],a_rest=res[('moment','rest')],c_thr=res[('cov95','thr')],c_rest=res[('cov95','rest')],p=p,am=am,ab=np.load('abin.npy')[0])
