import numpy as np
from scipy.optimize import brentq, least_squares
Q=np.load('qt.npz'); t=Q['t']; ok=t>0
k=np.rint((Q['mhh']*100-0.5)).astype(int); var=t*Q['tot']/Q['neff']
f,s=Q['FT_boot'],Q['FT_boot_sd']; r2=(t-f)**2
uid,inv=np.unique(Q['ID'],return_inverse=True); mhh=255+15*(np.arange(80)+0.5)
bins=[np.where(ok&(k==j))[0] for j in range(80)]
def perbin(w):
    out=np.zeros(80)
    for j,ix in enumerate(bins):
        ww=w[ix]
        g=lambda a: np.sum(ww*r2[ix]/(var[ix]*(1+a)+s[ix]**2))/ww.sum()-1
        out[j]=0 if g(0)<=0 else brentq(g,0,50)
    return out
def fit(am,n):
    fun=lambda p:(p[0]*np.exp(-(mhh-255)/p[1])+p[2]-am)*np.sqrt(n)
    return least_squares(fun,[0.6,100,0.08],bounds=([0,5,0],[10,2000,1])).x
n=np.array([len(b) for b in bins]); rng=np.random.default_rng(2); P=[]
for _ in range(100):
    w=np.bincount(rng.integers(0,len(uid),len(uid)),minlength=len(uid))[inv].astype(float)
    P.append(fit(perbin(w),n))
P=np.array(P); print('a0 %.2f±%.2f  lam %.0f±%.0f  a1 %.3f±%.3f'%(P[:,0].mean(),P[:,0].std(),P[:,1].mean(),P[:,1].std(),P[:,2].mean(),P[:,2].std()))
cur=P[:,[0]]*np.exp(-(mhh-255)/P[:,[1]])+P[:,[2]]; np.save('band.npy',np.percentile(cur,[16,84],axis=0))
