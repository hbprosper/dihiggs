import numpy as np, pandas as pd, torch
from dihiggs import hmc, heftnet as NN
torch.set_num_threads(2)
sp=pd.read_csv('/home/claude/work/data/heft_wide_gauss_testdata_13TeV.csv')
params=['klambda','CT','CTT','CGGH','CGGHH']; bins=[c for c in sp.columns if c.isdigit()]
F=sp[bins].to_numpy(float); th=sp[params].to_numpy(np.float32)
nS,nB=F.shape; xb=((np.arange(nB)+0.5)/100).astype(np.float32)
X=np.column_stack([np.tile(xb,nS), np.repeat(th,nB,axis=0)]).astype(np.float32)
x=torch.from_numpy(X)
bnn=hmc.BNN(NN.HEFTNet,'BNN_poisson')
K=len(bnn); print('K',K)
P=bnn.predict(x).double().numpy().reshape(K,nS,nB)   # 1000*80000*8B = 640MB
np.save('P_mean.npy',P.mean(0)); 
m=NN.HEFTNet(); m.load('/mnt/user-data/uploads/dihiggs/code/runs/L1Loss/dihiggs_params.pth')
with torch.no_grad(): L1=m(x).double().numpy().reshape(nS,nB)
NEFF=38024.
tot=F.sum(1,keepdims=True)
var=np.where(F>0,F*tot/NEFF,np.nan)
def chi2N(Y): 
    c=(Y-F)**2/var; return np.nanmean(c,axis=1)
mean=P.mean(0); sd=P.std(0)
lo,hi=np.quantile(P,[0.025,0.975],axis=0)
for name,Y in [('L1Loss warm start',L1),('BNN mean',mean),('BNN first sample',P[0]),('BNN last sample',P[-1])]:
    c=chi2N(Y); print(f'{name:18s} median chi2/N {np.median(c):.3f}  frac>2 {np.mean(c>2):.3f}  frac>5 {np.mean(c>5):.3f}  mean|res| {np.mean(np.abs(Y-F)):.3e}')
# per-region
thr=slice(0,20)
for name,Y in [('L1',L1),('BNN',mean)]:
    c=(Y-F)**2/var
    print(name,'mean chi2/bin 255-555', np.nanmean(c[:,thr]), 'above', np.nanmean(c[:,20:]))
noise=np.sqrt(var)
r=sd/noise
print('posterior sd / POWHEG noise: median',np.nanmedian(r),' 90%',np.nanquantile(r,0.9),' 99%',np.nanquantile(r,0.99))
print('posterior sd / POWHEG noise by region: thr', np.nanmedian(r[:,thr]), 'rest', np.nanmedian(r[:,20:]))
cov=np.mean((F>=lo)&(F<=hi)); print('fraction of POWHEG bins inside 95% credible band', cov)
d=(mean-L1)/np.where(sd>0,sd,np.nan)
print('|BNN mean - L1| / posterior sd: median', np.nanmedian(np.abs(d)), '90%', np.nanquantile(np.abs(d),0.9))
rel=np.abs(mean-L1)/np.maximum(np.abs(L1),1e-12)
print('rel change BNN vs L1: median', np.median(rel), 'thr bins median', np.median(rel[:,thr]))
# drift: first vs last 500 samples
a=P[:500].mean(0); b=P[500:].mean(0)
dd=np.abs(a-b)/np.where(sd>0,sd,np.nan)
print('|mean(first half)-mean(second half)|/sd median', np.nanmedian(dd), '90%', np.nanquantile(dd,0.9))
neg=np.mean(mean<0); print('negative BNN means', neg, 'L1', np.mean(L1<0))
np.save('L1.npy',L1); np.save('sd.npy',sd)
z=(F-mean)/np.sqrt(var+sd**2)
ok=np.isfinite(z)
print('pull sd all', np.std(z[ok]), 'frac|z|<2', np.mean(np.abs(z[ok])<2))
zt=z[:,thr]; zr=z[:,20:]
print('pull sd thr', np.nanstd(zt), 'rest', np.nanstd(zr))
zl=(F-L1)/np.sqrt(var); print('L1 pull sd (noise only)', np.nanstd(zl), 'thr', np.nanstd(zl[:,thr]), 'rest', np.nanstd(zl[:,20:]))
zb=(F-mean)/np.sqrt(var); print('BNN pull sd (noise only)', np.nanstd(zb), 'thr', np.nanstd(zb[:,thr]), 'rest', np.nanstd(zb[:,20:]))
