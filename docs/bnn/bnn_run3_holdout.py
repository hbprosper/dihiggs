import numpy as np, pandas as pd, torch
from dihiggs import hmc, ensemble, heftnet as NN
torch.set_num_threads(2)
sp=pd.read_csv('/home/claude/work/data/heft_wide_gauss_testdata_13TeV.csv')
params=['klambda','CT','CTT','CGGH','CGGHH']; bins=[c for c in sp.columns if c.isdigit()]
F=sp[bins].to_numpy(float); nS,nB=F.shape
th=sp[params].to_numpy(np.float32); xb=((np.arange(nB)+0.5)/100).astype(np.float32)
X=torch.from_numpy(np.column_stack([np.tile(xb,nS), np.repeat(th,nB,axis=0)]).astype(np.float32))
tot=F.sum(1,keepdims=True); var=np.where(F>0,F*tot/38024.,np.nan)
def stats(name,mean,sd=None):
    c=(mean-F)**2/var; cs=np.nanmean(c,1)
    out=f'{name:28s} med {np.median(cs):.3f} >2 {np.mean(cs>2):.3f} >5 {np.mean(cs>5):.3f} thr {np.nanmean(c[:,:20]):.2f} rest {np.nanmean(c[:,20:]):.3f} |res| {np.mean(np.abs(mean-F)):.3e} neg {np.mean(mean<0):.2e}'
    if sd is not None:
        z=(F-mean)/np.sqrt(var+sd**2); ok=np.isfinite(z); r=sd/np.sqrt(var)
        out+=f'\n{"":28s} band/noise med {np.nanmedian(r):.4f} (thr {np.nanmedian(r[:,:20]):.4f}, rest {np.nanmedian(r[:,20:]):.4f})  pull sd {np.std(z[ok]):.3f} thr {np.nanstd(z[:,:20]):.3f} rest {np.nanstd(z[:,20:]):.3f} |z|<2 {np.mean(np.abs(z[ok])<2):.3f}'
    print(out)
m=NN.HEFTNet(); m.load('/mnt/user-data/uploads/dihiggs/code/runs/L1Loss/dihiggs_params.pth')
with torch.no_grad(): L1=m(X).double().numpy().reshape(nS,nB)
stats('L1Loss (single)',L1)
# ensemble
ens=ensemble.Ensemble(NN.HEFTNet,'ENS_bootstrap')
PE=ens.predict(X).double().numpy().reshape(len(ens),nS,nB)
me,se=PE.mean(0),PE.std(0,ddof=1)
stats('ENS_bootstrap mean',me,se)
ms=[ (lambda c: (np.median(np.nanmean(c,1)), np.nanmean(c[:,:20])))(((PE[k]-F)**2/var)) for k in range(len(ens))]
print('ensemble members: median chi2/N median over members', np.median([a for a,b in ms]), 'thr', np.median([b for a,b in ms]), 'range thr', min(b for a,b in ms), max(b for a,b in ms))
# BNN full
b=hmc.BNN(NN.HEFTNet,'BNN_poisson_full'); PB=b.predict(X).double().numpy().reshape(len(b),nS,nB)
mb,sb=PB.mean(0),PB.std(0)
stats('BNN Poisson full',mb,sb)
# BNN 10%
b3=hmc.BNN(NN.HEFTNet,'/home/claude/ana3/BNN_poisson'); P3=b3.predict(X).double().numpy().reshape(len(b3),nS,nB)
m3,s3=P3.mean(0),P3.std(0)
stats('BNN Poisson 10% (run 3)',m3,s3)
# combined: BNN full mean with ensemble spread
stats('BNN full mean + ENS spread',mb,se)
# drift halves for full
a=PB[:500].mean(0); c=PB[500:].mean(0); d=np.abs(a-c)/np.where(sb>0,sb,np.nan); print('BNN full halves |diff|/sd median',np.nanmedian(d),'90%',np.nanquantile(d,.9))
dd=np.abs(mb-m3)/np.where(s3>0,s3,np.nan); print('|full - 10%| / sd(10%) median', np.nanmedian(dd), '90%', np.nanquantile(dd,.9))
print('ratio sd10/sdfull median', np.nanmedian(s3/np.where(sb>0,sb,np.nan)))
print('ratio ens spread / BNN full sd median', np.nanmedian(se/np.where(sb>0,sb,np.nan)))
np.savez('means.npz',L1=L1,me=me,se=se,mb=mb,sb=sb,m3=m3,s3=s3)
