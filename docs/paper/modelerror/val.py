import numpy as np, pandas as pd
sp=pd.read_csv('/home/claude/work/data/heft_wide_gauss_testdata_13TeV.csv')
bins=[c for c in sp.columns if c.isdigit()]; F=sp[bins].to_numpy(float)
tot=F.sum(1,keepdims=True); var=np.where(F>0,F*tot/38024.,np.nan)
a5=np.load('/home/claude/ana5/ana5.npz'); A=np.load('afit.npz')
mhh=255+15*(np.arange(80)+0.5); thr=np.arange(80)<20
p=A['p']; smooth=p[0]*np.exp(-(mhh-255)/p[1])+p[2]
models={'none':np.zeros(80),'two-region (variance)':np.where(thr,A['a_thr'],A['a_rest']),
        'two-region (95% cover)':np.where(thr,A['c_thr'],A['c_rest']),'smooth':smooth}
out={}
for ens in ['FT_boot','FT_seeds']:
  m,s=a5[ens],a5[ens+'_sd']
  print(ens)
  for name,a in models.items():
    z=(F-m)/np.sqrt(var*(1+a)+s**2); zz=z[np.isfinite(z)]
    sd=np.nanstd(z,axis=0); out[(ens,name)]=sd
    print(f'  {name:24s} sd {np.std(zz):.3f} thr {np.nanstd(z[:,thr]):.3f} rest {np.nanstd(z[:,~thr]):.3f} | |z|<1 {np.mean(np.abs(zz)<1):.4f} |z|<2 {np.mean(np.abs(zz)<2):.4f} |z|>3 {np.mean(np.abs(zz)>3):.4f} | median sqrt(a)+ s/sig all {np.nanmedian(np.sqrt(a*var+s**2)/np.sqrt(var)):.3f}')
np.savez('val.npz',**{f'{e}|{n}':v for (e,n),v in out.items()},smooth=smooth)
