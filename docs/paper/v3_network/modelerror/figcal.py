import numpy as np, pandas as pd, matplotlib.pyplot as plt
plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.size": 11})
sp=pd.read_csv('/home/claude/work/data/heft_wide_gauss_testdata_13TeV.csv')
bins=[c for c in sp.columns if c.isdigit()]; F=sp[bins].to_numpy(float)
tot=F.sum(1,keepdims=True); var=np.where(F>0,F*tot/38024.,np.nan)
a4=np.load('/home/claude/ana4/means.npz'); a5=np.load('/home/claude/ana5/ana5.npz'); A=np.load('afit.npz'); band=np.load('band.npy')
mhh=255+15*(np.arange(80)+0.5); thr=np.arange(80)<20; p=A['p']; sm=p[0]*np.exp(-(mhh-255)/p[1])+p[2]
C=dict(l1='#C0392B',ft='#1B6CA8',bnn='#7B4EA3')
fig,ax=plt.subplots(1,2,figsize=(10,3.9))
m,s=a5['FT_boot'],a5['FT_boot_sd']
for f,sd,a,lab,c,ls in [(a5['L1_seeds'],a5['L1_seeds_sd'],0,'stage-1 ensemble',C['l1'],'-'),(m,s,0,'fine-tuned ensemble',C['ft'],'-'),(m,s,sm,'fine-tuned ensemble + model error',C['ft'],'--'),(a4['mb'],a4['sb'],0,'Bayesian network',C['bnn'],':')]:
    z=(F-f)/np.sqrt(var*(1+a)+sd**2); ax[0].plot(mhh,np.nanstd(z,axis=0),color=c,ls=ls,label=lab)
ax[0].axhline(1,color='k',ls=':',lw=0.8); ax[0].set_ylim(0.8,1.8); ax[0].set_ylabel(r'standard deviation of the pull $z$'); ax[0].legend(frameon=False,fontsize=9)
ax[1].plot(mhh,A['am'],'o',ms=3,color='k',label=r'per bin, unused training rows')
ax[1].fill_between(mhh,band[0],band[1],color=C['ft'],alpha=0.25,lw=0)
ax[1].plot(mhh,sm,color=C['ft'],label='smooth fit')
ax[1].plot(mhh,np.where(thr,A['a_thr'],A['a_rest']),color='gray',ls='--',label='two regions')
ax[1].set_ylabel(r'model-error coefficient $a$'); ax[1].set_ylim(0,1.1); ax[1].legend(frameon=False,fontsize=9)
for a in ax: a.set_xlabel(r'$m_{hh}$ [GeV]'); a.set_xlim(255,1455)
fig.tight_layout(); fig.savefig('fig_app_calibration.pdf')
