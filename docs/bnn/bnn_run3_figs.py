import numpy as np, pandas as pd, matplotlib.pyplot as plt, matplotlib.ticker as mt
from dihiggs import hmc
plt.rcParams.update({"text.usetex": True, "font.family": "serif",
  "text.latex.preamble": r"\usepackage{mathpazo}", "font.size": 11})
C1,C2,C3,C4,C5='#1B6CA8','#C0392B','#2E933C','#7B4EA3','#E67E22'
t=hmc.read_csv('BNN_poisson_full/dihiggs_hmc_trace.csv'); it=t['iteration']
fig,ax=plt.subplots(2,2,figsize=(9.6,5.6)); ax=ax.ravel()
ax[0].plot(it,t['U']*1e-5,lw=0.5,color=C1); ax[0].set_ylim(1.638,1.643); ax[0].set_ylabel(r'$U$ ($10^5$)')
ax[1].plot(it,t['eps'],lw=0.5,color=C1); ax[1].set_yscale('log'); ax[1].set_ylabel(r'step size $\varepsilon$')
acc=np.convolve(t['accepted'],np.ones(50)/50,mode='same'); ax[2].plot(it,acc,color=C1); ax[2].set_ylim(0,1.05); ax[2].set_ylabel('acceptance (50-iteration average)')
ax[3].plot(it,t['m_1']*1e5,lw=0.5,color=C1); ax[3].set_ylim(1.6,1.75); ax[3].set_ylabel(r'SM, 997.5 GeV ($10^{-5}$ pb)')
for a in ax:
    for u in [100,150,250,450,900]: a.axvline(u,color=C3,ls=':',lw=0.8)
    a.axvline(1000,color='gray',ls='--',lw=1); a.set_xlabel('iteration')
fig.tight_layout(); fig.savefig('fig3_traces.pdf')
sp=pd.read_csv('/home/claude/work/data/heft_wide_gauss_testdata_13TeV.csv')
bins=[c for c in sp.columns if c.isdigit()]; F=sp[bins].to_numpy(float)
tot=F.sum(1,keepdims=True); var=np.where(F>0,F*tot/38024.,np.nan); sig=np.sqrt(var)
d=np.load('means.npz'); mhh=255+15*(np.arange(80)+0.5)
fig,ax=plt.subplots(1,3,figsize=(13.5,3.9))
for Y,lab,c in [(d['L1'],'L1Loss (single)',C2),(d['me'],'ensemble mean',C5),(d['m3'],r'BNN, 10\%',C4),(d['mb'],'BNN, full data',C1)]:
    ax[0].plot(mhh,np.nanmean((Y-F)**2/var,axis=0),color=c,label=lab)
ax[0].axhline(1,color='gray',ls=':'); ax[0].set_yscale('log'); ax[0].set_ylabel(r'mean $\chi^2$ per bin'); ax[0].legend(frameon=False,fontsize=9); ax[0].set_title('(a) accuracy',fontsize=11)
for s,lab,c in [(d['se'],'ensemble spread',C5),(d['s3'],r'BNN band, 10\%',C4),(d['sb'],'BNN band, full data',C1)]:
    ax[1].plot(mhh,np.nanmedian(s/sig,axis=0),color=c,label=lab)
ax[1].set_yscale('log'); ax[1].set_ylabel('width / POWHEG noise (median)'); ax[1].legend(frameon=False,fontsize=9); ax[1].set_title('(b) width',fontsize=11)
for Y,s,lab,c,ls in [(d['me'],d['se'],'ensemble',C5,'-'),(d['mb'],d['sb'],'BNN, full data',C1,'-'),(d['mb'],d['se'],'BNN mean + ensemble spread','k','--')]:
    z=(F-Y)/np.sqrt(var+s**2); ax[2].plot(mhh,np.nanstd(z,axis=0),color=c,ls=ls,label=lab)
ax[2].axhline(1,color='gray',ls=':'); ax[2].set_ylabel(r'standard deviation of the pull $z$'); ax[2].legend(frameon=False,fontsize=9); ax[2].set_title('(c) calibration',fontsize=11); ax[2].set_ylim(0.8,2.2)
for a in ax: a.set_xlabel(r'$m_{hh}$ (GeV)')
fig.tight_layout(); fig.savefig('fig3_holdout.pdf')
# pull histograms
fig,a=plt.subplots(figsize=(5.4,3.6)); u=np.linspace(-5,5,201)
for Y,s,lab,c in [(d['me'],d['se'],'ensemble',C5),(d['mb'],d['sb'],'BNN, full data',C1)]:
    z=((F-Y)/np.sqrt(var+s**2)).ravel(); z=z[np.isfinite(z)]
    a.hist(z,bins=np.linspace(-5,5,81),density=True,histtype='step',color=c,lw=1.3,label=lab)
a.plot(u,np.exp(-u**2/2)/np.sqrt(2*np.pi),'k:',label=r'$\mathcal{N}(0,1)$'); a.set_yscale('log'); a.set_ylim(1e-4,0.6); a.set_xlabel(r'pull $z$'); a.legend(frameon=False,fontsize=9)
fig.tight_layout(); fig.savefig('fig3_pulls.pdf')
