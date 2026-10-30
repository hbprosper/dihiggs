import numpy as np, pandas as pd, matplotlib.pyplot as plt
from dihiggs import hmc
plt.rcParams.update({"text.usetex": True, "font.family": "serif",
  "text.latex.preamble": r"\usepackage{mathpazo}", "font.size": 11})
C1,C2,C3='#1B6CA8','#C0392B','#2E933C'
t=hmc.read_csv('BNN_laplace/dihiggs_hmc_trace.csv'); it=t['iteration']; nw=500
# Fig 1: traces
fig,ax=plt.subplots(2,2,figsize=(9.6,5.6)); ax=ax.ravel()
for a,k,lab,sc in zip(ax,['U','scale','m_0','m_1'],[r'$U$',r'noise scale $b$ ($10^{-3}$ pb)',r'SM, 262.5 GeV ($10^{-5}$ pb)',r'SM, 997.5 GeV ($10^{-5}$ pb)'],[1e-5,1e3,1e5,1e5]):
    a.plot(it,t[k]*sc,lw=0.5,color=C1); a.axvline(nw,color='gray',ls='--',lw=1)
    a.set_xlabel('iteration'); a.set_ylabel(lab if k!='U' else r'$U$ ($10^5$)')
ax[2].set_ylim(3.0,4.7); ax[3].set_ylim(1.82,2.0)
fig.tight_layout(); fig.savefig('fig_traces.pdf')
# Fig 2: autocorrelation
fig,a=plt.subplots(figsize=(5.2,3.4)); s=t['phase']=='sample'
for k,lab,c in [('U',r'$U$',C3),('scale',r'$b$','0.5'),('m_0','SM, 262.5 GeV',C1),('m_1','SM, 997.5 GeV',C2)]:
    r=hmc.autocorrelation(t[k][s]); a.plot(r[:300],color=c,label=lab)
a.axhline(0,color='k',lw=0.5); a.set_xlabel('lag (iterations)'); a.set_ylabel('autocorrelation'); a.legend(frameon=False,fontsize=9)
fig.tight_layout(); fig.savefig('fig_acf.pdf')
# holdout figures
sp=pd.read_csv('/home/claude/work/data/heft_wide_gauss_testdata_13TeV.csv')
bins=[c for c in sp.columns if c.isdigit()]; F=sp[bins].to_numpy(float)
mean=np.load('P_mean.npy'); L1=np.load('L1.npy'); sd=np.load('sd.npy')
tot=F.sum(1,keepdims=True); var=np.where(F>0,F*tot/38024.,np.nan)
mhh=255+15*(np.arange(80)+0.5)
fig,ax=plt.subplots(1,2,figsize=(9.6,3.6))
for Y,lab,c in [(L1,'L1Loss (warm start)',C2),(mean,'BNN mean',C1)]:
    ax[0].plot(mhh,np.nanmean((Y-F)**2/var,axis=0),color=c,label=lab)
ax[0].axhline(1,color='gray',ls=':'); ax[0].set_yscale('log'); ax[0].set_xlabel(r'$m_{hh}$ (GeV)'); ax[0].set_ylabel(r'mean $\chi^2$ per bin'); ax[0].legend(frameon=False)
ax[0].set_title('(a) holdout spectra',fontsize=11)
r=(sd/np.sqrt(var)).ravel(); r=r[np.isfinite(r)&(r>0)]
ax[1].hist(np.log10(r),bins=80,color=C1,alpha=0.7)
ax[1].axvline(np.log10(0.36),color=C2,ls='--'); ax[1].text(np.log10(0.36)+0.05,ax[1].get_ylim()[1]*0.85 if False else 1,'',fontsize=9)
ax[1].set_xlabel(r'$\log_{10}$(posterior s.d. / POWHEG noise)'); ax[1].set_ylabel('bins')
ax[1].set_title('(b) width of the band',fontsize=11)
fig.tight_layout(); fig.savefig('fig_holdout.pdf')
print('median ratio',np.median(r))
