import numpy as np, pandas as pd, torch, matplotlib.pyplot as plt
from dihiggs import hmc, heftnet as NN
plt.rcParams.update({"text.usetex": True, "font.family": "serif",
  "text.latex.preamble": r"\usepackage{mathpazo}", "font.size": 11})
C1,C2,C3,C4='#1B6CA8','#C0392B','#2E933C','#7B4EA3'
t=hmc.read_csv('BNN_poisson/dihiggs_hmc_trace.csv'); it=t['iteration']
fig,ax=plt.subplots(3,2,figsize=(9.6,7.6)); ax=ax.ravel()
spec=[('U',r'$U$ ($10^4$)',1e-4,None),('eps',r'step size $\varepsilon$',1,'log'),
      ('m_0',r'SM, 262.5 GeV ($10^{-5}$ pb)',1e5,None),('m_1',r'SM, 997.5 GeV ($10^{-5}$ pb)',1e5,None),
      ('tau_7',r'$\tau_7$ (\texttt{P.6.weight})',1,None),('tau_9',r'$\tau_9$ (\texttt{P.8.weight})',1,None)]
for a,(k,lab,sc,ys) in zip(ax,spec):
    a.plot(it,t[k]*sc,lw=0.5,color=C1)
    a.axvline(1275,color=C3,ls=':',lw=1); a.axvline(1500,color='gray',ls='--',lw=1)
    a.set_ylabel(lab); a.set_xlabel('iteration')
    if ys: a.set_yscale(ys)
ax[0].set_ylim(1.66,1.76); ax[2].set_ylim(5.0,7.0)
fig.tight_layout(); fig.savefig('fig2_traces.pdf')
# holdout
sp=pd.read_csv('/home/claude/work/data/heft_wide_gauss_testdata_13TeV.csv')
bins=[c for c in sp.columns if c.isdigit()]; F=sp[bins].to_numpy(float)
tot=F.sum(1,keepdims=True); var=np.where(F>0,F*tot/38024.,np.nan)
L1=np.load('L1.npy'); MP=np.load('P_mean.npy'); SP=np.load('sd.npy'); ML=np.load('/home/claude/ana/P_mean.npy')
mhh=255+15*(np.arange(80)+0.5)
fig,ax=plt.subplots(1,2,figsize=(9.6,3.7))
for Y,lab,c in [(L1,'L1Loss (start)',C2),(ML,'BNN, Laplace (run 1)',C4),(MP,'BNN, Poisson (run 2)',C1)]:
    ax[0].plot(mhh,np.nanmean((Y-F)**2/var,axis=0),color=c,label=lab)
ax[0].axhline(1,color='gray',ls=':'); ax[0].set_yscale('log'); ax[0].set_xlabel(r'$m_{hh}$ (GeV)'); ax[0].set_ylabel(r'mean $\chi^2$ per bin'); ax[0].legend(frameon=False,fontsize=9)
ax[0].set_title('(a) accuracy',fontsize=11)
zl=(F-L1)/np.sqrt(var); zp=(F-MP)/np.sqrt(var+SP**2); zn=(F-MP)/np.sqrt(var)
for z,lab,c,ls in [(zl,'L1Loss, noise only',C2,'-'),(zn,'Poisson BNN mean, noise only',C1,'--'),(zp,'Poisson BNN, noise + band',C1,'-')]:
    ax[1].plot(mhh,np.nanstd(z,axis=0),color=c,ls=ls,label=lab)
ax[1].axhline(1,color='gray',ls=':'); ax[1].set_yscale('log'); import matplotlib.ticker as mt; ax[1].yaxis.set_major_formatter(mt.ScalarFormatter()); ax[1].yaxis.set_minor_formatter(mt.NullFormatter()); ax[1].set_yticks([1,2,3,5]); ax[1].set_xlabel(r'$m_{hh}$ (GeV)'); ax[1].set_ylabel(r'standard deviation of the pull $z$'); ax[1].legend(frameon=False,fontsize=9)
ax[1].set_title('(b) calibration',fontsize=11)
fig.tight_layout(); fig.savefig('fig2_holdout.pdf')
# SM spectrum near threshold
xs=np.linspace(0.0025,0.3,200).astype(np.float32)
X=torch.from_numpy(np.column_stack([xs,np.tile(np.array([1,1,0,0,0],np.float32),(len(xs),1))]).astype(np.float32))
m=NN.HEFTNet(); m.load('/mnt/user-data/uploads/dihiggs/code/runs/L1Loss/dihiggs_params.pth')
with torch.no_grad(): yl=m(X).numpy()
fig,a=plt.subplots(figsize=(6,3.8)); mm=255+1500*xs
a.plot(mm,yl*1e3,color=C2,label='L1Loss (start)')
for d,lab,c in [('/home/claude/ana/BNN_laplace','BNN, Laplace',C4),('BNN_poisson','BNN, Poisson',C1)]:
    b=hmc.BNN(NN.HEFTNet,d,max_samples=500); q=b.quantiles(X,[0.025,0.5,0.975]).numpy()
    a.fill_between(mm,q[0]*1e3,q[2]*1e3,color=c,alpha=0.35,lw=0); a.plot(mm,q[1]*1e3,color=c,label=lab+r' (median, 95\%)')
a.set_xlabel(r'$m_{hh}$ (GeV)'); a.set_ylabel(r'$\sigma$ per 15 GeV ($10^{-3}$ pb)'); a.legend(frameon=False,fontsize=9,loc='lower right'); a.set_xlim(255,420); a.set_yscale('log'); a.set_ylim(1e-2,3)
fig.tight_layout(); fig.savefig('fig2_sm.pdf')
print('pull sd per region', np.nanstd(zp[:,:20]), np.nanstd(zp[:,20:]))
