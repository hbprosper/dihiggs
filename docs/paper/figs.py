import numpy as np, pandas as pd, matplotlib.pyplot as plt, matplotlib.ticker as mt
plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.size": 11})
sp=pd.read_csv('/home/claude/work/data/heft_wide_gauss_testdata_13TeV.csv')
bins=[c for c in sp.columns if c.isdigit()]; F=sp[bins].to_numpy(float)
tot=F.sum(1,keepdims=True); var=np.where(F>0,F*tot/38024.,np.nan); sig=np.sqrt(var)
a4=np.load('/home/claude/ana4/means.npz'); a5=np.load('/home/claude/ana5/ana5.npz')
mhh=255+15*(np.arange(80)+0.5)
C=dict(l1='#C0392B',l1e='#E67E22',ft='#1B6CA8',bnn='#7B4EA3')
# fig A: accuracy
fig,ax=plt.subplots(figsize=(6.2,4))
for f,lab,c,ls in [(a4['L1'],'single network, L1','#999999','-'),(a5['L1_seeds'],'L1 ensemble (mean)',C['l1'],'-'),(a5['FT_boot'],'fine-tuned ensemble (mean)',C['ft'],'-'),(a4['mb'],'BNN (posterior mean)',C['bnn'],'--')]:
    ax.plot(mhh,np.nanmean((f-F)**2/var,axis=0),color=c,ls=ls,label=lab)
ax.axhline(1,color='k',ls=':',lw=0.8); ax.set_yscale('log'); ax.set_xlabel(r'$m_{hh}$ [GeV]'); ax.set_ylabel(r'$\langle \chi^2 \rangle$ per bin'); ax.legend(frameon=False,fontsize=9); ax.set_xlim(255,1455)
fig.tight_layout(); fig.savefig('fig_app_accuracy.pdf')
# fig B: calibration
rng=np.random.default_rng(0); idx=rng.permutation(len(F)); A,B=idx[:500],idx[500:]
m=a5['FT_boot']; s=a5['FT_boot_sd']; r2=(F-m)**2
e=np.nanmean((r2[A]-var[A]-s[A]**2)/var[A],axis=0)
aA=np.where(np.arange(80)<20,max(np.median(e[:20]),0),max(np.median(e[20:]),0))
fig,ax=plt.subplots(figsize=(6.2,4))
for f,sd,lab,c,ls,rows in [(a5['L1_seeds'],a5['L1_seeds_sd'],'L1 ensemble',C['l1'],'-',slice(None)),(m,s,'fine-tuned ensemble',C['ft'],'-',slice(None)),(m,np.sqrt(s**2+aA*var),r'fine-tuned ensemble + model error (test half)',C['ft'],'--',B),(a4['mb'],a4['sb'],'BNN',C['bnn'],':',slice(None))]:
    z=(F[rows]-f[rows])/np.sqrt(var[rows]+sd[rows]**2); ax.plot(mhh,np.nanstd(z,axis=0),color=c,ls=ls,label=lab)
ax.axhline(1,color='k',ls=':',lw=0.8); ax.set_ylim(0.8,1.8); ax.set_xlabel(r'$m_{hh}$ [GeV]'); ax.set_ylabel(r'standard deviation of the pull $z$'); ax.legend(frameon=False,fontsize=9); ax.set_xlim(255,1455)
fig.tight_layout(); fig.savefig('fig_app_calibration.pdf')
# fig C: decomposition
fig,ax=plt.subplots(1,2,figsize=(9.6,3.7),sharey=True)
for a,(lab,b,se) in zip(ax,[('L1','L1_boot','L1_seeds'),('fine-tuned','FT_boot','FT_seeds')]):
    vt=np.nanmedian(a5[b+'_sd']**2/var,axis=0); vo=np.nanmedian(a5[se+'_sd']**2/var,axis=0)
    vd=np.where(vt>vo,vt-vo,np.nan)
    for v,l,c in [(vt,'total (bootstrap)','k'),(vo,'optimisation (seeds)',C['ft']),(vd,'data (difference)',C['l1'])]:
        a.plot(mhh,np.sqrt(v),color=c,label=l)
    a.set_yscale('log'); a.set_xlabel(r'$m_{hh}$ [GeV]'); a.set_title(f'{lab} ensembles',fontsize=11); a.set_xlim(255,1455)
ax[0].set_ylabel(r'$[\mathrm{median}\,(s^2/\sigma_k^2)]^{1/2}$'); ax[0].legend(frameon=False,fontsize=9); ax[0].set_ylim(3e-3,1)
fig.tight_layout(); fig.savefig('fig_app_decomposition.pdf')
print('aA thr',aA[0],'rest',aA[25])
