import pandas as pd,numpy as np,matplotlib.pyplot as plt
plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.size": 11})
R=pd.read_csv('train_quanta.csv'); T=pd.read_csv('test_quanta.csv')
fig,ax=plt.subplots(1,2,figsize=(10,3.7))
a=ax[0]; bins=np.arange(14000,40501,500)
a.hist(R.N_in,bins=bins,color='#1B6CA8',alpha=0.8,label='training (3,989)'); a.hist(T.N_in,bins=bins,histtype='step',color='#C0392B',lw=1.2,label='test (1,000)')
a.set_yscale('log'); a.set_xlabel(r'events in 255--1455 GeV'); a.set_ylabel('spectra'); a.legend(frameon=False,fontsize=9,loc='upper left')
a.axvline(40000,color='k',ls=':',lw=0.8); a.axvline(38024,color='gray',ls='--',lw=0.8); a.text(37900,300,r'old $N_\mathrm{eff}$',ha='right',fontsize=8,color='gray')
M=40000-R.N_in.values; A=np.column_stack([R['first'],R['last']]); ok=R.N_in>30000
c=np.linalg.lstsq(A[ok],M[ok],rcond=None)[0]; p=A@c
a=ax[1]; a.loglog(p[ok],M[ok],'.',ms=2,color='#1B6CA8'); a.loglog(p[~ok],M[~ok],'o',ms=5,color='#C0392B',label='IDs 1814, 4284')
x=np.array([30,6000]); a.plot(x,x,'k:',lw=0.8)
a.set_xlabel(r'$%.2f\,n_\mathrm{first} + %.1f\,n_\mathrm{last}$'%tuple(c)); a.set_ylabel(r'$40{,}000 - $ events in range'); a.legend(frameon=False,fontsize=9)
fig.tight_layout(); fig.savefig('fig_counts.pdf'); print(np.corrcoef(p[ok],M[ok])[0,1])
