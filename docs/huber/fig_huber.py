import numpy as np, matplotlib.pyplot as plt
plt.rcParams.update({"text.usetex": True, "font.family": "serif",
  "text.latex.preamble": r"\usepackage{mathpazo}", "font.size": 11})
r=np.linspace(-5,5,401)
def hub(r,d): a=np.abs(r); return np.where(a<=d,0.5*r**2,d*(a-0.5*d))
fig,ax=plt.subplots(1,2,figsize=(9.6,3.6))
ax[0].plot(r,0.5*r**2,'k',label=r'$\frac12 r^2$ ($\chi^2$, $\delta=\infty$)')
for d,c in [(1,'#C0392B'),(2,'#1B6CA8'),(3,'#2E933C')]:
    ax[0].plot(r,hub(r,d),color=c,label=fr'Huber, $\delta={d}$')
ax[0].set_ylim(0,8); ax[0].set_xlabel(r'pull $r = (f - t)/\sigma_k$'); ax[0].set_ylabel(r'loss $\ell_\delta(r)$'); ax[0].legend(frameon=False,fontsize=9)
ax[0].set_title('(a) the loss',fontsize=11)
ax[1].plot(r,r,'k',label=r'$\chi^2$')
for d,c in [(1,'#C0392B'),(2,'#1B6CA8'),(3,'#2E933C')]:
    ax[1].plot(r,np.clip(r,-d,d),color=c,label=fr'$\delta={d}$')
ax[1].set_xlabel(r'pull $r$'); ax[1].set_ylabel(r"influence $\ell_\delta'(r)$"); ax[1].legend(frameon=False,fontsize=9); ax[1].set_title('(b) its derivative: the pull of one bin on the fit',fontsize=11)
ax[1].set_ylim(-5,5)
fig.tight_layout(); fig.savefig('fig_huber.pdf')
