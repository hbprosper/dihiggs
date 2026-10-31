import numpy as np, matplotlib.pyplot as plt
plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.size": 10, 'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True})
d = np.load('newerr.npz'); f = np.load('free.npz')
canc = d['canc']; q5 = np.quantile(canc, [0, .2, .4, .6, .8, 1])
def byq(z): return [np.nanmean(z[(canc >= a) & (canc <= b)]**2) for a, b in zip(q5[:-1], q5[1:])]
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
x = np.arange(1, 6)
for z, lab, c, ls, mk in [(d['z_none'], 'no discrepancy', 'gray', ':', 'o'), (d['z_old'], r'$a(m_{hh})\,\sigma_k^2$', '#C0392B', '--', 's'), (d['z_new'], r'$\varepsilon(m_{hh})^2\,(\sum_i|C_ia_i|)^2$', '#1B6CA8', '-', 'o')]:
    ax[0].plot(x, byq(z), ls=ls, marker=mk, color=c, label=lab, ms=4)
ax[0].plot(x, byq(f['z80']), ls='-', marker='^', color='#2E933C', ms=4, label=r'$23\times80$ model, $\varepsilon^2(\sum_i|C_ia_i|)^2$')
ax[0].axhline(1, color='k', lw=0.6); ax[0].set_xticks(x); ax[0].set_xlabel('cancellation quintile (1 = strongest)')
ax[0].set_ylabel(r'$\langle z^2\rangle$ on the test spectra'); ax[0].legend(frameon=False, fontsize=8.5); ax[0].set_ylim(0.8, 1.8)
ax[1].semilogy(d['mhh'], np.where(d['eps_bin'] > 0, 100 * d['eps_bin'], np.nan), 'o', ms=3, color='#1B6CA8', label='per bin (ensemble mixture)')
ax[1].semilogy(d['mhh'], 100 * d['eps_sm'], '-', color='#1B6CA8', label='smooth fit')
ax[1].semilogy(d['mhh'], 100 * f['eps2'], '--', color='#2E933C', label=r'smooth fit, $23\times80$ model')
ax[1].set_xlabel(r'$m_{hh}$ [GeV]'); ax[1].set_ylabel(r'$\varepsilon(m_{hh})$ [\%]'); ax[1].legend(frameon=False, fontsize=8.5); ax[1].set_xlim(255, 1455)
fig.tight_layout(); fig.savefig('fig_newerr.pdf')
