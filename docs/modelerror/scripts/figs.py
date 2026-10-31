import numpy as np, matplotlib.pyplot as plt
plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.size": 10, 'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True})
d = np.load('diag.npz'); T, qt, f_or, mu, dev, z = d['T'], d['qt'], d['f_or'], d['mu'], d['dev'], d['z']
mhh = 255 + 15 * (np.arange(80) + 0.5); E = 255 + 15 * np.arange(81)
n = T / qt[:, None]
def devf(m):
    mm = m / qt[:, None]; pos = mm > 0
    return np.where(pos, 2 * (mm - n + np.where(n > 0, n * np.log(np.where(n > 0, n, 1) / np.where(pos, mm, 1)), 0)), np.nan)
dev_en = devf(mu)
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
ax[0].stairs(np.nanmean(dev, 0), E, color='#C0392B', lw=1.3, label='free fit per bin (no network)')
ax[0].stairs(np.nanmean(dev_en, 0), E, color='#1B6CA8', lw=1.1, ls='--', label='fine-tuned ensemble')
ax[0].axhline(1, color='k', ls=':', lw=0.8); ax[0].set_ylim(0.8, 2.4); ax[0].set_xlim(255, 1455)
ax[0].set_xlabel(r'$m_{hh}$ [GeV]'); ax[0].set_ylabel('mean Poisson deviance per bin'); ax[0].legend(frameon=False, fontsize=9)
lags = np.arange(1, 11)
for lo, hi, lab, c in [(0, 20, r'255--555 GeV', '#C0392B'), (20, 79, r'$>555$ GeV', '#1B6CA8')]:
    r = []
    for L in lags:
        x1, x2 = z[:, lo:hi - L].ravel(), z[:, lo + L:hi].ravel(); g = np.isfinite(x1) & np.isfinite(x2)
        r.append(np.corrcoef(x1[g], x2[g])[0, 1])
    ax[1].plot(lags, r, 'o-', color=c, ms=4, label=lab)
ax[1].axhline(0, color='k', lw=0.6); ax[1].set_xlabel('separation (bins of 15 GeV)'); ax[1].set_ylabel('correlation of residuals'); ax[1].legend(frameon=False, fontsize=9)
ax[1].text(0.97, 0.95, 'Poisson noise: 0', transform=ax[1].transAxes, ha='right', va='top', fontsize=9)
fig.tight_layout(); fig.savefig('fig_floor.pdf')
canc, d_spec = d['canc'], d['d_spec']
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
ax[0].loglog(canc, d_spec, '.', ms=2.5, color='#1B6CA8', alpha=0.6)
qs = np.quantile(canc, np.linspace(0, 1, 11)); mids = np.sqrt(qs[1:] * qs[:-1])
means = [np.nanmean(d_spec[(canc >= a) & (canc <= b)]) for a, b in zip(qs[:-1], qs[1:])]
ax[0].plot(mids, means, 's-', color='#C0392B', ms=4, label='mean per decile')
ax[0].axhline(1, color='k', ls=':', lw=0.8)
ax[0].set_xlabel(r'$\sigma_\mathrm{tot} / \sum_k\sum_i |C_i a_i|$ \ (1 = no cancellation)'); ax[0].set_ylabel('deviance per bin, per spectrum'); ax[0].legend(frameon=False, fontsize=9)
s = int(np.nanargmax(d_spec))
ax[1].stairs(1e3 * T[s], E, color='#2E933C', fill=True, alpha=0.25, label='POWHEG')
ax[1].errorbar(mhh, 1e3 * T[s], yerr=1e3 * np.sqrt(T[s] * qt[s]), fmt='none', ecolor='#2E933C', lw=0.7)
ax[1].stairs(1e3 * f_or[s], E, color='#C0392B', lw=1.2, label='free fit per bin')
ax[1].stairs(1e3 * mu[s], E, color='#1B6CA8', lw=1.0, ls='--', label='ensemble')
ax[1].set_xlim(255, 1455); ax[1].set_xlabel(r'$m_{hh}$ [GeV]'); ax[1].set_ylabel(r'$\sigma$ per 15 GeV [fb]')
ax[1].set_title(f'test spectrum {s}: $\\sigma_{{\\rm tot}}$ = {1e3*T[s].sum():.1f} fb, cancellation ratio {canc[s]:.3f}', fontsize=9)
ax[1].legend(frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig('fig_cancel.pdf')
print('worst spectrum', s, 'canc', canc[s], 'canc quantile', np.mean(canc < canc[s]))
print('decile means', np.round(means, 3))
