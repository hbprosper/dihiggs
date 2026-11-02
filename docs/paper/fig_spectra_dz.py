# Paper figure: three test spectra, chosen by how strongly their 23 terms cancel
# (weak, typical, strong; in each case the spectrum with the median likelihood-ratio
# statistic of its cancellation fifth).  For each: the spectrum, the per-bin
# likelihood-ratio statistic -2 ln lambda, lambda = P(n; mu)/P(n; n), and the pull z(m) = (n - mu)/sd.
import numpy as np, matplotlib.pyplot as plt
exec(open('smooth_common.py').read())
plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.size": 13})
mg, dens, dvar, dS, epsg, th = D['mg'], D['dens'], D['dvar'], D['dS'], D['epsg'], D['th']
nS = len(T); edges = 255 + 15 * np.arange(81); mhh = edges[:-1] + 7.5

# per-bin -2 ln lambda (the 'Poisson deviance'; event counts n, predicted counts mu) and pulls
d = deviance
z_full = (T - f) / np.sqrt(noise + v + disc2)          # Poisson noise + fit + model-error term
z_part = (T - f) / np.sqrt(noise + v)                  # without the model-error term

# degree of cancellation of each spectrum: sigma / sum_i |C_i a_i|  (1 = no cancellation)
canc = f.sum(1) / S.sum(1)
fifth = np.digitize(canc, np.quantile(canc, [0.2, 0.4, 0.6, 0.8]))    # 0 = strongest
def median_dev(g):
    idx = np.where(fifth == g)[0]
    return idx[np.argsort(devN[idx])[len(idx) // 2]]
cols = [('weak cancellation', median_dev(4)), ('typical', median_dev(2)),
        ('strong cancellation', median_dev(0))]

names = [r'$\kappa_\lambda$', r'$c_t$', r'$c_{tt}$', r'$c_{ggh}$', r'$c_{gghh}$']
DARK, LIGHT, LINE, GREY = '#5B8FC7', '#C9DDF0', '#1B4F8A', '#9A9A9A'
fig = plt.figure(figsize=(7.2, 8.2))
outer = fig.add_gridspec(1, 3, wspace=0.34, left=0.11, right=0.99, top=0.845, bottom=0.135)
for j, (label, s) in enumerate(cols):
    g = outer[0, j].subgridspec(3, 1, height_ratios=[2.6, 1.15, 1.15], hspace=0.0)
    ax, dx, zx = (fig.add_subplot(g[k]) for k in range(3))

    # spectrum with fit and total bands
    y = 1e3 * dens[s]; sf = 1e3 * np.sqrt(dvar[s]); st = 1e3 * np.sqrt(dvar[s] + (epsg * dS[s])**2)
    ax.fill_between(mg, y - st, y + st, color=LIGHT, lw=0, label=r'total ($\pm1\sigma$)')
    ax.fill_between(mg, y - sf, y + sf, color=DARK, lw=0, label=r'fit ($\pm1\sigma$)')
    ax.plot(mg, y, color=LINE, lw=0.9, label=r'model')
    ok = T[s] > 0
    ax.errorbar(mhh[ok], 1e3 * T[s][ok] / 15, yerr=1e3 * np.sqrt(T[s][ok] * qt[s]) / 15,
                fmt='o', ms=1.6, lw=0.6, color='k', label='POWHEG')
    ax.set_yscale('log')
    lo = np.nanmin(np.where(ok, 1e3 * T[s] / 15, np.nan)); hi = 1e3 * T[s].max() / 15
    ax.set_ylim(max(lo / 3, 1e-6 * hi), hi * 4)
    # the five couplings and two summaries, written above the panel
    vals = [f'{nm}$ = {v_:.2f}$' for nm, v_ in zip(names, th[s])]
    info = (vals[0] + ', ' + vals[1] + '\n' + vals[2] + ', ' + vals[3] + '\n' + vals[4] + '\n'
            + rf'$\sigma/\sum_i|C_i a_i| = {canc[s]:.2f}$,  ' + rf'$\langle -2\ln\lambda\rangle = {devN[s]:.2f}$')
    ax.text(0.5, 1.02, info, transform=ax.transAxes, ha='center', va='bottom', fontsize=10, linespacing=1.15)
    ax.set_title(label, fontsize=14, pad=58)

    # -2 ln lambda per bin; reference: 1 (its mean for a correct model), 3.84 (95% point of chi2_1)
    dx.axhline(1, color=LINE, lw=0.6); dx.axhline(3.84, color=LINE, lw=0.6, ls='--')
    dx.plot(mhh, d[s], 'o', ms=1.8, color='k')
    dx.set_ylim(0, 12); dx.set_yticks([0, 4, 8])
    big = d[s] > 12
    dx.plot(mhh[big], np.full(big.sum(), 11.6), '^', ms=2.6, color='k')   # off scale

    # pulls: with (filled) and without (open grey) the model-error term
    zx.axhspan(-2, 2, color=LIGHT, lw=0); zx.axhspan(-1, 1, color=DARK, alpha=0.35, lw=0)
    zx.axhline(0, color=LINE, lw=0.6)
    zx.plot(mhh, np.clip(z_part[s], -5.8, 5.8), 'o', ms=2.2, mfc='none', mec=GREY, mew=0.5,
            label='noise + fit')
    zx.plot(mhh, np.clip(z_full[s], -5.8, 5.8), 'o', ms=1.8, color='k', label='+ model error')
    zx.set_ylim(-5.5, 5.5); zx.set_yticks([-4, 0, 4])

    for a_ in (ax, dx, zx):
        a_.set_xlim(250, 1455); a_.set_xticks([500, 1000])
        a_.tick_params(labelsize=12, direction='in', which='both', top=True, right=True)
    ax.tick_params(labelbottom=False); dx.tick_params(labelbottom=False)
    zx.set_xlabel(r'$m_{hh}$ [GeV]', fontsize=13)
    if j == 0:
        ax.set_ylabel(r'$d\sigma/dm_{hh}$ [fb/GeV]', fontsize=13)
        dx.set_ylabel(r'$-2\ln\lambda$', fontsize=13)
        zx.set_ylabel(r'pull $z$', fontsize=13)
    if j == 0:
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = zx.get_legend_handles_labels()
fig.legend(h1 + h2, l1 + ['pull: noise + fit', 'pull: + model error'], ncol=3, fontsize=10.5,
           frameon=False, loc='lower center', bbox_to_anchor=(0.55, 0.0), columnspacing=1.2,
           handletextpad=0.4)
fig.savefig('fig_spectra_dz.pdf'); fig.savefig('fig_spectra_dz.png', dpi=200)
for label, s in cols:
    print(label, int(s), 'canc', round(canc[s], 3), '<d>', round(devN[s], 3),
          'sd z full', round(np.nanstd(z_full[s]), 2), 'sd z part', round(np.nanstd(z_part[s]), 2))
