# Paper figure: the smooth model against POWHEG for nine of the 1,000 test spectra: the three with
# the largest Poisson deviance per bin, three around the median and the three smallest.
import numpy as np, matplotlib.pyplot as plt
exec(open('smooth_common.py').read())
plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.size": 11})
mg, dens, dvar, dS, epsg, th = D['mg'], D['dens'], D['dvar'], D['dS'], D['epsg'], D['th']
nS = len(T); edges = 255 + 15 * np.arange(81); mhh = edges[:-1] + 7.5
order = np.argsort(devN)[::-1]; mid = order[nS // 2 - 1: nS // 2 + 2]
rows = [('largest', order[:3]), ('median', mid), ('smallest', order[-3:][::-1])]
names = [r'$\kappa_\lambda$', r'$c_t$', r'$c_{tt}$', r'$c_{ggh}$', r'$c_{gghh}$']
DARK, LIGHT, LINE = '#5B8FC7', '#C9DDF0', '#1B4F8A'
fig = plt.figure(figsize=(7.2, 9.0))
outer = fig.add_gridspec(3, 3, hspace=0.13, wspace=0.27, left=0.115, right=0.995, top=0.995, bottom=0.05)
for i, (label, idx) in enumerate(rows):
    for j, s in enumerate(idx):
        g = outer[i, j].subgridspec(2, 1, height_ratios=[3, 1.15], hspace=0.0)
        ax, rx = fig.add_subplot(g[0]), fig.add_subplot(g[1])
        y = 1e3 * dens[s]; sf = 1e3 * np.sqrt(dvar[s]); st = 1e3 * np.sqrt(dvar[s] + (epsg * dS[s])**2)
        ax.fill_between(mg, y - st, y + st, color=LIGHT, lw=0, label=r'total ($\pm1\sigma$)')
        ax.fill_between(mg, y - sf, y + sf, color=DARK, lw=0, label=r'fit ($\pm1\sigma$)')
        ax.plot(mg, y, color=LINE, lw=0.9, label=r'model $d\sigma/dm_{hh}$')
        ok = T[s] > 0
        ax.errorbar(mhh[ok], 1e3 * T[s][ok] / 15, yerr=1e3 * np.sqrt(T[s][ok] * qt[s]) / 15, fmt='o', ms=1.6, lw=0.6, color='k', label='POWHEG / 15 GeV')
        ax.set_yscale('log'); ax.set_xlim(250, 1455)
        ax.tick_params(labelbottom=False, labelsize=9, direction='in', which='both', top=True, right=True)
        lo = np.nanmin(np.where(ok, 1e3 * T[s] / 15, np.nan)); hi = 1e3 * T[s].max() / 15
        ax.set_ylim(max(lo / 3, 1e-6 * hi), hi * 12); ax.set_xticks([400, 800, 1200])
        stot, ftot = T[s].sum(), f[s].sum(); unit, k = ('fb', 1e3) if stot < 1 else ('pb', 1.0)
        th1 = ', '.join(f'{nm}$={v:.2f}$' for nm, v in zip(names[:3], th[s, :3]))
        th2 = ', '.join(f'{nm}$={v:.2f}$' for nm, v in zip(names[3:], th[s, 3:]))
        info = th1 + '\n' + th2 + '\n' + rf'$\sigma_\mathrm{{tot}} = {stot*k:.3g}$ {unit} (model: {ftot*k:.3g})' + '\n' + rf'deviance$/N = {devN[s]:.2f}$'
        ax.text(0.975, 0.965, info, transform=ax.transAxes, ha='right', va='top', fontsize=7.4, linespacing=1.2)
        # ratio of POWHEG to the model's bin integrals, with the bin-level bands
        r1, r2 = np.sqrt(v[s]) / f[s], np.sqrt(v[s] + disc2[s]) / f[s]
        step = lambda a_: np.r_[a_, a_[-1]]
        rx.fill_between(edges, step(1 - r2), step(1 + r2), step='post', color=LIGHT, lw=0)
        rx.fill_between(edges, step(1 - r1), step(1 + r1), step='post', color=DARK, lw=0)
        rx.axhline(1, color=LINE, lw=0.6)
        rx.errorbar(mhh[ok], T[s][ok] / f[s][ok], yerr=np.sqrt(T[s][ok] * qt[s]) / f[s][ok], fmt='o', ms=1.4, lw=0.5, color='k')
        rx.set_xlim(250, 1455); rx.set_ylim(0.5, 1.5); rx.set_yticks([0.6, 1.0, 1.4]); rx.set_xticks([400, 800, 1200])
        rx.tick_params(labelsize=9, direction='in', which='both', top=True, right=True)
        if j == 0:
            ax.set_ylabel(r'$d\sigma/dm_{hh}$ [fb/GeV]', fontsize=10); rx.set_ylabel('ratio', fontsize=10)
            ax.text(-0.43, 0.38, label + ' deviance', transform=ax.transAxes, rotation=90, va='center', ha='center', fontsize=11)
        if i == 2: rx.set_xlabel(r'$m_{hh}$ [GeV]', fontsize=10)
        if i == 0 and j == 0: ax.legend(fontsize=7.0, frameon=False, loc='lower left', handlelength=1.3, borderaxespad=0.3)
fig.savefig('fig_spectra_panel.pdf')
print('rows', [(l, list(map(int, x)), np.round(devN[x], 2)) for l, x in rows])
