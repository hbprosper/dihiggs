# All 1,000 test spectra against the smooth model, in the style of heft_test.ipynb (linear scale,
# sorted by increasing total cross section), 20 per page.
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
exec(open('smooth_common.py').read())
plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm', 'font.size': 8, 'axes.linewidth': 0.6,
                     'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True})
mg, dens, dvar, dS, epsg, th = D['mg'], D['dens'], D['dvar'], D['dS'], D['epsg'], D['th']
nS = len(T); edges = 255 + 15 * np.arange(81); mhh = edges[:-1] + 7.5
tot, ftot = T.sum(1), f.sum(1); order = np.argsort(tot)
cols, rows = 4, 5; per_page = cols * rows; npages = int(np.ceil(nS / per_page))
GREEN, LINE, DARK, LIGHT = '#2E933C', '#1B4F8A', '#5B8FC7', '#C9DDF0'
fmt = lambda x: f'{x*1e3:.3g} fb' if x < 1 else f'{x:.3g} pb'
with PdfPages('all_test_spectra.pdf') as pdf:
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.5, 0.88, 'The smooth model: the 1,000 test spectra', ha='center', fontsize=18)
    fig.text(0.5, 0.84, r'POWHEG against the smooth $23\times80$ model of $d\sigma/dm_{hh}$', ha='center', fontsize=12)
    txt = ('Spectra are sorted by increasing total cross section (255–1455 GeV), 20 per page.\n\n'
           'Green histogram and error bars: POWHEG divided by the 15 GeV bin width, with the Poisson noise of\n'
           'the scaled event counts.  Blue line: the model density.  Dark band: fit uncertainty ($\\pm1\\sigma$, exact).\n'
           'Light band: total ($\\pm1\\sigma$), including the discrepancy $\\varepsilon(m_{hh})\\,\\sum_i|C_i\\,a_i|$.\n\n'
           'Each panel gives, above it, the test-spectrum index (row of the test file, from 0) and the HEFT\n'
           r'parameters ($\kappa_\lambda$, $c_t$, $c_{tt}$, $c_{ggh}$, $c_{gghh}$); inside, the total cross section, POWHEG (model),'
           '\nand the likelihood-ratio statistic $-2\\ln\\lambda$ per bin, $\\lambda$ = Poisson$(n;\\mu)$/Poisson$(n;n)$ (1 for a perfect model with Poisson noise only).')
    fig.text(0.1, 0.55, txt, fontsize=10.5, va='bottom', linespacing=1.5)
    fig.text(0.1, 0.48, f'Median $-2\\ln\\lambda$ per bin = {np.median(devN):.3f};   spectra above 2: {np.sum(devN > 2)};   above 5: {np.sum(devN > 5)}.', fontsize=10.5)
    fig.text(0.1, 0.06, 'P. Bhat (Fermilab), N. De Filippis (Politecnico di Bari), K.F. Johnson, H.B. Prosper, R. Starnes (Florida State University)', fontsize=8.5)
    pdf.savefig(fig); plt.close(fig)
    for page in range(npages):
        fig, axes = plt.subplots(rows, cols, figsize=(8.5, 11))
        fig.subplots_adjust(left=0.07, right=0.985, top=0.948, bottom=0.05, wspace=0.3, hspace=0.36)
        idx = order[page * per_page:(page + 1) * per_page]
        fig.text(0.5, 0.977, f'Test spectra {page*per_page+1}–{page*per_page+len(idx)} of {nS}, by increasing total cross section   (page {page+1} of {npages})', ha='center', fontsize=9)
        for ax, s in zip(axes.flat, idx):
            y = 1e3 * dens[s]; sf = 1e3 * np.sqrt(dvar[s]); st = 1e3 * np.sqrt(dvar[s] + (epsg * dS[s])**2)
            ax.fill_between(mg, y - st, y + st, color=LIGHT, lw=0); ax.fill_between(mg, y - sf, y + sf, color=DARK, lw=0)
            Ts = 1e3 * T[s] / 15; e = 1e3 * np.sqrt(noise[s]) / 15; ok = T[s] > 0
            ax.stairs(Ts, edges, fill=True, color=GREEN, alpha=0.22, lw=0)
            ax.errorbar(mhh[ok], Ts[ok], yerr=e[ok], fmt='none', ecolor=GREEN, lw=0.6)
            ax.plot(mg, y, color=LINE, lw=0.9)
            ax.set_xlim(250, 1455); ax.set_ylim(min(0.0, (y - st).min() * 1.1), 1.45 * max(Ts.max(), (y + st).max()))
            ax.set_xticks([400, 800, 1200]); ax.ticklabel_format(axis='y', style='plain', useOffset=False); ax.tick_params(labelsize=7)
            t_ = th[s]
            ax.set_title(f'#{s}:  {t_[0]:.2f}, {t_[1]:.2f}, {t_[2]:.2f}, {t_[3]:.2f}, {t_[4]:.2f}', fontsize=7, pad=2.5)
            ax.text(0.97, 0.95, f'$\\sigma$ = {fmt(tot[s])} ({fmt(ftot[s])})\n$\\langle -2\\ln\\lambda\\rangle$ = {devN[s]:.2f}', transform=ax.transAxes,
                    ha='right', va='top', fontsize=7, linespacing=1.3, color='#B03A2E' if devN[s] > 2 else 'k')
        for ax in axes.flat[len(idx):]: ax.axis('off')
        for ax in axes[-1]: ax.set_xlabel(r'$m_{hh}$ [GeV]', fontsize=8)
        for ax in axes[:, 0]: ax.set_ylabel(r'$d\sigma/dm_{hh}$ [fb/GeV]', fontsize=8)
        if page == 0:
            h = [Patch(color=GREEN, alpha=0.4, label='POWHEG / 15 GeV'), Line2D([], [], color=LINE, label='model'),
                 Patch(color=DARK, label=r'fit $\pm1\sigma$'), Patch(color=LIGHT, label=r'total $\pm1\sigma$')]
            fig.legend(handles=h, loc='lower center', ncol=4, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.0))
        pdf.savefig(fig); plt.close(fig)
print('done', npages)
