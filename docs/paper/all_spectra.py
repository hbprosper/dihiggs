# All 1,000 test spectra, POWHEG against the fine-tuned ensemble, in the style of
# heft_test.ipynb (linear scale, sorted by increasing total cross section),
# 20 spectra per page.
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
exec(open('spectra_common.py').read())
plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm', 'font.size': 8,
                     'axes.linewidth': 0.6, 'xtick.direction': 'in', 'ytick.direction': 'in',
                     'xtick.top': True, 'ytick.right': True})
tot = T.sum(1); ftot = mu.sum(1)
order = np.argsort(tot)
cols, rows = 4, 5
per_page = cols * rows
GREEN, BLUE, BAND_N, BAND_T = '#2E933C', '#1B4F8A', '#5B8FC7', '#C9DDF0'

def band(ax, lo, hi, color, alpha=1.0):
    ax.fill_between(edges, np.r_[lo, lo[-1]], np.r_[hi, hi[-1]], step='post', color=color, alpha=alpha, lw=0)

def fmt(v):
    return f'{v*1e3:.3g} fb' if v < 1 else f'{v:.3g} pb'

with PdfPages('all_test_spectra.pdf') as pdf:
    # cover page
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.5, 0.88, 'HEFTNet: the 1,000 test spectra', ha='center', fontsize=18)
    fig.text(0.5, 0.84, 'POWHEG against the mean of the 20 fine-tuned networks', ha='center', fontsize=12)
    txt = ('Spectra are sorted by increasing total cross section (255–1455 GeV), 20 per page.\n\n'
           'Green histogram and error bars: POWHEG, with the exact Poisson noise of the scaled event counts.\n'
           'Blue line: ensemble mean.  Dark band: network uncertainty ($\\pm1\\sigma$; spread between the\n'
           'networks and the Laplace uncertainty of each).  Light band: total ($\\pm1\\sigma$), including the\n'
           f'model-error term $a(m_{{hh}})\\sigma_k^2$, $a = {p[0]:.3f}\\,e^{{-(m_{{hh}}-255)/{p[1]:.0f}\\,\\mathrm{{GeV}}}} + {p[2]:.3f}$.\n\n'
           'Each panel gives: above it, the test-spectrum index (row of the test file, from 0) and the HEFT parameters\n'
           r'($\kappa_\lambda$, $c_t$, $c_{tt}$, $c_{ggh}$, $c_{gghh}$); the total cross section, POWHEG (model); and $\chi^2/N$'
           '\nbetween POWHEG and the ensemble mean, using the Poisson noise only.')
    fig.text(0.1, 0.55, txt, fontsize=10.5, va='bottom', linespacing=1.5)
    fig.text(0.1, 0.48, f'Median $\\chi^2/N$ = {np.median(chi2N):.3f};   spectra with $\\chi^2/N > 2$: {np.sum(chi2N > 2)};   $> 5$: {np.sum(chi2N > 5)}.', fontsize=10.5)
    fig.text(0.1, 0.06, 'P. Bhat (Fermilab), N. De Filippis (Politecnico di Bari), K.F. Johnson, H.B. Prosper, R. Starnes (Florida State University)', fontsize=8.5)
    pdf.savefig(fig); plt.close(fig)

    for page in range(int(np.ceil(nS / per_page))):
        fig, axes = plt.subplots(rows, cols, figsize=(8.5, 11))
        fig.subplots_adjust(left=0.07, right=0.985, top=0.948, bottom=0.05, wspace=0.3, hspace=0.36)
        idx = order[page * per_page:(page + 1) * per_page]
        fig.text(0.5, 0.977, f'Test spectra {page*per_page+1}–{page*per_page+len(idx)} of {nS}, by increasing total cross section'
                 f'   (page {page+1} of {int(np.ceil(nS/per_page))})', ha='center', fontsize=9)
        for ax, s in zip(axes.flat, idx):
            # everything in fb, so that no axis needs a power-of-ten offset
            m = 1e3 * mu[s]; sn = 1e3 * np.sqrt(s2_net[s]); st = 1e3 * np.sqrt(s2_tot[s]); e = 1e3 * np.sqrt(noise[s]); Ts = 1e3 * T[s]
            band(ax, m - st, m + st, BAND_T)
            band(ax, m - sn, m + sn, BAND_N, 0.8)
            ax.stairs(Ts, edges, fill=True, color=GREEN, alpha=0.22, lw=0)
            ok = T[s] > 0
            ax.errorbar(mhh[ok], Ts[ok], yerr=e[ok], fmt='none', ecolor=GREEN, lw=0.6)
            ax.stairs(m, edges, color=BLUE, lw=0.9)
            ymax = 1.45 * max(Ts.max(), (m + st).max())
            ymin = min(0.0, (m - st).min() * 1.1)
            ax.set_xlim(255, 1455); ax.set_ylim(ymin, ymax)
            ax.set_xticks([400, 800, 1200])
            ax.ticklabel_format(axis='y', style='plain', useOffset=False)
            ax.tick_params(labelsize=7)
            th = sp.iloc[s]
            ax.set_title(f'#{s}:  {th.klambda:.2f}, {th.CT:.2f}, {th.CTT:.2f}, {th.CGGH:.2f}, {th.CGGHH:.2f}',
                         fontsize=7, pad=2.5)
            ax.text(0.97, 0.95, f'$\\sigma$ = {fmt(tot[s])} ({fmt(ftot[s])})\n$\\chi^2/N$ = {chi2N[s]:.2f}',
                    transform=ax.transAxes, ha='right', va='top', fontsize=7, linespacing=1.3,
                    color='#B03A2E' if chi2N[s] > 2 else 'k')
        for ax in axes.flat[len(idx):]:
            ax.axis('off')
        for ax in axes[-1]:
            ax.set_xlabel(r'$m_{hh}$ [GeV]', fontsize=8)
        for ax in axes[:, 0]:
            ax.set_ylabel(r'$\sigma$ per 15 GeV [fb]', fontsize=8)
        if page == 0:
            h = [Patch(color=GREEN, alpha=0.4, label='POWHEG'), Line2D([], [], color=BLUE, label='ensemble mean'),
                 Patch(color=BAND_N, label=r'network $\pm1\sigma$'), Patch(color=BAND_T, label=r'total $\pm1\sigma$')]
            fig.legend(handles=h, loc='lower center', ncol=4, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.0))
        pdf.savefig(fig); plt.close(fig)
print('done')
