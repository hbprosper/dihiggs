# Paper figure: the 40 cubic B-splines B_j(m) of the smooth model (top) and the
# basis functions phi(m) B_j(m) that the coefficients beta_ij multiply (bottom).
# Breakpoints are uniform in sqrt(m - 250 GeV); the knot vector is clamped
# (each end point repeated), so the B_j sum to 1 everywhere on [250, 1500] GeV.
import numpy as np, matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))   # repo root
try:
    from dihiggs.smooth import SplineBasis
except ImportError:            # standalone copy of smooth.py next to this script
    from smooth import SplineBasis

plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.size": 13,
                     'xtick.direction': 'in', 'ytick.direction': 'in',
                     'xtick.top': True, 'ytick.right': True})
basis = SplineBasis(n_basis=40, m_lo=250.0, m_hi=1500.0, threshold=True)
m = np.linspace(250.0, 1500.0, 5001)
m[-1] -= 1e-9                                   # stay inside the half-open last interval

# B_j(m) without the threshold factor, and phi(m) B_j(m) with it
from scipy.interpolate import BSpline
Bj = BSpline.design_matrix(m, basis.knots, basis.k).toarray()      # [M, 40]
phi = basis.phi(m)
J = Bj.shape[1]
cmap = plt.cm.turbo
norm = Normalize(vmin=1, vmax=J)
cols = cmap(np.linspace(0.04, 0.96, J))

fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.2, 6.4), sharex=True,
                             gridspec_kw=dict(hspace=0.06, left=0.1, right=0.86, top=0.98, bottom=0.09))
for j in range(J):
    a1.plot(m, Bj[:, j], color=cols[j], lw=1.3)
    a2.plot(m, phi * Bj[:, j], color=cols[j], lw=1.3)
a1.plot(m, Bj.sum(1), 'k--', lw=1.0)
a1.text(330, 1.04, r'$\sum_j B_j(m_{hh}) = 1$', fontsize=12, va='bottom')
a2.plot(m, phi, 'k--', lw=1.0, label=r'$\phi(m) = \sqrt{1 - 4m_h^2/m^2}$')

# breakpoints (knots) as ticks along the bottom of each panel
for ax in (a1, a2):
    ax.plot(basis.breaks, np.full_like(basis.breaks, -0.035), '|', color='k', ms=7, mew=0.8,
            clip_on=False)
    ax.set_xlim(250, 1500)
    ax.axvspan(1455, 1500, color='0.9', lw=0, zorder=0)
a1.set_ylim(-0.05, 1.75); a2.set_ylim(-0.05, 1.2)
a1.set_yticks([0, 0.5, 1.0]); a2.set_yticks([0, 0.5, 1.0])
a1.set_ylabel(r'$B_j(m_{hh})$')
a2.legend(frameon=False, loc='upper right', fontsize=12)
a2.set_ylabel(r'$\phi(m_{hh})\,B_j(m_{hh})$')
a2.set_xlabel(r'$m_{hh}$ [GeV]')
a1.text(0.02, 0.97, r'40 cubic B-splines, clamped knots' + '\n' + r'uniform in $\sqrt{m_{hh}-250~\mathrm{GeV}}$',
        transform=a1.transAxes, ha='left', va='top', fontsize=11.5)

# inset: the crowded region just above threshold
ins = a1.inset_axes([0.585, 0.66, 0.39, 0.30])
z = m < 330
for j in range(J):
    ins.plot(m[z], Bj[z, j], color=cols[j], lw=1.1)
ins.plot(basis.breaks[basis.breaks < 330], np.full((basis.breaks < 330).sum(), -0.06), '|',
         color='k', ms=5, mew=0.7, clip_on=False)
ins.set_xlim(250, 330); ins.set_ylim(-0.08, 1.12)
ins.set_xticks([250, 270, 290, 310, 330]); ins.set_yticks([0, 0.5, 1])
ins.tick_params(labelsize=9, direction='in', top=True, right=True)
ins.text(0.97, 0.93, r'250--330 GeV', transform=ins.transAxes, ha='right', va='top', fontsize=9.5)

cax = fig.add_axes([0.885, 0.09, 0.022, 0.89])
cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
cb.set_label(r'basis function $j$'); cb.set_ticks([1, 10, 20, 30, 40])
fig.savefig('fig_basis.pdf'); fig.savefig('fig_basis.png', dpi=200)
print('breakpoints', len(basis.breaks), 'first few', np.round(basis.breaks[:5], 1),
      'max |sum-1|', np.abs(Bj.sum(1) - 1).max())
