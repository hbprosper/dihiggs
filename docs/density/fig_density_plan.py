# ------------------------------------------------------------------------------
# fig_density_plan.py: the figure of docs/density/density_plan.pdf.
#   (a) what the loss can see: the SM density from the smooth model, its bin
#       averages (the "staircase" a network trained on bin values learns), and
#       a density with an added wiggle that has exactly the same bin integrals;
#   (b) quadrature error in the first bin (255-270 GeV) versus the number of
#       function evaluations, trapezoid rule against Gauss-Legendre.
# Run from docs/density:  python fig_density_plan.py
# ------------------------------------------------------------------------------
import os, sys, numpy as np, matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..'))
from dihiggs.smooth import SmoothHEFT

model = SmoothHEFT.load(os.path.join(HERE, '..', '..', 'code', 'runs', 'SMOOTH_J40', 'smooth_model.npz'))
SM = [1.0, 1.0, 0.0, 0.0, 0.0]
dens = lambda m: 1e3 * model.dsigma_dm(SM, m)[0][0]          # fb/GeV

def gl(f, a, b, n):
    # n-point Gauss-Legendre on [a, b]: nodes c + h x, weights h w
    x, w = np.polynomial.legendre.leggauss(n); c, h = (a + b) / 2, (b - a) / 2
    return h * np.sum(w * f(c + h * x))
def trap(f, a, b, n):
    x = np.linspace(a, b, n + 1); return np.trapezoid(f(x), x)

plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.size": 10,
                     'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True})
fig, ax = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw={'width_ratios': [1.5, 1]})

# (a) ----------------------------------------------------------------------------
edges = 255.0 + 15.0 * np.arange(13)                           # 255-435 GeV
m = np.linspace(255, 435, 1801)
avg = np.array([gl(dens, a, a + 15, 16) / 15 for a in edges[:-1]])
k = np.minimum(((m - 255) // 15).astype(int), len(avg) - 1)
wiggle = dens(m) + 0.35 * avg[k] * np.sin(2 * np.pi * (m - edges[k]) / 15)   # zero integral over every bin
ax[0].plot(m, wiggle, color='#C0392B', lw=1.0, ls='--', label=r'density + wiggle (same bin integrals)')
ax[0].stairs(avg, edges, color='gray', lw=1.4, label="bin averages: the ``staircase''")
ax[0].plot(m, dens(m), color='#1B4F8A', lw=1.8, label=r'SM $d\sigma/dm_{hh}$ (smooth model)')
ax[0].plot(edges[:-1] + 7.5, avg, 'o', color='gray', ms=3.5, label='bin centres (where HEFTNet is trained)')
ax[0].set_xlim(255, 435); ax[0].set_ylim(0, None)
ax[0].set_xlabel(r'$m_{hh}$ [GeV]'); ax[0].set_ylabel(r'$d\sigma/dm_{hh}$ [fb/GeV]')
ax[0].legend(frameon=False, fontsize=8, loc='lower right')
ax[0].set_title(r'(a) three curves the bin integrals cannot tell apart', fontsize=10)

# (b) ----------------------------------------------------------------------------
f = dens; exact = gl(f, 255, 270, 64)
nt = np.array([1, 2, 4, 8, 15, 30, 60]); ng = np.array([1, 2, 3, 4, 6, 8, 12])
et = [abs(trap(f, 255, 270, n) / exact - 1) for n in nt]
eg = [max(abs(gl(f, 255, 270, n) / exact - 1), 1e-16) for n in ng]
ax[1].loglog(nt + 1, et, 's-', color='#E3A857', label='trapezoid')
ax[1].loglog(ng, eg, 'o-', color='#1B4F8A', label='Gauss--Legendre')
ax[1].axhline(0.03, color='gray', ls=':', lw=1); ax[1].text(70, 0.04, r'Poisson noise ($\approx$3\%)', fontsize=8, color='gray', ha='right')
ax[1].axhline(7.7e-4, color='gray', ls='--', lw=1); ax[1].text(70, 1.1e-3, r'systematic, $\varepsilon = 0.08\%$', fontsize=8, color='gray', ha='right')
ax[1].set_xlabel('function evaluations in the bin'); ax[1].set_ylabel('relative error of the bin integral')
ax[1].set_ylim(1e-8, 1); ax[1].set_xlim(0.8, 80); ax[1].legend(frameon=False, fontsize=9, loc='lower left')
ax[1].set_title(r'(b) quadrature error, first bin (255--270 GeV), SM', fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(HERE, 'fig_density_plan.pdf'))
print('trapezoid', dict(zip(nt + 1, np.round(et, 6)))); print('GL', dict(zip(ng, eg)))
