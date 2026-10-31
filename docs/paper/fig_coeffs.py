# The 23 coefficient functions a_i(m_hh) of HEFTNet with their uncertainty bands:
# mean over the 20 fine-tuned networks; band = within-network (Laplace) and
# between-network variance, +-1 sigma. The cross section in a bin is
# sum_i C_i(theta) a_i(m_hh), with C_i the monomial shown in each panel.
import numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.size": 9,
                     'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True})
d = np.load('coeffs_bins.npz')  # written by fig_coeffs_compute.py
x, A, Kd = d['x'], 1e3 * d['A'], 1e6 * d['Kdiag']            # in fb
# keep only the range spanned by the bin centres (262.5 to 1447.5 GeV), where the
# functions are constrained by data; outside it they are extrapolations
inside = (x >= 0.005 - 1e-9) & (x <= 0.795 + 1e-9)
x, A, Kd = x[inside], A[:, inside], Kd[:, inside]
mhh = 255 + 1500 * x
mean = A.mean(0); within = Kd.mean(0); between = A.var(0)
s_tot = np.sqrt(within + between); s_bet = np.sqrt(between)
kl, ct, ctt, cg, cgg = r'\kappa_\lambda ', 'c_t', 'c_{tt}', 'c_{ggh}', 'c_{gghh}'
C = [f'{ct}^4', f'{ctt}^2', f'{ct}^2{kl}^2', f'{cg}^2{kl}^2', f'{cgg}^2', f'{ctt}{ct}^2', f'{kl}{ct}^3',
     f'{ct}{kl}{ctt}', f'{cg}{kl}{ctt}', f'{ctt}{cgg}', f'{cg}{kl}{ct}^2', f'{cgg}{ct}^2', f'{kl}^2{cg}{ct}',
     f'{cgg}{ct}{kl}', f'{cg}{cgg}{kl}', f'{ct}^3{cg}', f'{ct}{ctt}{cg}', f'{ct}{cg}^2{kl}', f'{ct}{cg}{cgg}',
     f'{ct}^2{cg}^2', f'{ctt}{cg}^2', f'{cg}^3{kl}', f'{cg}^2{cgg}']
fig, axes = plt.subplots(6, 4, figsize=(7.2, 9.6), sharex=True)
fig.subplots_adjust(left=0.085, right=0.99, top=0.985, bottom=0.05, wspace=0.42, hspace=0.12)
for i, ax in enumerate(axes.flat[:23]):
    # small functions are shown in ab (1 fb = 1000 ab), so that no axis needs a x10^n offset
    k, unit = (1e3, 'ab') if np.abs(mean[:, i]).max() < 0.5 else (1.0, 'fb')
    E = 255 + 15 * np.arange(len(mhh) + 1)            # bin edges
    step = lambda y: np.r_[y, y[-1]]
    ax.fill_between(E, step(k * (mean[:, i] - s_tot[:, i])), step(k * (mean[:, i] + s_tot[:, i])), step='post', color='#C9DDF0', lw=0)
    ax.fill_between(E, step(k * (mean[:, i] - s_bet[:, i])), step(k * (mean[:, i] + s_bet[:, i])), step='post', color='#5B8FC7', lw=0)
    ax.stairs(k * mean[:, i], E, color='#1B4F8A', lw=0.9)
    ax.axhline(0, color='gray', lw=0.4, ls=':')
    lo = min(0.0, k * (mean[:, i] - s_tot[:, i]).min()); hi = max(0.0, k * (mean[:, i] + s_tot[:, i]).max())
    pad = 0.12 * (hi - lo); ax.set_ylim(lo - pad, hi + 0.35 * (hi - lo))
    ax.set_xlim(255, 1455); ax.set_xticks([400, 800, 1200])
    ax.ticklabel_format(axis='y', style='plain', useOffset=False)
    ax.tick_params(labelsize=7.5)
    ax.text(0.96, 0.92, f'$a_{{{i+1}}}$ [{unit}]: ${C[i]}$', transform=ax.transAxes, ha='right', va='top', fontsize=7.3,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.75, pad=1.0))
    if i % 4 == 0:
        ax.set_ylabel(r'$a_i$', fontsize=8.5)
ax = axes.flat[23]; ax.axis('off')
ax.legend(handles=[Line2D([], [], color='#1B4F8A', label='mean, 20 networks'),
                   Patch(color='#5B8FC7', label=r'between, $\pm1\sigma$'),
                   Patch(color='#C9DDF0', label=r'total, $\pm1\sigma$')],
          loc='center left', fontsize=7.5, frameon=False, borderaxespad=0.0)
for ax in axes[-1]:
    ax.set_xlabel(r'$m_{hh}$ [GeV]', fontsize=8.5)
axes[-1, -1].xaxis.set_tick_params(labelbottom=False)
fig.savefig('fig_coefficients.pdf')
r = s_tot / np.maximum(np.abs(mean), 1e-30)
print('within/between variance ratio, median over m and i: %.2f' % np.median(within / between))
for i in range(23):
    print(f'a_{i+1:2d}: median |s/mean| {np.median(r[:, i]):.3f}   max |mean| {np.abs(mean[:, i]).max():.3g} fb')
