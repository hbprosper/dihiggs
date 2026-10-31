import numpy as np, pickle, glob
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.size": 9, 'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True})
P = np.load('primary.npz'); mg, a, sd, ac = P['mg'], P['a'], P['sd'], P['ac']
net = 1e3 * np.load('/home/claude/coef/coeffs_bins.npz')['A'].mean(0)        # fb per 15 GeV bin
cen = 262.5 + 15 * np.arange(80)
kl, ct, ctt, cg, cgg = r'\kappa_\lambda ', 'c_t', 'c_{tt}', 'c_{ggh}', 'c_{gghh}'
Cn = [f'{ct}^4', f'{ctt}^2', f'{ct}^2{kl}^2', f'{cg}^2{kl}^2', f'{cgg}^2', f'{ctt}{ct}^2', f'{kl}{ct}^3', f'{ct}{kl}{ctt}', f'{cg}{kl}{ctt}', f'{ctt}{cgg}', f'{cg}{kl}{ct}^2', f'{cgg}{ct}^2', f'{kl}^2{cg}{ct}', f'{cgg}{ct}{kl}', f'{cg}{cgg}{kl}', f'{ct}^3{cg}', f'{ct}{ctt}{cg}', f'{ct}{cg}^2{kl}', f'{ct}{cg}{cgg}', f'{ct}^2{cg}^2', f'{ctt}{cg}^2', f'{cg}^3{kl}', f'{cg}^2{cgg}']
fig, axes = plt.subplots(6, 4, figsize=(7.2, 9.6), sharex=True)
fig.subplots_adjust(left=0.085, right=0.99, top=0.985, bottom=0.05, wspace=0.42, hspace=0.12)
for i, ax in enumerate(axes.flat[:23]):
    y, s = 15e3 * a[:, i], 15e3 * sd[:, i]                                     # fb per 15 GeV
    k, unit = (1e3, 'ab') if np.abs(y).max() < 0.5 else (1.0, 'fb')
    ax.fill_between(mg, k * (y - s), k * (y + s), color='#C9DDF0', lw=0)
    ax.plot(mg, k * y, color='#1B4F8A', lw=0.9)
    ax.plot(cen, k * net[:, i], '.', ms=1.8, color='#C0392B')
    ax.axhline(0, color='gray', lw=0.4, ls=':')
    lo, hi = min(0, (k * (y - s)).min()), max(0, (k * (y + s)).max()); pad = 0.12 * (hi - lo)
    ax.set_ylim(lo - pad, hi + 0.35 * (hi - lo)); ax.set_xlim(250, 1455); ax.set_xticks([400, 800, 1200])
    ax.ticklabel_format(axis='y', style='plain', useOffset=False); ax.tick_params(labelsize=7.5)
    ax.text(0.96, 0.92, f'$a_{{{i+1}}}$ [{unit}]: ${Cn[i]}$', transform=ax.transAxes, ha='right', va='top', fontsize=7.3, bbox=dict(facecolor='white', edgecolor='none', alpha=0.75, pad=1.0))
    if i % 4 == 0: ax.set_ylabel(r'$15\,\mathrm{GeV}\times a_i$', fontsize=8.5)
ax = axes.flat[23]; ax.axis('off')
ax.legend(handles=[Line2D([], [], color='#1B4F8A', label='smooth model'), Patch(color='#C9DDF0', label=r'$\pm1\sigma$ (exact)'),
                   Line2D([], [], color='#C0392B', marker='.', ls='', label='network, per bin')], loc='center left', fontsize=7.5, frameon=False)
for ax in axes[-1]: ax.set_xlabel(r'$m_{hh}$ [GeV]', fontsize=8.5)
fig.savefig('fig_smooth_coefficients.pdf'); plt.close(fig)
# calibration
d = np.load('inputs.npz'); T, qt = d['T'], d['q_t']; r = np.load('res_J40_thr.npz')
f1 = np.load('p1.npz')['f']
import pickle
m = pickle.load(open('model_primary.pkl', 'rb'))
nS = len(T); I = m.basis.integrals(255 + 15 * np.arange(81))
f, v = m.predict(d['C_t'], np.tile(I, (nS, 1))); f, v = f.reshape(nS, 80), v.reshape(nS, 80)
A = m.coefficient_integrals(I); St = np.abs(d['C_t'].reshape(nS, 80, 23) * A[None]).sum(2)
eps = P['eps']; noise = qt[:, None] * np.maximum(f, qt[:, None])
ok = (T > 0) | (f > 0)
z0 = np.where(ok, (T - f) / np.sqrt(noise + v), np.nan); z1 = np.where(ok, (T - f) / np.sqrt(noise + v + (eps[None] * St)**2), np.nan)
canc = T.sum(1) / St.sum(1); q5 = np.quantile(canc, [0, .2, .4, .6, .8, 1])
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.5))
ax[0].plot(cen, np.nanstd(z0, 0), color='gray', label='noise + fit uncertainty'); ax[0].plot(cen, np.nanstd(z1, 0), color='#1B6CA8', label=r'+ discrepancy $\varepsilon(m)\sum_i|C_ia_i|$')
ax[0].axhline(1, color='k', ls=':', lw=0.8); ax[0].set_ylim(0.8, 1.6); ax[0].set_xlim(255, 1455); ax[0].set_xlabel(r'$m_{hh}$ [GeV]'); ax[0].set_ylabel('s.d. of pull'); ax[0].legend(frameon=False, fontsize=8)
x = np.arange(1, 6)
for z, lab, c in [(z0, 'noise + fit uncertainty', 'gray'), (z1, '+ discrepancy', '#1B6CA8')]:
    ax[1].plot(x, [np.nanmean(z[(canc >= lo) & (canc <= hi)]**2) for lo, hi in zip(q5[:-1], q5[1:])], 'o-', color=c, ms=4, label=lab)
ax[1].axhline(1, color='k', lw=0.6); ax[1].set_xticks(x); ax[1].set_xlabel('cancellation fifth (1 = strongest)'); ax[1].set_ylabel(r'$\langle z^2\rangle$'); ax[1].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig('fig_smooth_calibration.pdf'); plt.close(fig)
print('primary: pull sd none %.3f, with eps %.3f (thr %.3f, rest %.3f), |z|<2 %.3f; quintiles' % (np.nanstd(z0), np.nanstd(z1), np.nanstd(z1[:, :20]), np.nanstd(z1[:, 20:]), np.nanmean(np.abs(z1) < 2)),
      np.round([np.nanmean(z1[(canc >= lo) & (canc <= hi)]**2) for lo, hi in zip(q5[:-1], q5[1:])], 2))
print('fit uncertainty / noise median %.3f; total / noise median %.3f' % (np.nanmedian(np.sqrt(v / noise)), np.nanmedian(np.sqrt((noise + v + (eps[None] * St)**2) / noise))))
np.savez('primary_test.npz', f=f, v=v, St=St, z1=z1)
# merged-bin figure
M = np.load('merged.npz'); D_obs = T[:, 0::2] - T[:, 1::2]
def sc(f): return np.nanmean((D_obs - (f[:, 0::2] - f[:, 1::2]))**2 / (qt[:, None] * (np.maximum(f[:, 0::2], qt[:, None]) + np.maximum(f[:, 1::2], qt[:, None]))), 0)
pc = 270 + 30 * np.arange(40)
fig, ax = plt.subplots(figsize=(5.2, 3.4))
ax.semilogy(pc, sc(M['f_h']), 's-', ms=3, color='#C0392B', label='free fit on 30 GeV bins, halved')
ax.semilogy(pc, sc(M['f_s']), 'o-', ms=3, color='#1B6CA8', label='smooth model fitted on 30 GeV bins')
ax.semilogy(pc, sc(f1), '^-', ms=3, color='gray', label='smooth model fitted on 15 GeV bins')
ax.axhline(1, color='k', ls=':', lw=0.8); ax.set_xlim(255, 1455); ax.set_xlabel(r'$m_{hh}$ of the 30 GeV pair [GeV]'); ax.set_ylabel(r'$\chi^2$ of the split within the pair'); ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig('fig_smooth_merged.pdf'); plt.close(fig)
# roughness
def rough(x): return np.median(np.sqrt(np.mean(np.diff(x, 2, axis=0)**2, 0)) / np.sqrt(np.mean(x**2, 0)))
fr = np.load('/home/claude/newerr/free.npz')['A']
print('roughness of the bin values: smooth %.3f, network %.3f, free fit %.3f' % (rough(ac), rough(net), rough(fr)))
# a16-a20 determination: median relative sd at bin centres
sdc = np.array([np.interp(cen, mg, sd[:, i]) for i in range(23)]).T; ac_c = ac
print('median |sd/value| at bin centres, a16-a20:', np.round(np.median(sdc[:, 15:20] / np.maximum(np.abs(ac_c[:, 15:20]), 1e-30), 0), 3))
