# Publication figure: ensemble predictions with uncertainty bands against POWHEG
# for 9 of the 1,000 test spectra: the 3 worst, 3 around the median and the 3 best,
# ranked by chi2/N (ensemble mean vs POWHEG, exact Poisson noise).
import glob, numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.optimize import brentq, least_squares
from dihiggs import counts as CN
plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.size": 11})
R = '/home/claude/work/code/runs/LAPLACE_ENS_bootstrap_huber_d2_1'
P = [np.load(f) for f in sorted(glob.glob(f'{R}/member_*/predictions.npz'))]
sp = pd.read_csv('/home/claude/work/data/heft_wide_gauss_testdata_13TeV.csv')
bins = [c for c in sp.columns if c.isdigit()]
T = sp[bins].to_numpy(float); nS, nB = T.shape
_, q, _ = CN.recover_counts_wide(T)
mhh = 255 + 15 * (np.arange(nB) + 0.5); edges = 255 + 15 * np.arange(nB + 1)
mu_m = np.array([p['mu_test'] for p in P]).reshape(len(P), nS, nB)
v_m = np.array([p['v_test'] for p in P]).reshape(len(P), nS, nB)
mu = mu_m.mean(0); within = v_m.mean(0); between = mu_m.var(0)
s2_net = within + between
# model-error coefficient a(m): fitted on the quick-test rows, as in heft_laplace.ipynb
df = pd.read_csv('/home/claude/work/data/heft_wide_gauss_traindata_13TeV.csv.gz')
N = len(df); tr = int(0.864 * N); te = int(0.120 * N); va = N - tr - te
counted, _ = CN.recover_counts(df); quick = counted.iloc[tr + va:]
mq = np.mean([p['mu_quick'] for p in P], 0)
s2q = np.mean([p['v_quick'] for p in P], 0) + np.var([p['mu_quick'] for p in P], 0)
tq = quick.sigma.to_numpy(); vq = tq * quick.quantum.to_numpy()
bq = np.rint(quick.mhh.to_numpy() * 100 - 0.5).astype(int)
def a_bin(r):
    g = lambda a: np.mean((tq - mq)[r]**2 / (vq[r] * (1 + a) + s2q[r])) - 1
    return 0.0 if g(0) <= 0 else brentq(g, 0, 50)
ab = np.array([a_bin(bq == b) for b in range(nB)]); nb = np.bincount(bq, minlength=nB)
p = least_squares(lambda p: (p[0]*np.exp(-(mhh-255)/p[1]) + p[2] - ab)*np.sqrt(nb),
                  [0.6, 100, 0.08], bounds=([0, 5, 0], [10, 2000, 1])).x
a = p[0] * np.exp(-(mhh - 255) / p[1]) + p[2]
print('a(m) parameters', p)
noise = T * q[:, None]                                  # exact Poisson variance
s2_tot = s2_net + a * noise
# ranking
chi2N = np.nanmean(np.where(T > 0, (T - mu)**2 / noise, np.nan), axis=1)
order = np.argsort(chi2N)[::-1]                         # worst first
mid = order[nS // 2 - 1: nS // 2 + 2]
rows = [('worst', order[:3]), ('median', mid), ('best', order[-3:][::-1])]
z = np.where(T > 0, (T - mu) / np.sqrt(noise + s2_tot), np.nan)
names = [r'$\kappa_\lambda$', r'$c_t$', r'$c_{tt}$', r'$c_{ggh}$', r'$c_{gghh}$']
cols = ['klambda', 'CT', 'CTT', 'CGGH', 'CGGHH']
C = dict(pw='k', mean='#1B6CA8', net='#1B6CA8', tot='#9ec5e8')
fig = plt.figure(figsize=(7.2, 9.0))
outer = fig.add_gridspec(3, 3, hspace=0.13, wspace=0.27, left=0.115, right=0.995, top=0.995, bottom=0.05)
for i, (label, idx) in enumerate(rows):
    for j, s in enumerate(idx):
        g = outer[i, j].subgridspec(2, 1, height_ratios=[3, 1.15], hspace=0.0)
        ax, rx = fig.add_subplot(g[0]), fig.add_subplot(g[1])
        m = mu[s]; st = np.sqrt(s2_tot[s]); sn = np.sqrt(s2_net[s]); e = np.sqrt(noise[s])
        ok = T[s] > 0
        ax.fill_between(edges, np.r_[m - st, (m - st)[-1]], np.r_[m + st, (m + st)[-1]], step='post', color=C['tot'], lw=0, label=r'total ($\pm1\sigma$)')
        ax.fill_between(edges, np.r_[m - sn, (m - sn)[-1]], np.r_[m + sn, (m + sn)[-1]], step='post', color=C['net'], alpha=0.55, lw=0, label=r'network ($\pm1\sigma$)')
        ax.stairs(m, edges, color=C['mean'], lw=0.8, label='ensemble mean')
        ax.errorbar(mhh[ok], T[s][ok], yerr=e[ok], fmt='o', ms=1.6, lw=0.6, color=C['pw'], label='POWHEG')
        ax.set_yscale('log'); ax.set_xlim(255, 1455); ax.tick_params(labelbottom=False, labelsize=9, direction='in', which='both', top=True, right=True)
        lo = np.nanmin(np.where(ok, T[s], np.nan)); ax.set_ylim(max(lo / 3, 1e-6 * T[s].max()), T[s].max() * 12)
        th = ', '.join(f'{n}$={sp[c].iloc[s]:.2f}$' for n, c in zip(names[:3], cols[:3]))
        th2 = ', '.join(f'{n}$={sp[c].iloc[s]:.2f}$' for n, c in zip(names[3:], cols[3:]))
        stot = T[s].sum(); ftot = m.sum()
        unit, k = ('fb', 1e3) if stot < 1 else ('pb', 1.0)
        info = (th + '\n' + th2 + '\n' + rf'$\sigma_\mathrm{{tot}} = {stot*k:.3g}$ {unit} (model: {ftot*k:.3g})' + '\n' + rf'$\chi^2/N = {chi2N[s]:.2f}$')
        ax.text(0.975, 0.965, info, transform=ax.transAxes, ha='right', va='top', fontsize=7.4, linespacing=1.2)
        # ratio panel: POWHEG / ensemble mean, with bands
        safe = np.where(m > 0, m, np.nan)
        rx.fill_between(edges, np.r_[1 - st / safe, (1 - st / safe)[-1]], np.r_[1 + st / safe, (1 + st / safe)[-1]], step='post', color=C['tot'], lw=0)
        rx.fill_between(edges, np.r_[1 - sn / safe, (1 - sn / safe)[-1]], np.r_[1 + sn / safe, (1 + sn / safe)[-1]], step='post', color=C['net'], alpha=0.55, lw=0)
        rx.axhline(1, color=C['mean'], lw=0.6)
        rx.errorbar(mhh[ok], T[s][ok] / safe[ok], yerr=e[ok] / safe[ok], fmt='o', ms=1.4, lw=0.5, color='k')
        rx.set_xlim(255, 1455); rx.set_ylim(0.5, 1.5); rx.set_yticks([0.6, 1.0, 1.4]); rx.set_xticks([400, 800, 1200]); ax.set_xticks([400, 800, 1200])
        rx.tick_params(labelsize=9, direction='in', which='both', top=True, right=True)
        if j == 0:
            ax.set_ylabel(r'$\sigma$ per 15 GeV [pb]', fontsize=10); rx.set_ylabel(r'ratio', fontsize=10)
            ax.text(-0.43, 0.38, {'worst': 'largest', 'median': 'median', 'best': 'smallest'}[label] + r' $\chi^2/N$', transform=ax.transAxes, rotation=90, va='center', ha='center', fontsize=11, fontweight='bold')
        if i == 2: rx.set_xlabel(r'$m_{hh}$ [GeV]', fontsize=10)
        if i == 0 and j == 0: ax.legend(fontsize=7.4, frameon=False, loc='lower left', handlelength=1.3, borderaxespad=0.3)
fig.savefig('fig_spectra_panel.pdf'); fig.savefig('fig_spectra_panel.png', dpi=200)
print('ranks', [(l, list(map(int, idx)), [round(float(chi2N[k]), 2) for k in idx]) for l, idx in rows])
print('pull sd all', np.nanstd(z), ' median total/noise', np.nanmedian(np.sqrt(s2_tot / noise)))
tot = T.sum(1)
print('total sigma: worst10 median %.3g, all median %.3g' % (np.median(tot[order[:10]]), np.median(tot)))
print('quantile of total sigma among worst 10:', np.round([np.mean(tot < tot[k]) for k in order[:10]], 3))
print('spearman', pd.Series(chi2N).corr(pd.Series(tot), method='spearman'))
