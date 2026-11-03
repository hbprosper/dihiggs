# ------------------------------------------------------------------------------
# model_error_study.py: numbers and figure for the section on the systematic
# uncertainty (the model-error term) in docs/dihiggs_model.tex.
#
# Run from docs/model:   python model_error_study.py
# Writes fig_model_error.pdf and prints the numbers quoted in the note.
# ------------------------------------------------------------------------------
import os, numpy as np, matplotlib.pyplot as plt
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.join(HERE, '..', 'paper')
# run docs/paper/smooth_common.py as if from docs/paper (so its cache lives there)
cwd = os.getcwd(); os.chdir(PAPER)
G = {'__file__': os.path.join(PAPER, 'smooth_common.py')}
exec(open('smooth_common.py').read(), G)
os.chdir(cwd)
D, T, qt, f, v, S, eps, z = G['D'], G['T'], G['qt'], G['f'], G['v'], G['S'], G['eps'], G['z']
noise, deviance = G['noise'], G['deviance']
model = G['SmoothHEFT'].load(G['MODEL_FILE'])
EDGES, CENTRES = G['EDGES'], G['CENTRES']
I = model.basis.integrals(EDGES); A = model.coefficient_integrals(I)      # [80, 23]

# ---- 1. epsilon bin by bin on the quick-test rows (as in heft_smooth.ipynb) ----
Cq, bq, tq, qq = G['quick_test_rows']()
fq, vq = model.predict(Cq, I[bq]); nq = qq * np.maximum(fq, qq)
Sq = np.abs(Cq * A[bq]).sum(1)                       # sum_i |C_i A_ik|
Qq = np.sqrt(((Cq * A[bq])**2).sum(1))               # sqrt(sum_i (C_i A_ik)^2), the quadrature variant
def per_bin(shape2):
    # for each bin, the c >= 0 with < r^2 / (noise + v + c * shape2) > = 1
    c = np.zeros(80)
    for k in range(80):
        s = bq == k; r2 = (tq - fq)[s]**2
        g = lambda x: np.mean(r2 / (nq[s] + vq[s] + x * shape2[s])) - 1
        c[k] = 0.0 if g(0) <= 0 else brentq(g, 0, 1e6 if shape2 is nq else 1.0)
    return c
e2_lin = per_bin(Sq**2); e2_quad = per_bin(Qq**2); a_noise = per_bin(nq)
u = (CENTRES - 855) / 600; nb = np.bincount(bq, minlength=80)
def smooth(c):
    ok = c > 0
    return np.exp(np.polyval(np.polyfit(u[ok], 0.5 * np.log(c[ok]), 2, w=np.sqrt(nb[ok])), u))
eps_lin, eps_quad, sa_noise = smooth(e2_lin), smooth(e2_quad), smooth(a_noise)
print('epsilon reproduces the stored one: max rel diff %.1e' % np.max(np.abs(eps_lin / eps - 1)))
print('epsilon (%%) at 262.5, 600, 1447.5 GeV: %s' % np.round(100 * eps[[0, 23, 79]], 3))
print('bins with epsilon_k = 0 (no excess):', int(np.sum(e2_lin == 0)))

# ---- 2. calibration on the test spectra for four treatments ----
use = (T > 0) | (f > 0)
Qt = np.sqrt(((D['C'][:, None, :] * A[None])**2).sum(2))
canc = T.sum(1) / S.sum(1); q5 = np.quantile(canc, [0, .2, .4, .6, .8, 1])
fifth = [(canc >= a) & (canc <= b) for a, b in zip(q5[:-1], q5[1:])]
forms = {'none': 0 * f, 'noise-proportional': sa_noise[None]**2 * noise,
         'linear (adopted)': (eps[None] * S)**2, 'quadrature': (eps_quad[None] * Qt)**2}
Z = {}
for k_, extra in forms.items():
    zz = np.where(use, (T - f) / np.sqrt(noise + v + extra), np.nan); Z[k_] = zz
    print(f'{k_:20s} s.d. {np.nanstd(zz):.3f} ({np.nanstd(zz[:, :20]):.3f}, {np.nanstd(zz[:, 20:]):.3f})  '
          f'|z|<2 {np.nanmean(np.abs(zz) < 2):.3f}  <z^2> by fifth {np.round([np.nanmean(zz[s]**2) for s in fifth], 2)}')
print('cancellation ratio: fifth boundaries', np.round(q5, 3))

# ---- 3. residual correlation between neighbouring bins, by fifth (noise only) ----
r = np.where(use, (T - f) / np.sqrt(noise), np.nan)
def lag1(rr):
    a_, b_ = rr[:, :-1].ravel(), rr[:, 1:].ravel(); ok = np.isfinite(a_) & np.isfinite(b_)
    return np.corrcoef(a_[ok], b_[ok])[0, 1]
print('lag-1 correlation 255-555 GeV by fifth:', np.round([lag1(r[s][:, :20]) for s in fifth], 2))
print('lag-1 correlation >555 GeV by fifth:  ', np.round([lag1(r[s][:, 20:]) for s in fifth], 2))

# ---- 4. size of the three variances relative to the Poisson noise ----
rel = np.sqrt((eps[None] * S)**2 / noise)
for lab, s in [('all', np.ones(len(T), bool)), ('strongest fifth', fifth[0]), ('weakest fifth', fifth[4])]:
    print(f'model error / noise, median ({lab}): 255-555 {np.median(rel[s][:, :20]):.3f}, >555 {np.median(rel[s][:, 20:]):.3f}; '
          f'fit/noise {np.median(np.sqrt(v[s] / noise[s])):.3f}')
relsig = (eps[None] * S) / np.abs(f)
print('relative model error eps*S/|f|: median %.4f, 99th pct %.3f; at m<555 median %.4f' % (
      np.median(relsig), np.percentile(relsig, 99), np.median(relsig[:, :20])))

# ---- 5. SM example: cross section in 255-1455 GeV, with each uncertainty ----
sm = np.array([[1.0, 1.0, 0.0, 0.0, 0.0]])
C0 = G['heft_monomials'](sm)
fsm, vsm = model.predict(np.repeat(C0, 80, 0), I)
g = np.kron(C0[0], I.sum(0))                         # gradient of the sum over the 80 bins w.r.t. beta
fit_sd = np.sqrt(g @ model.H_inverse() @ g)
d_k = eps * np.abs(C0[0] * A).sum(1)
print('SM sigma(255-1455) = %.2f fb; fit sd %.3f fb; model error: independent %.3f fb, fully correlated %.3f fb' % (
      1e3 * fsm.sum(), 1e3 * fit_sd, 1e3 * np.sqrt((d_k**2).sum()), 1e3 * d_k.sum()))
print('SM: relative model error per bin at 262.5, 600 GeV: %.4f %.4f; cancellation ratio %.3f' % (
      d_k[0] / fsm[0], d_k[23] / fsm[23], fsm.sum() / np.abs(C0[0] * A).sum()))

# ---- 6. the worst test spectrum ----
w = np.argmin(canc)
print('most cancelling test spectrum: ID %d, ratio %.4f, theta %s, mean -2lnL %.1f, <z^2> none %.1f, with %.2f' % (
      D['id'][w], canc[w], np.round(D['th'][w], 2), np.nanmean(deviance[w]), np.nanmean(Z['none'][w]**2),
      np.nanmean(Z['linear (adopted)'][w]**2)))

# ---- figure ----
plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.size": 10,
                     'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True})
fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
ok = e2_lin > 0
ax[0].semilogy(CENTRES[ok], 100 * np.sqrt(e2_lin[ok]), 'o', ms=3, color='gray', label=r'per bin, $\varepsilon_k$')
ax[0].semilogy(CENTRES, 100 * eps, color='#1B4F8A', lw=1.6, label=r'smoothed, $\varepsilon(m_{hh})$')
ax[0].set_xlabel(r'$m_{hh}$ [GeV]'); ax[0].set_ylabel(r'$\varepsilon$ [\%]'); ax[0].legend(frameon=False)
ax[0].set_title(r'(a) the model-error coefficient', fontsize=10)
for k_, c_, ls in [('none', 'gray', ':'), ('linear (adopted)', '#1B4F8A', '-')]:
    sd = [np.nanstd(Z[k_][:, k]) for k in range(80)]
    ax[1].stairs(sd, EDGES, color=c_, ls=ls, lw=1.3, label={'none': 'noise + fit only', 'linear (adopted)': 'with the model-error term'}[k_])
ax[1].axhline(1, color='k', lw=0.6); ax[1].set_xlim(255, 1455); ax[1].set_ylim(0.85, 1.35)
ax[1].set_xlabel(r'$m_{hh}$ [GeV]'); ax[1].set_ylabel(r'standard deviation of the pull'); ax[1].legend(frameon=False)
ax[1].set_title(r'(b) calibration versus $m_{hh}$', fontsize=10)
xs = np.arange(5); wdt = 0.2
for j, (k_, c_) in enumerate([('none', 'lightgray'), ('noise-proportional', '#E3A857'), ('linear (adopted)', '#1B4F8A'), ('quadrature', '#7FB3D5')]):
    ax[2].bar(xs + (j - 1.5) * wdt, [np.nanmean(Z[k_][s]**2) for s in fifth], wdt, color=c_,
              label={'none': 'none', 'noise-proportional': r'$a(m)\,\sigma^2_{\rm noise}$',
                     'linear (adopted)': r'$\varepsilon\sum_i|C_iA_i|$ (adopted)', 'quadrature': r'$\varepsilon\,[\sum_i(C_iA_i)^2]^{1/2}$'}[k_])
ax[2].axhline(1, color='k', lw=0.6); ax[2].set_ylim(0.8, 1.75)
ax[2].set_xticks(xs); ax[2].set_xticklabels(['1\nstrongest', '2', '3', '4', '5\nweakest'])
ax[2].set_xlabel('cancellation fifth'); ax[2].set_ylabel(r'$\langle z^2\rangle$'); ax[2].legend(frameon=False, fontsize=8)
ax[2].set_title(r'(c) calibration versus cancellation', fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(HERE, 'fig_model_error.pdf'))
print('wrote fig_model_error.pdf')
print('quadrature epsilon (%%) at 262.5, 600, 1447.5 GeV: %s' % np.round(100 * eps_quad[[0, 23, 79]], 3))
print('noise-proportional a(m) at 262.5, 600, 1447.5 GeV: %s' % np.round(sa_noise[[0, 23, 79]]**2, 3))
print('rows per bin in quick test: min %d median %d' % (nb.min(), np.median(nb)))
tot = np.sqrt(noise + v + (eps[None]*S)**2) / np.sqrt(noise)
print('total width / noise: median %.3f; strongest fifth %.3f' % (np.median(tot), np.median(tot[fifth[0]])))
