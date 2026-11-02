# Mean likelihood-ratio statistic -2 ln lambda per bin (the Poisson deviance) on the test spectra: smooth model, free per-bin fit, network ensemble
import numpy as np, glob, matplotlib.pyplot as plt
exec(open('smooth_common.py').read())
plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.size": 10, 'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True})
d = np.load('/home/claude/sm/inputs.npz'); nS = len(T)
C, b, t, q = d['C_fit'], d['b_fit'], d['t_fit'], d['q_fit']; Ct3 = d['C_t'].reshape(nS, 80, 23)
A = np.zeros((80, 23))
for k in range(80):                                   # free per-bin fit, Poisson weights (iterated)
    s = b == k; Ck, tk, qk = C[s], t[s], q[s]; var = qk * (tk + qk)
    for it in range(12):
        w = 1 / var; a = np.linalg.solve(Ck.T @ (w[:, None] * Ck), Ck.T @ (w * tk)); fk = Ck @ a
        var = np.sqrt(var * np.where(fk > 0, qk * np.maximum(fk, qk), qk * (tk + qk)))
    A[k] = a
f_free = np.einsum('ski,ki->sk', Ct3, A)
f_net = np.mean([np.load(x)['mu_test'] for x in glob.glob('/home/claude/work/code/runs/LAPLACE_ENS_bootstrap_huber_d2_1/member_*/predictions.npz')], 0).reshape(nS, 80)
def dev(fp):
    mu_ = fp / qt[:, None]; p_ = mu_ > 0
    return np.where(p_, 2 * (mu_ - n + np.where(n > 0, n * np.log(np.where(n > 0, n, 1) / np.where(p_, mu_, 1)), 0)), np.nan)
E = 255 + 15 * np.arange(81)
fig, ax = plt.subplots(figsize=(6.2, 3.8))
for fp, lab, c, ls in [(f, r'smooth $23\times80$ model', '#1B4F8A', '-'), (f_free, 'free fit per bin', '#C0392B', '--'), (f_net, 'network ensemble (cross-check)', 'gray', ':')]:
    ax.stairs(np.nanmean(dev(fp), 0), E, color=c, ls=ls, lw=1.2, label=lab)
ax.axhline(1, color='k', lw=0.6); ax.set_xlim(255, 1455); ax.set_ylim(0.8, 2.4)
ax.set_xlabel(r'$m_{hh}$ [GeV]'); ax.set_ylabel(r'mean $-2\ln\lambda$ per bin'); ax.legend(frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig('fig_app_accuracy.pdf')
for fp, lab in [(f, 'smooth'), (f_free, 'free'), (f_net, 'network')]:
    dv = dev(fp); print(lab, 'thr %.3f rest %.3f median/N %.3f  >5: %d' % (np.nanmean(dv[:, :20]), np.nanmean(dv[:, 20:]), np.median(np.nanmean(dv, 1)), np.sum(np.nanmean(dv, 1) > 5)))
