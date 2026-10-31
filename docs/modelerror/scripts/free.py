import numpy as np, pandas as pd, torch
from scipy.optimize import brentq, least_squares
from dihiggs import heftnet as NN, counts as CN, laplace as LA
exec(open('newerr.py').read().split('# --- fit both')[0])      # quick rows, test spectra, mixture predictions
# the 23 x 80 linear model: weighted least squares per bin on the training rows
train = counted.iloc[:tr]
Ctr = LA.wilson_coefficients(m0, torch.tensor(train[NN.FEATURES].to_numpy(np.float64))).numpy()
t = train.sigma.to_numpy(); q = train.quantum.to_numpy(); w = 1 / (q * (t + q))
b = np.rint(train.mhh.to_numpy() * 100 - 0.5).astype(int)
A = np.zeros((80, 23)); Cov = np.zeros((80, 23, 23))
for k in range(80):
    s = b == k; X = Ctr[s] * np.sqrt(w[s])[:, None]
    Cov[k] = np.linalg.inv(X.T @ X); A[k] = Cov[k] @ (X.T @ (t[s] * np.sqrt(w[s])))
f_q = np.einsum('ri,ri->r', Cq, A[bq]); v_q = np.einsum('ri,rij,rj->r', Cq, Cov[bq], Cq)
f_t = np.einsum('ski,ki->sk', Ct, A); v_t = np.einsum('ski,kij,skj->sk', Ct, Cov, Ct)
S_q = np.abs(Cq * A[bq]).sum(1); S_t = np.abs(Ct * A[None]).sum(2)
def per_bin(shape, base):
    out = np.zeros(80)
    for k in range(80):
        s = (bq == k) & (vq > 0)
        r2, v0, sh = (tq - f_q)[s]**2, vq[s] + base[s], shape[s]
        g = lambda c: np.mean(r2 / (v0 + c * sh)) - 1
        out[k] = 0.0 if g(0) <= 0 else brentq(g, 0, 1e12)
    return out
nb = np.bincount(bq, minlength=80); u = (mhh - 855) / 600
e2 = per_bin(S_q**2, v_q); ok = e2 > 0
pe2 = np.polyfit(u[ok], 0.5 * np.log(e2[ok]), 2, w=np.sqrt(nb[ok])); eps2 = np.exp(np.polyval(pe2, u))
ab = per_bin(vq, v_q)
pa2 = least_squares(lambda p: (p[0]*np.exp(-(mhh-255)/p[1]) + p[2] - ab)*np.sqrt(nb), [0.6, 100, 0.08], bounds=([0,5,0],[10,2000,1])).x
a2 = pa2[0]*np.exp(-(mhh-255)/pa2[1]) + pa2[2]
q5 = np.quantile(canc, [0, .2, .4, .6, .8, 1])
def report(lab, f, v, extra):
    z = (T - f) / np.sqrt(var + v + extra); zz = z[np.isfinite(z)]
    c = (T - f)**2 / var
    qz = [np.nanmean(z[(canc >= a) & (canc <= bb)]**2) for a, bb in zip(q5[:-1], q5[1:])]
    print(f'{lab:34s} med chi2/N {np.median(np.nanmean(c,1)):.3f}  sd(z) {np.std(zz):.3f} ({np.nanstd(z[:, :20]):.3f}, {np.nanstd(z[:, 20:]):.3f})  |z|<2 {np.mean(np.abs(zz)<2):.3f} | {np.round(qz, 2)}')
    return z
print('23x80 model: a(m) = %.3f exp(-(m-255)/%.0f) + %.3f; eps at 262, 600, 1447 GeV: %s %%' % (*pa2, np.round(100*np.exp(np.polyval(pe2, (np.array([262.5, 600, 1447.5])-855)/600)), 3)))
report('23x80, no discrepancy', f_t, v_t, 0)
report('23x80 + a(m) sigma^2', f_t, v_t, a2[None] * var)
z80 = report('23x80 + eps(m)^2 S^2', f_t, v_t, (eps2[None] * S_t)**2)
eps_sm = np.load('newerr.npz')['eps_sm']; report('ensemble mixture + eps(m)^2 S^2', mu, s2, (eps_sm[None] * St)**2)
print('median width (without the noise) / noise: 23x80 fit unc. %.3f, mixture %.3f' % (np.nanmedian(np.sqrt(v_t / var)), np.nanmedian(np.sqrt(s2 / var))))
# smoothness: rms of the second difference across bins, relative to the function's size
abar_net = np.load('/home/claude/coef/coeffs_bins.npz')['A'].mean(0)
def rough(a): return np.sqrt(np.mean(np.diff(a, 2, axis=0)**2, 0)) / np.sqrt(np.mean(a**2, 0))
print('roughness (rms 2nd difference / rms value), median over i: 23x80 %.3f, network %.3f' % (np.median(rough(A)), np.median(rough(abar_net))))
np.savez('free.npz', A=A, Cov=Cov, f_t=f_t, v_t=v_t, S_t=S_t, eps2=eps2, a2=a2, abar_net=abar_net, z80=z80)
