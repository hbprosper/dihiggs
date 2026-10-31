import glob, numpy as np, pandas as pd, torch
from scipy.optimize import brentq
from dihiggs import heftnet as NN, counts as CN, laplace as LA
R = '/home/claude/work/code/runs/LAPLACE_ENS_bootstrap_huber_d2_1'
P = [np.load(f) for f in sorted(glob.glob(f'{R}/member_*/predictions.npz'))]
abar = np.load('/home/claude/coef/coeffs_bins.npz')['A'].mean(0)          # [80, 23] in pb, mixture mean
m0 = NN.HEFTNet().double()
# quick-test rows of the training file
df = pd.read_csv('/home/claude/work/data/heft_wide_gauss_traindata_13TeV.csv.gz')
N = len(df); tr = int(0.864 * N); te = int(0.120 * N); va = N - tr - te
counted, _ = CN.recover_counts(df); quick = counted.iloc[tr + va:]
Xq = torch.tensor(quick[NN.FEATURES].to_numpy(np.float64))
Cq = LA.wilson_coefficients(m0, Xq).numpy()
bq = np.rint(quick.mhh.to_numpy() * 100 - 0.5).astype(int)
Sq = np.abs(Cq * abar[bq]).sum(1)
mu_q = np.mean([p['mu_quick'] for p in P], 0)
s2_q = np.mean([p['v_quick'] for p in P], 0) + np.var([p['mu_quick'] for p in P], 0)
tq = quick.sigma.to_numpy(); vq = tq * quick.quantum.to_numpy()
# test spectra
sp = pd.read_csv('/home/claude/work/data/heft_wide_gauss_testdata_13TeV.csv'); bins = [c for c in sp.columns if c.isdigit()]
T = sp[bins].to_numpy(float); nS = len(T); _, qt, _ = CN.recover_counts_wide(T)
xb = (np.arange(80) + 0.5) / 100; th = sp[['klambda','CT','CTT','CGGH','CGGHH']].to_numpy(np.float32).astype(np.float64)
Xt = torch.tensor(np.column_stack([np.tile(xb, nS), np.repeat(th, 80, axis=0)]))
Ct = LA.wilson_coefficients(m0, Xt).numpy().reshape(nS, 80, 23)
St = np.abs(Ct * abar[None]).sum(2)
mu = np.mean([p['mu_test'] for p in P], 0).reshape(nS, 80)
s2 = (np.mean([p['v_test'] for p in P], 0) + np.var([p['mu_test'] for p in P], 0)).reshape(nS, 80)
var = np.where(T > 0, T * qt[:, None], np.nan)
mhh = 255 + 15 * (np.arange(80) + 0.5)
canc = T.sum(1) / St.sum(1)
# --- fit both discrepancy models bin by bin on the quick-test rows ---
def per_bin(shape):
    out = np.zeros(80)
    for k in range(80):
        s = (bq == k) & (vq > 0)
        r2, v0, sh = (tq - mu_q)[s]**2, vq[s] + s2_q[s], shape[s]
        g = lambda c: np.mean(r2 / (v0 + c * sh)) - 1
        out[k] = 0.0 if g(0) <= 0 else brentq(g, 0, 1e12)
    return out
a_bin = per_bin(vq)                  # a(m): extra variance a * sigma^2
e2_bin = per_bin(Sq**2)              # eps(m)^2: extra variance eps^2 * S^2
# smooth versions: a(m) as in the appendix; log eps(m) as a quadratic in m
from scipy.optimize import least_squares
nb = np.bincount(bq, minlength=80)
pa = least_squares(lambda p: (p[0]*np.exp(-(mhh-255)/p[1]) + p[2] - a_bin)*np.sqrt(nb), [0.6, 100, 0.08], bounds=([0,5,0],[10,2000,1])).x
a_sm = pa[0]*np.exp(-(mhh-255)/pa[1]) + pa[2]
eps_bin = np.sqrt(e2_bin); ok = eps_bin > 0
u = (mhh - 855) / 600
pe = np.polyfit(u[ok], np.log(eps_bin[ok]), 2, w=np.sqrt(nb[ok]))
eps_sm = np.exp(np.polyval(pe, u))
print('a(m): %.3f exp(-(m-255)/%.0f) + %.3f' % tuple(pa))
print('eps(m) smooth: at 262, 400, 600, 900, 1200, 1447 GeV:', np.round(np.exp(np.polyval(pe, (np.array([262.5,400,600,900,1200,1447.5])-855)/600)) * 100, 3), '%')
models = {'none (mixture only)': np.zeros_like(var),
          'a(m) sigma^2 (current)': a_sm[None] * var,
          'eps(m)^2 S^2 (new)': (eps_sm[None] * St)**2,
          'both, refitted': None}
# both terms: fit eps^2 per bin on top of a(m) sigma^2? keep it simple: sum of the two smooth terms, each fitted alone, would double count; skip.
models.pop('both, refitted')
q5 = np.quantile(canc, [0, .2, .4, .6, .8, 1])
print(f"\n{'':26s} {'sd(z)':>6s} {'255-555':>8s} {'>555':>6s} {'|z|<1':>6s} {'|z|<2':>6s} | <z^2> by cancellation quintile (1 = strongest)")
res = {}
for lab, extra in models.items():
    z = (T - mu) / np.sqrt(var + s2 + extra); res[lab] = z
    zz = z[np.isfinite(z)]
    qz = [np.nanmean(z[(canc >= a) & (canc <= b)]**2) for a, b in zip(q5[:-1], q5[1:])]
    print(f'{lab:26s} {np.std(zz):6.3f} {np.nanstd(z[:, :20]):8.3f} {np.nanstd(z[:, 20:]):6.3f} {np.mean(np.abs(zz)<1):6.3f} {np.mean(np.abs(zz)<2):6.3f} | {np.round(qz, 2)}')
w815 = {lab: np.nanmean(z[815]**2) for lab, z in res.items()}
print('spectrum 815 <z^2>:', {k: round(v, 1) for k, v in w815.items()})
tot_new = np.sqrt(var + s2 + (eps_sm[None] * St)**2) / np.sqrt(var); tot_old = np.sqrt(var + s2 + a_sm[None] * var) / np.sqrt(var)
print('median total width / noise: old %.2f, new %.2f;  most-cancelling fifth: old %.2f new %.2f; least: old %.2f new %.2f' % (
      np.nanmedian(tot_old), np.nanmedian(tot_new), np.nanmedian(tot_old[canc <= q5[1]]), np.nanmedian(tot_new[canc <= q5[1]]),
      np.nanmedian(tot_old[canc >= q5[4]]), np.nanmedian(tot_new[canc >= q5[4]])))
np.savez('newerr.npz', mhh=mhh, a_bin=a_bin, a_sm=a_sm, eps_bin=eps_bin, eps_sm=eps_sm, pa=pa, pe=pe, St=St, canc=canc,
         z_old=res['a(m) sigma^2 (current)'], z_new=res['eps(m)^2 S^2 (new)'], z_none=res['none (mixture only)'])
