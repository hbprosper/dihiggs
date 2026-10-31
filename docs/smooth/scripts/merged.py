# merged-bin test: fit on 30 GeV bins (pairs of 15 GeV bins), predict the 15 GeV bins of the test spectra
import numpy as np, pandas as pd, torch
from dihiggs import heftnet as NN, counts as CN, laplace as LA
from dihiggs.smooth import SplineBasis, SmoothHEFT
df = pd.read_csv('/home/claude/work/data/heft_wide_gauss_traindata_13TeV.csv.gz')
counted, _ = CN.recover_counts(df); full = CN.add_empty_bins(counted)
full['bin'] = np.rint(full.mhh.to_numpy() * 100 - 0.5).astype(int); full['pair'] = full['bin'] // 2
# all rows of the training-file spectra: sum the two bins of each pair
g = full.groupby(['ID', 'pair']).agg(sigma=('sigma', 'sum'), quantum=('quantum', 'first'), nb=('bin', 'size'),
        klambda=('klambda', 'first'), CT=('CT', 'first'), CTT=('CTT', 'first'), CGGH=('CGGH', 'first'), CGGHH=('CGGHH', 'first')).reset_index()
g = g[g.nb == 2]
g['mhh'] = 0.0
m0 = NN.HEFTNet().double()
C = LA.wilson_coefficients(m0, torch.tensor(g[NN.FEATURES].to_numpy(np.float64))).numpy()
pair, t, q = g.pair.to_numpy(), g.sigma.to_numpy(), g.quantum.to_numpy()
e30 = 255.0 + 30.0 * np.arange(41); e15 = 255.0 + 15.0 * np.arange(81)
d = np.load('inputs.npz'); T, qt, Ct = d['T'], d['q_t'], d['C_t']; nS = len(T)
basis = SplineBasis(40, threshold=True)
m = SmoothHEFT(basis).fit_poisson(C, basis.integrals(e30)[pair], t, q)
f_s, v_s = m.predict(Ct, np.tile(basis.integrals(e15), (nS, 1))); f_s, v_s = f_s.reshape(nS, 80), v_s.reshape(nS, 80)
# free fit on the 30 GeV bins, each 15 GeV bin given half
Ct3 = Ct.reshape(nS, 80, 23); A = np.zeros((40, 23))
for k in range(40):
    s = pair == k; Ck, tk, qk = C[s], t[s], q[s]; var = qk * (tk + qk)
    for it in range(12):
        w = 1 / var; a = np.linalg.solve(Ck.T @ (w[:, None] * Ck), Ck.T @ (w * tk)); fk = Ck @ a
        var = np.sqrt(var * np.where(fk > 0, qk * np.maximum(fk, qk), qk * (tk + qk)))
    A[k] = a
f_h = 0.5 * np.einsum('ski,ki->sk', Ct3, np.repeat(A, 2, axis=0))
f_ref = np.load('res_J40_thr.npz')['f']                 # smooth model fitted on the 15 GeV bins
n = T / qt[:, None]
def dev(f):
    mu = f / qt[:, None]; pos = mu > 0
    return np.where(pos, 2 * (mu - n + np.where(n > 0, n * np.log(np.where(n > 0, n, 1) / np.where(pos, mu, 1)), 0)), np.nan)
# the split within each pair: difference of the two halves, in units of its noise
D_obs = T[:, 0::2] - T[:, 1::2]
def split_chi2(f):
    Dp = f[:, 0::2] - f[:, 1::2]; var = qt[:, None] * (np.maximum(f[:, 0::2], qt[:, None]) + np.maximum(f[:, 1::2], qt[:, None]))
    return (D_obs - Dp)**2 / var
print(f"{'':36s} {'deviance 255-555':>17s} {'>555':>6s} | {'split chi2 255-555':>18s} {'>555':>6s}")
for lab, f in [('free fit on 30 GeV, halved', f_h), ('smooth, fitted on 30 GeV bins', f_s), ('smooth, fitted on 15 GeV bins (ref)', f_ref)]:
    dv, sc = dev(f), split_chi2(f)
    print(f'{lab:36s} {np.nanmean(dv[:, :20]):17.3f} {np.nanmean(dv[:, 20:]):6.3f} | {np.nanmean(sc[:, :10]):18.3f} {np.nanmean(sc[:, 10:]):6.3f}')
np.savez('merged.npz', f_h=f_h, f_s=f_s, v_s=v_s)
