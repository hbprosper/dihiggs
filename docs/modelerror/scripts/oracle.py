import numpy as np, pandas as pd, torch
from dihiggs import heftnet as NN, counts as CN, laplace as LA
df = pd.read_csv('/home/claude/work/data/heft_wide_gauss_traindata_13TeV.csv.gz')
N = len(df); tr = int(0.864 * N)
out, _ = CN.recover_counts(df); train = out.iloc[:tr]
m = NN.HEFTNet()
Ctr = LA.wilson_coefficients(m.double(), torch.tensor(train[NN.FEATURES].to_numpy(np.float64))).numpy()
t = train.sigma.to_numpy(); q = train.quantum.to_numpy(); w = 1 / (q * (t + q))
b = np.rint(train.mhh.to_numpy() * 100 - 0.5).astype(int)
# per-bin weighted least squares: the best possible a_i(m_b) for the exact 23-term form
A = np.zeros((80, 23)); chi2_train = np.zeros(80)
for k in range(80):
    s = b == k; X = Ctr[s] * np.sqrt(w[s])[:, None]; y = t[s] * np.sqrt(w[s])
    A[k] = np.linalg.lstsq(X, y, rcond=None)[0]
    chi2_train[k] = np.mean((X @ A[k] - y)**2)
sp = pd.read_csv('/home/claude/work/data/heft_wide_gauss_testdata_13TeV.csv'); bins = [c for c in sp.columns if c.isdigit()]
T = sp[bins].to_numpy(float); _, qt, _ = CN.recover_counts_wide(T)
xb = (np.arange(80) + 0.5) / 100; th = sp[['klambda','CT','CTT','CGGH','CGGHH']].to_numpy()
Xt = torch.tensor(np.column_stack([np.tile(xb, len(T)), np.repeat(th, 80, axis=0)]))
Ct = LA.wilson_coefficients(m, Xt).numpy().reshape(len(T), 80, 23)
f_or = np.einsum('ski,ki->sk', Ct, A)
var = np.where(T > 0, T * qt[:, None], np.nan)
c_or = (f_or - T)**2 / var
P = np.load('/home/claude/work/code/runs/LAPLACE_ENS_bootstrap_huber_d2_1/member_01/predictions.npz')
import glob
mu = np.mean([np.load(f)['mu_test'] for f in glob.glob('/home/claude/work/code/runs/LAPLACE_ENS_bootstrap_huber_d2_1/member_*/predictions.npz')], 0).reshape(T.shape)
c_ens = (mu - T)**2 / var
for nm, c in [('per-bin linear fit (no network)', c_or), ('fine-tuned ensemble', c_ens)]:
    print(f'{nm:34s} median chi2/N {np.median(np.nanmean(c,1)):.3f}  thr {np.nanmean(c[:, :20]):.2f}  >555 {np.nanmean(c[:, 20:]):.3f}  spectra chi2/N>5: {np.sum(np.nanmean(c,1) > 5)}')
print('train chi2/row per-bin fit: thr %.3f rest %.3f' % (chi2_train[:20].mean(), chi2_train[20:].mean()))
tot = T.sum(1); worst = np.argsort(-np.nanmean(c_or, 1))[:5]
print('worst spectra under the per-bin fit:', worst, np.nanmean(c_or, 1)[worst].round(1), 'ensemble:', np.nanmean(c_ens, 1)[worst].round(1))
# Is the excess a per-spectrum normalisation error (POWHEG integration), common to all bins?
ok = ~np.isnan(var)
r = np.nansum(np.where(ok, T, 0), 1) / np.nansum(np.where(ok, f_or, 0), 1)
nev = np.nansum(np.where(ok, T / qt[:, None], 0), 1)
print('ratio POWHEG/fit totals: robust s.d. %.4f, expected from counts ~%.4f' % (1.4826 * np.median(np.abs(r - np.median(r))), np.median(1 / np.sqrt(nev))))
# best per-spectrum scale s minimising chi2, and the chi2 after it
s_hat = np.nansum(np.where(ok, T * f_or / var, 0), 1) / np.nansum(np.where(ok, f_or**2 / var, 0), 1)
c_sc = (s_hat[:, None] * f_or - T)**2 / var
print('with a free scale per spectrum: thr %.2f  >555 %.3f  (median scale %.4f, robust s.d. %.4f)' % (np.nanmean(c_sc[:, :20]), np.nanmean(c_sc[:, 20:]), np.median(s_hat), 1.4826*np.median(np.abs(s_hat-np.median(s_hat)))))
# neighbouring-bin correlation of the residuals within a spectrum (smooth vs independent)
z = (T - f_or) / np.sqrt(var)
for lo, hi, nm in [(0, 20, 'thr'), (20, 79, 'rest')]:
    a, bb = z[:, lo:hi-1].ravel(), z[:, lo+1:hi].ravel(); g = np.isfinite(a) & np.isfinite(bb)
    print(f'adjacent-bin residual correlation, {nm}: {np.corrcoef(a[g], bb[g])[0,1]:.3f}')
