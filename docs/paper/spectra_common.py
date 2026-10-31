import glob, numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.optimize import brentq, least_squares
from dihiggs import counts as CN
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
chi2N = np.nanmean(np.where(T > 0, (T - mu)**2 / np.where(noise > 0, noise, np.nan), np.nan), axis=1)
