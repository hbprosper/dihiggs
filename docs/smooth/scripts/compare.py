import numpy as np, glob, pickle, sys
d = np.load('inputs.npz'); T, qt, Ct = d['T'], d['q_t'], d['C_t']; nS = len(T)
C, b, t, q = d['C_fit'], d['b_fit'], d['t_fit'], d['q_fit']
Ct3 = Ct.reshape(nS, 80, 23)
def free_fit(poisson):
    A = np.zeros((80, 23))
    for k in range(80):
        s = b == k; Ck, tk, qk = C[s], t[s], q[s]; var = qk * (tk + qk)
        for it in range(12 if poisson else 1):
            w = 1 / var; a = np.linalg.solve(Ck.T @ (w[:, None] * Ck), Ck.T @ (w * tk))
            f = Ck @ a; new = np.where(f > 0, qk * np.maximum(f, qk), qk * (tk + qk)); var = np.sqrt(var * new)
        A[k] = a
    return np.einsum('ski,ki->sk', Ct3, A)
n = T / qt[:, None]
def metrics(f):
    mu = f / qt[:, None]; pos = mu > 0
    ney = np.where(n > 0, (n - mu)**2 / np.where(n > 0, n, 1), np.nan)
    pea = np.where(pos, (n - mu)**2 / np.where(pos, mu, 1), np.nan)
    dev = np.where(pos, 2 * (mu - n + np.where(n > 0, n * np.log(np.where(n > 0, n, 1) / np.where(pos, mu, 1)), 0)), np.nan)
    return [np.nanmean(x[:, sl]) for x in (ney, pea, dev) for sl in (slice(0, 20), slice(20, 80))] + [np.median(np.nanmean(dev, 1))]
mu_ens = np.mean([np.load(f)['mu_test'] for f in glob.glob('/home/claude/work/code/runs/LAPLACE_ENS_bootstrap_huber_d2_1/member_*/predictions.npz')], 0).reshape(nS, 80)
rows = {'ensemble (Neyman-type FT)': mu_ens, 'free fit, Neyman weights': free_fit(False), 'free fit, Poisson (IRLS)': free_fit(True),
        'smooth J=40, Poisson': np.load('pass1.npz')['f']}
for extra in sys.argv[1:]:
    rows[extra] = np.load(extra)['f']
print(f"{'':28s} {'Neyman thr':>10s} {'>555':>6s} | {'Pearson thr':>11s} {'>555':>6s} | {'deviance thr':>12s} {'>555':>6s} | med dev/N")
for k, f in rows.items():
    m = metrics(f); print(f'{k:28s} {m[0]:10.3f} {m[1]:6.3f} | {m[2]:11.3f} {m[3]:6.3f} | {m[4]:12.3f} {m[5]:6.3f} | {m[6]:.3f}')
