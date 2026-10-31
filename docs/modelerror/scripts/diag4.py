exec(open('oracle.py').read().split("# Is the excess")[0].replace("print(", "(lambda *a, **k: None)("))
from scipy.optimize import brentq
F = np.load('fitunc.npz'); v, canc = F['v'], F['canc']
S2 = (np.abs(Ct * A[None]).sum(2))**2          # (sum_i |C_i a_i|)^2 per test bin
r2 = (T - f_or)**2
rng = np.random.default_rng(0); idx = rng.permutation(len(T)); HA, HB = idx[:500], idx[500:]
def fit_bin(extra_shape, rows):
    out = np.zeros(80)
    for k in range(80):
        rr, vv, ee, vf = r2[rows, k], var[rows, k], extra_shape[rows, k], v[rows, k]
        g = np.isfinite(rr / vv)
        f = lambda c: np.mean(rr[g] / (vv[g] + vf[g] + c * ee[g])) - 1
        out[k] = 0.0 if f(0) <= 0 else brentq(f, 0, 1e6 if extra_shape is S2 else 50, maxiter=200)
    return out
dec = np.quantile(canc[HB], np.linspace(0, 1, 6))
for lab, shape in [('a(m) sigma^2 (current)', var), ('eps(m)^2 (sum|C_i a_i|)^2', S2)]:
    c = fit_bin(shape, HA)
    z2 = r2[HB] / (var[HB] + v[HB] + c[None] * shape[HB])
    m = [np.nanmean(z2[(canc[HB] >= a) & (canc[HB] <= bb)]) for a, bb in zip(dec[:-1], dec[1:])]
    print(f'{lab:28s} <z^2> half B: all {np.nanmean(z2):.3f} thr {np.nanmean(z2[:, :20]):.3f} rest {np.nanmean(z2[:, 20:]):.3f} | by cancellation quintile {np.round(m, 2)}')
    if shape is S2: print('   eps(m): first bins', np.sqrt(c[:5]).round(4), ' bins 30-34', np.sqrt(c[30:35]).round(4))
