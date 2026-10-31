exec(open('oracle.py').read().split("# Is the excess")[0].replace("print(", "(lambda *a, **k: None)("))
# covariance of the per-bin fit: (X^T W X)^{-1}, and the prediction variance at each test point
V = np.zeros((80, 23, 23))
for k in range(80):
    s = b == k; X = Ctr[s] * np.sqrt(w[s])[:, None]
    V[k] = np.linalg.inv(X.T @ X)
v = np.einsum('ski,kij,skj->sk', Ct, V, Ct)
r2 = (T - f_or)**2
canc = T.sum(1) / np.abs(Ct * A[None]).sum(2).sum(1)
dec = np.quantile(canc, np.linspace(0, 1, 11))
for lab, den in [('noise only', var), ('noise + fit uncertainty', var + v)]:
    c = r2 / den
    m = [np.nanmean(c[(canc >= a) & (canc <= bb)]) for a, bb in zip(dec[:-1], dec[1:])]
    print(f'{lab:26s} all {np.nanmean(c):.3f} thr {np.nanmean(c[:, :20]):.3f} rest {np.nanmean(c[:, 20:]):.3f} | by cancellation decile: {np.round(m, 2)}')
print('median sqrt(v)/sigma: all %.3f, lowest cancellation decile %.3f, spectrum 815 %.2f' % (np.nanmedian(np.sqrt(v / var)), np.nanmedian(np.sqrt(v / var)[canc <= dec[1]]), np.nanmedian(np.sqrt(v[815] / var[815]))))
c = r2 / (var + v); print('spectrum 815 chi2/N with fit uncertainty: %.1f' % np.nanmean(c[815]))
np.savez('fitunc.npz', v=v, canc=canc)
