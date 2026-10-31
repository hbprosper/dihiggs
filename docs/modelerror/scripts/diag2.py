exec(open('diag.py').read().split("cnt = ")[0])
a, b2, dev = stats(mu_or)
d_spec = np.nanmean(dev, 1); d_thr = np.nanmean(dev[:, :20], 1)
tot = T.sum(1)
import scipy.stats as S
print('spearman(spectrum deviance, total sigma) %.3f' % S.spearmanr(d_spec, tot).correlation)
for j, nm in enumerate(['klambda','CT','CTT','CGGH','CGGHH']):
    print(f'  spearman(threshold deviance, |{nm}|) {S.spearmanr(d_thr, np.abs(th[:, j])).correlation:+.3f}   (theta) {S.spearmanr(d_thr, th[:, j]).correlation:+.3f}')
# relative size of the cancellation: total / sum of |terms|
terms = np.abs(Ct * A[None]).sum(2).sum(1)
canc = tot / terms
print('spearman(spectrum deviance, cancellation ratio sigma/sum|terms|) %.3f' % S.spearmanr(d_spec, canc).correlation)
print('deviance by cancellation quintile:', [round(float(np.nanmean(d_spec[(canc >= lo) & (canc < hi)])), 3) for lo, hi in zip(np.quantile(canc, [0,.2,.4,.6,.8]), np.r_[np.quantile(canc, [.2,.4,.6,.8]), np.inf])])
# lag correlation of residuals in the threshold region
z = (n - mu_or) / np.sqrt(np.where(mu_or > 0, mu_or, np.nan))
for lag in [1, 2, 3, 5]:
    x1, x2 = z[:, :20 - lag].ravel(), z[:, lag:20].ravel(); g = np.isfinite(x1) & np.isfinite(x2)
    y1, y2 = z[:, 20:79 - lag].ravel(), z[:, 20 + lag:79].ravel(); h = np.isfinite(y1) & np.isfinite(y2)
    print(f'lag {lag}: corr thr {np.corrcoef(x1[g], x2[g])[0,1]:.3f}   rest {np.corrcoef(y1[h], y2[h])[0,1]:.3f}')
# training-set check with the same (data) variance convention as the test
np.savez('diag.npz', dev=dev, d_spec=d_spec, tot=tot, canc=canc, z=z, f_or=f_or, mu=mu, T=T, qt=qt, A=A)
