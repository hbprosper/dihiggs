import numpy as np
d = np.load('diag.npz'); z = d['z']; canc = np.load('fitunc.npz')['canc']
q5 = np.quantile(canc, [0, .2, .4, .6, .8, 1])
for i, (a, b) in enumerate(zip(q5[:-1], q5[1:])):
    s = (canc >= a) & (canc <= b)
    out = []
    for lo, hi in [(0, 20), (20, 79)]:
        x1, x2 = z[s, lo:hi - 1].ravel(), z[s, lo + 1:hi].ravel(); g = np.isfinite(x1) & np.isfinite(x2)
        out.append(np.corrcoef(x1[g], x2[g])[0, 1])
    print(f'cancellation quintile {i+1}: lag-1 residual correlation thr {out[0]:.3f}  rest {out[1]:.3f}   mean z^2 {np.nanmean(z[s]**2):.2f}')
