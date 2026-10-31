# ----------------------------------------------------------------------------
# smooth_toy.py: a test of dihiggs/smooth.py with known answers.
#
# Three smooth coefficient functions a_i(m) and three monomials C_i(x) of one
# parameter x define a density; 600 "spectra" are generated as Poisson counts
# of equal-weight events in 15 GeV bins, exactly like the POWHEG spectra. The
# smooth model must (1) reproduce the exact linear-algebra answer for fixed
# smoothness, and (2) recover the true functions within its quoted
# uncertainties: the pulls (estimate - truth) / sd should have s.d. about 1.
#
# P. Bhat, N. De Filippis, K.F. Johnson, H.B. Prosper, R. Starnes, Oct 31 2026
# ----------------------------------------------------------------------------
import numpy as np
from dihiggs.smooth import SplineBasis, SmoothHEFT

rng = np.random.default_rng(7)
edges = 255.0 + 15.0 * np.arange(81)
basis = SplineBasis(30, threshold=True)
phi = basis.phi

def true_a(m):
    # three smooth functions (pb/GeV), the second changes sign
    u = (m - 250.0) / 100.0
    return np.column_stack([phi(m) * np.exp(-u / 2.5),
                            phi(m) * 0.5 * (u - 1.0) * np.exp(-u / 1.5),
                            phi(m) * 0.6 * np.exp(-((u - 1.5) / 1.0)**2)])

# exact bin integrals of the true functions (fine trapezoid rule)
fine = np.linspace(250, 1455, 120001)
cum = np.vstack([np.zeros(3), np.cumsum(0.5 * (true_a(fine[1:]) + true_a(fine[:-1])) * np.diff(fine)[:, None], axis=0)])
A_bins = np.diff(np.array([np.interp(edges, fine, cum[:, i]) for i in range(3)]).T, axis=0)   # [80, 3]

S = 600
x = rng.uniform(-1.5, 1.5, S)
C = np.column_stack([np.ones(S), x, x**2 + 0.5])           # keeps the density positive
mu = C @ A_bins.T                                           # [S, 80] expected cross sections
q = mu.sum(1) / 40000.0                                     # one event's weight, ~40,000 events
n = rng.poisson(mu / q[:, None])
t = n * q[:, None]
rows = np.repeat(np.arange(S), 80); bins = np.tile(np.arange(80), S)
I = basis.integrals(edges)
Crow, Irow, trow = C[rows], I[bins], t.ravel()
var = (trow + q[rows]) * q[rows]

model = SmoothHEFT(basis, n_coeffs=3)
# (1) fixed smoothness: compare with a direct dense solve
lam = np.array([1e6, 1e6, 1e6])
model.fit(Crow, Irow, trow, var, lam=lam)
Phi = (Crow[:, :, None] * Irow[:, None, :]).reshape(len(trow), -1)
W = 1 / var
Hd = Phi.T @ (W[:, None] * Phi) + model.Lambda
bd = np.linalg.solve(Hd, Phi.T @ (W * trow))
err = np.max(np.abs(model.beta - bd)) / np.max(np.abs(bd))
print(f'exact solve, relative difference   {err:.1e}   {"PASS" if err < 1e-8 else "FAIL"}')

# (2) smoothness by the evidence; recover the truth within the uncertainties.
# fit_poisson weights each bin by the model's own Poisson variance, which
# avoids the downward bias of weighting by the observed counts
model.fit_poisson(Crow, Irow, trow, q[rows])
var = model.var
m = np.linspace(262.5, 1447.5, 80)
a, sd = model.coefficients(m)
pull = (a - true_a(m)) / sd
print(f'lambda by evidence: {np.array2string(model.lam, precision=2)}; gamma: {np.array2string(model.gamma(), precision=1)}')
print(f'pulls of a_i(m) vs truth: mean {pull.mean():+.2f}, s.d. {pull.std():.2f}   '
      f'{"PASS" if abs(pull.std() - 1) < 0.35 and abs(pull.mean()) < 0.5 else "FAIL"}')
f, v = model.predict(Crow, Irow)
z = (trow - f) / np.sqrt(var + v)
print(f'chi2 per bin of the fit: {np.mean(z**2):.3f} (expect about 1)   {"PASS" if abs(np.mean(z**2) - 1) < 0.05 else "FAIL"}')
for i in range(3):
    print(f'  a_{i+1}: pull mean {pull[:, i].mean():+.2f}, s.d. {pull[:, i].std():.2f}; rel. sd {np.median(sd[:, i] / np.abs(true_a(m)[:, i]).max()):.4f}')
print('  pulls by m region (all functions): ', [round(float(pull[k:k+20].mean()), 2) for k in range(0, 80, 20)])
