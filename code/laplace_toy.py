# ----------------------------------------------------------------------------
# laplace_toy.py: a test of dihiggs/laplace.py with an exact answer.
#
# For a model that is LINEAR in its parameters, f(x) = x . w + b, with a
# Gaussian likelihood of known variances and a Gaussian prior, the posterior
# is exactly Gaussian, and its precision, the prediction variances and the
# number of well-determined parameters have textbook formulae. The Laplace
# class must reproduce them to rounding error. Run with
#     python laplace_toy.py
# from the code folder.
#
# Harrison B. Prosper and Claude, Oct 31 2026
# ----------------------------------------------------------------------------
import numpy as np
import torch
import torch.nn as nn
from dihiggs.laplace import Laplace

torch.manual_seed(1)
N, D = 2000, 5
X = torch.randn(N, D, dtype=torch.float64)
V = torch.rand(N, dtype=torch.float64) + 0.5          # noise variances
model = nn.Linear(D, 1).double()                       # w: [1, D], b: [1]

lap = Laplace(model)                                   # 2 groups: w and b
lap.fit_hessian(X, V, method='rows')                   # the general route
tau = torch.tensor([3.0, 0.5], dtype=torch.float64)
lap.set_prior(tau)

# the textbook answer: design matrix [x, 1], weights 1/V
Phi = torch.cat([X, torch.ones(N, 1, dtype=torch.float64)], dim=1)
G = Phi.T @ (Phi / V[:, None])
H = G + torch.diag(torch.tensor([3.0] * D + [0.5], dtype=torch.float64))
Hinv = torch.linalg.inv(H)

Xt = torch.randn(100, D, dtype=torch.float64)
Phit = torch.cat([Xt, torch.ones(100, 1, dtype=torch.float64)], dim=1)
v_exact = torch.einsum('ki,ij,kj->k', Phit, Hinv, Phit)
gamma_exact = float(torch.trace(G @ Hinv))

_, v = lap.predict(Xt, method='rows')
gamma, _ = lap.gamma()
checks = {
    'G':        float((lap.G - G).abs().max() / G.abs().max()),
    'variance': float(((v - v_exact).abs() / v_exact).max()),
    'gamma':    abs(gamma - gamma_exact) / gamma_exact,
}
# sampling: the covariance of many draws must approach H^{-1}
W = lap.sample(200000, generator=torch.Generator().manual_seed(2))
cov = torch.cov(W.T)
checks['samples (MC)'] = float((cov - Hinv).abs().max() / Hinv.abs().max())

for name, err in checks.items():
    limit = 2e-2 if name.startswith('samples') else 1e-10
    print(f'{name:14s} relative error {err:.1e}   '
          f'{"PASS" if err < limit else "FAIL"}')

# MacKay's updates converge to a fixed point: tau_g = gamma_g / |w_g|^2
lap.fit_prior(n_iter=200, rtol=1e-12)
_, gam = lap.gamma()
ss = lap._group_sum(lap.w_hat**2)
fp = float(torch.max(torch.abs(lap.tau * ss / gam - 1)))
print(f'{"fixed point":14s} relative error {fp:.1e}   '
      f'{"PASS" if fp < 1e-8 else "FAIL"}')
