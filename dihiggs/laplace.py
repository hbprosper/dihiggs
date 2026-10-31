# ----------------------------------------------------------------------------
# laplace.py: the linearised Laplace approximation of the posterior of a
#             network, and mixtures of such approximations.
#
# The design is described in docs/bnn/bnn_plan2.pdf. In brief, near a trained
# network w_hat the network is replaced by its linearisation
#
#     f(x; w) ~ f(x; w_hat) + j(x) . (w - w_hat),     j(x) = df/dw at w_hat,
#
# so that, with a Gaussian likelihood of known variances sigma_k^2 and a
# Gaussian prior of precisions tau (one per group of parameters), the
# posterior is Gaussian with precision
#
#     H = G + diag(tau),    G = sum_k j_k j_k^T / sigma_k^2     (eq. 3)
#
# and the prediction at x is Gaussian with mean f(x; w_hat) and variance
#
#     v(x) = j(x)^T H^{-1} j(x)                                  (eq. 4)
#
# (MacKay 1992; the "GLM predictive" of Immer et al. 2021). A mixture of such
# approximations, one about each network of an ensemble, approximates a
# posterior with several modes (eq. 5).
#
# HEFTNet has a special structure, used here to make everything fast and
# exact: its output is
#
#     f(x) = sum_i C_i(theta) a_i(mhh; w),    i = 1, ..., 23,
#
# where the C_i are known polynomials of the HEFT parameters theta and the
# a_i are the outputs of the network, which depend on mhh ONLY. So
#
#     j(x) = sum_i C_i(theta) da_i(mhh)/dw = A(mhh)^T C(theta),
#
# with A(mhh) the 23 x n_par Jacobian of the 23 functions at that mhh.
# There are only 80 distinct values of mhh (the bins), so G needs 80
# Jacobians, not one per row:
#
#     G = sum_b A_b^T S_b A_b,   S_b = sum_{k in bin b} C_k C_k^T / sigma_k^2,
#
# and the prediction variance is C^T K_b C, where K_b = A_b H^{-1} A_b^T is
# the 23 x 23 posterior covariance of the functions a_i(mhh) in bin b.
# A general (slower) route, one gradient per row, is kept for other models
# and as a check.
#
# Classes and functions
#   wilson_coefficients(model, X)   the C_i(theta) of HEFTNet for every row
#   Laplace                         one network: G, tau, H, predictions,
#                                   MacKay's prior updates, evidence, samples
#   mixture(mus, vs, weights)       combine the predictions of several
#
# Harrison B. Prosper and Claude
# Created: Sat Oct 31 2026
# ----------------------------------------------------------------------------
import math
import copy
import numpy as np
import torch
from torch.func import functional_call, jacrev, vmap, grad

from dihiggs.hmc import ParameterView

__all__ = ['wilson_coefficients', 'Laplace', 'mixture']

# ----------------------------------------------------------------------------
# A note on torch.func, used below
#
#   functional_call(model, params, (x,))  evaluates the model with the
#        parameters given in the dictionary params (name -> tensor) instead
#        of its own. This lets us treat the model as a mathematical function
#        f(params, x) that can be differentiated with respect to params.
#   jacrev(fn)(p, x)   the Jacobian of fn with respect to its first argument
#        p: for an output of shape [B, 23] and a parameter tensor of shape
#        [25, 25], a tensor of shape [B, 23, 25, 25].
#   vmap(fn)           applies fn to every row of its input, as if in a loop,
#        but in one vectorised call.
# ----------------------------------------------------------------------------


def wilson_coefficients(model, X):
    '''
    The 23 coefficients C_i(theta) of HEFTNet for every row of X ([N, 6]),
    returned as a tensor [N, 23].

    HEFTNet computes f = sum_i C_i * A_i inside forward(), with
    A = self.coeffs(mhh). Since f is LINEAR in A, C is the derivative of f
    with respect to A. We obtain it without touching heftnet.py: a copy of
    the model is given a replacement coeffs() that returns a fixed tensor A
    (all ones) that requires a gradient; then f.sum().backward() puts
    df/dA = C in A.grad, row by row.
    '''
    m = copy.deepcopy(model)
    A = torch.ones(len(X), 23, dtype=X.dtype, device=X.device,
                   requires_grad=True)
    # Assigning a function to m.coeffs makes this copy use it instead of
    # the class's own method (an "instance attribute" hides the method).
    m.coeffs = lambda mhh: A
    f = m(X)
    f.sum().backward()
    C = A.grad.detach()
    # check: f must equal sum_i C_i * A_i = sum_i C_i (A is all ones)
    assert torch.allclose(f.detach(), C.sum(dim=1), rtol=1e-6, atol=0), \
        'wilson_coefficients: the model is not linear in its coeffs()'
    return C


class Laplace:
    '''
    The linearised Laplace approximation about one trained network.

    Arguments
    ---------
    model    : the trained network (it is copied and converted to double
               precision; the original is not changed).
    grouping : the groups of the prior, as in hmc.ParameterView ('tensor':
               one group per parameter tensor, 16 for HEFTNet).
    device   : where to compute.

    Typical use
    -----------
        lap = Laplace(model)
        lap.fit_hessian(X, V)       # G, eq. (3); V = noise variances
        lap.fit_prior()             # MacKay's updates of tau, eq. (6)
        mu, v = lap.predict(X_new)  # eq. (4)
    '''
    def __init__(self, model, grouping='tensor', device='cpu'):
        self.device = torch.device(device)
        # a private copy in double precision: G and H span many orders of
        # magnitude, and 64-bit numbers keep about 16 significant digits
        self.model = copy.deepcopy(model).to(self.device).double()
        self.model.eval()
        self.view = ParameterView(self.model, grouping)
        self.n = self.view.n
        self.w_hat = self.view.get()                 # the expansion point
        self.group_index = self.view.group_index.to(self.device)
        self.n_groups = len(self.view.groups)
        self.group_sizes = torch.tensor([g['size'] for g in self.view.groups],
                                        dtype=torch.float64, device=self.device)
        self.G = None          # the Gauss-Newton matrix
        self.tau = None        # prior precision of each group
        self.L = None          # Cholesky factor of H = G + diag(tau)
        self._factorised = hasattr(self.model, 'coeffs')
        # the parameters as a dictionary name -> tensor, for functional_call
        self._params = {k: v.detach() for k, v in
                        self.model.named_parameters()}
        self._wrapper = _CoeffsOnly(self.model)

    # ------------------------------------------------------------------------
    # Jacobians
    # ------------------------------------------------------------------------
    def _flatten(self, jac, lead):
        # jac: dictionary name -> tensor of shape lead + parameter shape.
        # Returns one tensor of shape lead + [n], in the order of the view.
        pieces = [jac[name].reshape(*lead, -1) for name in self.view.names]
        return torch.cat(pieces, dim=-1)

    def coeff_jacobian(self, mhh):
        '''
        The Jacobian of the 23 functions a_i(mhh) with respect to the
        parameters, at the given values of mhh (a 1-d tensor of length B).
        Returns a tensor [B, 23, n].
        '''
        mhh = torch.as_tensor(mhh, dtype=torch.float64, device=self.device)
        # jacrev differentiates _coeffs_with with respect to its first
        # argument, the dictionary of parameters: one Jacobian per tensor
        jac = jacrev(self._coeffs_with)(self._params, mhh)
        return self._flatten(jac, (len(mhh), 23))

    def _coeffs_with(self, params, x):
        # model.coeffs(x) evaluated with the parameters in the dictionary
        # params. functional_call always calls forward(), so we call it on
        # _CoeffsOnly, a wrapper whose forward() IS model.coeffs(). The
        # wrapper holds the model as its sub-module "m", so its parameter
        # names carry the prefix "m.".
        named = {'m.' + k: v for k, v in params.items()}
        return functional_call(self._wrapper, named, (x,))

    def row_jacobian(self, X):
        '''
        The gradient of the output with respect to the parameters for every
        row of X ([B, 6]): a tensor [B, n]. The general (slow) route.
        '''
        X = torch.as_tensor(X, dtype=torch.float64, device=self.device)

        def f_one(params, x):
            # the network's output for ONE row x (shape [6])
            return functional_call(self.model, params,
                                   (x.unsqueeze(0),)).reshape(-1)[0]
        jac = vmap(grad(f_one), in_dims=(None, 0))(self._params, X)
        return self._flatten(jac, (len(X),))

    # ------------------------------------------------------------------------
    # G, eq. (3)
    # ------------------------------------------------------------------------
    def fit_hessian(self, X, V, method='auto', chunk_size=4096):
        '''
        Compute G = sum_k j_k j_k^T / V_k over the rows of X ([N, 6]), with
        V the noise variance of each row ([N]).

        method : 'factorised' (HEFTNet structure, eq. above; fast and exact),
                 'rows' (one gradient per row; any model), or 'auto'.
        '''
        X = torch.as_tensor(X, dtype=torch.float64, device=self.device)
        W = 1.0 / torch.as_tensor(V, dtype=torch.float64, device=self.device)
        if method == 'auto':
            method = 'factorised' if self._factorised else 'rows'
        G = torch.zeros(self.n, self.n, dtype=torch.float64, device=self.device)

        if method == 'factorised':
            C = wilson_coefficients(self.model, X)              # [N, 23]
            # the distinct mhh values (bins) and, for every row, the number
            # of its bin: torch.unique(..., return_inverse=True)
            mhh, b_of_row = torch.unique(X[:, 0], return_inverse=True)
            A = self.coeff_jacobian(mhh)                         # [B, 23, n]
            for b in range(len(mhh)):
                rows = b_of_row == b                  # True for rows in bin b
                Cb = C[rows]                                     # [n_b, 23]
                # S_b = sum_k w_k C_k C_k^T, a 23 x 23 matrix
                Sb = Cb.T @ (W[rows, None] * Cb)
                G += A[b].T @ Sb @ A[b]
            self._mhh_bins = mhh
            self._A = A
        elif method == 'rows':
            for start in range(0, len(X), chunk_size):
                s = slice(start, start + chunk_size)
                J = self.row_jacobian(X[s])                      # [B, n]
                G += J.T @ (W[s, None] * J)
        else:
            raise ValueError(f'fit_hessian: unknown method {method!r}')
        # symmetrise (round-off can make G very slightly asymmetric)
        self.G = 0.5 * (G + G.T)
        if self.tau is None:
            self.set_prior(None)
        return self.G

    # ------------------------------------------------------------------------
    # the prior and H
    # ------------------------------------------------------------------------
    def set_prior(self, tau):
        '''
        Set the prior precisions, one per group (a sequence of length
        n_groups), and factorise H. With tau = None, each group starts at
        tau_g = n_g / sum(w_g^2), the value for which the expansion point
        is a typical draw from the prior.
        '''
        if tau is None:
            ss = self._group_sum(self.w_hat**2)
            tau = self.group_sizes / ss
        self.tau = torch.as_tensor(tau, dtype=torch.float64,
                                   device=self.device).clone()
        self._factorise()

    def _group_sum(self, x):
        # the sum of x over the elements of each group: a vector [n_groups]
        out = torch.zeros(self.n_groups, dtype=torch.float64,
                          device=self.device)
        return out.index_add_(0, self.group_index, x.double())

    def _factorise(self):
        # H = G + diag(tau of each parameter) = L L^T (Cholesky)
        H = self.G + torch.diag(self.tau[self.group_index])
        self.L = torch.linalg.cholesky(H)
        self._Hinv_diag = None

    def H_inverse_diagonal(self):
        # the diagonal of H^{-1}: the posterior variances of the parameters
        if self._Hinv_diag is None:
            self._Hinv_diag = torch.diagonal(torch.cholesky_inverse(self.L))
        return self._Hinv_diag

    def gamma(self):
        '''
        The number of well-determined parameters, per group and in total:
        gamma_g = n_g - tau_g * sum_{i in g} (H^{-1})_ii, and the total
        gamma = tr(G H^{-1}) (eq. 1 of the plan). Returns (total, per group).
        '''
        per_group = self.group_sizes - self.tau * \
            self._group_sum(self.H_inverse_diagonal())
        return float(per_group.sum()), per_group

    def fit_prior(self, n_iter=100, rtol=1e-4, damping=0.5, verbose=False):
        '''
        MacKay's evidence updates of the prior precisions (eq. 6):
            tau_g = gamma_g / sum_{i in g} w_hat_i^2,
        iterated until the relative change of every tau_g is below rtol.
        Each iteration costs one Cholesky factorisation. The plain update
        tends to overshoot and oscillate, so each step moves only part of
        the way, in the logarithm: ln tau <- ln tau + damping * (ln new -
        ln tau). damping = 1 is the plain update.
        '''
        ss = self._group_sum(self.w_hat**2)
        for it in range(n_iter):
            _, gam = self.gamma()
            new = torch.clamp(gam, min=1e-8) / ss
            change = float(torch.max(torch.abs(new / self.tau - 1)))
            self.tau = torch.exp(torch.log(self.tau) + damping *
                                 (torch.log(new) - torch.log(self.tau)))
            self._factorise()
            if verbose:
                print(f'  iteration {it + 1}: gamma = {self.gamma()[0]:.1f}, '
                      f'largest relative change of tau {change:.1e}')
            if change < rtol:
                break
        return self.tau

    # ------------------------------------------------------------------------
    # predictions, eq. (4)
    # ------------------------------------------------------------------------
    def coeff_covariance(self, mhh=None):
        '''
        The posterior covariance of the 23 functions a_i at each value of
        mhh (default: the bins used in fit_hessian): a tensor [B, 23, 23],
        K_b = A_b H^{-1} A_b^T. HEFTNet only.
        '''
        A = self._A if mhh is None else self.coeff_jacobian(mhh)
        # M_b = L^{-1} A_b^T, so that K_b = M_b^T M_b
        # A.permute(2, 0, 1) has shape [n, B, 23]; reshape lays the B
        # matrices side by side as one [n, B*23] right-hand side
        rhs = A.permute(2, 0, 1).reshape(self.n, -1)
        M = torch.linalg.solve_triangular(self.L, rhs, upper=False)
        M = M.reshape(self.n, len(A), 23).permute(1, 0, 2)      # [B, n, 23]
        return M.transpose(1, 2) @ M

    def predict(self, X, method='auto', chunk_size=4096):
        '''
        The mean f(x; w_hat) and the variance j^T H^{-1} j at every row of X.
        Returns two double-precision tensors [N].
        '''
        X = torch.as_tensor(X, dtype=torch.float64, device=self.device)
        with torch.no_grad():
            mu = self.model(X).detach().reshape(len(X), -1)[:, 0]
        if method == 'auto':
            method = 'factorised' if self._factorised else 'rows'
        if method == 'factorised':
            C = wilson_coefficients(self.model, X)
            mhh, b_of_row = torch.unique(X[:, 0], return_inverse=True)
            K = self.coeff_covariance(mhh)                       # [B, 23, 23]
            # v_k = C_k^T K_{b(k)} C_k for every row, in one call
            v = torch.einsum('ki,kij,kj->k', C, K[b_of_row], C).detach()
        else:
            v = torch.empty(len(X), dtype=torch.float64, device=self.device)
            for start in range(0, len(X), chunk_size):
                s = slice(start, start + chunk_size)
                J = self.row_jacobian(X[s])
                Z = torch.linalg.solve_triangular(self.L, J.T, upper=False)
                v[s] = (Z**2).sum(dim=0)
        return mu, v

    # ------------------------------------------------------------------------
    # evidence and samples
    # ------------------------------------------------------------------------
    def log_evidence(self, chi2, log_det_noise=0.0):
        '''
        The Laplace estimate of the log marginal likelihood (eq. 7 of the
        plan), up to terms common to all networks:
          -chi2/2 - sum_i tau_i w_i^2 / 2 + sum_g n_g ln(tau_g) / 2
          - ln det H / 2 - log_det_noise / 2,
        where chi2 = sum_k (f_k - t_k)^2 / V_k at w_hat and log_det_noise =
        sum_k ln V_k (needed only if V depends on the network).
        '''
        prior = float(torch.sum(self.tau[self.group_index] * self.w_hat**2))
        log_det_H = 2.0 * float(torch.sum(torch.log(torch.diagonal(self.L))))
        return (-0.5 * chi2 - 0.5 * prior
                + 0.5 * float(torch.sum(self.group_sizes * torch.log(self.tau)))
                - 0.5 * log_det_H - 0.5 * log_det_noise)

    def sample(self, n_samples, generator=None):
        '''
        Draw parameter vectors from N(w_hat, H^{-1}): w = w_hat + L^{-T} z,
        z standard normal. Returns a tensor [n_samples, n].
        '''
        z = torch.randn(self.n, n_samples, dtype=torch.float64,
                        device=self.device, generator=generator)
        d = torch.linalg.solve_triangular(self.L.T, z, upper=True)
        return (self.w_hat[:, None] + d).T

    def evaluate(self, w, X):
        '''
        The FULL (not linearised) network with parameters w at the rows of X.
        Used to check the linearisation. The expansion point is restored.
        '''
        X = torch.as_tensor(X, dtype=torch.float64, device=self.device)
        try:
            self.view.set(w)
            with torch.no_grad():
                return self.model(X)
        finally:
            self.view.set(self.w_hat)

    def eigenvalues(self):
        # the eigenvalues of H, smallest first
        H = self.G + torch.diag(self.tau[self.group_index])
        return torch.linalg.eigvalsh(H)

    # ------------------------------------------------------------------------
    def state(self):
        # what is worth saving. G must stay in double precision: its
        # eigenvalues span 13 orders of magnitude, and single precision
        # (7 digits) would make H fail to be positive definite.
        return {'w_hat': self.w_hat.cpu(), 'tau': self.tau.cpu(),
                'G': self.G.cpu()}

    def load_state(self, state):
        # restore G and tau (the model must be the same network)
        self.G = state['G'].to(self.device, torch.float64)
        self.set_prior(state['tau'])


class _CoeffsOnly(torch.nn.Module):
    # A thin wrapper whose forward() is the wrapped model's coeffs(), so that
    # functional_call can evaluate coeffs() with substituted parameters. The
    # wrapped model is registered as a sub-module called "m"; the parameter
    # names of the wrapper are therefore "m.P.0.weight" etc.
    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, x):
        return self.m.coeffs(x)


def mixture(mus, vs, weights=None):
    '''
    Combine the predictions of several Laplace approximations (eq. 5):
    mus and vs are lists (or arrays [M, ...]) of means and variances.
    Returns (mean, total variance, within variance, between variance).
    '''
    mus = np.asarray(mus, dtype=np.float64)
    vs = np.asarray(vs, dtype=np.float64)
    M = len(mus)
    w = np.full(M, 1.0 / M) if weights is None else \
        np.asarray(weights, dtype=np.float64) / np.sum(weights)
    # w.reshape(-1, *[1]*k) lets the weights multiply arrays of any shape
    w = w.reshape(-1, *([1] * (mus.ndim - 1)))
    mean = np.sum(w * mus, axis=0)
    within = np.sum(w * vs, axis=0)
    between = np.sum(w * (mus - mean)**2, axis=0)
    return mean, within + between, within, between
