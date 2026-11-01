# ----------------------------------------------------------------------------
# smooth.py: the smooth 23 x 80 model of the di-Higgs m_hh spectrum.
#
# The design is described in docs/smooth/smooth_plan.pdf. In brief, the
# density is
#
#     d sigma / dm (m; theta) = sum_i C_i(theta) a_i(m),     i = 1, ..., 23,
#
# with the 23 known monomials C_i of HEFTNet, and each coefficient function
# is a sum of J cubic B-splines times a threshold factor phi(m):
#
#     a_i(m) = phi(m) sum_j beta_ij B_j(m).
#
# POWHEG gives the INTEGRAL of the density over 15 GeV bins, and so does the
# model:  sigma_k = sum_ij C_i beta_ij I_jk,  I_jk = int_bin_k phi B_j dm.
# The prediction is linear in the 23*J numbers beta_ij. With a Gaussian
# likelihood and a Gaussian smoothness prior the posterior is therefore
# EXACTLY Gaussian: mean beta_hat, precision matrix H. Nothing is sampled
# and nothing is approximated.
#
# Classes
#   SplineBasis   the B-splines times the threshold factor; their values at
#                 any m and their integrals over any bins
#   SmoothHEFT    the model: fit (with the smoothness chosen by the
#                 evidence), predictions with uncertainties for any binning,
#                 the density and the coefficient functions at any m
#
# The parameters are stored in ONE vector beta of length 23*J, ordered
# function by function: beta[i*J + j] = beta_ij. Several matrices below are
# Kronecker products, np.kron(A, B), which builds the block matrix whose
# block (i, i') is A[i, i'] * B; with this ordering, np.kron(C, I_k) is the
# vector of all products C_i * I_jk, i.e. the derivative of the prediction
# for bin k with respect to beta.
#
# P. Bhat, N. De Filippis, K.F. Johnson, H.B. Prosper, R. Starnes
# Created: Sat Oct 31 2026
# ----------------------------------------------------------------------------
import numpy as np
from scipy.interpolate import BSpline

__all__ = ['SplineBasis', 'SmoothHEFT', 'heft_monomials']

M_H = 125.0            # Higgs boson mass in GeV; the threshold is 2 M_H


class SplineBasis:
    '''
    J cubic B-splines on [m_lo, m_hi] (GeV), with breakpoints spaced
    uniformly in u = sqrt(m - m_lo), i.e. denser just above threshold, times
    an optional threshold factor phi(m) = sqrt(1 - 4 M_H^2 / m^2), which
    vanishes at m = 2 M_H like the two-body phase space.

    Arguments
    ---------
    n_basis   : J, the number of basis functions (at least 4).
    m_lo      : lower end; also where the spacing in sqrt(m - m_lo) starts.
    m_hi      : upper end.
    threshold : multiply by phi(m) (True) or not (False).
    '''
    def __init__(self, n_basis=40, m_lo=250.0, m_hi=1500.0, threshold=True):
        self.J, self.m_lo, self.m_hi, self.threshold = n_basis, m_lo, m_hi, threshold
        k = 3                                          # cubic
        n_break = n_basis - k + 1                      # breakpoints incl. both ends
        u = np.linspace(0.0, np.sqrt(m_hi - m_lo), n_break)
        self.breaks = m_lo + u**2
        # Knot vector for SciPy's BSpline.
        #
        # A cubic B-spline is a smooth bump made of 4 cubic pieces joined at
        # 5 consecutive knots, with value, slope and curvature continuous at
        # every join.  Four pieces is the fewest that can rise from zero and
        # return to zero that smoothly, so each bump touches only 4 intervals
        # and at any m only 4 of the bumps are non-zero (a sparse, banded fit).
        #
        # To get a full set of bumps right up to the ends we "clamp" the knot
        # vector: repeat each end point k more times.
        #   [m_lo]*k  is list repetition: [250.0]*3 -> [250.0, 250.0, 250.0]
        #   np.r_[...] joins the three pieces into one array.
        # Total knots = k + n_break + k = 44 for the defaults, and SciPy's rule
        #   (number of B-splines) = (number of knots) - k - 1 = 40 = n_basis.
        # With clamping the B-splines sum to 1 everywhere on [m_lo, m_hi], and
        # only the first (last) one is non-zero at m_lo (m_hi).
        #
        # We do NOT use natural splines (zero curvature forced at the ends):
        # the coefficient functions curve strongly near threshold and in the
        # tail, so that constraint would bias them.  Smoothness comes instead
        # from the second-difference penalty on the coefficients (P-splines).
        self.knots = np.r_[[m_lo] * k, self.breaks, [m_hi] * k]
        self.k = k

    def phi(self, m):
        m = np.asarray(m, dtype=np.float64)
        if not self.threshold:
            return np.ones_like(m)
        # np.clip keeps the argument of sqrt non-negative below threshold
        return np.sqrt(np.clip(1.0 - 4.0 * M_H**2 / m**2, 0.0, None))

    def values(self, m):
        '''
        phi(m) B_j(m) for every m and j: an array [len(m), J]; zero outside
        [m_lo, m_hi].
        '''
        m = np.atleast_1d(np.asarray(m, dtype=np.float64))
        out = np.zeros((len(m), self.J))
        inside = (m >= self.m_lo) & (m <= self.m_hi)
        if inside.any():
            # BSpline.design_matrix returns a sparse matrix; .toarray() makes
            # it an ordinary array
            B = BSpline.design_matrix(m[inside], self.knots, self.k).toarray()
            out[inside] = B * self.phi(m[inside])[:, None]
        return out

    def integrals(self, edges, n_gauss=16):
        '''
        The integrals of phi B_j over the bins [edges[k], edges[k+1]]: an
        array [n_bins, J]. Each bin is cut at the spline breakpoints (where
        the B-splines change polynomial) and each piece is integrated with
        n_gauss-point Gauss-Legendre quadrature; pieces below threshold are
        zero. Near the threshold phi has a square-root behaviour, so pieces
        that start at the threshold use the substitution m = m_lo + v^2,
        which removes it.
        '''
        x, w = np.polynomial.legendre.leggauss(n_gauss)   # nodes, weights on [-1, 1]
        edges = np.asarray(edges, dtype=np.float64)
        out = np.zeros((len(edges) - 1, self.J))
        for k, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
            a, b = max(a, self.m_lo), min(b, self.m_hi)
            if b <= a:
                continue
            cuts = np.r_[a, self.breaks[(self.breaks > a) & (self.breaks < b)], b]
            for lo, hi in zip(cuts[:-1], cuts[1:]):
                if self.threshold and lo <= self.m_lo + 1e-9:
                    # substitution m = m_lo + v^2, dm = 2 v dv
                    vlo, vhi = 0.0, np.sqrt(hi - self.m_lo)
                    v = 0.5 * (vhi - vlo) * x + 0.5 * (vhi + vlo)
                    m = self.m_lo + v**2
                    out[k] += (0.5 * (vhi - vlo) * w * 2 * v) @ self.values(m)
                else:
                    m = 0.5 * (hi - lo) * x + 0.5 * (hi + lo)
                    out[k] += (0.5 * (hi - lo) * w) @ self.values(m)
        return out


class SmoothHEFT:
    '''
    The smooth 23 x 80 model: a Bayesian linear model with an exact Gaussian
    posterior.

    Arguments
    ---------
    basis      : a SplineBasis.
    n_coeffs   : number of monomials C_i (23 for HEFTNet).
    ridge      : a tiny ridge term, relative to the data curvature of each
                 function, that makes the prior proper (the curvature
                 penalty does not penalise straight lines).

    Typical use
    -----------
        model = SmoothHEFT(SplineBasis(40))
        model.fit(C, I_rows, t, var)        # C: [N, 23]; I_rows: [N, J]
        f, v = model.predict(C_new, I_new)  # bin cross sections and variances
    '''
    def __init__(self, basis, n_coeffs=23, ridge=1e-10):
        self.basis, self.n, self.J = basis, n_coeffs, basis.J
        self.P = n_coeffs * basis.J
        J = basis.J
        D = np.diff(np.eye(J), n=2, axis=0)           # second differences, [J-2, J]
        self.Q = D.T @ D                               # curvature penalty, rank J-2
        self.rank_Q = J - 2
        self.ridge = ridge
        self.lam = None                                # one strength per function

    # ------------------------------------------------------------------------
    @staticmethod
    def group_rows(I_rows):
        '''
        Many data rows share the same bin, i.e. the same row of integrals.
        Return (unique rows [K, J], index [N] of each data row's unique row).
        '''
        # np.unique on rows: the distinct rows and, for every row, which one
        uniq, inverse = np.unique(np.round(I_rows, 12), axis=0, return_inverse=True)
        return uniq, inverse.reshape(-1)

    def _data_terms(self, C, I_rows, t, w):
        # G = sum_rows w g g^T and h = sum_rows w t g, with g = kron(C, I_row).
        # Rows of the same bin share I_row, so G = sum_k kron(S_k, I_k I_k^T)
        # with S_k = sum_{rows in k} w C C^T (a 23 x 23 matrix): K terms
        # instead of one per row.
        uniq, idx = self.group_rows(I_rows)
        G = np.zeros((self.P, self.P)); h = np.zeros(self.P)
        for k, Ik in enumerate(uniq):
            s = idx == k
            Ck, wk, tk = C[s], w[s], t[s]
            S = Ck.T @ (wk[:, None] * Ck)
            r = Ck.T @ (wk * tk)
            G += np.kron(S, np.outer(Ik, Ik))
            h += np.kron(r, Ik)
        return G, h

    def _prior(self, lam):
        # block-diagonal prior precision: lam_i Q + ridge_i I for function i
        L = np.zeros((self.P, self.P))
        for i in range(self.n):
            b = slice(i * self.J, (i + 1) * self.J)
            L[b, b] = lam[i] * self.Q + self.ridge_i[i] * np.eye(self.J)
        return L

    def _solve(self):
        # posterior precision H = G + Lambda, its Cholesky factor, and the mean
        self.Lambda = self._prior(self.lam)
        self.H = self.G + self.Lambda
        self.L = np.linalg.cholesky(self.H)
        # beta_hat = H^{-1} h, by two triangular solves with L
        y = np.linalg.solve(self.L, self.h)
        self.beta = np.linalg.solve(self.L.T, y)
        self._Hinv = None

    def H_inverse(self):
        # the posterior covariance of beta
        if self._Hinv is None:
            Linv = np.linalg.inv(self.L)
            self._Hinv = Linv.T @ Linv
        return self._Hinv

    # ------------------------------------------------------------------------
    def fit(self, C, I_rows, t, var, lam=None, n_iter=200, rtol=1e-4,
            damping=0.5, verbose=False):
        '''
        Fit the model to data rows: C [N, 23] monomials, I_rows [N, J] the
        integrals of the basis over each row's bin, t [N] the cross sections,
        var [N] their variances. The smoothness strengths lam (one per
        function) are fixed if given, otherwise chosen by maximising the
        evidence with MacKay's updates (see gamma below).
        '''
        C = np.asarray(C, np.float64); I_rows = np.asarray(I_rows, np.float64)
        t = np.asarray(t, np.float64); w = 1.0 / np.asarray(var, np.float64)
        self.G, self.h = self._data_terms(C, I_rows, t, w)
        self.chi2_const = float(np.sum(w * t**2))      # for chi2(beta) below
        # the scale of each function's block of G sets the ridge
        diagG = np.diag(self.G).reshape(self.n, self.J)
        scale = diagG.mean(axis=1)
        self.ridge_i = self.ridge * scale
        if lam is not None:
            self.lam = np.asarray(lam, np.float64)
            self._solve()
            return self
        # start with a weak penalty: 1e-6 of the data curvature
        self.lam = 1e-6 * scale / np.trace(self.Q) * self.J
        self._solve()
        for it in range(n_iter):
            gam = self.gamma()
            Hinv = self.H_inverse()
            new = np.empty(self.n)
            for i in range(self.n):
                b = slice(i * self.J, (i + 1) * self.J)
                curv = float(self.beta[b] @ self.Q @ self.beta[b])
                new[i] = max(gam[i], 1e-6) / max(curv, 1e-300)
            change = np.max(np.abs(new / self.lam - 1))
            # move part of the way, in the logarithm (the plain update can
            # overshoot and oscillate)
            self.lam = np.exp(np.log(self.lam) + damping * (np.log(new) - np.log(self.lam)))
            self._solve()
            if verbose and (it % 10 == 0 or change < rtol):
                print(f'  iteration {it}: ln Z = {self.log_evidence():.2f}, '
                      f'largest change of lambda {change:.1e}')
            if change < rtol:
                break
        return self

    def fit_poisson(self, C, I_rows, t, q, extra_var=None, n_irls=12,
                    verbose=False, **kwargs):
        '''
        Fit to Poisson counts of equal-weight events: t = q * n. Weighting
        each bin by its OBSERVED count (Neyman's chi-squared) biases the fit
        low where counts are small, since downward fluctuations get larger
        weights. Instead the weights are taken from the model itself,
        var = q * max(f, q), and the fit is repeated with the new weights
        ("iteratively reweighted least squares"); at convergence this is the
        Poisson maximum-likelihood fit (plus the prior). The first fit uses
        q * (t + q). extra_var (optional, [N]) is added to every variance,
        e.g. the model-error term.
        '''
        C = np.asarray(C, np.float64); I_rows = np.asarray(I_rows, np.float64)
        t = np.asarray(t, np.float64); q = np.asarray(q, np.float64)
        ev = 0.0 if extra_var is None else np.asarray(extra_var, np.float64)
        var = q * (t + q) + ev
        lam = kwargs.pop('lam', None)
        for it in range(n_irls):
            # the smoothness is chosen by the evidence in the first pass and
            # then held fixed while the weights settle; it is re-chosen at the end
            self.fit(C, I_rows, t, var, lam=lam, **kwargs)
            lam = self.lam
            B = self.beta.reshape(self.n, self.J)
            f = np.einsum('ri,ij,rj->r', C, B, I_rows)
            # the model's Poisson variance, at least one event; where the
            # model predicts a negative cross section (it cannot be a Poisson
            # mean) the observed count is used instead
            new = np.where(f > 0, q * np.maximum(f, q), q * (t + q)) + ev
            # move half-way (geometric mean), which prevents oscillation
            new = np.sqrt(var * new)
            change = np.quantile(np.abs(new / var - 1), 0.99)
            var = new
            if verbose:
                print(f'  IRLS pass {it + 1}: 99% of the variances changed by less than {change:.1e}')
            if change < 1e-3:
                break
        self.fit(C, I_rows, t, var, **kwargs)
        self.var = var
        return self

    def gamma(self):
        '''
        For each function i, the number of curvature directions determined by
        the data: gamma_i = rank(Q) - lam_i tr(Q H^{-1}_ii). The evidence is
        maximal when lam_i = gamma_i / (beta_i^T Q beta_i) (MacKay 1992).
        '''
        Hinv = self.H_inverse()
        out = np.empty(self.n)
        for i in range(self.n):
            b = slice(i * self.J, (i + 1) * self.J)
            out[i] = self.rank_Q - self.lam[i] * np.trace(self.Q @ Hinv[b, b])
        return out

    def chi2(self):
        # chi2 at beta_hat: sum w (t - f)^2 = const - 2 beta.h + beta.G.beta
        return self.chi2_const - 2 * self.beta @ self.h + self.beta @ self.G @ self.beta

    def log_evidence(self):
        '''
        ln p(data | lam), up to a constant: -chi2/2 - beta.Lambda.beta/2 +
        ln det(Lambda)/2 - ln det(H)/2. Exact, since the model is linear.
        '''
        ld_L = sum(np.linalg.slogdet(self.Lambda[i*self.J:(i+1)*self.J, i*self.J:(i+1)*self.J])[1]
                   for i in range(self.n))
        ld_H = 2 * np.sum(np.log(np.diag(self.L)))
        return float(-0.5 * self.chi2() - 0.5 * self.beta @ self.Lambda @ self.beta
                     + 0.5 * ld_L - 0.5 * ld_H)

    # ------------------------------------------------------------------------
    def _linear(self, C, R, chunk=20000):
        # mean and variance of sum_ij C_i beta_ij R_j for each row: g = kron(C, R)
        C = np.asarray(C, np.float64); R = np.asarray(R, np.float64)
        N = len(C)
        B = self.beta.reshape(self.n, self.J)
        mean = np.einsum('ri,ij,rj->r', C, B, R)
        var = np.empty(N)
        for s in range(0, N, chunk):
            # g[r] = kron(C[r], R[r]) for all rows of the chunk at once
            g = (C[s:s+chunk, :, None] * R[s:s+chunk, None, :]).reshape(-1, self.P)
            z = np.linalg.solve(self.L, g.T)           # L^{-1} g
            var[s:s+chunk] = np.sum(z**2, axis=0)      # g^T H^{-1} g
        return mean, var

    def predict(self, C, I_rows):
        '''
        Cross section over each row's interval and its posterior variance:
        C [N, 23], I_rows [N, J] the basis integrals over the intervals.
        '''
        return self._linear(C, I_rows)

    def density(self, C, m):
        '''d sigma / dm at the masses m (one per row of C), and its variance.'''
        return self._linear(C, self.basis.values(m))

    def coefficients(self, m):
        '''
        The 23 coefficient functions a_i(m) (pb/GeV) and their posterior
        standard deviations: two arrays [len(m), 23].
        '''
        V = self.basis.values(m)                        # [M, J]
        B = self.beta.reshape(self.n, self.J)
        a = V @ B.T
        Hinv = self.H_inverse()
        sd = np.empty_like(a)
        for i in range(self.n):
            b = slice(i * self.J, (i + 1) * self.J)
            sd[:, i] = np.sqrt(np.einsum('mj,jk,mk->m', V, Hinv[b, b], V))
        return a, sd

    def coefficient_integrals(self, I_bins):
        '''The integrals of the 23 a_i over bins (rows of I_bins): [K, 23].'''
        return I_bins @ self.beta.reshape(self.n, self.J).T

    def state(self):
        return {'beta': self.beta, 'lam': self.lam, 'H': self.H,
                'breaks': self.basis.breaks, 'threshold': self.basis.threshold,
                'm_lo': self.basis.m_lo, 'm_hi': self.basis.m_hi}

    # ------------------------------------------------------------------------
    # Using a saved model
    # ------------------------------------------------------------------------
    @classmethod
    def load(cls, filename):
        '''
        Rebuild a fitted model from the file written by heft_smooth.ipynb
        (runs/SMOOTH_<tag>/smooth_model.npz). No data are needed.

            model = SmoothHEFT.load('runs/SMOOTH_J40/smooth_model.npz')
        '''
        d = np.load(filename)
        basis = SplineBasis(n_basis=len(d['breaks']) + 2, m_lo=float(d['m_lo']),
                            m_hi=float(d['m_hi']), threshold=bool(d['threshold']))
        if not np.allclose(basis.breaks, d['breaks']):
            raise ValueError('SmoothHEFT.load: the stored breakpoints do not match')
        model = cls(basis)
        model.beta, model.lam, model.H = d['beta'], d['lam'], d['H']
        model.L = np.linalg.cholesky(model.H)
        model._Hinv = None
        # the discrepancy: ln epsilon is a quadratic in u = (m - 855 GeV)/600 GeV
        model.eps_poly = d['eps_poly'] if 'eps_poly' in d else None
        return model

    def epsilon(self, m):
        '''The discrepancy coefficient epsilon(m) (a fraction), at the masses m.'''
        if getattr(self, 'eps_poly', None) is None:
            raise ValueError('this model has no discrepancy stored')
        return np.exp(np.polyval(self.eps_poly, (np.asarray(m, np.float64) - 855.0) / 600.0))

    def spectrum(self, theta, edges=255.0 + 15.0 * np.arange(81), discrepancy=True):
        '''
        The cross section (pb) in the bins [edges[k], edges[k+1]] for each HEFT
        point theta = (klambda, ct, ctt, cggh, cgghh), with its uncertainty.

        Arguments
        ---------
        theta       : array [5] or [N, 5].
        edges       : bin edges in GeV, within 250-1455 GeV (default: the 80
                      bins of 15 GeV from 255 to 1455 GeV).
        discrepancy : include the model-error term in the uncertainty.

        Returns (sigma, sd), each of shape [N, n_bins]: the posterior mean and
        the standard deviation of the model (fit uncertainty, and if asked the
        discrepancy). The Poisson noise of a POWHEG sample is NOT included.
        '''
        theta = np.atleast_2d(np.asarray(theta, np.float64))
        edges = np.asarray(edges, np.float64)
        N, K = len(theta), len(edges) - 1
        C = heft_monomials(theta)                               # [N, 23]
        I = self.basis.integrals(edges)                         # [K, J]
        # every (point, bin) pair is one row: C repeated K times, I tiled N times
        f, v = self.predict(np.repeat(C, K, axis=0), np.tile(I, (N, 1)))
        f, v = f.reshape(N, K), v.reshape(N, K)
        if discrepancy:
            A = self.coefficient_integrals(I)                   # [K, 23]
            S = np.abs(C[:, None, :] * A[None]).sum(2)          # sum_i |C_i a_ik|
            v = v + (self.epsilon(0.5 * (edges[:-1] + edges[1:]))[None] * S)**2
        return f, np.sqrt(v)

    def dsigma_dm(self, theta, m):
        '''
        The density d sigma / dm (pb/GeV) at the masses m for each HEFT point:
        arrays [N, len(m)] of the posterior mean and its standard deviation
        (fit uncertainty only).
        '''
        theta = np.atleast_2d(np.asarray(theta, np.float64))
        m = np.atleast_1d(np.asarray(m, np.float64))
        C = heft_monomials(theta)
        f, v = self.density(np.repeat(C, len(m), axis=0), np.tile(m, len(C)))
        return f.reshape(len(C), len(m)), np.sqrt(v).reshape(len(C), len(m))


def heft_monomials(theta):
    '''
    The 23 monomials C_i(theta) of HEFTNet, in the same order, for
    theta = (klambda, ct, ctt, cggh, cgghh): an array [N, 23].
    (Kept identical to heftnet.HEFTNet.forward; code/smooth_toy.py and
    heft_smooth.ipynb use HEFTNet itself.)
    '''
    t = np.atleast_2d(np.asarray(theta, np.float64))
    kl, ct, ctt, cggh, cgghh = t.T
    return np.column_stack((
        ct**4, ctt**2, ct**2*kl**2, cggh**2*kl**2, cgghh**2,
        ctt*ct**2, kl*ct**3, ct*kl*ctt, cggh*kl*ctt, ctt*cgghh,
        cggh*kl*ct**2, cgghh*ct**2, kl**2*cggh*ct, cgghh*ct*kl, cggh*cgghh*kl,
        ct**3*cggh, ct*ctt*cggh, ct*cggh**2*kl, ct*cggh*cgghh, ct**2*cggh**2,
        ctt*cggh**2, cggh**3*kl, cggh**2*cgghh))
