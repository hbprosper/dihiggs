# ----------------------------------------------------------------------------
# quadrature.py: Gauss-Legendre quadrature in PyTorch, for integrating the
#                density model of dihiggs/heftnet.py (HEFTDensity) over bins.
#
# The plan is docs/density/density_plan.pdf (section 3). In brief:
#
#   gauss_legendre(n)      the n nodes x_g and weights w_g on [-1, 1] such that
#                          sum_g w_g f(x_g) = integral of f over [-1, 1]
#                          exactly for every polynomial f of degree <= 2n - 1.
#   bin_nodes(edges, n)    the same rule mapped onto each bin [e_k, e_k+1]:
#                          nodes m [K*n] and weights w [K*n], so that
#                          (w * f(m)).reshape(K, n).sum(1) are the K bin integrals.
#
# Golub-Welsch method (Math. Comp. 23 (1969) 221): the normalised Legendre
# polynomials obey x p_k = b_{k+1} p_{k+1} + b_k p_{k-1}, b_k = k/sqrt(4k^2-1).
# Written as a matrix equation, x p = J p + b_n p_n e_n, with J the symmetric
# tridiagonal matrix that has b_1 ... b_{n-1} next to its (zero) diagonal. At a
# zero of p_n the last term vanishes, so the nodes are the eigenvalues of J;
# the weights are 2 * (first component of each normalised eigenvector)^2.
#
# The nodes and weights are computed once, in double precision, and then kept
# as constant tensors (on the GPU if the model is there).
#
# P. Bhat, N. De Filippis, K.F. Johnson, H.B. Prosper, R. Starnes
# Created: Wed Nov 4 2026
# ----------------------------------------------------------------------------
import numpy as np
import torch

__all__ = ['gauss_legendre', 'bin_nodes']


def gauss_legendre(n, dtype=torch.float64):
    '''
    The n-point Gauss-Legendre rule on [-1, 1].

    Returns (x, w), two tensors of length n: the nodes, in increasing order,
    and the weights (which are positive and sum to 2, the length of [-1, 1]).
    '''
    if n < 1:
        raise ValueError('gauss_legendre: n must be at least 1')
    if n == 1:
        # one node at the centre with weight 2 (the midpoint rule)
        return torch.zeros(1, dtype=dtype), torch.full((1,), 2.0, dtype=dtype)
    k = torch.arange(1, n, dtype=dtype)              # 1, 2, ..., n-1
    b = k / torch.sqrt(4 * k**2 - 1)                 # b_1 ... b_{n-1}
    # torch.diag(b, 1) puts b just above the diagonal, torch.diag(b, -1) just below
    J = torch.diag(b, 1) + torch.diag(b, -1)
    # eigh: eigenvalues (ascending) and eigenvectors (columns of V) of a
    # symmetric matrix; the eigenvectors are normalised to unit length
    x, V = torch.linalg.eigh(J)
    w = 2 * V[0]**2                                  # V[0] = first component of every eigenvector
    return x, w


def bin_nodes(edges, n=8, dtype=torch.float64):
    '''
    The n-point Gauss-Legendre rule in each bin [edges[k], edges[k+1]].

    Returns (m, w), tensors of length K*n (K = number of bins), ordered bin
    by bin: m[k*n:(k+1)*n] are the nodes of bin k and w[k*n:(k+1)*n] their
    weights, in the units of the edges (GeV). For any function f,
        (w * f(m)).reshape(K, n).sum(dim=1)
    gives its integrals over the K bins.
    '''
    x, wx = gauss_legendre(n, dtype)
    e = torch.as_tensor(np.asarray(edges, dtype=np.float64), dtype=dtype)
    c = 0.5 * (e[1:] + e[:-1])                       # bin centres
    h = 0.5 * (e[1:] - e[:-1])                       # half-widths
    # [K, 1] * [1, n] broadcasts to [K, n]: node g of bin k is c_k + h_k x_g,
    # and since dm = h dx its weight is h_k w_g; reshape(-1) flattens bin by bin
    m = (c[:, None] + h[:, None] * x[None, :]).reshape(-1)
    w = (h[:, None] * wx[None, :]).reshape(-1)
    return m, w


# ----------------------------------------------------------------------------
# Self-test:  python -m dihiggs.quadrature
# ----------------------------------------------------------------------------
if __name__ == '__main__':
    ok = True
    for n in (1, 2, 4, 8, 16):
        x, w = gauss_legendre(n)
        xn, wn = np.polynomial.legendre.leggauss(n)
        d = max(np.abs(x.numpy() - xn).max(), np.abs(w.numpy() - wn).max())
        # exact for x^p, p = 0 ... 2n-1: integral over [-1, 1] is 2/(p+1) for even p, 0 for odd p
        e = max(abs(float((w * x**p).sum()) - (2.0 / (p + 1) if p % 2 == 0 else 0.0)) for p in range(2 * n))
        good = d < 1e-13 and e < 1e-13
        ok &= good
        print(f'n = {n:2d}: |difference from numpy| {d:.1e}; worst error on x^0..x^{2*n-1} {e:.1e}  {"PASS" if good else "FAIL"}')
    # bin integrals of a polynomial of degree 15 over the 80 bins, with n = 8
    edges = 255.0 + 15.0 * np.arange(81)
    m, w = bin_nodes(edges, 8)
    u = (m - 855.0) / 600.0
    F = lambda v: v**16 / 16 + v**3                  # antiderivative (in u) of v^15 + 3 v^2
    approx = (w * (u**15 + 3 * u**2)).reshape(80, 8).sum(1).numpy()
    ue = (edges - 855.0) / 600.0
    exact = 600.0 * np.diff(F(ue))
    rel = np.max(np.abs(approx - exact) / np.maximum(np.abs(exact), 1e-300))
    print(f'bin integrals of a degree-15 polynomial, 8 nodes: worst relative error {rel:.1e}  {"PASS" if rel < 1e-10 else "FAIL"}')
    ok &= rel < 1e-10
    print('ALL PASS' if ok else 'SOME TESTS FAILED')
