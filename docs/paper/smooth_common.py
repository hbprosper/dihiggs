# Shared by the paper figures: the smooth model's predictions for the 1,000 test spectra,
# as bin integrals and as a continuous density, with the fit and total (fit + discrepancy) bands.
import numpy as np, pickle, os
OUT = 'smooth_test.npz'
if not os.path.exists(OUT):
    m = pickle.load(open('/home/claude/sm/model_primary.pkl', 'rb'))
    d = np.load('/home/claude/sm/inputs.npz'); P = np.load('/home/claude/sm/primary.npz')
    T, qt = d['T'], d['q_t']; nS = len(T)
    C = d['C_t'].reshape(nS, 80, 23)[:, 0, :]                 # C_i(theta) does not depend on m
    edges = 255.0 + 15.0 * np.arange(81); I = m.basis.integrals(edges)
    f, v = m.predict(np.repeat(C, 80, axis=0), np.tile(I, (nS, 1))); f, v = f.reshape(nS, 80), v.reshape(nS, 80)
    Ab = m.coefficient_integrals(I)                           # [80, 23]
    S = np.abs(C[:, None, :] * Ab[None]).sum(2)               # [nS, 80]
    eps = P['eps']
    # the density on a grid, with its fit variance: K(m) = V(m) Hinv_ij V(m)^T for each pair (i, j)
    mg = np.linspace(252.5, 1455, 482)
    V = m.basis.values(mg); B = m.beta.reshape(23, m.J); Hinv = m.H_inverse()
    a = V @ B.T                                               # [M, 23] in pb/GeV
    Hb = Hinv.reshape(23, m.J, 23, m.J)
    K = np.einsum('mj,ijkl,ml->mik', V, Hb, V)                # [M, 23, 23]
    dens = C @ a.T                                            # [nS, M]
    dvar = np.einsum('si,mik,sk->sm', C, K, C)
    epsg = np.interp(mg, edges[:-1] + 7.5, eps)
    dS = np.abs(C[:, None, :] * a[None]).sum(2)               # sum_i |C_i a_i(m)|
    np.savez(OUT, T=T, qt=qt, C=C, f=f, v=v, S=S, eps=eps, mg=mg, dens=dens, dvar=dvar, dS=dS, epsg=epsg,
             th=d['th'], id=d['id_t'])
D = dict(np.load(OUT))
T, qt, f, v, S, eps = D['T'], D['qt'], D['f'], D['v'], D['S'], D['eps']
noise = qt[:, None] * np.maximum(f, qt[:, None])
disc2 = (eps[None] * S)**2
n = T / qt[:, None]; mu = f / qt[:, None]; pos = mu > 0
deviance = np.where(pos, 2 * (mu - n + np.where(n > 0, n * np.log(np.where(n > 0, n, 1) / np.where(pos, mu, 1)), 0)), np.nan)
devN = np.nanmean(deviance, 1)
z = np.where((T > 0) | (f > 0), (T - f) / np.sqrt(noise + v + disc2), np.nan)
