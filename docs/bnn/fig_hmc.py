import numpy as np, matplotlib as mp, matplotlib.pyplot as plt
plt.rcParams.update({"text.usetex": False, "font.family": "serif", "font.serif": ["TeX Gyre Pagella", "Palatino", "DejaVu Serif"], "mathtext.fontset": "cm",
    "font.size": 11})
rng = np.random.default_rng(7)
rho = 0.98
S = np.array([[1, rho], [rho, 1]]); P = np.linalg.inv(S)
U = lambda q: 0.5 * q @ P @ q
gU = lambda q: P @ q
# random-walk Metropolis, 200 steps
q = np.array([-2.0, -2.0]); rw = [q]
for _ in range(200):
    p = q + 0.15 * rng.standard_normal(2)
    if np.log(rng.random()) < U(q) - U(p): q = p
    rw.append(q)
rw = np.array(rw)
# HMC: 5 trajectories, eps 0.12, L 25
q = np.array([-2.0, -2.0]); trajs = []; Hs = None
eps, L = 0.12, 45
for t in range(4):
    p = rng.standard_normal(2); q0, p0 = q.copy(), p.copy()
    path = [q.copy()]; HUK = [(U(q), 0.5*p@p)]
    p = p - 0.5*eps*gU(q)
    for i in range(L):
        q = q + eps*p
        g = gU(q)
        p = p - (eps if i < L-1 else 0.5*eps)*g
        path.append(q.copy())
        ph = p + 0.5*eps*g if i < L-1 else p
        HUK.append((U(q), 0.5*ph@ph))
    dH = (U(q)+0.5*p@p) - (U(q0)+0.5*p0@p0)
    if np.log(rng.random()) >= -dH: q = q0
    trajs.append(np.array(path))
    if t == 0: Hs = np.array(HUK)
fig, ax = plt.subplots(1, 2, figsize=(9.6, 4.0))
g = np.linspace(-3, 3, 200); X, Y = np.meshgrid(g, g)
Z = 0.5*(P[0,0]*X**2 + 2*P[0,1]*X*Y + P[1,1]*Y**2)
for a in ax[:1]:
    a.contour(X, Y, Z, levels=[0.5, 2, 4.5], colors='0.6', linewidths=0.8)
ax[0].plot(rw[:,0], rw[:,1], '-', color='#C0392B', lw=0.8, label='random-walk Metropolis (200 steps)')
for i, tr in enumerate(trajs):
    ax[0].plot(tr[:,0], tr[:,1], '.-', color='#1B6CA8', ms=3, lw=0.8,
               label='HMC (4 trajectories, 45 leapfrog steps each)' if i == 0 else None)
    ax[0].plot(tr[-1,0], tr[-1,1], 'o', color='#1B6CA8', ms=6, mfc='white')
ax[0].plot(-2.0, -2.0, 'k*', ms=10)
ax[0].set_xlim(-3, 3); ax[0].set_ylim(-3, 3); ax[0].set_aspect('equal')
ax[0].set_xlabel(r'$\omega_1$'); ax[0].set_ylabel(r'$\omega_2$')
ax[0].legend(fontsize=8.5, loc="upper left", frameon=False)
ax[0].set_title(r'(a) exploring a narrow ridge', fontsize=11)
s = np.arange(len(Hs))*eps
ax[1].plot(s, Hs[:,0], color='#2E933C', label=r'potential $U(\omega)$')
ax[1].plot(s, Hs[:,1], color='#7B4EA3', label=r'kinetic $K(p)$')
ax[1].plot(s, Hs.sum(1), 'k', lw=1.6, label=r'total $H = U + K$')
ax[1].set_xlabel(r'fictitious time $s = \ell\,\varepsilon$'); ax[1].set_ylabel('energy')
ax[1].set_ylim(0, 1.25*Hs.sum(1).max()); ax[1].legend(fontsize=9, frameon=False, loc="upper center", ncol=3)
ax[1].set_title(r'(b) energy along the first trajectory', fontsize=11)
fig.tight_layout(); fig.savefig('fig_hmc.pdf')
