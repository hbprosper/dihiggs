# ----------------------------------------------------------------------------
# density_toy.py: a test, with known answers, of the density network
#                 HEFTDensity (dihiggs/heftnet.py); see docs/density/density_plan.pdf.
#
# The toy is that of smooth_toy.py: three known smooth coefficient functions
# a_i(m) (pb/GeV, with the threshold factor), three monomials C_i(x) of one
# parameter x, and 600 spectra of about 40,000 equal-weight events in the 80
# bins of 15 GeV from 255 to 1455 GeV, with Poisson counts.
#
# Fitted to the same data:
#   density  HEFTDensity, a_i(m) a density, compared through its bin integrals
#            (Gauss-Legendre, 8 nodes per bin), threshold factor on, input
#            z ~ sqrt(m - 250 GeV), for three smooth activations (SiLU, tanh,
#            sine), three seeds each;
#   bins     the same network with the bin integral replaced by 15 GeV times
#            the value at the bin centre (a 1-node rule): this is what HEFTNet
#            assumes when it is trained on bin contents;
#   spline   the smooth spline model (dihiggs/smooth.py), as the reference.
#
# The question: how well is the TRUE density recovered, at the bin centres and
# between them? Printed: the r.m.s. error of each model's a_i(m) relative to
# the size of a_i, at the centres and on a 0.5 GeV grid, and in the threshold
# region; the spread of the seeds between and at the centres; PASS/FAIL.
#
# Run from code/:   python density_toy.py          (about 15 minutes on a laptop CPU)
#
# P. Bhat, N. De Filippis, K.F. Johnson, H.B. Prosper, R. Starnes
# Created: Wed Nov 4 2026
# ----------------------------------------------------------------------------
import os, sys, time
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from dihiggs.heftnet import HEFTDensity
from dihiggs.smooth import SplineBasis, SmoothHEFT

torch.set_num_threads(max(1, os.cpu_count() or 1))
ADAM_STEPS   = int(os.environ.get('TOY_ADAM_STEPS', 3000))   # warm-up steps per network
LBFGS_BLOCKS = int(os.environ.get('TOY_LBFGS_BLOCKS', 30))  # at most this many L-BFGS blocks of 50 iterations
SEEDS = (1, 2, 3)

# ----------------------------------------------------------------------------
# 1. The toy data (identical to smooth_toy.py)
# ----------------------------------------------------------------------------
rng = np.random.default_rng(7)
edges = 255.0 + 15.0 * np.arange(81)
centres = edges[:-1] + 7.5
basis = SplineBasis(30, threshold=True)
phi = basis.phi

def true_a(m):
    # three smooth functions (pb/GeV); the second changes sign
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
C = np.column_stack([np.ones(S), x, x**2 + 0.5])            # keeps the density positive
mu = C @ A_bins.T                                            # [S, 80] expected cross sections
q = mu.sum(1) / 40000.0                                      # one event's weight, ~40,000 events
n = rng.poisson(mu / q[:, None])
t = n * q[:, None]
rows = np.repeat(np.arange(S), 80); bins = np.tile(np.arange(80), S)

def toy_monomials(theta):
    # theta = the single toy parameter x in column 0 (the other columns are unused)
    xx = theta[:, 0]
    return torch.stack([torch.ones_like(xx), xx, xx**2 + 0.5], dim=1)

# training rows in the HEFTNet format: (mhh = (k + 0.5)/100, x, 0, 0, 0, 0)
X = np.zeros((len(rows), 6)); X[:, 0] = (bins + 0.5) / 100; X[:, 1] = x[rows]
X = torch.tensor(X, dtype=torch.float32)
T = torch.tensor(t.ravel(), dtype=torch.float32)
Qr = torch.tensor(q[rows], dtype=torch.float32)

# ----------------------------------------------------------------------------
# 2. Training: chi-squared with the Poisson variance; Adam, then L-BFGS
# ----------------------------------------------------------------------------
# Everything in double precision: L-BFGS's line search needs accurate values
# and gradients, and stalls in single precision.
X, T, Qr = X.double(), T.double(), Qr.double()

def chi2_of(model, var=None):
    with torch.no_grad():
        f = model(X)
        v = Qr * (torch.clamp(f, min=0.0) + Qr) if var is None else var
        return float(torch.mean((T - f)**2 / v))

def train(model, adam_steps=ADAM_STEPS, lbfgs_blocks=LBFGS_BLOCKS, lr=3e-3):
    '''
    Minimise the mean of (t - f)^2 / var over all rows (full batch).
    1. Adam, adam_steps steps, with var = q (t + q) from the data: robust from
       a random start, but slow to converge.
    2. L-BFGS, a quasi-Newton method that builds up curvature information
       from recent gradients, in blocks: at the start of each block var is
       recomputed from the model's own prediction, q (max(f, 0) + q), the
       Poisson variance (iteratively reweighted least squares, which avoids
       the downward bias of weighting by the observed counts), and held fixed
       during the block. Stop when chi2 changes by less than 1e-5.
    '''
    model.double()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    var = Qr * (T + Qr)
    for step in range(adam_steps):
        opt.zero_grad()
        loss = torch.mean((T - model(X))**2 / var)
        loss.backward(); opt.step()
    opt = torch.optim.LBFGS(model.parameters(), lr=1, max_iter=50, history_size=50,
                            line_search_fn='strong_wolfe')
    previous = np.inf
    for block in range(lbfgs_blocks):
        with torch.no_grad():
            var = Qr * (torch.clamp(model(X), min=0.0) + Qr)
        def closure():
            # L-BFGS calls this several times per step (line search)
            opt.zero_grad()
            loss = torch.mean((T - model(X))**2 / var)
            loss.backward()
            return loss
        opt.step(closure)
        c = chi2_of(model)
        if abs(previous - c) < 1e-5:
            break
        previous = c
    return chi2_of(model)

grid = np.arange(255.0, 1455.0 + 1e-9, 0.5)
truth_grid, truth_cen = true_a(grid), true_a(centres)
scale = np.abs(truth_grid).max(0)                           # size of each a_i
thr = grid <= 345.0                                         # threshold region, first 6 bins

def errors(a_grid, a_cen):
    '''r.m.s. of (a - truth)/max|truth| per function, on the grid, at the centres, near threshold.'''
    eg = (a_grid - truth_grid) / scale; ec = (a_cen - truth_cen) / scale
    return np.sqrt((eg**2).mean(0)), np.sqrt((ec**2).mean(0)), np.sqrt((eg[thr]**2).mean(0))

results = {}
curves = {}
t0 = time.time()
for label, make in [
        ('density SiLU', lambda: HEFTDensity(activation='nn.SiLU', n_coeffs=3, monomial_fn=toy_monomials)),
        ('density tanh', lambda: HEFTDensity(activation='nn.Tanh', n_coeffs=3, monomial_fn=toy_monomials)),
        ('density sine', lambda: HEFTDensity(activation='Sin',     n_coeffs=3, monomial_fn=toy_monomials)),
        # the bin-value model: the same network, but the integral over a bin is
        # replaced by 15 GeV x the value at the centre (the 1-node midpoint
        # rule), which is what HEFTNet assumes when trained on bin contents
        ('bins (1 node)', lambda: HEFTDensity(activation='nn.SiLU', n_nodes=1, n_coeffs=3, monomial_fn=toy_monomials))]:
    for seed in SEEDS:
        torch.manual_seed(seed)
        model = make()
        chi2 = train(model)
        a_grid, a_cen = model.coefficients(grid), model.coefficients(centres)
        results.setdefault(label, []).append((chi2,) + errors(a_grid, a_cen))
        curves.setdefault(label, []).append(a_grid)
        print(f'{label:13s} seed {seed}: chi2/bin {chi2:.3f}   ({time.time() - t0:.0f} s)', flush=True)

# the spline model, as the reference
I = basis.integrals(edges)
sm = SmoothHEFT(basis, n_coeffs=3).fit_poisson(C[rows], I[bins], t.ravel(), q[rows])
a_s_grid, a_s_cen = sm.coefficients(grid)[0], sm.coefficients(centres)[0]
results['spline'] = [(float('nan'),) + errors(a_s_grid, a_s_cen)]
curves['spline'] = [a_s_grid]

# ----------------------------------------------------------------------------
# 3. Summary
# ----------------------------------------------------------------------------
print('\nr.m.s. error of a_i(m) / max|a_i|, mean over the 3 functions and the seeds (in %):')
print(f'{"model":14s} {"chi2/bin":>9s} {"grid":>8s} {"centres":>8s} {"255-345":>8s}')
summary = {}
for label, r in results.items():
    chi2 = np.nanmean([v[0] for v in r]) if label != 'spline' else float('nan')
    g, c_, h = [100 * np.mean([v[j] for v in r]) for j in (1, 2, 3)]
    summary[label] = (g, c_, h)
    print(f'{label:14s} {chi2:9.3f} {g:8.3f} {c_:8.3f} {h:8.3f}')

# spread of the seeds: between bin centres against at the centres (relative to max|a_i|)
print('\nspread of the 3 seeds (s.d. / max|a_i|, median over m and functions, in %), 255-600 GeV:')
low = grid <= 600.0
on_centre = np.isin(grid, centres)
for label in ['density SiLU', 'density tanh', 'density sine', 'bins (1 node)']:
    sd = np.std(np.array(curves[label]), axis=0) / scale     # [M, 3]
    mid = low & ~on_centre
    print(f'{label:14s} at centres {100 * np.median(sd[low & on_centre]):.3f}   between {100 * np.median(sd[mid]):.3f}   '
          f'max between {100 * sd[mid].max():.3f}')

# bias near threshold: in the first two bins the 1-node rule misreads the bin
# content as 15 GeV x the density at the centre; a systematic error shows up
# as a mean error over the seeds that is large compared with their spread
print('\nerror of sum_i a_i(m) / sum_i max|a_i| in the first two bins (in %): mean over seeds (bias) +- s.d. of seeds')
first = grid < 285.0
bias = {}
for label in ['density SiLU', 'density sine', 'bins (1 node)', 'spline']:
    cv = np.array(curves[label])                              # [seeds, M, 3]
    err = 100 * ((cv - truth_grid[None]) / scale[None]).mean(2)   # [seeds, M], mean over the 3 functions
    e1 = err[:, first].mean(1)                                # mean error over 255-285 GeV, per seed
    sd = e1.std() if len(e1) > 1 else float('nan')
    print(f'{label:14s} {e1.mean():+7.3f} +- {sd:.3f}   (at 255.5 GeV: {err[:, 1].mean():+.3f})')
    bias[label] = (e1.mean(), sd)

ok_fit = all(abs(np.mean([v[0] for v in results[l]]) - 1) < 0.1 for l in results if l.startswith('density'))
ok_between = summary['density SiLU'][0] < 0.5 * summary['bins (1 node)'][0]
ok_thr = summary['density SiLU'][2] < 0.5 * summary['bins (1 node)'][2]
print(f'\ndensity networks fit the data (chi2/bin within 0.1 of 1)             {"PASS" if ok_fit else "FAIL"}')
print(f'density (SiLU) recovers the true a_i(m) at least 2x better than bins  {"PASS" if ok_between else "FAIL"}')
print(f'... and near threshold (255-345 GeV)                                  {"PASS" if ok_thr else "FAIL"}')
# added after the first run, which showed that the 2x criteria above (from the
# plan) are not met: in this toy the difference is a BIAS near threshold
ok_bias = (abs(bias['bins (1 node)'][0]) > 3 * bias['bins (1 node)'][1]
           and abs(bias['density SiLU'][0]) < 3 * max(bias['density SiLU'][1], 1e-3))
print(f'bins biased in 255-285 GeV (> 3 s.d. of the seeds), density not         {"PASS" if ok_bias else "FAIL"}')

# ----------------------------------------------------------------------------
# 4. Figure: truth and fits near threshold
# ----------------------------------------------------------------------------
try:
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = os.path.join(HERE, '..', 'docs', 'density', 'fig_density_toy.pdf')
    fig, ax = plt.subplots(2, 3, figsize=(12, 6), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    sel = grid <= 450
    styles = {'density SiLU': ('#1B4F8A', '-'), 'density sine': ('#2E933C', '-'), 'bins (1 node)': ('#C0392B', '-'), 'spline': ('k', ':')}
    for i in range(3):
        ax[0, i].plot(grid[sel], truth_grid[sel, i], color='gray', lw=4, alpha=0.4, label='truth')
        for label, (c, ls) in styles.items():
            for j, cv in enumerate(curves[label]):
                ax[0, i].plot(grid[sel], cv[sel, i], color=c, ls=ls, lw=1, label=label if j == 0 else None)
                ax[1, i].plot(grid[sel], 100 * (cv[sel, i] - truth_grid[sel, i]) / scale[i], color=c, ls=ls, lw=1)
        for e in edges[edges <= 450]:
            ax[1, i].axvline(e, color='gray', lw=0.3)
        ax[0, i].set_title(f'$a_{i+1}(m)$', fontsize=11)
        ax[1, i].set_xlabel(r'$m$ [GeV]'); ax[1, i].set_ylim(-3, 3)
    ax[0, 0].set_ylabel('pb/GeV'); ax[1, 0].set_ylabel(r'error / max$|a_i|$ [\%]')
    ax[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(out); print('wrote', os.path.relpath(out, HERE))
except Exception as exc:
    print('figure not made:', exc)
