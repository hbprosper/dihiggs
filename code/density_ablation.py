# ----------------------------------------------------------------------------
# density_ablation.py: which ingredient matters? A factorial test on toys with
# known answers, to decide whether the density network (HEFTDensity) is worth
# pursuing. See docs/density/density_plan.pdf, addendum.
#
# Every configuration is the same network, HEFTDensity, trained the same way
# (full-batch Adam warm-up, then L-BFGS in double precision to convergence),
# and differs only in:
#   activation : ReLU or SiLU
#   input      : 'linear' x = (m - 255)/1500 (as HEFTNet) or 'sqrt' z ~ sqrt(m - 250)
#   target     : 'bins'    = 1 node per bin: bin content = 15 GeV x a(centre),
#                            what HEFTNet assumes;
#                'density' = 8 Gauss-Legendre nodes: the true bin integral
#   threshold  : phi(m) on or off
# "old" = ReLU, linear, bins, no phi: equivalent to the original HEFTNet.
#
# Two toys:
#   simple    : the toy of smooth_toy.py / density_toy.py (3 monomials).
#   realistic : the truth is the smooth spline model's own 23 a_i(m)
#               (runs/SMOOTH_J40), at HEFT points of the real training
#               spectra, with Poisson counts of 40,000 events per spectrum:
#               real threshold curvature and real cancellations.
#
# Usage (from code/):   python density_ablation.py simple       (~12 min)
#                       python density_ablation.py realistic    (~15 min per run on a CPU)
# The simple toy uses a quick recipe (Adam warm-up + L-BFGS); the realistic
# toy uses the production recipe of the notebooks (train_production), which
# it needs: with the quick recipe it does not converge from a random start.
# Writes density_ablation_<toy>.npz next to this script and prints a table.
#
# P. Bhat, N. De Filippis, K.F. Johnson, H.B. Prosper, R. Starnes
# Created: Wed Nov 4 2026
# ----------------------------------------------------------------------------
import os, sys, time
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from dihiggs.heftnet import HEFTDensity, monomials
from dihiggs.smooth import SplineBasis, SmoothHEFT, heft_monomials

TOY = sys.argv[1] if len(sys.argv) > 1 else 'simple'
assert TOY in ('simple', 'realistic'), TOY
torch.set_num_threads(int(os.environ.get('THREADS', 1)))
edges = 255.0 + 15.0 * np.arange(81)
centres = edges[:-1] + 7.5
grid = np.arange(255.0, 1455.0 + 1e-9, 0.5)
on_centre = np.isin(grid, centres)

# ----------------------------------------------------------------------------
# 1. The toy data and the truth
# ----------------------------------------------------------------------------
rng = np.random.default_rng(7)
if TOY == 'simple':
    basis = SplineBasis(30, threshold=True); phi = basis.phi
    def true_a(m):
        u = (m - 250.0) / 100.0
        return np.column_stack([phi(m) * np.exp(-u / 2.5),
                                phi(m) * 0.5 * (u - 1.0) * np.exp(-u / 1.5),
                                phi(m) * 0.6 * np.exp(-((u - 1.5) / 1.0)**2)])
    fine = np.linspace(250, 1455, 120001)
    cum = np.vstack([np.zeros(3), np.cumsum(0.5 * (true_a(fine[1:]) + true_a(fine[:-1])) * np.diff(fine)[:, None], axis=0)])
    A_true = np.diff(np.array([np.interp(edges, fine, cum[:, i]) for i in range(3)]).T, axis=0)   # [80, 3]
    S = 600
    xpar = rng.uniform(-1.5, 1.5, S)
    theta = np.column_stack([xpar, np.zeros((S, 4))])
    C = np.column_stack([np.ones(S), xpar, xpar**2 + 0.5])
    def mono(th):
        xx = th[:, 0]
        return torch.stack([torch.ones_like(xx), xx, xx**2 + 0.5], dim=1)
    n_coeffs = 3
else:
    import pandas as pd
    truth_model = SmoothHEFT.load(os.path.join(HERE, 'runs', 'SMOOTH_J40', 'smooth_model.npz'))
    true_a = lambda m: truth_model.coefficients(np.asarray(m))[0]          # [M, 23] pb/GeV
    A_true = truth_model.coefficient_integrals(truth_model.basis.integrals(edges))   # [80, 23]
    df = pd.read_csv(os.path.join(HERE, '..', 'data', 'heft_wide_gauss_traindata_13TeV.csv.gz'))
    cols = ['klambda', 'CT', 'CTT', 'CGGH', 'CGGHH']
    th_all = df.groupby('ID')[cols].first().to_numpy(np.float64)
    S = int(os.environ.get('N_SPECTRA', 1500))
    theta = th_all[rng.choice(len(th_all), S, replace=False)]
    C = heft_monomials(theta)
    mono, n_coeffs = monomials, 23

mu = C @ A_true.T                                    # [S, 80] expected cross section per bin
mu = np.maximum(mu, 1e-12)                           # (guards against tiny negative truth values)
q = mu.sum(1) / 40000.0                              # one event's weight
n = rng.poisson(mu / q[:, None]); t = n * q[:, None]
rows = np.repeat(np.arange(S), 80); bins = np.tile(np.arange(80), S)
X = np.zeros((len(rows), 6)); X[:, 0] = (bins + 0.5) / 100; X[:, 1:] = theta[rows]
X = torch.tensor(X, dtype=torch.float64)
T = torch.tensor(t.ravel(), dtype=torch.float64)
Qr = torch.tensor(q[rows], dtype=torch.float64)

truth_grid = true_a(grid)                            # [M, n_coeffs]
# evaluation points: for the simple toy the 3 functions; for the realistic toy
# the density d sigma/dm itself at 300 HEFT points (the a_i of the realistic
# model include poorly determined combinations, so the density is the fair test)
if TOY == 'realistic':
    th_eval = theta[:300]
    C_eval = heft_monomials(th_eval)
    dens_true = C_eval @ truth_grid.T                # [300, M]

# ----------------------------------------------------------------------------
# 2. Training (identical for every configuration)
# ----------------------------------------------------------------------------
def chi2_bins(model):
    with torch.no_grad():
        f = model(X)
        r2 = ((T - f)**2 / (Qr * (torch.clamp(f, min=0.0) + Qr))).numpy().reshape(S, 80)
    return r2.mean(), r2[:, :4].mean()

def train(model, adam_steps=3000, lbfgs_blocks=40, lr=3e-3):
    model.double()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    var = Qr * (T + Qr)
    for step in range(adam_steps):
        opt.zero_grad(); loss = torch.mean((T - model(X))**2 / var); loss.backward(); opt.step()
    opt = torch.optim.LBFGS(model.parameters(), lr=1, max_iter=50, history_size=50, line_search_fn='strong_wolfe')
    previous = np.inf
    for block in range(lbfgs_blocks):
        with torch.no_grad():
            var = Qr * (torch.clamp(model(X), min=0.0) + Qr)
        def closure():
            opt.zero_grad(); loss = torch.mean((T - model(X))**2 / var); loss.backward(); return loss
        opt.step(closure)
        c = chi2_bins(model)[0]
        if abs(previous - c) < 1e-5:
            break
        previous = c

def train_production(model, l1_steps=200_000, batch=2048, ft_steps=10_000):
    '''
    The recipe of heft_train.ipynb + heft_finetune.ipynb, as used for the
    original networks: (1) L1 loss on the bin cross sections, Adam with
    mini-batches of 2,048 rows, 200,000 steps, learning rate 2e-3 halved
    every 50,000 steps; (2) fine-tune: Huber loss (delta = 2) of the pulls with
    the Poisson variance from the model, full-batch Adam, 10,000 steps,
    learning rate 1e-4 halved every 2,500 steps.
    '''
    model.double()
    g = torch.Generator().manual_seed(12345)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=l1_steps // 4, gamma=0.5)
    N = len(T)
    for step in range(l1_steps):
        idx = torch.randint(0, N, (batch,), generator=g)
        opt.zero_grad()
        loss = torch.mean(torch.abs(model(X[idx]) - T[idx]))
        loss.backward(); opt.step(); sched.step()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=ft_steps // 4, gamma=0.5)
    for step in range(ft_steps):
        opt.zero_grad()
        f = model(X)
        r = (f - T) / torch.sqrt(Qr * (torch.clamp(f.detach(), min=0.0) + Qr))
        a = torch.abs(r)
        loss = torch.mean(torch.where(a <= 2.0, 0.5 * r**2, 2.0 * (a - 1.0)))
        loss.backward(); opt.step(); sched.step()

RECIPE = os.environ.get('RECIPE', 'quick' if TOY == 'simple' else 'production')
if RECIPE == 'production':
    train = train_production

# ----------------------------------------------------------------------------
# 3. The configurations
# ----------------------------------------------------------------------------
configs = []
for act in ('nn.ReLU', 'nn.SiLU'):
    for imap in ('linear', 'sqrt'):
        for nodes in (1, 8):
            configs.append((act, imap, nodes, True))
configs.append(('nn.ReLU', 'linear', 1, False))      # the old HEFTNet
if TOY == 'realistic' and os.environ.get('SHORT', '1') == '1':
    # the decision-relevant subset: old; best bin-value; density; and the
    # two single changes from old (activation; input)
    configs = [('nn.ReLU', 'linear', 1, False), ('nn.SiLU', 'linear', 1, False), ('nn.ReLU', 'sqrt', 1, False),
               ('nn.SiLU', 'sqrt', 1, True), ('nn.SiLU', 'sqrt', 8, True)]
    only = os.environ.get('CONFIGS')      # e.g. CONFIGS=0,1 to run a subset (for parallel runs)
    if only:
        configs = [configs[int(i)] for i in only.split(',')]
SEEDS = (1, 2, 3) if TOY == 'simple' else (1, 2)

def label(c):
    act, imap, nodes, thr = c
    return f"{act.split('.')[-1]:4s} {imap:6s} {'density' if nodes > 1 else 'bins':7s} {'phi' if thr else 'no-phi'}"

out = {}
t0 = time.time()
for c in configs:
    act, imap, nodes, thr = c
    curves, chis = [], []
    for seed in SEEDS:
        torch.manual_seed(seed)
        model = HEFTDensity(activation=act, input_map=imap, n_nodes=nodes, threshold=thr,
                            n_coeffs=n_coeffs, monomial_fn=mono)
        train(model)
        chis.append(chi2_bins(model))
        a = model.coefficients(grid)                                  # [M, n_coeffs]
        curves.append(a if TOY == 'simple' else C_eval @ a.T)         # simple: a_i; realistic: densities [300, M]
        print(f'{label(c)}  seed {seed}: chi2/bin {chis[-1][0]:.3f} (first 4 bins {chis[-1][1]:.2f})  ({time.time() - t0:.0f} s)', flush=True)
    out[label(c)] = (np.array(curves), np.array(chis))

# ----------------------------------------------------------------------------
# 4. Summary
# ----------------------------------------------------------------------------
low = grid <= 600.0; first = grid < 285.0
print(f'\n=== {TOY} toy ===')
if TOY == 'simple':
    scale = np.abs(truth_grid).max(0)
    print('errors relative to max|a_i|, in %; spread = s.d. over seeds; "ratio" = spread between / at bin centres (255-600 GeV)')
    print(f'{"configuration":30s} {"chi2":>6s} {"chi2 1-4":>8s} {"err grid":>8s} {"err thr":>8s} {"spread ctr":>10s} {"between":>8s} {"ratio":>6s} {"max betw":>8s} {"bias 255-285":>14s}')
    for k, (cv, ch) in out.items():
        err = (cv - truth_grid[None]) / scale                       # [seeds, M, 3]
        e_grid = 100 * np.sqrt((err**2).mean(axis=(1, 2))).mean()
        e_thr = 100 * np.sqrt((err[:, grid <= 345]**2).mean(axis=(1, 2))).mean()
        sd = cv.std(0) / scale                                      # [M, 3]
        s_c, s_b = 100 * np.median(sd[low & on_centre]), 100 * np.median(sd[low & ~on_centre])
        b = 100 * err[:, first].mean(axis=(1, 2))
        print(f'{k:30s} {ch[:, 0].mean():6.3f} {ch[:, 1].mean():8.2f} {e_grid:8.3f} {e_thr:8.3f} {s_c:10.3f} {s_b:8.3f} {s_b / s_c:6.2f} {100 * sd[low & ~on_centre].max():8.3f} {b.mean():+7.3f}+-{b.std():.3f}')
else:
    print('density d sigma/dm at 300 HEFT points; relative errors in %; medians over the points')
    print(f'{"configuration":30s} {"chi2":>6s} {"chi2 1-4":>8s} {"err 255-600":>11s} {"err thr":>8s} {"spread ctr":>10s} {"between":>8s} {"ratio":>6s} {"bias 255-285":>14s}')
    for k, (cv, ch) in out.items():
        rel = (cv - dens_true[None]) / np.abs(dens_true[None]).max(axis=2, keepdims=True)   # relative to each spectrum's peak density
        e_low = 100 * np.median(np.sqrt((rel[:, :, low]**2).mean(2)))
        e_thr = 100 * np.median(np.sqrt((rel[:, :, grid <= 345]**2).mean(2)))
        sd = cv.std(0) / np.abs(dens_true).max(axis=1, keepdims=True)                         # [300, M]
        s_c, s_b = 100 * np.median(sd[:, low & on_centre]), 100 * np.median(sd[:, low & ~on_centre])
        # bias near threshold relative to the local density: mean over points and seeds
        rloc = (cv[:, :, first] - dens_true[None][:, :, first]) / np.abs(dens_true[None][:, :, first])
        b = 100 * np.median(rloc, axis=(1, 2))
        print(f'{k:30s} {ch[:, 0].mean():6.3f} {ch[:, 1].mean():8.2f} {e_low:11.3f} {e_thr:8.3f} {s_c:10.3f} {s_b:8.3f} {s_b / s_c:6.2f} {b.mean():+7.3f}+-{b.std():.3f}')
np.savez(os.path.join(HERE, f"density_ablation_{TOY}{os.environ.get('TAG', '')}.npz"), grid=grid,
         **{k.replace(' ', '_'): v[0] for k, v in out.items()})
print(f'total {time.time() - t0:.0f} s')
