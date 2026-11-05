# ------------------------------------------------------------------------------
# merged_test.py: the merged-bin test of the two published models (paper,
# appendices A and B).
#
# The training-file spectra are merged into 30 GeV bins (pairs of 15 GeV bins;
# the counts of a pair add, so the merged data are still Poisson). Each model
# is fitted to the merged data only and then predicts the 15 GeV bins of the
# 1,000 test spectra. A model of the bin contents (the free fit) can only give
# each half of a pair the same value; a model of the density d sigma/dm predicts
# how each 30 GeV bin splits into its two halves.
#
#   smooth model  : SmoothHEFT with the 30 GeV bin integrals (about 1 minute)
#   free fit      : 23 numbers per 30 GeV bin, each 15 GeV half given half
#   density model : HEFTDensity with 30 GeV edges, trained with the recipe of
#                   the published network: L1, mini-batch Adam, 200,000
#                   iterations; then the Huber loss (delta = 2) of the pulls
#                   with Poisson variances from the model, plus the smoothness
#                   penalty (lambda = 1e-6), full-batch Adam, 5,000 iterations.
#                   About 20 minutes on a laptop CPU, a few on a GPU.
#
# Run from docs/paper:   python merged_test.py
# Writes merged_test.npz (the 15 GeV predictions of the three models) and the
# density network's parameters in code/runs/DENS_merged30/; prints the table.
# A second run reuses the trained network (delete the folder to retrain).
# ------------------------------------------------------------------------------
import os, sys, time
import numpy as np, pandas as pd, torch

__file__ = os.path.abspath(__file__)
exec(open(os.path.join(os.path.dirname(__file__), 'smooth_common.py')).read())
from dihiggs.heftnet import HEFTDensity

E30 = 255.0 + 30.0 * np.arange(41)                 # the 40 merged bins
E15 = EDGES                                        # the 80 bins of 15 GeV
RUN = os.path.join(REPO, 'code', 'runs', 'DENS_merged30')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------------------------------------------------------------------------------
# 1. The merged training data: every row of the training-file spectra
#    (including the restored empty bins), summed over each pair of bins
# ------------------------------------------------------------------------------
df = pd.read_csv(TRAIN_FILE)
counted, _ = CN.recover_counts(df)
full = CN.add_empty_bins(counted)
full['bin'] = np.rint(full.mhh.to_numpy() * 100 - 0.5).astype(int)
full['pair'] = full['bin'] // 2
g = full.groupby(['ID', 'pair']).agg(sigma=('sigma', 'sum'), quantum=('quantum', 'first'), nb=('bin', 'size'),
                                     **{c: (c, 'first') for c in THETA_COLS}).reset_index()
g = g[g.nb == 2]                                    # only complete pairs
th_m = g[THETA_COLS].to_numpy(np.float64)
C_m = heft_monomials(th_m)
pair, t_m, q_m = g.pair.to_numpy(), g.sigma.to_numpy(), g.quantum.to_numpy()
print(f'merged training rows: {len(g)} (30 GeV bins of {g.ID.nunique()} spectra)')

theta_t, C_t = D['th'], D['C']
nS = len(T)

# ------------------------------------------------------------------------------
# 2. Smooth model and free fit on the merged bins
# ------------------------------------------------------------------------------
from dihiggs.smooth import SplineBasis
basis = SplineBasis(40, threshold=True)
sm30 = SmoothHEFT(basis).fit_poisson(C_m, basis.integrals(E30)[pair], t_m, q_m)
f_s, _ = sm30.predict(np.repeat(C_t, 80, axis=0), np.tile(basis.integrals(E15), (nS, 1)))
f_s = f_s.reshape(nS, 80)

A = np.zeros((40, 23))
for k in range(40):
    s = pair == k; Ck, tk, qk = C_m[s], t_m[s], q_m[s]; var = qk * (tk + qk)
    for it in range(12):
        w = 1 / var; a = np.linalg.solve(Ck.T @ (w[:, None] * Ck), Ck.T @ (w * tk)); fk = Ck @ a
        var = np.sqrt(var * np.where(fk > 0, qk * np.maximum(fk, qk), qk * (tk + qk)))
    A[k] = a
f_h = 0.5 * C_t @ np.repeat(A, 2, axis=0).T                      # each half of a pair gets half

# ------------------------------------------------------------------------------
# 3. The density network on the merged bins
# ------------------------------------------------------------------------------
X = np.zeros((len(g), 6)); X[:, 0] = (0.5 * (E30[pair] + E30[pair + 1]) - 255.0) / 1500.0; X[:, 1:] = th_m
X = torch.tensor(X, dtype=torch.float32, device=DEVICE)
T_m = torch.tensor(t_m, dtype=torch.float32, device=DEVICE)
Q_m = torch.tensor(q_m, dtype=torch.float32, device=DEVICE)

def make():
    return HEFTDensity(activation='nn.SiLU', n_nodes=8, threshold=True, input_map='sqrt', edges=E30)

params = os.path.join(RUN, 'dihiggs_params.pth')
if os.path.exists(params):
    net = make(); net.load(params); net.to(DEVICE)
    print(f'loaded {params}')
else:
    os.makedirs(RUN, exist_ok=True)
    torch.manual_seed(42)
    net = make().to(DEVICE)
    t0 = time.time()
    # (a) L1 loss, mini-batches of 2,048 rows, 200,000 iterations, lr 2e-3 halved every 50,000
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=50_000, gamma=0.5)
    gen = torch.Generator(device=DEVICE).manual_seed(42)
    for it in range(200_000):
        idx = torch.randint(0, len(T_m), (2048,), generator=gen, device=DEVICE)
        opt.zero_grad(); loss = torch.mean(torch.abs(net(X[idx]) - T_m[idx])); loss.backward()
        opt.step(); sched.step()
        if (it + 1) % 50_000 == 0:
            print(f'  L1 iteration {it + 1}: loss {loss.item():.3e}  ({(time.time() - t0) / 60:.1f} min)', flush=True)
    # (b) Huber (delta = 2) of the pulls, Poisson variance from the model (held fixed in
    #     each iteration), plus the smoothness penalty; full batch, 5,000 iterations,
    #     lr 1e-4 halved every 1,250
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1_250, gamma=0.5)
    for it in range(5_000):
        opt.zero_grad()
        pred = net(X)                    # (not f, which holds the smooth model's test predictions)
        r = (pred - T_m) / torch.sqrt(Q_m * (torch.clamp(pred.detach(), min=0.0) + Q_m))
        a_ = torch.abs(r)
        loss = torch.mean(torch.where(a_ <= 2.0, 0.5 * r**2, 2.0 * (a_ - 1.0))) + 1e-6 * net.curvature_penalty()
        loss.backward(); opt.step(); sched.step()
        if (it + 1) % 1_000 == 0:
            print(f'  fine-tune iteration {it + 1}: loss {loss.item():.5f}  ({(time.time() - t0) / 60:.1f} min)', flush=True)
    net.save(params)
    open(os.path.join(RUN, 'README.txt'), 'w').write(
        'Density network trained on 30 GeV bins for the merged-bin test (docs/paper/merged_test.py).\n'
        'HEFTDensity(activation="nn.SiLU", n_nodes=8, threshold=True, input_map="sqrt", edges=255+30*arange(41))\n')
net = net.cpu().double().eval()
# predict the 15 GeV bins: integrate the density over the 15 GeV bins
from dihiggs.quadrature import bin_nodes
m15, w15 = bin_nodes(E15, 8)
with torch.no_grad():
    a = net.density_coeffs(m15.double()) * w15.double()[:, None]
    A15 = a.view(80, 8, 23).sum(1).numpy()
f_d = C_t @ A15.T

# the reference: the published models, fitted on the 15 GeV bins
net15 = load_density_network('FTD_SiLU_huber_d2_pearson_curv1e-06')
f_d15 = net15.spectrum(theta_t)

# ------------------------------------------------------------------------------
# 4. Measures: -2 ln lambda of the 15 GeV predictions, and the chi-squared of the
#    split of each pair into its two halves, in units of its Poisson noise
# ------------------------------------------------------------------------------
def lr_stat(fp):
    mu_ = fp / qt[:, None]; p_ = mu_ > 0
    return np.where(p_, 2 * (mu_ - n + np.where(n > 0, n * np.log(np.where(n > 0, n, 1) / np.where(p_, mu_, 1)), 0)), np.nan)

D_obs = T[:, 0::2] - T[:, 1::2]
def split_chi2(fp):
    Dp = fp[:, 0::2] - fp[:, 1::2]
    var = qt[:, None] * (np.maximum(fp[:, 0::2], qt[:, None]) + np.maximum(fp[:, 1::2], qt[:, None]))
    return (D_obs - Dp)**2 / var

models = [('free fit on 30 GeV bins, halved', f_h), ('smooth model, fitted on 30 GeV bins', f_s),
          ('density model, trained on 30 GeV bins', f_d), ('smooth model, fitted on 15 GeV bins', f),
          ('density model, trained on 15 GeV bins', f_d15)]
print(f"\n{'':40s} {'-2lnL 255-555':>14s} {'>555':>6s} | {'split chi2 255-555':>18s} {'>555':>6s} {'first 3 pairs':>16s}")
for lab, fp in models:
    dv, sc = lr_stat(fp), split_chi2(fp)
    print(f'{lab:40s} {np.nanmean(dv[:, :20]):14.3f} {np.nanmean(dv[:, 20:]):6.3f} | {np.nanmean(sc[:, :10]):18.3f} '
          f'{np.nanmean(sc[:, 10:]):6.3f} {np.array2string(np.nanmean(sc[:, :3], 0), precision=1):>16s}')
np.savez(os.path.join(PAPER, 'merged_test.npz'),          # chi2 of the split per 30 GeV pair, averaged over the test spectra
         split_h=np.nanmean(split_chi2(f_h), 0), split_s=np.nanmean(split_chi2(f_s), 0),
         split_d=np.nanmean(split_chi2(f_d), 0), split_s15=np.nanmean(split_chi2(f), 0),
         split_d15=np.nanmean(split_chi2(f_d15), 0))
print('wrote merged_test.npz')
