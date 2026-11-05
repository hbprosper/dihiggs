# ------------------------------------------------------------------------------
# figs_appendices.py: the new figures of the paper's appendices A (smooth model)
# and B (neural-network density model), from the committed model files.
#
#   fig_A_accuracy.pdf      -2 ln lambda per bin vs m_hh: smooth model, free fit
#   fig_A_model_error.pdf   the model-error term (made by docs/model/model_error_study.py)
#   fig_A_merged.pdf        merged-bin test: smooth model, free fit (needs merged_test.npz)
#   fig_A_coefficients.pdf  the 23 a_i(m) of the smooth model with exact bands
#   fig_B_accuracy.pdf      -2 ln lambda per bin vs m_hh: smooth model, free fit, network
#   fig_B_calibration.pdf   epsilon(m) of both models; <z^2> by fifth of cancellation
#   fig_B_density.pdf       d sigma/dm of both models for three test spectra, and their ratio
#   fig_B_merged.pdf        merged-bin test including the network (needs merged_test.npz)
#   fig_B_coefficients.pdf  the network's a_i(m) over the smooth model's bands
# The other figures of the appendices are fig_basis.py and fig_spectra_dz.py.
#
# Run from docs/paper:   python figs_appendices.py     (a few minutes on a laptop)
# Prints the numbers quoted in the appendices.
# ------------------------------------------------------------------------------
import os, sys, shutil, subprocess
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from scipy.optimize import brentq

__file__ = os.path.abspath(__file__)
exec(open(os.path.join(os.path.dirname(__file__), 'smooth_common.py')).read())
NET_RUN = 'FTD_SiLU_huber_d2_pearson_curv1e-06'
plt.rcParams.update({"text.usetex": shutil.which('latex') is not None, "font.family": "serif", "font.size": 10,
                     'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True})
BLUE, RED, GREEN, GREY = '#1B4F8A', '#C0392B', '#2E933C', 'gray'
smooth = SmoothHEFT.load(MODEL_FILE)
theta, C_test, nS = D['th'], D['C'], len(T)
net = load_density_network(NET_RUN)

# ---- predictions --------------------------------------------------------------
C_fit, b_fit, t_fit, q_fit = training_rows()
A_free = np.zeros((80, 23))
for k in range(80):
    s = b_fit == k; Ck, tk, qk = C_fit[s], t_fit[s], q_fit[s]; var = qk * (tk + qk)
    for it in range(12):
        w = 1 / var; a = np.linalg.solve(Ck.T @ (w[:, None] * Ck), Ck.T @ (w * tk)); fk = Ck @ a
        var = np.sqrt(var * np.where(fk > 0, qk * np.maximum(fk, qk), qk * (tk + qk)))
    A_free[k] = a
f_free = C_test @ A_free.T
f_net = net.spectrum(theta)

def lr_stat(fp):
    mu_ = fp / qt[:, None]; p_ = mu_ > 0
    return np.where(p_, 2 * (mu_ - n + np.where(n > 0, n * np.log(np.where(n > 0, n, 1) / np.where(p_, mu_, 1)), 0)), np.nan)
LR = {'smooth model': lr_stat(f), 'free fit': lr_stat(f_free), 'neural-network density model': lr_stat(f_net)}
for k_, d_ in LR.items():
    print(f'{k_:30s} -2lnL: 255-555 {np.nanmean(d_[:, :20]):.3f}  >555 {np.nanmean(d_[:, 20:]):.3f}  median {np.median(np.nanmean(d_, 1)):.3f}')

# ---- accuracy figures -----------------------------------------------------------
def accuracy_fig(names, out):
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    style = {'smooth model': (BLUE, '-'), 'free fit': (RED, '--'), 'neural-network density model': (GREEN, ':')}
    for nm in names:
        c, ls = style[nm]
        ax.stairs(np.nanmean(LR[nm], 0), EDGES, color=c, ls=ls, lw=1.4, label=nm)
    ax.axhline(1, color='k', lw=0.6)
    ax.set_xlim(255, 1455); ax.set_ylim(0.8, 2.6)
    ax.set_xlabel(r'$m_{hh}$ [GeV]'); ax.set_ylabel(r'mean $-2\ln\lambda$ per bin'); ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
accuracy_fig(['smooth model', 'free fit'], 'fig_A_accuracy.pdf')
accuracy_fig(['smooth model', 'free fit', 'neural-network density model'], 'fig_B_accuracy.pdf')

# ---- the model-error figure of appendix A ---------------------------------------
model_dir = os.path.join(REPO, 'docs', 'model')
subprocess.run([sys.executable, os.path.join(model_dir, 'model_error_study.py')], check=True, cwd=model_dir,
               stdout=subprocess.DEVNULL)
shutil.copy(os.path.join(model_dir, 'fig_model_error.pdf'), 'fig_A_model_error.pdf')

# ---- coefficient functions --------------------------------------------------------
mc = np.arange(250.0, 1455.0 + 1e-9, 1.0)
a_sm, sd_sm = smooth.coefficients(mc)
a_nn = net.coefficients(mc)
def coef_fig(with_net, out):
    fig, axes = plt.subplots(6, 4, figsize=(11, 13.5), sharex=True)
    for i, ax in enumerate(axes.flat[:23]):
        y, e_ = 15e3 * a_sm[:, i], 15e3 * sd_sm[:, i]
        ax.fill_between(mc, y - e_, y + e_, color='#C9DDF0', lw=0)
        ax.plot(mc, y, color=BLUE, lw=1)
        if with_net:
            ax.plot(mc, 15e3 * a_nn[:, i], color=GREEN, lw=1, ls='--')
        ax.axhline(0, color='gray', lw=0.5, ls=':')
        ax.set_title(f'$a_{{{i + 1}}}$', fontsize=10)
    axes.flat[23].axis('off')
    for ax in axes[-1]: ax.set_xlabel(r'$m_{hh}$ [GeV]')
    for ax in axes[:, 0]: ax.set_ylabel(r'$15\,\mathrm{GeV}\times a_i$ [fb]')
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
coef_fig(False, 'fig_A_coefficients.pdf')
coef_fig(True, 'fig_B_coefficients.pdf')
rel = np.sqrt(np.mean((a_nn - a_sm)**2, 0)) / np.abs(a_sm).max(0)
print('a_i: r.m.s. difference network - smooth / max|a_i| (%):', np.round(100 * rel, 2))

# ---- calibration of the network: its own epsilon ----------------------------------
C_q, b_q, t_q, q_q = quick_test_rows()
u_c = (CENTRES - 855) / 600; n_rows = np.bincount(b_q, minlength=80)
def fit_epsilon(f_q, v_q, S_q):
    noise_q = q_q * np.maximum(f_q, q_q); e2 = np.zeros(80)
    for k in range(80):
        s = b_q == k; r2 = (t_q - f_q)[s]**2
        g = lambda c: np.mean(r2 / (noise_q[s] + v_q[s] + c * S_q[s]**2)) - 1
        e2[k] = 0.0 if g(0) <= 0 else brentq(g, 0, 1.0)
    ok = e2 > 0
    p = np.polyfit(u_c[ok], 0.5 * np.log(e2[ok]), 2, w=np.sqrt(n_rows[ok]))
    return np.exp(np.polyval(p, u_c)), np.sqrt(e2)
I15 = smooth.basis.integrals(EDGES); A_sm = smooth.coefficient_integrals(I15)
fq, vq = smooth.predict(C_q, I15[b_q])
eps_s, eps_s_bin = fit_epsilon(fq, vq, np.abs(C_q * A_sm[b_q]).sum(1))
with torch.no_grad():
    A_nn = net.bin_integrals().numpy()
fq_n = (C_q * A_nn[b_q]).sum(1)
eps_n, eps_n_bin = fit_epsilon(fq_n, np.zeros_like(fq_n), np.abs(C_q * A_nn[b_q]).sum(1))
print('epsilon (%) at 262.5, 600, 1447.5 GeV: smooth', np.round(100 * eps_s[[0, 23, 79]], 3), ' network', np.round(100 * eps_n[[0, 23, 79]], 3))
canc = T.sum(1) / S.sum(1); q5 = np.quantile(canc, [0, .2, .4, .6, .8, 1])
fifths = [(canc >= a_) & (canc <= b_) for a_, b_ in zip(q5[:-1], q5[1:])]
use = (T > 0) | (f > 0)
def pulls(fp, vp, Sp, e):
    nz = qt[:, None] * np.maximum(fp, qt[:, None])
    return np.where(use, (T - fp) / np.sqrt(nz + vp + (e[None] * Sp)**2), np.nan)
S_net = np.abs(C_test[:, None, :] * A_nn[None]).sum(2)
Z = {'smooth model': pulls(f, v, S, eps_s), 'smooth model, no model-error term': pulls(f, v, S, 0 * eps_s),
     'network': pulls(f_net, 0.0, S_net, eps_n), "network, smooth model's epsilon": pulls(f_net, 0.0, S_net, eps_s),
     'network, no model-error term': pulls(f_net, 0.0, S_net, 0 * eps_n)}
for k_, z_ in Z.items():
    print(f'{k_:36s} s.d. {np.nanstd(z_):.3f} ({np.nanstd(z_[:, :20]):.3f}, {np.nanstd(z_[:, 20:]):.3f})  |z|<2 {np.nanmean(np.abs(z_) < 2):.3f}  '
          f'<z^2> by fifth {np.round([np.nanmean(z_[s]**2) for s in fifths], 2)}')
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for eb, el, c, mk, lab in [(eps_s_bin, eps_s, BLUE, 'o', 'smooth model'), (eps_n_bin, eps_n, GREEN, 's', 'neural-network density model')]:
    ok = eb > 0
    ax[0].semilogy(CENTRES[ok], 100 * eb[ok], mk, ms=2.5, color=c, alpha=0.5)
    ax[0].semilogy(CENTRES, 100 * el, color=c, lw=1.6, label=lab)
ax[0].set_xlabel(r'$m_{hh}$ [GeV]'); ax[0].set_ylabel(r'$\varepsilon$ [\%]'); ax[0].legend(frameon=False, fontsize=9)
x5 = np.arange(5); w5 = 0.2
for j, (k_, c) in enumerate([('smooth model, no model-error term', '#BFD3EA'), ('smooth model', BLUE),
                             ('network, no model-error term', '#A9D8B2'), ('network', GREEN)]):
    ax[1].bar(x5 + (j - 1.5) * w5, [np.nanmean(Z[k_][s]**2) for s in fifths], w5, color=c, label=k_.replace('network', 'network'))
ax[1].axhline(1, color='k', lw=0.6); ax[1].set_ylim(0.8, 1.95)
ax[1].set_xticks(x5); ax[1].set_xticklabels(['1\nstrongest', '2', '3', '4', '5\nweakest'])
ax[1].set_xlabel('fifth of the cancellation ratio'); ax[1].set_ylabel(r'$\langle z^2\rangle$'); ax[1].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig('fig_B_calibration.pdf'); plt.close(fig)

# ---- the densities of the two models for three test spectra ------------------------
SPEC = [786, 878, 675]
mf = np.arange(255.0, 1455.0 + 1e-9, 0.5)
d_s, sdf = smooth.dsigma_dm(theta[SPEC], mf)
a_g = smooth.coefficients(mf)[0]
eps_g = np.exp(np.polyval(np.polyfit(u_c, np.log(eps_s), 2), (mf - 855) / 600))
dS = np.abs(C_test[SPEC][:, None, :] * a_g[None]).sum(2)
sd_tot = np.sqrt(sdf**2 + (eps_g[None] * dS)**2)
d_n = net.dsigma_dm(theta[SPEC], mf)
fig = plt.figure(figsize=(11, 5.6))
gs = fig.add_gridspec(2, 3, height_ratios=[2.2, 1], hspace=0.0, wspace=0.3)
for b, s in enumerate(SPEC):
    ax = fig.add_subplot(gs[0, b]); rx = fig.add_subplot(gs[1, b], sharex=ax)
    ax.fill_between(mf, 1e3 * (d_s[b] - sd_tot[b]), 1e3 * (d_s[b] + sd_tot[b]), color='#C9DDF0', lw=0)
    ax.plot(mf, 1e3 * d_s[b], color=BLUE, lw=1.4, label=r'smooth model, $\pm1\sigma$')
    ax.plot(mf, 1e3 * d_n[b], color=GREEN, lw=1.2, ls='--', label='neural-network density model')
    ax.plot(CENTRES, 1e3 * T[s] / 15.0, 'k.', ms=3, label='POWHEG / 15 GeV')
    ax.set_xlim(255, 800); ax.tick_params(labelbottom=False)
    if b == 0:
        ax.set_ylabel(r'$d\sigma/dm_{hh}$ [fb/GeV]'); ax.legend(frameon=False, fontsize=8)
    sel = mf <= 800
    rx.fill_between(mf, -sd_tot[b] / sd_tot[b], sd_tot[b] / sd_tot[b], color='#C9DDF0', lw=0)
    rx.plot(mf, (d_n[b] - d_s[b]) / sd_tot[b], color=GREEN, lw=1)
    rx.axhline(0, color=BLUE, lw=1)
    rx.set_ylim(-3, 3); rx.set_xlabel(r'$m_{hh}$ [GeV]')
    rx.plot(EDGES[EDGES <= 800], np.full((EDGES <= 800).sum(), -2.85), '|', color='gray', ms=4)   # bin edges, as ticks
    if b == 0:
        rx.set_ylabel(r'(NN $-$ smooth) / $\sigma$')
fig.savefig('fig_B_density.pdf', bbox_inches='tight'); plt.close(fig)

# density agreement over all test points, at and between bin centres
mg = np.arange(255.0, 1455.0 + 1e-9, 0.5); onc = np.isin(mg, CENTRES)
ds_all, sdf_all = smooth.dsigma_dm(theta, mg); dn_all = net.dsigma_dm(theta, mg)
a_all = smooth.coefficients(mg)[0]
eg_all = np.exp(np.polyval(np.polyfit(u_c, np.log(eps_s), 2), (mg - 855) / 600))
dS_all = np.zeros_like(ds_all)
for i0 in range(0, nS, 100):
    dS_all[i0:i0 + 100] = np.abs(C_test[i0:i0 + 100, None, :] * a_all[None]).sum(2)
st_all = np.sqrt(sdf_all**2 + (eg_all[None] * dS_all)**2)
r_ = np.abs(dn_all / ds_all - 1); z_ = np.abs(dn_all - ds_all) / st_all
thr = mg < 350
print(f'density below 350 GeV: median |NN/smooth - 1| {100 * np.median(r_[:, thr]):.3f}%, 95th percentile {100 * np.percentile(r_[:, thr], 95):.3f}%; '
      f'median smooth total sd / density {100 * np.median((st_all / np.abs(ds_all))[:, thr]):.3f}%; within 2 sigma {np.mean(z_[:, thr] < 2):.3f}')
print(f'density: median |NN/smooth - 1| {100 * np.median(r_):.3f}% (at centres {100 * np.median(r_[:, onc]):.3f}%, '
      f'between {100 * np.median(r_[:, ~onc]):.3f}%); within 1 sigma {np.mean(z_ < 1):.3f}, 2 sigma {np.mean(z_ < 2):.3f}')

# ---- merged-bin figures --------------------------------------------------------------
if os.path.exists('merged_test.npz'):
    M = np.load('merged_test.npz'); e30 = 255.0 + 30.0 * np.arange(41)
    def merged_fig(rows, out):
        fig, ax = plt.subplots(figsize=(6.2, 3.6))
        for key, lab, c, ls in rows:
            ax.stairs(M[key], e30, color=c, ls=ls, lw=1.4, label=lab)
        ax.axhline(1, color='k', lw=0.6); ax.set_yscale('log'); ax.set_xlim(255, 1455)
        ax.set_xlabel(r'$m_{hh}$ [GeV]'); ax.set_ylabel(r'$\chi^2$ of the split, per pair'); ax.legend(frameon=False, fontsize=8)
        fig.tight_layout(); fig.savefig(out); plt.close(fig)
    merged_fig([('split_h', 'free fit on 30 GeV bins, halved', RED, '--'),
                ('split_s', 'smooth model, fitted on 30 GeV bins', BLUE, '-'),
                ('split_s15', 'smooth model, fitted on 15 GeV bins', GREY, ':')], 'fig_A_merged.pdf')
    merged_fig([('split_h', 'free fit on 30 GeV bins, halved', RED, '--'),
                ('split_s', 'smooth model, 30 GeV bins', BLUE, '-'),
                ('split_d', 'neural-network density model, 30 GeV bins', GREEN, '-'),
                ('split_d15', 'neural-network density model, 15 GeV bins', GREY, ':')], 'fig_B_merged.pdf')
    for key in ['split_h', 'split_s', 'split_d', 'split_s15', 'split_d15']:
        sp = M[key]; c30 = e30[:-1] + 15
        print(f'{key:10s} split chi2: first 3 pairs {np.round(sp[:3], 1)}; mean 255-345 {sp[:3].mean():.2f}; '
              f'360-555 {sp[(c30 > 360) & (c30 < 555)].mean():.2f}; >555 {sp[c30 > 555].mean():.2f}; max above 360 {sp[c30 > 360].max():.2f}')
else:
    print('merged_test.npz not found: run merged_test.py first for fig_A_merged.pdf and fig_B_merged.pdf')
print('done')
