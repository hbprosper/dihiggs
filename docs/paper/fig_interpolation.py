# Network predictions interpolated between the bin centres, against the smooth model, for three test spectra
import glob, numpy as np, torch, matplotlib.pyplot as plt
from dihiggs import heftnet as NN
from dihiggs.smooth import SmoothHEFT
plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.size": 11,
                     'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True})
P = np.load('/home/claude/paper2/smooth_test.npz')
T, qt, th = P['T'], P['qt'], P['th']
SPEC = [786, 878, 675]                                       # the median-deviance row of the 9-panel figure
mg = np.arange(255.0, 1455.01, 0.5)                          # fine grid in m_hh (GeV)
x = (mg - 255.0) / 1500.0                                    # HEFTNet's input variable
members = sorted(glob.glob('/home/claude/ana5/ENS_bootstrap_huber_d2/member_*/dihiggs_params.pth'))
net = np.zeros((len(members), len(SPEC), len(mg)))
for a, f in enumerate(members):
    m = NN.HEFTNet(); m.load(f); m.eval()
    for b, s in enumerate(SPEC):
        X = torch.tensor(np.column_stack([x, np.tile(th[s], (len(mg), 1))]), dtype=torch.float32)
        with torch.no_grad():
            net[a, b] = m(X).double().numpy() / 15.0         # per 15 GeV -> per GeV
sm = SmoothHEFT.load('/home/claude/work/code/runs/SMOOTH_J40/smooth_model.npz')
dens, dsd = sm.dsigma_dm(th[SPEC], mg)
cen = 262.5 + 15 * np.arange(80); edges = 255 + 15 * np.arange(81)
np.savez('interp.npz', mg=mg, net=net, dens=dens, dsd=dsd)

fig = plt.figure(figsize=(10.5, 6.2))
gs = fig.add_gridspec(2, 3, height_ratios=[2.2, 1], hspace=0.0, wspace=0.28, left=0.07, right=0.99, top=0.9, bottom=0.1)
lo, hi = 255, 700
sel = (mg >= lo) & (mg <= hi)
names = [r'\kappa_\lambda', r'c_t', r'c_{tt}', r'c_{ggh}', r'c_{gghh}']
for b, s in enumerate(SPEC):
    ax = fig.add_subplot(gs[0, b]); rx = fig.add_subplot(gs[1, b], sharex=ax)
    k = 1e3                                                  # pb -> fb
    for a in range(len(members)):
        ax.plot(mg[sel], k * net[a, b, sel], color='#C0392B', lw=0.45, alpha=0.45, label='20 networks' if a == 0 else None)
    nm = net[:, b].mean(0); ns = net[:, b].std(0, ddof=1)
    ax.plot(mg[sel], k * nm[sel], color='#C0392B', lw=1.4, ls='--', label='network mean')
    ax.fill_between(mg[sel], k * (dens[b] - dsd[b])[sel], k * (dens[b] + dsd[b])[sel], color='#5B8FC7', alpha=0.5, lw=0)
    ax.plot(mg[sel], k * dens[b, sel], color='#1B4F8A', lw=1.6, label=r'smooth model, $\pm1\sigma$')
    ok = (cen <= hi) & (T[s] > 0)
    ax.errorbar(cen[ok], k * T[s][ok] / 15, yerr=k * np.sqrt(T[s][ok] * qt[s]) / 15, fmt='o', ms=2.6, lw=0.8, color='k', label='POWHEG / 15 GeV', zorder=5)
    for e in edges[edges <= hi]:
        ax.axvline(e, color='gray', lw=0.3, alpha=0.4); rx.axvline(e, color='gray', lw=0.3, alpha=0.4)
    ax.set_xlim(lo, hi); ax.tick_params(labelbottom=False)
    lab = [f'${n}={v:.2f}$' for n, v in zip(names, th[s])]
    ax.set_title(', '.join(lab[:3]) + '\n' + ', '.join(lab[3:]), fontsize=9.5)
    if b == 0:
        ax.set_ylabel(r'$d\sigma/dm_{hh}$ [fb/GeV]'); rx.set_ylabel(r'spread / value [\%]')
        ax.legend(frameon=False, fontsize=8.5, loc='upper right')
    # lower panel: relative spread of the 20 networks vs the smooth model's fit uncertainty
    rx.semilogy(mg[sel], 100 * (ns / np.abs(nm))[sel], color='#C0392B', lw=1.0, label='spread of the 20 networks')
    rx.semilogy(mg[sel], 100 * (dsd[b] / np.abs(dens[b]))[sel], color='#1B4F8A', lw=1.2, label='smooth model, fit uncertainty')
    rx.plot(cen[cen <= hi], 100 * (ns / np.abs(nm))[np.searchsorted(mg, cen[cen <= hi])], 'o', ms=2.4, color='#C0392B')
    rx.set_xlabel(r'$m_{hh}$ [GeV]'); rx.set_ylim(0.03, 30)
    if b == 0: rx.legend(frameon=False, fontsize=8, loc='upper right')
fig.savefig('fig_interpolation.pdf')
# numbers for the caption / message
sel2 = (mg >= 255) & (mg <= 600)
for b, s in enumerate(SPEC):
    rel = net[:, b].std(0, ddof=1) / np.abs(net[:, b].mean(0))
    atc = rel[np.searchsorted(mg, cen[cen <= 600])]
    print(s, 'network rel spread 255-600: at centres median %.2f%%, between centres max %.2f%%; smooth fit median %.3f%%' % (
        100 * np.median(atc), 100 * rel[sel2].max(), 100 * np.median((dsd[b] / np.abs(dens[b]))[sel2])))
