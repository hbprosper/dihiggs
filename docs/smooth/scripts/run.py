# smooth model: pass 1 (Poisson weights), discrepancy, pass 2 (weights include the discrepancy)
import numpy as np, sys, time, json
from scipy.optimize import brentq
from dihiggs.smooth import SplineBasis, SmoothHEFT
J = int(sys.argv[1]); thr = sys.argv[2] == '1'; tag = f'J{J}_{"thr" if thr else "nothr"}'
d = np.load('inputs.npz'); edges = 255.0 + 15.0 * np.arange(81); mhh = edges[:-1] + 7.5
basis = SplineBasis(J, threshold=thr); I = basis.integrals(edges)
C, b, t, q = d['C_fit'], d['b_fit'], d['t_fit'], d['q_fit']
Cq, bq, tq, qq = d['C_q'], d['b_q'], d['t_q'], d['q_q']
T, qt, Ct = d['T'], d['q_t'], d['C_t']; nS = len(T)
def S_of(model, Cr, br):                      # sum_i |C_i a_ik|, with a_ik the bin integral of a_i
    A = model.coefficient_integrals(I)        # [80, 23]
    return np.abs(Cr * A[br]).sum(1)
def eps_fit(model):
    fq, vq = model.predict(Cq, I[bq]); nq = qq * np.maximum(fq, qq); Sq = S_of(model, Cq, bq)
    e2 = np.zeros(80)
    for k in range(80):
        s = bq == k; r2 = (tq - fq)[s]**2
        g = lambda c: np.mean(r2 / (nq[s] + vq[s] + c * Sq[s]**2)) - 1
        e2[k] = 0.0 if g(0) <= 0 else brentq(g, 0, 1.0)
    nb = np.bincount(bq, minlength=80); u = (mhh - 855) / 600; ok = e2 > 0
    p = np.polyfit(u[ok], 0.5 * np.log(e2[ok]), 2, w=np.sqrt(nb[ok]))
    return np.exp(np.polyval(p, u)), np.sqrt(e2), p
def evaluate(model, eps, label):
    f, v = model.predict(Ct, np.tile(I, (nS, 1))); f, v = f.reshape(nS, 80), v.reshape(nS, 80)
    St = S_of(model, Ct, np.tile(np.arange(80), nS)).reshape(nS, 80)
    noise = qt[:, None] * np.maximum(f, qt[:, None])
    ok = T > 0
    out = {}
    for lab, extra in [('none', 0.0), ('eps', (eps[None] * St)**2)]:
        z = np.where(ok | (f > 0), (T - f) / np.sqrt(noise + v + extra), np.nan)
        canc = T.sum(1) / St.sum(1); q5 = np.quantile(canc, [0, .2, .4, .6, .8, 1])
        out[lab] = dict(sd=float(np.nanstd(z)), thr=float(np.nanstd(z[:, :20])), rest=float(np.nanstd(z[:, 20:])),
                        within2=float(np.nanmean(np.abs(z) < 2)),
                        quint=[float(np.nanmean(z[(canc >= a) & (canc <= bb)]**2)) for a, bb in zip(q5[:-1], q5[1:])])
        print(f'{label} [{lab:4s}] pull sd {out[lab]["sd"]:.3f} ({out[lab]["thr"]:.3f}, {out[lab]["rest"]:.3f}) |z|<2 {out[lab]["within2"]:.3f} quintiles {np.round(out[lab]["quint"], 2)}')
    return f, v, St, out
t0 = time.time()
m1 = SmoothHEFT(basis).fit_poisson(C, I[b], t, q)
print(f'{tag} pass 1: {time.time()-t0:.0f} s, ln Z {m1.log_evidence():.1f}, gamma {m1.gamma().sum():.0f}')
eps1, eb1, p1 = eps_fit(m1)
evaluate(m1, eps1, 'pass 1')
ev = (np.interp(b, np.arange(80), eps1) * S_of(m1, C, b))**2
m2 = SmoothHEFT(basis).fit_poisson(C, I[b], t, q, extra_var=ev)
eps2, eb2, p2 = eps_fit(m2)
f, v, St, out = evaluate(m2, eps2, 'pass 2')
print(f'eps(m) pass 1 at 262/600/1447: {np.round(100*eps1[[0,23,79]],3)} %;  pass 2: {np.round(100*eps2[[0,23,79]],3)} %')
np.savez(f'res_{tag}.npz', f=f, v=v, St=St, eps=eps2, eps_bin=eb2, eps1=eps1, beta=m2.beta, lam=m2.lam, H=m2.H, beta1=m1.beta, f1=m1.predict(Ct, np.tile(I, (nS, 1)))[0].reshape(nS, 80))
json.dump(dict(tag=tag, lnZ1=m1.log_evidence(), lnZ2=m2.log_evidence(), gamma=float(m2.gamma().sum()), out=out), open(f'res_{tag}.json', 'w'))
import pickle; pickle.dump(m2, open(f'model_{tag}.pkl', 'wb'))
print(f'total {time.time()-t0:.0f} s')
