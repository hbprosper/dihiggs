import numpy as np, pickle
from dihiggs.smooth import SplineBasis, SmoothHEFT
d = np.load('inputs.npz'); edges = 255.0 + 15.0 * np.arange(81)
basis = SplineBasis(40, threshold=True); I = basis.integrals(edges)
m = SmoothHEFT(basis).fit_poisson(d['C_fit'], I[d['b_fit']], d['t_fit'], d['q_fit'])
pickle.dump(m, open('model_primary.pkl', 'wb'))
r = np.load('res_J40_thr.npz')
mg = np.linspace(250, 1455, 1206)
a, sd = m.coefficients(mg)
ac = m.coefficient_integrals(I) / 15.0          # bin averages of a_i (pb/GeV)
np.savez('primary.npz', mg=mg, a=a, sd=sd, ac=ac, eps=r['eps1'], lam=m.lam, gamma=m.gamma())
print('done; gamma per function', np.round(m.gamma(), 1))
