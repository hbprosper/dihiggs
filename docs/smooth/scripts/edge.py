# edge-bin test: predict the unused bins 240-255 and 1455-1500 GeV of the test spectra
import numpy as np, pandas as pd, pickle, sys
src = pd.read_csv('/mnt/user-data/uploads/dihiggs/data/powheg_total_gauss_params_13TeV.csv.gz')
d = np.load('inputs.npz'); T, qt, Ct = d['T'], d['q_t'], d['C_t']; nS = len(T); ids = d['id_t']
src = src.set_index('ID').loc[ids]
cols = [str(k) for k in range(100)]
F = src[cols].to_numpy(float)
assert np.allclose(F[:, 17:97], T, rtol=1e-6, atol=1e-12), 'bins do not line up'
C1 = Ct.reshape(nS, 80, 23)[:, 0, :]               # the monomials do not depend on m
for tag in sys.argv[1:]:
    m = pickle.load(open(f'model_{tag}.pkl', 'rb'))
    for lab, (lo, hi), obs in [('240-255 GeV', (240, 255), F[:, 16]), ('1455-1500 GeV', (1455, 1500), F[:, 97:100].sum(1))]:
        Irow = m.basis.integrals(np.array([lo, hi]))[0]
        f, v = m.predict(C1, np.tile(Irow, (nS, 1)))
        noise = qt * np.maximum(f, qt)
        z = (obs - f) / np.sqrt(noise + v)
        print(f'{tag} {lab}: observed mean {obs.mean()*1e3:.3g} fb, predicted {f.mean()*1e3:.3g} fb, ratio of sums {obs.sum()/f.sum():.3f}, '
              f'pull mean {np.mean(z):+.2f} s.d. {np.std(z):.2f} (noise + fit only), median count {np.median(obs/qt):.0f}')
