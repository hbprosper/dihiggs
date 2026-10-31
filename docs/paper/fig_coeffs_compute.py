import glob, numpy as np, pandas as pd, torch
torch.set_num_threads(4)
from dihiggs import heftnet as NN, laplace as LA, counts as CN
df = pd.read_csv('/home/claude/work/data/heft_wide_gauss_traindata_13TeV.csv.gz')
N = len(df); tr = int(0.864 * N)
counted, _ = CN.recover_counts(df); full = CN.add_empty_bins(counted)
like = pd.concat([counted.iloc[:tr], full[full['count'] == 0]], ignore_index=True)
X = torch.tensor(like[NN.FEATURES].to_numpy(np.float64))
q = like.quantum.to_numpy(); t = like.sigma.to_numpy(); V = q * (t + q)
x = (torch.arange(80, dtype=torch.float64) + 0.5) / 100     # the 80 bin centres; mhh = 255 + 1500 x
A, K = [], []
for f in sorted(glob.glob('/home/claude/ana5/ENS_bootstrap_huber_d2/member_*/dihiggs_params.pth')):
    m = NN.HEFTNet(); m.load(f)
    lap = LA.Laplace(m); lap.fit_hessian(X, V); lap.fit_prior()
    with torch.no_grad(): A.append(lap.model.coeffs(x).numpy())
    K.append(np.diagonal(lap.coeff_covariance(x).detach().numpy(), axis1=1, axis2=2))
    print(f, flush=True)
np.savez('coeffs_bins.npz', x=x.numpy(), A=np.array(A), Kdiag=np.array(K))
