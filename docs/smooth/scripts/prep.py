# compute and cache the inputs: monomials C, bin index, cross sections, quanta
import numpy as np, pandas as pd, torch
from dihiggs import heftnet as NN, counts as CN, laplace as LA
df = pd.read_csv('/home/claude/work/data/heft_wide_gauss_traindata_13TeV.csv.gz')
N = len(df); tr = int(0.864 * N); te = int(0.120 * N); va = N - tr - te
counted, _ = CN.recover_counts(df)
full = CN.add_empty_bins(counted); empty = full[full['count'] == 0]
fitrows = pd.concat([counted.iloc[:tr + va], empty], ignore_index=True)
quick = counted.iloc[tr + va:]
m0 = NN.HEFTNet().double()
def C_of(frame): return LA.wilson_coefficients(m0, torch.tensor(frame[NN.FEATURES].to_numpy(np.float64))).numpy()
def bin_of(frame): return np.rint(frame.mhh.to_numpy() * 100 - 0.5).astype(int)
sp = pd.read_csv('/home/claude/work/data/heft_wide_gauss_testdata_13TeV.csv'); bins = [c for c in sp.columns if c.isdigit()]
T = sp[bins].to_numpy(float); nS = len(T); _, qt, _ = CN.recover_counts_wide(T)
th = sp[['klambda','CT','CTT','CGGH','CGGHH']].to_numpy(np.float32).astype(np.float64)
xb = (np.arange(80) + 0.5) / 100
Ct = LA.wilson_coefficients(m0, torch.tensor(np.column_stack([np.tile(xb, nS), np.repeat(th, 80, axis=0)]))).numpy()
np.savez('inputs.npz',
         C_fit=C_of(fitrows), b_fit=bin_of(fitrows), t_fit=fitrows.sigma.to_numpy(), q_fit=fitrows.quantum.to_numpy(), id_fit=fitrows.ID.to_numpy(),
         C_q=C_of(quick), b_q=bin_of(quick), t_q=quick.sigma.to_numpy(), q_q=quick.quantum.to_numpy(),
         C_t=Ct, T=T, q_t=qt, th=th, id_t=sp.ID.to_numpy())
print(len(fitrows), len(quick), T.shape)
