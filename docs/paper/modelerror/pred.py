import numpy as np, pandas as pd, torch
from dihiggs import ensemble, heftnet as NN, finetune as ft
torch.set_num_threads(4)
df=pd.read_csv('/home/claude/work/data/heft_wide_gauss_traindata_13TeV.csv.gz')
N=len(df); tr=int(0.864*N); te=int(0.120*N); va=N-tr-te
q=slice(tr+va,tr+va+te)
tot=df.groupby('ID')['sigma'].transform('sum').to_numpy(float)
neff=ft.estimate_n_eff(df); print('N',N,'quick test',te,'N_eff',neff)
d=df.iloc[q]; X=torch.from_numpy(d[NN.FEATURES].to_numpy(np.float32))
out=dict(t=d['sigma'].to_numpy(float),mhh=d['mhh'].to_numpy(float),tot=tot[q],ID=d['ID'].to_numpy(),neff=neff)
for k,(rd,pat) in {'FT_boot':('ENS_bootstrap_huber_d2','member_*/dihiggs_params.pth'),'FT_seeds':('ENS_seeds_huber_d2','member_*/dihiggs_params.pth'),'L1_boot':('ENS_bootstrap_huber_d2','member_*/dihiggs_params_L1.pth')}.items():
    e=ensemble.Ensemble(NN.HEFTNet,'/home/claude/ana5/'+rd,pattern=pat)
    P=e.predict(X).double().numpy().reshape(len(e),-1)
    out[k]=P.mean(0); out[k+'_sd']=P.std(0,ddof=1); print(k,len(e))
np.savez('qt.npz',**out)
