import torch
import torch.nn as nn
import numpy as np

name     = 'heftnet'
features = ['mhh', 'klambda', 'CT', 'CTT', 'CGGH', 'CGGHH']
target   = 'sigma'
nodes    = 25
nhidden  = 10
noutputs = 23

class Sin(nn.Module):

    def __init__(self):
        # initial base class (nn.Module)
        super().__init__()

    def forward(self, x):
        return torch.sin(x)
        
class HEFTNet(nn.Module):

    def __init__(self):

        # initial base class (nn.Module)
        super().__init__()

        # model the 23 functions a_i(m_hh) with a simple deep neural network 
        cmd = 'self.P = nn.Sequential(nn.Linear(1, nodes), nn.SiLU(),'
        for _ in range(nhidden):
            cmd += 'nn.Linear(nodes, nodes), nn.SiLU(),'
        cmd += 'nn.Linear(nodes, noutputs))'
        exec(cmd)

        self.Q = nn.Linear(1, noutputs, bias=False)
        
    # required method: this function computes the sqrt(cross section)
    def forward(self, x):
        # x.shape: (N, 6), where N is the batch size

        # compute vector of Wilson coefficient functions
        mhh, klambda, ct, ctt, cggh, cgghh = x.transpose(1, 0)

        C = torch.column_stack((
            ct**4,                 # A1
            ctt**2,
             ct**2*klambda**2,
             cggh**2*klambda**2,
             cgghh**2,
             ctt*ct**2,
             klambda*ct**3,
             ct*klambda*ctt,
             cggh*klambda*ctt,                   
             ctt*cgghh, 
             cggh*klambda*ct**2,
             cgghh*ct**2,                   
             klambda**2*cggh*ct, 
             cgghh*ct*klambda,
             cggh*cgghh*klambda,
             ct**3*cggh, 
             ct*ctt*cggh, 
             ct*cggh**2*klambda,
             ct*cggh*cgghh, 
             ct**2*cggh**2,
             ctt*cggh**2,                    
             cggh**3*klambda,    
             cggh**2*cgghh))

        # compute coefficients with NN
        A = self.coeffs(mhh)
        
        # compute cross section(s) per 15 GeV bin
        xsec = (C * A).sum(dim=1) 

        return xsec
        
    def coeffs(self, x):
        # must reshape input from (N, ) to (N, 1)
        return self.P(x.view(-1, 1)) * torch.exp(self.Q(x.view(-1, 1)))
