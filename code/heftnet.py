# ----------------------------------------------------------
import torch
import torch.nn as nn
import numpy as np
import dihiggs.nn as mlp
# ----------------------------------------------------------
NAME     = 'heftnet'
FEATURES = ['mhh', 'klambda', 'CT', 'CTT', 'CGGH', 'CGGHH']
TARGET   = 'sigma'

WIDTH    = 25
HIDDEN   =  5
ACTIVATION = 'nn.ReLU'
# ----------------------------------------------------------
class Sin(nn.Module):
    def __init__(self):
        # initial base class (nn.Module)
        super().__init__()

    def forward(self, x):
        return torch.sin(x)
        
class HEFTNet(mlp.Model):

    def __init__(self, width=WIDTH, hidden=HIDDEN, activation=ACTIVATION):

        # initial base class (nn.Module)
        super().__init__()

        # model the 23 functions a_i(m_hh) with a simple deep neural network 
        cmd = f'self.P = nn.Sequential(nn.Linear(1, width), {activation}(),'
        for _ in range(hidden):
            cmd += f'nn.Linear(width, width), {activation}(),'
        cmd += 'nn.Linear(width, 23), nn.Tanh())'
        exec(cmd)

        self.Q = nn.Linear(1, 23)
        
    def forward(self, x):
        # x.shape: [N, 6], where N is the batch size

        # compute vector of Wilson coefficient functions
        mhh, klambda, ct, ctt, cggh, cgghh = x.transpose(1, 0)

        C = torch.column_stack((
             ct**4, 
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
        cross_section = (C * A).sum(dim=1)

        return cross_section
        
    def coeffs(self, x):
        # must reshape input from (N, ) to (N, 1)
        x = x.view(-1, 1)
        P = self.P(x)
        Q = self.Q(x)
        return P * torch.exp(Q)
