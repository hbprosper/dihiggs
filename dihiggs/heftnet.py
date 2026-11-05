# ----------------------------------------------------------
# heftnet.py: physics-structured networks for the HEFT di-Higgs cross section
#
#   d sigma / d m (m; theta) = sum_i C_i(theta) a_i(m),   i = 1, ..., 23,
#
# where the 23 monomials C_i(theta) of theta = (klambda, ct, ctt, cggh, cgghh)
# are known exactly and the 23 functions a_i(m) are given by a network.
#
#   HEFTNet      the original model: forward(x) is the cross section in the
#                15 GeV bin whose centre is x (docs/dihiggs_model.pdf, sec. 9).
#   HEFTDensity  the network as a density: a_i(m) in pb/GeV, and forward(x)
#                is the INTEGRAL of the density over the bin of x, computed by
#                Gauss-Legendre quadrature (docs/density/density_plan.pdf).
#
# Both take the same input rows x = (mhh, klambda, CT, CTT, CGGH, CGGHH), with
# mhh = (m - 255 GeV)/1500 GeV at the bin centre, and return pb per bin, so the
# training notebooks, drwho's tools and finetune.py work with either.
# ----------------------------------------------------------
import torch
import torch.nn as nn
import numpy as np
from drwho import nn as mlp
from dihiggs.quadrature import bin_nodes
# ----------------------------------------------------------
NAME     = 'heftnet'
FEATURES = ['mhh', 'klambda', 'CT', 'CTT', 'CGGH', 'CGGHH']
TARGET   = 'sigma'

WIDTH    = 25
HIDDEN   =  5
ACTIVATION = 'nn.ReLU'           # HEFTNet (kept, so that saved models load)
DENSITY_ACTIVATION = 'nn.SiLU'   # HEFTDensity: a smooth activation, so d sigma/dm is smooth

M_LO    = 255.0                  # GeV: lower edge of the first bin
M_SCALE = 1500.0                 # GeV: mhh = (m - M_LO) / M_SCALE
M_H     = 125.0                  # GeV: Higgs boson mass, for the threshold factor
EDGES   = M_LO + 15.0 * np.arange(81)   # the 80 bins of 15 GeV, 255-1455 GeV
# ----------------------------------------------------------
class Sin(nn.Module):
    def __init__(self):
        # initial base class (nn.Module)
        super().__init__()

    def forward(self, x):
        return torch.sin(x)


def monomials(theta):
    '''
    The 23 monomials C_i(theta), in the order of Eq. (4.1) of arXiv:2304.01968
    (and of smooth.heft_monomials), for theta [N, 5] = (klambda, ct, ctt,
    cggh, cgghh): a tensor [N, 23].
    '''
    # NOTE: TorchScript cannot unpack a Tensor directly into a tuple of
    # variables ("Tensor cannot be used as a tuple"), so index explicitly.
    tt = theta.transpose(1, 0)
    klambda= tt[0]
    ct     = tt[1]
    ctt    = tt[2]
    cggh   = tt[3]
    cgghh  = tt[4]
    return torch.column_stack((
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


class HEFTNet(mlp.Model):

    def __init__(self, width=WIDTH, hidden=HIDDEN, activation=ACTIVATION,
                 n_coeffs=23, monomial_fn=monomials):
        # n_coeffs and monomial_fn exist only for tests with other functional
        # forms (code/density_toy.py); for HEFT keep the defaults.

        # initial base class (nn.Module)
        super().__init__()
        self.n_coeffs = n_coeffs
        self.monomial_fn = monomial_fn

        # model the 23 functions a_i(m_hh) with a simple deep neural network
        cmd = f'self.P = nn.Sequential(nn.Linear(1, width), {activation}(),'
        for _ in range(hidden):
            cmd += f'nn.Linear(width, width), {activation}(),'
        cmd += 'nn.Linear(width, n_coeffs), nn.Tanh())'
        exec(cmd)

        self.Q = nn.Linear(1, n_coeffs)

    def forward(self, x):
        # x.shape: [N, 6], where N is the batch size
        mhh = x[:, 0]

        # vector of the 23 monomials C_i(theta), [N, 23]
        C = self.monomial_fn(x[:, 1:])

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


class HEFTDensity(HEFTNet):
    '''
    HEFTNet as a density.

    The network gives the 23 coefficient functions as densities (pb/GeV),
        a_i(m) = phi(m) * P_i(z) * exp(Q_i(z)),   z = 2 sqrt((m - 250)/1205) - 1,
    with phi(m) = sqrt(1 - 4 m_h^2/m^2) the two-body threshold factor (if
    threshold=True). forward(x) returns, for each input row, the integral of
    d sigma/dm = sum_i C_i(theta) a_i(m) over the bin whose centre is x:
        f_k(theta) = sum_i C_i(theta) A_ik,   A_ik = integral of a_i over bin k,
    and A_ik is computed by n_nodes-point Gauss-Legendre quadrature in each bin.

    Because the a_i depend on m only, the network is evaluated at the fixed
    n_bins * n_nodes nodes once per call, however many rows there are; each
    row then only picks row k of the table A [n_bins, 23].

    Arguments
    ---------
    width, hidden, activation : as for HEFTNet; the default activation is
                    SiLU, so that the density is smooth (no kinks).
    n_nodes       : Gauss-Legendre nodes per bin (8 is accurate to < 3e-5).
    threshold     : multiply the a_i by the threshold factor phi(m).
    input_map     : 'sqrt' (default) or 'linear': the network's input variable;
                    see network_input.
    edges         : bin edges in GeV (default: the 80 bins of 15 GeV).
    n_coeffs, monomial_fn : for tests with other functional forms only.
    '''
    def __init__(self, width=WIDTH, hidden=HIDDEN, activation=DENSITY_ACTIVATION,
                 n_nodes=8, threshold=True, input_map='sqrt', edges=EDGES,
                 n_coeffs=23, monomial_fn=monomials):
        super().__init__(width=width, hidden=hidden, activation=activation,
                         n_coeffs=n_coeffs, monomial_fn=monomial_fn)
        edges = np.asarray(edges, dtype=np.float64)
        m, w = bin_nodes(edges, n_nodes)
        # register_buffer: tensors that move with model.to(device) and are
        # saved with the parameters, but are not trained
        self.register_buffer('m_nodes', m.float())
        self.register_buffer('w_nodes', w.float())
        self.register_buffer('edges', torch.tensor(edges, dtype=torch.float32))
        self.n_bins    = len(edges) - 1
        self.n_nodes   = n_nodes
        self.threshold = threshold
        assert input_map in ('sqrt', 'linear'), input_map
        self.input_map = input_map

    # ---- the density --------------------------------------------------------
    def phi(self, m):
        '''Threshold factor sqrt(1 - 4 m_h^2/m^2) (0 below threshold).'''
        return torch.sqrt(torch.clamp(1 - 4 * M_H**2 / m**2, min=0.0))

    def network_input(self, m):
        '''
        The network's input variable for the masses m (GeV).
        'sqrt'  : z = 2 sqrt((m - 250)/(1455 - 250)) - 1, from -0.87 at 255 GeV to
                  1 at 1455 GeV. Like the spline knots, it stretches the threshold
                  region, where the functions change fastest: 255-435 GeV is a
                  third of the range of z, against 12% of the range of x below.
        'linear': x = (m - 255)/1500, the input of HEFTNet.
        '''
        if self.input_map == 'sqrt':
            return 2 * torch.sqrt(torch.clamp(m - 2 * M_H, min=0.0) / (EDGES[-1] - 2 * M_H)) - 1
        return (m - M_LO) / M_SCALE

    def density_coeffs(self, m):
        '''The 23 functions a_i(m) in pb/GeV at the masses m (GeV): [len(m), 23].'''
        a = self.coeffs(self.network_input(m))           # P * exp(Q)
        if self.threshold:
            a = a * self.phi(m)[:, None]                 # [M, 1] broadcasts over the 23 columns
        return a

    def bin_integrals(self):
        '''The table A [n_bins, 23] of the integrals of the a_i over the bins (pb).'''
        a = self.density_coeffs(self.m_nodes) * self.w_nodes[:, None]
        # rows are ordered bin by bin, n_nodes per bin: sum over the nodes of each bin
        return a.view(self.n_bins, self.n_nodes, self.n_coeffs).sum(dim=1)

    # ---- the smoothness (curvature) penalty ---------------------------------
    def curvature_penalty(self, n_points=1201):
        '''
        The mean over the functions of the relative curvature of the network's
        own part of each coefficient function, g_i = P_i e^{Q_i} (that is,
        a_i without the threshold factor, which is physics):

            R = (1/n) sum_i  sum_j (d^2 g_i/dz^2 at z_j)^2 / sum_j g_i(z_j)^2,

        on n_points equally spaced values z_j of the network's input variable
        from 255 GeV to 1455 GeV, the second derivative by second differences.
        With the default input z ~ sqrt(m - 250 GeV), this is the variable in
        which the spline model's smoothness prior acts (its knots are uniform
        in sqrt(m - 250)). The denominator is held fixed (detach), so the
        penalty discourages curvature without shrinking the functions.
        Multiply by a strength lambda and add it to the loss (finetune.py,
        argument curvature). See docs/density/density_plan.pdf, section 6.
        '''
        p = next(self.parameters())
        lo = float(self.network_input(torch.tensor([M_LO], dtype=torch.float64)))
        hi = float(self.network_input(torch.tensor([EDGES[-1]], dtype=torch.float64)))
        z = torch.linspace(lo, hi, n_points, dtype=p.dtype, device=p.device)
        h = (hi - lo) / (n_points - 1)
        g = self.coeffs(z)                                    # [n_points, n_coeffs]
        d2 = (g[2:] - 2 * g[1:-1] + g[:-2]) / h**2            # second differences / h^2
        return ((d2**2).sum(0) / (g.detach()**2).sum(0)).mean()

    # ---- the prediction for training rows -----------------------------------
    def forward(self, x):
        # x.shape: [N, 6]; x[:, 0] = (m - 255 GeV)/1500 GeV at the centre of a
        # bin. The bin index k is found from the bin edges: bucketize returns,
        # for each m, the number of interior edges at or below it, which is the
        # index of the bin that contains m (points outside the range go to the
        # first or last bin). This works for any binning, e.g. 30 GeV bins.
        m = M_LO + M_SCALE * x[:, 0]
        k = torch.bucketize(m, self.edges[1:-1].to(m.dtype), right=True)
        A = self.bin_integrals()                          # [n_bins, 23]
        C = self.monomial_fn(x[:, 1:])                    # [N, 23]
        return (C * A[k]).sum(dim=1)                      # A[k]: row k of A for each input row

    # ---- using the model ----------------------------------------------------
    def _tensor(self, v):
        p = next(self.parameters())
        return torch.as_tensor(np.asarray(v, dtype=np.float64), dtype=p.dtype, device=p.device)

    def dsigma_dm(self, theta, m):
        '''
        d sigma/dm (pb/GeV) at the masses m (GeV) for each HEFT point theta
        (array [5] or [N, 5]): a numpy array [N, len(m)].
        '''
        with torch.no_grad():
            th = self._tensor(np.atleast_2d(theta))
            a = self.density_coeffs(self._tensor(np.atleast_1d(m)))   # [M, 23]
            return (self.monomial_fn(th) @ a.T).cpu().numpy()          # [N, 23] @ [23, M]

    def coefficients(self, m):
        '''The 23 functions a_i(m) (pb/GeV) at the masses m: a numpy array [len(m), 23].'''
        with torch.no_grad():
            return self.density_coeffs(self._tensor(np.atleast_1d(m))).cpu().numpy()

    def spectrum(self, theta):
        '''The cross section (pb) in each bin for each HEFT point: a numpy array [N, n_bins].'''
        with torch.no_grad():
            th = self._tensor(np.atleast_2d(theta))
            return (self.monomial_fn(th) @ self.bin_integrals().T).cpu().numpy()
