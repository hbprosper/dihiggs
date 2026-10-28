"""
heftnet_numpy.py -- the dihiggs HEFT di-Higgs model, re-implemented with numpy only.

What the model computes
-----------------------
For a di-Higgs invariant mass m_hh (in GeV) and the five HEFT Wilson coefficients

    theta = (klambda, ct, ctt, cggh, cgghh)

the model returns the gluon-fusion di-Higgs cross section, in pb, in a 15 GeV wide
bin centred at m_hh (proton-proton collisions at 13 TeV, POWHEG NLO). Divide by 15
to get dsigma/dm_hh in pb/GeV.

The cross section is the dot product of two 23-component vectors,

    sigma(m_hh, theta) = sum_i  c_i(theta) * a_i(m_hh),

where
  * c_i(theta) are 23 KNOWN monomials in the Wilson coefficients (see monomials()),
  * a_i(m_hh) are 23 functions of the mass, learned by a small neural network:

        a_i = P_i(x) * exp(Q_i(x)),      x = (m_hh - 255 GeV) / 1500 GeV,

    P is a fully connected network (ReLU hidden layers, tanh output) and
    Q is a single linear layer.

See docs/dihiggs_model.pdf in https://github.com/hbprosper/dihiggs for the physics.

Usage
-----
    from heftnet_numpy import HEFTNet

    model = HEFTNet("heftnet_weights.txt")
    sigma = model(400.0, [1, 1, 0, 0, 0])              # one point (SM)
    mhh   = np.arange(262.5, 1455, 15)                 # all 80 bin centres
    spec  = model(mhh, [1, 1, 0, 0, 0])                # a whole spectrum
    sig   = model(mhh_array, theta_array)              # [N] and [N, 5] -> [N]

Only numpy and the Python standard library are needed.
"""
import warnings
import numpy as np

FORMAT = "dihiggs-heftnet-text-v1"   # weights file format this code understands
N_COEFFS = 23                        # number of monomials / learned functions


# ---------------------------------------------------------------------------
# Reading the weights file
# ---------------------------------------------------------------------------
def read_weights(filename):
    """
    Read a plain-text weights file written by dihiggs.convert.

    The file is a sequence of whitespace-separated tokens; everything after a
    '#' on a line is a comment. It contains

        key value                          header entries, e.g. "activation relu"
        tensor NAME NDIM D1 ... DNDIM      followed by D1*...*DNDIM numbers,
                                           in row-major (C) order

    Returns (header, tensors): two dictionaries, the second mapping each tensor
    name to a numpy array of the given shape.
    """
    tokens = []
    with open(filename) as f:
        for line in f:
            tokens.extend(line.split("#", 1)[0].split())

    header, tensors = {}, {}
    it = iter(tokens)
    for key in it:
        if key == "tensor":
            name = next(it)
            ndim = int(next(it))
            shape = tuple(int(next(it)) for _ in range(ndim))
            size = int(np.prod(shape))
            values = [float(next(it)) for _ in range(size)]
            tensors[name] = np.array(values, dtype=np.float64).reshape(shape)
        else:
            header[key] = next(it)
    return header, tensors


# the activation functions the model may use between hidden layers
ACTIVATIONS = {
    "relu": lambda z: np.maximum(z, 0.0),
    "tanh": np.tanh,
    "sin":  np.sin,
}


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------
class HEFTNet:
    """The HEFT di-Higgs cross-section model (numpy version)."""

    def __init__(self, filename):
        header, tensors = read_weights(filename)

        if header.get("format") != FORMAT:
            raise ValueError(f"{filename}: expected format {FORMAT}, "
                             f"found {header.get('format')}")
        if int(header["n_coeffs"]) != N_COEFFS:
            raise ValueError(f"{filename}: expected {N_COEFFS} coefficients")

        name = header["activation"]
        if name not in ACTIVATIONS:
            raise ValueError(f"{filename}: unknown activation '{name}'; "
                             f"known: {sorted(ACTIVATIONS)}")
        self.activation = ACTIVATIONS[name]

        # The mass mapping used in training: x = (m_hh - mhh_min) / mhh_scale.
        # The model was trained on mhh_min <= m_hh < mhh_max.
        self.mhh_min = float(header["mhh_min_GeV"])
        self.mhh_max = float(header["mhh_max_GeV"])
        self.mhh_scale = float(header["mhh_scale_GeV"])

        # Network P: a list of (W, b) pairs, one per fully connected layer.
        # W has shape [n_out, n_in], b has shape [n_out].
        n_layers = int(header["p_layers"])
        self.P = [(tensors[f"P.{i}.weight"], tensors[f"P.{i}.bias"])
                  for i in range(n_layers)]

        # Network Q: a single linear layer, 1 input -> 23 outputs.
        self.Q = (tensors["Q.weight"], tensors["Q.bias"])

    # -- the mass variable ---------------------------------------------------
    def x_from_mhh(self, mhh):
        """Map m_hh in GeV to the dimensionless input x used in training."""
        mhh = np.asarray(mhh, dtype=np.float64)
        if np.any((mhh < self.mhh_min) | (mhh > self.mhh_max)):
            warnings.warn(f"m_hh outside the training range "
                          f"[{self.mhh_min:g}, {self.mhh_max:g}] GeV; "
                          "the model is extrapolating", stacklevel=3)
        return (mhh - self.mhh_min) / self.mhh_scale

    # -- the learned functions a_i(m_hh) -------------------------------------
    def coeffs(self, mhh):
        """
        Return the 23 learned functions a_i(m_hh), shape [N, 23],
        for masses mhh (GeV) of shape [N] (or a scalar).
        """
        # column vector of inputs, shape [N, 1]; each row is one mass
        x = self.x_from_mhh(np.atleast_1d(mhh)).reshape(-1, 1)

        # P: every layer is h -> h W^T + b, applied to all N rows at once.
        # Hidden layers are followed by the activation, the last one by tanh.
        h = x
        last = len(self.P) - 1
        for i, (W, b) in enumerate(self.P):
            h = h @ W.T + b
            h = np.tanh(h) if i == last else self.activation(h)

        # Q: one linear layer
        W, b = self.Q
        q = x @ W.T + b

        return h * np.exp(q)                     # a_i = P_i * exp(Q_i)

    # -- the known monomials c_i(theta) --------------------------------------
    @staticmethod
    def monomials(theta):
        """
        Return the 23 monomials c_i(theta), shape [N, 23], for Wilson
        coefficients theta = (klambda, ct, ctt, cggh, cgghh) of shape [N, 5]
        (or [5] for a single point).

        The order is that of Eq. (4.1) of arXiv:2304.01968 and of Table 2 of
        docs/dihiggs_model.pdf. Terms 1-15 suffice at leading order; terms
        16-23 appear at next-to-leading order.
        """
        theta = np.atleast_2d(np.asarray(theta, dtype=np.float64))
        kl, ct, ctt, cggh, cgghh = theta.T       # each of shape [N]

        return np.column_stack((
            ct**4,                   #  1
            ctt**2,                  #  2
            ct**2 * kl**2,           #  3
            cggh**2 * kl**2,         #  4
            cgghh**2,                #  5
            ctt * ct**2,             #  6
            kl * ct**3,              #  7
            ct * kl * ctt,           #  8
            cggh * kl * ctt,         #  9
            ctt * cgghh,             # 10
            cggh * kl * ct**2,       # 11
            cgghh * ct**2,           # 12
            kl**2 * cggh * ct,       # 13
            cgghh * ct * kl,         # 14
            cggh * cgghh * kl,       # 15
            ct**3 * cggh,            # 16  (NLO)
            ct * ctt * cggh,         # 17  (NLO)
            ct * cggh**2 * kl,       # 18  (NLO)
            ct * cggh * cgghh,       # 19  (NLO)
            ct**2 * cggh**2,         # 20  (NLO)
            ctt * cggh**2,           # 21  (NLO)
            cggh**3 * kl,            # 22  (NLO)
            cggh**2 * cgghh,         # 23  (NLO)
        ))

    # -- the cross section ---------------------------------------------------
    def __call__(self, mhh, theta):
        """
        Cross section in pb per 15 GeV bin centred at mhh (GeV).

        mhh:   scalar or array of shape [N]
        theta: (klambda, ct, ctt, cggh, cgghh), shape [5] or [N, 5]

        A single theta is used for every mass, and a single mass for every
        theta. Returns an array of shape [N], or a float if both inputs
        describe a single point.
        """
        scalar = np.ndim(mhh) == 0 and np.ndim(theta) == 1

        a = self.coeffs(mhh)                     # [N_m, 23]
        c = self.monomials(theta)                # [N_t, 23]
        a, c = np.broadcast_arrays(a, c)         # a single row is repeated

        # row-by-row dot product: sigma_n = sum_i c_ni * a_ni
        sigma = np.einsum("ni,ni->n", c, a)
        return float(sigma[0]) if scalar else sigma
