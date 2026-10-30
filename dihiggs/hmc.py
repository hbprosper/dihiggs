# ----------------------------------------------------------------------------
# hmc.py: Bayesian neural networks sampled with Hamiltonian Monte Carlo (HMC),
#         following Radford M. Neal.
#
#   R. M. Neal, "Bayesian Learning for Neural Networks", Springer (1996)
#   R. M. Neal, "MCMC using Hamiltonian dynamics", arXiv:1206.1901 (2011)
#   M. D. Hoffman and A. Gelman, JMLR 15 (2014) 1593 (step-size adaptation)
#
# The design is described in docs/bnn/bnn_plan.pdf. In brief:
#
#   ParameterView      finds the trainable parameters of ANY PyTorch model and
#                      converts between the model's tensors and one flat vector
#                      w (the "position" of the HMC particle).
#   HierarchicalPrior  Neal's prior: a Gaussian of unknown precision tau_g for
#                      each group g of parameters, with a Gamma hyperprior.
#   LaplaceLikelihood, GaussianLikelihood, PoissonLikelihood
#                      how the targets scatter about the model's prediction.
#   Posterior          the potential energy U(w) = -log(posterior) and its
#                      gradient, computed with PyTorch's autograd over ALL rows.
#   HMC                the sampler: leapfrog trajectories, accept/reject,
#                      Gibbs updates of the hyperparameters, step-size and mass
#                      adaptation during warm-up, sample files, checkpoints.
#   BNN                reads the sampled parameter files and computes the
#                      posterior mean, standard deviation, quantiles, etc.
#   autocorrelation, ess, rhat
#                      convergence diagnostics.
#
# This module deliberately imports nothing from dihiggs, so that it can be
# moved to drwho unchanged.
#
# Harrison B. Prosper and Claude
# Created: Thu Oct 29 2026
# ----------------------------------------------------------------------------
import os, csv, glob, math, time
import numpy as np
import torch
import torch.nn as nn

__all__ = ['ParameterView', 'HierarchicalPrior',
           'LaplaceLikelihood', 'GaussianLikelihood', 'PoissonLikelihood',
           'Posterior', 'HMC', 'BNN',
           'autocorrelation', 'ess', 'rhat']

# ----------------------------------------------------------------------------
# A note on three PyTorch idioms used throughout this file
#
#   with torch.no_grad():   Inside this block PyTorch does NOT record the
#                           operations for later differentiation. We use it
#                           whenever we change parameters by hand or only want
#                           numbers, not gradients. It saves time and memory.
#
#   t.detach()              Returns the same numbers as tensor t but cut off
#                           from the record of how t was computed, so that
#                           nothing done to the result can affect gradients.
#
#   x.copy_(y)              A trailing underscore means "in place": the numbers
#                           of y are copied INTO the existing tensor x. Without
#                           the underscore, PyTorch functions return new tensors.
# ----------------------------------------------------------------------------


# ============================================================================
# 1. Finding the parameters of a model
# ============================================================================
class ParameterView:
    '''
    A "view" of the trainable parameters of a PyTorch model as one flat vector.

    HMC treats all the parameters of a network as the coordinates of a single
    point w in a space of n dimensions (n = 3,944 for HEFTNet). PyTorch, on
    the other hand, keeps the parameters in many tensors of different shapes
    (one weight matrix and one bias vector per linear layer). This class
    translates between the two.

    It needs nothing from the model except what every nn.Module provides:
    the method named_parameters(), which lists the model's parameters by name.

    Arguments
    ---------
    model    : the PyTorch model (an instance of nn.Module).
    grouping : how the parameters are grouped for Neal's prior.
               'tensor' (default) - each parameter tensor is its own group.
                                    For a linear layer this means one group
                                    for its weights and one for its biases,
                                    which is Neal's choice.
               'all'              - all parameters form a single group.
               a dict             - {parameter name: group name}; every
                                    parameter must appear.
    '''
    # Layers that make the output random, or make it depend on the other rows
    # of a batch. HMC needs the model to be a fixed (deterministic) function
    # of its inputs and parameters, so models containing these are refused.
    # (The names starting with "_" are PyTorch's base classes for all dropout
    # and all batch-normalisation layers.)
    FORBIDDEN = (nn.modules.dropout._DropoutNd,
                 nn.modules.batchnorm._BatchNorm)

    def __init__(self, model, grouping='tensor'):
        self.model = model

        # --- 1. refuse models HMC cannot handle ----------------------------
        # model.modules() walks through the model and all its sub-modules.
        for module in model.modules():
            if isinstance(module, self.FORBIDDEN):
                raise TypeError(
                    f'ParameterView: the model contains a '
                    f'{type(module).__name__} layer. HMC needs a model whose '
                    f'output is a fixed function of its inputs and parameters;'
                    f' please remove dropout and batch-normalisation layers.')

        # --- 2. list the trainable parameters -------------------------------
        # named_parameters() yields (name, tensor) pairs. For HEFTNet:
        #   ('P.0.weight', shape [25, 1]), ('P.0.bias', shape [25]),
        #   ('P.2.weight', shape [25, 25]), ...,  ('Q.bias', shape [23])
        # The name says where the tensor lives: 'P.0.weight' is the weight
        # matrix of item 0 of the nn.Sequential called P.
        self.names  = []
        self.params = []
        for name, p in model.named_parameters():
            if p.requires_grad:          # skip parameters frozen by the user
                self.names.append(name)
                self.params.append(p)

        if len(self.params) == 0:
            raise ValueError('ParameterView: the model has no trainable '
                             'parameters')

        # p.numel() is the number of elements of tensor p, e.g. 25*25 = 625.
        self.shapes = [tuple(p.shape) for p in self.params]
        self.sizes  = [p.numel() for p in self.params]
        self.n      = sum(self.sizes)     # total number of parameters

        # Where each tensor sits inside the flat vector w. A Python slice
        # object slice(a, b) selects elements a, a+1, ..., b-1, so w[s] is
        # the part of w that belongs to that tensor.
        ends = np.cumsum(self.sizes)
        self.slices = [slice(int(e) - n, int(e))
                       for e, n in zip(ends, self.sizes)]

        # --- 3. form the groups of Neal's prior -----------------------------
        if grouping == 'tensor':
            group_of = {name: name for name in self.names}
        elif grouping == 'all':
            group_of = {name: 'all' for name in self.names}
        elif isinstance(grouping, dict):
            missing = [name for name in self.names if name not in grouping]
            if missing:
                raise KeyError(f'ParameterView: no group given for {missing}')
            group_of = grouping
        else:
            raise ValueError(f"ParameterView: unknown grouping {grouping!r}")

        # self.groups: a list with one dictionary per group, in the order in
        # which the groups first appear in the model.
        self.groups = []
        index_of_group = {}
        for name, shape, s in zip(self.names, self.shapes, self.slices):
            gname = group_of[name]
            if gname not in index_of_group:
                index_of_group[gname] = len(self.groups)
                self.groups.append({'name': gname, 'slices': [],
                                    'size': 0, 'fan_in': None,
                                    'is_bias': True})
            g = self.groups[index_of_group[gname]]
            g['slices'].append(s)
            g['size'] += s.stop - s.start
            # A weight matrix of a linear layer has shape [n_out, n_in], so
            # its "fan-in" (the number of inputs feeding each unit) is the
            # product of all dimensions after the first. A bias (1 dimension)
            # has no fan-in.
            if len(shape) >= 2:
                g['is_bias'] = False
                g['fan_in'] = int(np.prod(shape[1:]))

        # group_index[i] = number of the group to which element i of w
        # belongs. It lets us apply the group precisions to w in one step.
        group_index = torch.empty(self.n, dtype=torch.long)
        for k, g in enumerate(self.groups):
            for s in g['slices']:
                group_index[s] = k
        self.group_index = group_index

    # ------------------------------------------------------------------------
    @property
    def device(self):
        # the device (CPU or GPU) on which the model's parameters live
        return self.params[0].device

    @property
    def dtype(self):
        return self.params[0].dtype

    def get(self):
        '''
        Return a copy of all the model's parameters as one flat vector w.
        '''
        with torch.no_grad():
            # p.reshape(-1) lays out the elements of p in a single row
            # (-1 means "as many as needed"); torch.cat joins those rows.
            return torch.cat([p.reshape(-1) for p in self.params]).clone()

    def set(self, w):
        '''
        Copy the flat vector w into the model's parameters, in place.
        '''
        with torch.no_grad():
            for p, s in zip(self.params, self.slices):
                # w[s] is a 1-dimensional piece of w; view_as(p) gives it the
                # shape of p (e.g. [25, 25]) without copying it.
                p.copy_(w[s].view_as(p))

    def grad(self):
        '''
        Return the gradients currently stored in the parameters (p.grad, which
        PyTorch fills when .backward() is called) as one flat vector, in the
        same order as w.
        '''
        pieces = []
        for p in self.params:
            if p.grad is None:          # parameter not used in the output
                pieces.append(torch.zeros(p.numel(), device=p.device,
                                          dtype=p.dtype))
            else:
                pieces.append(p.grad.reshape(-1))
        return torch.cat(pieces).detach().clone()

    def zero_grad(self):
        # forget gradients from an earlier calculation
        for p in self.params:
            p.grad = None

    def summary(self):
        '''
        Return a printable table of the parameters and their groups.
        '''
        lines = [f'{"parameter":<22s} {"shape":<12s} {"size":>7s}   group',
                 '-' * 60]
        for name, shape, size in zip(self.names, self.shapes, self.sizes):
            k = int(self.group_index[self.slices[self.names.index(name)].start])
            lines.append(f'{name:<22s} {str(list(shape)):<12s} {size:7d}   {k + 1}')
        lines.append('-' * 60)
        lines.append(f'{"total":<35s}{self.n:7d}   '
                     f'{len(self.groups)} groups')
        return '\n'.join(lines)


# ============================================================================
# 2. Neal's hierarchical prior
# ============================================================================
class HierarchicalPrior:
    '''
    For each group g of parameters (see ParameterView):

        w_g | tau_g  ~  Normal(0, 1/tau_g)      (each element independently)
        tau_g        ~  Gamma(shape=alpha, rate=beta_g)

    tau_g is the PRECISION (1/variance) of the group. The Gamma prior is
    chosen so that its mean, alpha/beta_g, equals 1/sigma_g^2, where

        sigma_g = 1/sqrt(fan-in)   for weights (Neal's scaling),
        sigma_g = 1                for biases.

    Arguments
    ---------
    view   : a ParameterView.
    alpha  : shape of the Gamma hyperprior (default 1, a broad prior).
    sigmas : optional list of prior widths sigma_g, one per group, to
             replace the defaults above.
    fixed  : if True, the precisions are kept at their prior means and never
             updated (a non-hierarchical prior).
    '''
    def __init__(self, view, alpha=1.0, sigmas=None, fixed=False):
        self.view  = view
        self.alpha = float(alpha)
        self.fixed = fixed

        G = len(view.groups)
        if sigmas is None:
            sigmas = []
            for g in view.groups:
                if g['is_bias'] or g['fan_in'] is None:
                    sigmas.append(1.0)
                else:
                    sigmas.append(1.0 / math.sqrt(g['fan_in']))
        if len(sigmas) != G:
            raise ValueError(f'HierarchicalPrior: need {G} sigmas')

        self.sigmas = np.array(sigmas, dtype=np.float64)
        prior_mean  = 1.0 / self.sigmas**2      # prior mean of each tau_g
        self.beta   = self.alpha / prior_mean   # rate, so alpha/beta = mean
        self.sizes  = np.array([g['size'] for g in view.groups])

        # current precisions: start at the prior means
        self.tau = prior_mean.copy()

    # ------------------------------------------------------------------------
    def precision_vector(self, device, dtype):
        '''
        The precision of every element of w: tau[group of that element].
        '''
        tau = torch.as_tensor(self.tau, device=device, dtype=dtype)
        return tau[self.view.group_index.to(device)]

    def neg_log(self, w):
        '''
        Minus the log of the prior density of w, up to terms that do not
        depend on w:   sum_g tau_g * |w_g|^2 / 2.
        Returns (value as a Python float, its gradient tau * w as a tensor).
        '''
        prec = self.precision_vector(w.device, w.dtype)
        grad = prec * w
        # .double() converts to 64-bit floats for an accurate sum
        value = 0.5 * float(torch.sum(prec.double() * w.double()**2))
        return value, grad

    def sum_squares(self, w):
        # |w_g|^2 for every group g, as a numpy array
        w2 = (w.detach().double()**2).cpu()
        index = self.view.group_index
        # index_add_ adds w2[i] to element index[i] of the result
        ss = torch.zeros(len(self.view.groups), dtype=torch.float64)
        ss.index_add_(0, index, w2)
        return ss.numpy()

    def from_weights(self, w):
        '''
        Set each tau_g to the precision suggested by the weights themselves,
        tau_g = n_g / |w_g|^2 (used when starting from a trained model).
        '''
        if self.fixed:
            return
        ss = self.sum_squares(w)
        self.tau = self.sizes / np.maximum(ss, 1e-30)

    def gibbs(self, w, rng):
        '''
        Draw new precisions from their exact conditional density:

            tau_g | w_g ~ Gamma(shape = alpha + n_g/2,
                                rate  = beta_g + |w_g|^2/2)

        (see the box on Gibbs sampling in docs/bnn/bnn_plan.pdf).
        rng is a numpy random Generator. Note that numpy's gamma() is
        parametrised by the SCALE, which is 1/rate.
        '''
        if self.fixed:
            return
        ss    = self.sum_squares(w)
        shape = self.alpha + self.sizes / 2
        rate  = self.beta + ss / 2
        self.tau = rng.gamma(shape, 1.0 / rate)

    def state(self):
        return {'tau': self.tau.copy()}

    def load_state(self, state):
        self.tau = np.array(state['tau'], dtype=np.float64)


# ============================================================================
# 3. Likelihoods
# ============================================================================
# Each likelihood provides
#   neg_log(f, t)  minus the log-likelihood summed over the rows given,
#                  keeping only terms that depend on the predictions f;
#   statistic(f, t) the summed quantity needed for the Gibbs update of the
#                  noise scale (e.g. sum |t - f| for the Laplace likelihood);
#   gibbs(stat, n, rng)  draws a new noise scale;
#   noise(f, generator)  simulated noise, for predictive bands.
#
# The targets t may have several columns. Column 0 is always the value to be
# predicted; PoissonLikelihood reads the known noise variance from column 1.
# ----------------------------------------------------------------------------
def _target(t):
    # column 0 of the targets, as a 1-dimensional tensor
    return t[:, 0] if t.ndim == 2 else t


class LaplaceLikelihood:
    '''
    p(t | f, b) = exp(-|t - f| / b) / (2b)      (the counterpart of L1 loss)

    The scale b is unknown and Gibbs-sampled. Its prior is
    InvGamma(alpha, beta); the default alpha = beta = 0 is the scale-invariant
    (Jeffreys) prior p(b) ~ 1/b, under which

        b | w, data ~ InvGamma(shape = N, rate = sum_i |t_i - f_i|).

    Set learn=False to keep b fixed at the value given.
    '''
    name = 'laplace'

    def __init__(self, scale=1.0, alpha=0.0, beta=0.0, learn=True):
        self.scale = float(scale)
        self.alpha, self.beta, self.learn = float(alpha), float(beta), learn

    def neg_log(self, f, t):
        return torch.sum(torch.abs(_target(t) - f)) / self.scale

    def statistic(self, f, t):
        return torch.sum(torch.abs(_target(t) - f))

    def gibbs(self, stat, n, rng):
        if not self.learn:
            return
        # If g ~ Gamma(shape, rate) then 1/g ~ InvGamma(shape, rate).
        g = rng.gamma(self.alpha + n, 1.0 / (self.beta + stat))
        self.scale = 1.0 / g

    def initial_scale(self, stat, n):
        # the most probable b for given residuals: mean |t - f|
        return stat / n

    def noise(self, f, generator=None):
        # A Laplace variate is the difference of two exponential variates.
        e1 = torch.empty_like(f).exponential_(generator=generator)
        e2 = torch.empty_like(f).exponential_(generator=generator)
        return self.scale * (e1 - e2)

    def state(self):
        return {'scale': self.scale}

    def load_state(self, s):
        self.scale = float(s['scale'])


class GaussianLikelihood:
    '''
    p(t | f, s) = Normal(t; f, s^2)            (the counterpart of MSE loss)

    The variance s^2 (stored as self.scale = s) is Gibbs-sampled, with prior
    InvGamma(alpha, beta) on s^2; with the default alpha = beta = 0,

        s^2 | w, data ~ InvGamma(shape = N/2, rate = sum_i (t_i - f_i)^2 / 2).

    Set learn=False to keep s fixed at the value given (used in the toy check,
    where the noise is known).
    '''
    name = 'gaussian'

    def __init__(self, scale=1.0, alpha=0.0, beta=0.0, learn=True):
        self.scale = float(scale)
        self.alpha, self.beta, self.learn = float(alpha), float(beta), learn

    def neg_log(self, f, t):
        return torch.sum((_target(t) - f)**2) / (2 * self.scale**2)

    def statistic(self, f, t):
        return torch.sum((_target(t) - f)**2)

    def gibbs(self, stat, n, rng):
        if not self.learn:
            return
        g = rng.gamma(self.alpha + n / 2, 1.0 / (self.beta + stat / 2))
        self.scale = math.sqrt(1.0 / g)

    def initial_scale(self, stat, n):
        return math.sqrt(stat / n)

    def noise(self, f, generator=None):
        return self.scale * torch.randn(f.shape, generator=generator,
                                        device=f.device, dtype=f.dtype)

    def state(self):
        return {'scale': self.scale}

    def load_state(self, s):
        self.scale = float(s['scale'])


class PoissonLikelihood:
    '''
    p(t | f) = Normal(t; f, v),  v = the KNOWN noise variance of each target,
    read from column 1 of the targets (for a POWHEG bin, v = w_j * t, the
    variance of a histogram of events of weight w_j). There is no unknown
    noise parameter. EXPERIMENTAL: the corresponding CHI2 loss trained poorly.
    '''
    name = 'poisson'

    def __init__(self):
        self.scale = 1.0       # not used; kept so all likelihoods look alike
        self.learn = False

    def neg_log(self, f, t):
        return torch.sum((t[:, 0] - f)**2 / (2 * t[:, 1]))

    def statistic(self, f, t):
        return torch.sum((t[:, 0] - f)**2 / t[:, 1])     # chi-squared

    def gibbs(self, stat, n, rng):
        return

    def initial_scale(self, stat, n):
        return 1.0

    def noise(self, f, generator=None):
        raise NotImplementedError('PoissonLikelihood: predictive noise needs '
                                  'the variance at each point')

    def state(self):
        return {'scale': self.scale}

    def load_state(self, s):
        pass


# ============================================================================
# 4. The posterior: potential energy U(w) and its gradient
# ============================================================================
class Posterior:
    '''
    U(w) = -log p(data | w) - log prior(w)   (up to a constant), i.e.

        U(w) = likelihood.neg_log(f(x; w), t)  summed over ALL rows
             + sum_g tau_g |w_g|^2 / 2.

    Arguments
    ---------
    model      : the PyTorch model; its forward(x) must return one
                 prediction per row of x.
    likelihood : one of the likelihood objects above.
    prior      : a HierarchicalPrior (which holds the ParameterView).
    x, t       : all the training inputs and targets, as tensors on the
                 device on which the model lives.
    chunk_size : the rows are processed in chunks of this many, to limit the
                 memory used. The gradients of the chunks add up to exactly
                 the gradient over all rows.
    '''
    def __init__(self, model, likelihood, prior, x, t, chunk_size=65536):
        self.model      = model
        self.likelihood = likelihood
        self.prior      = prior
        self.view       = prior.view
        self.x, self.t  = x, t
        self.N          = len(x)
        self.chunk_size = int(chunk_size)
        model.eval()   # no dropout etc. (ParameterView has checked anyway)

    def _chunks(self):
        # yield (x, t) for consecutive blocks of rows
        for start in range(0, self.N, self.chunk_size):
            end = start + self.chunk_size
            yield self.x[start:end], self.t[start:end]

    def U_and_grad(self, w):
        '''
        Return (U, gradient of U, statistic):
          U          a Python float (computed in double precision),
          gradient   a tensor shaped like w,
          statistic  the likelihood's summed statistic (a Python float),
                     needed for the Gibbs update of the noise scale.
        '''
        self.view.set(w)          # put w into the model
        self.view.zero_grad()     # forget gradients of the previous call
        U_lik, stat = 0.0, 0.0
        for xc, tc in self._chunks():
            # model(xc) has shape [chunk] or [chunk, 1]; reshape(-1) makes
            # it [chunk]. .double() converts to 64-bit floats, so that the sum
            # over many rows keeps enough digits for the accept/reject step.
            f = self.model(xc).reshape(-1).double()
            nll = self.likelihood.neg_log(f, tc.double())
            # backward() computes d(nll)/d(parameter) for every parameter
            # and ADDS it to p.grad; over all chunks this gives the gradient
            # of the full sum.
            nll.backward()
            U_lik += float(nll.detach())
            stat  += float(self.likelihood.statistic(f.detach(),
                                                     tc.double()))
        U_prior, grad_prior = self.prior.neg_log(w)
        grad = self.view.grad() + grad_prior
        return U_lik + U_prior, grad, stat

    def U(self, w):
        '''
        U(w) only, without the gradient (cheaper).
        '''
        self.view.set(w)
        U_lik = 0.0
        with torch.no_grad():
            for xc, tc in self._chunks():
                f = self.model(xc).reshape(-1).double()
                U_lik += float(self.likelihood.neg_log(f, tc.double()))
        return U_lik + self.prior.neg_log(w)[0]

    def time_gradient(self, w, repeats=3):
        '''
        Average time in seconds of one call to U_and_grad.
        '''
        self.U_and_grad(w)                   # warm-up call (not timed)
        self._sync()
        start = time.time()
        for _ in range(repeats):
            self.U_and_grad(w)
        self._sync()
        return (time.time() - start) / repeats

    def _sync(self):
        # GPU work runs asynchronously; wait for it to finish before timing
        if self.x.is_cuda:
            torch.cuda.synchronize()


# ============================================================================
# 5. The HMC sampler
# ============================================================================
class HMC:
    '''
    Hamiltonian Monte Carlo for the parameters w, alternated with Gibbs draws
    of the prior precisions tau_g and the likelihood's noise scale.

    One ITERATION is:
      1. draw a momentum p ~ Normal(0, M);
      2. follow L leapfrog steps of size eps;
      3. accept the end point with probability min(1, exp(-dH));
      4. draw new tau_g given w, and a new noise scale given w (Gibbs).

    The first n_warmup iterations tune eps and the (diagonal) mass matrix M
    and are not saved. After that, every thin-th iteration writes the current
    parameters to outdir/samples/params_NNNN.pth.

    Arguments
    ---------
    posterior     : a Posterior.
    outdir        : the folder for this chain (created if needed).
    n_leapfrog    : L, the number of leapfrog steps per trajectory.
    step_size     : eps; if None, a starting value is found automatically.
    target_accept : the acceptance rate aimed at during warm-up (0.8).
    jitter        : eps and L are multiplied by a random factor between
                    1 - jitter and 1 + jitter for each trajectory (0.1).
    adapt_mass    : estimate a diagonal mass matrix during warm-up (True).
    seed          : seed of the random number generators.
    name          : stub for the file names, e.g. 'dihiggs'.
    monitor       : optional function monitor(model) returning a 1-d tensor
                    or list of numbers (e.g. predictions at chosen points),
                    which are written to the trace file after every iteration.
    checkpoint_every : seconds between checkpoints (default 300).
    max_dH        : a trajectory with dH larger than this is called a
                    divergence (rejected and counted).
    '''
    def __init__(self, posterior, outdir, n_leapfrog=100, step_size=None,
                 target_accept=0.8, jitter=0.1, adapt_mass=True, seed=42,
                 name='hmc', monitor=None, checkpoint_every=300,
                 max_dH=1000.0):
        self.post   = posterior
        self.view   = posterior.view
        self.outdir = outdir
        self.L      = int(n_leapfrog)
        self.eps    = step_size
        self.delta  = float(target_accept)
        self.jitter = float(jitter)
        self.adapt_mass = adapt_mass
        self.seed   = seed
        self.name   = name
        self.monitor = monitor
        self.checkpoint_every = checkpoint_every
        self.max_dH = max_dH

        self.sampledir  = os.path.join(outdir, 'samples')
        self.tracefile  = os.path.join(outdir, f'{name}_hmc_trace.csv')
        self.statefile  = os.path.join(outdir, f'{name}_hmc_state.pth')
        self.indexfile  = os.path.join(self.sampledir, 'index.csv')
        os.makedirs(self.sampledir, exist_ok=True)

        # Random numbers. numpy's Generator is used for everything except the
        # momenta, which are drawn with a torch Generator on the CPU and then
        # moved to the device: that way a run gives the same numbers on a CPU
        # and on a GPU.
        self.rng   = np.random.default_rng(seed)
        self.trng  = torch.Generator(device='cpu')
        self.trng.manual_seed(seed)

        # The current state of the chain
        device, dtype = self.view.device, self.view.dtype
        self.w = self.view.get()                                  # position
        self.mass_inv = torch.ones(self.view.n, device=device,
                                   dtype=dtype)                   # M^{-1}
        self.iteration = 0          # iterations completed
        self.n_saved   = 0          # sample files written
        self.n_accept  = 0          # accepted trajectories (sampling phase)
        self.n_diverge = 0          # divergent trajectories (sampling)
        self.n_diverge_warmup = 0   # divergent trajectories (warm-up),
                                    # expected while eps is being tuned
        self.n_warmup  = None       # set by run()
        self._da = None             # dual-averaging variables (see below)
        self._welford = None        # running variance of w (mass matrix)

    # ------------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------------
    def _momentum(self):
        # p ~ Normal(0, M) with M = diag(1/mass_inv): p_i = z_i / sqrt(mass_inv_i)
        z = torch.randn(self.view.n, generator=self.trng)
        z = z.to(device=self.w.device, dtype=self.w.dtype)
        return z / torch.sqrt(self.mass_inv)

    def _kinetic(self, p):
        # K(p) = p^T M^{-1} p / 2, in double precision
        return 0.5 * float(torch.sum(self.mass_inv.double() * p.double()**2))

    def _refresh(self):
        # U, gradient and statistic at the current position
        self.U, self.grad, self.stat = self.post.U_and_grad(self.w)

    # ------------------------------------------------------------------------
    # The leapfrog integrator
    # ------------------------------------------------------------------------
    def leapfrog(self, w, p, grad, eps, L):
        '''
        Follow Hamilton's equations for L steps of size eps, starting at
        position w with momentum p and gradient grad = dU/dw at w.
        Returns the end point (w, p, U, grad, statistic), or None if the
        trajectory blew up (non-finite energy).
        '''
        # half step in momentum
        p = p - 0.5 * eps * grad
        for step in range(L):
            # full step in position: dw/ds = M^{-1} p
            w = w + eps * self.mass_inv * p
            U, grad, stat = self.post.U_and_grad(w)
            if not math.isfinite(U):
                return None
            # full step in momentum, except after the last position step,
            # where only a half step is taken (below)
            if step < L - 1:
                p = p - eps * grad
        # final half step in momentum
        p = p - 0.5 * eps * grad
        return w, p, U, grad, stat

    def trajectory(self, eps, L):
        '''
        One HMC update of w. Returns (accepted?, dH, acceptance probability).
        '''
        p0 = self._momentum()
        H0 = self.U + self._kinetic(p0)
        end = self.leapfrog(self.w, p0, self.grad, eps, L)
        if end is None:
            dH = math.inf
        else:
            w1, p1, U1, g1, s1 = end
            dH = (U1 + self._kinetic(p1)) - H0
        if not math.isfinite(dH):
            dH = math.inf
        # acceptance probability min(1, exp(-dH)); exp(-inf) = 0
        prob = 1.0 if dH <= 0 else math.exp(-dH)
        accepted = (dH < self.max_dH) and (self.rng.random() < prob)
        if accepted:
            self.w, self.U, self.grad, self.stat = w1, U1, g1, s1
        return accepted, dH, prob

    # ------------------------------------------------------------------------
    # Step-size tuning
    # ------------------------------------------------------------------------
    def find_step_size(self, eps=1e-3, max_tries=60):
        '''
        A rough starting value for eps (Hoffman & Gelman 2014, Algorithm 4):
        double or halve eps until a single leapfrog step has an acceptance
        probability on the other side of 1/2.
        '''
        def prob(e):
            p = self._momentum()
            H0 = self.U + self._kinetic(p)
            end = self.leapfrog(self.w, p, self.grad, e, 1)
            if end is None:
                return 0.0
            w1, p1, U1, _, _ = end
            dH = U1 + self._kinetic(p1) - H0
            return 0.0 if not math.isfinite(dH) else min(1.0, math.exp(-dH))

        a = 1.0 if prob(eps) > 0.5 else -1.0      # go up (+1) or down (-1)?
        for _ in range(max_tries):
            pr = prob(eps)
            if (a > 0 and pr <= 0.5) or (a < 0 and pr > 0.5):
                break
            eps = eps * 2.0**a
        self.view.set(self.w)   # leapfrog left a trial point in the model
        return eps

    def _da_start(self, eps):
        # Dual averaging (Hoffman & Gelman 2014, Algorithm 5). The step size
        # is pushed up when the acceptance is above the target and down when
        # below; log_eps_bar is a running average that settles down.
        self._da = {'mu': math.log(10 * eps), 'Hbar': 0.0, 'm': 0,
                    'log_eps': math.log(eps), 'log_eps_bar': 0.0}

    def _da_update(self, prob):
        gamma, t0, kappa = 0.05, 10.0, 0.75     # the paper's constants
        d = self._da
        d['m'] += 1
        m = d['m']
        d['Hbar'] = ((1 - 1 / (m + t0)) * d['Hbar']
                     + (self.delta - prob) / (m + t0))
        d['log_eps'] = d['mu'] - math.sqrt(m) / gamma * d['Hbar']
        w = m**(-kappa)
        d['log_eps_bar'] = w * d['log_eps'] + (1 - w) * d['log_eps_bar']
        self.eps = math.exp(d['log_eps'])

    # ------------------------------------------------------------------------
    # Mass-matrix tuning: running variance of each parameter (Welford)
    # ------------------------------------------------------------------------
    def _welford_add(self, w):
        if self._welford is None:
            z = torch.zeros_like(w, dtype=torch.float64)
            self._welford = {'n': 0, 'mean': z.clone(), 'M2': z.clone()}
        s = self._welford
        x = w.double()
        s['n'] += 1
        delta = x - s['mean']
        s['mean'] += delta / s['n']
        s['M2'] += delta * (x - s['mean'])

    def _welford_mass(self):
        s = self._welford
        n = s['n']
        var = s['M2'] / max(n - 1, 1)
        # Shrink each variance towards the median variance; this guards
        # against a parameter that happened not to move during the window.
        med = torch.median(var)
        var = (n / (n + 5.0)) * var + (5.0 / (n + 5.0)) * med
        var = torch.clamp(var, min=float(med) * 1e-6 + 1e-30)
        self._welford = None
        return var.to(self.w.dtype)

    # ------------------------------------------------------------------------
    # Files
    # ------------------------------------------------------------------------
    def _trace_header(self):
        G = len(self.post.prior.tau)
        cols = ['iteration', 'phase', 'accepted', 'dH', 'prob', 'U', 'eps',
                'L', 'scale'] + [f'tau_{g+1}' for g in range(G)]
        if self.monitor is not None:
            with torch.no_grad():
                self.view.set(self.w)
                n = len(self._monitor_values())
            cols += [f'm_{k}' for k in range(n)]
        return cols

    def _monitor_values(self):
        with torch.no_grad():
            v = self.monitor(self.post.model)
        v = torch.as_tensor(v).reshape(-1).detach().cpu().double().numpy()
        return list(v)

    def _append(self, filename, header, row):
        new = not os.path.exists(filename)
        with open(filename, 'a', newline='') as f:
            writer = csv.writer(f)
            if new:
                writer.writerow(header)
            writer.writerow(row)

    def _save_sample(self):
        self.n_saved += 1
        fname = f'params_{self.n_saved:04d}.pth'
        self.view.set(self.w)
        # state_dict() is the standard PyTorch dictionary of a model's
        # parameters; model.load(file) or model.load_state_dict(torch.load(
        # file)) reads it back.
        torch.save(self.post.model.state_dict(),
                   os.path.join(self.sampledir, fname))
        G = len(self.post.prior.tau)
        header = ['file', 'iteration', 'U', 'scale'] + \
                 [f'tau_{g+1}' for g in range(G)]
        self._append(self.indexfile, header,
                     [fname, self.iteration, f'{self.U:.10g}',
                      f'{self.post.likelihood.scale:.6g}']
                     + [f'{t:.6g}' for t in self.post.prior.tau])

    def snapshot(self):
        '''
        A copy of everything needed to continue the chain exactly. It is
        taken at the END of every iteration, when the state of the chain and
        the files written so far agree. If the run is interrupted part-way
        through an iteration, this (consistent) copy is what is saved.
        clone() and copy() make independent copies, so that later changes to
        the chain do not alter the snapshot.
        '''
        return {
            'w': self.w.detach().cpu().clone(),
            'mass_inv': self.mass_inv.detach().cpu().clone(),
            'eps': self.eps, 'L': self.L, 'iteration': self.iteration,
            'n_saved': self.n_saved, 'n_accept': self.n_accept,
            'n_diverge': self.n_diverge,
            'n_diverge_warmup': self.n_diverge_warmup,
            'n_warmup': self.n_warmup,
            'da': None if self._da is None else dict(self._da),
            'welford': None if self._welford is None else
                       {k: (v.detach().cpu().clone() if torch.is_tensor(v)
                            else v)
                        for k, v in self._welford.items()},
            'prior': self.post.prior.state(),
            'likelihood': self.post.likelihood.state(),
            'rng': self.rng.bit_generator.state,       # a plain dictionary
            'trng': self.trng.get_state(),             # a tensor (a copy)
        }

    def checkpoint(self, state=None):
        '''
        Write a snapshot (by default, a fresh one) to the checkpoint file.
        '''
        if state is None:
            state = self.snapshot()
        tmp = self.statefile + '.tmp'
        torch.save(state, tmp)
        os.replace(tmp, self.statefile)   # never leave a half-written file

    def _load_checkpoint(self):
        # weights_only=False: the checkpoint holds plain Python objects too.
        s = torch.load(self.statefile, map_location='cpu',
                       weights_only=False)
        device = self.view.device
        self.w = s['w'].to(device)
        self.mass_inv = s['mass_inv'].to(device)
        self.eps, self.L = s['eps'], s['L']
        self.iteration, self.n_saved = s['iteration'], s['n_saved']
        self.n_accept, self.n_diverge = s['n_accept'], s['n_diverge']
        self.n_diverge_warmup = s['n_diverge_warmup']
        self.n_warmup, self._da = s['n_warmup'], s['da']
        self._welford = None
        if s['welford'] is not None:
            self._welford = {k: (v.to(device) if torch.is_tensor(v) else v)
                             for k, v in s['welford'].items()}
        self.post.prior.load_state(s['prior'])
        self.post.likelihood.load_state(s['likelihood'])
        self.rng.bit_generator.state = s['rng']
        self.trng.set_state(s['trng'])

        # Remove anything written after the checkpoint, so that the files
        # match the state we continue from.
        def truncate(filename, keep):
            if not os.path.exists(filename):
                return
            with open(filename) as f:
                rows = list(csv.reader(f))
            rows = [rows[0]] + [r for r in rows[1:] if keep(r)]
            with open(filename, 'w', newline='') as f:
                csv.writer(f).writerows(rows)
        truncate(self.tracefile, lambda r: int(r[0]) <= self.iteration)
        truncate(self.indexfile, lambda r: int(r[1]) <= self.iteration)
        for f in glob.glob(os.path.join(self.sampledir, 'params_*.pth')):
            k = int(os.path.basename(f)[7:11])
            if k > self.n_saved:
                os.remove(f)

    # ------------------------------------------------------------------------
    # The main loop
    # ------------------------------------------------------------------------
    def run(self, n_warmup=500, n_samples=1000, thin=1, print_every=10,
            resume=False, init_hyper=True):
        '''
        Run the chain: n_warmup tuning iterations, then n_samples * thin
        iterations of which every thin-th is saved.

        resume=True continues from the checkpoint file in outdir.
        init_hyper=True (new runs only) sets tau_g and the noise scale from
        the starting parameters, as is sensible after a warm start.
        '''
        lik, prior = self.post.likelihood, self.post.prior
        if resume:
            if not os.path.exists(self.statefile):
                raise FileNotFoundError(f'no checkpoint {self.statefile}')
            self._load_checkpoint()
            self._refresh()
            print(f'resumed at iteration {self.iteration}, '
                  f'{self.n_saved} samples saved')
        else:
            # start afresh: remove files of an earlier run in this folder
            for f in [self.tracefile, self.indexfile, self.statefile] + \
                    glob.glob(os.path.join(self.sampledir, 'params_*.pth')):
                if os.path.exists(f):
                    os.remove(f)
            self.n_warmup = int(n_warmup)
            if init_hyper:
                self._refresh()
                prior.from_weights(self.w)
                if lik.learn:
                    lik.scale = lik.initial_scale(self.stat, self.post.N)
            self._refresh()
            if self.eps is None:
                self.eps = self.find_step_size()
                self._refresh()
            self._da_start(self.eps)

        n_warmup = self.n_warmup
        n_total  = n_warmup + int(n_samples) * int(thin)
        # Warm-up windows: [0, w1) step size only; [w1, w2) also collect the
        # variance of w for the mass matrix; at w2 set M^{-1} and restart the
        # step-size tuning, which continues until the end of warm-up.
        w1 = int(0.15 * n_warmup)
        w2 = int(0.85 * n_warmup)
        mass_on = self.adapt_mass and n_warmup >= 20

        header = self._trace_header()
        self._last = self.snapshot()
        last_ckpt = time.time()
        t_start = time.time()
        it0 = self.iteration
        try:
            while self.iteration < n_total:
                warm = self.iteration < n_warmup

                # jittered step size and number of steps for this trajectory
                u = 2 * self.rng.random() - 1
                eps = self.eps * (1 + self.jitter * u)
                u = 2 * self.rng.random() - 1
                L = max(1, int(round(self.L * (1 + self.jitter * u))))

                # 1-3: the HMC update of w
                accepted, dH, prob = self.trajectory(eps, L)
                if dH > self.max_dH:
                    if warm:
                        self.n_diverge_warmup += 1
                    else:
                        self.n_diverge += 1

                # 4: Gibbs updates of the hyperparameters, given w; then the
                # energy and gradient at w change, so recompute them
                prior.gibbs(self.w, self.rng)
                lik.gibbs(self.stat, self.post.N, self.rng)
                self._refresh()

                self.iteration += 1
                it = self.iteration

                # warm-up tuning
                if warm:
                    self._da_update(prob)
                    if mass_on and w1 <= it <= w2:
                        self._welford_add(self.w)
                    if mass_on and it == w2:
                        self.mass_inv = self._welford_mass()
                        self.eps = self.find_step_size(self.eps)
                        self._refresh()
                        self._da_start(self.eps)
                    if it == n_warmup:
                        # freeze the step size at its averaged value
                        self.eps = math.exp(self._da['log_eps_bar'])
                elif accepted:
                    self.n_accept += 1

                # trace
                row = [it, 'warmup' if warm else 'sample', int(accepted),
                       f'{dH:.6g}', f'{prob:.4f}', f'{self.U:.10g}',
                       f'{eps:.6g}', L, f'{lik.scale:.6g}'] + \
                      [f'{t:.6g}' for t in prior.tau]
                if self.monitor is not None:
                    self.view.set(self.w)
                    row += [f'{v:.8g}' for v in self._monitor_values()]
                self._append(self.tracefile, header, row)

                # save a sample
                if not warm and (it - n_warmup) % thin == 0:
                    self._save_sample()

                # the iteration is complete: the state and the files agree
                self._last = self.snapshot()
                if time.time() - last_ckpt > self.checkpoint_every:
                    self.checkpoint(self._last)
                    last_ckpt = time.time()

                if print_every and (it % print_every == 0 or it == n_total):
                    rate = (time.time() - t_start) / max(it - it0, 1)
                    left = rate * (n_total - it) / 60
                    phase = 'warm-up' if warm else 'sampling'
                    acc = (f'acc {self.acceptance_rate():.2f}'
                           if not warm else f'p {prob:.2f}')
                    print(f'{it:6d}/{n_total} {phase:8s} {acc}  '
                          f'eps {eps:.3e}  U {self.U:.6e}  '
                          f'scale {lik.scale:.3e}  '
                          f'({left:.1f} min left)', flush=True)
        except KeyboardInterrupt:
            # save the state at the end of the last COMPLETE iteration
            self.checkpoint(self._last)
            print(f'\ninterrupted: checkpoint saved at iteration '
                  f'{self._last["iteration"]}; continue with '
                  f'run(..., resume=True)')
            raise

        self.checkpoint()
        self.view.set(self.w)      # leave the model at the last sample
        print(f'done: {self.n_saved} samples in {self.sampledir}; '
              f'acceptance rate {self.acceptance_rate():.3f}; '
              f'divergences {self.n_diverge} '
              f'(and {self.n_diverge_warmup} during warm-up, while the '
              f'step size was being tuned)')

    def acceptance_rate(self):
        n = self.iteration - (self.n_warmup or 0)
        return self.n_accept / n if n > 0 else float('nan')

    def trace(self):
        '''
        The trace file as a dictionary {column name: numpy array}.
        '''
        return read_csv(self.tracefile)


def read_csv(filename):
    '''
    Read a csv file written by HMC into {column: numpy array}; numeric
    columns become float arrays.
    '''
    with open(filename) as f:
        rows = list(csv.reader(f))
    cols = {}
    for k, name in enumerate(rows[0]):
        values = [r[k] for r in rows[1:]]
        try:
            cols[name] = np.array(values, dtype=float)
        except ValueError:
            cols[name] = np.array(values)
    return cols


# ============================================================================
# 6. The Bayesian neural network: an ensemble of sampled parameter sets
# ============================================================================
class BNN:
    '''
    The ensemble of parameter sets w_1, ..., w_K sampled by HMC.

    Arguments
    ---------
    model_factory : a function (or class) that, called with no arguments,
                    returns a new model, e.g. HEFTNet.
    rundir        : the run folder. Every file samples/params_*.pth inside it,
                    including inside chain_*/ sub-folders, is used.
    device        : where to compute (default: CPU).
    max_samples   : use at most this many samples (evenly spaced).

    Example
    -------
        bnn = BNN(HEFTNet, 'runs/BNN_laplace')
        mu  = bnn.mean(x)            # posterior mean, one value per row of x
        sd  = bnn.std(x)
        lo, hi = bnn.interval(x, 0.68)
    '''
    def __init__(self, model_factory, rundir, device='cpu', max_samples=None):
        pattern = os.path.join(rundir, '**', 'params_*.pth')
        files = sorted(f for f in glob.glob(pattern, recursive=True)
                       if os.path.basename(os.path.dirname(f)) == 'samples')
        if not files:
            raise FileNotFoundError(f'BNN: no samples/params_*.pth in {rundir}')
        if max_samples is not None and len(files) > max_samples:
            keep = np.linspace(0, len(files) - 1, max_samples).astype(int)
            files = [files[i] for i in keep]
        self.files  = files
        self.device = device
        # every parameter set, as a state_dict, held in memory (16 kB each
        # for HEFTNet)
        self.states = [torch.load(f, map_location=device, weights_only=True)
                       for f in files]
        # the noise scale of each sample, from samples/index.csv
        self.scales = []
        for f in files:
            index = read_csv(os.path.join(os.path.dirname(f), 'index.csv'))
            k = list(index['file']).index(os.path.basename(f))
            self.scales.append(float(index['scale'][k]))
        self.model = model_factory().to(device)
        self.model.eval()

    def __len__(self):
        return len(self.states)

    def models(self):
        '''
        Loop over the samples: yields the (same) model object loaded, in turn,
        with each sampled parameter set. (A Python "generator": use it as
        "for model in bnn.models(): ...".)
        '''
        for state in self.states:
            self.model.load_state_dict(state)
            yield self.model

    def apply(self, fn, x):
        '''
        fn(model, x) for every sample, stacked: shape [K, ...]. For example
            bnn.apply(lambda m, x: m.coeffs(x[:, 0]), x)
        gives the 23 coefficient functions of HEFTNet for every sample.
        '''
        x = torch.as_tensor(x, device=self.device)
        with torch.no_grad():
            return torch.stack([fn(m, x) for m in self.models()])

    def predict(self, x):
        '''
        f(x; w_k) for every sample k: a tensor of shape [K, N].
        '''
        return self.apply(lambda m, x: m(x).reshape(-1), x)

    def _moments(self, x):
        # Running mean and variance over the samples (Welford's method), so
        # that the [K, N] table of predictions is never held in memory.
        x = torch.as_tensor(x, device=self.device)
        n, mean, M2 = 0, None, None
        with torch.no_grad():
            for m in self.models():
                f = m(x).reshape(-1).double()
                n += 1
                if mean is None:
                    mean, M2 = f.clone(), torch.zeros_like(f)
                else:
                    delta = f - mean
                    mean += delta / n
                    M2 += delta * (f - mean)
        var = M2 / max(n - 1, 1)
        return mean, var

    def mean(self, x):
        '''
        Posterior mean of the output at each row of x (Eq. 2 of the plan).
        '''
        return self._moments(x)[0]

    def std(self, x):
        '''
        Posterior standard deviation of the output at each row of x.
        '''
        return torch.sqrt(self._moments(x)[1])

    def quantiles(self, x, q):
        '''
        Posterior quantiles of the output: q a list of probabilities, e.g.
        [0.16, 0.5, 0.84]. Returns a tensor of shape [len(q), N].
        '''
        f = self.predict(x).double()
        q = torch.as_tensor(q, dtype=torch.float64, device=f.device)
        return torch.quantile(f, q, dim=0)

    def interval(self, x, cl=0.68):
        '''
        Central credible interval of probability content cl: (lower, upper).
        '''
        lo, hi = self.quantiles(x, [(1 - cl) / 2, (1 + cl) / 2])
        return lo, hi

    def predictive(self, x, likelihood, generator=None):
        '''
        One simulated target per sample and row: f(x; w_k) + noise drawn from
        the likelihood with sample k's noise scale. Shape [K, N]. Quantiles
        of this give predictive intervals (epistemic + noise).
        '''
        f = self.predict(x)
        out = torch.empty_like(f)
        saved = likelihood.scale
        for k in range(len(self)):
            likelihood.scale = self.scales[k]
            out[k] = f[k] + likelihood.noise(f[k], generator)
        likelihood.scale = saved
        return out


# ============================================================================
# 7. Convergence diagnostics
# ============================================================================
def autocorrelation(y):
    '''
    Normalised autocorrelation rho(lag) of the sequence y, for
    lag = 0, 1, ..., len(y)-1, computed with the fast Fourier transform.
    '''
    y = np.asarray(y, dtype=float)
    n = len(y)
    y = y - y.mean()
    # zero-pad to a power of 2 at least 2n long to avoid wrap-around
    size = 1 << (2 * n - 1).bit_length()
    F = np.fft.rfft(y, size)
    acf = np.fft.irfft(F * np.conjugate(F), size)[:n]
    return acf / acf[0] if acf[0] > 0 else np.ones(n)


def ess(y):
    '''
    Effective sample size of the sequence y: the number of independent draws
    that would give the same precision for the mean of y. Uses Geyer's
    initial positive sequence: sum the autocorrelations in pairs until a
    pair sum becomes negative.
    '''
    rho = autocorrelation(y)
    n = len(rho)
    total = 0.0
    for k in range(0, n - 1, 2):
        pair = rho[k] + rho[k + 1]
        if pair < 0:
            break
        total += pair
    tau = 2 * total - 1        # integrated autocorrelation time
    return n / max(tau, 1e-12)


def rhat(chains):
    '''
    Gelman-Rubin potential scale reduction factor for a list of chains
    (equal lengths) of one quantity. Values close to 1 (say < 1.05) suggest
    the chains have converged to the same distribution.
    '''
    c = np.asarray(chains, dtype=float)     # shape [m chains, n draws]
    m, n = c.shape
    means = c.mean(axis=1)
    B = n * means.var(ddof=1)               # between-chain variance
    W = c.var(axis=1, ddof=1).mean()        # within-chain variance
    var_hat = (n - 1) / n * W + B / n
    return math.sqrt(var_hat / W)
