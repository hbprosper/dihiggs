# ----------------------------------------------------------------------------
# ensemble.py: an ensemble of trained networks of the same architecture,
#              for example members trained with different random seeds or
#              on bootstrap resamples of the training data.
#
# The design is described in docs/ensemble/ensemble_plan.pdf. The class
# Ensemble has the same methods as hmc.BNN (mean, std, quantiles, interval,
# apply), so that analysis code can use either.
#
# This module deliberately imports nothing from dihiggs, so that it can be
# moved to drwho unchanged.
#
# Harrison B. Prosper and Claude
# Created: Fri Oct 30 2026
# ----------------------------------------------------------------------------
import os, re, csv, glob
import numpy as np
import torch

__all__ = ['Ensemble', 'read_csv']

# ----------------------------------------------------------------------------
# A note on two PyTorch idioms used below
#
#   with torch.no_grad():   Inside this block PyTorch does NOT record the
#                           operations for later differentiation. We only
#                           want numbers here, not gradients, so this saves
#                           time and memory.
#
#   torch.stack(list)       Joins a list of tensors of the same shape along a
#                           new first dimension: K tensors of shape [N] become
#                           one tensor of shape [K, N].
# ----------------------------------------------------------------------------


def read_csv(filename):
    '''
    Read a small csv file into a dictionary {column name: numpy array};
    columns that hold numbers become float arrays.
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


class Ensemble:
    '''
    An ensemble of trained networks: the same architecture, one parameter
    file per member.

    Arguments
    ---------
    model_factory : a function (or class) that, called with no arguments,
                    returns a new model, e.g. HEFTNet.
    rundir        : the ensemble folder, e.g. 'runs/ENS_bootstrap'.
    pattern       : where the parameter files are, relative to rundir
                    (default 'member_*/dihiggs_params.pth'). Any glob pattern
                    may be given; '*' matches any characters in a name.
    device        : where to compute (default: CPU).
    members       : optionally, a list of member folder names to use, e.g.
                    ['member_01', 'member_02']; default: all that are found.

    Example
    -------
        ens = Ensemble(HEFTNet, 'runs/ENS_bootstrap')
        mu  = ens.mean(x)              # ensemble mean, one value per row of x
        sd  = ens.std(x)               # spread of the members
        lo, hi = ens.interval(x, 0.68) # central 68% of the members
        A   = ens.apply(lambda m, x: m.coeffs(x), xs)   # any function
    '''
    def __init__(self, model_factory, rundir,
                 pattern='member_*/dihiggs_params.pth',
                 device='cpu', members=None):
        files = glob.glob(os.path.join(rundir, pattern))
        if members is not None:
            # keep only the files inside the requested member folders
            files = [f for f in files
                     if os.path.basename(os.path.dirname(f)) in members]
        if not files:
            raise FileNotFoundError(f'Ensemble: no {pattern} in {rundir}')

        # Sort by the member NUMBER in the folder name, so that member_10
        # comes after member_9 (sorting the names as text would not do that).
        def number(f):
            digits = re.findall(r'\d+', os.path.basename(os.path.dirname(f)))
            return int(digits[-1]) if digits else 0
        self.files  = sorted(files, key=number)
        self.rundir = rundir
        self.device = device

        # every member's parameters, as a state_dict (a dictionary of
        # tensors), held in memory; about 16 kB each for HEFTNet
        self.states = [torch.load(f, map_location=device, weights_only=True)
                       for f in self.files]
        self.model = model_factory().to(device)
        self.model.eval()

    def __len__(self):
        return len(self.states)

    def summary(self):
        '''
        The contents of rundir/ensemble.csv (one row per member, written by
        heft_train_ensemble.ipynb), or None if there is no such file.
        '''
        filename = os.path.join(self.rundir, 'ensemble.csv')
        return read_csv(filename) if os.path.exists(filename) else None

    def models(self):
        '''
        Loop over the members: yields the (same) model object loaded, in turn,
        with each member's parameters. (A Python "generator": use it as
        "for model in ens.models(): ...".)
        '''
        for state in self.states:
            self.model.load_state_dict(state)
            yield self.model

    def apply(self, fn, x):
        '''
        fn(model, x) for every member, stacked: a tensor of shape [K, ...].
        For HEFTNet, ens.apply(lambda m, x: m.coeffs(x), xs) gives the 23
        coefficient functions a_i(m_hh) of every member.
        '''
        x = torch.as_tensor(x, device=self.device)
        with torch.no_grad():
            return torch.stack([fn(m, x) for m in self.models()])

    def predict(self, x):
        '''
        f(x; member k) for every member k: a tensor of shape [K, N].
        '''
        return self.apply(lambda m, x: m(x).reshape(-1), x)

    def _moments(self, x):
        # Running mean and variance over the members (Welford's method), so
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
        var = M2 / max(n - 1, 1)        # unbiased: divide by K - 1
        return mean, var

    def mean(self, x):
        '''
        The ensemble mean of the output at each row of x.
        '''
        return self._moments(x)[0]

    def std(self, x):
        '''
        The standard deviation of the members' outputs at each row of x: the
        estimate of the standard error of the trained network.
        '''
        return torch.sqrt(self._moments(x)[1])

    def quantiles(self, x, q):
        '''
        Quantiles of the members' outputs: q is a list of probabilities, e.g.
        [0.16, 0.5, 0.84]. Returns a tensor of shape [len(q), N]. With a
        few dozen members only the central quantiles are well determined.
        '''
        f = self.predict(x).double()
        q = torch.as_tensor(q, dtype=torch.float64, device=f.device)
        return torch.quantile(f, q, dim=0)

    def interval(self, x, cl=0.68):
        '''
        The central interval containing a fraction cl of the members:
        (lower, upper).
        '''
        lo, hi = self.quantiles(x, [(1 - cl) / 2, (1 + cl) / 2])
        return lo, hi
