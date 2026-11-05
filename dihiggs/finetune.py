# ----------------------------------------------------------------------------
# finetune.py: fine-tune a trained HEFTNet (or any model) with the Huber loss
#              applied to the pulls, i.e. the residuals in units of the
#              POWHEG noise of each bin.
#
# The design is described in docs/huber/huber_plan.pdf. In brief:
#
#   estimate_n_eff  the effective number of equal-weight events per POWHEG
#                   spectrum, from the bin-to-bin jitter of the spectra.
#   noise_variance  the noise variance of every row (bin) of a data frame.
#   HuberPull       the loss: Huber function of r = (f - t) / sigma.
#   finetune        stage 2 of the training: full-batch Adam, starting from
#                   a trained model, keeping the parameters with the best
#                   validation loss.
#   finetune_lbfgs  the same with L-BFGS in double precision, for the density
#                   model HEFTDensity (docs/density/density_plan.pdf), whose
#                   full-batch evaluation is cheap; it converges far faster
#                   than Adam (code/density_toy.py).
#
# The noise variances are NOT a second target column: they are a separate
# tensor, aligned row by row with the inputs and targets, and the rows of a
# batch are chosen by their row numbers (indices), which pick out inputs,
# targets and variances together.
#
# P. Bhat, N. De Filippis, K.F. Johnson, H.B. Prosper, R. Starnes
# Created: Sat Oct 31 2026
# ----------------------------------------------------------------------------
import os, math, time
import numpy as np
import torch
import torch.nn as nn

__all__ = ['estimate_n_eff', 'noise_variance', 'HuberPull', 'finetune', 'finetune_lbfgs']

# ----------------------------------------------------------------------------
# A note on three PyTorch idioms used below
#
#   with torch.no_grad():  inside this block PyTorch does not record the
#                          operations for differentiation; we use it when we
#                          only need numbers (validation), not gradients.
#
#   t.detach()             the same numbers as t, cut off from the record of
#                          how t was computed, so that gradients do not flow
#                          through it.
#
#   loss.backward()        computes d(loss)/d(parameter) for every parameter
#                          and ADDS it to parameter.grad; calling it once per
#                          chunk of rows therefore accumulates the gradient of
#                          the sum over all rows.
# ----------------------------------------------------------------------------


# ============================================================================
# 1. The POWHEG noise
# ============================================================================
def estimate_n_eff(df, x_min=(38 - 17 + 0.5) / 100):
    '''
    Estimate N_eff, the effective number of equal-weight events per POWHEG
    spectrum, from the bin-to-bin jitter of the spectra in a data frame with
    columns ID, mhh and sigma (one row per bin), as in heft_train.ipynb.

    For each bin, D = f_k - (f_{k-1} + f_{k+1})/2 compares the bin with the
    average of its neighbours. If each spectrum is a histogram of N_eff
    events, then
        q = D^2 / (sigma_tot * (f_k + (f_{k-1} + f_{k+1})/4))
    is 1/N_eff times the square of a standard normal variable, whose median
    is 0.4549. Only bins with x >= x_min (570 GeV), where the spectra are
    smooth, are used.
    '''
    # one row per spectrum, one column per bin; missing bins (dropped from
    # the file because their cross section is zero) come back as 0
    spectra = df.pivot_table(index='ID', columns='mhh', values='sigma',
                             fill_value=0.0)
    F     = spectra.to_numpy(np.float64)
    F_tot = F.sum(axis=1, keepdims=True)
    # F[:, 1:-1] is every bin except the first and last; F[:, :-2] and
    # F[:, 2:] are their left and right neighbours
    f_mid, f_lo, f_hi = F[:, 1:-1], F[:, :-2], F[:, 2:]
    D = f_mid - (f_lo + f_hi) / 2
    good = (f_mid > 0) & (f_lo > 0) & (f_hi > 0)
    q = np.full_like(D, np.nan)
    q[good] = D[good]**2 / (F_tot * (f_mid + (f_lo + f_hi) / 4))[good]
    smooth = spectra.columns.to_numpy()[1:-1] >= x_min
    return float(0.4549 / np.nanmedian(q[:, smooth]))


def noise_variance(df, n_eff):
    '''
    The noise variance of every row of a data frame (columns ID, sigma):

        var_k = (t_k + unit_k) * unit_k,   unit_k = sigma_tot / N_eff,

    where sigma_tot is the total cross section of the spectrum to which the
    row belongs, and unit_k is the cross section of ONE event of that
    spectrum. The term unit_k^2 is the variance of one extra event, so that
    no bin claims to be known better than that.

    Returns (var, unit), two numpy arrays with one entry per row.
    '''
    # transform('sum') gives every row the sum over the rows with its ID
    sigma_tot = df.groupby('ID')['sigma'].transform('sum').to_numpy(np.float64)
    unit = sigma_tot / n_eff
    t = df['sigma'].to_numpy(np.float64)
    return (t + unit) * unit, unit


# ============================================================================
# 2. The loss
# ============================================================================
class HuberPull(nn.Module):
    '''
    The Huber loss of the pulls r = (f - t) / sqrt(v):

        l(r) = r^2 / 2                   if |r| <= delta,
             = delta * (|r| - delta/2)   if |r| >  delta.

    With delta = None, l(r) = r^2 / 2 for all r (plain chi-squared / 2).

    forward(f, t, v, reduction='mean') takes the predictions f, the targets
    t and the noise variances v, all of shape [B], and returns the mean (or,
    with reduction='sum', the sum) of l over the B rows.
    '''
    def __init__(self, delta=None):
        super().__init__()
        self.delta = None if delta is None else float(delta)

    def forward(self, f, t, v, reduction='mean'):
        r = (f - t) / torch.sqrt(v)
        if self.delta is None:
            loss = 0.5 * r**2
        else:
            a = torch.abs(r)
            # torch.where(condition, x, y) takes x where condition is True
            # and y elsewhere, element by element
            loss = torch.where(a <= self.delta, 0.5 * r**2,
                               self.delta * (a - 0.5 * self.delta))
        return loss.sum() if reduction == 'sum' else loss.mean()

    def __repr__(self):
        return f'HuberPull(delta={self.delta})'


# ============================================================================
# 3. The fine-tune
# ============================================================================
def _variance(f, var, unit, variance_from):
    # The noise variance of the rows in a chunk. With 'data' it is the fixed
    # var computed from the POWHEG value t (Neyman's chi-squared). With
    # 'model' the POWHEG value is replaced by the model's own prediction,
    # held fixed for this step (detach) and kept non-negative (Pearson's
    # chi-squared): (f + unit) * unit.
    if variance_from == 'data':
        return var
    return (torch.clamp(f.detach(), min=0.0) + unit) * unit


def _average_loss(model, loss_fn, X, T, V, U, variance_from, chunk_size,
                  with_grad):
    '''
    The average loss over all rows of (X, T, V, U), computed in chunks. With
    with_grad=True the gradients of the average are accumulated in the
    parameters' .grad (exactly as if all rows were one batch).
    '''
    N = len(X)
    total = 0.0
    for start in range(0, N, chunk_size):
        # a Python slice selects rows start, start+1, ..., start+chunk_size-1
        rows = slice(start, start + chunk_size)
        f = model(X[rows]).view(-1)
        v = _variance(f, V[rows], U[rows], variance_from)
        # the sum over the chunk, divided by N: the chunks add up to the
        # average over all rows
        loss = loss_fn(f, T[rows], v, reduction='sum') / N
        if with_grad:
            loss.backward()
        total += float(loss.detach())
    return total


def finetune(model, X, T, V, U, Xv, Tv, Vv, Uv, loss_fn,
             steps=5000, lr=1e-4, n_lr_steps=4, gamma=0.5,
             variance_from='data', chunk_size=65536, val_every=50,
             paramsfile=None, lossfile=None, print_every=500, curvature=0.0):
    '''
    Stage 2 of the training: fine-tune a trained model with full-batch Adam.

    Arguments
    ---------
    model          : the model, already trained (stage 1) and on the device
                     of the tensors below.
    X, T, V, U     : training inputs [N, 6], targets [N], noise variances [N]
                     and one-event units [N] (see noise_variance), tensors on
                     the model's device. U is used only with
                     variance_from='model'.
    Xv, Tv, Vv, Uv : the same for the validation rows.
    loss_fn        : e.g. HuberPull(delta=2).
    steps          : number of full-batch steps.
    lr, n_lr_steps, gamma : the learning rate starts at lr and is multiplied
                     by gamma n_lr_steps - 1 times, at equal intervals.
    variance_from  : 'data' (variance from the POWHEG value) or 'model'
                     (from the model's prediction).
    chunk_size     : rows per chunk; the gradient of the full batch is
                     accumulated chunk by chunk to limit memory.
    val_every      : compute the validation loss every this many steps.
    paramsfile     : where to save the parameters with the lowest validation
                     loss (model.save); they are reloaded at the end.
    lossfile       : csv file for the loss history.
    curvature      : strength lambda of the smoothness penalty
                     lambda * model.curvature_penalty() (HEFTDensity only),
                     added to the training AND the validation loss, so that
                     the parameters kept are the best for the penalised loss.

    Returns a dictionary of numpy arrays: step, train, val, best, lr.
    '''
    assert variance_from in ('data', 'model'), variance_from
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def penalty(with_grad):
        # the smoothness penalty, once per step (it does not depend on the rows)
        if not curvature:
            return 0.0
        pen = curvature * model.curvature_penalty()
        if with_grad:
            pen.backward()
        return float(pen.detach())
    # StepLR multiplies the learning rate by gamma every step_size steps
    step_size = max(1, steps // n_lr_steps)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size,
                                                gamma=gamma)
    history = {'step': [], 'train': [], 'val': [], 'best': [], 'lr': []}
    if lossfile is not None:
        open(lossfile, 'w').write('step,train,val,best,lr\n')

    def validation_loss():
        model.eval()
        with torch.no_grad():
            return _average_loss(model, loss_fn, Xv, Tv, Vv, Uv,
                                 variance_from, chunk_size, with_grad=False) + penalty(False)

    best = validation_loss()        # the starting model is the first "best"
    if paramsfile is not None:
        model.save(paramsfile)
    t0 = time.time()
    for step in range(1, steps + 1):
        model.train()
        optimizer.zero_grad()       # forget the previous gradients
        train = _average_loss(model, loss_fn, X, T, V, U, variance_from,
                              chunk_size, with_grad=True) + penalty(True)
        if not math.isfinite(train):
            print(f'\nstep {step}: the training loss is not finite; '
                  'stopping and keeping the best parameters so far')
            break
        optimizer.step()            # one step downhill
        scheduler.step()            # perhaps lower the learning rate

        if step % val_every == 0 or step == steps:
            val = validation_loss()
            if val < best:
                best = val
                if paramsfile is not None:
                    model.save(paramsfile)
            current_lr = optimizer.param_groups[0]['lr']
            for key, value in zip(['step', 'train', 'val', 'best', 'lr'],
                                  [step, train, val, best, current_lr]):
                history[key].append(value)
            if lossfile is not None:
                open(lossfile, 'a').write(f'{step},{train:.8g},{val:.8g},'
                                          f'{best:.8g},{current_lr:.3g}\n')
            if print_every and (step % print_every == 0 or step == steps):
                left = (time.time() - t0) / step * (steps - step) / 60
                print(f'{step:6d}/{steps}  train {train:.5g}  val {val:.5g}  '
                      f'best {best:.5g}  lr {current_lr:.2e}  '
                      f'({left:.1f} min left)', flush=True)

    # reload the parameters with the lowest validation loss
    if paramsfile is not None:
        device = next(model.parameters()).device
        model.load(paramsfile)      # drwho's Model.load: onto the CPU
        model.to(device)
    model.eval()
    return {k: np.array(v) for k, v in history.items()}


# ============================================================================
# 4. The fine-tune with L-BFGS
# ============================================================================
def finetune_lbfgs(model, X, T, V, U, Xv, Tv, Vv, Uv, loss_fn,
                   blocks=30, iters_per_block=50, history_size=50,
                   variance_from='model', tol=1e-7,
                   paramsfile=None, lossfile=None, verbose=True, curvature=0.0):
    '''
    Stage 2 of the training with L-BFGS, a quasi-Newton method: from the
    gradients of the last history_size steps it builds an approximation to
    the inverse of the Hessian (the matrix of second derivatives of the loss)
    and steps in the direction it implies, with a line search for the step
    length. For a smooth loss of a few thousand parameters, evaluated on all
    rows at once, it needs far fewer steps than Adam.

    Intended for HEFTDensity: its forward pass evaluates the network only at
    the fixed quadrature nodes, so a full-batch evaluation over all training
    rows is cheap. Everything is done in double precision, because the line
    search needs accurate losses and gradients (in single precision L-BFGS
    stalls); at the end the model is returned in single precision.

    The work is done in blocks of iters_per_block L-BFGS iterations. With
    variance_from='model' the noise variances are recomputed from the model's
    prediction at the start of each block and held fixed within it
    (iteratively reweighted least squares: at convergence, the Poisson fit,
    without the downward bias of weighting by the observed counts). After
    each block the validation loss is computed and the best parameters kept.
    Stops when the training loss changes by less than tol (relative).

    Arguments as for finetune (including curvature); blocks, iters_per_block
    and history_size control L-BFGS. Returns a dictionary of numpy arrays: step (block),
    train, val, best.
    '''
    assert variance_from in ('data', 'model'), variance_from
    model.double()
    X, T, V, U = (a.double() for a in (X, T, V, U))
    Xv, Tv, Vv, Uv = (a.double() for a in (Xv, Tv, Vv, Uv))
    optimizer = torch.optim.LBFGS(model.parameters(), lr=1, max_iter=iters_per_block,
                                  history_size=history_size,
                                  line_search_fn='strong_wolfe')

    def loss_on(Xa, Ta, Va, Ua, var=None):
        # the data loss plus, if curvature > 0, the smoothness penalty
        f = model(Xa).view(-1)
        v = _variance(f, Va, Ua, variance_from) if var is None else var
        loss = loss_fn(f, Ta, v)
        if curvature:
            loss = loss + curvature * model.curvature_penalty()
        return loss

    def validation_loss():
        model.eval()
        with torch.no_grad():
            return float(loss_on(Xv, Tv, Vv, Uv))

    history = {'step': [], 'train': [], 'val': [], 'best': []}
    if lossfile is not None:
        open(lossfile, 'w').write('step,train,val,best,lr\n')
    best = validation_loss()
    if paramsfile is not None:
        model.save(paramsfile)
    previous = math.inf
    t0 = time.time()
    for block in range(1, blocks + 1):
        model.train()
        # the variances for this block, from the current model (or the data)
        with torch.no_grad():
            var = _variance(model(X).view(-1), V, U, variance_from)

        def closure():
            # L-BFGS calls this (possibly several times per iteration, in
            # the line search) to get the loss and its gradient
            optimizer.zero_grad()
            loss = loss_on(X, T, V, U, var)
            loss.backward()
            return loss

        optimizer.step(closure)
        with torch.no_grad():
            train = float(loss_on(X, T, V, U))
        if not math.isfinite(train):
            print(f'block {block}: the training loss is not finite; stopping')
            break
        val = validation_loss()
        if val < best:
            best = val
            if paramsfile is not None:
                model.save(paramsfile)
        for key, value in zip(['step', 'train', 'val', 'best'], [block, train, val, best]):
            history[key].append(value)
        if lossfile is not None:
            open(lossfile, 'a').write(f'{block},{train:.10g},{val:.10g},{best:.10g},1\n')
        if verbose:
            print(f'block {block:3d}/{blocks}  train {train:.7g}  val {val:.7g}  best {best:.7g}  '
                  f'({(time.time() - t0) / 60:.1f} min)', flush=True)
        if abs(previous - train) < tol * abs(train):
            break
        previous = train

    # reload the parameters with the lowest validation loss
    if paramsfile is not None:
        device = next(model.parameters()).device
        model.load(paramsfile)
        model.to(device)
    model.float()
    model.eval()
    return {k: np.array(v) for k, v in history.items()}
