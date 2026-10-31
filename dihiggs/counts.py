# ----------------------------------------------------------------------------
# counts.py: recover the event counts behind the POWHEG spectra.
#
# Every POWHEG spectrum is a histogram of events that all carry the SAME
# weight q (in pb). So the cross section in bin k of a spectrum is
#
#     t_k = q * n_k,     n_k = 0, 1, 2, ...  (an integer: the event count)
#
# with one scale factor q per spectrum. Given the t_k of a spectrum, q is the
# largest number of which every t_k is an integer multiple (a "greatest
# common divisor" of real numbers), and n_k = t_k / q.
#
# Why it matters: the count n_k is Poisson distributed, so the noise
# variance of t_k is exactly
#
#     Var(t_k) = q^2 * Var(n_k) = q^2 * n_k = q * t_k,
#
# with no "effective number of events" to estimate.
#
# Functions
#   find_quantum(values)        q and the counts of ONE spectrum
#   recover_counts(df)          the same for every spectrum of a long-format
#                               data frame (one row per bin, as the training
#                               file); adds columns 'count' and 'quantum'
#   recover_counts_wide(F)      the same for a 2-d array, one row per
#                               spectrum, one column per bin (the test file)
#   add_empty_bins(df)          restore the bins with zero events, which were
#                               dropped from the training file
#   poisson_variance(df)        q * t for every row
#
# Harrison B. Prosper and Claude
# Created: Sat Oct 31 2026
# ----------------------------------------------------------------------------
import warnings
import numpy as np
import pandas as pd

__all__ = ['find_quantum', 'recover_counts', 'recover_counts_wide',
           'add_empty_bins', 'poisson_variance']


def find_quantum(values, max_count=5000, tol=0.01):
    '''
    Find the scale factor q and the integer counts of one spectrum.

    Arguments
    ---------
    values    : the non-zero cross sections of the bins of ONE spectrum.
    max_count : the largest event count tried for the smallest bin.
    tol       : the largest allowed distance of any n_k from an integer.

    Returns (q, counts, max_dev): the scale factor, the counts (a numpy
    array of floats holding integers), and the largest distance of any
    t_k / q from its integer. Raises ValueError if no q is found.

    Method
    ------
    Let t_min be the smallest value. If the smallest bin holds c events,
    then q = t_min / c, and every ratio t_k / t_min times c must be an
    integer. We try c = 1, 2, 3, ... and keep the FIRST c that works; that
    gives the largest q, i.e. the greatest common divisor. (A smaller q,
    q/2 say, would also make every value an integer multiple; the largest
    one is the event weight unless every count happens to be even, which,
    for about 80 bins, does not happen in practice.) Finally q is refined
    using all bins: q = sum(t) / sum(n).
    '''
    # np.asarray(..., dtype=np.float64) makes a numpy array of 64-bit floats
    t = np.asarray(values, dtype=np.float64)
    t = t[t > 0]
    if len(t) == 0:
        raise ValueError('find_quantum: no non-zero values')
    ratios = t / t.min()
    for c in range(1, max_count + 1):
        n = c * ratios
        # np.rint rounds to the nearest integer (and returns floats)
        if np.max(np.abs(n - np.rint(n))) < tol:
            counts = np.rint(n)
            q = t.sum() / counts.sum()          # refined with all bins
            max_dev = float(np.max(np.abs(t / q - counts)))
            return q, counts, max_dev
    raise ValueError(f'find_quantum: no scale factor found with a smallest '
                     f'count up to {max_count}')


def recover_counts(df, id_col='ID', target='sigma', **kwargs):
    '''
    Recover the counts of every spectrum of a long-format data frame (one row
    per bin; columns id_col and target, as in the training file).

    Returns (out, summary):
      out     - a copy of df with two new columns: 'count' (the number of
                events in the bin) and 'quantum' (the scale factor q of the
                spectrum to which the row belongs), so that
                df[target] == out['quantum'] * out['count'];
      summary - one row per spectrum: id_col, quantum, n_events (the number
                of events inside the binned range), n_bins (non-zero bins),
                max_dev (see find_quantum).
    '''
    out = df.copy()
    out['count'] = np.nan
    out['quantum'] = np.nan
    rows = []
    # groupby(id_col).indices: a dictionary {spectrum ID: row positions}
    for sid, pos in df.groupby(id_col).indices.items():
        t = df[target].to_numpy(np.float64)[pos]
        q, n, dev = find_quantum(t, **kwargs)
        # find_quantum dropped zero values; put the counts back in the order
        # of the rows, with 0 for any zero value
        counts = np.zeros(len(t))
        counts[t > 0] = n
        # .iloc[pos, column number] addresses rows by position
        out.iloc[pos, out.columns.get_loc('count')] = counts
        out.iloc[pos, out.columns.get_loc('quantum')] = q
        rows.append({id_col: sid, 'quantum': q, 'n_events': counts.sum(),
                     'n_bins': int(np.sum(t > 0)), 'max_dev': dev})
    summary = pd.DataFrame(rows)
    if summary['max_dev'].max() > 0.01:
        warnings.warn('recover_counts: some spectra are not exact multiples '
                      'of a single scale factor (see summary.max_dev)')
    return out, summary


def recover_counts_wide(F, **kwargs):
    '''
    The same for a 2-d array F with one row per spectrum and one column per
    bin (the layout of the test file). Returns (counts, q, max_dev): an array
    of the same shape as F, and two arrays with one entry per spectrum.
    '''
    F = np.asarray(F, dtype=np.float64)
    counts = np.zeros_like(F)
    q = np.zeros(len(F))
    max_dev = np.zeros(len(F))
    for i, t in enumerate(F):
        q[i], n, max_dev[i] = find_quantum(t, **kwargs)
        counts[i, t > 0] = n
    return counts, q, max_dev


def add_empty_bins(df, id_col='ID', x_col='mhh', target='sigma'):
    '''
    Restore the bins with no events. The training file has one row per
    non-empty bin; under a Poisson likelihood an empty bin is information
    (it says the cross section there is small), so the likelihood needs it.

    The bins are taken to be all the distinct values of x_col that occur in
    df. For each spectrum, every missing bin is added as a new row with the
    spectrum's other columns (its HEFT parameters, and, if present, its
    'quantum') copied, target = 0 and, if present, 'count' = 0.
    Returns a new data frame, sorted by id_col and x_col.
    '''
    grid = np.sort(df[x_col].unique())
    new_rows = []
    for sid, g in df.groupby(id_col):
        # np.setdiff1d(a, b): the values of a that are not in b
        missing = np.setdiff1d(grid, g[x_col].to_numpy())
        if len(missing) == 0:
            continue
        # one copy of the spectrum's first row for each missing bin
        extra = pd.concat([g.iloc[[0]]] * len(missing), ignore_index=True)
        extra[x_col] = missing
        extra[target] = 0.0
        if 'count' in extra:
            extra['count'] = 0.0
        new_rows.append(extra)
    if not new_rows:
        return df.copy()
    full = pd.concat([df] + new_rows, ignore_index=True)
    return full.sort_values([id_col, x_col]).reset_index(drop=True)


def poisson_variance(df, target='sigma', prediction=None):
    '''
    The Poisson noise variance of every row: q * t, with q the 'quantum'
    column made by recover_counts. If a prediction f (one value per row) is
    given, q * f is returned instead: the variance the model itself implies,
    which is the natural choice for a Poisson likelihood and is not zero in
    empty bins. Negative predictions are set to zero.
    '''
    q = df['quantum'].to_numpy(np.float64)
    if prediction is None:
        return q * df[target].to_numpy(np.float64)
    return q * np.clip(np.asarray(prediction, dtype=np.float64), 0.0, None)
