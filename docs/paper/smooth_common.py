# ------------------------------------------------------------------------------
# smooth_common.py: shared by the paper figures in docs/paper.
#
# Each figure script runs this file with
#       exec(open('smooth_common.py').read())
# which defines, in the script's own namespace, the smooth model's predictions
# for the 1,000 test spectra and the quantities derived from them (listed at
# the end of this file).  Run the figure scripts from docs/paper.
#
# Inputs (all in the repository; nothing outside it is read):
#   code/runs/SMOOTH_J40/smooth_model.npz        the fitted model (heft_smooth.ipynb)
#   data/heft_wide_gauss_testdata_13TeV.csv      the 1,000 test spectra
#   data/heft_wide_gauss_traindata_13TeV.csv.gz  the training rows (training_rows(), quick_test_rows())
#   code/runs/ENS_bootstrap_huber_d2_1/member_*  the 20 networks (network_bins(), network_density())
#   code/runs/<run>                              a density network (load_density_network(run))
#
# The predictions take about a minute to compute, so they are cached in
# docs/paper/smooth_test.npz.  The cache is rebuilt automatically when it is
# missing or older than the model file or the test file; delete it to force
# a rebuild.
# ------------------------------------------------------------------------------
import os, sys, glob
import numpy as np, pandas as pd

# Where am I?  With exec(...), __file__ is the name of the figure script that
# ran this file; both live in docs/paper, so the repository root is two
# directories up.  (If __file__ is not defined, e.g. in an interactive
# session, the current directory is assumed to be docs/paper.)
try:
    PAPER = os.path.dirname(os.path.abspath(__file__))
except NameError:
    PAPER = os.getcwd()
REPO = os.path.abspath(os.path.join(PAPER, '..', '..'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)          # so that "import dihiggs" finds the repository's package

from dihiggs.smooth import SmoothHEFT, heft_monomials
from dihiggs import counts as CN

MODEL_FILE = os.path.join(REPO, 'code', 'runs', 'SMOOTH_J40', 'smooth_model.npz')
TEST_FILE  = os.path.join(REPO, 'data', 'heft_wide_gauss_testdata_13TeV.csv')
TRAIN_FILE = os.path.join(REPO, 'data', 'heft_wide_gauss_traindata_13TeV.csv.gz')
NET_DIR    = os.path.join(REPO, 'code', 'runs', 'ENS_bootstrap_huber_d2_1')
CACHE      = os.path.join(PAPER, 'smooth_test.npz')

EDGES   = 255.0 + 15.0 * np.arange(81)     # the 80 bins of 15 GeV, 255-1455 GeV
CENTRES = EDGES[:-1] + 7.5
THETA_COLS = ['klambda', 'CT', 'CTT', 'CGGH', 'CGGHH']


def load_test_spectra():
    '''
    The 1,000 test spectra: T [nS, 80] (pb per bin), q_t [nS] (pb per event),
    theta [nS, 5] and the spectrum IDs [nS].
    '''
    spectra = pd.read_csv(TEST_FILE)
    bins = [c for c in spectra.columns if c.isdigit()]     # the columns '17' ... '96'
    assert len(bins) == 80 and bins[0] == '17', 'unexpected bin columns in the test file'
    T = spectra[bins].to_numpy(np.float64)
    _, q_t, _ = CN.recover_counts_wide(T)                  # cross section per event, per spectrum
    # theta passes through float32 exactly as in heft_smooth.ipynb, so that the
    # C_i(theta) are bit-for-bit those used when the model was evaluated there
    theta = spectra[THETA_COLS].to_numpy(np.float32).astype(np.float64)
    return T, q_t, theta, spectra['ID'].to_numpy()


def _build_cache():
    print(f'smooth_common: building {os.path.basename(CACHE)} from the repository files ...')
    model = SmoothHEFT.load(MODEL_FILE)
    T, qt, th, ids = load_test_spectra()
    nS = len(T)
    C = heft_monomials(th)                                  # [nS, 23]; C_i(theta) does not depend on m

    # bin integrals of the model and their fit variance, for every (spectrum, bin) pair
    I = model.basis.integrals(EDGES)                        # [80, J]
    f, v = model.predict(np.repeat(C, 80, axis=0), np.tile(I, (nS, 1)))
    f, v = f.reshape(nS, 80), v.reshape(nS, 80)
    A = model.coefficient_integrals(I)                      # [80, 23]: integrals of the a_i over each bin
    S = np.abs(C[:, None, :] * A[None]).sum(2)              # [nS, 80]: sum_i |C_i A_ik|
    eps = model.epsilon(CENTRES)                            # model-error coefficient at the bin centres

    # the density d sigma/dm on a fine grid, with its fit variance.  For each m,
    # K(m)_ik = sum_jl B_j(m) Hinv[(i,j),(k,l)] B_l(m) is the 23x23 covariance
    # of the a_i(m); the variance of sum_i C_i a_i(m) is then C K C^T.
    mg = np.linspace(252.5, 1455.0, 482)
    V = model.basis.values(mg)                              # [M, J], includes the threshold factor phi(m)
    B = model.beta.reshape(23, model.J)
    a = V @ B.T                                             # [M, 23] in pb/GeV
    Hb = model.H_inverse().reshape(23, model.J, 23, model.J)
    K = np.einsum('mj,ijkl,ml->mik', V, Hb, V)              # [M, 23, 23]
    dens = C @ a.T                                          # [nS, M]
    dvar = np.einsum('si,mik,sk->sm', C, K, C)
    dS = np.abs(C[:, None, :] * a[None]).sum(2)             # [nS, M]: sum_i |C_i a_i(m)|
    epsg = model.epsilon(mg)

    np.savez(CACHE, T=T, qt=qt, C=C, f=f, v=v, S=S, eps=eps, mg=mg, dens=dens, dvar=dvar,
             dS=dS, epsg=epsg, th=th, id=ids)


def _cache_is_stale():
    if not os.path.exists(CACHE):
        return True
    t_cache = os.path.getmtime(CACHE)
    return any(os.path.getmtime(x) > t_cache for x in (MODEL_FILE, TEST_FILE))


_SPLIT = None
def _split_rows():
    '''
    Read the training file and split it exactly as heft_smooth.ipynb does:
    'fit' = the rows the smooth model was fitted to (training + validation
    rows, plus the restored empty bins); 'quick' = the 12% quick-test rows,
    never used in the fit, on which the model-error coefficient epsilon(m)
    was determined.  Read once and kept.
    '''
    global _SPLIT
    if _SPLIT is None:
        df = pd.read_csv(TRAIN_FILE)
        N = len(df); tr = int(0.864 * N); te = int(0.120 * N); va = N - tr - te
        counted, _ = CN.recover_counts(df)
        full = CN.add_empty_bins(counted)
        fit = pd.concat([counted.iloc[:tr + va], full[full['count'] == 0]], ignore_index=True)
        _SPLIT = {'fit': fit, 'quick': counted.iloc[tr + va:]}
    return _SPLIT


def _arrays(frame):
    C = heft_monomials(frame[THETA_COLS].to_numpy(np.float64))
    b = np.rint(frame['mhh'].to_numpy() * 100 - 0.5).astype(int)   # mhh = (m - 255 GeV)/1500 GeV at the bin centre
    return C, b, frame.sigma.to_numpy(), frame.quantum.to_numpy()


def training_rows():
    '''
    The rows the smooth model was fitted to, built exactly as in
    heft_smooth.ipynb: C [N, 23], bin index b [N], cross section t [N] (pb),
    quantum q [N] (pb per event).  Used for the free per-bin fit.
    '''
    return _arrays(_split_rows()['fit'])


def quick_test_rows():
    '''The quick-test rows (same four arrays as training_rows()), used to determine epsilon(m).'''
    return _arrays(_split_rows()['quick'])


def load_density_network(run):
    '''
    A trained density network (dihiggs.heftnet.HEFTDensity), rebuilt from the
    run folder's dihiggs_config.yaml (activation, threshold factor, nodes per
    bin, input variable) and loaded with its parameters, in evaluation mode,
    on the CPU, in double precision. run: a folder name in code/runs (e.g.
    'FTD_SiLU_huber_d2_pearson') or a path to a run folder.
    '''
    import yaml
    from dihiggs.heftnet import HEFTDensity
    folder = run if os.path.isdir(run) else os.path.join(REPO, 'code', 'runs', run)
    cfg = yaml.safe_load(open(os.path.join(folder, 'dihiggs_config.yaml')))
    assert cfg.get('model') == 'density', f'{run} is not a density-model run'
    d = cfg.get('density', {})
    net = HEFTDensity(activation=cfg['activation'], threshold=d.get('threshold', True),
                      n_nodes=d.get('n_nodes', 8), input_map=d.get('input_map', 'sqrt'))
    net.load(os.path.join(folder, 'dihiggs_params.pth'))     # drwho's Model.load: CPU, eval mode
    return net.double()


def _networks():
    '''The 20 fine-tuned bootstrap networks (HEFTNet), in evaluation mode.'''
    from dihiggs import heftnet as NN                       # needs torch; imported only when used
    files = sorted(glob.glob(os.path.join(NET_DIR, 'member_*', 'dihiggs_params.pth')))
    assert len(files) > 0, f'no networks found in {NET_DIR}'
    nets = []
    for fn in files:
        net = NN.HEFTNet(); net.load(fn); net.eval(); nets.append(net)
    return nets


def network_density(theta, m):
    '''
    The networks' prediction at the masses m (GeV) for each HEFT point:
    an array [n_networks, n_points, len(m)] in pb per 15 GeV.  Divide by 15
    for pb/GeV.  The input variable of HEFTNet is x = (m - 255 GeV)/1500 GeV.
    '''
    import torch
    theta = np.atleast_2d(theta); x = (np.asarray(m, np.float64) - 255.0) / 1500.0
    nets = _networks()
    out = np.zeros((len(nets), len(theta), len(x)))
    for a, net in enumerate(nets):
        for b, th_ in enumerate(theta):
            X = torch.tensor(np.column_stack([x, np.tile(th_, (len(x), 1))]), dtype=torch.float32)
            with torch.no_grad():
                out[a, b] = net(X).double().numpy()
    return out


def network_bins(theta):
    '''The networks' bin predictions (pb per 15 GeV bin, at the bin centres): [n_networks, n_points, 80].'''
    return network_density(theta, CENTRES)


# ------------------------------------------------------------------------------
# Build (if needed) and load the cache; define the quantities the figures use
# ------------------------------------------------------------------------------
if _cache_is_stale():
    _build_cache()
D = dict(np.load(CACHE))
T, qt, f, v, S, eps = D['T'], D['qt'], D['f'], D['v'], D['S'], D['eps']

# Variances per (spectrum, bin), all in pb^2:
#   noise : Poisson noise of the POWHEG sample, from the model: q * max(f, q)
#   v     : fit uncertainty of the model (posterior variance of the bin integral)
#   disc2 : the model-error term (systematic), (epsilon(m) * sum_i |C_i A_ik|)^2
noise = qt[:, None] * np.maximum(f, qt[:, None])
disc2 = (eps[None] * S)**2

# Likelihood-ratio statistic -2 ln lambda per bin, lambda = Poisson(n; mu)/Poisson(n; n),
# n = POWHEG event count, mu = model prediction in events.  (np.where guards the
# n = 0 and mu <= 0 cases; bins with mu <= 0 give NaN.)  The variable keeps its
# old name 'deviance' because the figure scripts use it.
n = T / qt[:, None]; mu = f / qt[:, None]; pos = mu > 0
deviance = np.where(pos, 2 * (mu - n + np.where(n > 0, n * np.log(np.where(n > 0, n, 1) / np.where(pos, mu, 1)), 0)), np.nan)
devN = np.nanmean(deviance, 1)                       # mean per bin, for each spectrum

# Pull with all three variances (bins where both POWHEG and the model are zero are skipped)
z = np.where((T > 0) | (f > 0), (T - f) / np.sqrt(noise + v + disc2), np.nan)
