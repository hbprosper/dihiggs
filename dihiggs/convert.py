"""
dihiggs.convert -- write self-contained, non-PyTorch versions of a trained HEFTNet.

Given the configuration file written by heft_train.ipynb (which points to the
trained parameters), this module writes

    <outdir>/python/   heftnet_numpy.py  example.py   heftnet_weights.txt  reference.txt  README.md
    <outdir>/cpp/      heftnet.hpp       example.cpp  heftnet_weights.txt  reference.txt  README.md

  heftnet_weights.txt  the trained parameters in a plain-text format that both
                       versions read (described at the top of the file itself)
  reference.txt        inputs and outputs of the original PyTorch model, which
                       example.py / example.cpp use to check the copy
  heftnet_numpy.py     the model in numpy (numpy + Python standard library only)
  heftnet.hpp          the model in header-only C++17 (standard library only)

The model code is not generated: it is fixed, hand-written code in
dihiggs/templates that reads the architecture (number of layers, their sizes,
the activation) from the weights file, so retraining needs no code changes.

After writing, the copies are validated against the PyTorch model, computed in
double precision: the numpy version on every point of the held-out test set and
the C++ version (if a C++ compiler is available) on the reference points.

Usage, from Python or a notebook:

    from dihiggs import convert
    convert.export('runs/L1Loss/dihiggs_config.yaml', outdir='exported')

or from the command line:

    python -m dihiggs.convert runs/L1Loss/dihiggs_config.yaml --out exported
"""
import argparse
import datetime
import hashlib
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from dihiggs import heftnet as NN

TEMPLATES = Path(__file__).parent / 'templates'
FORMAT = 'dihiggs-heftnet-text-v1'

# Mass mapping used when the training data were prepared (heft_prepare_data.ipynb).
# POWHEG bin k covers [15k, 15(k+1)) GeV; bins 17-96 are kept and mapped to
# x = (m_hh - 255 GeV) / 1500 GeV, so that one 15 GeV bin is 0.01 in x.
MHH_MIN_GEV = 255.0
MHH_MAX_GEV = 1455.0
MHH_SCALE_GEV = 1500.0

# activation names in heftnet.py / the training configuration -> weights file
ACTIVATIONS = {'nn.ReLU': 'relu', 'nn.Tanh': 'tanh', 'Sin': 'sin'}

# files copied from dihiggs/templates: (template name, name in the output folder)
FILES = {
    'python': [('heftnet_numpy.py', 'heftnet_numpy.py'),
               ('example.py', 'example.py'),
               ('README_python.md', 'README.md')],
    'cpp':    [('heftnet.hpp', 'heftnet.hpp'),
               ('example.cpp', 'example.cpp'),
               ('README_cpp.md', 'README.md')],
}

N_COEFFS = 23     # monomials c_i(theta) / learned functions a_i(m_hh)

TESTDATA = 'heft_wide_gauss_testdata_13TeV.csv'
PARAMS = ['klambda', 'CT', 'CTT', 'CGGH', 'CGGHH']

# A copy must agree with PyTorch (in double precision) to this relative accuracy.
TOLERANCE = 1e-8


# ---------------------------------------------------------------------------
# Loading the trained model
# ---------------------------------------------------------------------------
def _resolve(path, config_file):
    """Paths in the configuration are relative to the folder training ran in,
    which is two levels above the configuration file (code/runs/<name>/)."""
    path = Path(path)
    if path.is_absolute() or path.exists():
        return path
    return Path(config_file).resolve().parents[2] / path


def load_trained(config_file):
    """
    Rebuild the trained HEFTNet from its configuration file.

    The width and number of hidden layers are read off the saved parameters,
    the activation from the configuration. Returns a dictionary with the model
    (in double precision, eval mode), its layers as numpy arrays, the
    activation name for the weights file and the parameter file used.
    """
    with open(config_file) as f:
        cfg = yaml.safe_load(f)

    params_file = _resolve(cfg['file']['params'], config_file)
    state = torch.load(params_file, map_location='cpu', weights_only=True)

    # The Linear layers of P are P.0, P.2, ..., in order (odd indices are the
    # activations, which have no parameters).
    p_index = sorted({int(k.split('.')[1]) for k in state
                      if k.startswith('P.') and k.endswith('.weight')})
    width = state[f'P.{p_index[0]}.weight'].shape[0]
    hidden = len(p_index) - 2

    activation = cfg.get('activation', NN.ACTIVATION)
    if activation not in ACTIVATIONS:
        raise ValueError(f'activation {activation} is not supported; '
                         f'supported: {list(ACTIVATIONS)}')

    model = NN.HEFTNet(width=width, hidden=hidden, activation=activation)
    model.load_state_dict(state)
    model.double().eval()

    def as_numpy(name):
        return state[name].detach().cpu().numpy()

    layers = [(as_numpy(f'P.{i}.weight'), as_numpy(f'P.{i}.bias')) for i in p_index]
    Q = (as_numpy('Q.weight'), as_numpy('Q.bias'))

    return dict(model=model, layers=layers, Q=Q,
                activation=ACTIVATIONS[activation],
                params_file=Path(params_file), config_file=Path(config_file))


def torch_sigma(model, mhh, theta):
    """Cross section from the PyTorch model, for mhh in GeV and theta [N, 5]."""
    x = (np.asarray(mhh, dtype=np.float64) - MHH_MIN_GEV) / MHH_SCALE_GEV
    inputs = np.column_stack((x, theta))
    with torch.no_grad():
        return model(torch.from_numpy(inputs)).numpy()


# ---------------------------------------------------------------------------
# Writing the weights and reference files
# ---------------------------------------------------------------------------
def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_weights(trained, filename):
    """Write the parameters in the plain-text format read by both copies."""
    layers, Q = trained['layers'], trained['Q']

    def numbers(values):
        # repr gives the shortest decimal string that reads back as exactly
        # the same double, so the copies use bit-for-bit the trained weights.
        return ' '.join(repr(float(v)) for v in values)

    out = [
        '# dihiggs HEFTNet parameters (plain text)',
        f'# written by dihiggs.convert on {datetime.date.today().isoformat()}',
        f'# from {trained["params_file"]}',
        f'# sha256 {_sha256(trained["params_file"])}',
        '#',
        '# Format: whitespace-separated tokens; text after "#" is a comment.',
        '#   key value                      a header entry',
        '#   tensor NAME NDIM D1 ... DNDIM  followed by D1*...*DNDIM numbers in',
        '#                                  row-major order (one matrix row per line)',
        '# The network P has p_layers fully connected layers P.0, P.1, ...;',
        '# P.i.weight has shape [n_out, n_in]. Hidden layers are followed by',
        '# "activation", the last one by tanh. Q is one linear layer. The input is',
        '# x = (m_hh - mhh_min_GeV) / mhh_scale_GeV.',
        '#',
        f'format         {FORMAT}',
        f'activation     {trained["activation"]}',
        'output         tanh',
        f'p_layers       {len(layers)}',
        f'n_coeffs       {N_COEFFS}',
        f'mhh_min_GeV    {MHH_MIN_GEV:g}',
        f'mhh_max_GeV    {MHH_MAX_GEV:g}',
        f'mhh_scale_GeV  {MHH_SCALE_GEV:g}',
    ]

    def tensor(name, array):
        out.append('')
        out.append(f'tensor {name} {array.ndim} ' + ' '.join(map(str, array.shape)))
        rows = array if array.ndim == 2 else [array]
        out.extend(numbers(row) for row in rows)

    for i, (W, b) in enumerate(layers):
        tensor(f'P.{i}.weight', W)
        tensor(f'P.{i}.bias', b)
    tensor('Q.weight', Q[0])
    tensor('Q.bias', Q[1])

    Path(filename).write_text('\n'.join(out) + '\n')



def test_points(config_file, rng=None):
    """
    Return (mhh, theta) for the validation: every bin of every held-out test
    spectrum if the test data file can be found, otherwise random points
    spread like the training data.
    """
    import pandas as pd

    candidates = [Path(config_file).resolve().parents[3] / 'data' / TESTDATA,
                  Path('../data') / TESTDATA]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            bins = [c for c in df.columns if c.isdigit()]
            centres = np.array([15.0 * int(k) + 7.5 for k in bins])
            theta = np.repeat(df[PARAMS].to_numpy(np.float64), len(bins), axis=0)
            mhh = np.tile(centres, len(df))
            return mhh, theta, f'{len(df)} test spectra in {path}'

    rng = rng or np.random.default_rng(42)
    n = 40_000
    theta = rng.normal([1, 1, 0, 0, 0], [5, 2, 2, 2, 2], size=(n, 5))
    mhh = rng.choice(np.arange(262.5, 1455, 15), size=n)
    return mhh, theta, f'{n} random points (test data file not found)'


def write_reference(trained, filename):
    """
    Write reference inputs and outputs of the PyTorch model: the full spectrum
    (80 bins) for the Standard Model and for six other parameter points.
    """
    points = np.array([
        [1.0, 1.0, 0.0, 0.0, 0.0],       # Standard Model
        [5.0, 1.0, 0.0, 0.0, 0.0],
        [-2.0, 1.2, 0.5, 0.0, 0.0],
        [1.0, 0.8, -0.5, 0.3, 0.2],
        [2.4, 1.0, 0.0, 0.66, 0.4],
        [-8.4, -0.3, -0.3, 0.3, 0.5],
        [7.2, 2.9, 0.1, 0.1, -0.1],
    ])
    centres = np.arange(262.5, 1455, 15)
    theta = np.repeat(points, len(centres), axis=0)
    mhh = np.tile(centres, len(points))
    sigma = torch_sigma(trained['model'], mhh, theta)

    lines = ['# Reference values from the PyTorch model (double precision)',
             f'# parameters: {trained["params_file"]}',
             '# columns: mhh(GeV) klambda ct ctt cggh cgghh sigma(pb per 15 GeV bin)']
    for m, t, s in zip(mhh, theta, sigma):
        lines.append(' '.join(repr(float(v)) for v in (m, *t, s)))
    Path(filename).write_text('\n'.join(lines) + '\n')


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def relative_error(values, reference):
    """|values - reference| / |reference|, protected where reference ~ 0."""
    scale = np.abs(reference).max()
    return np.abs(values - reference) / (np.abs(reference) + 1e-12 * scale)


def validate_python(trained, folder):
    """Compare heftnet_numpy.py with PyTorch on the test points, then run example.py."""
    spec = importlib.util.spec_from_file_location('heftnet_numpy',
                                                  folder / 'heftnet_numpy.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    copy = module.HEFTNet(folder / 'heftnet_weights.txt')
    mhh, theta, description = test_points(trained['config_file'])
    reference = torch_sigma(trained['model'], mhh, theta)
    error = relative_error(copy(mhh, theta), reference).max()
    print(f'  numpy vs PyTorch on {description}:')
    print(f'  largest relative difference {error:.2e}')
    if error > TOLERANCE:
        raise RuntimeError('the numpy copy does not reproduce the PyTorch model')

    run = subprocess.run([sys.executable, 'example.py'], cwd=folder,
                         capture_output=True, text=True)
    print('  example.py: ' + (run.stdout.strip().splitlines() or ['?'])[-1])
    if run.returncode != 0:
        raise RuntimeError('example.py failed:\n' + run.stdout + run.stderr)
    shutil.rmtree(folder / '__pycache__', ignore_errors=True)


def validate_cpp(folder):
    """Compile and run example.cpp, if a C++ compiler is available."""
    compiler = next((c for c in ('c++', 'g++', 'clang++') if shutil.which(c)), None)
    if compiler is None:
        print('  no C++ compiler found: C++ validation skipped')
        return
    build = subprocess.run([compiler, '-std=c++17', '-O2', '-Wall', '-o', 'example',
                            'example.cpp'], cwd=folder, capture_output=True, text=True)
    if build.returncode != 0:
        raise RuntimeError('example.cpp did not compile:\n' + build.stderr)
    run = subprocess.run(['./example'], cwd=folder, capture_output=True, text=True)
    print(f'  example.cpp ({compiler}): ' + (run.stdout.strip().splitlines() or ['?'])[-1])
    if run.returncode != 0:
        raise RuntimeError('example.cpp failed:\n' + run.stdout + run.stderr)
    (folder / 'example').unlink()                 # do not ship the binary


# ---------------------------------------------------------------------------
# Writing the copies
# ---------------------------------------------------------------------------
def _write(language, config_file, outdir, validate, trained=None):
    trained = trained or load_trained(config_file)
    folder = Path(outdir) / language
    folder.mkdir(parents=True, exist_ok=True)

    for template, name in FILES[language]:
        shutil.copy(TEMPLATES / template, folder / name)
    write_weights(trained, folder / 'heftnet_weights.txt')
    write_reference(trained, folder / 'reference.txt')
    print(f'wrote {language} version to {folder}/')

    if validate:
        if language == 'python':
            validate_python(trained, folder)
        else:
            validate_cpp(folder)
    return folder


def to_python(config_file, outdir='exported', validate=True):
    """Write the numpy version to <outdir>/python/."""
    return _write('python', config_file, outdir, validate)


def to_cpp(config_file, outdir='exported', validate=True):
    """Write the C++ version to <outdir>/cpp/."""
    return _write('cpp', config_file, outdir, validate)


def export(config_file, outdir='exported', languages=('python', 'cpp'), validate=True):
    """Write the requested versions (both by default)."""
    trained = load_trained(config_file)
    return [_write(language, config_file, outdir, validate, trained)
            for language in languages]


def main():
    parser = argparse.ArgumentParser(
        description='Write numpy and C++ versions of a trained dihiggs HEFTNet.')
    parser.add_argument('config', help='configuration file, e.g. '
                                       'runs/L1Loss/dihiggs_config.yaml')
    parser.add_argument('--out', default='exported', help='output folder')
    parser.add_argument('--lang', nargs='+', choices=['python', 'cpp'],
                        default=['python', 'cpp'])
    parser.add_argument('--no-validate', action='store_true')
    args = parser.parse_args()
    export(args.config, args.out, args.lang, validate=not args.no_validate)


if __name__ == '__main__':
    main()
