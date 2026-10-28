# dihiggs HEFT di-Higgs model — numpy version

A self-contained copy of the trained `dihiggs` model
([github.com/hbprosper/dihiggs](https://github.com/hbprosper/dihiggs)) that needs
only **numpy** and the Python standard library. No PyTorch.

| file | contents |
| :--- | :--- |
| `heftnet_numpy.py` | the model (one class, `HEFTNet`) |
| `heftnet_weights.txt` | the trained parameters, in plain text |
| `reference.txt` | inputs and outputs of the original PyTorch model |
| `example.py` | prints the SM spectrum and checks the model against `reference.txt` |

## What it computes

For a di-Higgs mass `mhh` in GeV and the HEFT Wilson coefficients
`theta = (klambda, ct, ctt, cggh, cgghh)`, the model returns the gluon-fusion
di-Higgs cross section **in pb in a 15 GeV wide bin centred at `mhh`**
(pp at 13 TeV, POWHEG NLO). Divide by 15 for dσ/dm_hh in pb/GeV.
The Standard Model is `theta = (1, 1, 0, 0, 0)`.

The model was trained for 255 ≤ m_hh ≤ 1455 GeV; outside that range it
extrapolates, and `HEFTNet` issues a warning.

## Usage

```python
import numpy as np
from heftnet_numpy import HEFTNet

model = HEFTNet("heftnet_weights.txt")

model(400.0, [1, 1, 0, 0, 0])            # one point -> float
mhh = np.arange(262.5, 1455, 15)          # the 80 bin centres (GeV)
model(mhh, [1, 1, 0, 0, 0])               # one spectrum -> array [80]
model(mhh_array, theta_array)             # [N] and [N, 5] -> array [N]

model.coeffs(mhh)                         # the 23 learned functions a_i(m_hh), [N, 23]
HEFTNet.monomials(theta)                  # the 23 monomials c_i(theta), [N, 23]
```

## Checking your copy

```bash
python example.py
```

ends with `PASS` if the model reproduces the original PyTorch values.

## How it works

`sigma = sum_i c_i(theta) * a_i(m_hh)`: the dependence on the Wilson coefficients
is a known polynomial with 23 terms, and a small neural network supplies the 23
functions of the mass, `a_i = P_i(x) * exp(Q_i(x))` with
`x = (m_hh - 255 GeV) / 1500 GeV`. The code is short and commented; see
`docs/dihiggs_model.pdf` in the repository for the physics.
