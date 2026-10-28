# dihiggs HEFT di-Higgs model — C++ version

A self-contained copy of the trained `dihiggs` model
([github.com/hbprosper/dihiggs](https://github.com/hbprosper/dihiggs)) as a single
header that uses **only the C++17 standard library**. No PyTorch, no other
libraries. It also works in ROOT macros.

| file | contents |
| :--- | :--- |
| `heftnet.hpp` | the model (header-only; class `dihiggs::HEFTNet`) |
| `heftnet_weights.txt` | the trained parameters, in plain text |
| `reference.txt` | inputs and outputs of the original PyTorch model |
| `example.cpp` | prints the SM spectrum and checks the model against `reference.txt` |

## What it computes

For a di-Higgs mass `mhh` in GeV and the HEFT Wilson coefficients
`(klambda, ct, ctt, cggh, cgghh)`, the model returns the gluon-fusion di-Higgs
cross section **in pb in a 15 GeV wide bin centred at `mhh`** (pp at 13 TeV,
POWHEG NLO). Divide by 15 for dσ/dm_hh in pb/GeV. The Standard Model is
`{1, 1, 0, 0, 0}`.

The model was trained for 255 ≤ m_hh ≤ 1455 GeV; use `model.in_range(mhh)` to
check whether a mass is inside that range.

## Usage

```cpp
#include "heftnet.hpp"

dihiggs::HEFTNet model("heftnet_weights.txt");
dihiggs::Theta   theta{1, 1, 0, 0, 0};     // klambda, ct, ctt, cggh, cgghh

double sigma = model(400.0, theta);        // pb per 15 GeV bin at m_hh = 400 GeV
auto a = model.coeffs(400.0);              // the 23 learned functions a_i(m_hh)
auto c = dihiggs::HEFTNet::monomials(theta);  // the 23 monomials c_i(theta)
```

## Building and checking the example

```bash
g++ -std=c++17 -O2 -o example example.cpp      # or clang++
./example
```

ends with `PASS` if the model reproduces the original PyTorch values.

In ROOT: `.L heftnet.hpp` (or `#include` it in a macro), then use it as above.

## How it works

`sigma = sum_i c_i(theta) * a_i(m_hh)`: the dependence on the Wilson coefficients
is a known polynomial with 23 terms, and a small neural network supplies the 23
functions of the mass, `a_i = P_i(x) * exp(Q_i(x))` with
`x = (m_hh - 255 GeV) / 1500 GeV`. The code is short and commented; see
`docs/dihiggs_model.pdf` in the repository for the physics.
