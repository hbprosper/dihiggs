# Run folder naming convention

Each subfolder of `code/runs/` holds the saved model, loss history, and
configuration for one training run. The folder name encodes the key settings.

## Scheme

```
<loss>_<curriculum?>_<difficulty>_<pacing>
```

| segment | values | meaning |
| :--- | :--- | :--- |
| `<loss>` | `L1` · `MSE` | loss function (`L1` = `nn.L1Loss`, `MSE` = `nn.MSELoss`) |
| `_pos` (optional) | present / absent | positivity penalty included |
| `_curr` | present / absent | curriculum learning active |
| `<difficulty>` | `b` · `s` · `snr` | difficulty criterion (see below) |
| `<pacing>` | `lin` · `log` | ramp shape (see below) |

### Difficulty codes

| code | full name | sort key |
| :--- | :--- | :--- |
| `b` | `bin_sigma` | cross section of the individual bin, σₖ |
| `s` | `spectrum_sigma` | total cross section of the parent spectrum, σ_tot |
| `snr` | `spectrum_snr` | σ_tot / √w_j (not yet implemented) |

Rows are sorted **descending**: largest (easiest, highest SNR) first.

### Pacing codes

| code | full name | formula |
| :--- | :--- | :--- |
| `lin` | `linear` | f(i) = f₀ + (1 − f₀) · i / i_ramp |
| `log` | `log` | f(i) = f₀^(1 − i/i_ramp) — slow start, then faster |

In both cases f(0) = START_FRACTION and f(i_ramp) = 1.

## Examples

| folder | meaning |
| :--- | :--- |
| `L1Loss` | plain `nn.L1Loss`, no curriculum (baseline) |
| `L1_curr_b_log` | `nn.L1Loss`, curriculum on, `bin_sigma` difficulty, log pacing |
| `L1_curr_b_lin` | `nn.L1Loss`, curriculum on, `bin_sigma` difficulty, linear pacing |
| `L1_curr_s_log` | `nn.L1Loss`, curriculum on, `spectrum_sigma` difficulty, log pacing |
| `L1_pos_curr_b_log` | same as above with positivity penalty |

## Contents of each run folder

| file | description |
| :--- | :--- |
| `dihiggs_config.yaml` | all hyperparameters recorded at run time |
| `dihiggs_losses.csv` | train / validation loss per monitor step |
| `dihiggs_params.pt` | saved model weights (best validation loss) |
