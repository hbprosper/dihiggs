"""
example.py -- use the numpy version of the dihiggs model, and check it.

    python example.py [heftnet_weights.txt] [reference.txt]

1. Prints the Standard Model spectrum at a few masses.
2. Compares the model with reference values computed by the original
   PyTorch model, and reports PASS or FAIL.
"""
import sys
import numpy as np
from heftnet_numpy import HEFTNet

weights = sys.argv[1] if len(sys.argv) > 1 else "heftnet_weights.txt"
reference = sys.argv[2] if len(sys.argv) > 2 else "reference.txt"

model = HEFTNet(weights)

# --- 1. the SM spectrum ------------------------------------------------------
SM = [1.0, 1.0, 0.0, 0.0, 0.0]        # klambda, ct, ctt, cggh, cgghh
mhh = np.arange(262.5, 1455, 15)       # the 80 bin centres, in GeV
spectrum = model(mhh, SM)              # pb per 15 GeV bin

print("Standard Model, sigma per 15 GeV bin")
print("  m_hh (GeV)   sigma (pb)")
for m, s in list(zip(mhh, spectrum))[::10]:
    print(f"  {m:9.1f}   {s:12.5e}")
print(f"  total over 255-1455 GeV: {spectrum.sum():.5e} pb\n")

# --- 2. check against the PyTorch reference values ---------------------------
# each row: mhh klambda ct ctt cggh cgghh sigma
ref = np.loadtxt(reference)
mhh, theta, sigma_ref = ref[:, 0], ref[:, 1:6], ref[:, 6]

sigma = model(mhh, theta)
scale = np.abs(sigma_ref).max()
error = np.abs(sigma - sigma_ref) / (np.abs(sigma_ref) + 1e-12 * scale)
ok = np.all(error < 1e-8)

print(f"checked {len(ref)} points against {reference}")
print(f"largest relative difference: {error.max():.2e}")
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
