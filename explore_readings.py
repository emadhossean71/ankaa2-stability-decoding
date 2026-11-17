"""The two quick looks that used to sit at the end of preprocess.py, moved here unchanged.

1. Ancilla 36's soft (IQ) readings in circuit_24, coloured by the hard bit the hardware recorded
   (first 5,000 shots, all 26 readings).
2. Ancilla 36's firing rate in the first detector round of every run, next to the run's time
   stamp: the first sign of the drift across the session (guide, Section 9).

Run preprocess.py first (it writes detectors.npz).
"""
import h5py
import matplotlib.pyplot as plt
import numpy as np

import ankaa_core as core

# 1. IQ points of ancilla 36
with h5py.File(core.data("stability_8_without_resets_raw_data.h5"), "r") as f:
    r = f["circuit_24/result"]
    iq = r["soft_measurements/36"][:5000].ravel()    # ancilla 36, all 26 rounds
    bit = r["hard_measurements/36"][:5000].ravel()

plt.scatter(iq.real, iq.imag, c=bit, s=1, alpha=0.3, cmap="coolwarm")
plt.xlabel("I"); plt.ylabel("Q"); plt.title("Ancilla 36: soft readings, coloured by hard bit")
plt.show()

# 2. first-round firing rate against time
d = np.load(core.data("detectors.npz"))
print(f"{'run':<12}{'time':<10}{'firing rate':>12}")
for run in core.runs_of(d):
    dets, anc, t = d[run + "/dets"], d[run + "/anc"], d[run + "/t"]
    time = str(d[run + "/utc"])[11:19]                  # keep only hh:mm:ss
    rate = dets[:, (anc == 36) & (t == 0)].mean()
    print(f"{run:<12}{time:<10}{rate:>12.4f}")
