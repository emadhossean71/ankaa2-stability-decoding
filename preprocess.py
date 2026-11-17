"""Riverlane's Ankaa-2 stability-8 data (no resets)  ->  detection events.

Reads stability_8_without_resets_raw_data.h5 (from this folder or the one above) and writes
detectors.npz into this folder. For every run (circuit_2 ... circuit_24) it holds:

    dets    (shots, detectors)   1 = this detector fired in this shot
    obs     (shots,)             1 = the logical observable flipped
    fpga    (shots,)             1 = Riverlane's FPGA decoder predicted a flip
    anc     (detectors,)         the ancilla each detector watches: 36, 38, 50 or 52
    t       (detectors,)         the detector round, 0 = first
    rounds, utc, run_s           rounds in the circuit, time stamp, run time [s]

Then it checks the result against the numbers Riverlane published next to the data.
The reading itself is core.read_runs(), shared with preprocess_resets.py. Only the hard (0/1)
readings are used; explore_readings.py looks at the soft (IQ) readings.
"""
import numpy as np

import ankaa_core as core

runs, out = core.read_runs("stability_8_without_resets_raw_data.h5")
np.savez_compressed(core.out("detectors.npz"), **out)
print(f"wrote detectors.npz with {len(runs)} runs\n")

# ---- checks against the numbers Riverlane published next to the data ----
lep = {int(r): v for r, v, err in np.loadtxt(core.data("lep_fpga.txt"), skiprows=1)}
print(f"{'run':<12}{'rounds':>7}{'detectors':>10}{'firing rate':>12}{'FPGA LER':>10}{'Riverlane':>11}")
for run in runs:
    d, o, p, t = out[run + "/dets"], out[run + "/obs"], out[run + "/fpga"], out[run + "/t"]
    n = t.max() + 1                          # detector rounds = "decoding rounds" in lep_fpga.txt
    theirs = f"{lep[n]:.6f}" if n in lep else "-"
    print(f"{run:<12}{out[run + '/rounds']:>7}{d.shape[1]:>10}{d.mean():>12.4f}{(p != o).mean():>10.6f}{theirs:>11}")

ref = np.loadtxt(core.data("defect_rates_per_round.txt"), skiprows=1)    # columns: round, 50, 52, 36, 38
d, anc, t = out["circuit_24/dets"], out["circuit_24/anc"], out["circuit_24/t"]
mine = np.array([[d[:, (anc == q) & (t == r)].mean() for q in (50, 52, 36, 38)] for r in range(len(ref))])
print("\ncircuit_24, per-ancilla firing rates vs defect_rates_per_round.txt:")
print("largest difference =", np.abs(mine - ref[:, 1:]).max())
