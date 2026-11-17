"""Riverlane's Ankaa-2 stability-8 data WITH ancilla resets  ->  detection events.

Reads stability_8_with_resets_raw_data.h5 (from this folder or the one above) and writes
detectors_with_resets.npz into this folder, with dets, obs, anc, t, rounds, utc, run_s for every
run (circuit_2, 6, 10, 14, 18, 22). The reading is core.read_runs(), the same as preprocess.py;
the detectors come from Riverlane's own circuit, so the resets need no special handling.
These runs have no FPGA verdicts and no published defect rates to check against.
"""
import numpy as np

import ankaa_core as core

runs, out = core.read_runs("stability_8_with_resets_raw_data.h5")
np.savez_compressed(core.out("detectors_with_resets.npz"), **out)
print(f"wrote detectors_with_resets.npz with {len(runs)} runs\n")
for run in runs:
    d_, t_ = out[run + "/dets"], out[run + "/t"]
    print(f"{run:<12} readings {out[run + '/rounds']:>3}  detectors {d_.shape[1]:>4}  detector rounds {t_.max() + 1:>3}  "
          f"firing rate {d_.mean():.4f}  observable flips {out[run + '/obs'].mean():.4f}")
