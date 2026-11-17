"""The frozen procedure on Riverlane's WITH-resets stability data: an independent test.

Run preprocess_resets.py first. Every decoder piece comes from ankaa_core.py, unchanged from the
no-reset analysis: Riverlane's noise model with p = 0.03, the p_ij estimator and its clipping, the
blend with LAM = 0.75 (doors kept generic), the soft-information recipe (two Gaussians fitted by EM
on the learning half, fitted prior kept, negative evidence clipped), two-fold grading.

The two changes the resets force are already general in the shared module:
  - core.reading_detectors reads off the circuit which detectors each ancilla reading enters
    (with resets a misread of reading t fires two consecutive detectors);
  - core.soft_prices prices an edge the model also fills with other errors as "misread XOR the rest".
These runs have no FPGA verdicts, so there is no FPGA column. Takes about 5-10 minutes.
"""
import h5py
import numpy as np

import ankaa_core as core

d = np.load(core.data("detectors_with_resets.npz"))
hdf = h5py.File(core.data("stability_8_with_resets_raw_data.h5"), "r")
runs = ["circuit_6", "circuit_10", "circuit_14", "circuit_18", "circuit_22"]     # circuit_2 is too short, as before
RIVERLANE_BEST = {int(n): v for n, v, err in np.loadtxt(core.data("lep_mwpm_soft_info_pairwise_correlation_graph_with_resets.txt"), skiprows=1)}

rows, all_g, all_b = [], [], []
for run in runs:
    dets, obs = d[run + "/dets"], d[run + "/obs"]
    N, T = int(d[run + "/rounds"]), int(d[run + "/t"].max() + 1)
    circuit = core.load_circuit(hdf, run)
    generic, edges, w_generic = core.graph(circuit)

    wrong_generic = generic.decode_batch(dets)[:, 0] != obs
    wrong_pij, _, _ = core.two_fold(edges, dets, obs)
    wrong_blend = core.two_fold_blend(edges, w_generic, dets, obs)

    rmap = core.reading_detectors(circuit, core.chip_map(hdf, run, circuit))
    shots = np.arange(len(obs))
    keys, W_even, _ = core.soft_prices(hdf, run, w_generic, rmap, N, shots[0::2])     # clouds fitted on the even shots
    _, W_odd, _ = core.soft_prices(hdf, run, w_generic, rmap, N, shots[1::2])         # ... and on the odd shots
    W = {"even": W_even, "odd": W_odd}
    wrong_gs = core.two_fold_soft(edges, w_generic, keys, W, dets, obs, blend=False)
    wrong_bs = core.two_fold_soft(edges, w_generic, keys, W, dets, obs, blend=True)
    all_g.append(wrong_gs); all_b.append(wrong_bs)
    rows.append((run, T, wrong_generic.mean(), wrong_pij.mean(), wrong_blend.mean(), wrong_gs.mean(), wrong_bs.mean(),
                 RIVERLANE_BEST.get(T, float("nan")), core.paired(wrong_bs, wrong_gs)))
    print(f"done {run}", flush=True)

print(f"\n{'run':<12}{'rounds':>7}{'our MWPM':>10}{'p_ij':>8}{'blend':>8}{'MWPM+soft':>11}{'blend+soft':>12}{'Riverlane best':>16}"
      f"{'blend+soft - MWPM+soft [95%]':>32}")
for run, T, g, p, b, gs, bs, rl, (dd, lo, hi, pv) in rows:
    print(f"{run:<12}{T:>7}{g:>10.4f}{p:>8.4f}{b:>8.4f}{gs:>11.4f}{bs:>12.4f}{rl:>16.4f}     {dd:+.2f} [{lo:+.2f}, {hi:+.2f}] p={pv:.1e}")
mean = lambda i: np.mean([r[i] for r in rows])
print(f"{'pooled':<12}{'':>7}{mean(2):>10.4f}{mean(3):>8.4f}{mean(4):>8.4f}{mean(5):>11.4f}{mean(6):>12.4f}{mean(7):>16.4f}")
dd, lo, hi, pv = core.paired(np.concatenate(all_b), np.concatenate(all_g))
print(f"\nblend + soft beats Riverlane's best in {sum(r[6] < r[7] for r in rows)} of {len(rows)} runs, "
      f"by {min(r[7] - r[6] for r in rows) * 100:.2f} to {max(r[7] - r[6] for r in rows) * 100:.2f} points")
print(f"blend beats our MWPM (hard measurements only) in {sum(r[4] < r[2] for r in rows)} of {len(rows)} runs")
print(f"the blend improves on our MWPM + soft in {sum(r[6] < r[5] for r in rows)} of {len(rows)} runs; "
      f"pooled {dd:+.2f} points [{lo:+.2f}, {hi:+.2f}], p = {pv:.1e}")
