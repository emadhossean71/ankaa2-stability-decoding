"""p_ij: learn every edge's error probability from the data, then decode with it (no resets).

Run after preprocess.py. For every run except circuit_2:

  1. get the decoding graph's edges from the circuit                         core.graph
  2. calculate p_ij twice: once from the even shots, once from the odd shots  core.pij
  3. decode each half with the p_ij learned from the OTHER half              core.two_fold
  4. repeat 2-3 with EVERY pair of detectors as an edge, not just the circuit's
  5. blend: keep the generic price on the boundary doors, move the pair prices
     a fraction LAM of the way towards p_ij (the doors are p_ij's weak spot)  core.two_fold_blend
  6. soft information: price every misread edge shot by shot from how far its
     reading's IQ point sat from the dividing line between the two clouds. The two
     clouds are fitted on the learning half only, like everything else that is learned.
     This is an approximation fitted from the experiment's own readings (Riverlane used
     a separate calibration run with prepared |0> and |1> states, which is not published)
                                                                              core.soft_prices, core.two_fold_soft
  7. print the edges for circuit_24, then every run's logical error rate next
     to the generic weights (Riverlane's noise model), the FPGA and Riverlane's decoders
"""
import h5py
import numpy as np

import ankaa_core as core

SOFT = True                              # soft-information columns; they add 15-30 minutes

# Riverlane's best decoder (MWPM + soft information + p_ij), per number of decoding rounds,
# from lep_mwpm_soft_info_pairwise_correlation_graph.txt, Zenodo record 15364358
RIVERLANE_SOFT = {5: 0.26818, 7: 0.24422, 9: 0.22745, 11: 0.22190, 13: 0.22143, 15: 0.20752,
                  17: 0.20747, 19: 0.19434, 21: 0.18241, 23: 0.19390, 25: 0.17623}

d = np.load(core.data("detectors.npz"))
hdf = h5py.File(core.data("stability_8_without_resets_raw_data.h5"), "r")
runs = core.runs_of(d)
runs.remove("circuit_2")                 # too short, and Riverlane published no number for it
FLOOR = core.FLOOR


def kind(u, v, anc, t):
    """Name a circuit edge by the footprint it leaves."""
    if v is None:
        return "boundary (one detector)"
    dt = abs(t[u] - t[v])
    if anc[u] == anc[v]:
        return {1: "same ancilla, next round", 2: "same ancilla, 2 rounds on"}[dt]
    return "neighbours, same round" if dt == 0 else f"neighbours, dt={dt}"


lep = {int(n): v for n, v, err in np.loadtxt(core.data("lep_mwpm.txt"), skiprows=1)}
rows = []
for run in runs:
    dets, obs, fpga = d[run + "/dets"], d[run + "/obs"], d[run + "/fpga"]
    anc, t = d[run + "/anc"], d[run + "/t"]
    circuit = core.load_circuit(hdf, run)

    # 1. circuit edges: (detector, detector or None for boundary, flips the observable?)
    generic, edges, w_generic = core.graph(circuit)

    # 4. every pair of detectors. No pair edge in this experiment flips the observable, so the added ones don't either.
    have = {(u, v) for u, v, _ in edges}
    n = dets.shape[1]
    every = edges + [(u, v, False) for u in range(n) for v in range(u + 1, n) if (u, v) not in have]

    # 2-3. learn on one half, decode the other, for both graphs; 5. the blend
    wrong_c, p_even, p_odd = core.two_fold(edges, dets, obs)
    wrong_a, pa_even, pa_odd = core.two_fold(every, dets, obs)
    wrong_b = core.two_fold_blend(edges, w_generic, dets, obs)
    wrong_generic = generic.decode_batch(dets)[:, 0] != obs

    # 6. soft information, on top of the generic prices and on top of the blend
    soft_g = soft_b = float("nan")
    if SOFT:
        rmap = core.reading_detectors(circuit, core.chip_map(hdf, run, circuit))
        N, shots = int(d[run + "/rounds"]), np.arange(len(obs))
        keys, W_even, _ = core.soft_prices(hdf, run, w_generic, rmap, N, shots[0::2])   # clouds fitted on even shots
        _, W_odd, _ = core.soft_prices(hdf, run, w_generic, rmap, N, shots[1::2])       # ... and on odd shots
        W = {"even": W_even, "odd": W_odd}
        soft_g = core.two_fold_soft(edges, w_generic, keys, W, dets, obs, blend=False).mean()
        soft_b = core.two_fold_soft(edges, w_generic, keys, W, dets, obs, blend=True).mean()

    rows.append((run, t.max() + 1, (fpga != obs).mean(), wrong_generic.mean(), wrong_c.mean(), wrong_a.mean(),
                 wrong_b.mean(), soft_g, soft_b))
    print(f"done {run}", flush=True)

    if run == "circuit_24":                   # 7a. what the p_ij look like
        print(f"\ncircuit_24: {n} detectors, {len(edges)} circuit edges, {len(every)} pairs + boundaries in all\n")
        print(f"{'edge type':<28}{'count':>6}{'median p, even':>16}{'median p, odd':>15}{'at floor':>10}")
        for k in sorted({kind(u, v, anc, t) for u, v, _ in edges}):
            e = [(u, v) for u, v, _ in edges if kind(u, v, anc, t) == k]
            pe, po = np.array([p_even[x] for x in e]), np.array([p_odd[x] for x in e])
            floor = f"{(pe <= FLOOR).sum()} / {(po <= FLOOR).sum()}"
            print(f"{k:<28}{len(e):>6}{np.median(pe):>16.4f}{np.median(po):>15.4f}{floor:>10}")
        e = [(u, v) for u, v, _ in every if (u, v) not in have]
        pe, po = np.array([pa_even[x] for x in e]), np.array([pa_odd[x] for x in e])
        floor = f"{(pe <= FLOOR).sum()} / {(po <= FLOOR).sum()}"
        print(f"{'added pairs (every pair)':<28}{len(e):>6}{np.median(pe):>16.4f}{np.median(po):>15.4f}{floor:>10}")
        b = [(u, v) for u, v, f in edges if v is None and f]
        print(f"\nboundary edges that flip the observable: {len(b)}; at the floor (even half): "
              f"{sum(p_even[x] <= FLOOR for x in b)} with circuit edges, {sum(pa_even[x] <= FLOOR for x in b)} with every pair\n")

# 7b. logical error rates, every shot graded once
print(f"\n{'run':<12}{'rounds':>7}{'FPGA':>9}{'generic':>9}{'p_ij circuit':>14}{'p_ij every pair':>17}"
      f"{'blend':>9}{'generic+soft':>14}{'blend+soft':>12}{'Riverlane MWPM':>16}{'Riverlane soft':>16}")
for run, n, f, g, pc, pa, pb, sg, sb in rows:
    print(f"{run:<12}{n:>7}{f:>9.4f}{g:>9.4f}{pc:>14.4f}{pa:>17.4f}{pb:>9.4f}{sg:>14.4f}{sb:>12.4f}"
          f"{lep.get(n, float('nan')):>16.4f}{RIVERLANE_SOFT.get(n, float('nan')):>16.4f}")
print(f"\nevery pair beats circuit edges in {sum(r[5] < r[4] for r in rows)} of {len(rows)} runs")
print(f"every pair beats the generic weights in {sum(r[5] < r[3] for r in rows)} of {len(rows)} runs")
print(f"blend (LAM = {core.LAM}) beats the generic weights in {sum(r[6] < r[3] for r in rows)} of {len(rows)} runs")
if SOFT:
    print(f"soft information improves the blend in {sum(r[8] < r[6] for r in rows)} of {len(rows)} runs")
    print(f"blend + soft beats Riverlane's best (soft + p_ij) in "
          f"{sum(r[8] < RIVERLANE_SOFT[r[1]] for r in rows)} of {len(rows)} runs")
