"""Why resets help more as the stability experiment gets longer: the time-like distance (guide, Section 13.2).

1. For every run of both experiments, Stim's shortest undetectable error that flips the observable
   ("graph-like distance", the one a matching decoder sees), twice:
     - with readout errors only (every measurement flipped with p, nothing else);
     - with Riverlane's full noise model (core.with_noise).
   Without resets a misread fires two detectors two rounds apart, so the shortest chain needs
   only every other reading (101010...): ceil(N/2). With resets it needs every reading (111111...): N.
2. How much reset advantage would that structure alone give? Both circuits are simulated with
   Riverlane's model (at p = 0.03, and at p = 0.082, where the simulated detectors fire as often as
   the device's) and decoded with the matching decoder of the same model.

Run after nothing (it needs only the two raw-data files). Takes about 2 minutes.
Writes reset_distance.json into this folder.
"""
import json

import h5py
import numpy as np
import pymatching
import stim

import ankaa_core as core

SHOTS = 200_000


def readout_only(circuit, p=0.03):
    """The circuit with every measurement result flipped with probability p, and no other noise."""
    out = stim.Circuit()
    for ins in circuit.flattened():
        if ins.name == "M":
            out.append("M", [x.value for x in ins.targets_copy()], p)
        else:
            out.append(ins)
    return out


res = {"distance": {}, "simulated": {}}
for h5, name in (("stability_8_without_resets_raw_data.h5", "no resets"), ("stability_8_with_resets_raw_data.h5", "with resets")):
    hdf = h5py.File(core.data(h5), "r")
    runs = sorted([k for k in hdf if k.startswith("circuit_")], key=lambda k: int(k.split("_")[1]))
    print(f"\n{name}")
    for run in runs:
        c = core.load_circuit(hdf, run)
        N = sum(1 for ins in c.flattened() if ins.name == "M")              # readings per ancilla
        d_ro = len(readout_only(c).detector_error_model(decompose_errors=True).shortest_graphlike_error())
        d_all = len(core.with_noise(c).detector_error_model(decompose_errors=True).shortest_graphlike_error())
        res["distance"][f"{name}/{run}"] = dict(N=N, T=N - 1, readout_only=d_ro, full_model=d_all)
        print(f"  {run:<12} N = {N:>2}   readout errors only: {d_ro:>2}   Riverlane's full model: {d_all:>2}")
    for n in (6, 10, 14, 18, 22):
        c = core.load_circuit(hdf, f"circuit_{n}")
        for p in (0.03, 0.082):
            noisy = core.with_noise(c, p)
            m = pymatching.Matching.from_detector_error_model(noisy.detector_error_model(decompose_errors=True))
            dets, obs = noisy.compile_detector_sampler(seed=n).sample(SHOTS, separate_observables=True)
            res["simulated"][f"{name}/{n + 1}/{p}"] = dict(ler=float((m.decode_batch(dets)[:, 0] != obs[:, 0]).mean()),
                                                         firing=float(dets.mean()))

print("\nRiverlane's model, simulated and decoded with its own matching decoder (logical error rate, %):")
for p in (0.03, 0.082):
    print(f"p = {p}")
    for T in (7, 11, 15, 19, 23):
        a, b = res["simulated"][f"no resets/{T}/{p}"], res["simulated"][f"with resets/{T}/{p}"]
        se = 196 * np.sqrt(a["ler"] * (1 - a["ler"]) / SHOTS + b["ler"] * (1 - b["ler"]) / SHOTS)
        print(f"  T = {T:>2}   no resets {100 * a['ler']:6.2f}   with resets {100 * b['ler']:6.2f}   "
              f"advantage {100 * (a['ler'] - b['ler']):+6.2f} +- {se:.2f}   (detectors firing {a['firing']:.3f} / {b['firing']:.3f})")
json.dump(res, open(core.out("reset_distance.json"), "w"), indent=1)
