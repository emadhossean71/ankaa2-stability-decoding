"""With resets against without resets: every number and figure of Part II of the guide.

Run after preprocess.py and preprocess_resets.py. It also reads the PPO logs and knob files of the
cloud runs in cloud_results/ (in this folder or the one above). It writes compare_resets.json, with
the numbers behind Part II, and the eight figures (compare_figs_draw.py) into compare_figs/ in this
folder. Takes about a minute. Riverlane's noise model and the decoding graph come from ankaa_core.py.

The decoder error rates per run are not recomputed here (that takes pij.py about 40 minutes and
resets_test.py about 10); they are typed in below exactly as those two scripts print them.
"""
import json
import os
import re
from datetime import datetime

import h5py
import numpy as np
import stim

import ankaa_core as core

# ---------------------------------------------------------------- inputs
ANCILLAS, DATA_QUBIT = core.ANCILLAS, core.DATA_QUBIT
OPPOSITE = {frozenset((36, 52)), frozenset((38, 50))}
FLOOR, CEIL = core.FLOOR, core.CEIL
SHOTS = 100_000

# logical error rates, every shot graded once (two-fold for anything learned), keyed by decoding rounds T.
# Without resets: pij.py (hard decoders) and pij.py with SOFT = True (soft decoders); fpga, rl_mwpm and
# rl_best are Riverlane's (recorded verdicts, lep_mwpm.txt, their best published decoder).
NO_RESETS = {
     5: dict(fpga=0.2814, rl_mwpm=0.2876, generic=0.2863, pij=0.2962, blend=0.2747, generic_soft=0.24917, blend_soft=0.23878, rl_best=0.26818),
     7: dict(fpga=0.2663, rl_mwpm=0.2675, generic=0.2693, pij=0.2837, blend=0.2561, generic_soft=0.23043, blend_soft=0.22017, rl_best=0.24422),
     9: dict(fpga=0.2484, rl_mwpm=0.2485, generic=0.2497, pij=0.2764, blend=0.2391, generic_soft=0.21443, blend_soft=0.20404, rl_best=0.22745),
    11: dict(fpga=0.2438, rl_mwpm=0.2421, generic=0.2438, pij=0.2723, blend=0.2331, generic_soft=0.20941, blend_soft=0.19998, rl_best=0.22190),
    13: dict(fpga=0.2424, rl_mwpm=0.2406, generic=0.2412, pij=0.2760, blend=0.2321, generic_soft=0.21407, blend_soft=0.20245, rl_best=0.22143),
    15: dict(fpga=0.2325, rl_mwpm=0.2295, generic=0.2306, pij=0.2681, blend=0.2207, generic_soft=0.20183, blend_soft=0.19125, rl_best=0.20752),
    17: dict(fpga=0.2332, rl_mwpm=0.2282, generic=0.2294, pij=0.3382, blend=0.2208, generic_soft=0.20160, blend_soft=0.19133, rl_best=0.20747),
    19: dict(fpga=0.2191, rl_mwpm=0.2175, generic=0.2181, pij=0.3421, blend=0.2094, generic_soft=0.19044, blend_soft=0.17990, rl_best=0.19434),
    21: dict(fpga=0.2118, rl_mwpm=0.2070, generic=0.2076, pij=0.2771, blend=0.1965, generic_soft=0.18201, blend_soft=0.17007, rl_best=0.18241),
    23: dict(fpga=0.2225, rl_mwpm=0.2182, generic=0.2195, pij=0.2864, blend=0.2088, generic_soft=0.19448, blend_soft=0.18320, rl_best=0.19390),
    25: dict(fpga=0.2055, rl_mwpm=0.2001, generic=0.2012, pij=0.3228, blend=0.1917, generic_soft=0.17641, blend_soft=0.16521, rl_best=0.17623),
}
# With resets: resets_test.py; rl_best from lep_mwpm_soft_info_pairwise_correlation_graph_with_resets.txt
# (Riverlane published no plain-MWPM number and recorded no FPGA verdicts for these runs).
WITH_RESETS = {
     7: dict(generic=0.25309, pij=0.24712, blend=0.24022, generic_soft=0.23200, blend_soft=0.22462, rl_best=0.23203),
    11: dict(generic=0.21548, pij=0.21512, blend=0.20553, generic_soft=0.19758, blend_soft=0.19448, rl_best=0.20092),
    15: dict(generic=0.20161, pij=0.19951, blend=0.19200, generic_soft=0.18209, blend_soft=0.18011, rl_best=0.18511),
    19: dict(generic=0.18437, pij=0.18295, blend=0.17552, generic_soft=0.16548, blend_soft=0.16342, rl_best=0.16858),
    23: dict(generic=0.16968, pij=0.16938, blend=0.16187, generic_soft=0.15084, blend_soft=0.15163, rl_best=0.15901),
}
DATASETS = {
    "no resets": dict(npz="detectors.npz", h5="stability_8_without_resets_raw_data.h5", ppo={23: "ppo_circuit_22", 25: "ppo_circuit_24"}),
    "with resets": dict(npz="detectors_with_resets.npz", h5="stability_8_with_resets_raw_data.h5", ppo={23: "ppo_resets_circuit_22"}),
}
RESULTS = "cloud_results"


# ---------------------------------------------------------------- small tools
with_noise, runs_of = core.with_noise, core.runs_of


def pair_kind(a, b):
    """'same', 'neighbours' or 'opposite' for two ancillas."""
    if a == b:
        return "same"
    return "neighbours" if frozenset((a, b)) in DATA_QUBIT else "opposite"


def excess_matrix(dets):
    """Excess chance Delta (percentage points) of every detector pair, averaged over both directions."""
    X = dets.astype(np.float64)
    x = X.mean(0)
    cov = X.T @ X / len(X) - np.outer(x, x)
    D = cov / (x * (1 - x))[:, None]
    return 100 * (D + D.T) / 2


def excess_by_separation(D, anc, t, max_sep=20):
    """Average excess per (pair kind, separation in rounds)."""
    out = {k: {} for k in ("same", "neighbours", "opposite")}
    n = len(anc)
    for i in range(n):
        for j in range(i + 1, n):
            k, s = pair_kind(anc[i], anc[j]), abs(int(t[i]) - int(t[j]))
            if (k == "same" and s == 0) or s > max_sep:
                continue
            out[k].setdefault(s, []).append(D[i, j])
    return {k: {s: float(np.mean(v)) for s, v in sorted(d.items())} for k, d in out.items()}


def reach(circuit, anc, t):
    """Largest round separation that ONE error of Riverlane's model can create, per pair kind."""
    dem = with_noise(circuit).detector_error_model(decompose_errors=False)
    far = {"same": 0, "neighbours": 0, "opposite": None}
    for ins in dem.flattened():
        if ins.type != "error":
            continue
        ds = [x.val for x in ins.targets_copy() if x.is_relative_detector_id()]
        for i in ds:
            for j in ds:
                if i < j:
                    k, s = pair_kind(anc[i], anc[j]), abs(int(t[i]) - int(t[j]))
                    far[k] = s if far[k] is None else max(far[k], s)
    return far


def simulate_like(circuit, rate, shots=SHOTS, seed=7):
    """Riverlane's model with p raised until the simulated detectors fire as often as the device's."""
    lo, hi = 0.01, 0.2
    for _ in range(14):
        p = (lo + hi) / 2
        r = with_noise(circuit, p).compile_detector_sampler(seed=seed).sample(20_000).mean()
        lo, hi = (p, hi) if r < rate else (lo, p)
    p = (lo + hi) / 2
    return p, with_noise(circuit, p).compile_detector_sampler(seed=seed + 1).sample(shots)


def door_leftovers(dets, edges):
    """p_ij on every pair edge, then each door's leftover b = (x - q) / (1 - 2q), NOT clipped.
    Returns (detector, flips the observable, b, firing rate x, claimed by its pair edges q, number of pair edges)."""
    x = dets.mean(0)
    p = {}
    for u, v, _ in edges:
        if v is not None:
            cov = (dets[:, u] & dets[:, v]).mean() - x[u] * x[v]
            y = cov / ((1 - 2 * x[u]) * (1 - 2 * x[v]))
            p[u, v] = np.clip(0.5 - 0.5 / np.sqrt(max(1 + 4 * y, 1e-12)), FLOOR, CEIL)
    out = []
    for u, v, f in edges:
        if v is None:
            q, k = 0.0, 0
            for (a, b), pp in p.items():
                if u in (a, b):
                    q, k = q * (1 - pp) + pp * (1 - q), k + 1
            out.append((int(u), bool(f), float((x[u] - q) / (1 - 2 * q)), float(x[u]), float(q), k))
    return out


def graph(circuit):
    """The decoding graph's edges (detector, detector or None, flips the observable?), from ankaa_core."""
    return core.graph(circuit)[1]


def fit_clouds(z, b):
    """The two-Gaussian EM fit of pij.py on one ancilla's IQ points (all shots, for display);
    returns the axis, the fit, and each reading's position along and across the axis."""
    m0, m1 = z[~b].mean(), z[b].mean()
    u, c = (m1 - m0) / abs(m1 - m0), (m0 + m1) / 2
    w = (z - c) * np.conj(u)
    x, y = w.real.astype(np.float64), w.imag.astype(np.float64)

    def log_ratio(pi1, mu0, mu1, s0, s1):
        return (np.log(pi1) - np.log(s1) - (x - mu1) ** 2 / (2 * s1 ** 2)) - (np.log(1 - pi1) - np.log(s0) - (x - mu0) ** 2 / (2 * s0 ** 2))

    pi1, mu0, mu1, s0, s1 = b.mean(), x[~b].mean(), x[b].mean(), x[~b].std(), x[b].std()
    for _ in range(40):
        r1 = 1 / (1 + np.exp(-np.clip(log_ratio(pi1, mu0, mu1, s0, s1), -50, 50)))
        w1, w0 = r1.sum(), (1 - r1).sum()
        pi1 = w1 / r1.size
        mu0, mu1 = ((1 - r1) * x).sum() / w0, (r1 * x).sum() / w1
        s0, s1 = np.sqrt(((1 - r1) * (x - mu0) ** 2).sum() / w0), np.sqrt((r1 * (x - mu1) ** 2).sum() / w1)
    lam = log_ratio(pi1, mu0, mu1, s0, s1)
    r1 = 1 / (1 + np.exp(-np.clip(lam, -50, 50)))
    nu0, nu1 = ((1 - r1) * y).sum() / (1 - r1).sum(), (r1 * y).sum() / r1.sum()          # centres across the axis
    t0 = np.sqrt(((1 - r1) * (y - nu0) ** 2).sum() / (1 - r1).sum())
    t1 = np.sqrt((r1 * (y - nu1) ** 2).sum() / r1.sum())
    return dict(pi1=pi1, mu0=mu0, mu1=mu1, s0=s0, s1=s1, nu0=nu0, nu1=nu1, t0=t0, t1=t1), x, y, np.where(b, lam, -lam)


def ppo_log(path):
    """Per-epoch lines of a ppo.py / ppo_resets.py log: [epoch, error at mu, untuned, mean width] per direction."""
    dirs, cur = [], None
    for line in open(path):
        if line.startswith("=== "):
            cur = []
            dirs.append(cur)
        m = re.match(r"epoch\s+(\d+)\s+error rate at mu ([\d.]+)\s+untuned ([\d.]+)\s+mean width ([\d.]+)", line)
        if m:
            cur.append([float(v) for v in m.groups()])
    return [np.array(d) for d in dirs]


def disagreements(js, n=50_000):
    """n10 / n01 back out of a PPO result file's difference and 95 % range (eq. for SE in the guide)."""
    D = js["diff_pts"] / 100 * n
    se = (js["ci"][1] - js["ci"][0]) / 2 / 1.96 / 100
    tot = (n * se) ** 2 + D ** 2 / n
    return int(round((tot + D) / 2)), int(round((tot - D) / 2))


# ---------------------------------------------------------------- the analysis
def analyse():
    R = {"decoders": {"no resets": NO_RESETS, "with resets": WITH_RESETS}}
    R["matched"] = {}
    for T in sorted(WITH_RESETS):
        row = {}
        for k in ("generic", "pij", "blend", "generic_soft", "blend_soft", "rl_best"):
            a, b = NO_RESETS[T][k], WITH_RESETS[T][k]
            row[k] = [100 * (a - b), 196 * np.sqrt(a * (1 - a) / SHOTS + b * (1 - b) / SHOTS)]
        R["matched"][T] = row

    for name, ds in DATASETS.items():
        d = np.load(core.data(ds["npz"]))
        hdf = h5py.File(core.data(ds["h5"]), "r")
        out = R[name] = {"runs": {}}
        t00 = None
        for run in runs_of(d):
            dets, obs, anc, t = d[run + "/dets"], d[run + "/obs"], d[run + "/anc"], d[run + "/t"]
            T = int(t.max() + 1)
            first = dets[:, t < 2].mean(1)                      # each shot's firing rate in detector rounds 1-2
            n = dets.sum(1)
            when = datetime.fromisoformat(str(d[run + "/utc"]))
            r = out["runs"][run] = dict(T=T, N=int(d[run + "/rounds"]), utc=str(d[run + "/utc"]),
                                        minute=when.hour * 60 + when.minute + when.second / 60,
                                        run_s=float(d[run + "/run_s"]), firing=float(dets.mean()), observable=float(obs.mean()),
                                        first2=float(first.mean()), first2_se=float(first.std() / np.sqrt(len(first))),
                                        var_ratio=float(n.var() / (dets.mean(0) * (1 - dets.mean(0))).sum()))
            if T >= 5:                                          # p_ij doors, learned on the even half as in pij.py
                circuit = stim.Circuit(hdf[run + "/stim_circuits/stim_circuit_0"][()].decode())
                doors = door_leftovers(dets[0::2], graph(circuit))
                r["doors"] = len(doors)
                r["doors_at_floor"] = sum(b <= FLOOR for _, _, b, *_ in doors)
                r["observable_doors"] = sum(f for _, f, *_ in doors)
                r["observable_doors_at_floor"] = sum(f and b <= FLOOR for _, f, b, *_ in doors)
            print(f"{name:<12}{run:<12} T={T:>2}  firing {r['firing']:.4f}  rounds 1-2 {r['first2']:.4f}  Var ratio {r['var_ratio']:.2f}"
                  + (f"  doors at floor {r['doors_at_floor']}/{r['doors']}" if T >= 5 else ""), flush=True)

        # homogeneity of the rounds-1-2 firing rate across the runs of this session
        f2 = np.array([v["first2"] for v in out["runs"].values()])
        se = np.array([v["first2_se"] for v in out["runs"].values()])
        mins = np.array([v["minute"] for v in out["runs"].values()])
        mean = (f2 / se ** 2).sum() / (1 / se ** 2).sum()
        rank = lambda a: np.argsort(np.argsort(a))
        out["first2_chi2"] = [float((((f2 - mean) / se) ** 2).sum()), len(f2) - 1]
        out["first2_spearman_time"] = float(np.corrcoef(rank(f2), rank(mins))[0, 1])

        # ---- circuit_22 (23 decoding rounds) in detail: the like-for-like run
        run = "circuit_22"
        dets, anc, t = d[run + "/dets"], d[run + "/anc"], d[run + "/t"]
        circuit = stim.Circuit(hdf[run + "/stim_circuits/stim_circuit_0"][()].decode())
        c22 = out["c22"] = {}
        c22["rate"] = {int(a): [float(dets[:, (anc == a) & (t == k)].mean()) for k in range(int(t.max()) + 1)] for a in ANCILLAS}
        c22["reach"] = reach(circuit, anc, t)
        D = excess_matrix(dets)
        order = np.lexsort((t, anc))
        np.save(core.out(f"compare_figs/excess_{name.replace(' ', '_')}.npy"), D[np.ix_(order, order)])
        c22["excess"] = excess_by_separation(D, anc, t)
        c22["excess_per_ancilla_3_10"] = {int(a): float(np.mean([D[i, j] for i in range(len(anc)) for j in range(len(anc))
                                                                 if anc[i] == a == anc[j] and 3 <= t[j] - t[i] <= 10])) for a in ANCILLAS}
        p_sim, sim = simulate_like(circuit, dets.mean())
        c22["sim_p"], c22["sim_firing"] = p_sim, float(sim.mean())
        c22["sim_excess"] = excess_by_separation(excess_matrix(sim), anc, t)
        c22["sim_var_ratio"] = float(sim.sum(1).var() / (sim.mean(0) * (1 - sim.mean(0))).sum())
        c22["rate_all"] = [float(dets[:, t == k].mean()) for k in range(int(t.max()) + 1)]      # device, the four ancillas together
        c22["rate_all_sim"] = [float(sim[:, t == k].mean()) for k in range(int(t.max()) + 1)]   # the model at the device's firing rate
        c22["doors"] = [(int(anc[u]), int(t[u]) + 1, f, b, xx, q, k) for u, f, b, xx, q, k in door_leftovers(dets[0::2], graph(circuit))]
        print(f"{name}: one error reaches {c22['reach']}; simulated p = {p_sim:.4f} fires {sim.mean():.4f} (device {dets.mean():.4f})", flush=True)

        # IQ readout of every ancilla
        iq = c22["iq"] = {}
        for a in ANCILLAS:
            z = hdf[f"{run}/result/soft_measurements/{a}"][()]
            b = hdf[f"{run}/result/hard_measurements/{a}"][()].astype(bool)
            fit, x, y, e = fit_clouds(z.ravel(), b.ravel())
            x, y, e = x.reshape(z.shape), y.reshape(z.shape), e.reshape(z.shape)
            odd = y < fit["nu1"] - 3 * fit["t1"]                 # more than 3 widths below the 1 cloud, across the axis
            both = (odd[:, 1:] & odd[:, :-1]).mean()
            if a == 50:                                         # ancilla 50's IQ plane, for the figure
                H, xe, ye = np.histogram2d(x.ravel(), y.ravel(), bins=220)
                np.savez(core.out(f"compare_figs/iq50_{name.replace(' ', '_')}.npz"), H=H, xe=xe, ye=ye, cut=fit["nu1"] - 3 * fit["t1"])
            iq[a] = dict(separation=float((fit["mu1"] - fit["mu0"]) / np.sqrt((fit["s0"] ** 2 + fit["s1"] ** 2) / 2)),
                         misread=float((1 / (1 + np.exp(np.maximum(e, 0)))).mean()), wrong_side=float((e < 0).mean()),
                         odd=float(odd.mean()), odd_per_reading=[float(v) for v in odd.mean(0)],
                         odd_first5=float(odd[:, :5].mean()), odd_last5=float(odd[:, -5:].mean()),
                         odd_memory=float(both / (odd[:, :-1].mean() * odd[:, 1:].mean())),
                         **{k: float(v) for k, v in fit.items()})
            print(f"{name}: ancilla {a}  separation {iq[a]['separation']:.2f} widths  misread {iq[a]['misread']:.4f}  "
                  f"odd {iq[a]['odd']:.5f}  memory x{iq[a]['odd_memory']:.1f}", flush=True)

        # PPO
        ppo = out["ppo"] = {}
        for T, stem in ds["ppo"].items():
            js = {h: json.load(open(core.data(f"{RESULTS}/{stem}_learn_{h}.json"))) for h in ("even", "odd")}
            nn = {h: disagreements(js[h]) for h in js}
            ppo[T] = dict(untuned={h: js[h]["untuned"] for h in js}, tuned={h: js[h]["tuned"] for h in js},
                          diff={h: [js[h]["diff_pts"], *js[h]["ci"], js[h]["p"]] for h in js}, n10_n01=nn,
                          knobs={h: js[h]["knobs"] for h in js}, curves=[c.tolist() for c in ppo_log(core.data(f"{RESULTS}/{stem}.log"))])
            n10, n01 = sum(v[0] for v in nn.values()), sum(v[1] for v in nn.values())
            dd, se = (n10 - n01) / SHOTS, np.sqrt(n10 + n01 - (n10 - n01) ** 2 / SHOTS) / SHOTS
            ppo[T]["both"] = dict(untuned=float(np.mean(list(ppo[T]["untuned"].values()))), tuned=float(np.mean(list(ppo[T]["tuned"].values()))),
                                  diff=[100 * dd, 100 * (dd - 1.96 * se), 100 * (dd + 1.96 * se)], n10_n01=[n10, n01])
            lm = {h: np.log([js[h]["knobs"][k][0] for k in js["even"]["knobs"]]) for h in js}
            ppo[T]["r_directions"] = float(np.corrcoef(lm["even"], lm["odd"])[0, 1])
            ppo[T]["same_direction"] = int(((lm["even"] > 0) == (lm["odd"] > 0)).sum())
            ppo[T]["n_knobs"] = len(lm["even"])

    # knobs learned on different data: averaged over the two directions, log scale
    def knob_mean(name, T):
        k = R[name]["ppo"][T]["knobs"]
        return {n: float(np.mean([np.log(k[h][n][0]) for h in k])) for n in k["even"]}
    A, B, C = knob_mean("no resets", 23), knob_mean("with resets", 23), knob_mean("no resets", 25)
    shared = [n for n in B if n in A]
    R["knob_r_resets_vs_noresets_T23"] = [float(np.corrcoef([A[n] for n in shared], [B[n] for n in shared])[0, 1]), len(shared)]
    R["knob_r_T23_vs_T25_noresets"] = [float(np.corrcoef([A[n] for n in A], [C[n] for n in A])[0, 1]), len(A)]
    json.dump(R, open(core.out("compare_resets.json"), "w"), indent=1, default=float)
    return R


if __name__ == "__main__":
    os.makedirs(core.out("compare_figs"), exist_ok=True)
    analyse()
    import compare_figs_draw                       # the figures (compare_figs_draw.py, next to this file)
    compare_figs_draw.draw(json.load(open(core.out("compare_resets.json"))))
