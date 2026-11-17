"""ankaa_core: the pieces every Ankaa-2 script in this folder shares.

Every script here starts with  `import ankaa_core as core`  and calls these functions, so each
piece of the method is written once:

  files             data(name)  finds an input in this folder or in the folder above
                    out(name)   puts an output in this folder (nothing above is ever overwritten)
  reading the data  read_runs, runs_of, load_circuit, chip_map
  the model         with_noise (Riverlane's noise model), graph (edges and generic weights)
  learned weights   pij, blend_prices, decoder, matching, two_fold, two_fold_blend
  soft information  soft_evidence, reading_detectors, soft_prices, two_fold_soft
  statistics        paired
  PPO               ppo_problem, init_worker, wrong_shots, n_wrong, train, grade

The functions are the ones of pij.py, drift_test.py, ppo.py, resets_test.py and ppo_resets.py,
moved here unchanged in what they compute; the README says how this was checked.
"""
import math
import time
from pathlib import Path

import h5py
import numpy as np
import pymatching
import stim

# ---------------------------------------------------------------- settings shared by every script
FLOOR, CEIL = 1e-4, 0.45          # keep every p in a range where log((1-p)/p) is sensible
LAM = 0.75                        # blend: how far pair prices move from generic (0) towards p_ij (1)
P_MEAS = 0.03                     # Riverlane's measurement-flip probability
ANCILLAS = (36, 38, 50, 52)
DATA_QUBIT = {frozenset((36, 38)): 37, frozenset((38, 52)): 45, frozenset((52, 50)): 51, frozenset((50, 36)): 43}

# ---------------------------------------------------------------- files
HERE = Path(__file__).resolve().parent


def data(name):
    """An input file: looked for in this folder first, then in the folder above (where the raw data lives)."""
    for folder in (HERE, HERE.parent):
        if (folder / name).exists():
            return folder / name
    raise FileNotFoundError(f"{name} is neither in {HERE} nor in {HERE.parent}")


def out(name):
    """Every output goes into this folder, so nothing in the folder above is overwritten."""
    return HERE / name


# ---------------------------------------------------------------- reading the data
def runs_of(npz):
    """The runs in a detectors file, shortest first."""
    return sorted({k.split("/")[0] for k in npz.files}, key=lambda k: int(k.split("_")[1]))


def load_circuit(hdf, run):
    """The run's (noiseless) Stim circuit, as deposited by Riverlane."""
    return stim.Circuit(hdf[run + "/stim_circuits/stim_circuit_0"][()].decode())


def chip_map(hdf, run, circuit):
    """Stim's qubit number -> the chip's qubit number (36, 37, ...), matched by coordinates."""
    at = {(x, y): q for x, y, q in hdf[run + "/qubit_mapping"][()].tolist()}
    return {i: at[int(x), int(y)] for i, (x, y) in circuit.get_final_qubit_coordinates().items()}


def read_runs(h5_name):
    """Raw HDF file -> detection events of every run, as the dictionary that preprocess.py saves.
    Works for both experiments; the FPGA verdicts are kept where the file has them."""
    out_ = {}
    with h5py.File(data(h5_name), "r") as f:
        runs = sorted([k for k in f if k.startswith("circuit_")], key=lambda k: int(k.split("_")[1]))
        for run in runs:
            g = f[run]
            circuit = load_circuit(f, run)
            at = {(x, y): q for x, y, q in g["qubit_mapping"][()].tolist()}
            chip = chip_map(f, run, circuit)

            # The file keeps each qubit's readings in its own table. Put them back in the
            # order the circuit measured them: each time a qubit is measured, take its next column.
            order = [chip[q.value] for ins in circuit.flattened() if ins.name == "M" for q in ins.targets_copy()]
            hard = {q: g[f"result/hard_measurements/{q}"][()] for q in set(order)}
            m = np.zeros((len(hard[order[0]]), len(order)), dtype=bool)
            col = {q: 0 for q in hard}
            for k, q in enumerate(order):
                m[:, k] = hard[q][:, col[q]]
                col[q] += 1
            assert all(col[q] == hard[q].shape[1] for q in hard)        # every reading used exactly once

            # Stim XORs the readings into detectors and the observable, as the circuit says.
            dets, obs = circuit.compile_m2d_converter().convert(measurements=m, separate_observables=True)

            # A detector's coordinates are (x, y, round); its ancilla sits at (2x, 2y) on the chip map.
            c = circuit.get_detector_coordinates()
            a = g["result"].attrs
            out_[run + "/dets"] = dets
            out_[run + "/obs"] = obs[:, 0]
            if "decoder_shot_results" in g["result"]:                    # the no-reset runs only
                out_[run + "/fpga"] = g["result/decoder_shot_results"][:, 0] == 1   # 1 = flip; 0 or 2 = no flip
            out_[run + "/anc"] = np.array([at[2 * int(c[d][0]), 2 * int(c[d][1])] for d in range(len(c))])
            out_[run + "/t"] = np.array([int(c[d][2]) for d in range(len(c))])
            out_[run + "/rounds"] = int(a["rounds"])
            out_[run + "/utc"] = str(a["utc_time_taken"])
            out_[run + "/run_s"] = float(a["running_time"])
    return runs, out_


# ---------------------------------------------------------------- the model
def with_noise(circuit, p=0.03):
    """The deposited circuits are noiseless. Add Riverlane's own noise model, the one
    they used to weight their MWPM decoder (Caune et al., arXiv:2410.05202, Appendix B):
      - both qubits depolarised with p after every two-qubit gate (CZ)
      - p/10 depolarising after every single-qubit gate, after every measurement,
        and on every qubit left idle during a time step
      - every measurement result flipped with p
    p = 0.03 was chosen by Riverlane from Ankaa-2's median CZ fidelity.
    This gives the edges (which detectors each error can flip) and the generic weights."""
    out_, busy = stim.Circuit(), set()
    for ins in circuit.flattened():
        q = [x.value for x in ins.targets_copy() if x.is_qubit_target]
        if ins.name == "TICK":                               # end of a time step
            idle = [k for k in range(circuit.num_qubits) if k not in busy]
            if busy and idle:
                out_.append("DEPOLARIZE1", idle, p / 10)     # qubits that did nothing this step
            busy.clear()
        if ins.name == "M":
            out_.append("M", q, p)                           # measurement result flipped
            out_.append("DEPOLARIZE1", q, p / 10)
        else:
            out_.append(ins)
        if ins.name in ("SQRT_X", "SQRT_X_DAG", "S"):
            out_.append("DEPOLARIZE1", q, p / 10)
        if ins.name == "CZ":
            out_.append("DEPOLARIZE2", q, p)
        if ins.name in ("M", "CZ", "SQRT_X", "SQRT_X_DAG", "S"):
            busy.update(q)
    return out_


def graph(circuit):
    """The decoding graph of Riverlane's model: the generic matching decoder, the edges
    (detector, detector or None for a door, flips the observable?) and each edge's generic weight."""
    generic = pymatching.Matching.from_detector_error_model(with_noise(circuit).detector_error_model(decompose_errors=True))
    edges, w_generic = [], {}
    for u, v, a in generic.edges():
        k = (u, None) if v is None else (min(u, v), max(u, v))
        edges.append((*k, 0 in a["fault_ids"]))
        w_generic[k] = a["weight"]                       # Riverlane's price for this edge
    return generic, edges, w_generic


# ---------------------------------------------------------------- learned weights
def pij(dets, edges):
    """Probability of every edge, from how often its detectors fire together."""
    x = dets.mean(0)                                            # each detector's firing rate
    p = {}
    for u, v, _ in edges:                                       # edges between two detectors
        if v is not None:
            cov = (dets[:, u] & dets[:, v]).mean() - x[u] * x[v]
            y = cov / ((1 - 2 * x[u]) * (1 - 2 * x[v]))
            p[u, v] = np.clip(0.5 - 0.5 / np.sqrt(max(1 + 4 * y, 1e-12)), FLOOR, CEIL)
    for u, v, _ in edges:                                       # boundary edges: one detector
        if v is None:
            q = 0.0                                             # chance the other edges fire u
            for (a, b), pp in list(p.items()):
                if b is not None and u in (a, b):
                    q = q * (1 - pp) + pp * (1 - q)
            p[u, None] = np.clip((x[u] - q) / (1 - 2 * q), FLOOR, CEIL)   # the leftover
    return p


def blend_prices(edges, w_generic, dets):
    """Generic price on every boundary door; each pair price moved a fraction LAM towards p_ij."""
    p = pij(dets, edges)
    return {k: w_generic[k] if k[1] is None else (1 - LAM) * w_generic[k] + LAM * np.log((1 - p[k]) / p[k])
            for k in w_generic}


def decoder(edges, p):
    """A matching decoder from edge probabilities."""
    m = pymatching.Matching()
    for u, v, flips in edges:
        w = np.log((1 - p[u, v]) / p[u, v])
        f = {0} if flips else set()                   # does this error flip the observable?
        if v is None:
            m.add_boundary_edge(u, weight=w, fault_ids=f)
        else:
            m.add_edge(u, v, weight=w, fault_ids=f)
    return m


def matching(edges, w):
    """A matching decoder from edge weights."""
    m = pymatching.Matching()
    for u, v, flips in edges:
        f = {0} if flips else set()
        if v is None:
            m.add_boundary_edge(u, weight=w[u, v], fault_ids=f)
        else:
            m.add_edge(u, v, weight=w[u, v], fault_ids=f)
    return m


def two_fold(edges, dets, obs):
    """Learn p_ij from each half, decode the other half. Returns 1 per shot the decoder got wrong."""
    even, odd = np.arange(0, len(obs), 2), np.arange(1, len(obs), 2)
    p_even, p_odd = pij(dets[even], edges), pij(dets[odd], edges)
    wrong = np.zeros(len(obs), dtype=bool)
    wrong[odd] = decoder(edges, p_even).decode_batch(dets[odd])[:, 0] != obs[odd]
    wrong[even] = decoder(edges, p_odd).decode_batch(dets[even])[:, 0] != obs[even]
    return wrong, p_even, p_odd


def two_fold_blend(edges, w_generic, dets, obs):
    """Same halves, decoded with the blend prices learned on the other half."""
    even, odd = np.arange(0, len(obs), 2), np.arange(1, len(obs), 2)
    wrong = np.zeros(len(obs), dtype=bool)
    for learn, test in ((even, odd), (odd, even)):
        w = blend_prices(edges, w_generic, dets[learn])
        p_blend = {k: 1 / (1 + np.exp(w[k])) for k in w}           # price back to probability
        wrong[test] = decoder(edges, p_blend).decode_batch(dets[test])[:, 0] != obs[test]
    return wrong


# ---------------------------------------------------------------- soft information
def soft_evidence(hdf, run, a, fit):
    """For every reading of ancilla a (shots x readings): how strongly its IQ point supports
    the recorded 0/1. Two Gaussians are fitted along the line joining the two clouds (EM),
    using only the shots in `fit` (the learning half); the fit is then applied to every shot.
    Lambda = ln[P(1|z) / P(0|z)] includes the fitted population of 1s (pi1), so it is a
    posterior log-odds, not a pure readout likelihood ratio. The evidence for the recorded
    bit is +Lambda for a 1 and -Lambda for a 0; negative evidence means the point favours
    the other bit."""
    z_all = hdf[f"{run}/result/soft_measurements/{a}"][()]
    b_all = hdf[f"{run}/result/hard_measurements/{a}"][()].astype(bool)
    z, b = z_all[fit], b_all[fit]
    m0, m1 = z[~b].mean(), z[b].mean()
    axis = lambda zz: ((zz - (m0 + m1) / 2) * np.conj((m1 - m0) / abs(m1 - m0))).real.astype(np.float64)
    x = axis(z)

    def log_ratio(pi1, mu0, mu1, s0, s1):                   # ln[pi1 P(x|1)] - ln[pi0 P(x|0)]
        return (np.log(pi1) - np.log(s1) - (x - mu1) ** 2 / (2 * s1 ** 2)) \
             - (np.log(1 - pi1) - np.log(s0) - (x - mu0) ** 2 / (2 * s0 ** 2))

    pi1, mu0, mu1, s0, s1 = b.mean(), x[~b].mean(), x[b].mean(), x[~b].std(), x[b].std()
    for _ in range(40):                                     # expectation-maximisation
        r1 = 1 / (1 + np.exp(-np.clip(log_ratio(pi1, mu0, mu1, s0, s1), -50, 50)))   # chance each point is a 1
        w1, w0 = r1.sum(), (1 - r1).sum()
        pi1 = w1 / r1.size
        mu0, mu1 = ((1 - r1) * x).sum() / w0, (r1 * x).sum() / w1
        s0, s1 = np.sqrt(((1 - r1) * (x - mu0) ** 2).sum() / w0), np.sqrt((r1 * (x - mu1) ** 2).sum() / w1)
    x = axis(z_all)                                         # apply the learning-half fit to every shot
    lam = log_ratio(pi1, mu0, mu1, s0, s1)
    return np.where(b_all, lam, -lam)


def reading_detectors(circuit, chip):
    """(qubit, reading number) -> the detectors that reading enters, read off the circuit's DETECTOR lines.
    Without resets reading t enters detectors t-1 and t+1 of its ancilla; with resets, t-1 and t."""
    recs, enters, ndet = [], {}, 0
    for ins in circuit.flattened():
        if ins.name == "M":
            recs += [chip[t.value] for t in ins.targets_copy()]
        elif ins.name == "DETECTOR":
            for t in ins.targets_copy():
                enters.setdefault(len(recs) + t.value, []).append(ndet)
            ndet += 1
    count, out_ = {}, {}
    for i, q in enumerate(recs):
        count[q] = count.get(q, 0) + 1
        out_[q, count[q]] = sorted(enters.get(i, []))
    return out_


def soft_prices(hdf, run, w_generic, rmap, N, fit):
    """Every ancilla reading's misread edge, and its price in every shot (shots x edges).
    Returns the edge keys, the price table and the ancilla that owns each column.
    An edge the model also fills with other errors (p_generic != 0.03) is priced as
    "misread XOR the rest", rest = (p_generic - 0.03) / (1 - 2 x 0.03). Without resets this
    touches only the last doors; with resets, every misread edge."""
    keys, cols, owner = [], [], []
    for a in ANCILLAS:
        evidence = soft_evidence(hdf, run, a, fit)
        for r in range(1, N + 1):
            ids = rmap[a, r]
            if not ids:
                continue
            key = (ids[0], ids[1]) if len(ids) == 2 else (ids[0], None)
            w = np.maximum(evidence[:, r - 1], 0)                   # IQ favouring the other bit -> a coin toss
            p_gen = 1 / (1 + np.exp(w_generic[key]))
            if abs(p_gen - P_MEAS) > 1e-6:                           # the edge also holds other errors
                q = 1 / (1 + np.exp(w))
                rest = max((p_gen - P_MEAS) / (1 - 2 * P_MEAS), 1e-6)
                p = q * (1 - rest) + rest * (1 - q)
                w = np.log((1 - p) / p)
            keys.append(key)
            cols.append(w)
            owner.append(a)
    return keys, np.column_stack(cols), owner


def two_fold_soft(edges, w_generic, keys, W, dets, obs, blend):
    """Generic (or blend) prices everywhere, then each shot's own misread prices from its IQ points.
    W[half] holds the soft prices from clouds fitted on that half; a half is decoded with the other half's."""
    flips = {(u, v): ({0} if f else set()) for u, v, f in edges}
    even, odd = np.arange(0, len(obs), 2), np.arange(1, len(obs), 2)
    wrong = np.zeros(len(obs), dtype=bool)
    for name, learn, test in (("even", even, odd), ("odd", odd, even)):
        w = blend_prices(edges, w_generic, dets[learn]) if blend else w_generic
        m = decoder(edges, {k: 1 / (1 + np.exp(w[k])) for k in w})
        for s in test:
            for (u, v), ws in zip(keys, W[name][s]):
                if v is None:
                    m.add_boundary_edge(u, weight=ws, fault_ids=flips[u, v], merge_strategy="replace")
                else:
                    m.add_edge(u, v, weight=ws, fault_ids=flips[u, v], merge_strategy="replace")
            wrong[s] = m.decode(dets[s])[0] != obs[s]
    return wrong


# ---------------------------------------------------------------- statistics
def paired(a, b):
    """Two decoders on the same shots (a, b: True where wrong). Returns the difference a - b in
    % points, its 95 % range, and the exact two-sided p (only shots where they disagree count)."""
    n10, n01, n = int((a & ~b).sum()), int((b & ~a).sum()), len(a)
    diff, se = (n10 - n01) / n, math.sqrt(n10 + n01 - (n10 - n01) ** 2 / n) / n
    p = min(1.0, 2 * sum(math.exp(math.lgamma(n10 + n01 + 1) - math.lgamma(i + 1) - math.lgamma(n10 + n01 - i + 1) - (n10 + n01) * math.log(2))
                         for i in range(min(n10, n01) + 1)))
    return 100 * diff, 100 * (diff - 1.96 * se), 100 * (diff + 1.96 * se), p


# ---------------------------------------------------------------- PPO
def ppo_problem(npz_name, h5_name, run, learn_on):
    """One run, one learning half: the blend and the IQ cloud fit from the learning half, the
    per-shot soft prices, and the knobs every other edge is tied to:
      flip  : same ancilla, next round, not soft-priced     per ancilla    x region (start / middle / end)
      data  : neighbouring ancillas, same round             per data qubit x region
      diag  : neighbouring ancillas, one round apart        per data qubit x region
      door1 : a round-1 door that is not soft-priced        per ancilla
      soft  : a scale on the per-shot soft prices           per ancilla
    Without resets this gives 44 knobs, with resets 28 (no flip or door1 edges are left)."""
    d = np.load(data(npz_name))
    hdf = h5py.File(data(h5_name), "r")
    dets, obs = d[run + "/dets"], d[run + "/obs"]
    anc, t = d[run + "/anc"], d[run + "/t"]
    N = int(d[run + "/rounds"]); T = int(t.max() + 1)
    circuit = load_circuit(hdf, run)
    generic, edges, w_generic = graph(circuit)
    shots = np.arange(len(obs))
    learn, test = (shots[0::2], shots[1::2]) if learn_on == "even" else (shots[1::2], shots[0::2])

    base = blend_prices(edges, w_generic, dets[learn])                       # learned on the learning half only
    rmap = reading_detectors(circuit, chip_map(hdf, run, circuit))
    keys, W, owner = soft_prices(hdf, run, w_generic, rmap, N, learn)        # clouds fitted on the learning half
    assert len(set(keys)) == len(keys), "two readings share one edge"

    names, group = [], {}

    def knob(name):
        if name not in names:
            names.append(name)
        return names.index(name)

    def region(r):
        return "start" if r <= 1 else "end" if r >= T - 2 else "mid"

    soft = set(keys)
    for u, v, _ in edges:
        if (u, v) in soft:
            continue
        if v is None:
            group[u, v] = knob(f"door1 {anc[u]}" if t[u] == 0 else f"door {anc[u]} {region(t[u])}")
        elif anc[u] == anc[v]:
            group[u, v] = knob(f"flip {anc[u]} {region(min(t[u], t[v]))}")
        else:
            kind = "data" if t[u] == t[v] else "diag"
            group[u, v] = knob(f"{kind} {DATA_QUBIT[frozenset((int(anc[u]), int(anc[v])))]} {region(min(t[u], t[v]))}")
    soft_knob = np.array([knob(f"soft {a}") for a in owner])
    flips = {(u, v): ({0} if f else set()) for u, v, f in edges}

    # references on the same graded shots: our MWPM, our blend, and Riverlane's FPGA where recorded
    blend = matching([(u, v, f) for u, v, f in edges], base)
    return dict(names=names, dets=dets, obs=obs, learn=learn, test=test, W=W, T=T,
                mwpm_wrong=generic.decode_batch(dets[test])[:, 0] != obs[test],
                blend_wrong=blend.decode_batch(dets[test])[:, 0] != obs[test],
                fpga_wrong=(d[run + "/fpga"][test] != obs[test]) if run + "/fpga" in d.files else None,
                edges=[(u, v, flips[u, v]) for u, v, _ in edges],
                edge_w=np.array([base[u, v] for u, v, _ in edges]),
                edge_knob=np.array([group.get((u, v), -1) for u, v, _ in edges]),   # -1: replaced by a soft price
                soft=[(u, v, flips[u, v]) for u, v in keys], soft_knob=soft_knob)


PROB = None                                   # the problem, one copy in every worker process


def init_worker(prob):
    global PROB
    PROB = prob


def wrong_shots(job):
    """Decode the given shots with knobs theta; True where the decoder got the logical outcome wrong."""
    theta, shots = job
    mult = np.exp(np.append(theta, 0.0))       # knob -1 -> multiplier 1
    m = pymatching.Matching()
    for (u, v, f), w in zip(PROB["edges"], PROB["edge_w"] * mult[PROB["edge_knob"]]):
        if v is None:
            m.add_boundary_edge(u, weight=w, fault_ids=f)
        else:
            m.add_edge(u, v, weight=w, fault_ids=f)
    scale = mult[PROB["soft_knob"]]
    wrong = np.zeros(len(shots), dtype=bool)
    for i, s in enumerate(shots):              # this shot's own misread prices
        for (u, v, f), w in zip(PROB["soft"], PROB["W"][s] * scale):
            if v is None:
                m.add_boundary_edge(u, weight=w, fault_ids=f, merge_strategy="replace")
            else:
                m.add_edge(u, v, weight=w, fault_ids=f, merge_strategy="replace")
        wrong[i] = m.decode(PROB["dets"][s])[0] != PROB["obs"][s]
    return wrong


def n_wrong(job):
    return int(wrong_shots(job).sum())


def train(pool, prob, B, NSHOT, K, EPOCHS, EPS, ETA, SIG0, GCLIP, SIG_MIN):
    """PPO on the knobs (Sivak, Newman & Klimov, PRL 133, 150603 (2024)); returns the policy (mu, rho)."""
    P = len(prob["names"])
    rng = np.random.default_rng(1)
    mu, rho = np.zeros(P), np.full(P, np.log(SIG0))           # the policy: centre and log-width of each knob
    m_mu, v_mu, m_rho, v_rho, step = np.zeros(P), np.zeros(P), np.zeros(P), np.zeros(P), 0
    t0 = time.time()
    for ep in range(EPOCHS):
        shots = rng.choice(prob["learn"], NSHOT, replace=False)   # fresh shots, the same for every candidate
        sig = np.exp(rho)
        cand = mu + sig * rng.standard_normal((B, P))              # 1. wobble
        fails = pool.map(n_wrong, [(c, shots) for c in cand] + [(mu, shots), (np.zeros(P), shots)])   # 2-3. decode
        R = -np.log10(np.maximum(np.array(fails[:B]), 0.5) / NSHOT)                # 4. score
        alpha = (R - R.mean()) / (R.std() + 1e-8)                                   # 5. grade on a curve
        lold = (-(cand - mu) ** 2 / (2 * sig ** 2) - np.log(sig)).sum(1)            # 6. freeze the policy
        for _ in range(K):                                                          # 7. clipped policy steps
            sig = np.exp(rho)
            chi = np.exp((-(cand - mu) ** 2 / (2 * sig ** 2) - rho).sum(1) - lold)
            flat = ((alpha > 0) & (chi > 1 + EPS)) | ((alpha < 0) & (chi < 1 - EPS))
            w = np.where(flat, 0.0, chi * alpha)
            z = (cand - mu) / sig
            g_mu = np.clip((w[:, None] * (z / sig)).mean(0), -GCLIP, GCLIP)
            g_rho = np.clip((w[:, None] * (z ** 2 - 1)).mean(0), -GCLIP, GCLIP)
            step += 1
            m_mu, v_mu = 0.9 * m_mu + 0.1 * g_mu, 0.999 * v_mu + 0.001 * g_mu ** 2
            m_rho, v_rho = 0.9 * m_rho + 0.1 * g_rho, 0.999 * v_rho + 0.001 * g_rho ** 2
            mu = mu + ETA * (m_mu / (1 - 0.9 ** step)) / (np.sqrt(v_mu / (1 - 0.999 ** step)) + 1e-8)
            rho = rho + ETA * (m_rho / (1 - 0.9 ** step)) / (np.sqrt(v_rho / (1 - 0.999 ** step)) + 1e-8)
            rho = np.maximum(rho, np.log(SIG_MIN))
        print(f"epoch {ep + 1:3d}   error rate at mu {fails[B] / NSHOT:.4f}   untuned {fails[B + 1] / NSHOT:.4f}   "
              f"mean width {np.exp(rho).mean():.3f}   {(time.time() - t0) / 60:5.1f} min", flush=True)
    return mu, rho


def grade(pool, prob, mu, shots, cores):
    """Tuned and untuned decoders on the same shots; returns the two 'wrong' arrays."""
    parts = np.array_split(shots, 4 * cores)
    tuned = np.concatenate(pool.map(wrong_shots, [(mu, p) for p in parts]))
    plain = np.concatenate(pool.map(wrong_shots, [(np.zeros_like(mu), p) for p in parts]))
    return tuned, plain
