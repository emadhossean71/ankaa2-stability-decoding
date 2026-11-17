"""lambda_scan.py: a fine scan of the blend's shrinkage lambda (no resets, hard readings).

The guide's Table 3 tried five values (0, 0.25, 0.5, 0.75, 1). This script tries 29, every 0.05
from 0 to 1 and every 0.025 between 0.6 and 1, and asks three questions:

  1. Where is the minimum, and how flat is the curve around it?
  2. Is the best grid value really better than 0.75, or is the gap within noise?
     (paired on the same shots: exact McNemar test and a 95 % range for the difference)
  3. What error rate does the CHOOSING itself give, when lambda is picked without looking
     at the shots it is graded on? (nested: lambda is chosen inside the learning half,
     by splitting it in two again, then applied to the other half)

Everything else is exactly pij.py's blend (core.pij, core.decoder): p_ij learned on one half
of the shots, decoded on the other half, every shot graded once, circuit_2 left out.
Writes lambda_scan.json and lambda_scan.png into this folder; nothing is overwritten.
Run after preprocess.py. About 15-25 minutes with CORES = 2.
"""
import json
import multiprocessing as mp

import h5py
import numpy as np
from scipy.stats import binomtest

import ankaa_core as core

CORES = 2
H5 = "stability_8_without_resets_raw_data.h5"
LAMBDAS = sorted({round(x, 3) for x in [*np.arange(0, 1.0001, 0.05), *np.arange(0.625, 1.0, 0.05)]})
I75 = LAMBDAS.index(0.75)


def circuit_of(run):
    with h5py.File(core.data(H5), "r") as f:
        return core.load_circuit(f, run)


def blend_p(w_generic, p, lam):
    """core.blend_prices with LAM = lam, turned back into probabilities as core.two_fold_blend does."""
    w = {k: w_generic[k] if k[1] is None else (1 - lam) * w_generic[k] + lam * np.log((1 - p[k]) / p[k])
         for k in w_generic}
    return {k: 1 / (1 + np.exp(w[k])) for k in w}


def sweep(edges, w_generic, dets, obs, learn, test):
    """Learn p_ij on the shots `learn`, decode the shots `test` at every lambda.
    Returns a (lambdas x test shots) array: True where the decoder got the shot wrong."""
    p = core.pij(dets[learn], edges)
    return np.array([core.decoder(edges, blend_p(w_generic, p, lam)).decode_batch(dets[test])[:, 0] != obs[test]
                     for lam in LAMBDAS])


def one_run(run):
    d = np.load(core.data("detectors.npz"))
    dets, obs = d[run + "/dets"], d[run + "/obs"]
    generic, edges, w_generic = core.graph(circuit_of(run))
    n = len(obs)
    even, odd = np.arange(0, n, 2), np.arange(1, n, 2)
    outer = np.zeros((len(LAMBDAS), n), dtype=bool)     # every shot graded once, at every lambda
    inner = {}                                          # errors inside each learning half, per lambda
    for fold, learn, test in (("learn_even", even, odd), ("learn_odd", odd, even)):
        outer[:, test] = sweep(edges, w_generic, dets, obs, learn, test)
        a, b = learn[0::2], learn[1::2]                 # split the learning half in two again
        e = np.zeros((len(LAMBDAS), len(learn)), dtype=bool)
        e[:, 1::2] = sweep(edges, w_generic, dets, obs, a, b)
        e[:, 0::2] = sweep(edges, w_generic, dets, obs, b, a)
        inner[fold] = e.sum(1)
    wrong_generic = generic.decode_batch(dets)[:, 0] != obs
    print(f"done {run}", flush=True)
    return run, int(d[run + "/t"].max() + 1), outer, inner, wrong_generic


def paired(a, b):
    """a, b: per-shot errors of two decoders on the same shots. Difference of rates (a - b)
    with a 95 % range, and the exact McNemar p-value."""
    n10, n01, n = int((a & ~b).sum()), int((~a & b).sum()), len(a)
    diff = (n10 - n01) / n
    se = np.sqrt(n10 + n01 - (n10 - n01) ** 2 / n) / n
    p = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
    return diff, diff - 1.96 * se, diff + 1.96 * se, p


def main():
    d = np.load(core.data("detectors.npz"))
    runs = [r for r in core.runs_of(d) if r != "circuit_2"]
    with mp.Pool(CORES) as pool:
        res = sorted(pool.map(one_run, runs), key=lambda r: r[1])

    T = [r[1] for r in res]
    W = np.concatenate([r[2] for r in res], axis=1)                 # lambdas x all 1.1 M shots
    G = np.concatenate([r[4] for r in res])
    ler = W.mean(1)
    best = int(np.argmin(ler))

    print(f"\nchecks: generic weights {G.mean():.4%}, lambda = 0 {ler[0]:.4%}  (guide: 23.61 %);  "
          f"lambda = 0.75 {ler[I75]:.4%}  (guide: 22.57 %)\n")
    print(f"{'lambda':>7}{'pooled':>10}{'vs 0.75 (points)':>19}{'95 % range':>22}{'McNemar p':>12}")
    rows = []
    for i, lam in enumerate(LAMBDAS):
        diff, lo, hi, p = paired(W[i], W[I75])
        rows.append(dict(lam=lam, ler=ler[i], diff=diff, lo=lo, hi=hi, p=p))
        mark = "  <- lowest" if i == best else ("  <- guide" if i == I75 else "")
        print(f"{lam:>7.3f}{ler[i]:>10.4%}{100 * diff:>+19.3f}{f'{100 * lo:+.3f} to {100 * hi:+.3f}':>22}{p:>12.3g}{mark}")

    print("\nlowest lambda in each run separately:")
    per_run = {}
    for run, t, outer, _, _ in res:
        l = outer.mean(1)
        per_run[run] = dict(T=t, best=LAMBDAS[int(np.argmin(l))], ler_best=l.min(), ler_075=l[I75])
        print(f"  {run:<11} T={t:>2}: best {per_run[run]['best']:.3f}  ({l.min():.4%}; at 0.75: {l[I75]:.4%})")

    # nested: lambda chosen inside each learning half, applied to the other half
    chosen = {f: int(np.argmin(sum(r[3][f] for r in res))) for f in ("learn_even", "learn_odd")}
    nested = []
    for run, t, outer, _, _ in res:
        w = np.zeros(outer.shape[1], dtype=bool)
        w[1::2] = outer[chosen["learn_even"], 1::2]                  # odd shots: learned on even
        w[0::2] = outer[chosen["learn_odd"], 0::2]                   # even shots: learned on odd
        nested.append(w)
    nested = np.concatenate(nested)
    diff, lo, hi, p = paired(nested, W[I75])
    print(f"\nnested choice: lambda = {LAMBDAS[chosen['learn_even']]} (learning on even shots), "
          f"{LAMBDAS[chosen['learn_odd']]} (learning on odd shots)")
    print(f"  pooled error rate {nested.mean():.4%};  against fixed 0.75: {100 * diff:+.3f} points "
          f"({100 * lo:+.3f} to {100 * hi:+.3f}), McNemar p = {p:.3g}")

    # choose on the short runs (T = 5-13), test on the long ones (T = 15-25)
    short = np.concatenate([r[2] for r in res if r[1] <= 13], axis=1).mean(1)
    long_ = np.concatenate([r[2] for r in res if r[1] >= 15], axis=1).mean(1)
    i_s = int(np.argmin(short))
    print(f"\nchosen on the short runs: lambda = {LAMBDAS[i_s]}; on the long runs that gives {long_[i_s]:.4%} "
          f"(0.75: {long_[I75]:.4%}; best possible there: {long_.min():.4%} at {LAMBDAS[int(np.argmin(long_))]})")

    json.dump(dict(lambdas=LAMBDAS, pooled=rows, per_run=per_run,
                   nested=dict(chosen={f: LAMBDAS[i] for f, i in chosen.items()}, ler=nested.mean(),
                               diff_vs_075=diff, lo=lo, hi=hi, p=p),
                   short_long=dict(chosen=LAMBDAS[i_s], long_at_chosen=long_[i_s], long_at_075=long_[I75])),
              open(core.out("lambda_scan.json"), "w"), indent=1, default=float)
    draw(rows)
    print(f"\nwrote {core.out('lambda_scan.json')} and {core.out('lambda_scan.png')}")


def draw(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, line, band = "#1f2a30", "#6b7780", "#1d5f86", "#cfe0ea"
    lam = np.array([r["lam"] for r in rows])
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.2, 3.4), constrained_layout=True)
    a.plot(lam, [100 * r["ler"] for r in rows], "-o", color=line, lw=2, ms=4)
    a.set_xlabel("shrinkage $\\lambda$"); a.set_ylabel("logical error rate (%)")
    a.set_title("A   Pooled over 11 runs (1.1 M shots)", loc="left", fontsize=10, color=ink)
    m = lam >= 0.5
    d, lo, hi = (np.array([100 * r[k] for r in rows])[m] for k in ("diff", "lo", "hi"))
    b.fill_between(lam[m], lo, hi, color=band, lw=0)
    b.plot(lam[m], d, "-o", color=line, lw=2, ms=4)
    b.axhline(0, color=muted, lw=0.8)
    b.set_xlabel("shrinkage $\\lambda$"); b.set_ylabel("difference from $\\lambda$ = 0.75 (points)")
    b.set_title("B   Same shots, paired; band = 95 % range", loc="left", fontsize=10, color=ink)
    for ax in (a, b):
        ax.grid(alpha=0.25, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.savefig(core.out("lambda_scan.png"), dpi=200)


if __name__ == "__main__":
    main()
