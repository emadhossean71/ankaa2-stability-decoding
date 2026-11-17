"""PPO tuning of tied weight multipliers on the WITH-resets data.

Run after preprocess_resets.py, in PyCharm. The same method and settings as ppo.py (the decoder
pieces and the PPO loop are in ankaa_core.py). What the resets change is handled inside
core.ppo_problem: every same-ancilla edge and every door is a misread edge with its own per-shot
soft price, so 28 knobs remain (data 12, diag 12, soft 4). There are no FPGA verdicts for these
runs, so the ladder is measured against Riverlane's best published decoder.

circuit_22 with both directions takes about half an hour with CORES = 4.
Results are written to ppo_resets_{run}_learn_{half}.json in this folder.
"""
import json
import multiprocessing as mp

import numpy as np

import ankaa_core as core

# ---------------------------------------------------------------- settings
RUNS = ["circuit_22"]            # any of circuit_6, circuit_10, circuit_14, circuit_18, circuit_22
LEARN_ON = ["even", "odd"]       # both directions, so every shot is graded once; ["even"] alone takes half the time
CORES = 4                        # processes decoding in parallel
PPO = dict(B=24, NSHOT=8000, K=20, EPOCHS=50,
           EPS=0.2, ETA=0.002, SIG0=0.15, GCLIP=0.1, SIG_MIN=0.02)   # identical to ppo.py


def ladder(run, T, mwpm, blend, plain, tuned, best):
    """Every decoder's error rate next to Riverlane's best published one (all 100,000 shots;
    its own statistical spread is about +-0.12 points)."""
    n = len(tuned)
    here = f"these {n:,} shots"
    rows = [("Our MWPM (hard readings, generic weights)", mwpm.mean(), here),
            ("Our blend", blend.mean(), here),
            ("Riverlane's best (soft information + p_ij)", best, "published, all 100,000 shots"),
            ("Our blend + soft", plain.mean(), here),
            ("Blend + soft + PPO-tuned knobs", tuned.mean(), here)]
    print(f"\nWhere it sits on the ladder ({run}, {T} decoding rounds, with resets)")
    print(f"{'decoder':<46}{'error rate':>11}{'vs Riverlane best':>19}   graded on")
    for name, ler, where in rows:
        vs = "-" if name.startswith("Riverlane's best") else f"{100 * (ler - best):+.2f} pts"
        print(f"{name:<46}{100 * ler:>10.2f}%{vs:>19}   {where}")
    return [(name, float(ler), where) for name, ler, where in rows]


def main():
    best = {int(n): v for n, v, err in np.loadtxt(core.data("lep_mwpm_soft_info_pairwise_correlation_graph_with_resets.txt"), skiprows=1)}
    for run in RUNS:
        pooled_t, pooled_p, pooled_m, pooled_b = [], [], [], []
        for half in LEARN_ON:
            print(f"\n=== {run} (with resets): learn on the {half} shots, grade on the others ===", flush=True)
            prob = core.ppo_problem("detectors_with_resets.npz", "stability_8_with_resets_raw_data.h5", run, half)
            print(f"{len(prob['names'])} knobs: {', '.join(prob['names'])}", flush=True)
            with mp.get_context("spawn").Pool(CORES, initializer=core.init_worker, initargs=(prob,)) as pool:
                mu, rho = core.train(pool, prob, **PPO)
                tuned, plain = core.grade(pool, prob, mu, prob["test"], CORES)
            pooled_t.append(tuned); pooled_p.append(plain)
            pooled_m.append(prob["mwpm_wrong"]); pooled_b.append(prob["blend_wrong"])
            d_, lo, hi, p = core.paired(tuned, plain)
            print(f"\nheld-out shots: untuned {plain.mean():.4f}   tuned {tuned.mean():.4f}   "
                  f"difference {d_:+.2f} % points [{lo:+.2f}, {hi:+.2f}]   p = {p:.1e}")
            print("\nknob                  multiplier   width")
            for name, m_, r_ in sorted(zip(prob["names"], mu, rho), key=lambda x: x[1]):
                print(f"{name:<22}{np.exp(m_):>10.3f}{np.exp(r_):>8.3f}")
            rows = ladder(run, prob["T"], prob["mwpm_wrong"], prob["blend_wrong"], plain, tuned, best.get(prob["T"], float("nan")))
            json.dump(dict(run=run, resets=True, learn_on=half, untuned=plain.mean(), tuned=tuned.mean(), diff_pts=d_, ci=[lo, hi], p=p,
                           ladder=rows, knobs={n: [float(np.exp(m_)), float(np.exp(r_))] for n, m_, r_ in zip(prob["names"], mu, rho)}),
                      open(core.out(f"ppo_resets_{run}_learn_{half}.json"), "w"), indent=1, default=float)
        if len(LEARN_ON) == 2:                   # both directions: every shot graded exactly once
            d_, lo, hi, p = core.paired(np.concatenate(pooled_t), np.concatenate(pooled_p))
            print(f"\n{run}, every shot graded once: untuned {np.concatenate(pooled_p).mean():.4f}   "
                  f"tuned {np.concatenate(pooled_t).mean():.4f}   difference {d_:+.2f} % points [{lo:+.2f}, {hi:+.2f}]   p = {p:.1e}")
            ladder(run, prob["T"], np.concatenate(pooled_m), np.concatenate(pooled_b), np.concatenate(pooled_p),
                   np.concatenate(pooled_t), best.get(prob["T"], float("nan")))


if __name__ == "__main__":
    main()
