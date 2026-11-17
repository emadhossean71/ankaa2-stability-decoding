"""PPO tuning of tied weight multipliers on top of the blend + soft-information decoder (no resets).

Run after preprocess.py, in PyCharm. The decoder pieces and the PPO loop are in ankaa_core.py;
this file holds the settings, the ladder printout and the main block.

The idea (Sivak, Newman & Klimov, PRL 133, 150603 (2024)):
  - start from the blend + soft decoder (every knob = 1)
  - tie the 400 edge weights into 44 groups ("knobs"), each a multiplier on its group
    (core.ppo_problem lists them: flip, data, diag, door1, soft)
  - wobble the knobs (B noisy copies per epoch), decode, score each copy by -log10(error rate),
    and move the knobs towards the copies that did better (PPO with a clipped ratio, Adam)
  - learn on one half of the shots (even or odd), then grade on the other half; everything learned
    (the blend, the IQ cloud fit and the knobs) comes from the learning half only

circuit_24 with the default settings takes about 12 minutes per direction with CORES = 4.
Results are written to ppo_{run}_learn_{half}.json in this folder.
"""
import json
import multiprocessing as mp

import numpy as np

import ankaa_core as core

# ---------------------------------------------------------------- settings
RUNS = ["circuit_24"]            # any of circuit_4 ... circuit_24
LEARN_ON = ["even", "odd"]       # both directions, so every shot is graded once; ["even"] alone takes half the time
CORES = 4                        # processes decoding in parallel
PPO = dict(B=24, NSHOT=8000, K=20, EPOCHS=50,                # candidates per epoch, shots per epoch, policy steps, epochs
           EPS=0.2, ETA=0.002, SIG0=0.15, GCLIP=0.1, SIG_MIN=0.02)   # clip, learning rate, initial width, gradient clip, width floor
# Riverlane's best decoder (MWPM + soft information + p_ij), per number of decoding rounds,
# from lep_mwpm_soft_info_pairwise_correlation_graph.txt, Zenodo record 15364358 (their MWPM is read from lep_mwpm.txt)
RIVERLANE_SOFT = {5: 0.26818, 7: 0.24422, 9: 0.22745, 11: 0.22190, 13: 0.22143, 15: 0.20752,
                  17: 0.20747, 19: 0.19434, 21: 0.18241, 23: 0.19390, 25: 0.17623}


def ladder(run, T, fpga, blend, plain, tuned, mwpm):
    """Every decoder's error rate next to Riverlane's FPGA. Ours are on the graded shots; Riverlane's
    offline numbers are their published ones (all 100,000 shots)."""
    n = len(tuned)
    here = f"these {n:,} shots"
    pub = "published, all 100,000 shots"
    rows = [("Riverlane's FPGA (real time)", fpga.mean(), here),
            ("Riverlane's offline MWPM", mwpm, pub),
            ("Our blend", blend.mean(), here),
            ("Riverlane's best (soft information + p_ij)", RIVERLANE_SOFT.get(T, float("nan")), pub),
            ("Our blend + soft", plain.mean(), here),
            ("Blend + soft + PPO-tuned knobs", tuned.mean(), here)]
    f = fpga.mean()
    print(f"\nWhere it sits on the ladder ({run}, {T} decoding rounds)")
    print(f"{'decoder':<46}{'error rate':>11}{'vs FPGA':>11}{'fewer failures':>16}   graded on")
    for name, ler, where in rows:
        vs = "-" if name.startswith("Riverlane's FPGA") else f"{100 * (ler - f):+.2f} pts"
        fewer = "-" if name.startswith("Riverlane's FPGA") else f"{100 * (f - ler) / f:.0f}%"
        print(f"{name:<46}{100 * ler:>10.2f}%{vs:>11}{fewer:>16}   {where}")
    return [(name, float(ler), where) for name, ler, where in rows]


def main():
    mwpm = {int(n): v for n, v, err in np.loadtxt(core.data("lep_mwpm.txt"), skiprows=1)}     # Riverlane's published MWPM
    for run in RUNS:
        pooled_t, pooled_p, pooled_f, pooled_b = [], [], [], []
        for half in LEARN_ON:
            print(f"\n=== {run}: learn on the {half} shots, grade on the others ===", flush=True)
            prob = core.ppo_problem("detectors.npz", "stability_8_without_resets_raw_data.h5", run, half)
            print(f"{len(prob['names'])} knobs", flush=True)
            with mp.get_context("spawn").Pool(CORES, initializer=core.init_worker, initargs=(prob,)) as pool:
                mu, rho = core.train(pool, prob, **PPO)
                tuned, plain = core.grade(pool, prob, mu, prob["test"], CORES)
            pooled_t.append(tuned); pooled_p.append(plain)
            pooled_f.append(prob["fpga_wrong"]); pooled_b.append(prob["blend_wrong"])
            d_, lo, hi, p = core.paired(tuned, plain)
            print(f"\nheld-out shots: untuned {plain.mean():.4f}   tuned {tuned.mean():.4f}   "
                  f"difference {d_:+.2f} % points [{lo:+.2f}, {hi:+.2f}]   p = {p:.1e}")
            print("\nknob                  multiplier   width")
            for name, m_, r_ in sorted(zip(prob["names"], mu, rho), key=lambda x: x[1]):
                print(f"{name:<22}{np.exp(m_):>10.3f}{np.exp(r_):>8.3f}")
            rows = ladder(run, prob["T"], prob["fpga_wrong"], prob["blend_wrong"], plain, tuned, mwpm.get(prob["T"], float("nan")))
            json.dump(dict(run=run, learn_on=half, untuned=plain.mean(), tuned=tuned.mean(), diff_pts=d_, ci=[lo, hi], p=p, ladder=rows,
                           knobs={n: [float(np.exp(m_)), float(np.exp(r_))] for n, m_, r_ in zip(prob["names"], mu, rho)}),
                      open(core.out(f"ppo_{run}_learn_{half}.json"), "w"), indent=1, default=float)
        if len(LEARN_ON) == 2:                   # both directions: every shot graded exactly once
            d_, lo, hi, p = core.paired(np.concatenate(pooled_t), np.concatenate(pooled_p))
            print(f"\n{run}, every shot graded once: untuned {np.concatenate(pooled_p).mean():.4f}   "
                  f"tuned {np.concatenate(pooled_t).mean():.4f}   difference {d_:+.2f} % points [{lo:+.2f}, {hi:+.2f}]   p = {p:.1e}")
            ladder(run, prob["T"], np.concatenate(pooled_f), np.concatenate(pooled_b), np.concatenate(pooled_p),
                   np.concatenate(pooled_t), mwpm.get(prob["T"], float("nan")))


if __name__ == "__main__":
    main()
