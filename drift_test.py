"""Does drift hurt decoding? Learn the blend weights from one part of a run, decode another part.

Run after preprocess.py. Every shot is decoded exactly once in each of these ways of choosing the
shots the decoder learns from:

  spread : learn from the even shots, decode the odd ones, and swap (what pij.py does).
           The learning shots are spread over the whole run, so drift averages out.
  halves : learn from the first half of the run, decode the second half, and swap.
           The learning shots were taken about 30-60 seconds away from the shots being decoded.
  near   : cut the run into 10 blocks of 10,000 shots; decode each block with weights learned
           from its 2 nearest blocks
  far    : the same, but learned from its 2 farthest blocks (same amount of data as near)

If drift matters for decoding, halves loses to spread and far loses to near.
A second check: do the learned weights themselves move? Compare the weights learned from the
two time halves with those learned from the even and odd shots (which differ only by chance).
"""
import h5py
import numpy as np

import ankaa_core as core

d = np.load(core.data("detectors.npz"))
hdf = h5py.File(core.data("stability_8_without_resets_raw_data.h5"), "r")
runs = core.runs_of(d)
runs.remove("circuit_2")


def decode(plan, edges, w_generic, dets, obs):
    """plan = list of (shots to learn from, shots to decode). Returns 1 per shot decoded wrongly."""
    wrong = np.zeros(len(obs), dtype=bool)
    for learn, test in plan:
        m = core.matching(edges, core.blend_prices(edges, w_generic, dets[learn]))
        wrong[test] = m.decode_batch(dets[test])[:, 0] != obs[test]
    return wrong


def compare(a, b):
    """Paired test: shots only a got wrong vs shots only b got wrong. Returns (difference in % points, p-value)."""
    diff, _, _, p = core.paired(a, b)
    return diff, p


def ci95(a, b):
    """95% range of the paired difference (a - b) in % points."""
    _, lo, hi, _ = core.paired(a, b)
    return lo, hi


rows, pooled = [], {k: [] for k in ("spread", "halves", "near", "far")}
for run in runs:
    dets, obs = d[run + "/dets"], d[run + "/obs"]
    n = len(obs)
    _, edges, w_generic = core.graph(core.load_circuit(hdf, run))

    shots = np.arange(n)
    even, odd = shots[0::2], shots[1::2]
    first, second = shots[: n // 2], shots[n // 2:]
    block = np.array_split(shots, 10)
    near_of = {b: ([1, 2] if b == 0 else [8, 7] if b == 9 else [b - 1, b + 1]) for b in range(10)}
    far_of = {b: ([9, 8] if b <= 4 else [0, 1]) for b in range(10)}

    wrong = {
        "spread": decode([(even, odd), (odd, even)], edges, w_generic, dets, obs),
        "halves": decode([(first, second), (second, first)], edges, w_generic, dets, obs),
        "near": decode([(np.concatenate([block[c] for c in near_of[b]]), block[b]) for b in range(10)], edges, w_generic, dets, obs),
        "far": decode([(np.concatenate([block[c] for c in far_of[b]]), block[b]) for b in range(10)], edges, w_generic, dets, obs),
    }
    for k in pooled:
        pooled[k].append(wrong[k])

    # do the weights themselves move? (pair edges only; the doors are fixed)
    pairs = [k for k in w_generic if k[1] is not None]
    w_e, w_o = core.blend_prices(edges, w_generic, dets[even]), core.blend_prices(edges, w_generic, dets[odd])
    w_1, w_2 = core.blend_prices(edges, w_generic, dets[first]), core.blend_prices(edges, w_generic, dets[second])
    chance = np.sqrt(np.mean([(w_e[k] - w_o[k]) ** 2 for k in pairs]))       # differ only by chance
    in_time = np.sqrt(np.mean([(w_1[k] - w_2[k]) ** 2 for k in pairs]))      # chance + drift
    shift = np.mean([w_2[k] - w_1[k] for k in pairs])                        # < 0: errors more likely later in the run

    rows.append((run, int(d[run + "/rounds"]) - 1, {k: v.mean() for k, v in wrong.items()},
                 compare(wrong["halves"], wrong["spread"]), compare(wrong["far"], wrong["near"]), chance, in_time, shift))
    print(f"done {run}", flush=True)

print(f"\n{'run':<12}{'rounds':>7}{'spread':>9}{'halves':>9}{'near':>9}{'far':>9}"
      f"{'halves-spread':>15}{'p':>8}{'far-near':>11}{'p':>8}{'weight change: chance':>23}{'time':>7}{'2nd-1st':>9}")
for run, T, ler, (dh, ph), (df, pf), chance, in_time, shift in rows:
    print(f"{run:<12}{T:>7}{ler['spread']:>9.4f}{ler['halves']:>9.4f}{ler['near']:>9.4f}{ler['far']:>9.4f}"
          f"{dh:>+15.3f}{ph:>8.3f}{df:>+11.3f}{pf:>8.3f}{chance:>23.3f}{in_time:>7.3f}{shift:>+9.3f}")

all_ = {k: np.concatenate(v) for k, v in pooled.items()}
dh, ph = compare(all_["halves"], all_["spread"])
df, pf = compare(all_["far"], all_["near"])
print(f"\npooled over {len(all_['spread']):,} shots:")
print(f"  halves - spread = {dh:+.3f} % points, 95% range [{ci95(all_['halves'], all_['spread'])[0]:+.3f}, "
      f"{ci95(all_['halves'], all_['spread'])[1]:+.3f}]  (p = {ph:.2g})")
print(f"  far    - near   = {df:+.3f} % points, 95% range [{ci95(all_['far'], all_['near'])[0]:+.3f}, "
      f"{ci95(all_['far'], all_['near'])[1]:+.3f}]  (p = {pf:.2g})")
print(f"  time halves differ by more than chance in {sum(r[6] > r[5] for r in rows)} of {len(rows)} runs")
print(f"  weights from the two time halves differ by {np.mean([r[6] for r in rows]):.3f} on average; "
      f"even vs odd (pure chance) differ by {np.mean([r[5] for r in rows]):.3f}")
