"""The eight figures of Part II of the guide, drawn from compare_resets.json (run compare_resets.py),
saved into compare_figs/ in this folder."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import ankaa_core as core

INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, RED, PURPLE, GOLD = "#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#4a3aa7", "#c99a06"
ANC_COL = {"36": RED, "38": BLUE, "50": AQUA, "52": PURPLE}
NR, WR = "no resets", "with resets"
STYLE = {NR: dict(ls="--", mfc="white"), WR: dict(ls="-")}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": MUTED})


def style(ax, title, grid="y"):
    ax.set_title(title, loc="left", color=INK, fontsize=10, weight="bold")
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def dataset_legend(ax, loc="upper right", extra=()):
    h = [Line2D([], [], color=INK2, ls="-", marker="o", ms=4, label="with resets"),
         Line2D([], [], color=INK2, ls="--", marker="o", mfc="white", ms=4, label="no resets"), *extra]
    return ax.legend(handles=h, frameon=False, fontsize=8, loc=loc)


def save(fig, name):
    fig.savefig(core.out(f"compare_figs/{name}.pdf"))
    fig.savefig(core.out(f"compare_figs/{name}.png"), dpi=110)
    plt.close(fig)


def draw(R):
    D = R["decoders"]
    TN = sorted(int(t) for t in D[NR]); TW = sorted(int(t) for t in D[WR])
    col = lambda name, T, k: np.array([100 * D[name][str(t) if str(t) in D[name] else t][k] for t in T])
    ppo = {name: R[name]["ppo"] for name in (NR, WR)}

    # ---------------------------------------------------- 1. error rate against rounds
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), sharey=True)
    ax = axes[0]
    for k, lab, c in (("generic", "our MWPM (Riverlane's noise model)", AQUA), ("pij", "p$_{ij}$", RED), ("blend", "blend", BLUE)):
        ax.plot(TN, col(NR, TN, k), color=c, marker="o", ms=4, lw=1.6, **STYLE[NR])
        ax.plot(TW, col(WR, TW, k), color=c, marker="o", ms=4, lw=2.0, label=lab, **STYLE[WR])
    ax.plot(TN, col(NR, TN, "rl_mwpm"), color=MUTED, ls=":", lw=1.4, label="Riverlane MWPM, no resets (published)")
    style(ax, "A   Hard measurements only"); ax.set_xlabel("decoding rounds  T"); ax.set_ylabel("logical error rate (%)")
    ax.set_xticks(TN)
    l1 = ax.legend(frameon=False, fontsize=8, loc="upper right"); ax.add_artist(l1); dataset_legend(ax, loc="lower left")
    ax = axes[1]
    for k, lab, c in (("generic_soft", "our MWPM + soft", AQUA), ("blend_soft", "blend + soft", BLUE), ("rl_best", "Riverlane best (published)", INK2)):
        ax.plot(TN, col(NR, TN, k), color=c, marker="o", ms=4, lw=1.6, **STYLE[NR])
        ax.plot(TW, col(WR, TW, k), color=c, marker="o", ms=4, lw=2.0, label=lab, **STYLE[WR])
    ax.plot([23], [100 * ppo[WR]["23"]["both"]["tuned"]], marker="*", ms=15, color=ORANGE, mec="white", ls="none", label="+ PPO-tuned knobs, with resets")
    ax.plot([23, 25], [100 * ppo[NR][t]["both"]["tuned"] for t in ("23", "25")], marker="*", ms=14, color="white", mec=ORANGE, mew=1.4, ls="none",
            label="+ PPO-tuned knobs, no resets")
    style(ax, "B   With soft information"); ax.set_xlabel("decoding rounds  T"); ax.set_xticks(TN)
    l1 = ax.legend(frameon=False, fontsize=8, loc="upper right"); ax.add_artist(l1); dataset_legend(ax, loc="lower left")
    ax.set_ylim(13.5, 35.5)
    plt.tight_layout(); save(fig, "wr_ler")

    # ---------------------------------------------------- 2. what the resets are worth, and the soft gain
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.1))
    ax = axes[0]; ax.axhline(0, color=AXIS, lw=1)
    items = (("generic", "our MWPM", AQUA), ("blend", "blend", BLUE), ("rl_best", "Riverlane best", INK2),
             ("generic_soft", "our MWPM + soft", GOLD), ("blend_soft", "blend + soft", ORANGE))
    for i, (k, lab, c) in enumerate(items):
        T = np.array(TW) + (i - 2) * 0.32
        y = [R["matched"][str(t)][k][0] for t in TW]; e = [R["matched"][str(t)][k][1] for t in TW]
        ax.errorbar(T, y, yerr=e, color=c, marker="o", ms=4, lw=1.6, capsize=2, label=lab)
    style(ax, "A   Resets advantage at equal length"); ax.set_xticks(TW); ax.set_xlabel("decoding rounds  T")
    ax.set_ylabel("no resets minus with resets (points)"); ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax = axes[1]
    for name in (NR, WR):
        T = TN if name == NR else TW
        ax.plot(T, col(name, T, "blend") - col(name, T, "blend_soft"), color=BLUE, marker="o", ms=4, lw=1.8, **STYLE[name])
        ax.plot(T, col(name, T, "generic") - col(name, T, "generic_soft"), color=AQUA, marker="o", ms=4, lw=1.5, **STYLE[name])
    style(ax, "B   What soft information gains"); ax.set_xticks(TN); ax.set_xlabel("decoding rounds  T")
    ax.set_ylabel("error rate removed (points)"); ax.set_ylim(0, 4.5)
    dataset_legend(ax, loc="lower left", extra=(Line2D([], [], color=BLUE, lw=2, label="on the blend"),
                                                Line2D([], [], color=AQUA, lw=2, label="on our MWPM")))
    ax = axes[2]; ax.axhline(0, color=AXIS, lw=1)
    for name in (NR, WR):
        T = TN if name == NR else TW
        ax.plot(T, col(name, T, "pij") - col(name, T, "generic"), color=RED, marker="o", ms=4, lw=1.8, **STYLE[name])
        ax.plot(T, col(name, T, "blend") - col(name, T, "generic"), color=BLUE, marker="o", ms=4, lw=1.8, **STYLE[name])
    style(ax, "C   Learned weights against the model"); ax.set_xticks(TN); ax.set_xlabel("decoding rounds  T")
    ax.set_ylabel("minus our MWPM (points)")
    dataset_legend(ax, loc="upper left", extra=(Line2D([], [], color=RED, lw=2, label="p$_{ij}$"), Line2D([], [], color=BLUE, lw=2, label="blend")))
    plt.tight_layout(); save(fig, "wr_gap")

    # ---------------------------------------------------- 3. the device: firing per round, the session clock, busy shots
    fig = plt.figure(figsize=(12.5, 4.3)); gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.2, 1], wspace=0.3, left=0.06, right=0.98, top=0.91, bottom=0.12)
    ax = fig.add_subplot(gs[0])
    for name in (NR, WR):
        for a, rates in R[name]["c22"]["rate"].items():
            ax.plot(np.arange(1, len(rates) + 1), 100 * np.array(rates), color=ANC_COL[a], lw=1.6, ls=STYLE[name]["ls"],
                    label=f"ancilla {a}" if name == WR else None)
    style(ax, "A   Firing rate through the shot (T = 23)"); ax.set_xlabel("detector round  k"); ax.set_ylabel("detectors firing (%)")
    ax.set_ylim(26.3, 39.5)
    l1 = ax.legend(frameon=False, fontsize=7.5, loc="upper center", ncol=4, columnspacing=0.8, handlelength=1.4); ax.add_artist(l1)
    ax.legend(handles=[Line2D([], [], color=INK2, ls="-", label="with resets"), Line2D([], [], color=INK2, ls="--", label="no resets")],
              frameon=False, fontsize=7.5, loc="lower right", ncol=2)
    ax = fig.add_subplot(gs[1])
    for name, c in ((NR, MUTED), (WR, BLUE)):
        runs = R[name]["runs"]
        m = np.array([v["minute"] - 14 * 60 for v in runs.values()])
        f = 100 * np.array([v["first2"] for v in runs.values()]); e = 196 * np.array([v["first2_se"] for v in runs.values()])
        ax.errorbar(m, f, yerr=e, color=c, marker="o", ms=4, lw=1.4, capsize=2, ls=STYLE[name]["ls"], mfc=STYLE[name].get("mfc", c),
                    label=f"{name} ({len(runs)} runs, shortest first)")
        runs_last = list(runs)[-1]
        for (run, v), mm, ff in zip(runs.items(), m, f):
            if run in ("circuit_2", runs_last):
                ax.annotate(run.replace("circuit_", "c"), (mm, ff), textcoords="offset points", xytext=(0, 7), fontsize=7, color=c, ha="center")
    style(ax, "B   Rounds 1-2 against the clock"); ax.set_xlabel("minutes after 14:00, 14 June 2024"); ax.set_ylabel("detectors firing in rounds 1-2 (%)")
    ax.legend(frameon=False, fontsize=7.5, loc="center right")
    ax = fig.add_subplot(gs[2])
    for name, c in ((NR, MUTED), (WR, BLUE)):
        runs = R[name]["runs"]
        ax.plot([v["T"] for v in runs.values()], [v["var_ratio"] for v in runs.values()], color=c, marker="o", ms=4, lw=1.6,
                ls=STYLE[name]["ls"], mfc=STYLE[name].get("mfc", c), label=f"device, {name}")
        ax.plot([23], [R[name]["c22"]["sim_var_ratio"]], marker="D", ms=6, color=c, mfc="white" if name == NR else c, ls="none",
                label=f"independent errors, {name}")
    ax.axhline(1, color=AXIS, lw=1)
    style(ax, "C   Busy shots"); ax.set_xlabel("decoding rounds  T"); ax.set_ylabel(r"Var(n) / $\Sigma_i\,x_i(1-x_i)$")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left"); ax.set_ylim(0.8, 4.0)
    save(fig, "wr_device")

    # ---------------------------------------------------- 4. lingering
    fig = plt.figure(figsize=(12, 7.9)); gs = fig.add_gridspec(2, 6, height_ratios=[1, 1.35], hspace=0.32, wspace=0.9, left=0.06, right=0.9, top=0.95, bottom=0.05)
    for i, (kind, title) in enumerate((("same", "A   Same ancilla"), ("neighbours", "B   Neighbouring ancillas"), ("opposite", "C   Opposite ancillas"))):
        ax = fig.add_subplot(gs[0, 2 * i:2 * i + 2]); ax.axhline(0, color=AXIS, lw=1)
        for name, c in ((NR, MUTED), (WR, BLUE)):
            e, s = R[name]["c22"]["excess"][kind], R[name]["c22"]["sim_excess"][kind]
            sep = [int(k) for k in e]
            ax.plot(sep, list(e.values()), color=c, lw=1.8, marker="o", ms=3, ls=STYLE[name]["ls"], mfc=STYLE[name].get("mfc", c), label=f"device, {name}")
            ax.plot(sep, list(s.values()), color=c, lw=1.0, ls=":", label=f"independent errors, {name}")
            reach = R[name]["c22"]["reach"][kind]
            if reach is not None:
                ax.axvline(reach + 0.5, color=c, lw=0.8, ls=STYLE[name]["ls"], alpha=0.6)
        style(ax, title); ax.set_xlabel("rounds apart"); ax.set_xlim(-0.3, 20.5)
        if i == 0:
            ax.set_ylabel("excess chance (points)"); ax.legend(frameon=False, fontsize=7, loc="center right")
        ax.set_ylim(-0.6, 8.5)
        if kind == "same":
            v = [R[name]["c22"]["excess"]["same"]["1"] for name in (NR, WR)]
            ax.annotate(f"1 round apart (off scale):\n{v[0]:.1f} no resets, {v[1]:.1f} with resets", (1, 8.4), xytext=(3.2, 6.6),
                        fontsize=7, color=INK2, arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8))
    for i, name in enumerate((NR, WR)):
        ax = fig.add_subplot(gs[1, 3 * i:3 * i + 3])
        M = np.load(core.out(f"compare_figs/excess_{name.replace(' ', '_')}.npy")).copy(); np.fill_diagonal(M, np.nan)
        im = ax.imshow(M, cmap="magma_r", vmin=0, vmax=8, interpolation="nearest")
        n = len(M) // 4
        for k in range(1, 4):
            ax.axhline(k * n - 0.5, color="white", lw=0.8); ax.axvline(k * n - 0.5, color="white", lw=0.8)
        ax.set_xticks([n * k + n / 2 - 0.5 for k in range(4)]); ax.set_xticklabels(["36", "38", "50", "52"])
        ax.set_yticks([n * k + n / 2 - 0.5 for k in range(4)]); ax.set_yticklabels(["36", "38", "50", "52"])
        ax.set_title(f"{'DE'[i]}   Every detector pair, {name} (T = 23)", loc="left", color=INK, fontsize=10, weight="bold")
        ax.tick_params(length=0)
    cax = fig.add_axes([0.92, 0.08, 0.012, 0.36]); fig.colorbar(im, cax=cax).set_label("excess chance (points)", color=INK2)
    save(fig, "wr_linger")

    # ---------------------------------------------------- 5. why p_ij works with resets: the doors
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.0), gridspec_kw=dict(width_ratios=[1.5, 1]))
    ax = axes[0]; ax.axhline(0, color=INK2, lw=1)
    x0 = 0
    ticks, labels = [], []
    for name, c in ((NR, MUTED), (WR, BLUE)):
        doors = sorted(R[name]["c22"]["doors"], key=lambda d: (d[1], d[0]))
        for j, (a, k, f, b, *_) in enumerate(doors):
            ax.bar(x0 + j, b, color=c if f else "white", edgecolor=c, hatch=None if f else "///", lw=1)
            ticks.append(x0 + j); labels.append(f"{a}\nk={k}")
        ax.text(x0 + len(doors) / 2 - 0.5, 0.21, name, ha="center", color=c, fontsize=9, weight="bold")
        x0 += len(doors) + 1.5
    ax.set_xticks(ticks); ax.set_xticklabels(labels, fontsize=6.3)
    ax.set_ylim(-0.8, 0.26); style(ax, "A   Door leftover b = (x - q) / (1 - 2q), T = 23, even half")
    ax.set_ylabel("leftover probability b (not clipped)")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=INK2, label="door flips the observable"),
                       plt.Rectangle((0, 0), 1, 1, fc="white", ec=INK2, hatch="///", label="door does not")], frameon=False, fontsize=7.5, loc="lower left")
    ax = axes[1]
    for name, c in ((NR, MUTED), (WR, BLUE)):
        runs = [v for v in R[name]["runs"].values() if "doors" in v]
        ax.plot([v["T"] for v in runs], [100 * v["doors_at_floor"] / v["doors"] for v in runs], color=c, marker="o", ms=4, lw=1.8,
                ls=STYLE[name]["ls"], mfc=STYLE[name].get("mfc", c), label=f"all doors, {name}")
        ax.plot([v["T"] for v in runs], [100 * v["observable_doors_at_floor"] / v["observable_doors"] for v in runs], color=c, marker="s", ms=4,
                lw=1.0, ls=":", label=f"observable doors, {name}")
    style(ax, "B   Doors shut by p$_{ij}$ (leftover at the floor)"); ax.set_xlabel("decoding rounds  T"); ax.set_ylabel("doors at the floor (%)")
    ax.set_xticks(TN); ax.set_ylim(-5, 105); ax.legend(frameon=False, fontsize=7.5, loc="center right")
    plt.tight_layout(); save(fig, "wr_doors")

    # ---------------------------------------------------- 6. readout and odd IQ points
    fig = plt.figure(figsize=(12, 7.3)); gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.3, left=0.06, right=0.98, top=0.95, bottom=0.07)
    for i, name in enumerate((NR, WR)):
        ax = fig.add_subplot(gs[0, i]); z = np.load(core.out(f"compare_figs/iq50_{name.replace(' ', '_')}.npz"))
        ax.imshow(np.log10(z["H"].T + 1), origin="lower", extent=[z["xe"][0], z["xe"][-1], z["ye"][0], z["ye"][-1]], aspect="auto", cmap="magma")
        ax.axhline(float(z["cut"]), color="white", lw=1, ls="--")
        ax.text(z["xe"][0] + 0.2, float(z["cut"]) - 0.35, "odd: below this line", color="white", fontsize=7.5)
        style(ax, f"{'AB'[i]}   Ancilla 50's IQ points, {name}", grid=None)
        ax.set_xlabel("along the axis, 0 cloud to 1 cloud"); ax.set_ylabel("across the axis")
    ax = fig.add_subplot(gs[0, 2]); w = 0.38; A = ["36", "38", "50", "52"]
    for j, (name, c) in enumerate(((NR, MUTED), (WR, BLUE))):
        iq = R[name]["c22"]["iq"]
        ax.bar(np.arange(4) + (j - 0.5) * w, [100 * iq[a]["misread"] for a in A], w, color=c if name == WR else "white", edgecolor=c, label=name)
        for k, a in enumerate(A):
            ax.text(k + (j - 0.5) * w, 100 * iq[a]["misread"] + 0.1, f"{iq[a]['separation']:.2f}", ha="center", fontsize=6.5, color=c, rotation=90, va="bottom")
    ax.set_xticks(range(4)); ax.set_xticklabels(A); ax.set_xlabel("ancilla"); ax.set_ylim(0, 7.5)
    style(ax, "C   Readout quality, T = 23"); ax.set_ylabel("average soft misread probability (%)")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left"); ax.text(3.55, 7.3, "numbers above bars:\ncloud separation in widths", fontsize=7, color=MUTED, ha="right", va="top")
    ax = fig.add_subplot(gs[1, :2])
    for name in (NR, WR):
        for a in A:
            v = 100 * np.array(R[name]["c22"]["iq"][a]["odd_per_reading"])
            ax.plot(np.arange(1, len(v) + 1), v, color=ANC_COL[a], lw=2.0 if a == "50" else 1.0, ls=STYLE[name]["ls"],
                    label=f"ancilla {a}" if name == WR else None)
    style(ax, "D   Odd readings through the shot (T = 23)"); ax.set_xlabel("reading number"); ax.set_ylabel("odd readings (%)")
    l1 = ax.legend(frameon=False, fontsize=7.5, loc="upper left", ncol=4); ax.add_artist(l1)
    ax.legend(handles=[Line2D([], [], color=INK2, ls="-", label="with resets"), Line2D([], [], color=INK2, ls="--", label="no resets")],
              frameon=False, fontsize=7.5, loc="lower right")
    ax = fig.add_subplot(gs[1, 2])
    for j, (name, c) in enumerate(((NR, MUTED), (WR, BLUE))):
        iq = R[name]["c22"]["iq"]
        ax.bar(np.arange(4) + (j - 0.5) * w, [iq[a]["odd_memory"] for a in A], w, color=c if name == WR else "white", edgecolor=c, label=name)
    ax.axhline(1, color=INK2, lw=1, ls=":"); ax.text(3.55, 1.25, "no memory", fontsize=7, color=INK2, ha="right")
    ax.set_xticks(range(4)); ax.set_xticklabels(A); ax.set_xlabel("ancilla")
    style(ax, "E   Do odd readings come in streaks?"); ax.set_ylabel("odd after odd, relative to chance")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    save(fig, "wr_iq")

    # ---------------------------------------------------- 7. PPO
    fig = plt.figure(figsize=(12, 7.6)); gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], hspace=0.34, wspace=0.25, left=0.07, right=0.97, top=0.95, bottom=0.07)
    ax = fig.add_subplot(gs[0, :]); ax.axhline(0, color=AXIS, lw=1)
    for name, c in ((NR, MUTED), (WR, BLUE)):
        for j, d in enumerate(ppo[name]["23"]["curves"]):
            d = np.array(d); diff = 100 * (d[:, 1] - d[:, 2])
            avg = np.array([diff[max(0, i - 4):i + 1].mean() for i in range(len(diff))])
            ax.plot(d[:, 0], avg, color=c, lw=2.0 if j == 0 else 1.2, ls=STYLE[name]["ls"], label=f"{name}, learn {('even', 'odd')[j]}")
    style(ax, "A   Tuned minus untuned on each epoch's learning shots (5-epoch average), T = 23")
    ax.set_xlabel("epoch"); ax.set_ylabel("points"); ax.legend(frameon=False, fontsize=7.5, loc="lower left", ncol=2)
    kind_col = {"flip": RED, "data": AQUA, "diag": PURPLE, "door1": GOLD, "soft": ORANGE}

    def knob_points(name, T):
        k = ppo[name][T]["knobs"]
        return {n: (np.mean([k[h][n][0] for h in k]), abs(k["even"][n][0] - k["odd"][n][0]) / 2) for n in k["even"]}
    for i, (A_, B_, xl, yl, title) in enumerate(((knob_points(NR, "23"), knob_points(WR, "23"), "no resets, T = 23", "with resets, T = 23",
                                                  f"B   Same knobs, two experiments (r = {R['knob_r_resets_vs_noresets_T23'][0]:.2f})"),
                                                 (knob_points(NR, "23"), knob_points(NR, "25"), "no resets, T = 23", "no resets, T = 25",
                                                  f"C   Same experiment, two lengths (r = {R['knob_r_T23_vs_T25_noresets'][0]:.2f})"))):
        ax = fig.add_subplot(gs[1, i])
        ax.plot([0.55, 1.75], [0.55, 1.75], color=AXIS, lw=1)
        ax.axhline(1, color=GRID, lw=0.8); ax.axvline(1, color=GRID, lw=0.8)
        for n in B_:
            if n in A_:
                kind = n.split()[0]
                ax.errorbar(A_[n][0], B_[n][0], xerr=A_[n][1], yerr=B_[n][1], color=kind_col[kind], marker="o", ms=4.5, lw=0.8, capsize=0)
                if abs(np.log(A_[n][0])) > 0.25 or abs(np.log(B_[n][0])) > 0.25:
                    ax.annotate(n, (A_[n][0], B_[n][0]), textcoords="offset points", xytext=(4, 3), fontsize=6.5, color=INK2)
        ax.set_xscale("log"); ax.set_yscale("log")
        tk = [0.6, 0.8, 1.0, 1.25, 1.6]
        ax.set_xticks(tk); ax.set_xticklabels(tk); ax.set_yticks(tk); ax.set_yticklabels(tk); ax.minorticks_off()
        ax.set_xlim(0.55, 1.75); ax.set_ylim(0.55, 1.75)
        ax.set_xlabel(f"multiplier, {xl}"); ax.set_ylabel(f"multiplier, {yl}"); style(ax, title, grid=None)
        ax.legend(handles=[Line2D([], [], color=c, marker="o", ls="none", label=k) for k, c in kind_col.items()
                           if any(n.split()[0] == k for n in B_)], frameon=False, fontsize=7.5, loc="upper left")
    save(fig, "wr_ppo")

    # ---------------------------------------------------- 8. the ladder at T = 23
    fig, ax = plt.subplots(figsize=(10.5, 3.6))
    rows = [("our MWPM (hard readings)", "generic"), ("blend", "blend"), ("Riverlane best (published)", "rl_best"),
            ("blend + soft", "blend_soft"), ("blend + soft + PPO knobs", None)]
    val = {name: [100 * (D[name]["23"][k] if k else ppo[name]["23"]["both"]["tuned"]) for _, k in rows] for name in (NR, WR)}
    for i in range(len(rows)):
        a, b = val[NR][i], val[WR][i]
        ax.plot([b, a], [i, i], color=GRID, lw=3, zorder=1)
        ax.plot(a, i, "o", ms=9, mfc="white", mec=MUTED, mew=1.6, zorder=2, label="no resets" if i == 0 else None)
        ax.plot(b, i, "o", ms=9, color=BLUE, zorder=2, label="with resets" if i == 0 else None)
        ax.text(a + 0.18, i, f"{a:.2f} %", va="center", fontsize=8, color=MUTED)
        ax.text(b - 0.18, i, f"{b:.2f} %", va="center", ha="right", fontsize=8, color=BLUE)
        ax.text((a + b) / 2, i - 0.28, f"{a - b:.2f} points", ha="center", fontsize=7, color=INK2)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows]); ax.invert_yaxis()
    ax.set_xlim(12.5, 23.5); ax.set_ylim(len(rows) - 0.5, -0.7); ax.set_xlabel("logical error rate (%), every shot graded once")
    style(ax, "The ladder at equal length (T = 23, circuit_22 of each experiment)", grid="x")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    plt.tight_layout(); save(fig, "wr_ladder")
    print("figures written to compare_figs/")


if __name__ == "__main__":
    draw(json.load(open(core.out("compare_resets.json"))))
