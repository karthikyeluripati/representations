"""Paper figures (static, light mode) from the verified result archives.
  Fig 1  the gap: share of uncertain-outcome episodes whose next action is a blind retry, untrained policies
  Fig 2  transfer + scaling: held-out TravelAPI, implicit timeout, per-seed points and means
  Fig 3  tau2 diagnosis: checks before re-booking vs re-cancelling
Palette: reference categorical slots 1-3 (validated all-pairs, light mode); neutral grey for reference lines.
Usage: python scripts/make_figures.py --uo DIR --out paper/figures
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "savefig.facecolor": SURF, "font.family": "DejaVu Sans"})


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig1(uo, out):
    pols = [("Qwen3-4B (base)", "base4b"), ("Qwen3-8B", "q8b"), ("Qwen3-30B-A3B", "q30b_a3b"), ("π₀ corrective SFT", "pi0"), ("B verified DPO", "B_s0")]
    rows = []
    for name, f in pols:
        s = json.load(open(os.path.join(uo, f + ".summary.json")))
        for pr, lab in (("std", ""), ("check", " + check prompt")):
            if pr not in s:
                continue
            tot = ret = 0
            for w in ("explicit", "para", "implicit"):
                ae = s[pr][f"ACa_{w}"]["after_error"]
                n = sum(v for k, v in ae.items() if k != "no_error_seen")
                tot += n; ret += ae.get("retry", 0)
            rows.append((name + lab, ret / tot))
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    y = range(len(rows))[::-1]
    ax.barh(list(y), [r[1] for r in rows], height=0.6, color=S1)
    for yi, (n, v) in zip(y, rows):
        ax.text(v + 0.01, yi, f"{v:.0%}", va="center", color=INK, fontsize=8)
    ax.set_yticks(list(y)); ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlim(0, 1.08); ax.set_xlabel("next action is a blind retry (ACa, all wordings)")
    ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_title("Untrained agents re-issue the uncertain call", loc="left", color=INK, fontsize=10)
    save(fig, out, "fig1_blind_retry")


def fig2(out):
    arms = [("control\n(retry convention)", [0.514, 0.514, 0.500]), ("32 synthetic\nclasses", [0.700, 0.600, 0.529]),
            ("64 synthetic\nclasses", [0.729, 0.771, 0.500])]
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for i, (n, v) in enumerate(arms):
        m = sum(v) / 3
        ax.bar(i, m, width=0.55, color=S1 if i else "#b9b8b2")
        ax.scatter([i - 0.12, i, i + 0.12], v, s=22, color=INK, zorder=3, edgecolor=SURF, linewidth=1.5)
        ax.text(i, m + 0.03, f"{m:.3f}", ha="center", color=INK, fontsize=8)
    ax.axhline(0.5, color=INK2, lw=1, ls="--"); ax.text(0.5, 0.493, "blind-retry\nlevel", color=INK2, fontsize=7, va="top", ha="center")
    ax.set_xticks(range(3)); ax.set_xticklabels([a[0] for a in arms]); ax.set_ylim(0.4, 0.9)
    ax.set_ylabel("success, held-out API class\n(implicit timeout)"); ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_title("More API-class diversity, more transfer (dots = seeds)", loc="left", color=INK, fontsize=10)
    save(fig, out, "fig2_transfer_scaling")


def fig3(out):
    arms = ["π_B", "control", "32 classes", "64 classes"]
    book = [0 / 12, 2 / 36, 10 / 36, 10 / 36]; cancel = [0 / 66, 0 / 198, 0 / 198, 0 / 198]
    fig, ax = plt.subplots(figsize=(4.6, 3.0)); w = 0.36
    xs = range(len(arms))
    ax.bar([x - w / 2 - 0.01 for x in xs], book, width=w, color=S1, label="before re-booking (create-type)")
    ax.bar([x + w / 2 + 0.01 for x in xs], cancel, width=w, color=S2, label="before re-cancelling (refunds again)")
    for x, b, c in zip(xs, book, cancel):
        ax.text(x - w / 2, b + 0.01, f"{b:.0%}", ha="center", fontsize=7, color=INK)
        ax.text(x + w / 2, c + 0.01, f"{c:.0%}", ha="center", fontsize=7, color=INK)
    ax.set_xticks(list(xs)); ax.set_xticklabels(arms); ax.set_ylim(0, 0.45)
    ax.set_ylabel("checks right after the error\n(τ²-bench airline, zero-shot)"); ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.set_title("The learned rule follows the verb, not the side effect", loc="left", color=INK, fontsize=10)
    save(fig, out, "fig3_tau2_by_function")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--uo", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    fig1(a.uo, a.out); fig2(a.out); fig3(a.out); print("figures written to", a.out)
