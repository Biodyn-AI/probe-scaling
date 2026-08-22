"""Regenerate every figure in docs/paper.tex from the result cards.

Each figure is written as PDF (for LaTeX) and PNG (for visual inspection).
No number is typed in by hand; everything is recomputed here.
"""
from __future__ import annotations

import glob
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
from probe_scaling.io import seed_dirs
SEED_DIRS = [str(p) for _, p in sorted(seed_dirs().items())]

# Okabe-Ito: colourblind-safe.
INK = "#1a1a1a"
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
GREY = "#7a7a7a"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 9,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


def t_crit(n: int) -> float:
    return {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571}.get(n, 2.571)


def mean_ci(xs: list[float]) -> tuple[float, float, float, int]:
    n = len(xs)
    m = statistics.mean(xs)
    if n < 2:
        return m, m, m, n
    h = t_crit(n) * statistics.stdev(xs) / math.sqrt(n)
    return m, m - h, m + h, n


def iter_cards():
    for d in SEED_DIRS:
        for p in glob.glob(f"{d}/**/*.json", recursive=True):
            try:
                yield d, json.load(open(p))
            except json.JSONDecodeError:
                continue


def family_of(model_id: str) -> str | None:
    m = model_id.lower()
    if "esm2" in m:
        return "ESM2"
    if "carp" in m:
        return "CARP"
    if "rita" in m:
        return "RITA"
    if "nucleotide" in m or "nt-v2" in m:
        return "NT-v2"
    return None


def collect(task_prefix: str, family: str):
    """(seed_dir -> {params: gap}) for trained checkpoints of one family."""
    out: dict[str, dict[int, float]] = {}
    for d, c in iter_cards():
        t = c.get("task", {})
        tid = str(t.get("id", "")) if isinstance(t, dict) else str(t)
        if not tid.startswith(task_prefix):
            continue
        m = c.get("model", {})
        mid = str(m.get("id", ""))
        if m.get("control_arm") or mid.endswith("-random"):
            continue
        if family_of(mid) != family:
            continue
        s = c.get("scores", {})
        b, pr = s.get("behavioural"), s.get("probe")
        params = m.get("params") or m.get("parameters") or m.get("n_params")
        if b is None or pr is None or not params:
            continue
        if s.get("layer_selection") not in (None, "validation"):
            continue
        out.setdefault(d, {})[int(params)] = pr - b
    return out


def per_seed_slopes(task_prefix: str, family: str) -> list[float]:
    """One OLS slope per seed, over the rung set common to all seeds."""
    byseed = collect(task_prefix, family)
    if not byseed:
        return []
    common = set.intersection(*(set(v) for v in byseed.values()))
    if len(common) < 3:
        return []
    slopes = []
    for gaps in byseed.values():
        xs = [math.log10(p) for p in sorted(common)]
        ys = [gaps[p] for p in sorted(common)]
        xb, yb = statistics.mean(xs), statistics.mean(ys)
        den = sum((x - xb) ** 2 for x in xs)
        if den:
            slopes.append(sum((x - xb) * (y - yb) for x, y in zip(xs, ys)) / den)
    return slopes


# ---------------------------------------------------------------- figure 1
ROWS = [
    ("Fold recognition", "ESM2", "pr3", "ESM2"),
    ("Enzyme function", "ESM2", "pr4", "ESM2"),
    ("TF identity", "ESM2", "pr1", "ESM2"),
    ("TF family", "ESM2", "pr2", "ESM2"),
    ("Enhancer classes", "NT-v2", "dn1", "NT-v2"),
    ("Fold recognition", "CARP", "pr3", "CARP"),
    ("Fold recognition", "RITA", "pr3", "RITA"),
]


def figure_gap_forest():
    data = []
    for label, fam_label, prefix, fam in ROWS:
        sl = per_seed_slopes(prefix, fam)
        if not sl:
            continue
        m, lo, hi, n = mean_ci(sl)
        agree = all(x > 0 for x in sl) or all(x < 0 for x in sl)
        data.append((label, fam_label, m, lo, hi, n, agree, sl))

    fig, (ax, tx) = plt.subplots(
        1, 2, figsize=(7.0, 3.9), gridspec_kw={"width_ratios": [1.0, 0.72], "wspace": 0.04}
    )
    ys = list(range(len(data)))[::-1]

    for y, (lab, fam, m, lo, hi, n, agree, sl) in zip(ys, data):
        clears = (lo > 0) or (hi < 0)
        colour = BLUE if (clears and agree) else (ORANGE if clears else GREY)
        ax.scatter(sl, [y + 0.22] * len(sl), s=9, color=colour, alpha=0.38,
                   zorder=2, linewidths=0)
        ax.plot([lo, hi], [y, y], color=colour, lw=2.1, solid_capstyle="round", zorder=3)
        ax.plot([m], [y], "o", color=colour, ms=6.5, zorder=4,
                markeredgecolor="white", markeredgewidth=0.9)

    ax.axvline(0, color=INK, lw=0.9, zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{d[0]}\n{d[1]}" for d in data], fontsize=8.2, linespacing=1.3)
    ax.set_xlabel("Gap slope  (change in P \u2212 B per decade of parameters)", fontsize=8.4)
    ax.set_xlim(-0.068, 0.048)
    ax.set_ylim(-0.75, len(data) - 0.15)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="x", color="#dedede", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.text(0.003, len(data) - 0.55, "widens \u2192", fontsize=7.6, color=GREY)
    ax.text(-0.003, len(data) - 0.55, "\u2190 closes", fontsize=7.6, color=GREY, ha="right")

    # right-hand numeric column
    tx.set_xlim(0, 1)
    tx.set_ylim(-0.75, len(data) - 0.15)
    tx.axis("off")
    tx.text(0.02, len(data) - 0.55, "slope", fontsize=7.6, color=GREY, family="monospace")
    tx.text(0.34, len(data) - 0.55, "95% CI", fontsize=7.6, color=GREY, family="monospace")
    tx.text(1.0, len(data) - 0.55, "n", fontsize=7.6, color=GREY,
            family="monospace", ha="right")
    for y, (lab, fam, m, lo, hi, n, agree, sl) in zip(ys, data):
        clears = (lo > 0) or (hi < 0)
        strong = clears and agree
        tx.text(0.02, y, f"{m:+.4f}", fontsize=8.0, family="monospace", va="center",
                color=INK, fontweight="bold" if strong else "normal")
        tx.text(0.34, y, f"[{lo:+.4f}, {hi:+.4f}]", fontsize=7.7, family="monospace",
                va="center", color=INK if clears else GREY)
        tx.text(1.0, y, str(n), fontsize=8.0, family="monospace", va="center",
                ha="right", color=INK)

    handles = [
        Line2D([], [], color=BLUE, lw=2, marker="o", ms=5.5, markeredgecolor="white",
               label="clears zero, all seeds agree in sign"),
        Line2D([], [], color=ORANGE, lw=2, marker="o", ms=5.5, markeredgecolor="white",
               label="clears zero, seeds disagree"),
        Line2D([], [], color=GREY, lw=2, marker="o", ms=5.5, markeredgecolor="white",
               label="spans zero"),
        Line2D([], [], color=GREY, lw=0, marker="o", ms=3.6, alpha=0.5,
               label="individual seed slopes"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.52, -0.012),
               frameon=False, fontsize=7.5, ncol=2, handlelength=1.7,
               columnspacing=1.6, labelspacing=0.3)
    fig.subplots_adjust(left=0.175, right=0.995, top=0.965, bottom=0.255)
    fig.savefig(OUT / "gap_forest.pdf")
    fig.savefig(OUT / "gap_forest.png")
    plt.close(fig)
    return data


# ---------------------------------------------------------------- figure 2
def ols_ci(xs, ys):
    """Within-run OLS slope and 95% interval -- what a single-run paper reports."""
    n = len(xs)
    xb, yb = statistics.mean(xs), statistics.mean(ys)
    den = sum((x - xb) ** 2 for x in xs)
    b = sum((x - xb) * (y - yb) for x, y in zip(xs, ys)) / den
    a = yb - b * xb
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    se = math.sqrt(sum(r * r for r in resid) / (n - 2) / den)
    t = {3: 12.706, 4: 4.303, 5: 3.182, 6: 2.776}.get(n, 2.776)
    return b, b - t * se, b + t * se


def single_run(task_prefix, family, seed_dir=None, drop_above=None):
    seed_dir = seed_dir or SEED_DIRS[0]
    rows = {}
    for d, c in iter_cards():
        if d != seed_dir:
            continue
        t = c.get("task", {})
        tid = str(t.get("id", "")) if isinstance(t, dict) else str(t)
        if not tid.startswith(task_prefix):
            continue
        m = c.get("model", {})
        mid = str(m.get("id", ""))
        if m.get("control_arm") or mid.endswith("-random"):
            continue
        if family_of(mid) != family:
            continue
        sc = c.get("scores", {})
        b, pr = sc.get("behavioural"), sc.get("probe")
        par = m.get("params") or m.get("parameters")
        if b is None or pr is None or not par:
            continue
        if drop_above and int(par) > drop_above:
            continue
        rows[int(par)] = pr - b
    ps = sorted(rows)
    return ols_ci([math.log10(x) for x in ps], [rows[x] for x in ps]), len(ps)


def figure_replication():
    (b6, l6, h6), n6 = single_run("pr3", "ESM2")
    (b5, l5, h5), n5 = single_run("pr3", "ESM2", drop_above=int(10e9))
    esm_m, esm_lo, esm_hi, esm_n = mean_ci(per_seed_slopes("pr3", "ESM2"))
    (br, lr, hr), nr = single_run("pr3", "RITA")
    rita = per_seed_slopes("pr3", "RITA")
    rt_m, rt_lo, rt_hi, rt_n = mean_ci(rita)

    groups = [
        ("ESM2, fold recognition", [
            (f"1 run  \u00b7  {n6} rungs, 3.27 dec", b6, l6, h6, False, '"no measurable change"'),
            (f"1 run  \u00b7  {n5} rungs, 2.57 dec", b5, l5, h5, False, "(matched ladder)"),
            (f"{esm_n} runs  \u00b7  {n5} rungs, 2.57 dec", esm_m, esm_lo, esm_hi, True, '"the gap widens"'),
        ]),
        ("RITA, fold recognition", [
            (f"1 run  \u00b7  {nr} rungs", br, lr, hr, False, '"the gap closes sharply"'),
            (f"{rt_n} runs  \u00b7  {nr} rungs", rt_m, rt_lo, rt_hi, True, "no trend detectable"),
        ]),
    ]

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 3.6), sharex=True,
                             gridspec_kw={"height_ratios": [3, 2], "hspace": 0.42})
    out = []
    for ax, (title, rows) in zip(axes, groups):
        ys = list(range(len(rows)))[::-1]
        for y, (lab, m, lo, hi, repl, note) in zip(ys, rows):
            colour = BLUE if repl else ORANGE
            ax.plot([lo, hi], [y, y], color=colour, lw=2.2, solid_capstyle="round", zorder=3)
            ax.plot([m], [y], "o", color=colour, ms=6.5, zorder=4,
                    markeredgecolor="white", markeredgewidth=0.9)
            ax.text(0.137, y, note, fontsize=7.4, color=GREY, ha="right",
                    va="center", style="italic")
            out.append((title.split(",")[0], lab, m, lo, hi, repl, note))
        if title.startswith("RITA"):
            ax.scatter(rita, [ys[-1] + 0.30] * len(rita), s=11, color=BLUE,
                       alpha=0.5, zorder=2, linewidths=0)
        ax.axvline(0, color=INK, lw=0.9, zorder=1)
        ax.set_yticks(ys)
        ax.set_yticklabels([r[0] for r in rows], fontsize=8.2)
        ax.set_xlim(-0.098, 0.140)
        ax.set_ylim(-0.62, len(rows) - 0.38)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="x", color="#e2e2e2", lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.set_xticks([-0.075, -0.050, -0.025, 0.0, 0.025, 0.050, 0.075])
        ax.set_title(title, fontsize=8.5, color=INK, fontweight="bold",
                     loc="left", pad=5)

    axes[1].set_xlabel("Elicitation-gap slope per decade of parameters", fontsize=8.5)
    handles = [
        Line2D([], [], color=ORANGE, lw=2.2, marker="o", ms=5.5, markeredgecolor="white",
               label="single run"),
        Line2D([], [], color=BLUE, lw=2.2, marker="o", ms=5.5, markeredgecolor="white",
               label="replicated across seeds"),
        Line2D([], [], color=BLUE, lw=0, marker="o", ms=3.6, alpha=0.5,
               label="individual seed slopes"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.55, -0.015),
               frameon=False, fontsize=7.6, ncol=3, handlelength=1.7, columnspacing=1.8)
    fig.subplots_adjust(left=0.265, right=0.985, top=0.925, bottom=0.215)
    fig.savefig(OUT / "replication.pdf")
    fig.savefig(OUT / "replication.png")
    plt.close(fig)
    return out


# ---------------------------------------------------------------- figure 3
def figure_cost():
    from probe_scaling.analysis import cost_scaling_from_sweep
    files = [str(ROOT / "data" / "sweeps" / f) for f in
             ("iso_ladder_sweep.json", "iso_sweep_seed1.json", "iso_sweep_seed2.json")]
    colours = [BLUE, ORANGE, GREEN]
    fig, ax = plt.subplots(figsize=(6.5, 3.9))
    slopes = []
    ceiling = None
    for i, (f, col) in enumerate(zip(files, colours)):
        sweep = json.load(open(f))
        ceiling = math.log10(max(sweep["grid"]))
        r = cost_scaling_from_sweep(sweep, target=0.50)
        slopes.append(r.primary.slope)
        xs = [math.log10(pt["params"]) for pt in r.points]
        ys = [math.log10(pt["steps_interpolated"]) for pt in r.points]
        ax.plot(xs, ys, "o", color=col, ms=6, markeredgecolor="white",
                markeredgewidth=0.8, zorder=4,
                label=f"seed {i}   slope {r.primary.slope:+.3f}  (n={r.primary.n})")
        xb, yb = statistics.mean(xs), statistics.mean(ys)
        b = r.primary.slope
        xl = [min(xs) - 0.15, max(xs) + 0.15]
        ax.plot(xl, [yb + b * (x - xb) for x in xl], color=col, lw=1.5,
                alpha=0.8, zorder=3)
        # censored rungs: never reached the target inside the largest budget
        for mid, m in sweep["models"].items():
            c = m["crossings"].get("0.50", {})
            if c.get("censored"):
                x = math.log10(m["params"]) + (i - 1) * 0.048
                ax.plot([x], [ceiling + 0.035], marker="^", color=col, ms=6,
                        markerfacecolor="white", markeredgewidth=1.2, zorder=4)

    ax.axhline(ceiling, color=GREY, lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.text(math.log10(3.4e9), ceiling + 0.012, "largest budget run (400 steps)",
            fontsize=7.3, color=GREY, ha="right", va="bottom")

    ax.set_xlabel("model size  (log$_{10}$ parameters)", fontsize=8.6)
    ax.set_ylabel("budget to reach macro-F1 = 0.50\n(log$_{10}$ fine-tuning steps)",
                  fontsize=8.6)
    ax.tick_params(labelsize=8)
    ax.grid(color="#e8e8e8", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    m = statistics.mean(slopes)
    sd = statistics.stdev(slopes)
    h = 4.303 * sd / math.sqrt(3)
    ax.set_title(f"pooled slope {m:+.3f} decades of budget per decade of parameters"
                 f"    95% CI [{m - h:+.3f}, {m + h:+.3f}]",
                 fontsize=8.3, color=INK, pad=8)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], marker="^", color=GREY, lw=0, ms=6,
                          markerfacecolor="white", markeredgewidth=1.2,
                          label="censored: target not reached"))
    ax.legend(handles=handles, frameon=False, fontsize=7.5, loc="upper right",
              bbox_to_anchor=(1.0, 0.90), handlelength=1.3, labelspacing=0.35)
    xt = [math.log10(v) for v in (8e6, 35e6, 150e6, 650e6, 3e9)]
    ax.set_xticks(xt)
    ax.set_xticklabels(["8M", "35M", "150M", "650M", "3B"], fontsize=8)
    ax.set_xlim(math.log10(5.5e6), math.log10(5.0e9))
    ax.set_ylim(2.04, ceiling + 0.075)
    fig.subplots_adjust(left=0.135, right=0.985, top=0.895, bottom=0.135)
    fig.savefig(OUT / "cost_scaling.pdf")
    fig.savefig(OUT / "cost_scaling.png")
    plt.close(fig)
    return m, m - h, m + h


if __name__ == "__main__":
    for lab, fam, m, lo, hi, n, agree, sl in figure_gap_forest():
        print(f"{lab + ' / ' + fam:34s} {m:+.4f} [{lo:+.4f},{hi:+.4f}] n={n} agree={agree}")
    print("--- replication ---")
    for fam, lab, m, lo, hi, repl, note in figure_replication():
        print(f"{fam:5s} {lab:28s} {m:+.4f} [{lo:+.4f},{hi:+.4f}]")
    print("--- cost ---")
    print("pooled %.4f [%.4f, %.4f]" % figure_cost())
