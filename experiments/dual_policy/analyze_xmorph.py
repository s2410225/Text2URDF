"""Aggregate results_xmorph/*.json into results_xmorph.md and figs/xmorph_*.png."""
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
MODES = ["indep", "mutual", "better", "better_q"]
COL = {"indep": "#222222", "mutual": "#1f77b4", "better": "#d62728", "better_q": "#9467bd"}


def ms(a):
    a = np.asarray(a, float)
    se = a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0
    return f"{a.mean():.1f} ± {se:.1f}"


lines = []
for sp in ["keypoint", "joint"]:
    lines += [f"\n## {sp} space\n",
              "| mode | n | radial eval | radial dx [m] | bilateral eval | bilateral dx [m] | radial teaches | bilateral policy on radial body dx |",
              "|---|---|---|---|---|---|---|---|"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for m in MODES:
        L = [json.load(open(f)) for f in sorted(glob.glob(str(HERE / f"results_xmorph/{sp}__{m}__s*.json")))]
        if not L:
            continue
        t = np.array([l["teacher"] for l in L])
        lines.append(
            f"| {m} | {len(L)} | {ms([l['eval_radial'][-1] for l in L])} | {ms([l['dx_radial'][-1] for l in L])} "
            f"| {ms([l['eval_bilateral'][-1] for l in L])} | {ms([l['dx_bilateral'][-1] for l in L])} "
            f"| {np.mean(t == 0):.0%} | {ms([l['cross']['bilateral_policy_on_radial'][1] for l in L])} |")
        for ax, b in zip(axes, ["radial", "bilateral"]):
            x = np.array(L[0]["eval_steps"])
            Y = np.array([l[f"dx_{b}"] for l in L])
            ax.plot(x, Y.mean(0), color=COL[m], label=m, lw=2)
            if len(Y) > 1:
                se = Y.std(0, ddof=1) / np.sqrt(len(Y))
                ax.fill_between(x, Y.mean(0) - se, Y.mean(0) + se, color=COL[m], alpha=0.15)
    for ax, b in zip(axes, ["radial", "bilateral"]):
        ax.set_title(f"{sp} space: {b} body"); ax.set_xlabel("env steps (per body)")
        ax.set_ylabel("forward distance per episode [m]"); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(HERE / f"figs/xmorph_{sp}.png", dpi=130)
(HERE / "results_xmorph.md").write_text("# Cross-morphology results (mean ± s.e. over seeds, greedy eval at the end)\n" + "\n".join(lines) + "\n")
print("\n".join(lines))
