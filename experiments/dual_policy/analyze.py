"""Aggregate results/*.json into a table (results.md) and figures (figs/)."""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
MODES = ["ppo", "ppo2", "mutual", "disagree", "pred_reg", "pred_gate", "pred_sil"]
ENVS = ["CartPole-v1", "Acrobot-v1", "LunarLander-v3"]
COLORS = dict(zip(MODES, ["#222222", "#999999", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]))

runs = defaultdict(list)
for f in sorted((HERE / "results").glob("*.json")):
    env, mode, _ = f.stem.split("__")
    runs[(env, mode)].append(json.loads(f.read_text()))


def interp(log, key="ret", grid=None):
    x, y = np.array(log["steps"]), np.array(log[key], dtype=float)
    ok = ~np.isnan(y)
    return np.interp(grid, x[ok], y[ok])


def mse(a):
    a = np.asarray(a, dtype=float)
    return a.mean(), (a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0)


lines = ["| env | mode | n | final greedy eval | train return (last 10%) | train AUC | predictor eval |",
         "|---|---|---|---|---|---|---|"]
summary = {}
for env in ENVS:
    for mode in MODES:
        L = runs.get((env, mode))
        if not L:
            continue
        fin = [l["eval"][-1] for l in L]
        n10 = [np.nanmean(l["ret"][int(0.9 * len(l["ret"])):]) for l in L]
        auc = [np.nanmean(l["ret"]) for l in L]
        pe = [l["eval_pred"][-1] for l in L if l["eval_pred"]]
        f, fs = mse(fin); t, ts = mse(n10); a, as_ = mse(auc)
        summary[(env, mode)] = (f, fs, t, ts, a, as_)
        ptxt = f"{mse(pe)[0]:.1f} ± {mse(pe)[1]:.1f}" if pe else "–"
        lines.append(f"| {env} | {mode} | {len(L)} | {f:.1f} ± {fs:.1f} | {t:.1f} ± {ts:.1f} | {a:.1f} ± {as_:.1f} | {ptxt} |")
(HERE / "results.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

figs = HERE / "figs"
figs.mkdir(exist_ok=True)
envs = [e for e in ENVS if any((e, m) in runs for m in MODES)]
fig, axes = plt.subplots(1, len(envs), figsize=(5.2 * len(envs), 4), squeeze=False)
for ax, env in zip(axes[0], envs):
    for mode in MODES:
        L = runs.get((env, mode))
        if not L:
            continue
        grid = np.linspace(L[0]["steps"][0], L[0]["steps"][-1], 100)
        Y = np.stack([interp(l, "ret", grid) for l in L])
        m, s = Y.mean(0), Y.std(0, ddof=1) / np.sqrt(len(Y)) if len(Y) > 1 else 0 * Y[0]
        lw = 2.2 if mode.startswith("pred") or mode == "ppo" else 1.4
        ax.plot(grid, m, color=COLORS[mode], label=mode, lw=lw)
        ax.fill_between(grid, m - s, m + s, color=COLORS[mode], alpha=0.15)
    ax.set_title(env); ax.set_xlabel("env steps"); ax.set_ylabel("training episode return")
    ax.grid(alpha=0.3)
axes[0][0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(figs / "learning_curves.png", dpi=130)

# prediction error / inter-policy divergence over training
fig, axes = plt.subplots(1, len(envs), figsize=(5.2 * len(envs), 3.6), squeeze=False)
for ax, env in zip(axes[0], envs):
    for mode in MODES:
        L = runs.get((env, mode))
        if not L:
            continue
        key = "pe" if mode.startswith("pred") else "kl12"
        if not L[0][key] or all(np.isnan(L[0][key])):
            continue
        Y = np.stack([np.array(l[key], dtype=float) for l in L])
        ax.plot(L[0]["steps"][: Y.shape[1]], np.nanmean(Y, 0), color=COLORS[mode], label=mode)
    ax.set_yscale("log"); ax.set_title(f"{env}: KL between the two policies")
    ax.set_xlabel("env steps"); ax.grid(alpha=0.3)
axes[0][0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(figs / "policy_divergence.png", dpi=130)
print("figures written to", figs)
