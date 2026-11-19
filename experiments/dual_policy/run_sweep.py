"""Run every mode x env x seed in parallel and store one JSON log per run."""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import itertools
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import dpl

MODES = ["ppo", "ppo2", "mutual", "disagree", "pred_reg", "pred_gate", "pred_sil"]
ENVS = {
    "CartPole-v1": ["--total-steps", "150000", "--lr", "2.5e-4"],
    "Acrobot-v1": ["--total-steps", "200000", "--lr", "2.5e-4"],
    "LunarLander-v3": ["--total-steps", "1000000", "--lr", "3e-4", "--eval-every", "50"],
}
OUT = Path(__file__).parent / "results"


def run(job):
    env, mode, seed, extra = job
    out = OUT / f"{env}__{mode}__s{seed}.json"
    if out.exists():
        return str(out)
    cfg = dpl.parse(["--env", env, "--mode", mode, "--seed", str(seed), *ENVS[env], *extra])
    log = dpl.train(cfg)
    out.write_text(json.dumps(log))
    print(f"{env:15s} {mode:10s} s{seed}  eval {log['eval'][-1]:8.1f}  {log['time']:.0f}s", flush=True)
    return str(out)


if __name__ == "__main__":
    envs = sys.argv[1].split(",") if len(sys.argv) > 1 else list(ENVS)
    seeds = range(int(sys.argv[2]) if len(sys.argv) > 2 else 5)
    OUT.mkdir(exist_ok=True)
    jobs = [(e, m, s, []) for e, m, s in itertools.product(envs, MODES, seeds)]
    # longest jobs first so the pool stays busy
    jobs.sort(key=lambda j: -int(ENVS[j[0]][1]))
    with Pool(os.cpu_count()) as p:
        list(p.imap_unordered(run, jobs))
