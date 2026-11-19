"""Run the cross-morphology sweep: modes x seeds, one JSON per run."""
import json, os, sys
from multiprocessing import Pool
from pathlib import Path

import xmorph

OUT = Path(__file__).parent / "results_xmorph"
# keypoint space (different joint structures) is the main experiment;
# joint space (same joint tree) is a smaller control
PLAN = {"keypoint": (["indep", "mutual", "better", "better_q"], 4),
        "joint": (["indep", "better"], 3)}


def run(job):
    space, mode, seed = job
    out = OUT / f"{space}__{mode}__s{seed}.json"
    if out.exists():
        return
    log = xmorph.train(xmorph.parse(["--space", space, "--mode", mode, "--seed", str(seed)]))
    out.write_text(json.dumps(log))
    print("DONE", space, mode, seed, log["cross"], flush=True)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    spaces = sys.argv[1].split(",") if len(sys.argv) > 1 else list(PLAN)
    jobs = [(sp, m, s) for sp in spaces for s in range(PLAN[sp][1]) for m in PLAN[sp][0]]
    with Pool(os.cpu_count()) as p:
        list(p.imap_unordered(run, jobs))
