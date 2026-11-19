"""Run the cross-morphology sweep: modes x seeds, one JSON per run."""
import itertools, json, os, sys
from multiprocessing import Pool
from pathlib import Path

import xmorph

OUT = Path(__file__).parent / "results_xmorph"
MODES = ["indep", "mutual", "better", "better_q"]


def run(job):
    mode, seed = job
    out = OUT / f"{mode}__s{seed}.json"
    if out.exists():
        return
    log = xmorph.train(xmorph.parse(["--mode", mode, "--seed", str(seed)]))
    out.write_text(json.dumps(log))
    print("DONE", mode, seed, log["cross"], flush=True)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    seeds = range(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
    jobs = [(m, s) for s, m in itertools.product(seeds, MODES)]
    with Pool(os.cpu_count()) as p:
        list(p.imap_unordered(run, jobs))
