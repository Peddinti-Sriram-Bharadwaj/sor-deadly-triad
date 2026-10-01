"""Runs the full grid and writes results/results.jsonl and results/summary.md.

    python run.py                     # about 10-15 minutes on 7 cores
    python run.py --steps 2000 --seeds 1 --workers 2   # quick check
"""

import argparse
import itertools
import json
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

from sor_triad import PROBLEMS, expected_run, neural_run

WS = [0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 2.0]
EXPECTED_STEP_SIZES = [0.01, 0.1, 0.5]
NEURAL = [("sgd", 1e-3), ("sgd", 1e-2), ("adam", 1e-4), ("adam", 1e-3)]
HERE = os.path.dirname(os.path.abspath(__file__))


def _neural(args):
    problem, w, optimizer, lr, seed, steps = args
    r = neural_run(PROBLEMS[problem](), w, optimizer, lr, steps, seed)
    return {"problem": problem, "representation": "neural", "optimizer": optimizer, "lr": lr, "w": w,
            "seed": seed, **r}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=os.cpu_count() - 1)
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = [{"problem": p, "representation": rep, "optimizer": "expected", "lr": a, "w": w, "seed": 0,
             **expected_run(PROBLEMS[p](), rep, w, a, args.steps)}
            for p, rep, a, w in itertools.product(PROBLEMS, ("tabular", "linear"), EXPECTED_STEP_SIZES, WS)]
    tasks = [(p, w, o, lr, s, args.steps) for p in PROBLEMS for o, lr in NEURAL for w in WS
             for s in range(args.seeds)]
    with Pool(args.workers) as pool:
        rows += pool.map(_neural, tasks)

    with open(os.path.join(args.out, "results.jsonl"), "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    md = summarise(pd.DataFrame(rows))
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write(md)
    print(md)


def _cell(g):
    if len(g) == 1:
        return "DIV" if g.diverged.iloc[0] else f"{g.max_abs_q.iloc[0]:.1e}"
    ok = g.max_abs_q[~g.diverged]
    med = f"{np.median(ok):.1e}" if len(ok) else "-"
    return f"{med} ({int(g.diverged.sum())}/{len(g)} div)" if g.diverged.any() else med


def summarise(df):
    """One Markdown table per problem: rows are representation x optimizer x step size, columns w.
    Cells give max |Q| after training (median over seeds for the neural learner) or DIV."""
    out = []
    for problem, gp in df.groupby("problem", sort=False):
        out.append(f"### {problem}\n")
        out.append("| representation | update | step size | " + " | ".join(f"w={w}" for w in WS) + " |")
        out.append("|---|---|---|" + "---|" * len(WS))
        for (rep, opt, lr), g in gp.groupby(["representation", "optimizer", "lr"], sort=False):
            cells = [_cell(g[g.w == w]) for w in WS]
            out.append(f"| {rep} | {opt} | {lr:g} | " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    main()
