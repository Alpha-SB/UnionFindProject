"""
experiments.py

Run empirical experiments comparing ListUF and TreeUF.

Generates a CSV file with rows:
structure, use_rank, path_compression, n, m, workload, trial, runtime, pointer_updates
"""

from __future__ import annotations

import argparse
import csv
import random
import time

from list_uf import ListUF
from tree_uf import TreeUF


# -------------------- workload generators -------------------- #

def workload_random(uf, n: int, m: int, union_ratio: float = 0.5):
    """
    Random workload: mix of unions and finds.

    :param union_ratio: fraction of operations that are unions (0 <= r <= 1).
    """
    num_unions = int(union_ratio * m)
    num_finds = m - num_unions

    pairs = [(random.randrange(n), random.randrange(n)) for _ in range(num_unions)]
    queries = [random.randrange(n) for _ in range(num_finds)]

    start = time.perf_counter()
    for x, y in pairs:
        uf.union(x, y)
    for q in queries:
        uf.find(q)
    end = time.perf_counter()

    return end - start


def workload_adversarial(uf, n: int, m: int):
    """
    Adversarial workload:
    - First, create a long chain (or path) using unions.
    - Then, perform m finds on random nodes.
    """
    start = time.perf_counter()
    # chain unions
    for i in range(n - 1):
        uf.union(i, i + 1)
    # random finds
    for _ in range(m):
        q = random.randrange(n)
        uf.find(q)
    end = time.perf_counter()
    return end - start


def workload_batched_finds(uf, n: int, m: int):
    """
    Batched workload:
    - m unions
    - then m finds
    """
    pairs = [(random.randrange(n), random.randrange(n)) for _ in range(m)]
    queries = [random.randrange(n) for _ in range(m)]

    start = time.perf_counter()
    for x, y in pairs:
        uf.union(x, y)
    for q in queries:
        uf.find(q)
    end = time.perf_counter()
    return end - start


def workload_gnp(uf, n: int, m: int):
    """
    Approximate G(n,p) connectivity workload:
    - Instead of trying all O(n^2) pairs, generate m random edges.
    - union endpoints of those edges
    - then run m random queries

    This is an approximation of working with a sparse G(n, p) graph.
    """
    edges = [(random.randrange(n), random.randrange(n)) for _ in range(m)]
    queries = [random.randrange(n) for _ in range(m)]

    start = time.perf_counter()
    for u, v in edges:
        uf.union(u, v)
    for q in queries:
        uf.find(q)
    end = time.perf_counter()
    return end - start


# Map workload names to functions
WORKLOAD_FUNCS = {
    "random_50_50": lambda uf, n, m: workload_random(uf, n, m, union_ratio=0.5),
    "random_20_80": lambda uf, n, m: workload_random(uf, n, m, union_ratio=0.2),
    "adversarial": workload_adversarial,
    "batched_finds": workload_batched_finds,
    "gnp": workload_gnp,
}


def run_single_experiment(structure: str,
                          n: int,
                          m: int,
                          workload: str,
                          use_rank: bool,
                          path_compression: bool) -> tuple[float, int]:
    """
    Build the requested UF structure and run the chosen workload.

    Returns (runtime, pointer_updates).
    """
    if structure == "list":
        uf = ListUF(n)
    elif structure == "tree":
        uf = TreeUF(n, use_rank=use_rank, path_compression=path_compression)
    else:
        raise ValueError(f"Unknown structure: {structure}")

    func = WORKLOAD_FUNCS[workload]
    runtime = func(uf, n, m)

    # pointer_updates attribute name differs by class; unify it:
    pointer_updates = getattr(uf, "head_updates", 0) + getattr(uf, "parent_updates", 0)

    return runtime, pointer_updates


def run_all_experiments(args):
    random.seed(args.seed)

    start_time = time.perf_counter()
    ns = args.ns
    m_multipliers = args.ms
    workloads = args.workloads
    trials = args.trials

    rows = []

    for n in ns:
        for mul in m_multipliers:
            m = mul * n
            for workload in workloads:
                # list-based
                for trial in range(1, trials + 1):
                    runtime, ptrs = run_single_experiment(
                        structure="list",
                        n=n,
                        m=m,
                        workload=workload,
                        use_rank=False,
                        path_compression=False,  # not used here
                    )
                    rows.append([
                        "list", False, False, n, m, workload, trial, runtime, ptrs
                    ])
                    print(f"[list] n={n}, m={m}, workload={workload}, "
                          f"trial={trial}, time={runtime:.4f}s, ptrs={ptrs}")

                # tree-based: test both with and without rank; with path compression
                for use_rank in (False, True):
                    for trial in range(1, trials + 1):
                        runtime, ptrs = run_single_experiment(
                            structure="tree",
                            n=n,
                            m=m,
                            workload=workload,
                            use_rank=use_rank,
                            path_compression=True,
                        )
                        rows.append([
                            "tree", use_rank, True, n, m, workload, trial, runtime, ptrs
                        ])
                        label = "rank" if use_rank else "size"
                        print(f"[tree-{label}] n={n}, m={m}, workload={workload}, "
                              f"trial={trial}, time={runtime:.4f}s, ptrs={ptrs}")

    # write CSV
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "structure",
            "use_rank",
            "path_compression",
            "n",
            "m",
            "workload",
            "trial",
            "runtime",
            "pointer_updates",
        ])
        writer.writerows(rows)

    total_experiments = len(rows)
    total_time = time.perf_counter() - start_time

    print(f"\nSaved results to {args.output}")
    print(f"Total experiments: {total_experiments}")
    print(f"Total time: {total_time:.2f}s")

    return total_experiments, total_time


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Union-Find experiments and output CSV results."
    )
    parser.add_argument(
        "--ns",
        type=int,
        nargs="+",
        default=[10_000, 50_000, 100_000],
        help="List of n values to test (default: 10000 50000 100000)",
    )
    parser.add_argument(
        "--ms",
        type=int,
        nargs="+",
        default=[1, 5, 10],
        help="List of m multipliers; m = multiplier * n (default: 1 5 10)",
    )
    parser.add_argument(
        "--workloads",
        type=str,
        nargs="+",
        default=["random_50_50", "adversarial", "batched_finds", "gnp"],
        choices=list(WORKLOAD_FUNCS.keys()),
        help="Workloads to run.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=3,
        help="Number of trials per configuration (default: 3)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results.csv",
        help="Output CSV file (default: results.csv)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_all_experiments(args)
