"""
plot_results.py

Plot results from experiments.csv.

Usage (basic):
    python plot_results.py

This will:
- Load results from results.csv
- Produce runtime and pointer-update plots for each workload.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict

# Matplotlib is optional; we can fall back to text-mode charts.
try:
    import matplotlib.pyplot as plt  # type: ignore
except ImportError:  # pragma: no cover - environment without matplotlib
    plt = None


def load_results(filename: str):
    rows = []
    with open(filename, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["use_rank"] = (r["use_rank"] == "True")
            r["path_compression"] = (r["path_compression"] == "True")
            r["n"] = int(r["n"])
            r["m"] = int(r["m"])
            r["trial"] = int(r["trial"])
            r["runtime"] = float(r["runtime"])
            r["pointer_updates"] = int(r["pointer_updates"])
            rows.append(r)
    return rows


def aggregate_by_mean(rows, group_keys, value_key):
    """
    Group rows by 'group_keys' and compute mean of 'value_key'.
    Returns dict: key_tuple -> mean_value.
    """
    sums = defaultdict(float)
    counts = defaultdict(int)
    for r in rows:
        key = tuple(r[k] for k in group_keys)
        sums[key] += r[value_key]
        counts[key] += 1
    means = {k: sums[k] / counts[k] for k in sums}
    return means


def plot_runtime_vs_n(rows, workload, structure_filter=None):
    """
    Plot mean runtime vs n for a given workload.

    structure_filter:
        - "list" for only ListUF
        - "tree-size" for tree use_rank=False
        - "tree-rank" for tree use_rank=True
        - None for all three
    """
    # group by (structure, use_rank, n)
    group_keys = ["structure", "use_rank", "n"]
    means = aggregate_by_mean(
        [r for r in rows if r["workload"] == workload],
        group_keys,
        "runtime",
    )

    plt.figure(figsize=(8, 5))

    def label_for(structure, use_rank):
        if structure == "list":
            return "ListUF"
        if not use_rank:
            return "TreeUF (union-by-size)"
        return "TreeUF (union-by-rank)"

    # structure_filter control
    def should_plot(structure, use_rank):
        if structure_filter is None:
            return True
        if structure_filter == "list":
            return structure == "list"
        if structure_filter == "tree-size":
            return structure == "tree" and (not use_rank)
        if structure_filter == "tree-rank":
            return structure == "tree" and use_rank
        return True

    series = defaultdict(list)
    for (structure, use_rank, n), mean_runtime in means.items():
        if not should_plot(structure, use_rank):
            continue
        series[(structure, use_rank)].append((n, mean_runtime))

    for (structure, use_rank), data in series.items():
        data.sort(key=lambda x: x[0])
        xs = [n for n, _ in data]
        ys = [t for _, t in data]
        plt.plot(xs, ys, marker="o", label=label_for(structure, use_rank))

    plt.title(f"Runtime vs n — workload={workload}")
    plt.xlabel("n (number of elements)")
    plt.ylabel("Runtime (seconds)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_pointer_updates_vs_n(rows, workload):
    """
    Plot mean pointer_updates vs n for list vs tree structures.
    """
    group_keys = ["structure", "use_rank", "n"]
    means = aggregate_by_mean(
        [r for r in rows if r["workload"] == workload],
        group_keys,
        "pointer_updates",
    )

    plt.figure(figsize=(8, 5))

    def label_for(structure, use_rank):
        if structure == "list":
            return "ListUF"
        if not use_rank:
            return "TreeUF (union-by-size)"
        return "TreeUF (union-by-rank)"

    series = defaultdict(list)
    for (structure, use_rank, n), mean_ptr in means.items():
        series[(structure, use_rank)].append((n, mean_ptr))

    for (structure, use_rank), data in series.items():
        data.sort(key=lambda x: x[0])
        xs = [n for n, _ in data]
        ys = [p for _, p in data]
        plt.plot(xs, ys, marker="o", label=label_for(structure, use_rank))

    plt.title(f"Pointer Updates vs n — workload={workload}")
    plt.xlabel("n (number of elements)")
    plt.ylabel("Pointer updates (mean)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def text_plot(series, title, y_label):
    """
    Render a simple text bar chart for a dict:
    series[label] = list[(x, y)] where x is n and y is value.
    """
    print(f"\n{title}")
    print(f"{y_label} (scaled bars)\n")

    # flatten to find max for scaling
    max_y = max((y for data in series.values() for _, y in data), default=0.0)
    if max_y == 0:
        print("No data to plot.")
        return

    bar_width = 50
    for label, data in series.items():
        print(label)
        data.sort(key=lambda t: t[0])
        for x, y in data:
            filled = max(1, int((y / max_y) * bar_width))
            bar = "#" * filled
            print(f" n={x:<8} {bar} {y:.4f}")
        print()


def text_runtime_vs_n(rows, workload, structure_filter=None):
    group_keys = ["structure", "use_rank", "n"]
    means = aggregate_by_mean(
        [r for r in rows if r["workload"] == workload],
        group_keys,
        "runtime",
    )

    def label_for(structure, use_rank):
        if structure == "list":
            return "ListUF"
        if not use_rank:
            return "TreeUF (union-by-size)"
        return "TreeUF (union-by-rank)"

    def should_plot(structure, use_rank):
        if structure_filter is None:
            return True
        if structure_filter == "list":
            return structure == "list"
        if structure_filter == "tree-size":
            return structure == "tree" and (not use_rank)
        if structure_filter == "tree-rank":
            return structure == "tree" and use_rank
        return True

    series = defaultdict(list)
    for (structure, use_rank, n), mean_runtime in means.items():
        if not should_plot(structure, use_rank):
            continue
        series[label_for(structure, use_rank)].append((n, mean_runtime))

    text_plot(series, f"Runtime vs n — workload={workload}", "Runtime (seconds)")


def text_pointer_updates_vs_n(rows, workload):
    group_keys = ["structure", "use_rank", "n"]
    means = aggregate_by_mean(
        [r for r in rows if r["workload"] == workload],
        group_keys,
        "pointer_updates",
    )

    def label_for(structure, use_rank):
        if structure == "list":
            return "ListUF"
        if not use_rank:
            return "TreeUF (union-by-size)"
        return "TreeUF (union-by-rank)"

    series = defaultdict(list)
    for (structure, use_rank, n), mean_ptr in means.items():
        series[label_for(structure, use_rank)].append((n, mean_ptr))

    text_plot(
        series,
        f"Pointer Updates vs n — workload={workload}",
        "Pointer updates (mean)",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot results from Union-Find experiments."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="results.csv",
        help="Input CSV file (default: results.csv)",
    )
    parser.add_argument(
        "--workload",
        type=str,
        default="random_50_50",
        help="Workload name to plot (must match experiments.csv)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["matplotlib", "text"],
        default="matplotlib",
        help="Use 'matplotlib' for real plots (if available) or 'text' for ASCII bars.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    rows = load_results(args.input)

    use_text = args.mode == "text" or plt is None

    if use_text:
        if plt is None and args.mode == "matplotlib":
            print("matplotlib not available; falling back to text mode.")
        text_runtime_vs_n(rows, args.workload)
        text_pointer_updates_vs_n(rows, args.workload)
    else:
        # Example: create two plots for the selected workload
        plot_runtime_vs_n(rows, args.workload)
        plot_pointer_updates_vs_n(rows, args.workload)
