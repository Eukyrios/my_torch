# Draws the figures of docs/hyperparameter_search.md from the CSV logs written by
# scripts/run_hyperparameter_benchmark.sh, and prints the summary table in markdown.
# Usage: python3 -m scripts.plot_hyperparameter_search [RESULTS_DIR] [IMAGES_DIR]

import csv
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHODS = {"pso": "Particle swarm", "ga": "Genetic algorithm"}
COLORS = {"pso": "#2a78d6", "ga": "#eb6834"}
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
LOSS_CAP = 2.0
CONVERGENCE_RANGE = (0.64, 0.9)


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT_SECONDARY, "axes.titlecolor": TEXT,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": TEXT,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "legend.frameon": False,
    })


def read_rows(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def best_so_far(scores):
    curve, best = [], float("inf")
    for score in scores:
        best = min(best, score)
        curve.append(best)
    return curve


def runs_of(results, method):
    runs = []
    for path in sorted(results.glob(f"{method}_seed*_evaluations.csv")):
        rows = read_rows(path)
        runs.append({"rows": rows, "curve": best_so_far([float(row["score"]) for row in rows])})
    return runs


def median_curve(curves):
    length = min(len(curve) for curve in curves)
    return [statistics.median(curve[i] for curve in curves) for i in range(length)]


def end_label_offsets(medians):
    finals = sorted((curve[-1], method) for method, curve in medians.items())
    return {method: -9 + 18 * rank for rank, (_, method) in enumerate(finals)}


def plot_convergence(runs, path):
    figure, axes = plt.subplots(figsize=(8, 4.5))
    medians = {method: median_curve([run["curve"] for run in runs[method]]) for method in METHODS}
    offsets = end_label_offsets(medians)
    for method, label in METHODS.items():
        for run in runs[method]:
            axes.plot(range(1, len(run["curve"]) + 1), run["curve"], color=COLORS[method], linewidth=1, alpha=0.3)
        median = medians[method]
        axes.plot(range(1, len(median) + 1), median, color=COLORS[method], linewidth=2.2, label=f"{label} (median of {len(runs[method])} seeds)")
        axes.annotate(f"{median[-1]:.3f}", (len(median), median[-1]), xytext=(8, offsets[method]), textcoords="offset points", va="center", color=TEXT_SECONDARY)
    axes.set_ylim(*CONVERGENCE_RANGE)
    axes.set_title("Best validation loss found so far")
    axes.set_xlabel(f"networks trained (evaluations); the first ones, above {CONVERGENCE_RANGE[1]}, are cut")
    axes.set_ylabel("validation cross-entropy (lower is better)")
    axes.legend(loc="upper right")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_final(summary, path):
    figure, (loss_axes, accuracy_axes) = plt.subplots(1, 2, figsize=(8, 3.8))
    for position, (method, label) in enumerate(METHODS.items()):
        rows = [row for row in summary if row["method"] == method]
        losses = [float(row["best_validation_loss"]) for row in rows]
        accuracies = [100 * float(row["test_accuracy"]) for row in rows]
        for axes, values in ((loss_axes, losses), (accuracy_axes, accuracies)):
            axes.scatter([position] * len(values), values, s=64, color=COLORS[method], edgecolor=SURFACE, linewidth=2, zorder=3, label=label)
            axes.hlines(statistics.mean(values), position - 0.25, position + 0.25, color=TEXT_SECONDARY, linewidth=2, zorder=2)
    for axes, title in ((loss_axes, "Best validation loss per seed"), (accuracy_axes, "Test accuracy of the best (%)")):
        axes.set_xticks(range(len(METHODS)), list(METHODS.values()))
        axes.set_xlim(-0.6, len(METHODS) - 0.4)
        axes.set_title(title)
    loss_axes.set_ylabel("validation cross-entropy")
    accuracy_axes.set_ylabel("correct predictions on the test set (%)")
    figure.text(0.5, 0.01, "one dot per seed, bar = mean", ha="center", color=MUTED)
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_explored(runs, path):
    figure, axes = plt.subplots(figsize=(8, 4.5))
    for method, label in METHODS.items():
        rates = [float(row["learning_rate"]) for run in runs[method] for row in run["rows"]]
        scores = [min(float(row["score"]), LOSS_CAP) for run in runs[method] for row in run["rows"]]
        axes.scatter(rates, scores, s=36, color=COLORS[method], edgecolor=SURFACE, linewidth=1, alpha=0.75, label=label)
    axes.set_xscale("log")
    axes.set_title("Every configuration tried: learning rate against validation loss")
    axes.set_xlabel("learning rate (log scale)")
    axes.set_ylabel(f"validation cross-entropy (capped at {LOSS_CAP})")
    axes.legend(loc="lower center", bbox_to_anchor=(0.5, 1.06), ncol=len(METHODS))
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def evaluations_to_reach(curve, target):
    for index, best in enumerate(curve):
        if best <= target:
            return index + 1
    return None


def mean_and_spread(values, digits):
    if len(values) < 2:
        return f"{values[0]:.{digits}f}"
    return f"{statistics.mean(values):.{digits}f} ± {statistics.stdev(values):.{digits}f}"


def summary_table(summary, runs):
    target = max(run["curve"][-1] for method in METHODS for run in runs[method])
    lines = [
        f"| Method | Best validation loss | Test loss | Test accuracy | Networks trained to reach {target:.3f} | Minutes per search |",
        "|---|---|---|---|---|---|",
    ]
    for method, label in METHODS.items():
        rows = [row for row in summary if row["method"] == method]
        reach = [evaluations_to_reach(run["curve"], target) for run in runs[method]]
        lines.append("| " + " | ".join([
            label,
            mean_and_spread([float(row["best_validation_loss"]) for row in rows], 3),
            mean_and_spread([float(row["test_loss"]) for row in rows], 3),
            mean_and_spread([100 * float(row["test_accuracy"]) for row in rows], 1) + " %",
            mean_and_spread([float(value) for value in reach], 1),
            mean_and_spread([float(row["seconds"]) / 60 for row in rows], 1),
        ]) + " |")
    return "\n".join(lines)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    results = Path(argv[0] if argv else "docs/benchmarks/hyperparameter_search")
    images = Path(argv[1] if len(argv) > 1 else "docs/images")
    images.mkdir(parents=True, exist_ok=True)
    style()
    runs = {method: runs_of(results, method) for method in METHODS}
    summary = read_rows(results / "summary.csv")
    plot_convergence(runs, images / "hyperparameter_convergence.png")
    plot_final(summary, images / "hyperparameter_final_results.png")
    plot_explored(runs, images / "hyperparameter_learning_rates.png")
    print(summary_table(summary, runs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
