# Draws the figures of docs/refined_search.md from the CSV files of scripts/run_refined_search.sh and of the
# full-scale check (scripts/compare_final_networks.sh), and prints the result tables in markdown.
# Usage: python3 -m scripts.plot_refined_search [RESULTS_DIR] [FINAL_CHECK_DIR] [IMAGES_DIR]

import csv
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")
POINT_COLOR = "#2a78d6"
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"


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


def accuracy_of(error):
    return 100 * (1 - float(error))


def best_so_far(values):
    curve, best = [], float("-inf")
    for value in values:
        best = max(best, value)
        curve.append(best)
    return curve


def searches(results):
    found = []
    for path in sorted(results.glob("refined_seed*_evaluations.csv")):
        rows = read_rows(path)
        found.append({"seed": path.name.split("_")[1].removeprefix("seed"), "rows": rows, "curve": best_so_far([accuracy_of(row["score"]) for row in rows])})
    return found


def save(figure, path):
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_convergence(runs, path):
    figure, axes = plt.subplots(figsize=(8, 4.5))
    for index, run in enumerate(runs):
        color = SEED_COLORS[index % len(SEED_COLORS)]
        axes.plot(range(1, len(run["curve"]) + 1), run["curve"], color=color, linewidth=2, label=f"search {run['seed']}")
        axes.annotate(f"search {run['seed']}: {run['curve'][-1]:.2f} %", (len(run["curve"]), run["curve"][-1]), xytext=(8, -12 + 12 * index), textcoords="offset points", va="center", color=color)
    start = accuracy_of(runs[0]["rows"][0]["score"])
    axes.axhline(start, color=MUTED, linewidth=1.5, linestyle="--")
    axes.annotate(f"current default (2 × 47): {start:.2f} %", (1, start), xytext=(4, -12), textcoords="offset points", color=MUTED)
    axes.set_title("Refined search: best validation accuracy found so far")
    axes.set_xlabel("networks trained (evaluations)")
    axes.set_ylabel("validation accuracy (%), 20,000 boards")
    axes.set_xlim(right=len(runs[0]["curve"]) * 1.25)
    save(figure, path)


def plot_regularisation(runs, path):
    rows = [row for run in runs for row in run["rows"]]
    accuracies = [accuracy_of(row["score"]) for row in rows]
    figure, (dropout_axes, decay_axes) = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    dropout_axes.scatter([float(row["dropout"]) for row in rows], accuracies, s=28, color=POINT_COLOR, edgecolor=SURFACE, linewidth=1, alpha=0.75)
    decay_axes.scatter([float(row["weight_decay"]) for row in rows], accuracies, s=28, color=POINT_COLOR, edgecolor=SURFACE, linewidth=1, alpha=0.75)
    decay_axes.set_xscale("log")
    dropout_axes.set_title("Dropout rate")
    decay_axes.set_title("L2 weight decay (log scale)")
    dropout_axes.set_ylabel("validation accuracy (%)")
    floor = sorted(accuracies)[len(accuracies) // 20]
    dropout_axes.set_ylim(floor - 1, max(accuracies) + 0.5)
    figure.text(0.5, 0.01, "every configuration of the 3 searches; the lowest 5 % are below the axis", ha="center", color=MUTED)
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    figure.savefig(path, dpi=150)
    plt.close(figure)


def final_rows(final_check):
    path = final_check / "summary.csv"
    return read_rows(path) if path.exists() else []


def plot_final_check(rows, path):
    configs = list(dict.fromkeys(row["config"] for row in rows))
    figure, (validation_axes, test_axes) = plt.subplots(1, 2, figsize=(10, 4.2))
    for position, config in enumerate(configs):
        chosen = [row for row in rows if row["config"] == config]
        for axes, key in ((validation_axes, "validation_accuracy"), (test_axes, "test_accuracy")):
            values = [float(row[key]) for row in chosen]
            axes.scatter([position] * len(values), values, s=64, color=POINT_COLOR, edgecolor=SURFACE, linewidth=2, zorder=3)
            axes.hlines(statistics.mean(values), position - 0.25, position + 0.25, color=TEXT_SECONDARY, linewidth=2, zorder=2)
    for axes, title in ((validation_axes, "Validation accuracy, 20,000 boards (%)"), (test_axes, "Test accuracy, 20,000 boards (%)")):
        axes.set_xticks(range(len(configs)), [config.replace("_network", "").replace("_best", "") for config in configs], rotation=20)
        axes.set_title(title)
    figure.text(0.5, 0.01, "trained on all 160,000 boards, one dot per seed, bar = mean", ha="center", color=MUTED)
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    figure.savefig(path, dpi=150)
    plt.close(figure)


def mean_and_spread(values, digits):
    if len(values) < 2:
        return f"{values[0]:.{digits}f}"
    return f"{statistics.mean(values):.{digits}f} ± {statistics.stdev(values):.{digits}f}"


def searches_table(results):
    lines = ["| Search | Networks trained | Starting point (current default) | Best found | Minutes | Best configuration |", "|---|---|---|---|---|---|"]
    for row in read_rows(results / "summary.csv"):
        lines.append(f"| {row['seed']} | {row['evaluations']} | {accuracy_of(row['first_start_validation_error']):.2f} % | {accuracy_of(row['best_validation_error']):.2f} % | {float(row['seconds']) / 60:.0f} | {row['hyperparameters']} |")
    return "\n".join(lines)


def final_table(rows):
    lines = ["| Config | Networks | Epochs kept | Validation accuracy | Test accuracy | Training minutes |", "|---|---|---|---|---|---|"]
    for config in dict.fromkeys(row["config"] for row in rows):
        chosen = [row for row in rows if row["config"] == config]
        lines.append("| " + " | ".join([
            config, str(len(chosen)),
            mean_and_spread([float(row["epochs_kept"]) for row in chosen], 1),
            mean_and_spread([float(row["validation_accuracy"]) for row in chosen], 2) + " %",
            mean_and_spread([float(row["test_accuracy"]) for row in chosen], 2) + " %",
            mean_and_spread([float(row["seconds"]) / 60 for row in chosen], 1),
        ]) + " |")
    return "\n".join(lines)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    results = Path(argv[0] if argv else "docs/benchmarks/refined_search")
    final_check = Path(argv[1] if len(argv) > 1 else results / "final_check")
    images = Path(argv[2] if len(argv) > 2 else "docs/images/refined_search")
    images.mkdir(parents=True, exist_ok=True)
    style()
    runs = searches(results)
    plot_convergence(runs, images / "refined_convergence.png")
    plot_regularisation(runs, images / "refined_regularisation.png")
    print(searches_table(results))
    rows = final_rows(final_check)
    if rows:
        plot_final_check(rows, images / "refined_final_check.png")
        print()
        print(final_table(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
