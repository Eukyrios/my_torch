# Draws the figures of docs/hyperparameter_search.md from the CSV files written by
# scripts/run_hyperparameter_benchmark.sh, and prints its result tables in markdown.
# Usage: python3 -m scripts.plot_hyperparameter_search [RESULTS_DIR] [IMAGES_DIR]

import csv
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHODS = {"pso": "Particle swarm", "ga": "Genetic algorithm"}
METHOD_COLORS = {"pso": "#2a78d6", "ga": "#eb6834"}
ACTIVATIONS = ("relu", "tanh")
LOSSES = ("cross_entropy", "mean_squared_error")
ACTIVATION_COLORS = {"relu": "#2a78d6", "tanh": "#eb6834"}
LOSS_STYLES = {"cross_entropy": "-", "mean_squared_error": "--"}
LOSS_HATCHES = {"cross_entropy": "", "mean_squared_error": "///"}
LOSS_LABELS = {"cross_entropy": "cross-entropy", "mean_squared_error": "MSE"}
ACTIVATION_LABELS = {"relu": "ReLU", "tanh": "tanh"}
CLASSES = ("Nothing", "Check White", "Check Black", "Checkmate White", "Checkmate Black")
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
        "legend.frameon": False, "hatch.color": SURFACE,
    })


def read_rows(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def accuracy_of(error):
    return 100 * (1 - float(error))


def pair_of(row):
    return ACTIVATIONS[round(float(row["activation"]))], LOSSES[round(float(row["loss"]))]


def pair_label(activation, loss):
    return f"{ACTIVATION_LABELS[activation]} + {LOSS_LABELS[loss]}"


def best_so_far(values):
    curve, best = [], float("-inf")
    for value in values:
        best = max(best, value)
        curve.append(best)
    return curve


def runs_of(results, method):
    runs = []
    for path in sorted(results.glob(f"{method}_seed*_evaluations.csv")):
        rows = read_rows(path)
        runs.append({"rows": rows, "curve": best_so_far([accuracy_of(row["score"]) for row in rows])})
    return runs


def median_curve(curves):
    length = min(len(curve) for curve in curves)
    return [statistics.median(curve[i] for curve in curves) for i in range(length)]


def end_label_offsets(finals):
    ranked = sorted((value, key) for key, value in finals.items())
    return {key: -9 + 18 * rank for rank, (_, key) in enumerate(ranked)}


def save(figure, path):
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_convergence(runs, path):
    figure, axes = plt.subplots(figsize=(8, 4.5))
    medians = {method: median_curve([run["curve"] for run in runs[method]]) for method in METHODS}
    offsets = end_label_offsets({method: curve[-1] for method, curve in medians.items()})
    for method, label in METHODS.items():
        for run in runs[method]:
            axes.plot(range(1, len(run["curve"]) + 1), run["curve"], color=METHOD_COLORS[method], linewidth=1, alpha=0.3)
        median = medians[method]
        axes.plot(range(1, len(median) + 1), median, color=METHOD_COLORS[method], linewidth=2.2, label=f"{label} (median of {len(runs[method])} seeds)")
        axes.annotate(f"{median[-1]:.1f} %", (len(median), median[-1]), xytext=(8, offsets[method]), textcoords="offset points", va="center", color=TEXT_SECONDARY)
    axes.set_title("Best validation accuracy found so far")
    axes.set_xlabel("networks trained (evaluations)")
    axes.set_ylabel("validation accuracy (%)")
    axes.legend(loc="lower right")
    save(figure, path)


def plot_final(summary, path):
    figure, (validation_axes, test_axes) = plt.subplots(1, 2, figsize=(8, 3.8))
    for position, (method, label) in enumerate(METHODS.items()):
        rows = [row for row in summary if row["method"] == method]
        validation = [accuracy_of(row["best_validation_error"]) for row in rows]
        test = [100 * float(row["test_accuracy"]) for row in rows]
        for axes, values in ((validation_axes, validation), (test_axes, test)):
            axes.scatter([position] * len(values), values, s=64, color=METHOD_COLORS[method], edgecolor=SURFACE, linewidth=2, zorder=3)
            axes.hlines(statistics.mean(values), position - 0.25, position + 0.25, color=TEXT_SECONDARY, linewidth=2, zorder=2)
    for axes, title in ((validation_axes, "Best validation accuracy per seed (%)"), (test_axes, "Test accuracy of the best (%)")):
        axes.set_xticks(range(len(METHODS)), list(METHODS.values()))
        axes.set_xlim(-0.6, len(METHODS) - 0.4)
        axes.set_title(title)
    figure.text(0.5, 0.01, "one dot per seed, bar = mean", ha="center", color=MUTED)
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    figure.savefig(path, dpi=150)
    plt.close(figure)


def every_evaluation(runs):
    return [row for method in METHODS for run in runs[method] for row in run["rows"]]


def plot_pairs(runs, path):
    rows = every_evaluation(runs)
    figure, axes = plt.subplots(figsize=(8, 4.5))
    pairs = [(activation, loss) for activation in ACTIVATIONS for loss in LOSSES]
    for position, (activation, loss) in enumerate(pairs):
        values = [accuracy_of(row["score"]) for row in rows if pair_of(row) == (activation, loss)]
        if not values:
            continue
        offsets = [position + 0.3 * ((index * 0.618) % 1 - 0.5) for index in range(len(values))]
        axes.scatter(offsets, values, s=18, color=ACTIVATION_COLORS[activation], alpha=0.55, edgecolor=SURFACE, linewidth=0.5, marker="o" if loss == "cross_entropy" else "s")
        axes.hlines(max(values), position - 0.3, position + 0.3, color=TEXT, linewidth=2)
        axes.annotate(f"best {max(values):.1f} %\nn = {len(values)}", (position, max(values)), xytext=(0, 8), textcoords="offset points", ha="center", color=TEXT_SECONDARY)
    axes.set_xticks(range(len(pairs)), [pair_label(activation, loss) for activation, loss in pairs])
    axes.set_ylim(20, 82)
    axes.set_title("Every configuration tried, by hidden activation and loss")
    axes.set_ylabel("validation accuracy (%), 20 % = guessing")
    save(figure, path)


def plot_learning_rates(runs, path):
    figure, axes = plt.subplots(figsize=(8, 4.5))
    for method, label in METHODS.items():
        rows = [row for run in runs[method] for row in run["rows"]]
        axes.scatter([float(row["learning_rate"]) for row in rows], [accuracy_of(row["score"]) for row in rows], s=30, color=METHOD_COLORS[method], edgecolor=SURFACE, linewidth=1, alpha=0.75, label=label)
    axes.set_xscale("log")
    axes.set_ylim(bottom=20)
    axes.set_title("Every configuration tried: learning rate against validation accuracy")
    axes.set_xlabel("learning rate (log scale)")
    axes.set_ylabel("validation accuracy (%)")
    axes.legend(loc="lower center", bbox_to_anchor=(0.5, 1.06), ncol=len(METHODS))
    save(figure, path)


def curve_files(results):
    curves = {}
    for activation in ACTIVATIONS:
        for loss in LOSSES:
            path = results / f"curves_{activation}_{loss}.csv"
            if path.exists():
                curves[(activation, loss)] = read_rows(path)
    return curves


def plot_curves(curves, path):
    figure, (accuracy_axes, entropy_axes) = plt.subplots(1, 2, figsize=(10, 4.2))
    for (activation, loss), rows in curves.items():
        epochs = [int(row["epoch"]) for row in rows]
        common = {"color": ACTIVATION_COLORS[activation], "linestyle": LOSS_STYLES[loss], "linewidth": 2, "label": pair_label(activation, loss)}
        accuracy_axes.plot(epochs, [100 * float(row["validation_accuracy"]) for row in rows], **common)
        entropy_axes.plot(epochs, [float(row["validation_cross_entropy"]) for row in rows], **common)
    accuracy_axes.set_title("Validation accuracy per epoch (%)")
    entropy_axes.set_title("Validation cross-entropy per epoch")
    for axes in (accuracy_axes, entropy_axes):
        axes.set_xlabel("epoch")
    entropy_axes.legend(loc="upper right")
    figure.text(0.5, 0.01, "colour = hidden activation, line = loss (solid cross-entropy, dashed MSE); each is the best configuration of its pair", ha="center", color=MUTED)
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_classes(results, curves, path):
    figure, axes = plt.subplots(figsize=(10, 4.2))
    pairs = list(curves)
    width = 0.8 / max(len(pairs), 1)
    for index, (activation, loss) in enumerate(pairs):
        rows = read_rows(results / f"curves_{activation}_{loss}_classes.csv")
        positions = [label + (index - (len(pairs) - 1) / 2) * width for label in range(len(rows))]
        axes.bar(positions, [100 * float(row["f1"]) for row in rows], width=width * 0.9, color=ACTIVATION_COLORS[activation], hatch=LOSS_HATCHES[loss], edgecolor=SURFACE, linewidth=0.5, label=pair_label(activation, loss))
    axes.set_xticks(range(len(CLASSES)), CLASSES)
    axes.set_ylabel("F1 on the test set (%)")
    axes.set_title("Per-class F1 of the best configuration of each pair")
    axes.legend(loc="lower center", bbox_to_anchor=(0.5, 1.08), ncol=len(pairs))
    save(figure, path)


def mean_and_spread(values, digits):
    if len(values) < 2:
        return f"{values[0]:.{digits}f}"
    return f"{statistics.mean(values):.{digits}f} ± {statistics.stdev(values):.{digits}f}"


def evaluations_to_reach(curve, target):
    for index, best in enumerate(curve):
        if best >= target:
            return index + 1
    return None


def methods_table(summary, runs):
    target = min(run["curve"][-1] for method in METHODS for run in runs[method])
    lines = [
        f"| Method | Best validation accuracy | Test accuracy | Networks trained to reach {target:.1f} % | Minutes per search |",
        "|---|---|---|---|---|",
    ]
    for method, label in METHODS.items():
        rows = [row for row in summary if row["method"] == method]
        reach = [evaluations_to_reach(run["curve"], target) for run in runs[method]]
        lines.append("| " + " | ".join([
            label,
            mean_and_spread([accuracy_of(row["best_validation_error"]) for row in rows], 1) + " %",
            mean_and_spread([100 * float(row["test_accuracy"]) for row in rows], 1) + " %",
            mean_and_spread([float(value) for value in reach], 1),
            mean_and_spread([float(row["seconds"]) / 60 for row in rows], 1),
        ]) + " |")
    return "\n".join(lines)


def pairs_table(runs):
    rows = every_evaluation(runs)
    lines = ["| Hidden activation + loss | Networks trained | Median validation accuracy | Best validation accuracy |", "|---|---|---|---|"]
    for activation in ACTIVATIONS:
        for loss in LOSSES:
            values = [accuracy_of(row["score"]) for row in rows if pair_of(row) == (activation, loss)]
            if values:
                lines.append(f"| {pair_label(activation, loss)} | {len(values)} | {statistics.median(values):.1f} % | {max(values):.1f} % |")
    return "\n".join(lines)


def curves_table(results):
    path = results / "curves_summary.csv"
    if not path.exists():
        return ""
    lines = ["| Hidden activation + loss | Found by | Best epoch | Validation accuracy | Test accuracy | Test macro-F1 |", "|---|---|---|---|---|---|"]
    for row in read_rows(path):
        activation, loss = row["pair"].split("_", 1)
        lines.append(f"| {pair_label(activation, loss)} | {row['found_by']} | {row['best_epoch']} | {100 * float(row['validation_accuracy']):.1f} % | {100 * float(row['test_accuracy']):.1f} % | {100 * float(row['test_macro_f1']):.1f} % |")
    return "\n".join(lines)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    results = Path(argv[0] if argv else "docs/benchmarks/hyperparameter_search")
    images = Path(argv[1] if len(argv) > 1 else "docs/images/hyperparameter_search")
    images.mkdir(parents=True, exist_ok=True)
    style()
    runs = {method: runs_of(results, method) for method in METHODS}
    summary = read_rows(results / "summary.csv")
    plot_convergence(runs, images / "hyperparameter_convergence.png")
    plot_final(summary, images / "hyperparameter_final_results.png")
    plot_pairs(runs, images / "hyperparameter_pairs.png")
    plot_learning_rates(runs, images / "hyperparameter_learning_rates.png")
    curves = curve_files(results)
    if curves:
        plot_curves(curves, images / "learning_curves.png")
        plot_classes(results, curves, images / "learning_curves_classes.png")
    print(methods_table(summary, runs))
    print()
    print(pairs_table(runs))
    print()
    print(curves_table(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
