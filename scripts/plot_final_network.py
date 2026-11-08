# Draws the figures of docs/benchmarks/final_network/ from the files of scripts/compare_final_networks.sh: the
# learning curves of every network trained on one graph per loss, and the accuracy of each seed.
# Usage: python3 -m scripts.plot_final_network [RESULTS_DIR] [IMAGES_DIR]

import csv
import re
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

CONFIG_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#8a5cd1")
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
EPOCH = re.compile(r"^epoch (\d+): loss ([0-9.]+), validation loss ([0-9.]+)")
KEPT = re.compile(r"kept the weights after epoch (\d+)")


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


def read_curve(path):
    curve = {"epochs": [], "loss": [], "validation_loss": [], "kept": None}
    for line in Path(path).read_text().splitlines():
        epoch = EPOCH.match(line)
        kept = KEPT.search(line)
        if epoch:
            curve["epochs"].append(int(epoch.group(1)))
            curve["loss"].append(float(epoch.group(2)))
            curve["validation_loss"].append(float(epoch.group(3)))
        if kept:
            curve["kept"] = int(kept.group(1))
    return curve


def config_names(rows):
    return list(dict.fromkeys(row["config"] for row in rows))


def short_name(config):
    return config.replace("_network", "")


def color_of(config, configs):
    return CONFIG_COLORS[configs.index(config) % len(CONFIG_COLORS)]


def save(figure, path):
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_curve(axes, curve, key, color, label):
    axes.plot(curve["epochs"], curve[key], color=color, linewidth=1.8, alpha=0.85, label=label)
    if curve["kept"] in curve["epochs"]:
        index = curve["epochs"].index(curve["kept"])
        axes.scatter([curve["kept"]], [curve[key][index]], s=36, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3)


def plot_curves(results, rows, path):
    configs = config_names(rows)
    figure, (training_axes, validation_axes) = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    labelled = set()
    for row in rows:
        curve = read_curve(results / f"{row['network']}_training.log")
        color = color_of(row["config"], configs)
        label = None if row["config"] in labelled else short_name(row["config"])
        labelled.add(row["config"])
        plot_curve(training_axes, curve, "loss", color, label)
        plot_curve(validation_axes, curve, "validation_loss", color, None)
    training_axes.set_title("Training loss (dropout on)")
    validation_axes.set_title("Validation loss, 16,000 boards set aside")
    for axes in (training_axes, validation_axes):
        axes.set_xlabel("epoch")
        axes.xaxis.set_major_locator(MaxNLocator(integer=True))
    training_axes.set_ylabel("cross-entropy")
    training_axes.legend(loc="upper right")
    figure.text(0.5, 0.01, "one line per seed, dot = epoch whose weights are kept by early stopping", ha="center", color=MUTED)
    save(figure, path)


def plot_accuracies(rows, path):
    configs = config_names(rows)
    figure, (validation_axes, test_axes) = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for position, config in enumerate(configs):
        chosen = [row for row in rows if row["config"] == config]
        plot_seeds(validation_axes, position, [float(row["validation_accuracy"]) for row in chosen], color_of(config, configs))
        plot_seeds(test_axes, position, [float(row["test_accuracy"]) for row in chosen], color_of(config, configs))
    validation_axes.set_title("Validation accuracy, 20,000 boards")
    test_axes.set_title("Test accuracy, 20,000 boards")
    validation_axes.set_ylabel("accuracy (%)")
    for axes in (validation_axes, test_axes):
        axes.set_xticks(range(len(configs)), [short_name(config) for config in configs])
        axes.set_xlim(-0.6, len(configs) - 0.4)
    figure.text(0.5, 0.01, "trained on 160,000 boards, one dot per seed, bar = mean", ha="center", color=MUTED)
    save(figure, path)


def plot_seeds(axes, position, values, color):
    axes.scatter([position] * len(values), values, s=64, color=color, edgecolor=SURFACE, linewidth=2, zorder=3)
    axes.hlines(statistics.mean(values), position - 0.25, position + 0.25, color=TEXT_SECONDARY, linewidth=2, zorder=2)
    axes.annotate(f"{statistics.mean(values):.1f} %", (position + 0.28, statistics.mean(values)), va="center", color=TEXT_SECONDARY)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    results = Path(argv[0] if argv else "docs/benchmarks/final_network")
    images = Path(argv[1] if len(argv) > 1 else "docs/images/final_network")
    images.mkdir(parents=True, exist_ok=True)
    style()
    rows = read_rows(results / "summary.csv")
    plot_curves(results, rows, images / "final_network_curves.png")
    plot_accuracies(rows, images / "final_network_accuracy.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
