# Draws the figures of docs/my_torch.md from the my_torch code itself, into docs/images/.
# Usage: python -m scripts.plot_docs

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from my_torch.activations.activation import Linear, ReLU, Sigmoid, Softmax, Tanh
from my_torch.layers.dense_layer import DenseLayer
from my_torch.losses.loss import CrossEntropy, MeanSquaredError
from my_torch.network.network import Network
from my_torch.optimizers.optimizer import SGD, Momentum
from my_torch.optimizers.schedule import Constant, ExponentialDecay, StepDecay, TimeDecay

OUTPUT = Path("docs/images")
SURFACE, GRID, TEXT, TEXT_SECONDARY = "#fcfcfb", "#e4e3df", "#0b0b0b", "#52514e"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
Z = np.linspace(-5, 5, 401)


def style_axes(axes, title: str, x_label: str, y_label: str) -> None:
    axes.set_facecolor(SURFACE)
    axes.set_title(title, color=TEXT, fontsize=11, loc="left")
    axes.set_xlabel(x_label, color=TEXT_SECONDARY, fontsize=9)
    axes.set_ylabel(y_label, color=TEXT_SECONDARY, fontsize=9)
    axes.grid(True, color=GRID, linewidth=0.8)
    axes.set_axisbelow(True)
    axes.tick_params(colors=TEXT_SECONDARY, labelsize=8)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(GRID)


def new_figure(rows: int, columns: int, width: float, height: float):
    figure, axes = plt.subplots(rows, columns, figsize=(width, height), facecolor=SURFACE, squeeze=False)
    return figure, axes


def end_label(axes, x: float, y: float, text: str) -> None:
    axes.annotate(text, (x, y), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8, color=TEXT)


def save(figure, name: str, top: float = 1.0) -> None:
    figure.tight_layout(rect=(0, 0, 1, top))
    figure.savefig(OUTPUT / name, dpi=150, facecolor=SURFACE)
    plt.close(figure)
    print(f"wrote {OUTPUT / name}")


def plot_function_and_derivative(axes, activation, title: str) -> None:
    style_axes(axes, title, "z", "value")
    axes.plot(Z, activation(Z), color=BLUE, linewidth=2, label="f(z)")
    axes.plot(Z, activation.derivative(Z), color=ORANGE, linewidth=2, label="f'(z)")


def plot_activations() -> None:
    figure, axes = new_figure(2, 2, 9, 6.5)
    plot_function_and_derivative(axes[0, 0], Linear(), "Linear: f(z) = z")
    plot_function_and_derivative(axes[0, 1], ReLU(), "ReLU: f(z) = max(0, z)")
    plot_function_and_derivative(axes[1, 0], Sigmoid(), "Sigmoid: f(z) = 1 / (1 + e^-z)")
    plot_function_and_derivative(axes[1, 1], Tanh(), "Tanh: f(z) = tanh(z)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, ["function f(z)", "derivative f'(z), the slope used by backpropagation"], loc="upper center", ncol=2, frameon=False, fontsize=9, labelcolor=TEXT)
    save(figure, "activations.png", top=0.94)


def plot_softmax() -> None:
    figure, axes = new_figure(1, 2, 9, 3.6)
    scores = np.stack([Z, np.zeros_like(Z), np.ones_like(Z)], axis=1)
    probabilities = Softmax()(scores)
    jacobians = Softmax().derivative(scores)
    style_axes(axes[0, 0], "Softmax outputs for z = (z0, 0, 1)", "z0", "probability")
    for column, color, name in ((0, BLUE, "p0"), (1, ORANGE, "p1"), (2, AQUA, "p2")):
        axes[0, 0].plot(Z, probabilities[:, column], color=color, linewidth=2, label=name)
    axes[0, 0].legend(frameon=False, fontsize=8, labelcolor=TEXT)
    style_axes(axes[0, 1], "Jacobian entries, row p0", "z0", "slope")
    axes[0, 1].plot(Z, jacobians[:, 0, 0], color=BLUE, linewidth=2, label="dp0/dz0 = p0 (1 - p0)")
    axes[0, 1].plot(Z, jacobians[:, 0, 1], color=ORANGE, linewidth=2, label="dp0/dz1 = -p0 p1")
    axes[0, 1].legend(frameon=False, fontsize=8, labelcolor=TEXT)
    save(figure, "softmax.png")


def plot_losses() -> None:
    figure, axes = new_figure(2, 2, 9, 6.5)
    probability = np.linspace(0.01, 1.0, 400)
    cross_entropy = CrossEntropy()
    ce_loss = [cross_entropy([p, 1 - p], [1.0, 0.0]) for p in probability]
    ce_gradient = [cross_entropy.gradient([p, 1 - p], [1.0, 0.0])[0, 0] for p in probability]
    style_axes(axes[0, 0], "Cross-entropy loss, expected class 1", "predicted probability of the right class p", "-log(p)")
    axes[0, 0].plot(probability, ce_loss, color=BLUE, linewidth=2)
    style_axes(axes[0, 1], "Cross-entropy gradient", "predicted probability of the right class p", "-1 / p")
    axes[0, 1].plot(probability, ce_gradient, color=ORANGE, linewidth=2)
    prediction = np.linspace(-1.0, 2.0, 400)
    squared_error = MeanSquaredError()
    mse_loss = [squared_error([p], [1.0]) for p in prediction]
    mse_gradient = [squared_error.gradient([p], [1.0])[0, 0] for p in prediction]
    style_axes(axes[1, 0], "Mean squared error, expected value 1", "prediction", "(prediction - 1)^2")
    axes[1, 0].plot(prediction, mse_loss, color=BLUE, linewidth=2)
    style_axes(axes[1, 1], "Mean squared error gradient", "prediction", "2 (prediction - 1)")
    axes[1, 1].plot(prediction, mse_gradient, color=ORANGE, linewidth=2)
    save(figure, "losses.png")


def descend(optimizer, steps: int) -> np.ndarray:
    position = np.array([-8.0, 2.0])
    path = [position.copy()]
    for _ in range(steps):
        gradient = np.array([position[0], 20.0 * position[1]])
        optimizer.step([(position, gradient)])
        path.append(position.copy())
    return np.array(path)


def valley_height(path: np.ndarray) -> np.ndarray:
    return 0.5 * (path[:, 0] ** 2 + 20.0 * path[:, 1] ** 2)


def plot_momentum() -> None:
    figure, axes = new_figure(1, 2, 9, 3.8)
    sgd_path, momentum_path = descend(SGD(0.09), 40), descend(Momentum(0.02, 0.8), 40)
    x, y = np.meshgrid(np.linspace(-9, 2, 200), np.linspace(-2.5, 2.5, 200))
    style_axes(axes[0, 0], "Path on a narrow valley f = (x² + 20y²) / 2", "x", "y")
    axes[0, 0].contour(x, y, 0.5 * (x ** 2 + 20 * y ** 2), levels=12, colors=GRID, linewidths=0.8)
    axes[0, 0].plot(sgd_path[:, 0], sgd_path[:, 1], color=ORANGE, linewidth=2, marker="o", markersize=3, label="SGD(0.09)")
    axes[0, 0].plot(momentum_path[:, 0], momentum_path[:, 1], color=BLUE, linewidth=2, marker="o", markersize=3, label="Momentum(0.02, 0.8)")
    axes[0, 0].legend(frameon=False, fontsize=8, labelcolor=TEXT, loc="lower right")
    style_axes(axes[0, 1], "Height f after each step", "step", "f (log scale)")
    axes[0, 1].plot(valley_height(sgd_path), color=ORANGE, linewidth=2, label="SGD")
    axes[0, 1].plot(valley_height(momentum_path), color=BLUE, linewidth=2, label="Momentum")
    axes[0, 1].set_yscale("log")
    axes[0, 1].legend(frameon=False, fontsize=8, labelcolor=TEXT)
    save(figure, "momentum.png")


def plot_schedules() -> None:
    figure, axes = new_figure(1, 1, 7, 3.8)
    epochs = np.arange(40)
    schedules = ((Constant(), BLUE, "Constant"), (TimeDecay(0.1), ORANGE, "TimeDecay(0.1)"), (ExponentialDecay(0.9), AQUA, "ExponentialDecay(0.9)"), (StepDecay(0.5, 10), YELLOW, "StepDecay(0.5, every=10)"))
    style_axes(axes[0, 0], "Learning rate per epoch, starting at 0.1", "epoch", "learning rate")
    for schedule, color, name in schedules:
        rates = [schedule(0.1, epoch) for epoch in epochs]
        axes[0, 0].plot(epochs, rates, color=color, linewidth=2, label=name, drawstyle="steps-post" if isinstance(schedule, StepDecay) else "default")
        end_label(axes[0, 0], epochs[-1], rates[-1], name)
    axes[0, 0].set_xlim(0, 52)
    axes[0, 0].legend(frameon=False, fontsize=8, labelcolor=TEXT, loc="center right")
    save(figure, "schedules.png")


def plot_xor() -> None:
    inputs = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    labels = np.eye(2)[[0, 1, 1, 0]]
    rng = np.random.default_rng(0)
    network = Network([DenseLayer.create(2, 8, Tanh(), rng), DenseLayer.create(8, 2, Softmax(), rng)])
    losses = network.train(inputs, labels, CrossEntropy(), SGD(0.5), epochs=500, batch_size=4, rng=rng)
    figure, axes = new_figure(1, 1, 7, 3.4)
    style_axes(axes[0, 0], "Learning XOR: cross-entropy per epoch (2 → 8 tanh → 2 softmax, SGD 0.5)", "epoch", "mean loss")
    axes[0, 0].plot(losses, color=BLUE, linewidth=2)
    end_label(axes[0, 0], len(losses) - 1, losses[-1], f"{losses[-1]:.3f}")
    save(figure, "xor.png")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plot_activations()
    plot_softmax()
    plot_losses()
    plot_momentum()
    plot_schedules()
    plot_xor()


if __name__ == "__main__":
    main()
