# A network runs its layers in order, on one sample or a batch: network(inputs), and
# backward runs them in reverse, from the loss gradient back to the inputs.
# Its shape comes from the first and last layers with a fixed size, which must chain.
# train shuffles the data each epoch into mini-batches and returns the mean loss per epoch.

import numpy as np

from my_torch.layers.layer import Layer
from my_torch.losses.loss import Loss
from my_torch.optimizers.optimizer import Optimizer


def is_sized(layer: Layer) -> bool:
    return layer.shape is not None


def check_sizes_chain(layers: list[Layer]) -> None:
    received = None
    for position, layer in enumerate(layers):
        if not is_sized(layer):
            continue
        n_inputs, n_outputs = layer.shape
        if received is not None and n_inputs != received:
            raise ValueError(f"layer {position} expects {n_inputs} inputs but receives {received}")
        received = n_outputs


def check_layers(layers: list) -> None:
    for layer in layers:
        if not isinstance(layer, Layer):
            raise TypeError(f"a network is made of layers, got {type(layer).__name__}")
    if not any(is_sized(layer) for layer in layers):
        raise ValueError("a network needs at least one layer with a fixed size, such as a dense layer")
    check_sizes_chain(layers)


def check_training_data(inputs: np.ndarray, labels: np.ndarray) -> None:
    if inputs.shape[0] != labels.shape[0]:
        raise ValueError(f"expected one label per sample, got {inputs.shape[0]} samples and {labels.shape[0]} labels")


def check_training_settings(epochs: int, batch_size: int) -> None:
    if epochs < 1 or batch_size < 1:
        raise ValueError(f"epochs and batch size must be at least 1, got {epochs} and {batch_size}")


def shuffled_batches(n_samples: int, batch_size: int, rng: np.random.Generator) -> list[np.ndarray]:
    order = rng.permutation(n_samples)
    return [order[start:start + batch_size] for start in range(0, n_samples, batch_size)]


class Network:
    def __init__(self, layers: list[Layer]):
        self.layers = list(layers)
        check_layers(self.layers)

    @property
    def shape(self) -> tuple[int, int]:
        sized_layers = [layer for layer in self.layers if is_sized(layer)]
        return sized_layers[0].shape[0], sized_layers[-1].shape[1]

    @property
    def parameters(self) -> list[tuple[np.ndarray, np.ndarray | None]]:
        return [parameter for layer in self.layers for parameter in layer.parameters]

    def __call__(self, inputs) -> np.ndarray:
        return self.forward(inputs)

    def forward(self, inputs) -> np.ndarray:
        outputs = np.asarray(inputs, dtype=float)
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, gradient) -> np.ndarray:
        gradient = np.asarray(gradient, dtype=float)
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)
        return gradient

    def train(self, inputs, labels, loss: Loss, optimizer: Optimizer, epochs: int = 1, batch_size: int = 32, rng: np.random.Generator | None = None) -> list[float]:
        inputs = np.atleast_2d(np.asarray(inputs, dtype=float))
        labels = np.atleast_2d(np.asarray(labels, dtype=float))
        check_training_data(inputs, labels)
        check_training_settings(epochs, batch_size)
        rng = rng or np.random.default_rng()
        return [self.train_epoch(inputs, labels, loss, optimizer, batch_size, rng, epoch) for epoch in range(epochs)]

    def train_epoch(self, inputs: np.ndarray, labels: np.ndarray, loss: Loss, optimizer: Optimizer, batch_size: int, rng: np.random.Generator, epoch: int = 0) -> float:
        optimizer.start_epoch(epoch)
        total = 0.0
        for batch in shuffled_batches(len(inputs), batch_size, rng):
            total += self.train_batch(inputs[batch], labels[batch], loss, optimizer) * len(batch)
        return total / len(inputs)

    def train_batch(self, inputs: np.ndarray, labels: np.ndarray, loss: Loss, optimizer: Optimizer) -> float:
        predictions = self.forward(inputs)
        self.backward(loss.gradient(predictions, labels))
        optimizer.step(self.parameters)
        return loss(predictions, labels)
