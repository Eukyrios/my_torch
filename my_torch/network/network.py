# A network runs its layers in order, on one sample or a batch: network(inputs), and
# backward runs them in reverse, from the loss gradient back to the inputs.
# Its shape comes from the first and last layers with a fixed size, which must chain.

import numpy as np

from my_torch.layers.layer import Layer


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


class Network:
    def __init__(self, layers: list[Layer]):
        self.layers = list(layers)
        check_layers(self.layers)

    @property
    def shape(self) -> tuple[int, int]:
        sized_layers = [layer for layer in self.layers if is_sized(layer)]
        return sized_layers[0].shape[0], sized_layers[-1].shape[1]

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
