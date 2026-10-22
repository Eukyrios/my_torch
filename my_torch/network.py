# A network: an ordered list of layers, each one reading the outputs of the previous one.
# Its forward pass runs every layer in order, on one sample of shape (n_inputs,) or a batch
# of shape (batch, n_inputs), and a network is called like a function: network(inputs)
# runs network.forward(inputs).
# It works with any Layer. Layers with a fixed size (a shape, like a dense layer) give the
# network its shape (n_inputs, n_outputs): n_inputs from the first of them is how many
# numbers each sample must have, n_outputs from the last how many scores it gives back.
# Layers without a fixed size work on whatever they receive and are skipped there.
# The checks come first: check_layers rejects anything that is not a layer, a network
# without a layer of fixed size, and sized layers that do not chain (the outputs of one
# must be the inputs of the next sized layer).

import numpy as np

from my_torch.layer import Layer


def is_sized(layer: Layer) -> bool:
    return layer.shape is not None


def check_sizes_chain(layers: list[Layer]) -> None:
    sized = [(position, layer) for position, layer in enumerate(layers) if is_sized(layer)]
    for (previous_position, previous), (following_position, following) in zip(sized, sized[1:]):
        previous_outputs, following_inputs = previous.shape[1], following.shape[0]
        if previous_outputs != following_inputs:
            raise ValueError(
                f"layer {previous_position} gives {previous_outputs} outputs but layer "
                f"{following_position} expects {following_inputs} inputs"
            )


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
