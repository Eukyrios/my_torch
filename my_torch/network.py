# A network: an ordered list of layers, each one reading the outputs of the previous one.
# Its forward pass runs every layer in order, on one sample of shape (n_inputs,) or a batch
# of shape (batch, n_inputs), and a network is called like a function: network(inputs)
# runs network.forward(inputs).
# n_inputs and n_outputs come from its first and last dense layers: n_inputs is how many
# numbers each sample must have, n_outputs how many scores the network gives back.
# The checks come first: check_layers rejects anything that is not a layer, a network
# without a dense layer, and dense layers whose sizes do not chain (the outputs of one
# must be the inputs of the next dense layer).

import numpy as np

from my_torch.layer import DenseLayer, Layer


def check_sizes_chain(dense_layers: list[DenseLayer]) -> None:
    for position, (previous, following) in enumerate(zip(dense_layers, dense_layers[1:])):
        if previous.n_outputs != following.n_inputs:
            raise ValueError(
                f"dense layer {position} gives {previous.n_outputs} outputs but dense layer "
                f"{position + 1} expects {following.n_inputs} inputs"
            )


def check_layers(layers: list) -> None:
    for layer in layers:
        if not isinstance(layer, Layer):
            raise TypeError(f"a network is made of layers, got {type(layer).__name__}")
    dense_layers = [layer for layer in layers if isinstance(layer, DenseLayer)]
    if not dense_layers:
        raise ValueError("a network needs at least one dense layer")
    check_sizes_chain(dense_layers)


class Network:
    def __init__(self, layers: list[Layer]):
        self.layers = list(layers)
        check_layers(self.layers)

    @property
    def dense_layers(self) -> list[DenseLayer]:
        return [layer for layer in self.layers if isinstance(layer, DenseLayer)]

    @property
    def n_inputs(self) -> int:
        return self.dense_layers[0].n_inputs

    @property
    def n_outputs(self) -> int:
        return self.dense_layers[-1].n_outputs

    def __call__(self, inputs) -> np.ndarray:
        return self.forward(inputs)

    def forward(self, inputs) -> np.ndarray:
        outputs = np.asarray(inputs, dtype=float)
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs
