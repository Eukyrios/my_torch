# Layer is the base class of every layer: it takes the layer's name, which a saved network
# will use to know which kind of layer to rebuild; a subclass passes its own name and
# implements forward, and a layer is called like a function, layer(inputs) running
# layer.forward(inputs). Its shape is (n_inputs, n_outputs) for a layer with a fixed size
# and None by default, for layers that work on any size; DenseLayer overrides it with the
# shape of its weight matrix.
# DenseLayer: n_outputs neurons that all read the same n_inputs values.
# Column j of the weight matrix (shape n_inputs x n_outputs) holds the weights of
# neuron j, so the whole layer is one matrix product: outputs = inputs . weights + biases.
# The same product runs a whole batch at once: inputs of shape (batch, n_inputs), one
# sample per row, give outputs of shape (batch, n_outputs), with the biases added to
# every row. A single sample of shape (n_inputs,) still gives (n_outputs,).
# The layer then applies its Activation (linear by default) to z = inputs . weights + biases.
# It keeps its last inputs and z, which backpropagation needs to compute the gradients.
# The checks come first: check_parameters validates the weights and biases a layer is
# built with, and check_input_size the inputs given to forward.

from abc import ABC, abstractmethod

import numpy as np

from my_torch.activations import Activation, Linear


def check_parameters(weights: np.ndarray, biases: np.ndarray) -> None:
    if weights.ndim != 2:
        raise ValueError(f"weights must be a matrix (n_inputs x n_outputs), got shape {weights.shape}")
    if biases.shape != (weights.shape[1],):
        raise ValueError(f"expected {weights.shape[1]} biases, one per neuron, got shape {biases.shape}")


def check_input_size(inputs: np.ndarray, n_inputs: int) -> None:
    if inputs.ndim not in (1, 2) or inputs.shape[-1] != n_inputs:
        raise ValueError(
            f"the layer expects {n_inputs} inputs per sample, as (n_inputs,) or (batch, n_inputs), "
            f"got shape {inputs.shape}"
        )


class Layer(ABC):
    def __init__(self, name: str):
        self.name = name

    @property
    def shape(self) -> tuple[int, int] | None:
        return None

    def __call__(self, inputs) -> np.ndarray:
        return self.forward(inputs)

    @abstractmethod
    def forward(self, inputs) -> np.ndarray:
        raise NotImplementedError


class DenseLayer(Layer):
    def __init__(self, weights, biases, activation: Activation | None = None):
        super().__init__("dense")
        self.weights = np.asarray(weights, dtype=float)
        self.biases = np.asarray(biases, dtype=float)
        check_parameters(self.weights, self.biases)
        self.activation = activation or Linear()
        self.inputs = None
        self.z = None

    @property
    def shape(self) -> tuple[int, int]:
        return self.weights.shape

    def forward(self, inputs) -> np.ndarray:
        self.inputs = np.asarray(inputs, dtype=float)
        check_input_size(self.inputs, self.shape[0])
        self.z = self.inputs @ self.weights + self.biases
        return self.activation(self.z)
