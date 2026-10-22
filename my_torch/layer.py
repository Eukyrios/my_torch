# A dense layer: n_outputs neurons that all read the same n_inputs values.
# Column j of the weight matrix (shape n_inputs x n_outputs) holds the weights of
# neuron j, so the whole layer is one matrix product: outputs = inputs . weights + biases.
# The same product runs a whole batch at once: inputs of shape (batch, n_inputs), one
# sample per row, give outputs of shape (batch, n_outputs), with the biases added to
# every row. A single sample of shape (n_inputs,) still gives (n_outputs,).
# The layer then applies its Activation (linear by default) to z = inputs . weights + biases.
# It keeps its last inputs and z, which backpropagation needs to compute the gradients.

import numpy as np

from my_torch.activations import Activation, Linear


class Dense:
    def __init__(self, weights, biases, activation: Activation | None = None):
        self.weights = np.asarray(weights, dtype=float)
        self.biases = np.asarray(biases, dtype=float)
        check_weights_are_a_matrix(self.weights)
        check_one_bias_per_neuron(self.weights, self.biases)
        self.activation = activation or Linear()
        self.inputs = None
        self.z = None

    @property
    def n_inputs(self) -> int:
        return self.weights.shape[0]

    @property
    def n_outputs(self) -> int:
        return self.weights.shape[1]

    def forward(self, inputs) -> np.ndarray:
        self.inputs = np.asarray(inputs, dtype=float)
        check_input_size(self.inputs, self.n_inputs)
        self.z = self.inputs @ self.weights + self.biases
        return self.activation(self.z)


def check_weights_are_a_matrix(weights: np.ndarray) -> None:
    if weights.ndim != 2:
        raise ValueError(f"weights must be a matrix (n_inputs x n_outputs), got shape {weights.shape}")


def check_one_bias_per_neuron(weights: np.ndarray, biases: np.ndarray) -> None:
    if biases.shape != (weights.shape[1],):
        raise ValueError(f"expected {weights.shape[1]} biases, one per neuron, got shape {biases.shape}")


def check_input_size(inputs: np.ndarray, n_inputs: int) -> None:
    if inputs.ndim not in (1, 2) or inputs.shape[-1] != n_inputs:
        raise ValueError(
            f"the layer expects {n_inputs} inputs per sample, as (n_inputs,) or (batch, n_inputs), "
            f"got shape {inputs.shape}"
        )
