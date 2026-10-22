# A dense layer: n_outputs neurons that all read the same n_inputs values.
# Column j of the weight matrix (shape n_inputs x n_outputs) holds the weights of
# neuron j, so the whole layer is one matrix product: outputs = inputs . weights + biases.

import numpy as np


class Dense:
    def __init__(self, weights, biases):
        self.weights = np.asarray(weights, dtype=float)
        self.biases = np.asarray(biases, dtype=float)
        check_weights_are_a_matrix(self.weights)
        check_one_bias_per_neuron(self.weights, self.biases)

    @property
    def n_inputs(self) -> int:
        return self.weights.shape[0]

    @property
    def n_outputs(self) -> int:
        return self.weights.shape[1]

    def forward(self, inputs) -> np.ndarray:
        inputs = np.asarray(inputs, dtype=float)
        check_input_size(inputs, self.n_inputs)
        return inputs @ self.weights + self.biases


def check_weights_are_a_matrix(weights: np.ndarray) -> None:
    if weights.ndim != 2:
        raise ValueError(f"weights must be a matrix (n_inputs x n_outputs), got shape {weights.shape}")


def check_one_bias_per_neuron(weights: np.ndarray, biases: np.ndarray) -> None:
    if biases.shape != (weights.shape[1],):
        raise ValueError(f"expected {weights.shape[1]} biases, one per neuron, got shape {biases.shape}")


def check_input_size(inputs: np.ndarray, n_inputs: int) -> None:
    if inputs.shape != (n_inputs,):
        raise ValueError(f"the layer expects {n_inputs} inputs, got shape {inputs.shape}")
