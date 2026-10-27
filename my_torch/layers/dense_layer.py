# DenseLayer computes activation(inputs . weights + biases); column j of the weights belongs
# to neuron j. create() starts from random weights (He for ReLU, Xavier otherwise); backward
# stores the weight and bias gradients and returns the gradient for the previous layer.

import numpy as np

from my_torch.activations.activation import Activation, Linear, ReLU
from my_torch.layers.layer import Layer


def check_sizes(n_inputs: int, n_outputs: int) -> None:
    if n_inputs < 1 or n_outputs < 1:
        raise ValueError(f"a dense layer needs at least 1 input and 1 output, got {n_inputs} and {n_outputs}")


def initial_scale(n_inputs: int, n_outputs: int, activation: Activation) -> float:
    if isinstance(activation, ReLU):
        return float(np.sqrt(2.0 / n_inputs))
    return float(np.sqrt(2.0 / (n_inputs + n_outputs)))


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


def check_backward(z: np.ndarray | None, gradient: np.ndarray) -> None:
    if z is None:
        raise RuntimeError("backward needs the values of a forward pass: call forward first")
    if gradient.shape != z.shape:
        raise ValueError(f"the gradient must have the shape of the last outputs {z.shape}, got {gradient.shape}")


class DenseLayer(Layer):
    def __init__(self, weights, biases, activation: Activation | None = None):
        super().__init__("dense")
        self.weights = np.asarray(weights, dtype=float)
        self.biases = np.asarray(biases, dtype=float)
        check_parameters(self.weights, self.biases)
        self.activation = activation or Linear()
        self.inputs = None
        self.z = None
        self.weights_gradient = None
        self.biases_gradient = None

    @classmethod
    def create(
        cls, n_inputs: int, n_outputs: int, activation: Activation | None = None, rng: np.random.Generator | None = None
    ) -> "DenseLayer":
        check_sizes(n_inputs, n_outputs)
        activation = activation or Linear()
        rng = rng or np.random.default_rng()
        scale = initial_scale(n_inputs, n_outputs, activation)
        weights = rng.normal(0.0, scale, size=(n_inputs, n_outputs))
        return cls(weights, np.zeros(n_outputs), activation)

    @property
    def shape(self) -> tuple[int, int]:
        return self.weights.shape

    @property
    def parameters(self) -> list[tuple[np.ndarray, np.ndarray | None]]:
        return [(self.weights, self.weights_gradient), (self.biases, self.biases_gradient)]

    def forward(self, inputs) -> np.ndarray:
        self.inputs = np.asarray(inputs, dtype=float)
        check_input_size(self.inputs, self.shape[0])
        self.z = self.inputs @ self.weights + self.biases
        return self.activation(self.z)

    def backward(self, gradient) -> np.ndarray:
        gradient = np.asarray(gradient, dtype=float)
        check_backward(self.z, gradient)
        z_gradient = self.activation.backward(self.z, gradient)
        self.weights_gradient = np.atleast_2d(self.inputs).T @ np.atleast_2d(z_gradient)
        self.biases_gradient = np.atleast_2d(z_gradient).sum(axis=0)
        return z_gradient @ self.weights.T
