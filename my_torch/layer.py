# Layers of a network. Layer is the base class: a name (used to save and reload a network),
# an optional shape (n_inputs, n_outputs), and a forward pass, also run by calling the layer.
# DenseLayer computes activation(inputs . weights + biases) for one sample or a batch
# (one sample per row); column j of the weights belongs to neuron j.
# It keeps its last inputs and z for backpropagation.

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
