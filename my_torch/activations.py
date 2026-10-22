# Activation functions, applied to a layer's weighted sums z.
# linear keeps z as is, relu cuts negative values to 0, sigmoid squashes into (0, 1),
# tanh into (-1, 1), and softmax turns each row of scores into probabilities summing to 1.
# sigmoid is computed as 0.5 * (1 + tanh(z / 2)), the same function without the overflow
# of exp(-z) for large negative z; softmax subtracts each row's maximum before exp for the
# same reason, which leaves the result unchanged.
# An Activation holds its name and its function and is called like the function. The name
# is how a saved network will store it, and find_activation turns a name back into it.

from dataclasses import dataclass
from typing import Callable

import numpy as np


def linear(z: np.ndarray) -> np.ndarray:
    return z


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def tanh(z: np.ndarray) -> np.ndarray:
    return np.tanh(z)


def softmax(z: np.ndarray) -> np.ndarray:
    exps = np.exp(z - np.max(z, axis=-1, keepdims=True))
    return exps / np.sum(exps, axis=-1, keepdims=True)


@dataclass(frozen=True)
class Activation:
    name: str
    function: Callable[[np.ndarray], np.ndarray]

    def __call__(self, z: np.ndarray) -> np.ndarray:
        return self.function(z)


LINEAR = Activation("linear", linear)
RELU = Activation("relu", relu)
SIGMOID = Activation("sigmoid", sigmoid)
TANH = Activation("tanh", tanh)
SOFTMAX = Activation("softmax", softmax)

ALL_ACTIVATIONS = (LINEAR, RELU, SIGMOID, TANH, SOFTMAX)


def activation_names() -> list[str]:
    return [activation.name for activation in ALL_ACTIVATIONS]


def find_activation(name: str) -> Activation:
    for activation in ALL_ACTIVATIONS:
        if activation.name == name:
            return activation
    raise ValueError(f"unknown activation {name!r}, expected one of: {', '.join(activation_names())}")
