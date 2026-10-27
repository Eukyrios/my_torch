# A single neuron: the weighted sum of its inputs plus its bias, called like a function.

import numpy as np


def check_parameters(weights: np.ndarray) -> None:
    if weights.ndim != 1:
        raise ValueError(f"a neuron's weights must be a vector, one per input, got shape {weights.shape}")


def check_input_size(inputs: np.ndarray, n_inputs: int) -> None:
    if inputs.shape != (n_inputs,):
        raise ValueError(f"the neuron needs one input per weight ({n_inputs}), got shape {inputs.shape}")


class Neuron:
    def __init__(self, weights, bias: float):
        self.weights = np.asarray(weights, dtype=float)
        self.bias = float(bias)
        check_parameters(self.weights)

    @property
    def n_inputs(self) -> int:
        return self.weights.shape[0]

    def __call__(self, inputs) -> float:
        return self.forward(inputs)

    def forward(self, inputs) -> float:
        inputs = np.asarray(inputs, dtype=float)
        check_input_size(inputs, self.n_inputs)
        return float(np.dot(inputs, self.weights) + self.bias)
