# A single neuron: the weighted sum of its inputs plus a bias,
# z = x1*w1 + x2*w2 + ... + xn*wn + b.
# This is the value an activation function is later applied to.

import numpy as np


def weighted_sum(inputs, weights, bias: float) -> float:
    inputs = np.asarray(inputs, dtype=float)
    weights = np.asarray(weights, dtype=float)
    check_one_weight_per_input(inputs, weights)
    return float(np.dot(inputs, weights) + bias)


def check_one_weight_per_input(inputs: np.ndarray, weights: np.ndarray) -> None:
    if inputs.ndim != 1 or inputs.shape != weights.shape:
        raise ValueError(
            f"a neuron needs one weight per input, got inputs of shape {inputs.shape} "
            f"and weights of shape {weights.shape}"
        )
