# Loss functions: how wrong a network's predictions are compared to the expected outputs,
# averaged over the batch. Loss is the base class: a name, forward giving the loss (also run
# by calling it) and gradient giving d(loss)/d(predictions), where backpropagation starts.
# CrossEntropy is for classification: predictions are probabilities (softmax output) and
# expected is one-hot; predictions are clipped to at least epsilon (1e-12 by default) so
# log never sees 0.

from abc import ABC, abstractmethod

import numpy as np


def prepare(predictions, expected) -> tuple[np.ndarray, np.ndarray]:
    predictions = np.atleast_2d(np.asarray(predictions, dtype=float))
    expected = np.atleast_2d(np.asarray(expected, dtype=float))
    if predictions.shape != expected.shape:
        raise ValueError(
            f"predictions and expected outputs must have the same shape, "
            f"got {predictions.shape} and {expected.shape}"
        )
    return predictions, expected


def check_epsilon(epsilon: float) -> None:
    if not 0.0 < epsilon < 1.0:
        raise ValueError(f"epsilon must be between 0 and 1, got {epsilon}")


class Loss(ABC):
    def __init__(self, name: str):
        self.name = name

    def __call__(self, predictions, expected) -> float:
        return self.forward(predictions, expected)

    @abstractmethod
    def forward(self, predictions, expected) -> float:
        raise NotImplementedError

    @abstractmethod
    def gradient(self, predictions, expected) -> np.ndarray:
        raise NotImplementedError


class MeanSquaredError(Loss):
    def __init__(self):
        super().__init__("mean_squared_error")

    def forward(self, predictions, expected) -> float:
        predictions, expected = prepare(predictions, expected)
        return float(np.mean((predictions - expected) ** 2))

    def gradient(self, predictions, expected) -> np.ndarray:
        predictions, expected = prepare(predictions, expected)
        return 2.0 * (predictions - expected) / predictions.size


class CrossEntropy(Loss):
    def __init__(self, epsilon: float = 1e-12):
        super().__init__("cross_entropy")
        check_epsilon(epsilon)
        self.epsilon = epsilon

    def forward(self, predictions, expected) -> float:
        predictions, expected = prepare(predictions, expected)
        clipped = np.clip(predictions, self.epsilon, 1.0)
        return float(-np.sum(expected * np.log(clipped)) / predictions.shape[0])

    def gradient(self, predictions, expected) -> np.ndarray:
        predictions, expected = prepare(predictions, expected)
        clipped = np.clip(predictions, self.epsilon, 1.0)
        return -expected / clipped / predictions.shape[0]
