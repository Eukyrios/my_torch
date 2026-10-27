# Optimizers update a network's parameters from their gradients after each backward pass.
# Optimizer is the base class: a name, a learning rate and step(parameters); SGD moves each
# parameter against its gradient, value -= learning_rate * gradient, in place.

from abc import ABC, abstractmethod

import numpy as np


def check_learning_rate(learning_rate: float) -> None:
    if learning_rate <= 0:
        raise ValueError(f"the learning rate must be positive, got {learning_rate}")


def check_gradient(gradient: np.ndarray | None) -> None:
    if gradient is None:
        raise RuntimeError("a parameter has no gradient yet: run backward before step")


class Optimizer(ABC):
    def __init__(self, name: str, learning_rate: float):
        check_learning_rate(learning_rate)
        self.name = name
        self.learning_rate = learning_rate

    def step(self, parameters: list[tuple[np.ndarray, np.ndarray | None]]) -> None:
        for value, gradient in parameters:
            check_gradient(gradient)
            self.update(value, gradient)

    @abstractmethod
    def update(self, value: np.ndarray, gradient: np.ndarray) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, learning_rate: float = 0.01):
        super().__init__("sgd", learning_rate)

    def update(self, value: np.ndarray, gradient: np.ndarray) -> None:
        value -= self.learning_rate * gradient
