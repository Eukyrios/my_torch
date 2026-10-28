# Layer is the base class of every layer: a name (used to save and reload a network), an
# optional shape (n_inputs, n_outputs), None for layers that work on any size, a forward
# pass (also run by calling the layer), a backward pass returning the input gradient, and
# its parameters as (value, gradient) pairs for the optimizer, none by default.

from abc import ABC, abstractmethod

import numpy as np


class Layer(ABC):
    def __init__(self, name: str):
        self.name = name

    @property
    def shape(self) -> tuple[int, int] | None:
        return None

    @property
    def parameters(self) -> list[tuple[np.ndarray, np.ndarray | None]]:
        return []

    def __call__(self, inputs) -> np.ndarray:
        return self.forward(inputs)

    @abstractmethod
    def forward(self, inputs) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def backward(self, gradient) -> np.ndarray:
        raise NotImplementedError
