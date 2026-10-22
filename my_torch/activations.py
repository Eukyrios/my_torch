# Activation functions, applied to a layer's weighted sums z.
# Activation is the base class: it takes the activation's name, each subclass passes its
# own name and implements forward, and an instance is called like a function, activation(z).
# Linear keeps z as is, ReLU cuts negative values to 0, Sigmoid squashes into (0, 1),
# Tanh into (-1, 1), and Softmax turns each row of scores into probabilities summing to 1.
# Sigmoid is computed as 0.5 * (1 + tanh(z / 2)), the same function without the overflow
# of exp(-z) for large negative z; Softmax subtracts each row's maximum before exp for the
# same reason, which leaves the result unchanged.
# The name is how a saved network will store an activation: Activation.from_name turns it
# back into an instance by searching Activation.subclasses().

from abc import ABC, abstractmethod

import numpy as np


class Activation(ABC):
    def __init__(self, name: str):
        self.name = name

    @classmethod
    def subclasses(cls) -> list[type["Activation"]]:
        return Activation.__subclasses__()

    @classmethod
    def names(cls) -> list[str]:
        return [subclass().name for subclass in Activation.subclasses()]

    @classmethod
    def from_name(cls, name: str) -> "Activation":
        for subclass in Activation.subclasses():
            activation = subclass()
            if activation.name == name:
                return activation
        raise ValueError(f"unknown activation {name!r}, expected one of: {', '.join(Activation.names())}")

    def __call__(self, z: np.ndarray) -> np.ndarray:
        return self.forward(z)

    @abstractmethod
    def forward(self, z: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class Linear(Activation):
    def __init__(self):
        super().__init__("linear")

    def forward(self, z: np.ndarray) -> np.ndarray:
        return z


class ReLU(Activation):
    def __init__(self):
        super().__init__("relu")

    def forward(self, z: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, z)


class Sigmoid(Activation):
    def __init__(self):
        super().__init__("sigmoid")

    def forward(self, z: np.ndarray) -> np.ndarray:
        return 0.5 * (1.0 + np.tanh(0.5 * z))


class Tanh(Activation):
    def __init__(self):
        super().__init__("tanh")

    def forward(self, z: np.ndarray) -> np.ndarray:
        return np.tanh(z)


class Softmax(Activation):
    def __init__(self):
        super().__init__("softmax")

    def forward(self, z: np.ndarray) -> np.ndarray:
        exps = np.exp(z - np.max(z, axis=-1, keepdims=True))
        return exps / np.sum(exps, axis=-1, keepdims=True)

