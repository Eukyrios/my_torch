# Learning-rate schedules: the rate to use at an epoch, from the initial one. Decaying it
# takes big steps early and finer ones near the end. Constant keeps it, the others lower it.

from abc import ABC, abstractmethod


def check_factor(factor: float) -> None:
    if not 0.0 < factor <= 1.0:
        raise ValueError(f"the decay factor must be in (0, 1], got {factor}")


def check_decay(decay: float) -> None:
    if decay < 0:
        raise ValueError(f"the decay must not be negative, got {decay}")


def check_every(every: int) -> None:
    if every < 1:
        raise ValueError(f"the step must be at least 1 epoch, got {every}")


class Schedule(ABC):
    def __init__(self, name: str):
        self.name = name

    def __call__(self, initial_rate: float, epoch: int) -> float:
        return self.rate(initial_rate, epoch)

    @abstractmethod
    def rate(self, initial_rate: float, epoch: int) -> float:
        raise NotImplementedError


class Constant(Schedule):
    def __init__(self):
        super().__init__("constant")

    def rate(self, initial_rate: float, epoch: int) -> float:
        return initial_rate


class TimeDecay(Schedule):
    def __init__(self, decay: float):
        super().__init__("time_decay")
        check_decay(decay)
        self.decay = decay

    def rate(self, initial_rate: float, epoch: int) -> float:
        return initial_rate / (1.0 + self.decay * epoch)


class ExponentialDecay(Schedule):
    def __init__(self, factor: float):
        super().__init__("exponential_decay")
        check_factor(factor)
        self.factor = factor

    def rate(self, initial_rate: float, epoch: int) -> float:
        return initial_rate * self.factor ** epoch


class StepDecay(Schedule):
    def __init__(self, factor: float, every: int):
        super().__init__("step_decay")
        check_factor(factor)
        check_every(every)
        self.factor = factor
        self.every = every

    def rate(self, initial_rate: float, epoch: int) -> float:
        return initial_rate * self.factor ** (epoch // self.every)
