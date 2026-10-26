import numpy as np
import pytest

from my_torch.layers.layer import Layer


class AnySize(Layer):
    def __init__(self):
        super().__init__("any_size")

    def forward(self, inputs):
        return inputs


def test_keeps_its_name():
    assert AnySize().name == "any_size"


def test_a_layer_has_no_shape_by_default():
    assert AnySize().shape is None


def test_calling_a_layer_runs_its_forward():
    inputs = np.array([1.0, 2.0])
    assert AnySize()(inputs) is inputs


def test_the_base_layer_cannot_be_used_on_its_own():
    with pytest.raises(TypeError):
        Layer("dense")


def test_the_base_forward_is_abstract():
    with pytest.raises(NotImplementedError):
        Layer.forward(AnySize(), [1, 2])
