import numpy as np
import pytest

from my_torch.activations import (
    ALL_ACTIVATIONS,
    LINEAR,
    RELU,
    SIGMOID,
    SOFTMAX,
    TANH,
    Activation,
    activation_names,
    find_activation,
    linear,
    relu,
    sigmoid,
    softmax,
    tanh,
)

Z = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])


def test_linear_returns_z_unchanged():
    assert linear(Z) == pytest.approx(Z)


def test_relu_cuts_negative_values_to_zero():
    assert relu(Z) == pytest.approx([0.0, 0.0, 0.0, 0.5, 2.0])


def test_sigmoid_matches_its_definition():
    assert sigmoid(Z) == pytest.approx(1 / (1 + np.exp(-Z)))


def test_sigmoid_is_one_half_at_zero():
    assert sigmoid(np.array(0.0)) == pytest.approx(0.5)


def test_sigmoid_does_not_overflow_on_large_values():
    with np.errstate(all="raise"):
        assert sigmoid(np.array([-1000.0, 1000.0])) == pytest.approx([0.0, 1.0])


def test_tanh_stays_between_minus_one_and_one():
    outputs = tanh(np.array([-1000.0, 0.0, 1000.0]))
    assert outputs == pytest.approx([-1.0, 0.0, 1.0])


def test_softmax_matches_the_hand_computation():
    assert softmax(np.array([2.0, 1.0, 0.1])) == pytest.approx([0.659, 0.242, 0.099], abs=1e-3)


def test_softmax_rows_sum_to_one():
    batch = np.random.default_rng(0).normal(size=(16, 3))
    assert softmax(batch).sum(axis=1) == pytest.approx(np.ones(16))


def test_softmax_works_row_by_row_on_a_batch():
    batch = np.array([[2.0, 1.0, 0.1], [0.0, 0.0, 0.0]])
    outputs = softmax(batch)
    assert outputs[0] == pytest.approx(softmax(batch[0]))
    assert outputs[1] == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_softmax_does_not_overflow_on_large_scores():
    with np.errstate(all="raise"):
        assert softmax(np.array([1000.0, 1000.0, 1000.0])) == pytest.approx([1 / 3, 1 / 3, 1 / 3])


@pytest.mark.parametrize(
    "activation, name, function",
    [
        (LINEAR, "linear", linear),
        (RELU, "relu", relu),
        (SIGMOID, "sigmoid", sigmoid),
        (TANH, "tanh", tanh),
        (SOFTMAX, "softmax", softmax),
    ],
)
def test_each_activation_holds_its_name_and_function(activation, name, function):
    assert activation.name == name
    assert activation.function is function


def test_an_activation_is_called_like_its_function():
    assert RELU(Z) == pytest.approx(relu(Z))


def test_a_custom_activation_needs_only_a_name_and_a_function():
    double = Activation("double", lambda z: 2 * z)
    assert double(Z) == pytest.approx(2 * Z)


def test_activation_names_are_unique():
    assert len(set(activation_names())) == len(ALL_ACTIVATIONS)


@pytest.mark.parametrize("activation", ALL_ACTIVATIONS)
def test_find_activation_finds_every_activation_by_name(activation):
    assert find_activation(activation.name) is activation


def test_find_activation_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="unknown activation 'swish', expected one of: linear, relu"):
        find_activation("swish")
