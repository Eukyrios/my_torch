import numpy as np
import pytest

from my_torch.activations import ACTIVATIONS, get_activation, linear, relu, sigmoid, softmax, tanh

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


@pytest.mark.parametrize("name", ["linear", "relu", "sigmoid", "tanh", "softmax"])
def test_get_activation_finds_every_activation_by_name(name):
    assert get_activation(name) is ACTIVATIONS[name]


def test_get_activation_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="unknown activation 'swish'"):
        get_activation("swish")
