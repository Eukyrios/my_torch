import numpy as np
import pytest

from my_torch.activations.activation import (
    Activation,
    Linear,
    ReLU,
    Sigmoid,
    Softmax,
    Tanh,
)

Z = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])


def test_linear_returns_z_unchanged():
    assert Linear()(Z) == pytest.approx(Z)


def test_relu_cuts_negative_values_to_zero():
    assert ReLU()(Z) == pytest.approx([0.0, 0.0, 0.0, 0.5, 2.0])


def test_sigmoid_matches_its_definition():
    assert Sigmoid()(Z) == pytest.approx(1 / (1 + np.exp(-Z)))


def test_sigmoid_is_one_half_at_zero():
    assert Sigmoid()(np.array(0.0)) == pytest.approx(0.5)


def test_sigmoid_does_not_overflow_on_large_values():
    with np.errstate(all="raise"):
        assert Sigmoid()(np.array([-1000.0, 1000.0])) == pytest.approx([0.0, 1.0])


def test_tanh_stays_between_minus_one_and_one():
    assert Tanh()(np.array([-1000.0, 0.0, 1000.0])) == pytest.approx([-1.0, 0.0, 1.0])


def test_softmax_matches_the_hand_computation():
    assert Softmax()(np.array([2.0, 1.0, 0.1])) == pytest.approx([0.659, 0.242, 0.099], abs=1e-3)


def test_softmax_rows_sum_to_one():
    batch = np.random.default_rng(0).normal(size=(16, 3))
    assert Softmax()(batch).sum(axis=1) == pytest.approx(np.ones(16))


def test_softmax_works_row_by_row_on_a_batch():
    softmax = Softmax()
    batch = np.array([[2.0, 1.0, 0.1], [0.0, 0.0, 0.0]])
    outputs = softmax(batch)
    assert outputs[0] == pytest.approx(softmax(batch[0]))
    assert outputs[1] == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_softmax_does_not_overflow_on_large_scores():
    with np.errstate(all="raise"):
        assert Softmax()(np.array([1000.0, 1000.0, 1000.0])) == pytest.approx([1 / 3, 1 / 3, 1 / 3])


@pytest.mark.parametrize(
    "activation_class, name",
    [(Linear, "linear"), (ReLU, "relu"), (Sigmoid, "sigmoid"), (Tanh, "tanh"), (Softmax, "softmax")],
)
def test_each_activation_is_an_activation_with_its_name(activation_class, name):
    activation = activation_class()
    assert isinstance(activation, Activation)
    assert activation.name == name


def test_calling_an_activation_runs_its_forward():
    relu = ReLU()
    assert relu(Z) == pytest.approx(relu.forward(Z))


def test_the_base_class_cannot_be_used_on_its_own():
    with pytest.raises(TypeError):
        Activation()


def test_the_base_forward_is_abstract():
    with pytest.raises(NotImplementedError):
        Activation.forward(Linear(), Z)


def test_the_base_derivative_is_abstract():
    with pytest.raises(NotImplementedError):
        Activation.derivative(Linear(), Z)


def test_every_subclass_is_found():
    assert set(Activation.subclasses()) >= {Linear, ReLU, Sigmoid, Tanh, Softmax}


def test_activation_names_are_unique():
    assert len(set(Activation.names())) == len(Activation.subclasses())


@pytest.mark.parametrize("activation_class", [Linear, ReLU, Sigmoid, Tanh, Softmax])
def test_from_name_builds_the_activation_named(activation_class):
    assert type(Activation.from_name(activation_class().name)) is activation_class


def test_from_name_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="unknown activation 'swish', expected one of: linear, relu"):
        Activation.from_name("swish")


def test_names_lists_every_activation():
    assert Activation.names() == ["linear", "relu", "sigmoid", "tanh", "softmax"]


ALL = [Linear, ReLU, Sigmoid, Tanh, Softmax]
VALUE_BY_VALUE = [Linear, ReLU, Sigmoid, Tanh]
BATCH = np.random.default_rng(3).normal(size=(4, 3)) * 2


def numerical_derivative(activation: Activation, z: np.ndarray) -> np.ndarray:
    step = 1e-6
    return (activation(z + step) - activation(z - step)) / (2 * step)


def numerical_backward(activation: Activation, z: np.ndarray, gradient: np.ndarray) -> np.ndarray:
    step = 1e-6
    result = np.zeros_like(z)
    for index in np.ndindex(z.shape):
        higher, lower = z.copy(), z.copy()
        higher[index] += step
        lower[index] -= step
        result[index] = np.sum(gradient * (activation(higher) - activation(lower))) / (2 * step)
    return result


def test_linear_derivative_is_one():
    assert Linear().derivative(Z) == pytest.approx(np.ones(5))


def test_relu_derivative_is_one_for_positive_values_and_zero_otherwise():
    assert ReLU().derivative(Z) == pytest.approx([0.0, 0.0, 0.0, 1.0, 1.0])


def test_sigmoid_derivative_is_one_quarter_at_zero():
    assert Sigmoid().derivative(np.array(0.0)) == pytest.approx(0.25)


def test_tanh_derivative_is_one_at_zero():
    assert Tanh().derivative(np.array(0.0)) == pytest.approx(1.0)


@pytest.mark.parametrize("activation_class", [Linear, Sigmoid, Tanh])
def test_derivative_matches_the_numerical_derivative(activation_class):
    activation = activation_class()
    assert activation.derivative(Z) == pytest.approx(numerical_derivative(activation, Z), rel=1e-5, abs=1e-8)


def test_relu_derivative_matches_the_numerical_derivative_away_from_zero():
    z = np.array([-2.0, -0.5, 0.5, 2.0])
    assert ReLU().derivative(z) == pytest.approx(numerical_derivative(ReLU(), z))


@pytest.mark.parametrize("activation_class", VALUE_BY_VALUE)
def test_backward_multiplies_the_gradient_by_the_derivative(activation_class):
    activation = activation_class()
    gradient = np.random.default_rng(4).normal(size=BATCH.shape)
    assert activation.backward(BATCH, gradient) == pytest.approx(gradient * activation.derivative(BATCH))


@pytest.mark.parametrize("activation_class", ALL)
def test_backward_matches_the_numerical_gradient(activation_class):
    activation = activation_class()
    gradient = np.random.default_rng(5).normal(size=BATCH.shape)
    assert activation.backward(BATCH, gradient) == pytest.approx(
        numerical_backward(activation, BATCH, gradient), rel=1e-5, abs=1e-8
    )


def test_softmax_derivative_is_one_jacobian_per_sample():
    assert Softmax().derivative(BATCH).shape == (4, 3, 3)


def test_softmax_derivative_matches_the_numerical_jacobian():
    z = np.array([2.0, 1.0, 0.1])
    step = 1e-6
    columns = [(Softmax()(z + step * np.eye(3)[j]) - Softmax()(z - step * np.eye(3)[j])) / (2 * step) for j in range(3)]
    assert Softmax().derivative(z) == pytest.approx(np.array(columns).T, abs=1e-8)


def test_softmax_backward_is_the_jacobian_times_the_gradient():
    gradient = np.random.default_rng(6).normal(size=BATCH.shape)
    jacobians = Softmax().derivative(BATCH)
    assert Softmax().backward(BATCH, gradient) == pytest.approx(np.einsum("bij,bi->bj", jacobians, gradient))
