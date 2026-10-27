import numpy as np
import pytest

from my_torch.activations.activation import Softmax, Tanh
from my_torch.layers.dense_layer import DenseLayer
from my_torch.losses.loss import CrossEntropy
from my_torch.network.network import Network
from my_torch.optimizers.optimizer import SGD, Optimizer


def test_sgd_moves_each_value_against_its_gradient():
    value = np.array([1.0, 2.0, 3.0])
    SGD(learning_rate=0.1).step([(value, np.array([10.0, -10.0, 0.0]))])
    assert value == pytest.approx([0.0, 3.0, 3.0])


def test_sgd_updates_the_arrays_in_place():
    value = np.array([1.0])
    original = value
    SGD(learning_rate=0.5).step([(value, np.array([1.0]))])
    assert value is original and value[0] == pytest.approx(0.5)


def test_sgd_updates_every_parameter_of_a_network():
    rng = np.random.default_rng(0)
    network = Network([DenseLayer.create(4, 3, Tanh(), rng), DenseLayer.create(3, 2, Softmax(), rng)])
    before = [value.copy() for value, _ in network.parameters]
    inputs, labels = rng.normal(size=(5, 4)), np.eye(2)[[0, 1, 1, 0, 1]]
    predictions = network(inputs)
    network.backward(CrossEntropy().gradient(predictions, labels))
    SGD(learning_rate=0.1).step(network.parameters)
    for old, (new, gradient) in zip(before, network.parameters):
        assert new == pytest.approx(old - 0.1 * gradient)


def test_a_small_sgd_step_lowers_the_loss():
    rng = np.random.default_rng(1)
    network = Network([DenseLayer.create(4, 6, Tanh(), rng), DenseLayer.create(6, 3, Softmax(), rng)])
    inputs, labels = rng.normal(size=(8, 4)), np.eye(3)[rng.integers(0, 3, size=8)]
    loss = CrossEntropy()
    before = loss(network(inputs), labels)
    network.backward(loss.gradient(network(inputs), labels))
    SGD(learning_rate=0.05).step(network.parameters)
    assert loss(network(inputs), labels) < before


def test_sgd_learning_rate_defaults_to_0_01():
    assert SGD().learning_rate == 0.01
    assert SGD().name == "sgd"


@pytest.mark.parametrize("learning_rate", [0.0, -0.1])
def test_rejects_a_learning_rate_that_is_not_positive(learning_rate):
    with pytest.raises(ValueError, match="learning rate must be positive"):
        SGD(learning_rate=learning_rate)


def test_step_before_backward_is_refused():
    layer = DenseLayer.create(3, 2)
    with pytest.raises(RuntimeError, match="run backward before step"):
        SGD().step(layer.parameters)


def test_the_base_class_cannot_be_used_on_its_own():
    with pytest.raises(TypeError):
        Optimizer("optimizer", 0.1)


def test_the_base_update_is_abstract():
    with pytest.raises(NotImplementedError):
        Optimizer.update(SGD(), np.zeros(1), np.zeros(1))
