import numpy as np
import pytest

from my_torch.activations.activation import Softmax, Tanh
from my_torch.layers.dense_layer import DenseLayer
from my_torch.losses.loss import CrossEntropy
from my_torch.network.network import Network
from my_torch.optimizers.optimizer import SGD, Momentum, Optimizer


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


def test_momentum_first_step_is_a_plain_sgd_step():
    value = np.array([1.0, 2.0])
    Momentum(learning_rate=0.1, momentum=0.9).step([(value, np.array([1.0, -1.0]))])
    assert value == pytest.approx([0.9, 2.1])


def test_momentum_builds_up_speed_when_the_gradient_keeps_its_direction():
    value, optimizer = np.array([0.0]), Momentum(learning_rate=0.1, momentum=0.9)
    positions = []
    for _ in range(3):
        optimizer.step([(value, np.array([1.0]))])
        positions.append(value[0])
    assert positions == pytest.approx([-0.1, -0.29, -0.561])


def test_momentum_cancels_out_gradients_that_flip_direction():
    sgd_value, momentum_value = np.array([0.0]), np.array([0.0])
    sgd, momentum = SGD(learning_rate=0.1), Momentum(learning_rate=0.1, momentum=0.9)
    sgd_moves, momentum_moves = [], []
    for step in range(10):
        gradient = np.array([1.0 if step % 2 == 0 else -1.0])
        before_sgd, before_momentum = sgd_value[0], momentum_value[0]
        sgd.step([(sgd_value, gradient)])
        momentum.step([(momentum_value, gradient)])
        sgd_moves.append(abs(sgd_value[0] - before_sgd))
        momentum_moves.append(abs(momentum_value[0] - before_momentum))
    assert sum(momentum_moves[2:]) < sum(sgd_moves[2:])


def test_momentum_keeps_one_velocity_per_parameter():
    first, second = np.array([0.0]), np.array([0.0])
    optimizer = Momentum(learning_rate=0.1, momentum=0.9)
    optimizer.step([(first, np.array([1.0])), (second, np.array([-1.0]))])
    optimizer.step([(first, np.array([1.0])), (second, np.array([-1.0]))])
    assert first == pytest.approx([-0.29]) and second == pytest.approx([0.29])


def test_momentum_zero_behaves_like_sgd():
    sgd_value, momentum_value = np.array([1.0, -1.0]), np.array([1.0, -1.0])
    for gradient in ([0.5, 0.2], [-0.3, 0.8], [0.1, 0.1]):
        SGD(0.1).step([(sgd_value, np.array(gradient))])
        Momentum(0.1, momentum=0.0).step([(momentum_value, np.array(gradient))])
    assert momentum_value == pytest.approx(sgd_value)


def final_loss_after_training(optimizer: Optimizer) -> float:
    data_rng, network_rng = np.random.default_rng(3), np.random.default_rng(4)
    inputs, labels = data_rng.normal(size=(40, 4)), np.eye(3)[data_rng.integers(0, 3, size=40)]
    network = Network([DenseLayer.create(4, 8, Tanh(), network_rng), DenseLayer.create(8, 3, Softmax(), network_rng)])
    return network.train(inputs, labels, CrossEntropy(), optimizer, epochs=30, batch_size=8, rng=np.random.default_rng(5))[-1]


def test_momentum_trains_a_network_faster_than_sgd():
    assert final_loss_after_training(Momentum(0.05, 0.9)) < final_loss_after_training(SGD(0.05))


def test_momentum_defaults():
    optimizer = Momentum()
    assert (optimizer.name, optimizer.learning_rate, optimizer.momentum) == ("momentum", 0.01, 0.9)


@pytest.mark.parametrize("momentum", [-0.1, 1.0, 1.5])
def test_rejects_a_momentum_outside_0_and_1(momentum):
    with pytest.raises(ValueError, match=r"momentum must be in \[0, 1\)"):
        Momentum(momentum=momentum)
