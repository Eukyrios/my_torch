import numpy as np
import pytest

from my_torch.activations.activation import ReLU, Sigmoid, Softmax, Tanh
from my_torch.layers.dense_layer import DenseLayer
from my_torch.losses.loss import CrossEntropy, Loss, MeanSquaredError
from my_torch.network.network import Network

RNG = np.random.default_rng(20)
INPUTS = RNG.normal(size=(6, 5))
LABELS = np.eye(3)[RNG.integers(0, 3, size=6)]


def build_network(hidden_activation, output_activation) -> Network:
    rng = np.random.default_rng(21)
    return Network([
        DenseLayer(rng.normal(size=(5, 4)), rng.normal(size=4), hidden_activation),
        DenseLayer(rng.normal(size=(4, 4)), rng.normal(size=4), hidden_activation),
        DenseLayer(rng.normal(size=(4, 3)), rng.normal(size=3), output_activation),
    ])


def loss_value(network: Network, loss: Loss, inputs: np.ndarray) -> float:
    return loss(network(inputs), LABELS)


def numerical_gradient(network: Network, loss: Loss, inputs: np.ndarray, array: np.ndarray) -> np.ndarray:
    step = 1e-6
    result = np.zeros_like(array)
    for index in np.ndindex(array.shape):
        original = array[index]
        array[index] = original + step
        higher = loss_value(network, loss, inputs)
        array[index] = original - step
        lower = loss_value(network, loss, inputs)
        array[index] = original
        result[index] = (higher - lower) / (2 * step)
    return result


def backpropagate(network: Network, loss: Loss, inputs: np.ndarray) -> np.ndarray:
    predictions = network(inputs)
    return network.backward(loss.gradient(predictions, LABELS))


CASES = [
    ("tanh + softmax + cross-entropy", Tanh(), Softmax(), CrossEntropy()),
    ("relu + softmax + cross-entropy", ReLU(), Softmax(), CrossEntropy()),
    ("sigmoid + sigmoid + mean squared error", Sigmoid(), Sigmoid(), MeanSquaredError()),
]


@pytest.mark.parametrize("hidden, output, loss", [case[1:] for case in CASES], ids=[case[0] for case in CASES])
def test_every_weight_and_bias_gradient_matches_the_numerical_gradient(hidden, output, loss):
    network = build_network(hidden, output)
    backpropagate(network, loss, INPUTS)
    for layer in network.layers:
        computed_weights, computed_biases = layer.weights_gradient.copy(), layer.biases_gradient.copy()
        assert computed_weights == pytest.approx(
            numerical_gradient(network, loss, INPUTS, layer.weights), rel=1e-4, abs=1e-7
        )
        assert computed_biases == pytest.approx(
            numerical_gradient(network, loss, INPUTS, layer.biases), rel=1e-4, abs=1e-7
        )


@pytest.mark.parametrize("hidden, output, loss", [case[1:] for case in CASES], ids=[case[0] for case in CASES])
def test_the_input_gradient_matches_the_numerical_gradient(hidden, output, loss):
    network = build_network(hidden, output)
    inputs = INPUTS.copy()
    input_gradient = backpropagate(network, loss, inputs)
    assert input_gradient == pytest.approx(numerical_gradient(network, loss, inputs, inputs), rel=1e-4, abs=1e-7)
