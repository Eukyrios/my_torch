import numpy as np

from my_torch.activations.activation import Softmax, Tanh
from my_torch.layers.dense_layer import DenseLayer
from my_torch.losses.loss import CrossEntropy
from my_torch.network.network import Network
from my_torch.optimizers.optimizer import SGD

XOR_INPUTS = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
XOR_OUTPUTS = np.array([0, 1, 1, 0])


def train_xor(seed: int) -> tuple[Network, list[float]]:
    rng = np.random.default_rng(seed)
    network = Network([DenseLayer.create(2, 8, Tanh(), rng), DenseLayer.create(8, 2, Softmax(), rng)])
    losses = network.train(XOR_INPUTS, np.eye(2)[XOR_OUTPUTS], CrossEntropy(), SGD(0.5),
                           epochs=500, batch_size=4, rng=rng)
    return network, losses


def test_the_network_learns_xor():
    network, losses = train_xor(seed=0)
    assert list(network(XOR_INPUTS).argmax(axis=1)) == list(XOR_OUTPUTS)
    assert losses[-1] < 0.05 < losses[0]


def test_xor_is_learned_from_several_random_starts():
    for seed in range(5):
        network, _ = train_xor(seed)
        assert list(network(XOR_INPUTS).argmax(axis=1)) == list(XOR_OUTPUTS), f"seed {seed}"
