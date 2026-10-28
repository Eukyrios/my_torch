# Training mode of the analyzer. For now it trains a small network on XOR, without reading
# the chess file or loading and saving networks, to check the training path end to end.

import numpy as np

from my_torch.activations.activation import Softmax, Tanh
from my_torch.layers.dense_layer import DenseLayer
from my_torch.losses.loss import CrossEntropy
from my_torch.network.network import Network
from my_torch.optimizers.optimizer import SGD

XOR_INPUTS = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
XOR_OUTPUTS = np.array([0, 1, 1, 0])


def build_xor_network(rng: np.random.Generator) -> Network:
    return Network([DenseLayer.create(2, 8, Tanh(), rng), DenseLayer.create(8, 2, Softmax(), rng)])


def train_xor(epochs: int = 500, seed: int = 0) -> tuple[Network, list[float]]:
    rng = np.random.default_rng(seed)
    network = build_xor_network(rng)
    losses = network.train(XOR_INPUTS, np.eye(2)[XOR_OUTPUTS], CrossEntropy(), SGD(0.5), epochs=epochs, batch_size=4, rng=rng)
    return network, losses


def xor_report(network: Network, losses: list[float]) -> str:
    lines = [f"trained on XOR for {len(losses)} epochs: loss {losses[0]:.3f} -> {losses[-1]:.3f}"]
    for inputs, probabilities, expected in zip(XOR_INPUTS, network(XOR_INPUTS), XOR_OUTPUTS):
        predicted = int(probabilities.argmax())
        status = "ok" if predicted == expected else "wrong"
        lines.append(f"{int(inputs[0])} xor {int(inputs[1])} = {predicted} (expected {expected}, confidence {probabilities.max():.1%}) {status}")
    return "\n".join(lines)
