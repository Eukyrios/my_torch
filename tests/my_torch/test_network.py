import numpy as np
import pytest

from my_torch.activations import ReLU, Softmax
from my_torch.layer import DenseLayer, Layer
from my_torch.network import Network


def hidden_layer() -> DenseLayer:
    return DenseLayer([[0.5, -1.0], [0.25, 0.5]], [0.1, -0.3], ReLU())


def output_layer() -> DenseLayer:
    return DenseLayer([[1.0, 0.5, -1.0], [0.0, 1.0, 1.0]], [0.0, 0.0, 0.0], Softmax())


def random_network(sizes: list[int]) -> Network:
    rng = np.random.default_rng(0)
    layers = [
        DenseLayer(rng.normal(size=(n_in, n_out)), rng.normal(size=n_out), ReLU())
        for n_in, n_out in zip(sizes, sizes[1:])
    ]
    return Network(layers)


class PassThrough(Layer):
    def __init__(self):
        super().__init__("pass_through")

    def forward(self, inputs):
        return inputs


def test_forward_matches_the_hand_computation():
    network = Network([hidden_layer(), output_layer()])
    assert network.forward([1, 2]) == pytest.approx([0.593, 0.342, 0.066], abs=1e-3)


def test_forward_runs_the_layers_in_order():
    hidden, output = hidden_layer(), output_layer()
    inputs = np.array([1.0, 2.0])
    assert Network([hidden, output]).forward(inputs) == pytest.approx(output(hidden(inputs)))


def test_calling_a_network_runs_its_forward():
    network = Network([hidden_layer(), output_layer()])
    assert network([1, 2]) == pytest.approx(network.forward([1, 2]))


def test_a_batch_gives_one_output_row_per_sample():
    network = random_network([768, 64, 3])
    assert network(np.zeros((32, 768))).shape == (32, 3)


def test_each_batch_row_matches_the_single_sample_forward():
    network = random_network([5, 4, 3])
    batch = np.random.default_rng(1).normal(size=(6, 5))
    outputs = network(batch)
    for row in range(6):
        assert outputs[row] == pytest.approx(network(batch[row]))


def test_softmax_output_gives_probabilities():
    network = Network([hidden_layer(), output_layer()])
    batch = np.array([[1.0, 2.0], [0.0, 0.0], [3.0, -1.0]])
    assert network(batch).sum(axis=1) == pytest.approx(np.ones(3))


def test_shape_comes_from_the_first_and_last_sized_layers():
    assert random_network([768, 256, 64, 3]).shape == (768, 3)


def test_keeps_its_layers_in_order():
    hidden, output = hidden_layer(), output_layer()
    assert Network([hidden, output]).layers == [hidden, output]


def test_a_layer_without_weights_can_sit_between_dense_layers():
    hidden, output = hidden_layer(), output_layer()
    network = Network([hidden, PassThrough(), output])
    assert network.shape == (2, 3)
    assert network([1, 2]) == pytest.approx(Network([hidden, output])([1, 2]))


def test_rejects_a_network_without_layers():
    with pytest.raises(ValueError, match="at least one layer with a fixed size"):
        Network([])


def test_rejects_a_network_without_a_layer_of_fixed_size():
    with pytest.raises(ValueError, match="at least one layer with a fixed size"):
        Network([PassThrough()])


def test_rejects_something_that_is_not_a_layer():
    with pytest.raises(TypeError, match="made of layers, got list"):
        Network([hidden_layer(), [1, 2, 3]])


def test_rejects_layers_whose_sizes_do_not_chain():
    first = DenseLayer(np.zeros((768, 64)), np.zeros(64))
    second = DenseLayer(np.zeros((32, 3)), np.zeros(3))
    with pytest.raises(ValueError, match="layer 1 expects 32 inputs but receives 64"):
        Network([first, second])


def test_rejects_inputs_of_the_wrong_size():
    with pytest.raises(ValueError, match="expects 2 inputs per sample"):
        Network([hidden_layer(), output_layer()])([1, 2, 3])


def test_size_errors_name_the_positions_in_the_whole_network():
    first = DenseLayer(np.zeros((768, 64)), np.zeros(64))
    second = DenseLayer(np.zeros((32, 3)), np.zeros(3))
    with pytest.raises(ValueError, match="layer 2 expects 32 inputs but receives 64"):
        Network([first, PassThrough(), second])


def test_shape_skips_layers_without_a_fixed_size():
    hidden, output = hidden_layer(), output_layer()
    network = Network([PassThrough(), hidden, PassThrough(), output, PassThrough()])
    assert network.shape == (2, 3)
