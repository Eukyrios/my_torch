import numpy as np
import pytest

from my_torch.layer import Dense
from my_torch.neuron import weighted_sum

WEIGHTS = [[0.5, -1.0], [0.25, 0.5]]
BIASES = [0.1, -0.3]


def test_forward_matches_the_hand_computation():
    layer = Dense(WEIGHTS, BIASES)
    assert layer.forward([1, 2]) == pytest.approx([1.1, -0.3])


def test_each_output_is_the_neuron_of_its_column():
    rng = np.random.default_rng(0)
    weights = rng.normal(size=(4, 3))
    biases = rng.normal(size=3)
    inputs = rng.normal(size=4)
    outputs = Dense(weights, biases).forward(inputs)
    for neuron in range(3):
        assert outputs[neuron] == pytest.approx(weighted_sum(inputs, weights[:, neuron], biases[neuron]))


def test_sizes_come_from_the_weight_matrix():
    layer = Dense(np.zeros((768, 256)), np.zeros(256))
    assert (layer.n_inputs, layer.n_outputs) == (768, 256)


def test_rejects_weights_that_are_not_a_matrix():
    with pytest.raises(ValueError, match="must be a matrix"):
        Dense([1, 2, 3], [0])


@pytest.mark.parametrize("biases", [[0.1], [0.1, 0.2, 0.3], [[0.1, 0.2]]])
def test_rejects_a_bias_count_that_does_not_match_the_neurons(biases):
    with pytest.raises(ValueError, match="one per neuron"):
        Dense(WEIGHTS, biases)


@pytest.mark.parametrize("inputs", [[1], [1, 2, 3]])
def test_rejects_the_wrong_number_of_inputs(inputs):
    with pytest.raises(ValueError, match="expects 2 inputs"):
        Dense(WEIGHTS, BIASES).forward(inputs)
