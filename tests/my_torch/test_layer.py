import numpy as np
import pytest

from my_torch.layer import Dense
from my_torch.neuron import weighted_sum

WEIGHTS = [[0.5, -1.0], [0.25, 0.5]]
BIASES = [0.1, -0.3]


def test_forward_matches_the_hand_computation():
    layer = Dense(WEIGHTS, BIASES)
    assert layer.forward([1, 2]) == pytest.approx([1.1, -0.3])


def test_forward_applies_the_activation():
    layer = Dense(WEIGHTS, BIASES, activation="relu")
    assert layer.forward([1, 2]) == pytest.approx([1.1, 0.0])


def test_default_activation_is_linear():
    assert Dense(WEIGHTS, BIASES).activation_name == "linear"


def test_forward_keeps_its_inputs_and_z_for_backpropagation():
    layer = Dense(WEIGHTS, BIASES, activation="relu")
    layer.forward([1, 2])
    assert layer.inputs == pytest.approx([1.0, 2.0])
    assert layer.z == pytest.approx([1.1, -0.3])


def test_softmax_layer_outputs_one_probability_row_per_sample():
    rng = np.random.default_rng(2)
    layer = Dense(rng.normal(size=(6, 3)), rng.normal(size=3), activation="softmax")
    outputs = layer.forward(rng.normal(size=(4, 6)))
    assert outputs.sum(axis=1) == pytest.approx(np.ones(4))


def test_rejects_an_unknown_activation():
    with pytest.raises(ValueError, match="unknown activation"):
        Dense(WEIGHTS, BIASES, activation="swish")


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


def test_batch_gives_one_output_row_per_sample():
    outputs = Dense(np.zeros((768, 3)), np.zeros(3)).forward(np.zeros((32, 768)))
    assert outputs.shape == (32, 3)


def test_each_batch_row_matches_the_single_sample_forward():
    rng = np.random.default_rng(1)
    layer = Dense(rng.normal(size=(5, 4)), rng.normal(size=4))
    batch = rng.normal(size=(8, 5))
    outputs = layer.forward(batch)
    for row in range(8):
        assert outputs[row] == pytest.approx(layer.forward(batch[row]))


def test_biases_are_added_to_every_row():
    outputs = Dense(np.zeros((2, 2)), BIASES).forward(np.ones((3, 2)))
    assert outputs == pytest.approx(np.tile(BIASES, (3, 1)))


def test_batch_of_one_keeps_its_batch_dimension():
    assert Dense(WEIGHTS, BIASES).forward([[1, 2]]).shape == (1, 2)


@pytest.mark.parametrize("inputs", [[1], [1, 2, 3], [[1, 2, 3]], np.zeros((2, 2, 2))])
def test_rejects_the_wrong_input_shape(inputs):
    with pytest.raises(ValueError, match="expects 2 inputs per sample"):
        Dense(WEIGHTS, BIASES).forward(inputs)
