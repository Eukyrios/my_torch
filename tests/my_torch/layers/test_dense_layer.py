import numpy as np
import pytest

from my_torch.activations.activation import Linear, ReLU, Sigmoid, Softmax, Tanh
from my_torch.layers.dense_layer import DenseLayer
from my_torch.layers.layer import Layer
from my_torch.layers.neuron import Neuron

WEIGHTS = [[0.5, -1.0], [0.25, 0.5]]
BIASES = [0.1, -0.3]


def test_forward_matches_the_hand_computation():
    layer = DenseLayer(WEIGHTS, BIASES)
    assert layer.forward([1, 2]) == pytest.approx([1.1, -0.3])


def test_forward_applies_the_activation():
    layer = DenseLayer(WEIGHTS, BIASES, activation=ReLU())
    assert layer.forward([1, 2]) == pytest.approx([1.1, 0.0])


def test_default_activation_is_linear():
    assert isinstance(DenseLayer(WEIGHTS, BIASES).activation, Linear)


def test_forward_keeps_its_inputs_and_z_for_backpropagation():
    layer = DenseLayer(WEIGHTS, BIASES, activation=ReLU())
    layer.forward([1, 2])
    assert layer.inputs == pytest.approx([1.0, 2.0])
    assert layer.z == pytest.approx([1.1, -0.3])


def test_softmax_layer_outputs_one_probability_row_per_sample():
    rng = np.random.default_rng(2)
    layer = DenseLayer(rng.normal(size=(6, 3)), rng.normal(size=3), activation=Softmax())
    outputs = layer.forward(rng.normal(size=(4, 6)))
    assert outputs.sum(axis=1) == pytest.approx(np.ones(4))


def test_each_output_is_the_neuron_of_its_column():
    rng = np.random.default_rng(0)
    weights = rng.normal(size=(4, 3))
    biases = rng.normal(size=3)
    inputs = rng.normal(size=4)
    outputs = DenseLayer(weights, biases).forward(inputs)
    for neuron in range(3):
        assert outputs[neuron] == pytest.approx(Neuron(weights[:, neuron], biases[neuron]).forward(inputs))


def test_shape_is_the_shape_of_the_weight_matrix():
    assert DenseLayer(np.zeros((768, 256)), np.zeros(256)).shape == (768, 256)


def test_rejects_weights_that_are_not_a_matrix():
    with pytest.raises(ValueError, match="must be a matrix"):
        DenseLayer([1, 2, 3], [0])


@pytest.mark.parametrize("biases", [[0.1], [0.1, 0.2, 0.3], [[0.1, 0.2]]])
def test_rejects_a_bias_count_that_does_not_match_the_neurons(biases):
    with pytest.raises(ValueError, match="one per neuron"):
        DenseLayer(WEIGHTS, biases)


def test_batch_gives_one_output_row_per_sample():
    outputs = DenseLayer(np.zeros((768, 3)), np.zeros(3)).forward(np.zeros((32, 768)))
    assert outputs.shape == (32, 3)


def test_each_batch_row_matches_the_single_sample_forward():
    rng = np.random.default_rng(1)
    layer = DenseLayer(rng.normal(size=(5, 4)), rng.normal(size=4))
    batch = rng.normal(size=(8, 5))
    outputs = layer.forward(batch)
    for row in range(8):
        assert outputs[row] == pytest.approx(layer.forward(batch[row]))


def test_biases_are_added_to_every_row():
    outputs = DenseLayer(np.zeros((2, 2)), BIASES).forward(np.ones((3, 2)))
    assert outputs == pytest.approx(np.tile(BIASES, (3, 1)))


def test_batch_of_one_keeps_its_batch_dimension():
    assert DenseLayer(WEIGHTS, BIASES).forward([[1, 2]]).shape == (1, 2)


@pytest.mark.parametrize("inputs", [[1], [1, 2, 3], [[1, 2, 3]], np.zeros((2, 2, 2))])
def test_rejects_the_wrong_input_shape(inputs):
    with pytest.raises(ValueError, match="expects 2 inputs per sample"):
        DenseLayer(WEIGHTS, BIASES).forward(inputs)


def test_a_dense_layer_is_a_layer():
    assert isinstance(DenseLayer(WEIGHTS, BIASES), Layer)


def test_a_dense_layer_is_named_dense():
    assert DenseLayer(WEIGHTS, BIASES).name == "dense"


def test_calling_a_layer_runs_its_forward():
    layer = DenseLayer(WEIGHTS, BIASES, activation=ReLU())
    assert layer([1, 2]) == pytest.approx(layer.forward([1, 2]))


def random_layer(activation, seed=10) -> DenseLayer:
    rng = np.random.default_rng(seed)
    return DenseLayer(rng.normal(size=(4, 3)), rng.normal(size=3), activation)


def weighted_output(layer: DenseLayer, inputs: np.ndarray, gradient: np.ndarray) -> float:
    return float(np.sum(gradient * layer.forward(inputs)))


def numerical_gradient(value, array: np.ndarray) -> np.ndarray:
    step = 1e-6
    result = np.zeros_like(array)
    for index in np.ndindex(array.shape):
        original = array[index]
        array[index] = original + step
        higher = value()
        array[index] = original - step
        lower = value()
        array[index] = original
        result[index] = (higher - lower) / (2 * step)
    return result


ACTIVATIONS = [Linear(), ReLU(), Sigmoid(), Tanh(), Softmax()]
INPUTS = np.random.default_rng(11).normal(size=(5, 4))
OUTPUT_GRADIENT = np.random.default_rng(12).normal(size=(5, 3))


@pytest.mark.parametrize("activation", ACTIVATIONS, ids=lambda activation: activation.name)
def test_weights_gradient_matches_the_numerical_gradient(activation):
    layer = random_layer(activation)
    layer.forward(INPUTS)
    layer.backward(OUTPUT_GRADIENT)
    expected = numerical_gradient(lambda: weighted_output(layer, INPUTS, OUTPUT_GRADIENT), layer.weights)
    assert layer.weights_gradient == pytest.approx(expected, rel=1e-5, abs=1e-7)


@pytest.mark.parametrize("activation", ACTIVATIONS, ids=lambda activation: activation.name)
def test_biases_gradient_matches_the_numerical_gradient(activation):
    layer = random_layer(activation)
    layer.forward(INPUTS)
    layer.backward(OUTPUT_GRADIENT)
    expected = numerical_gradient(lambda: weighted_output(layer, INPUTS, OUTPUT_GRADIENT), layer.biases)
    assert layer.biases_gradient == pytest.approx(expected, rel=1e-5, abs=1e-7)


@pytest.mark.parametrize("activation", ACTIVATIONS, ids=lambda activation: activation.name)
def test_input_gradient_matches_the_numerical_gradient(activation):
    layer = random_layer(activation)
    inputs = INPUTS.copy()
    layer.forward(inputs)
    input_gradient = layer.backward(OUTPUT_GRADIENT)
    expected = numerical_gradient(lambda: weighted_output(layer, inputs, OUTPUT_GRADIENT), inputs)
    assert input_gradient == pytest.approx(expected, rel=1e-5, abs=1e-7)


def test_gradients_have_the_shapes_of_what_they_belong_to():
    layer = random_layer(Tanh())
    layer.forward(INPUTS)
    input_gradient = layer.backward(OUTPUT_GRADIENT)
    assert layer.weights_gradient.shape == layer.weights.shape
    assert layer.biases_gradient.shape == layer.biases.shape
    assert input_gradient.shape == INPUTS.shape


def test_backward_on_a_single_sample_matches_a_batch_of_one():
    single, batch = random_layer(Tanh()), random_layer(Tanh())
    single.forward(INPUTS[0])
    batch.forward(INPUTS[:1])
    single_input_gradient = single.backward(OUTPUT_GRADIENT[0])
    batch_input_gradient = batch.backward(OUTPUT_GRADIENT[:1])
    assert single_input_gradient.shape == (4,)
    assert single_input_gradient == pytest.approx(batch_input_gradient[0])
    assert single.weights_gradient == pytest.approx(batch.weights_gradient)
    assert single.biases_gradient == pytest.approx(batch.biases_gradient)


def test_gradients_are_empty_before_any_backward():
    layer = random_layer(Tanh())
    assert layer.weights_gradient is None and layer.biases_gradient is None


def test_backward_before_forward_is_refused():
    with pytest.raises(RuntimeError, match="call forward first"):
        random_layer(Tanh()).backward(OUTPUT_GRADIENT)


def test_backward_rejects_a_gradient_of_the_wrong_shape():
    layer = random_layer(Tanh())
    layer.forward(INPUTS)
    with pytest.raises(ValueError, match=r"must have the shape of the last outputs \(5, 3\)"):
        layer.backward(OUTPUT_GRADIENT[:, :2])


def test_create_builds_a_layer_of_the_requested_size():
    layer = DenseLayer.create(769, 256, ReLU())
    assert layer.shape == (769, 256)
    assert layer.biases.shape == (256,)
    assert isinstance(layer.activation, ReLU)


def test_create_starts_with_zero_biases_and_random_weights():
    layer = DenseLayer.create(50, 40, Tanh())
    assert layer.biases == pytest.approx(np.zeros(40))
    assert len(np.unique(layer.weights)) == layer.weights.size


def test_create_uses_he_initialization_for_relu():
    weights = DenseLayer.create(800, 400, ReLU(), np.random.default_rng(0)).weights
    assert weights.mean() == pytest.approx(0.0, abs=0.005)
    assert weights.std() == pytest.approx(np.sqrt(2 / 800), rel=0.02)


def test_create_uses_xavier_initialization_otherwise():
    weights = DenseLayer.create(800, 400, Tanh(), np.random.default_rng(0)).weights
    assert weights.std() == pytest.approx(np.sqrt(2 / 1200), rel=0.02)


def test_create_defaults_to_a_linear_activation():
    assert isinstance(DenseLayer.create(3, 2).activation, Linear)


def test_create_is_reproducible_with_the_same_random_generator():
    first = DenseLayer.create(5, 4, Tanh(), np.random.default_rng(42))
    second = DenseLayer.create(5, 4, Tanh(), np.random.default_rng(42))
    assert first.weights == pytest.approx(second.weights)


@pytest.mark.parametrize("n_inputs, n_outputs", [(0, 3), (3, 0), (-1, 2)])
def test_create_rejects_sizes_below_one(n_inputs, n_outputs):
    with pytest.raises(ValueError, match="at least 1 input and 1 output"):
        DenseLayer.create(n_inputs, n_outputs)
