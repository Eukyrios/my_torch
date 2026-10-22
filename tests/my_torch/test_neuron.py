import pytest

from my_torch.neuron import Neuron


def test_forward_matches_the_hand_computation():
    assert Neuron([0.5, 0.25], 0.1).forward([1, 2]) == pytest.approx(1.1)


def test_bias_alone_when_every_input_is_zero():
    assert Neuron([3, -2, 7], -0.5).forward([0, 0, 0]) == pytest.approx(-0.5)


def test_negative_weights_lower_the_sum():
    assert Neuron([-1, 0.5], -0.3).forward([1, 2]) == pytest.approx(-0.3)


def test_forward_returns_a_plain_float():
    assert isinstance(Neuron([1], 0).forward([1]), float)


def test_n_inputs_is_the_number_of_weights():
    assert Neuron([1, 2, 3], 0).n_inputs == 3


def test_keeps_its_weights_and_bias():
    neuron = Neuron([1, 2], 3)
    assert neuron.weights == pytest.approx([1.0, 2.0])
    assert neuron.bias == 3.0


def test_rejects_weights_that_are_not_a_vector():
    with pytest.raises(ValueError, match="must be a vector"):
        Neuron([[1, 2]], 0)


@pytest.mark.parametrize("inputs", [[1], [1, 2, 3], [[1, 2]]])
def test_rejects_inputs_that_do_not_match_the_weights(inputs):
    with pytest.raises(ValueError, match="one input per weight"):
        Neuron([1, 2], 0).forward(inputs)
