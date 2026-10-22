import pytest

from my_torch.neuron import weighted_sum


def test_weighted_sum_matches_the_hand_computation():
    assert weighted_sum([1, 2], [0.5, 0.25], 0.1) == pytest.approx(1.1)


def test_bias_alone_when_every_input_is_zero():
    assert weighted_sum([0, 0, 0], [3, -2, 7], -0.5) == pytest.approx(-0.5)


def test_negative_weights_lower_the_sum():
    assert weighted_sum([1, 2], [-1, 0.5], -0.3) == pytest.approx(-0.3)


def test_returns_a_plain_float():
    assert isinstance(weighted_sum([1], [1], 0), float)


@pytest.mark.parametrize(
    "inputs, weights",
    [
        ([1, 2, 3], [1, 2]),
        ([1, 2], [1, 2, 3]),
        ([[1, 2]], [[1, 2]]),
    ],
)
def test_rejects_inputs_and_weights_that_do_not_match(inputs, weights):
    with pytest.raises(ValueError, match="one weight per input"):
        weighted_sum(inputs, weights, 0)
