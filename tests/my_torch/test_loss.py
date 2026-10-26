import numpy as np
import pytest

from my_torch.loss import CrossEntropy, Loss, MeanSquaredError

PREDICTIONS = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
EXPECTED = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def numerical_gradient(loss: Loss, predictions: np.ndarray, expected: np.ndarray) -> np.ndarray:
    step = 1e-6
    gradient = np.zeros_like(predictions)
    for index in np.ndindex(predictions.shape):
        higher, lower = predictions.copy(), predictions.copy()
        higher[index] += step
        lower[index] -= step
        gradient[index] = (loss(higher, expected) - loss(lower, expected)) / (2 * step)
    return gradient


def test_cross_entropy_matches_the_hand_computation():
    expected_loss = -(np.log(0.7) + np.log(0.8)) / 2
    assert CrossEntropy()(PREDICTIONS, EXPECTED) == pytest.approx(expected_loss)


def test_cross_entropy_is_zero_for_perfect_predictions():
    assert CrossEntropy()(EXPECTED, EXPECTED) == pytest.approx(0.0)


def test_cross_entropy_grows_when_the_right_class_gets_less_probability():
    confident = CrossEntropy()([0.9, 0.05, 0.05], [1, 0, 0])
    unsure = CrossEntropy()([0.4, 0.3, 0.3], [1, 0, 0])
    assert unsure > confident


def test_cross_entropy_stays_finite_when_the_right_class_gets_zero():
    with np.errstate(all="raise"):
        assert np.isfinite(CrossEntropy()([0.0, 1.0, 0.0], [1, 0, 0]))


def test_cross_entropy_gradient_matches_the_numerical_gradient():
    loss = CrossEntropy()
    assert loss.gradient(PREDICTIONS, EXPECTED) == pytest.approx(
        numerical_gradient(loss, PREDICTIONS, EXPECTED), rel=1e-4
    )


def test_mean_squared_error_matches_the_hand_computation():
    assert MeanSquaredError()([1.0, 2.0], [0.0, 4.0]) == pytest.approx((1 + 4) / 2)


def test_mean_squared_error_is_zero_for_perfect_predictions():
    assert MeanSquaredError()(PREDICTIONS, PREDICTIONS) == pytest.approx(0.0)


def test_mean_squared_error_gradient_matches_the_numerical_gradient():
    loss = MeanSquaredError()
    assert loss.gradient(PREDICTIONS, EXPECTED) == pytest.approx(
        numerical_gradient(loss, PREDICTIONS, EXPECTED), rel=1e-4
    )


@pytest.mark.parametrize("loss", [CrossEntropy(), MeanSquaredError()])
def test_the_loss_is_averaged_over_the_batch(loss):
    one = loss(PREDICTIONS[:1], EXPECTED[:1])
    doubled = loss(np.vstack([PREDICTIONS[:1]] * 2), np.vstack([EXPECTED[:1]] * 2))
    assert doubled == pytest.approx(one)


@pytest.mark.parametrize("loss", [CrossEntropy(), MeanSquaredError()])
def test_a_single_sample_is_a_batch_of_one(loss):
    assert loss(PREDICTIONS[0], EXPECTED[0]) == pytest.approx(loss(PREDICTIONS[:1], EXPECTED[:1]))
    assert loss.gradient(PREDICTIONS[0], EXPECTED[0]).shape == (1, 3)


@pytest.mark.parametrize("loss", [CrossEntropy(), MeanSquaredError()])
def test_the_gradient_has_the_shape_of_the_predictions(loss):
    assert loss.gradient(PREDICTIONS, EXPECTED).shape == PREDICTIONS.shape


@pytest.mark.parametrize("loss", [CrossEntropy(), MeanSquaredError()])
def test_rejects_predictions_and_expected_of_different_shapes(loss):
    with pytest.raises(ValueError, match="must have the same shape"):
        loss(PREDICTIONS, EXPECTED[:, :2])


@pytest.mark.parametrize("loss, name", [(CrossEntropy(), "cross_entropy"), (MeanSquaredError(), "mean_squared_error")])
def test_each_loss_has_its_name(loss, name):
    assert isinstance(loss, Loss)
    assert loss.name == name


def test_the_base_class_cannot_be_used_on_its_own():
    with pytest.raises(TypeError):
        Loss("loss")


def test_the_base_methods_are_abstract():
    with pytest.raises(NotImplementedError):
        Loss.forward(CrossEntropy(), PREDICTIONS, EXPECTED)
    with pytest.raises(NotImplementedError):
        Loss.gradient(CrossEntropy(), PREDICTIONS, EXPECTED)
