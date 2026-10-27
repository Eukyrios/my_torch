import numpy as np
import pytest

from my_torch.activations.activation import Softmax, Tanh
from my_torch.layers.dense_layer import DenseLayer
from my_torch.losses.loss import CrossEntropy
from my_torch.network.network import Network, shuffled_batches
from my_torch.optimizers.optimizer import SGD

RNG = np.random.default_rng(30)
INPUTS = RNG.normal(size=(20, 4))
LABELS = np.eye(3)[(INPUTS[:, 0] > 0).astype(int) + (INPUTS[:, 1] > 0).astype(int)]


def small_network(seed: int = 31) -> Network:
    rng = np.random.default_rng(seed)
    return Network([DenseLayer.create(4, 8, Tanh(), rng), DenseLayer.create(8, 3, Softmax(), rng)])


def test_train_returns_one_mean_loss_per_epoch():
    losses = small_network().train(INPUTS, LABELS, CrossEntropy(), SGD(0.1), epochs=5, batch_size=4)
    assert len(losses) == 5 and all(np.isfinite(losses))


def test_the_loss_goes_down_while_training():
    losses = small_network().train(INPUTS, LABELS, CrossEntropy(), SGD(0.1), epochs=200, batch_size=4,
                                   rng=np.random.default_rng(0))
    assert losses[-1] < losses[0] / 2


def test_the_first_epoch_loss_is_the_mean_over_all_samples_with_one_full_batch():
    network = small_network()
    expected = CrossEntropy()(network(INPUTS), LABELS)
    losses = network.train(INPUTS, LABELS, CrossEntropy(), SGD(0.1), epochs=1, batch_size=len(INPUTS))
    assert losses[0] == pytest.approx(expected)


def test_training_is_reproducible_with_the_same_seeds():
    first = small_network().train(INPUTS, LABELS, CrossEntropy(), SGD(0.1), 10, 4, np.random.default_rng(5))
    second = small_network().train(INPUTS, LABELS, CrossEntropy(), SGD(0.1), 10, 4, np.random.default_rng(5))
    assert first == pytest.approx(second)


@pytest.mark.parametrize("batch_size", [1, 3, 20, 50])
def test_any_batch_size_trains_on_every_sample(batch_size):
    losses = small_network().train(INPUTS, LABELS, CrossEntropy(), SGD(0.05), epochs=2, batch_size=batch_size)
    assert len(losses) == 2


def test_shuffled_batches_cover_every_sample_once():
    batches = shuffled_batches(10, 3, np.random.default_rng(0))
    assert [len(batch) for batch in batches] == [3, 3, 3, 1]
    assert sorted(np.concatenate(batches)) == list(range(10))


def test_shuffled_batches_change_order_between_calls():
    rng = np.random.default_rng(0)
    assert not np.array_equal(np.concatenate(shuffled_batches(10, 10, rng)), np.concatenate(shuffled_batches(10, 10, rng)))


def test_rejects_a_different_number_of_samples_and_labels():
    with pytest.raises(ValueError, match="one label per sample, got 20 samples and 19 labels"):
        small_network().train(INPUTS, LABELS[:19], CrossEntropy(), SGD())


@pytest.mark.parametrize("epochs, batch_size", [(0, 4), (3, 0)])
def test_rejects_epochs_or_batch_size_below_one(epochs, batch_size):
    with pytest.raises(ValueError, match="must be at least 1"):
        small_network().train(INPUTS, LABELS, CrossEntropy(), SGD(), epochs=epochs, batch_size=batch_size)
