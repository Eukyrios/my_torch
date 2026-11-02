import numpy as np
import pytest

from analyzer.prediction import LABELS, STATES, predicted_labels, predicted_states, state_of


def one_hot(label: str) -> list[float]:
    return list(np.eye(len(LABELS))[LABELS.index(label)])


def test_labels_follow_the_dataset_order():
    from dataset.labels import LABELS as DATASET_LABELS

    assert LABELS == DATASET_LABELS


def test_every_label_maps_to_its_state():
    states = [state_of(label) for label in LABELS]
    assert states == ["Nothing", "Check", "Check", "Checkmate", "Checkmate"]
    assert set(states) == set(STATES)


def test_unknown_label_is_rejected():
    with pytest.raises(ValueError, match="unknown label 'Stalemate'"):
        state_of("Stalemate")


def test_the_highest_output_wins():
    assert predicted_labels([0.1, 0.05, 0.05, 0.1, 0.7]) == ["Checkmate Black"]
    assert predicted_states([0.1, 0.05, 0.05, 0.1, 0.7]) == ["Checkmate"]


def test_one_state_per_board_in_the_same_order():
    outputs = [one_hot(label) for label in ("Check White", "Nothing", "Checkmate White", "Check Black")]
    assert predicted_states(outputs) == ["Check", "Nothing", "Checkmate", "Check"]


def test_scores_do_not_need_to_be_probabilities():
    assert predicted_states([[-3.0, 2.5, -1.0, 0.0, 1.0]]) == ["Check"]


def test_a_network_with_its_own_labels():
    basic = ("Nothing", "Check", "Checkmate")
    assert predicted_labels([[0.2, 0.1, 0.7], [0.9, 0.05, 0.05]], basic) == ["Checkmate", "Nothing"]
    assert predicted_states([0.1, 0.8, 0.1], basic) == ["Check"]


def test_no_board_gives_no_state():
    assert predicted_states(np.empty((0, len(LABELS)))) == []


@pytest.mark.parametrize("outputs", [[0.5, 0.5], np.zeros((2, 3)), np.zeros((2, 2, 5))])
def test_outputs_must_have_one_score_per_label(outputs):
    with pytest.raises(ValueError, match="expected 5 outputs per board"):
        predicted_states(outputs)
