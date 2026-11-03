import pytest

from analyzer.prediction import COLOURS, LABELS, STATES, colour_of, is_label, predicted_labels
from analyzer.prediction import predicted_states, prediction_report, state_of


def one_hot(label: str) -> list[float]:
    return [1.0 if other == label else 0.0 for other in LABELS]


def test_labels_follow_the_dataset_order():
    from dataset.labels import LABELS as DATASET_LABELS

    assert LABELS == DATASET_LABELS


def test_every_label_maps_to_its_state():
    states = [state_of(label) for label in LABELS]
    assert states == ["Nothing", "Check", "Check", "Checkmate", "Checkmate"]
    assert set(states) == set(STATES)


def test_every_label_maps_to_its_colour():
    colours = [colour_of(label) for label in LABELS]
    assert colours == [None, "White", "Black", "White", "Black"]
    assert set(colours) - {None} == set(COLOURS)


@pytest.mark.parametrize("label", ["Nothing", "Check", "Checkmate", "Check White", "Checkmate Black"])
def test_valid_labels(label):
    assert is_label(label)


@pytest.mark.parametrize(
    "label",
    ["Stalemate", "", "check White", "Check Green", "Nothing White", "Check  White", "Check White Black", "Check White "],
)
def test_invalid_labels(label):
    assert not is_label(label)
    with pytest.raises(ValueError, match="unknown label"):
        state_of(label)
    with pytest.raises(ValueError, match="unknown label"):
        colour_of(label)


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


def test_a_network_with_an_invalid_label_is_rejected():
    with pytest.raises(ValueError, match="unknown label 'Check Green'"):
        predicted_labels([0.5, 0.5], ("Nothing", "Check Green"))


def test_report_gives_the_full_label_with_its_colour():
    outputs = [one_hot(label) for label in ("Checkmate Black", "Nothing", "Nothing", "Check White", "Nothing")]
    assert prediction_report(outputs) == "Checkmate Black\nNothing\nNothing\nCheck White\nNothing"


def test_report_of_a_network_without_colours():
    basic = ("Nothing", "Check", "Checkmate")
    assert prediction_report([[0.1, 0.8, 0.1], [0.9, 0.0, 0.1]], basic) == "Check\nNothing"


def test_integer_scores_and_tuples_are_accepted():
    assert predicted_labels((0, 0, 1, 0, 0)) == ["Check Black"]


def test_a_tie_gives_the_first_label():
    assert predicted_labels([0.5, 0.5, 0.0, 0.0, 0.0]) == ["Nothing"]


def test_no_board_gives_no_state():
    assert predicted_states([]) == []
    assert prediction_report([]) == ""


@pytest.mark.parametrize(
    "outputs",
    [
        [0.5, 0.5],
        [[0.0] * 3, [0.0] * 3],
        [[[0.0] * 5] * 2] * 2,
        [0.5, [0.0] * 5],
        [[0.0, 0.0, "1", 0.0, 0.0]],
        [[True, False, False, False, False]],
    ],
)
def test_outputs_must_have_one_score_per_label(outputs):
    with pytest.raises(ValueError, match="expected 5 scores per board"):
        predicted_states(outputs)
