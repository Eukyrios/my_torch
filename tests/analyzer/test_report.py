import pytest

from analyzer.prediction import LABELS
from analyzer.report import build_report, expected_as, format_report, percent

BASIC = ("Nothing", "Check", "Checkmate")

EXPECTED = ["Nothing", "Nothing", "Check White", "Check Black", "Checkmate White", "Checkmate Black"]
PREDICTED = ["Nothing", "Check White", "Check White", "Check White", "Checkmate White", "Nothing"]


def test_counts_every_expected_and_predicted_pair():
    report = build_report(EXPECTED, PREDICTED)
    assert report.labels == LABELS
    assert report.confusion == (
        (1, 1, 0, 0, 0),
        (0, 1, 0, 0, 0),
        (0, 1, 0, 0, 0),
        (0, 0, 0, 1, 0),
        (1, 0, 0, 0, 0),
    )


def test_accuracy_is_the_share_of_boards_predicted_right():
    report = build_report(EXPECTED, PREDICTED)
    assert (report.correct, report.total) == (3, 6)
    assert report.accuracy == 0.5


def test_precision_and_recall_of_each_label():
    report = build_report(EXPECTED, PREDICTED)
    check_white = LABELS.index("Check White")
    assert report.precision(check_white) == pytest.approx(1 / 3)
    assert report.recall(check_white) == 1.0
    assert report.precision(LABELS.index("Nothing")) == 0.5
    assert report.recall(LABELS.index("Nothing")) == 0.5


def test_a_label_never_predicted_or_never_expected_has_no_score():
    report = build_report(EXPECTED, PREDICTED)
    assert report.precision(LABELS.index("Check Black")) is None
    assert report.precision(LABELS.index("Checkmate Black")) is None
    assert report.recall(LABELS.index("Checkmate Black")) == 0.0
    assert build_report([], []).accuracy is None


def test_a_network_without_colours_is_judged_on_the_state():
    report = build_report(["Check White", "Checkmate Black", "Nothing"], ["Check", "Check", "Nothing"], BASIC)
    assert report.confusion == ((1, 0, 0), (0, 1, 0), (0, 1, 0))
    assert report.accuracy == pytest.approx(2 / 3)


def test_expected_labels_must_match_the_network():
    assert expected_as("Check White", LABELS) == "Check White"
    assert expected_as("Check White", BASIC) == "Check"
    assert expected_as("Check", BASIC) == "Check"
    with pytest.raises(ValueError, match="expected output 'Check' is not one the network predicts"):
        expected_as("Check", LABELS)
    with pytest.raises(ValueError, match="unknown label 'Draw'"):
        build_report(["Draw"], ["Nothing"])
    with pytest.raises(ValueError, match="expected output 'Stalemate' is not one the network predicts"):
        build_report(["Stalemate"], ["Nothing"])


def test_predictions_must_be_labels_of_the_network():
    with pytest.raises(ValueError, match="predicted output 'Check' is not one of"):
        build_report(["Nothing"], ["Check"])


def test_needs_one_prediction_per_board():
    with pytest.raises(ValueError, match="got 1 for 2 boards"):
        build_report(["Nothing", "Nothing"], ["Nothing"])


def test_percent():
    assert percent(0.5) == "50.0%"
    assert percent(1 / 3) == "33.3%"
    assert percent(None) == "n/a"


def test_format_report():
    text = format_report(build_report(["Check White", "Checkmate Black", "Nothing", "Nothing"], ["Check", "Check", "Nothing", "Nothing"], BASIC))
    assert text == "\n".join([
        "accuracy: 75.0% (3/4 boards)",
        "",
        "confusion matrix (rows: expected, columns: predicted)",
        "expected \\ predicted  Nothing  Check  Checkmate",
        "Nothing                     2      0          0",
        "Check                       0      1          0",
        "Checkmate                   0      1          0",
        "",
        "label      precision  recall  boards",
        "Nothing       100.0%  100.0%       2",
        "Check          50.0%  100.0%       1",
        "Checkmate        n/a    0.0%       1",
    ])


def test_format_report_has_one_line_per_label():
    lines = format_report(build_report(EXPECTED, PREDICTED)).splitlines()
    assert lines[0] == "accuracy: 50.0% (3/6 boards)"
    assert lines[3].split()[-2:] == ["Checkmate", "Black"]
    assert len(lines) == 2 + 1 + 6 + 1 + 6
