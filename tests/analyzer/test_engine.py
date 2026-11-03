import zipfile

import pytest

import analyzer.engine as engine
from analyzer.chess_file import Board
from analyzer.encoder import active_inputs
from analyzer.engine import EngineError, boards_text, network_labels, parse_outputs, predict_outputs, train
from analyzer.fen import parse_fen
from analyzer.prediction import LABELS

from tests.analyzer.conftest import BLACK_TO_MOVE, WHITE_TO_MOVE, write_network

BOARDS = [Board(parse_fen(WHITE_TO_MOVE), None, 1), Board(parse_fen(BLACK_TO_MOVE), "Nothing", 2)]


def test_boards_are_sent_as_input_indices():
    white = " ".join(map(str, active_inputs(BOARDS[0].position)))
    black = " ".join(map(str, active_inputs(BOARDS[1].position)))
    assert boards_text(BOARDS, 5) == f"769 5\n- {white}\n- {black}\n"
    assert boards_text(BOARDS, 5, [3, 0]) == f"769 5\n3 {white}\n0 {black}\n"
    assert boards_text([], 3) == "769 3\n"


def test_outputs_are_read_one_line_per_board():
    assert parse_outputs("0.25 0.75\n1 0\n") == [[0.25, 0.75], [1.0, 0.0]]
    assert parse_outputs("") == []
    with pytest.raises(EngineError, match="not a list of outputs"):
        parse_outputs("USAGE\n")


def test_predict_gives_the_outputs_of_the_network(side_network):
    outputs = predict_outputs(side_network, BOARDS, LABELS)
    assert len(outputs) == 2
    assert max(range(5), key=outputs[0].__getitem__) == LABELS.index("Check White")
    assert max(range(5), key=outputs[1].__getitem__) == LABELS.index("Check Black")
    assert sum(outputs[0]) == pytest.approx(1.0)


def test_engine_errors_carry_its_message(tmp_path, side_network):
    with pytest.raises(EngineError, match="cannot open the network file"):
        predict_outputs(str(tmp_path / "missing.nn"), BOARDS, LABELS)
    with pytest.raises(EngineError, match="every board needs its class"):
        engine.run_engine(["train", side_network, side_network], "769 5\n- 1\n0 2\n", capture_output=True)


def test_a_missing_engine_is_reported(monkeypatch, tmp_path):
    monkeypatch.setattr(engine, "ENGINE", tmp_path / "my_torch_engine")
    with pytest.raises(EngineError, match="my_torch_engine is missing, build it with make"):
        predict_outputs("net.nn", BOARDS, LABELS)


def test_train_writes_the_trained_network(tmp_path, side_network):
    saved = tmp_path / "trained.nn"
    train(side_network, str(saved), BOARDS * 5, [0, 0] * 5, LABELS)
    assert "training" in zipfile.ZipFile(saved).read("header.json").decode()


def test_labels_come_from_the_metadata_or_default_to_the_5_classes(tmp_path):
    basic = ["Nothing", "Check", "Checkmate"]
    weights = [[0.0] * 3]
    assert network_labels(write_network(tmp_path / "basic.nn", weights, [0.0] * 3, {"labels": basic})) == tuple(basic)
    assert network_labels(write_network(tmp_path / "plain.nn", weights, [0.0] * 3)) == LABELS
    assert network_labels(write_network(tmp_path / "bad.nn", weights, [0.0] * 3, {"labels": ["Stalemate"]})) == LABELS
    assert network_labels(write_network(tmp_path / "empty.nn", weights, [0.0] * 3, {"labels": []})) == LABELS
    assert network_labels(str(tmp_path / "missing.nn")) == LABELS
    (tmp_path / "text.nn").write_text("not a zip")
    assert network_labels(str(tmp_path / "text.nn")) == LABELS
