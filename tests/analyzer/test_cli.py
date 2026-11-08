import subprocess
import sys
from pathlib import Path

import pytest

from analyzer.cli import EXIT_ERROR, EXIT_SUCCESS, USAGE, Arguments, UsageError, main, parse_args

from tests.analyzer.conftest import BLACK_TO_MOVE, WHITE_TO_MOVE, side_network, write_boards

ROOT = Path(__file__).resolve().parents[2]


PARSE_ARGS_ACCEPTS_THE_SUBJECT_USAGE_CASES = [
    (["--predict", "net.nn", "boards.txt"], Arguments("predict", "net.nn", "boards.txt")),
    (["--train", "net.nn", "boards.txt"], Arguments("train", "net.nn", "boards.txt")),
    (["--train", "--save", "out.nn", "net.nn", "boards.txt"], Arguments("train", "net.nn", "boards.txt", "out.nn")),
    (["net.nn", "--train", "boards.txt", "--save", "out.nn"], Arguments("train", "net.nn", "boards.txt", "out.nn")),
]


def test_parse_args_accepts_the_subject_usage():
    for argv, expected in PARSE_ARGS_ACCEPTS_THE_SUBJECT_USAGE_CASES:
        assert parse_args(argv) == expected


PARSE_ARGS_REJECTS_BAD_LINES_CASES = [
    ([], "expected --predict or --train"),
    (["net.nn", "boards.txt"], "expected --predict or --train"),
    (["--predict", "--train", "net.nn", "boards.txt"], "only one of"),
    (["--predict", "--save", "out.nn", "net.nn", "boards.txt"], "only works in train mode"),
    (["--train", "net.nn", "boards.txt", "--save"], "needs a SAVEFILE"),
    (["--train", "--save", "a.nn", "--save", "b.nn", "net.nn", "boards.txt"], "given twice"),
    (["--predict", "net.nn"], "expected LOADFILE and CHESSFILE"),
    (["--predict", "net.nn", "boards.txt", "extra"], "expected LOADFILE and CHESSFILE"),
    (["--verbose", "--predict", "net.nn", "boards.txt"], "unknown option --verbose"),
]


def test_parse_args_rejects_bad_lines():
    for argv, message in PARSE_ARGS_REJECTS_BAD_LINES_CASES:
        with pytest.raises(UsageError, match=message):
            parse_args(argv)


def test_help_prints_usage_and_exits_0(capsys):
    for flag in ["--help", "-h"]:
        assert main([flag]) == 0
        assert capsys.readouterr().out.strip() == USAGE


def test_usage_error_goes_to_stderr_and_exits_84(capsys):
    assert main(["--predict"]) == EXIT_ERROR
    out, err = capsys.readouterr()
    assert out == ""
    assert err.startswith("my_torch_analyzer: ")




def test_main_reads_sys_argv(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["my_torch_analyzer", "--help"])
    assert main() == 0


def test_launcher_runs_as_a_program():
    run = subprocess.run(
        [sys.executable, str(ROOT / "my_torch_analyzer.py"), "--help"],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert run.returncode == 0
    assert run.stdout.startswith("USAGE")


def test_launcher_predicts_like_in_the_subject(tmp_path):
    network_file = side_network(tmp_path)
    boards = write_boards(tmp_path, [WHITE_TO_MOVE, BLACK_TO_MOVE])
    run = subprocess.run(
        [sys.executable, str(ROOT / "my_torch_analyzer.py"), "--predict", network_file, boards],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert (run.returncode, run.stdout, run.stderr) == (0, "Check White\nCheck Black\n", "")


# --predict

def test_predict_prints_one_label_per_board_in_order_and_nothing_else(tmp_path, capsys):
    network_file = side_network(tmp_path)
    boards = write_boards(tmp_path, [BLACK_TO_MOVE, WHITE_TO_MOVE, "", BLACK_TO_MOVE])
    assert main(["--predict", network_file, boards]) == EXIT_SUCCESS
    out, err = capsys.readouterr()
    assert out == "Check Black\nCheck White\nCheck Black\n"
    assert err == ""


def test_predict_reports_the_accuracy_on_stderr_when_outputs_are_expected(tmp_path, capsys):
    network_file = side_network(tmp_path)
    boards = write_boards(tmp_path, [f"{WHITE_TO_MOVE} Check White", BLACK_TO_MOVE, f"{BLACK_TO_MOVE} Nothing"])
    assert main(["--predict", network_file, boards]) == EXIT_SUCCESS
    out, err = capsys.readouterr()
    assert out == "Check White\nCheck Black\nCheck Black\n"
    assert err.startswith("accuracy: 50.0% (1/2 boards)\n")


def test_predict_on_an_empty_file_prints_nothing(tmp_path, capsys):
    network_file = side_network(tmp_path)
    assert main(["--predict", network_file, write_boards(tmp_path, [])]) == EXIT_SUCCESS
    assert capsys.readouterr() == ("", "")


PREDICT_ERRORS_EXIT_84_WITH_A_MESSAGE_CASES = [
    ("missing.nn", [WHITE_TO_MOVE], "cannot open the network file"),
    ("side", ["not a fen"], "line 1: expected at least 6 fields"),
    ("side", [f"{WHITE_TO_MOVE} Check"], "expected output 'Check' is not one the network predicts"),
]


def test_predict_errors_exit_84_with_a_message(tmp_path, capsys):
    for network, lines, message in PREDICT_ERRORS_EXIT_84_WITH_A_MESSAGE_CASES:
        network_file = side_network(tmp_path)
        path = network_file if network == "side" else str(tmp_path / network)
        assert main(["--predict", path, write_boards(tmp_path, lines)]) == EXIT_ERROR
        out, err = capsys.readouterr()
        assert out == "" or message.startswith("expected output")
        assert err.startswith("my_torch_analyzer: ")
        assert message in err


# --train

def training_lines(count: int) -> list[str]:
    return [f"{WHITE_TO_MOVE} Nothing", f"{BLACK_TO_MOVE} Checkmate White"] * count


def test_train_saves_to_savefile_and_keeps_loadfile(tmp_path, capfd):
    network_file = side_network(tmp_path)
    before = Path(network_file).read_bytes()
    saved = tmp_path / "trained.nn"
    assert main(["--train", "--save", str(saved), network_file, write_boards(tmp_path, training_lines(100))]) == EXIT_SUCCESS
    out, _ = capfd.readouterr()
    assert "training on 200 boards, 20 kept aside for validation" in out
    assert f"saved to {saved}" in out
    assert Path(network_file).read_bytes() == before
    assert main(["--predict", str(saved), write_boards(tmp_path, [WHITE_TO_MOVE, BLACK_TO_MOVE], "check.txt")]) == EXIT_SUCCESS
    assert capfd.readouterr().out == "Nothing\nCheckmate White\n"


def test_train_without_save_overwrites_loadfile(tmp_path, capfd):
    network_file = side_network(tmp_path)
    before = Path(network_file).read_bytes()
    assert main(["--train", network_file, write_boards(tmp_path, training_lines(10))]) == EXIT_SUCCESS
    assert f"saved to {network_file}" in capfd.readouterr().out
    assert Path(network_file).read_bytes() != before


def test_train_needs_every_expected_output(tmp_path, capsys):
    network_file = side_network(tmp_path)
    assert main(["--train", network_file, write_boards(tmp_path, [f"{WHITE_TO_MOVE} Nothing", BLACK_TO_MOVE])]) == EXIT_ERROR
    assert "line 2: training needs the expected output" in capsys.readouterr().err


def test_train_refuses_labels_the_network_does_not_predict(tmp_path, capsys):
    network_file = side_network(tmp_path)
    assert main(["--train", network_file, write_boards(tmp_path, [f"{WHITE_TO_MOVE} Nothing", f"{BLACK_TO_MOVE} Check"])]) == EXIT_ERROR
    assert "line 2: expected output 'Check' is not one the network predicts" in capsys.readouterr().err


def test_train_errors_of_the_engine_exit_84(tmp_path, capsys):
    assert main(["--train", str(tmp_path / "missing.nn"), write_boards(tmp_path, training_lines(2))]) == EXIT_ERROR
    assert "cannot open the network file" in capsys.readouterr().err


def test_a_file_that_is_not_a_network_exits_84(tmp_path, capfd):
    for mode in ["--predict", "--train"]:
        network = tmp_path / "corrupt.nn"
        network.write_bytes(b"not a network")
        assert main([mode, str(network), write_boards(tmp_path, [f"{WHITE_TO_MOVE} Nothing"])]) == EXIT_ERROR
        out, err = capfd.readouterr()
        assert out == ""
        assert "is not a valid my_torch network" in err
        assert network.read_bytes() == b"not a network"
