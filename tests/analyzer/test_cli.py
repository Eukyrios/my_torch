import subprocess
import sys
from pathlib import Path

import pytest

from analyzer.cli import EXIT_ERROR, EXIT_SUCCESS, USAGE, Arguments, UsageError, main, parse_args

from tests.analyzer.conftest import BLACK_TO_MOVE, WHITE_TO_MOVE

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "argv, expected",
    [
        (["--predict", "net.nn", "boards.txt"], Arguments("predict", "net.nn", "boards.txt")),
        (["--train", "net.nn", "boards.txt"], Arguments("train", "net.nn", "boards.txt")),
        (["--train", "--save", "out.nn", "net.nn", "boards.txt"], Arguments("train", "net.nn", "boards.txt", "out.nn")),
        (["net.nn", "--train", "boards.txt", "--save", "out.nn"], Arguments("train", "net.nn", "boards.txt", "out.nn")),
    ],
)
def test_parse_args_accepts_the_subject_usage(argv, expected):
    assert parse_args(argv) == expected


@pytest.mark.parametrize(
    "argv, message",
    [
        ([], "expected --predict or --train"),
        (["net.nn", "boards.txt"], "expected --predict or --train"),
        (["--predict", "--train", "net.nn", "boards.txt"], "only one of"),
        (["--predict", "--save", "out.nn", "net.nn", "boards.txt"], "only works in train mode"),
        (["--train", "net.nn", "boards.txt", "--save"], "needs a SAVEFILE"),
        (["--train", "--save", "a.nn", "--save", "b.nn", "net.nn", "boards.txt"], "given twice"),
        (["--predict", "net.nn"], "expected LOADFILE and CHESSFILE"),
        (["--predict", "net.nn", "boards.txt", "extra"], "expected LOADFILE and CHESSFILE"),
        (["--verbose", "--predict", "net.nn", "boards.txt"], "unknown option --verbose"),
    ],
)
def test_parse_args_rejects_bad_lines(argv, message):
    with pytest.raises(UsageError, match=message):
        parse_args(argv)


@pytest.mark.parametrize("flag", ["--help", "-h"])
def test_help_prints_usage_and_exits_0(flag, capsys):
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


def test_launcher_predicts_like_in_the_subject(side_network, write_boards):
    boards = write_boards([WHITE_TO_MOVE, BLACK_TO_MOVE])
    run = subprocess.run(
        [sys.executable, str(ROOT / "my_torch_analyzer.py"), "--predict", side_network, boards],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert (run.returncode, run.stdout, run.stderr) == (0, "Check White\nCheck Black\n", "")


# --predict

def test_predict_prints_one_label_per_board_in_order_and_nothing_else(side_network, write_boards, capsys):
    boards = write_boards([BLACK_TO_MOVE, WHITE_TO_MOVE, "", BLACK_TO_MOVE])
    assert main(["--predict", side_network, boards]) == EXIT_SUCCESS
    out, err = capsys.readouterr()
    assert out == "Check Black\nCheck White\nCheck Black\n"
    assert err == ""


def test_predict_reports_the_accuracy_on_stderr_when_outputs_are_expected(side_network, write_boards, capsys):
    boards = write_boards([f"{WHITE_TO_MOVE} Check White", BLACK_TO_MOVE, f"{BLACK_TO_MOVE} Nothing"])
    assert main(["--predict", side_network, boards]) == EXIT_SUCCESS
    out, err = capsys.readouterr()
    assert out == "Check White\nCheck Black\nCheck Black\n"
    assert err.startswith("accuracy: 50.0% (1/2 boards)\n")


def test_predict_on_an_empty_file_prints_nothing(side_network, write_boards, capsys):
    assert main(["--predict", side_network, write_boards([])]) == EXIT_SUCCESS
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize(
    "network, lines, message",
    [
        ("missing.nn", [WHITE_TO_MOVE], "cannot open the network file"),
        ("side", ["not a fen"], "line 1: expected at least 6 fields"),
        ("side", [f"{WHITE_TO_MOVE} Check"], "expected output 'Check' is not one the network predicts"),
    ],
)
def test_predict_errors_exit_84_with_a_message(side_network, write_boards, tmp_path, capsys, network, lines, message):
    path = side_network if network == "side" else str(tmp_path / network)
    assert main(["--predict", path, write_boards(lines)]) == EXIT_ERROR
    out, err = capsys.readouterr()
    assert out == "" or message.startswith("expected output")
    assert err.startswith("my_torch_analyzer: ")
    assert message in err


# --train

def training_lines(count: int) -> list[str]:
    return [f"{WHITE_TO_MOVE} Nothing", f"{BLACK_TO_MOVE} Checkmate White"] * count


def test_train_saves_to_savefile_and_keeps_loadfile(side_network, write_boards, tmp_path, capfd):
    before = Path(side_network).read_bytes()
    saved = tmp_path / "trained.nn"
    assert main(["--train", "--save", str(saved), side_network, write_boards(training_lines(100))]) == EXIT_SUCCESS
    out, _ = capfd.readouterr()
    assert "training on 200 boards, 20 kept aside for validation" in out
    assert f"saved to {saved}" in out
    assert Path(side_network).read_bytes() == before
    assert main(["--predict", str(saved), write_boards([WHITE_TO_MOVE, BLACK_TO_MOVE], "check.txt")]) == EXIT_SUCCESS
    assert capfd.readouterr().out == "Nothing\nCheckmate White\n"


def test_train_without_save_overwrites_loadfile(side_network, write_boards, capfd):
    before = Path(side_network).read_bytes()
    assert main(["--train", side_network, write_boards(training_lines(10))]) == EXIT_SUCCESS
    assert f"saved to {side_network}" in capfd.readouterr().out
    assert Path(side_network).read_bytes() != before


def test_train_needs_every_expected_output(side_network, write_boards, capsys):
    assert main(["--train", side_network, write_boards([f"{WHITE_TO_MOVE} Nothing", BLACK_TO_MOVE])]) == EXIT_ERROR
    assert "line 2: training needs the expected output" in capsys.readouterr().err


def test_train_refuses_labels_the_network_does_not_predict(side_network, write_boards, capsys):
    assert main(["--train", side_network, write_boards([f"{WHITE_TO_MOVE} Nothing", f"{BLACK_TO_MOVE} Check"])]) == EXIT_ERROR
    assert "line 2: expected output 'Check' is not one the network predicts" in capsys.readouterr().err


def test_train_errors_of_the_engine_exit_84(tmp_path, write_boards, capsys):
    assert main(["--train", str(tmp_path / "missing.nn"), write_boards(training_lines(2))]) == EXIT_ERROR
    assert "cannot open the network file" in capsys.readouterr().err
