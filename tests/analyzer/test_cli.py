import subprocess
import sys
from pathlib import Path

import pytest

from analyzer.cli import EXIT_ERROR, USAGE, Arguments, UsageError, main, parse_args

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


def test_valid_line_exits_84_until_modes_exist(capsys):
    assert main(["--predict", "net.nn", "boards.txt"]) == EXIT_ERROR
    assert "not implemented" in capsys.readouterr().err


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
