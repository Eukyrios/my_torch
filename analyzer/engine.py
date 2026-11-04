# Runs my_torch_engine, the C++ program that owns the network (my_torch/engine), and talks to it
# in the format of scripts/encode_dataset.py: a first line "n_inputs n_classes", then one board
# per line, its class index ("-" when unknown) and the indices of its inputs that are 1.
# predict reads back one line of outputs per board; train lets the engine print its progress.

import json
import subprocess
import zipfile
from pathlib import Path

from analyzer.chess_file import Board
from analyzer.encoder import INPUT_SIZE, active_inputs
from analyzer.prediction import LABELS, is_label

ENGINE = Path(__file__).resolve().parent.parent / "my_torch" / "my_torch_engine"
UNKNOWN_CLASS = "-"


class EngineError(Exception):
    pass


def boards_text(boards: list[Board], n_classes: int, classes: list[int] | None = None) -> str:
    lines = [f"{INPUT_SIZE} {n_classes}"]
    for row, board in enumerate(boards):
        label = UNKNOWN_CLASS if classes is None else str(classes[row])
        lines.append(" ".join([label] + [str(index) for index in active_inputs(board.position)]))
    return "\n".join(lines) + "\n"


def run_engine(arguments: list[str], text: str, capture_output: bool) -> str:
    if not ENGINE.is_file():
        raise EngineError(f"{ENGINE} is missing, build it with make")
    stdout = subprocess.PIPE if capture_output else None
    result = subprocess.run([str(ENGINE)] + arguments, input=text, stdout=stdout, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise EngineError(result.stderr.strip() or f"my_torch_engine exited with {result.returncode}")
    return result.stdout or ""


def parse_outputs(text: str) -> list[list[float]]:
    try:
        return [[float(value) for value in line.split()] for line in text.splitlines()]
    except ValueError:
        raise EngineError("my_torch_engine printed something that is not a list of outputs")


def predict_outputs(network_file: str, boards: list[Board], labels: tuple[str, ...]) -> list[list[float]]:
    return parse_outputs(run_engine(["predict", network_file], boards_text(boards, len(labels)), capture_output=True))


def train(network_file: str, save_file: str, boards: list[Board], classes: list[int], labels: tuple[str, ...]) -> None:
    run_engine(["train", network_file, save_file], boards_text(boards, len(labels), classes), capture_output=False)


def network_labels(network_file: str) -> tuple[str, ...]:
    # the labels a network was trained on can be stored in its metadata; LABELS otherwise. A file
    # that can't be read keeps LABELS here: the engine will say what is wrong with it
    try:
        with zipfile.ZipFile(network_file) as archive:
            labels = json.loads(archive.read("header.json"))["metadata"]["labels"]
    except (OSError, KeyError, TypeError, ValueError, zipfile.BadZipFile):
        return LABELS
    if not isinstance(labels, list) or not labels or not all(isinstance(label, str) and is_label(label) for label in labels):
        return LABELS
    return tuple(labels)
