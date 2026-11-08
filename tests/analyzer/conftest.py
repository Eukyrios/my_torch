import json
import struct
import zipfile

import pytest

from analyzer.encoder import INPUT_SIZE, SIDE_TO_MOVE_INDEX
from analyzer.engine import ENGINE
from analyzer.prediction import LABELS

WHITE_TO_MOVE = "8/8/8/8/8/8/8/k1K5 w - - 0 1"
BLACK_TO_MOVE = "8/8/8/8/8/8/8/k1K5 b - - 0 1"


def npy(rows: list[list[float]], vector: bool = False) -> bytes:
    shape = f"({len(rows[0])},)" if vector else f"({len(rows)}, {len(rows[0])})"
    header = f"{{'descr': '<f8', 'fortran_order': False, 'shape': {shape}, }}"
    header += " " * (63 - (10 + len(header)) % 64) + "\n"
    values = [value for row in rows for value in row]
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header.encode() + struct.pack(f"<{len(values)}d", *values)


def write_network(path, weights: list[list[float]], biases: list[float], metadata: dict | None = None) -> str:
    header = {
        "format": "my_torch",
        "version": 1,
        "layers": [{"type": "dense", "activation": "softmax", "shape": [len(weights), len(biases)]}],
        "metadata": metadata or {},
    }
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("header.json", json.dumps(header))
        archive.writestr("layer_0_weights.npy", npy(weights))
        archive.writestr("layer_0_biases.npy", npy([biases], vector=True))
    return str(path)


def side_network(tmp_path) -> str:
    """A 769 -> 5 network that answers Check White when white is to move, Check Black otherwise."""
    if not ENGINE.is_file():
        pytest.fail(f"{ENGINE} is missing, build it with make")
    weights = [[0.0] * len(LABELS) for _ in range(INPUT_SIZE)]
    weights[SIDE_TO_MOVE_INDEX][LABELS.index("Check White")] = 4.0
    biases = [0.0] * len(LABELS)
    biases[LABELS.index("Check Black")] = 1.0
    return write_network(tmp_path / "side.nn", weights, biases)


def write_boards(tmp_path, lines: list[str], name: str = "boards.txt") -> str:
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n")
    return str(path)
