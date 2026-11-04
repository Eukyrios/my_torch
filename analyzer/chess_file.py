# Reads the CHESSFILE given to the analyzer: one board per line, a FEN optionally followed by its
# expected output ("Checkmate Black"), empty lines skipped. Every FEN and label is checked, and an
# error names the line it comes from.

from dataclasses import dataclass

from analyzer.fen import Position, parse_fen, split_fen_line
from analyzer.prediction import check_label


class ChessFileError(ValueError):
    pass


@dataclass(frozen=True)
class Board:
    position: Position
    expected: str | None
    line: int


def read_board(text: str, line: int) -> Board:
    fen, expected = split_fen_line(text)
    position = parse_fen(fen)
    if expected is not None:
        check_label(expected)
    return Board(position, expected, line)


def read_lines(path: str) -> list[str]:
    try:
        with open(path, encoding="utf-8") as file:
            return file.read().splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise ChessFileError(f"cannot read the chess file '{path}': {error.strerror if isinstance(error, OSError) else error}")


def read_chess_file(path: str) -> list[Board]:
    boards = []
    for line, text in enumerate(read_lines(path), start=1):
        if not text.strip():
            continue
        try:
            boards.append(read_board(text, line))
        except ValueError as error:
            raise ChessFileError(f"{path} line {line}: {error}")
    return boards


def check_expected_outputs(boards: list[Board], path: str) -> None:
    for board in boards:
        if board.expected is None:
            raise ChessFileError(f"{path} line {board.line}: training needs the expected output after the FEN")
