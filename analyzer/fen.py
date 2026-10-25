"""FEN parsing, validation and serialisation.

Square numbering used everywhere in the project:
    index = row * 8 + col
where row 0 is rank 8 (the first rank written in a FEN string) and col 0 is
file a. So a8 = 0, h8 = 7, a1 = 56, h1 = 63.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

PIECES = "PNBRQKpnbrqk"
EMPTY = "."
FILES = "abcdefgh"
STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


class FenError(ValueError):
    """Raised when a string is not a valid FEN."""


@dataclass(frozen=True)
class Position:
    """A chess position. `board` has 64 entries, EMPTY ('.') for empty squares."""

    board: tuple[str, ...]
    white_to_move: bool
    castling: str = "-"
    en_passant: int | None = None
    halfmove: int = 0
    fullmove: int = 1


def square_index(name: str) -> int:
    """'e2' -> 52."""
    if len(name) != 2 or name[0] not in FILES or name[1] not in "12345678":
        raise FenError(f"invalid square: {name!r}")
    return (8 - int(name[1])) * 8 + FILES.index(name[0])


def square_name(index: int) -> str:
    """52 -> 'e2'."""
    if not 0 <= index < 64:
        raise ValueError(f"square index out of range: {index}")
    return FILES[index % 8] + str(8 - index // 8)


def _parse_placement(text: str) -> tuple[str, ...]:
    ranks = text.split("/")
    if len(ranks) != 8:
        raise FenError(f"expected 8 ranks, got {len(ranks)}")
    board: list[str] = []
    for number, rank in enumerate(ranks, start=1):
        width = 0
        previous_was_digit = False
        for ch in rank:
            if ch in "12345678":
                if previous_was_digit:
                    raise FenError(f"rank {number}: consecutive digits")
                width += int(ch)
                board.extend(EMPTY * int(ch))
                previous_was_digit = True
            elif ch in PIECES:
                width += 1
                board.append(ch)
                previous_was_digit = False
            else:
                raise FenError(f"rank {number}: invalid character {ch!r}")
        if width != 8:
            raise FenError(f"rank {number}: describes {width} squares instead of 8")
    return tuple(board)


def _check_material(board: tuple[str, ...]) -> None:
    counts = Counter(board)
    if counts["K"] != 1 or counts["k"] != 1:
        raise FenError("each side must have exactly one king")
    for pawn in "Pp":
        if counts[pawn] > 8:
            raise FenError("a side cannot have more than 8 pawns")
    for side in ("PNBRQK", "pnbrqk"):
        if sum(counts[p] for p in side) > 16:
            raise FenError("a side cannot have more than 16 pieces")
    for square, piece in enumerate(board):
        if piece in ("P", "p") and square // 8 in (0, 7):
            raise FenError("pawns cannot stand on the first or last rank")


def _parse_castling(text: str) -> str:
    if text == "-":
        return "-"
    if any(c not in "KQkq" for c in text) or len(set(text)) != len(text):
        raise FenError(f"invalid castling field: {text!r}")
    return "".join(c for c in "KQkq" if c in text)


def _parse_counter(text: str, name: str, minimum: int) -> int:
    if not (text.isascii() and text.isdigit()):
        raise FenError(f"invalid {name} counter: {text!r}")
    value = int(text)
    if value < minimum:
        raise FenError(f"{name} counter must be >= {minimum}")
    return value


def parse_fen(text: str) -> Position:
    """Parse a 6-field FEN string. Raises FenError if it is invalid."""
    if not isinstance(text, str):
        raise FenError("FEN must be a string")
    fields = text.split()
    if len(fields) != 6:
        raise FenError(f"expected 6 fields, got {len(fields)}")
    placement, side, castling, en_passant, halfmove, fullmove = fields

    board = _parse_placement(placement)
    _check_material(board)

    if side not in ("w", "b"):
        raise FenError(f"invalid side to move: {side!r}")
    white_to_move = side == "w"

    ep_square = None
    if en_passant != "-":
        ep_square = square_index(en_passant)
        # After white's double push the target is on rank 3 (black to move), and vice versa.
        expected_row = 5 if not white_to_move else 2
        if ep_square // 8 != expected_row:
            raise FenError(f"en passant square {en_passant!r} does not match the side to move")

    return Position(
        board=board,
        white_to_move=white_to_move,
        castling=_parse_castling(castling),
        en_passant=ep_square,
        halfmove=_parse_counter(halfmove, "halfmove", 0),
        fullmove=_parse_counter(fullmove, "fullmove", 1),
    )


def to_fen(position: Position) -> str:
    """Serialise a position back to a FEN string."""
    rows = []
    for row in range(8):
        out = ""
        empty = 0
        for ch in position.board[row * 8 : row * 8 + 8]:
            if ch == EMPTY:
                empty += 1
                continue
            if empty:
                out += str(empty)
                empty = 0
            out += ch
        if empty:
            out += str(empty)
        rows.append(out)
    ep = "-" if position.en_passant is None else square_name(position.en_passant)
    side = "w" if position.white_to_move else "b"
    return f"{'/'.join(rows)} {side} {position.castling} {ep} {position.halfmove} {position.fullmove}"


def split_fen_line(line: str) -> tuple[str, str | None]:
    """Split a dataset line into (fen, expected_label).

    A line is a 6-field FEN optionally followed by a label (e.g. 'Checkmate Black').
    """
    fields = line.split()
    if len(fields) < 6:
        raise FenError(f"expected at least 6 fields, got {len(fields)}")
    label = " ".join(fields[6:]) or None
    return " ".join(fields[:6]), label