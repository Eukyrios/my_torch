"""Encode a chess position into the neural network input vector.

Layout (769 inputs):
    [0, 768)  one-hot: PLANE[piece] * 64 + square   (12 planes x 64 squares)
    768       side to move (1.0 = white, 0.0 = black)

Plane order is P N B R Q K p n b r q k, squares follow analyzer.fen numbering
(a8 = 0 ... h1 = 63).
"""

from __future__ import annotations

from .fen import PIECES, EMPTY, Position, parse_fen

NUM_SQUARES = 64
NUM_PLANES = len(PIECES)
SIDE_TO_MOVE_INDEX = NUM_PLANES * NUM_SQUARES
INPUT_SIZE = SIDE_TO_MOVE_INDEX + 1  # 769

PLANE = {piece: index for index, piece in enumerate(PIECES)}


def encode_position(position: Position) -> list[float]:
    vector = [0.0] * INPUT_SIZE
    for square, piece in enumerate(position.board):
        if piece != EMPTY:
            vector[PLANE[piece] * NUM_SQUARES + square] = 1.0
    vector[SIDE_TO_MOVE_INDEX] = 1.0 if position.white_to_move else 0.0
    return vector


def encode_fen(fen: str) -> list[float]:
    """Parse and encode a FEN. Raises FenError if the FEN is invalid."""
    return encode_position(parse_fen(fen))