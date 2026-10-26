import pytest

from analyzer.encoder import INPUT_SIZE,  encode_fen, encode_position
from analyzer.encoder import NUM_SQUARES, PLANE, SIDE_TO_MOVE_INDEX
from analyzer.fen import STARTING_FEN, FenError, square_index, parse_fen


def test_input_size_is_769():
    assert INPUT_SIZE == 769
    assert SIDE_TO_MOVE_INDEX == 768


def test_vector_length_and_binary_values():
    vector = encode_fen(STARTING_FEN)
    assert len(vector) == 769
    assert set(vector) <= {0.0, 1.0}


def test_starting_position_has_32_pieces():
    vector = encode_fen(STARTING_FEN)
    assert sum(vector[:768]) == 32


def test_known_indices():
    vector = encode_fen(STARTING_FEN)
    assert vector[PLANE["P"] * NUM_SQUARES + square_index("e2")] == 1.0
    assert vector[PLANE["k"] * NUM_SQUARES + square_index("e8")] == 1.0
    assert vector[PLANE["k"] * NUM_SQUARES + square_index("e8")] == vector[708]
    assert vector[PLANE["K"] * NUM_SQUARES + square_index("e8")] == 0.0


def test_side_to_move_bit():
    assert encode_fen("8/8/8/8/8/8/8/k1K5 w - - 0 1")[768] == 1.0
    assert encode_fen("8/8/8/8/8/8/8/k1K5 b - - 0 1")[768] == 0.0


def test_each_square_has_at_most_one_active_plane():
    vector = encode_fen("rnbqkbnr/pppp2pp/8/4pp1Q/3P4/4P3/PPP2PPP/RNB1KBNR b KQkq - 1 3")
    for square in range(64):
        assert sum(vector[plane * 64 + square] for plane in range(12)) <= 1.0


def test_empty_squares_stay_zero():
    vector = encode_fen("8/8/8/8/8/8/8/k1K5 w - - 0 1")
    assert sum(vector[:768]) == 2


def test_encode_position_matches_encode_fen():
    assert encode_position(parse_fen(STARTING_FEN)) == encode_fen(STARTING_FEN)


def test_encoding_ignores_clocks_and_castling():
    a = encode_fen("8/8/8/8/8/8/8/k1K5 w - - 0 1")
    b = encode_fen("8/8/8/8/8/8/8/k1K5 w - - 40 70")
    assert a == b


def test_invalid_fen_raises():
    with pytest.raises(FenError):
        encode_fen("not a fen")