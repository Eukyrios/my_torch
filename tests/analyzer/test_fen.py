import pytest

from analyzer.fen import (
    STARTING_FEN,
    FenError,
    parse_fen,
    split_fen_line,
    square_index,
    square_name,
    to_fen,
)

# FENs taken from the subject
SUBJECT_FENS = [
    "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1",
    "rnbqkbnr/pppp2pp/8/4pp1Q/3P4/4P3/PPP2PPP/RNB1KBNR b KQkq - 1 3",
    "8/8/8/8/8/8/8/k1K5 w - - 0 1",
]


@pytest.mark.parametrize("fen", SUBJECT_FENS)
def test_subject_fens_round_trip(fen):
    assert to_fen(parse_fen(fen)) == fen


def test_starting_position_fields():
    pos = parse_fen(STARTING_FEN)
    assert len(pos.board) == 64
    assert pos.board[0] == "r" and pos.board[4] == "k" and pos.board[60] == "K"
    assert pos.white_to_move is True
    assert pos.castling == "KQkq"
    assert pos.en_passant is None
    assert (pos.halfmove, pos.fullmove) == (0, 1)


def test_en_passant_and_black_to_move():
    pos = parse_fen("rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1")
    assert pos.white_to_move is False
    assert pos.en_passant == square_index("d3")


def test_castling_is_normalised_and_extra_whitespace_ignored():
    pos = parse_fen("  8/8/8/8/8/8/8/k1K5   w   qkQK  -  0  1 ")
    assert pos.castling == "KQkq"


def test_square_helpers():
    assert square_index("a8") == 0
    assert square_index("h1") == 63
    assert square_index("e2") == 52
    assert square_name(52) == "e2"
    assert all(square_index(square_name(i)) == i for i in range(64))


@pytest.mark.parametrize("name", ["", "e", "e22", "i3", "a0", "a9", "E2"])
def test_square_index_rejects_garbage(name):
    with pytest.raises(FenError):
        square_index(name)


@pytest.mark.parametrize("index", [-1, 64])
def test_square_name_out_of_range(index):
    with pytest.raises(ValueError):
        square_name(index)


INVALID_FENS = {
    "empty string": "",
    "too few fields": "8/8/8/8/8/8/8/k1K5 w - -",
    "too many fields": "8/8/8/8/8/8/8/k1K5 w - - 0 1 extra",
    "seven ranks": "8/8/8/8/8/8/k1K5 w - - 0 1",
    "nine ranks": "8/8/8/8/8/8/8/8/k1K5 w - - 0 1",
    "invalid character": "8/8/8/8/8/8/8/k1X5 w - - 0 1",
    "digit zero": "8/8/8/8/8/8/8/k1K50 w - - 0 1",
    "digit nine": "9/8/8/8/8/8/8/k1K5 w - - 0 1",
    "rank too long": "8/8/8/8/8/8/8/k1K6 w - - 0 1",
    "rank too short": "8/8/8/8/8/8/8/k1K4 w - - 0 1",
    "empty rank": "8/8/8/8/8/8/8//k1K5 w - - 0 1",
    "consecutive digits": "8/8/8/8/8/8/8/k11K4 w - - 0 1",
    "no kings": "8/8/8/8/8/8/8/8 w - - 0 1",
    "no black king": "8/8/8/8/8/8/8/2K5 w - - 0 1",
    "two white kings": "8/8/8/8/8/8/8/kKK5 w - - 0 1",
    "pawn on first rank": "8/8/8/8/8/8/8/kPK5 w - - 0 1",
    "pawn on last rank": "kp6/8/8/8/8/8/8/2K5 w - - 0 1",
    "nine white pawns": "k7/8/8/8/8/PPPPPPPP/P7/K7 w - - 0 1",
    "seventeen white pieces": "k7/8/8/8/QQQQQQQQ/QQQQQQQQ/8/K7 w - - 0 1",
    "bad side": "8/8/8/8/8/8/8/k1K5 x - - 0 1",
    "uppercase side": "8/8/8/8/8/8/8/k1K5 W - - 0 1",
    "bad castling char": "8/8/8/8/8/8/8/k1K5 w KX - 0 1",
    "duplicate castling": "8/8/8/8/8/8/8/k1K5 w KK - 0 1",
    "bad ep square": "8/8/8/8/8/8/8/k1K5 w - z9 0 1",
    "ep rank wrong for white": "8/8/8/8/8/8/8/k1K5 w - d3 0 1",
    "ep rank wrong for black": "8/8/8/8/8/8/8/k1K5 b - d6 0 1",
    "negative halfmove": "8/8/8/8/8/8/8/k1K5 w - - -1 1",
    "non numeric halfmove": "8/8/8/8/8/8/8/k1K5 w - - x 1",
    "unicode digit halfmove": "8/8/8/8/8/8/8/k1K5 w - - ² 1",
    "zero fullmove": "8/8/8/8/8/8/8/k1K5 w - - 0 0",
}


@pytest.mark.parametrize("fen", INVALID_FENS.values(), ids=INVALID_FENS.keys())
def test_invalid_fens_are_rejected(fen):
    with pytest.raises(FenError):
        parse_fen(fen)


def test_non_string_is_rejected():
    with pytest.raises(FenError):
        parse_fen(None)  # type: ignore[arg-type]


def test_fen_error_is_a_value_error():
    assert issubclass(FenError, ValueError)


def test_split_fen_line_without_label():
    fen = "8/8/8/8/8/8/8/k1K5 w - - 0 1"
    assert split_fen_line(fen) == (fen, None)


def test_split_fen_line_with_one_and_two_word_labels():
    fen = "8/8/8/8/8/8/8/k1K5 w - - 0 1"
    assert split_fen_line(f"{fen} Nothing") == (fen, "Nothing")
    assert split_fen_line(f"{fen} Checkmate Black\n") == (fen, "Checkmate Black")


def test_split_fen_line_too_short():
    with pytest.raises(FenError):
        split_fen_line("8/8/8/8/8/8/8/k1K5 w")