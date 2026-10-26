import chess
import pytest

from dataset.labels import LABELS, attacker, label_board

# positions and labels from the database given with the project
SAMPLE_LINES = [
    "8/8/R2k4/4r1p1/8/5K2/5P2/8 b - - 7 59 Check White",
    "8/8/K3Q2k/6p1/6Q1/4B3/7P/8 b - - 2 61 Check White",
    "3R4/2Pk4/KP6/8/8/8/8/5r2 b - - 4 79 Check White",
    "4rk2/1R3R1p/rp3b2/8/5P2/4P3/5PKP/8 b - - 0 36 Check White",
    "2rr4/pp4k1/4RR1p/3p1Np1/5n2/2P5/PP3PPP/6K1 b - - 6 26 Check White",
    "2Q5/p1q5/2p5/1pP2k2/1P3b2/6p1/P4PB1/3R1K2 b - - 0 43 Check White",
    "2r2b1k/5Pp1/4q2p/2pbP2P/3p2R1/5N1Q/3BnP2/1q3NK1 w - - 2 41 Check Black",
    "3r1r1k/p6p/1pq5/4ppn1/PPB3p1/4P1P1/1QN2PKP/3R4 w - - 4 29 Check Black",
    "2r2rk1/q3b1p1/pn3p2/N3p2p/3nN2P/3PP1P1/QP1B1PK1/R1R5 b - - 2 25 Check White",
    "6k1/4nppp/4p3/8/1pNnPP2/pP1B4/r1r3PP/K2R1R2 w - - 0 26 Check Black",
]


def play(*moves):
    board = chess.Board()
    for move in moves:
        board.push_san(move)
    return board


@pytest.mark.parametrize("line", SAMPLE_LINES)
def test_matches_labels_of_the_given_database(line):
    fen, expected = line.rsplit(" ", 2)[0], " ".join(line.split()[6:])
    assert label_board(chess.Board(fen)) == expected


def test_start_position_is_nothing():
    assert label_board(chess.Board()) == "Nothing"


def test_label_names_the_side_giving_check():
    assert label_board(play("e4", "f5", "Qh5+")) == "Check White"
    assert label_board(play("d4", "e5", "dxe5", "Bb4+")) == "Check Black"


def test_label_names_the_side_giving_mate():
    assert label_board(play("f3", "e5", "g4", "Qh4#")) == "Checkmate Black"
    assert label_board(play("e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6", "Qxf7#")) == "Checkmate White"


def test_stalemate_is_nothing():
    assert label_board(chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")) == "Nothing"


def test_attacker_is_the_opposite_of_the_side_to_move():
    assert attacker(chess.Board()) == "Black"
    assert attacker(chess.Board("8/8/8/8/8/8/8/k1K5 b - - 0 1")) == "White"


def test_labels_cover_the_five_classes():
    assert len(set(LABELS)) == 5
