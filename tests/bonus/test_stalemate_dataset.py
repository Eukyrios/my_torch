import random

import chess

from analyzer.fen import parse_fen
from bonus.stalemate_dataset import STALEMATE, build, draw_endgames, label_of, random_endgame, relabel

# black to move, king in the corner, no legal move and not in check
STALEMATE_FEN = "k7/2Q5/1K6/8/8/8/8/8 b - - 0 1"


def test_a_stalemate_gets_its_own_label():
    assert label_of(chess.Board(STALEMATE_FEN)) == STALEMATE
    assert label_of(chess.Board("k7/1Q6/1K6/8/8/8/8/8 b - - 0 1")) == "Checkmate White"
    assert label_of(chess.Board()) == "Nothing"


def test_random_endgames_are_legal_and_the_side_to_move_is_nearly_bare():
    rng = random.Random(1)
    boards = [board for board in (random_endgame(rng) for _ in range(300)) if board is not None]
    assert len(boards) > 100
    for board in boards:
        assert board.is_valid()
        mover = board.occupied_co[board.turn]
        assert chess.popcount(mover) <= 2
        assert 2 <= chess.popcount(board.occupied_co[not board.turn]) <= 4


def test_draws_the_stalemates_and_others_asked_without_duplicates():
    lines = draw_endgames(20, 10, random.Random(3))
    labels = [" ".join(line.split()[6:]) for line in lines]
    assert labels.count(STALEMATE) == 20
    assert len(lines) == 30
    assert len({" ".join(line.split()[:6]) for line in lines}) == 30
    for line in lines:
        parse_fen(" ".join(line.split()[:6]))


def test_relabels_the_real_stalemates_of_the_dataset():
    assert relabel(f"{STALEMATE_FEN} Nothing") == f"{STALEMATE_FEN} {STALEMATE}"
    start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 Nothing"
    assert relabel(start) == start


def test_builds_the_three_splits_with_the_drawn_stalemates(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    for split in ("train", "validation", "test"):
        (data / f"{split}.txt").write_text(f"{STALEMATE_FEN} Nothing\n\n8/8/8/8/8/8/8/k1K5 w - - 0 1 Nothing\n")
    counts = build(data, tmp_path / "out", 30, seed=0)
    assert sum(counts.values()) == 6 + 30 + 15
    lines = "".join((tmp_path / "out" / f"{split}.txt").read_text() for split in ("train", "validation", "test")).splitlines()
    assert sum(line.endswith(STALEMATE) for line in lines) == 3 + 30
    assert build(data, tmp_path / "again", 30, seed=0) == counts
    assert (tmp_path / "again" / "train.txt").read_text() == (tmp_path / "out" / "train.txt").read_text()
