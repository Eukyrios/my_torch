import random
from collections import Counter

import chess

from dataset.games import fen_of, read_games, sample_game


def labels_of(samples):
    return Counter(label for label, _ in samples)


def test_read_games_skips_games_with_illegal_moves(pgn_file, game_count):
    sites = [game.headers["Site"] for game in read_games(pgn_file)]
    assert len(sites) == game_count - 1
    assert "https://lichess.org/illegal1" not in sites


def test_fen_keeps_the_en_passant_square_like_the_subject():
    board = chess.Board()
    board.push_san("d4")
    assert fen_of(board).endswith(" b KQkq d3 0 1")


def test_checkmate_position_is_always_kept(load_game):
    samples = sample_game(load_game(0), random.Random(0), per_game=10)
    assert labels_of(samples) == {"Nothing": 3, "Checkmate Black": 1}


def test_scholars_mate_is_labelled_for_white(load_game):
    samples = sample_game(load_game(1), random.Random(0), per_game=10)
    assert labels_of(samples)["Checkmate White"] == 1


def test_checks_are_capped_per_game(load_game):
    assert labels_of(sample_game(load_game(2), random.Random(0), per_game=1))["Check White"] == 1
    assert labels_of(sample_game(load_game(2), random.Random(0), per_game=5))["Check White"] == 2


def test_black_checks_are_labelled_for_black(load_game):
    samples = sample_game(load_game(3), random.Random(0), per_game=5)
    assert labels_of(samples)["Check Black"] == 2


def test_quiet_positions_are_capped_per_game(load_game):
    samples = sample_game(load_game(3), random.Random(0), per_game=1)
    assert labels_of(samples)["Nothing"] <= 1


def test_game_without_moves_gives_nothing(load_game):
    assert sample_game(load_game(4), random.Random(0), per_game=3) == []


def test_sampling_is_reproducible(load_game):
    first = sample_game(load_game(3), random.Random(5), per_game=2)
    assert first == sample_game(load_game(3), random.Random(5), per_game=2)
