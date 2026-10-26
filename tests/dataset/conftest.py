import io

import chess.pgn
import pytest

HEADER = '[Event "test"]\n[Site "https://lichess.org/{site}"]\n[Result "{result}"]\n\n'

GAMES = [
    ("fool0001", "0-1", "1. f3 e5 2. g4 Qh4# 0-1"),
    ("scholar1", "1-0", "1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6 4. Qxf7# 1-0"),
    ("whitechk", "*", "1. e4 f5 2. Qh5+ g6 3. Qxg6+ hxg6 *"),
    ("blackchk", "*", "1. d4 e5 2. dxe5 Bb4+ 3. c3 Bxc3+ 4. Nxc3 *"),
    ("nomoves1", "0-1", "0-1"),
    ("illegal1", "*", "1. e4 e5 2. Ke4 *"),
]


def pgn_text(games=GAMES):
    return "\n\n".join(HEADER.format(site=site, result=result) + moves for site, result, moves in games) + "\n"


@pytest.fixture
def pgn_file(tmp_path):
    path = tmp_path / "games.pgn"
    path.write_text(pgn_text())
    return path


@pytest.fixture
def load_game():
    def load(index):
        site, result, moves = GAMES[index]
        return chess.pgn.read_game(io.StringIO(HEADER.format(site=site, result=result) + moves))

    return load


@pytest.fixture
def game_count():
    return len(GAMES)
