import chess.pgn

from .labels import label_board


def read_games(path):
    with open(path, encoding="utf-8", errors="replace") as handle:
        while (game := chess.pgn.read_game(handle)) is not None:
            if not game.errors:
                yield game


def fen_of(board):
    return board.fen(en_passant="fen")


def sample_game(game, rng, per_game):
    moves = list(game.mainline_moves())
    if not moves:
        return []
    quiet_plies = set(rng.sample(range(len(moves)), min(per_game, len(moves))))
    board = game.board()
    quiet, checks, mates = [], [], []
    for ply, move in enumerate(moves):
        board.push(move)
        if board.is_check():
            label = label_board(board)
            target = mates if label.startswith("Checkmate") else checks
            target.append((label, fen_of(board)))
        elif ply in quiet_plies:
            quiet.append(("Nothing", fen_of(board)))
    return quiet + rng.sample(checks, min(per_game, len(checks))) + mates
