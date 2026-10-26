import chess

LABELS = ("Nothing", "Check White", "Check Black", "Checkmate White", "Checkmate Black")


def attacker(board):
    # the side to move is the one in check, the label names the side giving it
    return "Black" if board.turn == chess.WHITE else "White"


def label_board(board):
    if not board.is_check():
        return "Nothing"
    kind = "Checkmate" if board.is_checkmate() else "Check"
    return f"{kind} {attacker(board)}"
