# Bonus: builds the dataset of the stalemate network, the 5 classes of data/ plus Stalemate.
# Real games almost never end in stalemate, so stalemates are drawn: random endgames where the
# side to move has its king and at most a pawn, against a king and 1 to 3 pieces, its king often
# on the edge of the board where stalemates happen; python-chess keeps the legal ones and says
# which are stalemate. Half as many endgames drawn the same way but that are not stalemate are
# added with their own label, so the network can't learn "few pieces means stalemate".
#
#     python -m bonus.stalemate_dataset --data data --out data/stalemate --per-class 40000

import argparse
import random
import sys
from pathlib import Path

import chess

from dataset.labels import label_board
from dataset.splits import SPLITS

EXIT_ERROR = 84
STALEMATE = "Stalemate"
RATIOS = (8, 1, 1)
EDGE_SQUARES = [square for square in chess.SQUARES if chess.square_file(square) in (0, 7) or chess.square_rank(square) in (0, 7)]
ATTACKERS = (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN)


def random_endgame(rng: random.Random) -> chess.Board | None:
    board = chess.Board(None)
    side = rng.choice(chess.COLORS)
    king = rng.choice(EDGE_SQUARES) if rng.random() < 0.7 else rng.choice(chess.SQUARES)
    squares = rng.sample([square for square in chess.SQUARES if square != king], 5)
    board.set_piece_at(king, chess.Piece(chess.KING, side))
    board.set_piece_at(squares[0], chess.Piece(chess.KING, not side))
    if rng.random() < 0.3:
        board.set_piece_at(squares[1], chess.Piece(chess.PAWN, side))
    for square in squares[2:2 + rng.randint(1, 3)]:
        board.set_piece_at(square, chess.Piece(rng.choice(ATTACKERS), not side))
    board.turn = side
    return board if board.is_valid() else None


def label_of(board: chess.Board) -> str:
    return STALEMATE if board.is_stalemate() else label_board(board)


def draw_endgames(stalemates: int, others: int, rng: random.Random) -> list[str]:
    lines = []
    seen = set()
    wanted = {True: stalemates, False: others}
    while wanted[True] or wanted[False]:
        board = random_endgame(rng)
        if board is None or board.fen() in seen:
            continue
        is_stalemate = board.is_stalemate()
        if wanted[is_stalemate]:
            wanted[is_stalemate] -= 1
            seen.add(board.fen())
            lines.append(f"{board.fen()} {label_of(board)}")
    return lines


def relabel(line: str) -> str:
    # data/ labels a stalemate Nothing: give the few real ones their own class
    fields = line.split()
    board = chess.Board(" ".join(fields[:6]))
    return f"{' '.join(fields[:6])} {STALEMATE}" if board.is_stalemate() else line


def split_lines(lines: list[str], rng: random.Random) -> dict[str, list[str]]:
    splits = {split: [] for split in SPLITS}
    for line in lines:
        splits[rng.choices(SPLITS, weights=RATIOS)[0]].append(line)
    return splits


def build(data: Path, out: Path, per_class: int, seed: int) -> dict[str, int]:
    rng = random.Random(seed)
    drawn = split_lines(draw_endgames(per_class, per_class // 2, rng), rng)
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in SPLITS:
        lines = [relabel(line) for line in (data / f"{split}.txt").read_text().splitlines() if line.strip()]
        lines += drawn[split]
        rng.shuffle(lines)
        (out / f"{split}.txt").write_text("".join(f"{line}\n" for line in lines))
        counts[split] = len(lines)
    return counts


def parse_args(argv):
    parser = argparse.ArgumentParser(description="build the 6-class dataset of the stalemate bonus network")
    parser.add_argument("--data", type=Path, default=Path("data"), help="folder of the 5-class train/validation/test files")
    parser.add_argument("--out", type=Path, default=Path("data/stalemate"))
    parser.add_argument("--per-class", type=int, default=40000, help="stalemates drawn, over the three splits")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        counts = build(args.data, args.out, args.per_class, args.seed)
    except (OSError, ValueError) as error:
        print(f"stalemate_dataset: {error}", file=sys.stderr)
        return EXIT_ERROR
    for split, count in counts.items():
        print(f"{split}: {count} boards -> {args.out / f'{split}.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
