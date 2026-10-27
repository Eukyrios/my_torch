import argparse
import random
import sys
from pathlib import Path

from analyzer.fen import FenError, parse_fen

from .balance import balance
from .games import read_games, sample_game
from .labels import LABELS
from .splits import SPLITS, game_digest, quotas, split_of

EXIT_ERROR = 84
PROGRESS_EVERY = 50_000


def is_valid(fen):
    try:
        parse_fen(fen)
    except FenError:
        return False
    return True


def label_of(line):
    return " ".join(line.split()[6:])


def piece_count(line):
    return sum(ch.isalpha() for ch in line.split()[0])


class Collector:
    def __init__(self, limits):
        self.limits = limits
        self.lines = {split: {label: [] for label in LABELS} for split in SPLITS}
        self.seen = set()
        self.rejected = 0

    def split_is_full(self, split):
        return all(len(lines) >= self.limits[split] for lines in self.lines[split].values())

    def is_full(self):
        return all(self.split_is_full(split) for split in SPLITS)

    def add(self, split, label, fen):
        lines = self.lines[split][label]
        key = " ".join(fen.split()[:4])
        if len(lines) >= self.limits[split] or key in self.seen:
            return
        if not is_valid(fen):
            self.rejected += 1
            return
        self.seen.add(key)
        lines.append(f"{fen} {label}")

    def shortfalls(self):
        return [
            (split, label, len(lines), self.limits[split])
            for split in SPLITS
            for label, lines in self.lines[split].items()
            if len(lines) < self.limits[split]
        ]


def collect(pgn_path, per_class, ratios, per_game, seed):
    collector = Collector(quotas(per_class, ratios))
    for count, game in enumerate(read_games(pgn_path), start=1):
        digest = game_digest(game.headers.get("Site", "?"), seed)
        split = split_of(digest, ratios)
        if not collector.split_is_full(split):
            for label, fen in sample_game(game, random.Random(digest), per_game):
                collector.add(split, label, fen)
        if collector.is_full():
            break
        if count % PROGRESS_EVERY == 0:
            print(f"{count} games read", file=sys.stderr)
    return collector


def write_splits(collector, out_dir, seed):
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    written = {}
    for split in SPLITS:
        lines = balance(collector.lines[split], rng)
        (out_dir / f"{split}.txt").write_text("".join(f"{line}\n" for line in lines))
        written[split] = lines
    return written


def describe(lines):
    stats = {}
    for label in LABELS:
        counts = [piece_count(line) for line in lines if label_of(line) == label]
        mean = sum(counts) / len(counts) if counts else 0.0
        stats[label] = (len(counts), mean)
    return stats


def report(collector, written, out_dir):
    print(f"written to {out_dir}/ ({collector.rejected} invalid FEN rejected)")
    for split, lines in written.items():
        print(f"\n{split}: {len(lines)} lines")
        for label, (count, mean) in describe(lines).items():
            print(f"  {label:<16} {count:>8}   mean pieces {mean:5.1f}")
    for split, label, got, wanted in collector.shortfalls():
        print(f"warning: {split} / {label}: only {got} of {wanted} positions found", file=sys.stderr)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog="python -m dataset.generate",
        description="Build balanced train/validation/test files from a Lichess PGN database.",
    )
    parser.add_argument("pgn", type=Path)
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--per-class", type=int, default=20_000, help="positions wanted per class, all splits together")
    parser.add_argument("--ratios", type=int, nargs=3, default=(80, 10, 10), metavar=("TRAIN", "VALID", "TEST"))
    parser.add_argument("--per-game", type=int, default=3, help="max quiet and check positions taken from one game")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.per_class < 1 or args.per_game < 1:
        parser.error("--per-class and --per-game must be positive")
    if min(args.ratios) < 0 or sum(args.ratios) == 0:
        parser.error("--ratios must be non-negative and not all zero")
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        collector = collect(args.pgn, args.per_class, tuple(args.ratios), args.per_game, args.seed)
    except OSError as error:
        print(f"dataset: {error}", file=sys.stderr)
        return EXIT_ERROR
    written = write_splits(collector, args.out, args.seed)
    report(collector, written, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
