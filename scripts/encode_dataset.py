# Turns the FEN files of a dataset (data/train.txt, ...) into files the C++ benchmark reads:
# a first line "769 5", then one board per line, its class index followed by the indices of
# its inputs that are 1 (every input of the encoder is 0 or 1, so the rest are 0).

import argparse
import sys
from pathlib import Path

from analyzer.encoder import INPUT_SIZE, encode_fen
from dataset.labels import LABELS
from dataset.splits import SPLITS

EXIT_ERROR = 84


def split_line(line):
    fields = line.split()
    return " ".join(fields[:6]), " ".join(fields[6:])


def encoded_line(line):
    fen, label = split_line(line)
    active = [str(index) for index, value in enumerate(encode_fen(fen)) if value]
    return " ".join([str(LABELS.index(label))] + active)


def encode_file(source, target):
    lines = [line for line in source.read_text().splitlines() if line.strip()]
    rows = [f"{INPUT_SIZE} {len(LABELS)}"] + [encoded_line(line) for line in lines]
    target.write_text("\n".join(rows) + "\n")
    return len(lines)


def parse_args(argv):
    parser = argparse.ArgumentParser(description="encode a FEN dataset for the C++ benchmarks")
    parser.add_argument("data", type=Path, nargs="?", default=Path("data"))
    parser.add_argument("--out", type=Path, default=None, help="defaults to DATA/encoded")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    out = args.out or args.data / "encoded"
    out.mkdir(parents=True, exist_ok=True)
    try:
        for split in SPLITS:
            count = encode_file(args.data / f"{split}.txt", out / f"{split}.txt")
            print(f"{split}: {count} boards -> {out / f'{split}.txt'}")
    except (OSError, ValueError) as error:
        print(f"encode_dataset: {error}", file=sys.stderr)
        return EXIT_ERROR
    return 0


if __name__ == "__main__":
    sys.exit(main())
