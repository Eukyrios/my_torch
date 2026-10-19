# Command line of my_torch_analyzer: argument parsing, --help and the exit codes.

import sys
from dataclasses import dataclass

EXIT_ERROR = 84

USAGE = """USAGE
    ./my_torch_analyzer [--predict | --train [--save SAVEFILE]] LOADFILE CHESSFILE

DESCRIPTION
    --train     Launch the neural network in training mode. Each chessboard in FILE must
                contain inputs to send to the neural network in FEN notation and the expected
                output separated by space. If specified, the newly trained neural network will
                be saved in SAVEFILE. Otherwise, it will be saved in the original LOADFILE.
    --predict   Launch the neural network in prediction mode. Each chessboard in FILE must
                contain inputs to send to the neural network in FEN notation, and optionally
                an expected output.
    --save      Save neural network into SAVEFILE. Only works in train mode.
    LOADFILE    File containing an artificial neural network
    CHESSFILE   File containing chessboards"""


class UsageError(Exception):
    pass


@dataclass(frozen=True)
class Arguments:
    mode: str
    load_file: str
    chess_file: str
    save_file: str | None = None


def parse_args(argv: list[str]) -> Arguments:
    mode = None
    save_file = None
    positional = []
    args = iter(argv)
    for arg in args:
        if arg in ("--predict", "--train"):
            if mode is not None:
                raise UsageError("only one of --predict and --train can be given")
            mode = arg.removeprefix("--")
        elif arg == "--save":
            if save_file is not None:
                raise UsageError("--save given twice")
            save_file = next(args, None)
            if save_file is None:
                raise UsageError("--save needs a SAVEFILE")
        elif arg.startswith("--"):
            raise UsageError(f"unknown option {arg}")
        else:
            positional.append(arg)
    if mode is None:
        raise UsageError("expected --predict or --train")
    if save_file is not None and mode != "train":
        raise UsageError("--save only works in train mode")
    if len(positional) != 2:
        raise UsageError("expected LOADFILE and CHESSFILE")
    return Arguments(mode, positional[0], positional[1], save_file)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv in (["--help"], ["-h"]):
        print(USAGE)
        return 0
    try:
        args = parse_args(argv)
    except UsageError as error:
        print(f"my_torch_analyzer: {error} (see --help)", file=sys.stderr)
        return EXIT_ERROR
    print(f"my_torch_analyzer: --{args.mode} is not implemented yet", file=sys.stderr)
    return EXIT_ERROR
