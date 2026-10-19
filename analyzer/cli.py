# Command line of my_torch_analyzer: argument parsing, --help and the exit codes.
# Parsing is two passes: read_arguments collects the options in any order,
# then validate_arguments checks the combination and builds the final Arguments.

import sys
from dataclasses import dataclass, field

EXIT_SUCCESS = 0
EXIT_ERROR = 84

HELP_FLAGS = (["--help"], ["-h"])
MODE_FLAGS = ("--predict", "--train")
SAVE_FLAG = "--save"

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


@dataclass
class RawArguments:
    mode: str | None = None
    save_file: str | None = None
    files: list[str] = field(default_factory=list)


def read_mode(raw: RawArguments, flag: str) -> None:
    if raw.mode is not None:
        raise UsageError("only one of --predict and --train can be given")
    raw.mode = flag.removeprefix("--")


def read_save_file(raw: RawArguments, remaining) -> None:
    if raw.save_file is not None:
        raise UsageError("--save given twice")
    raw.save_file = next(remaining, None)
    if raw.save_file is None:
        raise UsageError("--save needs a SAVEFILE")


def read_arguments(argv: list[str]) -> RawArguments:
    raw = RawArguments()
    remaining = iter(argv)
    for arg in remaining:
        if arg in MODE_FLAGS:
            read_mode(raw, arg)
        elif arg == SAVE_FLAG:
            read_save_file(raw, remaining)
        elif arg.startswith("--"):
            raise UsageError(f"unknown option {arg}")
        else:
            raw.files.append(arg)
    return raw


def check_mode_is_given(raw: RawArguments) -> None:
    if raw.mode is None:
        raise UsageError("expected --predict or --train")


def check_save_only_in_train_mode(raw: RawArguments) -> None:
    if raw.save_file is not None and raw.mode != "train":
        raise UsageError("--save only works in train mode")


def check_two_files_are_given(raw: RawArguments) -> None:
    if len(raw.files) != 2:
        raise UsageError("expected LOADFILE and CHESSFILE")


def validate_arguments(raw: RawArguments) -> Arguments:
    check_mode_is_given(raw)
    check_save_only_in_train_mode(raw)
    check_two_files_are_given(raw)
    load_file, chess_file = raw.files
    return Arguments(raw.mode, load_file, chess_file, raw.save_file)


def parse_args(argv: list[str]) -> Arguments:
    return validate_arguments(read_arguments(argv))


def is_help_request(argv: list[str]) -> bool:
    return argv in HELP_FLAGS


def print_error(message: str) -> None:
    print(f"my_torch_analyzer: {message}", file=sys.stderr)


def run(args: Arguments) -> int:
    print_error(f"--{args.mode} is not implemented yet")
    return EXIT_ERROR


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if is_help_request(argv):
        print(USAGE)
        return EXIT_SUCCESS
    try:
        args = parse_args(argv)
    except UsageError as error:
        print_error(f"{error} (see --help)")
        return EXIT_ERROR
    return run(args)
