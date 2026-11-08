# Command line of my_torch_analyzer: argument parsing in two passes (read, then validate),
# --help, the two modes and the exit codes. --predict and --train read the chess file here and
# leave the network to my_torch_engine, the C++ side.

import sys

from analyzer.chess_file import ChessFileError, check_expected_outputs, read_chess_file
from analyzer.engine import EngineError, network_labels, predict_outputs, train
from analyzer.prediction import predicted_labels
from analyzer.report import build_report, expected_as, format_report

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


class Arguments:
    def __init__(self, mode: str, load_file: str, chess_file: str, save_file: str | None = None):
        self.mode = mode
        self.load_file = load_file
        self.chess_file = chess_file
        self.save_file = save_file

    def fields(self) -> tuple:
        return (self.mode, self.load_file, self.chess_file, self.save_file)

    def __eq__(self, other) -> bool:
        return isinstance(other, Arguments) and self.fields() == other.fields()

    def __repr__(self) -> str:
        return f"Arguments{self.fields()!r}"


class RawArguments:
    def __init__(self):
        self.mode: str | None = None
        self.save_file: str | None = None
        self.files: list[str] = []


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


def run_predict(args: Arguments) -> None:
    # stdout carries the predictions only, one per board; the accuracy report goes to stderr
    boards = read_chess_file(args.chess_file)
    labels = network_labels(args.load_file)
    predicted = predicted_labels(predict_outputs(args.load_file, boards, labels), labels)
    if predicted:
        print("\n".join(predicted))
    checked = [(board.expected, label) for board, label in zip(boards, predicted) if board.expected is not None]
    if checked:
        expected, guesses = zip(*checked)
        print(format_report(build_report(list(expected), list(guesses), labels)), file=sys.stderr)


def run_train(args: Arguments) -> None:
    boards = read_chess_file(args.chess_file)
    check_expected_outputs(boards, args.chess_file)
    labels = network_labels(args.load_file)
    classes = []
    for board in boards:
        try:
            classes.append(labels.index(expected_as(board.expected, labels)))
        except ValueError as error:
            raise ChessFileError(f"{args.chess_file} line {board.line}: {error}")
    train(args.load_file, args.save_file or args.load_file, boards, classes, labels)


def run(args: Arguments) -> int:
    try:
        if args.mode == "predict":
            run_predict(args)
        else:
            run_train(args)
    except (ChessFileError, EngineError, ValueError) as error:
        print_error(str(error))
        return EXIT_ERROR
    return EXIT_SUCCESS


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
