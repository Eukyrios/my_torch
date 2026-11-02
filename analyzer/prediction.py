# Turns the network outputs into game states. The network gives one score per label, as one
# row of numbers for one board or a list of rows for several, and the highest score is the
# prediction. A label is a state, "Checkmate", "Check" or "Nothing", followed for checks and
# checkmates by the colour giving it, "White" or "Black": the full output is the label
# itself, the state alone drops the colour. Plain Python: the analyzer has no dependency.

# same order as the dataset files (dataset.labels), which the network learns from
LABELS = ("Nothing", "Check White", "Check Black", "Checkmate White", "Checkmate Black")
STATES = ("Checkmate", "Check", "Nothing")
COLOURS = ("White", "Black")
NO_COLOUR_STATE = "Nothing"


def is_label(label: str) -> bool:
    words = label.split(" ")
    if words[0] not in STATES or len(words) > 2:
        return False
    return len(words) == 1 or (words[0] != NO_COLOUR_STATE and words[1] in COLOURS)


def check_label(label: str) -> None:
    if not is_label(label):
        raise ValueError(
            f"unknown label {label!r}, expected one of {', '.join(STATES)}, "
            f"followed by {' or '.join(COLOURS)} for a check or a checkmate"
        )


def is_score(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def as_rows(outputs) -> list:
    outputs = list(outputs)
    single_board = len(outputs) > 0 and all(is_score(value) for value in outputs)
    return [outputs] if single_board else outputs


def check_row(row, labels: tuple[str, ...]) -> None:
    if not isinstance(row, (list, tuple)) or len(row) != len(labels) or not all(is_score(value) for value in row):
        raise ValueError(f"expected {len(labels)} scores per board, one per label, got {row!r}")


def best_index(row) -> int:
    # the first one on a tie
    return max(range(len(row)), key=row.__getitem__)


def predicted_labels(outputs, labels: tuple[str, ...] = LABELS) -> list[str]:
    for label in labels:
        check_label(label)
    rows = as_rows(outputs)
    for row in rows:
        check_row(row, labels)
    return [labels[best_index(row)] for row in rows]


def state_of(label: str) -> str:
    check_label(label)
    return label.split(" ")[0]


def colour_of(label: str) -> str | None:
    check_label(label)
    words = label.split(" ")
    return words[1] if len(words) == 2 else None


def predicted_states(outputs, labels: tuple[str, ...] = LABELS) -> list[str]:
    return [state_of(label) for label in predicted_labels(outputs, labels)]


def prediction_report(outputs, labels: tuple[str, ...] = LABELS) -> str:
    # one line per board, in the order of the chess file, as --predict prints them
    return "\n".join(predicted_labels(outputs, labels))
