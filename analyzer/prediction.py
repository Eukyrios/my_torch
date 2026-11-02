# Turns the network outputs into game states. The network gives one score per label (one row
# per board) and the highest score is the prediction. A label is a state, "Checkmate", "Check"
# or "Nothing", followed for checks and checkmates by the colour giving it, "White" or "Black":
# the full output is the label itself, the state alone drops the colour.

import numpy as np

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


def check_outputs(outputs: np.ndarray, labels: tuple[str, ...]) -> None:
    if outputs.ndim not in (1, 2) or outputs.shape[-1] != len(labels):
        raise ValueError(
            f"expected {len(labels)} outputs per board, one per label, as (n_labels,) or (boards, n_labels), "
            f"got shape {outputs.shape}"
        )


def predicted_labels(outputs, labels: tuple[str, ...] = LABELS) -> list[str]:
    for label in labels:
        check_label(label)
    outputs = np.asarray(outputs, dtype=float)
    check_outputs(outputs, labels)
    return [labels[index] for index in np.atleast_2d(outputs).argmax(axis=1)]


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
