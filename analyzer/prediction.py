# Turns the network outputs into game states. The network gives one score per label (one row
# per board), the highest score is the prediction, and the state is its label without the
# colour: "Checkmate", "Check" or "Nothing".

import numpy as np

# same order as the dataset files (dataset.labels), which the network learns from
LABELS = ("Nothing", "Check White", "Check Black", "Checkmate White", "Checkmate Black")
STATES = ("Checkmate", "Check", "Nothing")


def check_outputs(outputs: np.ndarray, labels: tuple[str, ...]) -> None:
    if outputs.ndim not in (1, 2) or outputs.shape[-1] != len(labels):
        raise ValueError(
            f"expected {len(labels)} outputs per board, one per label, as (n_labels,) or (boards, n_labels), "
            f"got shape {outputs.shape}"
        )


def predicted_labels(outputs, labels: tuple[str, ...] = LABELS) -> list[str]:
    outputs = np.asarray(outputs, dtype=float)
    check_outputs(outputs, labels)
    return [labels[index] for index in np.atleast_2d(outputs).argmax(axis=1)]


def state_of(label: str) -> str:
    state = label.split()[0]
    if state not in STATES:
        raise ValueError(f"unknown label {label!r}, expected one of {', '.join(STATES)} with an optional colour")
    return state


def predicted_states(outputs, labels: tuple[str, ...] = LABELS) -> list[str]:
    return [state_of(label) for label in predicted_labels(outputs, labels)]
