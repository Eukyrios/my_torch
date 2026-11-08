# Accuracy report of --predict when the chess file gives the expected outputs: the share of
# boards predicted right, the confusion matrix (how many boards of each expected label got each
# predicted label) and the precision and recall of every label. It goes to stderr, as stdout
# only carries the predictions, one per board, which is what the autograder reads.

from analyzer.prediction import LABELS, check_label, state_of


def expected_as(label: str, labels: tuple[str, ...]) -> str:
    # a network without colours is judged on the state alone: "Check White" expected, "Check" predicted
    check_label(label)
    if label in labels:
        return label
    if state_of(label) in labels:
        return state_of(label)
    raise ValueError(f"expected output {label!r} is not one the network predicts ({', '.join(labels)})")


def ratio(part: int, whole: int) -> float | None:
    return part / whole if whole else None


class Report:
    def __init__(self, labels: tuple[str, ...], confusion: tuple[tuple[int, ...], ...]):
        self.labels = labels
        self.confusion = confusion  # confusion[expected][predicted]

    def total(self) -> int:
        return sum(map(sum, self.confusion))

    def correct(self) -> int:
        return sum(self.confusion[index][index] for index in range(len(self.labels)))

    def accuracy(self) -> float | None:
        return ratio(self.correct(), self.total())

    def expected_count(self, index: int) -> int:
        return sum(self.confusion[index])

    def predicted_count(self, index: int) -> int:
        return sum(row[index] for row in self.confusion)

    def precision(self, index: int) -> float | None:
        # of the boards predicted as this label, the share that really were
        return ratio(self.confusion[index][index], self.predicted_count(index))

    def recall(self, index: int) -> float | None:
        # of the boards that really were this label, the share predicted as such
        return ratio(self.confusion[index][index], self.expected_count(index))


def build_report(expected: list[str], predicted: list[str], labels: tuple[str, ...] = LABELS) -> Report:
    if len(expected) != len(predicted):
        raise ValueError(f"expected one prediction per board, got {len(predicted)} for {len(expected)} boards")
    index = {label: position for position, label in enumerate(labels)}
    confusion = [[0] * len(labels) for _ in labels]
    for expected_label, predicted_label in zip(expected, predicted):
        if predicted_label not in index:
            raise ValueError(f"predicted output {predicted_label!r} is not one of {', '.join(labels)}")
        confusion[index[expected_as(expected_label, labels)]][index[predicted_label]] += 1
    return Report(tuple(labels), tuple(map(tuple, confusion)))


def percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def format_table(header: list[str], rows: list[list[str]]) -> list[str]:
    # first column on the left, the others on the right, each as wide as its longest cell
    widths = [max(len(row[column]) for row in [header] + rows) for column in range(len(header))]
    lines = []
    for row in [header] + rows:
        cells = [row[0].ljust(widths[0])] + [cell.rjust(width) for cell, width in zip(row[1:], widths[1:])]
        lines.append("  ".join(cells).rstrip())
    return lines


def format_report(report: Report) -> str:
    labels = list(report.labels)
    confusion = [[label] + [str(count) for count in row] for label, row in zip(labels, report.confusion)]
    scores = [
        [label, percent(report.precision(index)), percent(report.recall(index)), str(report.expected_count(index))]
        for index, label in enumerate(labels)
    ]
    lines = [f"accuracy: {percent(report.accuracy())} ({report.correct()}/{report.total()} boards)", ""]
    lines.append("confusion matrix (rows: expected, columns: predicted)")
    lines += format_table(["expected \\ predicted"] + labels, confusion)
    lines.append("")
    lines += format_table(["label", "precision", "recall", "boards"], scores)
    return "\n".join(lines)
