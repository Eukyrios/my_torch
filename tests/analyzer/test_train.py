import numpy as np

from analyzer.cli import EXIT_SUCCESS, main
from analyzer.train import XOR_INPUTS, XOR_OUTPUTS, build_xor_network, train_xor, xor_report


def test_train_xor_learns_the_four_cases():
    network, losses = train_xor()
    assert list(network(XOR_INPUTS).argmax(axis=1)) == list(XOR_OUTPUTS)
    assert losses[-1] < losses[0]


def test_train_xor_runs_the_requested_number_of_epochs():
    _, losses = train_xor(epochs=7)
    assert len(losses) == 7


def test_the_xor_network_has_2_inputs_and_2_outputs():
    assert build_xor_network(np.random.default_rng(0)).shape == (2, 2)


def test_report_shows_the_loss_and_every_case():
    network, losses = train_xor()
    report = xor_report(network, losses)
    assert report.splitlines()[0].startswith("trained on XOR for 500 epochs: loss ")
    assert "0 xor 1 = 1 (expected 1" in report
    assert report.count(" ok") == 4


def test_report_marks_wrong_predictions():
    network, losses = train_xor(epochs=1)
    report = xor_report(network, losses)
    assert report.count(" ok") + report.count(" wrong") == 4


def test_train_mode_trains_on_xor_and_exits_0(capsys):
    assert main(["--train", "net.nn", "boards.txt"]) == EXIT_SUCCESS
    output = capsys.readouterr().out
    assert "trained on XOR" in output and output.count(" ok") == 4
