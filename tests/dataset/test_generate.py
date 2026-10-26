import pytest

from analyzer import parse_fen, split_fen_line
from dataset import generate
from dataset.generate import Collector, collect, describe, main, write_splits
from dataset.labels import LABELS

TRAIN_ONLY = (1, 0, 0)


def run(pgn_file, out, *extra):
    return main([str(pgn_file), "--out", str(out), "--per-class", "1", "--ratios", "1", "0", "0", *extra])


def test_collector_ignores_duplicate_positions():
    collector = Collector({"train": 5, "validation": 5, "test": 5})
    fen = "8/8/8/8/8/8/8/k1K5 w - - 0 1"
    collector.add("train", "Nothing", fen)
    collector.add("validation", "Nothing", fen.replace(" 0 1", " 3 9"))
    assert len(collector.lines["train"]["Nothing"]) == 1
    assert collector.lines["validation"]["Nothing"] == []


def test_collector_rejects_invalid_fen():
    collector = Collector({"train": 5, "validation": 5, "test": 5})
    collector.add("train", "Nothing", "8/8/8/8/8/8/8/8 w - - 0 1")
    assert collector.rejected == 1
    assert collector.lines["train"]["Nothing"] == []


def test_collector_stops_at_the_quota():
    collector = Collector({"train": 1, "validation": 0, "test": 0})
    collector.add("train", "Nothing", "8/8/8/8/8/8/8/k1K5 w - - 0 1")
    collector.add("train", "Nothing", "8/8/8/8/8/8/8/1k1K4 w - - 0 1")
    assert len(collector.lines["train"]["Nothing"]) == 1
    assert collector.split_is_full("train") is False


def test_collect_fills_every_class(pgn_file):
    collector = collect(pgn_file, per_class=1, ratios=TRAIN_ONLY, per_game=5, seed=1)
    assert collector.is_full()
    assert all(len(collector.lines["train"][label]) == 1 for label in LABELS)
    assert collector.shortfalls() == []


def test_collect_reports_missing_positions(pgn_file):
    collector = collect(pgn_file, per_class=50, ratios=TRAIN_ONLY, per_game=5, seed=1)
    assert not collector.is_full()
    assert ("train", "Checkmate White", 1, 50) in collector.shortfalls()


def test_collect_prints_progress(pgn_file, monkeypatch, capsys):
    monkeypatch.setattr(generate, "PROGRESS_EVERY", 1)
    collect(pgn_file, per_class=50, ratios=TRAIN_ONLY, per_game=5, seed=1)
    assert "games read" in capsys.readouterr().err


def test_write_splits_balances_and_writes_files(pgn_file, tmp_path):
    collector = collect(pgn_file, per_class=1, ratios=TRAIN_ONLY, per_game=5, seed=1)
    written = write_splits(collector, tmp_path / "out", seed=1)
    assert len(written["train"]) == 5
    assert written["validation"] == [] and written["test"] == []
    assert (tmp_path / "out" / "train.txt").read_text().count("\n") == 5


def test_describe_counts_labels_and_pieces():
    lines = ["8/8/8/8/8/8/8/k1K5 w - - 0 1 Nothing", "8/8/8/8/8/8/8/kPK5 w - - 0 1 Nothing"]
    stats = describe(lines)
    assert stats["Nothing"] == (2, 2.5)
    assert stats["Check White"] == (0, 0.0)


def test_main_writes_valid_labelled_lines(pgn_file, tmp_path, capsys):
    assert run(pgn_file, tmp_path / "data") == 0
    lines = (tmp_path / "data" / "train.txt").read_text().splitlines()
    assert len(lines) == 5
    for line in lines:
        fen, label = split_fen_line(line)
        parse_fen(fen)
        assert label in LABELS
    assert "mean pieces" in capsys.readouterr().out


def test_same_seed_gives_identical_files(pgn_file, tmp_path):
    run(pgn_file, tmp_path / "a", "--seed", "9")
    run(pgn_file, tmp_path / "b", "--seed", "9")
    assert (tmp_path / "a" / "train.txt").read_text() == (tmp_path / "b" / "train.txt").read_text()


def test_main_warns_when_a_class_is_short(pgn_file, tmp_path, capsys):
    main([str(pgn_file), "--out", str(tmp_path / "d"), "--per-class", "9", "--ratios", "1", "0", "0"])
    assert "warning: train / Checkmate White" in capsys.readouterr().err


def test_missing_file_exits_with_84(tmp_path, capsys):
    assert main([str(tmp_path / "missing.pgn"), "--out", str(tmp_path / "d")]) == 84
    assert "dataset:" in capsys.readouterr().err


@pytest.mark.parametrize(
    "bad",
    [["--per-class", "0"], ["--per-game", "0"], ["--ratios", "0", "0", "0"], ["--ratios", "-1", "1", "1"]],
)
def test_bad_options_are_rejected(pgn_file, bad):
    with pytest.raises(SystemExit):
        main([str(pgn_file), *bad])
