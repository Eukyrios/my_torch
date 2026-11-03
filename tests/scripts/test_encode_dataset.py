from scripts.encode_dataset import EXIT_ERROR, encoded_line, main

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
MATE = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"


def test_a_line_starts_with_the_class_index_then_the_active_inputs():
    fields = encoded_line(f"{START} Nothing").split()
    assert fields[0] == "0"
    assert len(fields) == 1 + 32 + 1
    assert fields[-1] == "768"


def test_labels_with_a_colour_map_to_their_index():
    assert encoded_line(f"{MATE} Checkmate Black").split()[0] == "4"


def test_main_encodes_every_split(tmp_path):
    for split in ("train", "validation", "test"):
        (tmp_path / f"{split}.txt").write_text(f"{START} Nothing\n{MATE} Checkmate Black\n")
    assert main([str(tmp_path)]) == 0
    lines = (tmp_path / "encoded" / "train.txt").read_text().splitlines()
    assert lines[0] == "769 5" and len(lines) == 3


def test_a_missing_split_exits_84(tmp_path, capsys):
    assert main([str(tmp_path)]) == EXIT_ERROR
    assert "encode_dataset:" in capsys.readouterr().err
