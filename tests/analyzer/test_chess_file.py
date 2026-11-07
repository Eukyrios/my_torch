import pytest

from analyzer.chess_file import ChessFileError, check_expected_outputs, read_chess_file
from analyzer.fen import parse_fen

from tests.analyzer.conftest import BLACK_TO_MOVE, WHITE_TO_MOVE


def test_reads_every_board_in_order_with_its_expected_output(write_boards):
    boards = read_chess_file(write_boards([WHITE_TO_MOVE, "", f"{BLACK_TO_MOVE} Checkmate Black", "   "]))
    assert [board.position for board in boards] == [parse_fen(WHITE_TO_MOVE), parse_fen(BLACK_TO_MOVE)]
    assert [board.expected for board in boards] == [None, "Checkmate Black"]
    assert [board.line for board in boards] == [1, 3]


def test_an_empty_file_has_no_board(write_boards):
    assert read_chess_file(write_boards([])) == []


@pytest.mark.parametrize(
    "line, message",
    [
        ("8/8/8/8/8/8/8/k1K5 w", "line 2: expected at least 6 fields"),
        ("8/8/8/8/8/8/8/k1K4 w - - 0 1", "line 2: "),
        (f"{WHITE_TO_MOVE} Draw", "line 2: unknown label 'Draw'"),
        (f"{WHITE_TO_MOVE} Stalemate White", "line 2: unknown label 'Stalemate White'"),
        (f"{WHITE_TO_MOVE} Check Green", "line 2: unknown label 'Check Green'"),
    ],
)
def test_a_bad_line_is_refused_with_its_number(write_boards, line, message):
    path = write_boards([WHITE_TO_MOVE, line])
    with pytest.raises(ChessFileError, match=message) as error:
        read_chess_file(path)
    assert str(error.value).startswith(path)


def test_a_file_that_cannot_be_read_is_refused(tmp_path):
    with pytest.raises(ChessFileError, match="cannot read the chess file .*No such file"):
        read_chess_file(str(tmp_path / "missing.txt"))
    with pytest.raises(ChessFileError, match="cannot read the chess file"):
        read_chess_file(str(tmp_path))
    (tmp_path / "binary.txt").write_bytes(b"\xff\xfe\x00")
    with pytest.raises(ChessFileError, match="cannot read the chess file"):
        read_chess_file(str(tmp_path / "binary.txt"))


def test_training_needs_every_expected_output(write_boards):
    path = write_boards([f"{WHITE_TO_MOVE} Nothing", BLACK_TO_MOVE])
    boards = read_chess_file(path)
    with pytest.raises(ChessFileError, match="line 2: training needs the expected output"):
        check_expected_outputs(boards, path)
    check_expected_outputs(boards[:1], path)
