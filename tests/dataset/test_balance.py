import random

from dataset.balance import balance


def test_every_class_is_cut_to_the_smallest():
    result = balance({"a": [1, 2, 3, 4], "b": [5, 6]}, random.Random(0))
    assert len(result) == 4
    assert len([x for x in result if x <= 4]) == 2
    assert len([x for x in result if x >= 5]) == 2


def test_same_seed_gives_same_result():
    data = {"a": list(range(50)), "b": list(range(50, 80))}
    assert balance(data, random.Random(3)) == balance(data, random.Random(3))


def test_an_empty_class_empties_everything():
    assert balance({"a": [1, 2], "b": []}, random.Random(0)) == []
