from dataset.splits import SPLITS, game_digest, quotas, split_of


def test_digest_is_stable_and_depends_on_seed():
    assert game_digest("game", 1) == game_digest("game", 1)
    assert game_digest("game", 1) != game_digest("game", 2)
    assert game_digest("game", 1) != game_digest("other", 1)


def test_single_split_ratios_send_everything_there():
    digests = [game_digest(i, 0) for i in range(500)]
    assert {split_of(d, (1, 0, 0)) for d in digests} == {"train"}
    assert {split_of(d, (0, 1, 0)) for d in digests} == {"validation"}
    assert {split_of(d, (0, 0, 1)) for d in digests} == {"test"}


def test_split_proportions_follow_the_ratios():
    total = 20_000
    picks = [split_of(game_digest(i, 7), (80, 10, 10)) for i in range(total)]
    for split, expected in zip(SPLITS, (0.8, 0.1, 0.1)):
        assert abs(picks.count(split) / total - expected) < 0.02


def test_quotas_are_shared_out_by_ratio():
    assert quotas(1000, (80, 10, 10)) == {"train": 800, "validation": 100, "test": 100}
    assert quotas(10, (1, 0, 0)) == {"train": 10, "validation": 0, "test": 0}
