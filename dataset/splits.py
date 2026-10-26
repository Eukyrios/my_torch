import hashlib
from bisect import bisect_right
from itertools import accumulate

SPLITS = ("train", "validation", "test")


def game_digest(key, seed):
    data = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return int.from_bytes(data[:8], "big")


def split_of(digest, ratios):
    edges = list(accumulate(ratios))
    point = digest % 10_000 * edges[-1] / 10_000
    return SPLITS[bisect_right(edges, point)]


def quotas(per_class, ratios):
    total = sum(ratios)
    return {split: round(per_class * ratio / total) for split, ratio in zip(SPLITS, ratios)}
