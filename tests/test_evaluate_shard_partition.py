"""Sharded evaluation must cover every result exactly once.

Two processes writing the same result.json is the failure this guards: an
overlapping partition races on the file, and a partition with a gap silently
leaves results unevaluated while both shards report success.
"""
import zlib
import pytest


def _members(entries, idx, total):
    return [e for e in entries if zlib.crc32(e.encode()) % total == idx]


@pytest.mark.parametrize("total", [2, 3, 6, 8])
def test_partition_is_exact_cover(total):
    entries = [f"{i:016x}" for i in range(1000)]
    seen = []
    for idx in range(total):
        seen.extend(_members(entries, idx, total))
    assert sorted(seen) == sorted(entries), "shards must cover every entry exactly once"


@pytest.mark.parametrize("total", [2, 6])
def test_shards_are_disjoint(total):
    entries = [f"{i:016x}" for i in range(1000)]
    sets = [set(_members(entries, i, total)) for i in range(total)]
    for i in range(total):
        for j in range(i + 1, total):
            assert not (sets[i] & sets[j]), f"shard {i} and {j} overlap -- they would race on the same result.json"


def test_membership_is_stable_under_listing_changes():
    """A later pass with new results must not move an existing entry's shard."""
    before = [f"{i:016x}" for i in range(100)]
    after = before + [f"{i:016x}" for i in range(100, 150)]
    m1 = _members(before, 2, 6)
    m2 = [e for e in _members(after, 2, 6) if e in before]
    assert m1 == m2


def test_shards_are_roughly_balanced():
    entries = [f"{i:016x}" for i in range(6000)]
    sizes = [len(_members(entries, i, 6)) for i in range(6)]
    assert max(sizes) < 2 * min(sizes), f"badly skewed partition: {sizes}"
