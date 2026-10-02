"""Frozen train-only sequence replay; no annotations enter model prediction."""

import hashlib
from collections import Counter

import numpy as np

ARMS = ("uniform", "failure_aware")
BATCH_SIZE = 8
UPDATES_PER_EPOCH = 320


def validate_partition(shards, partition):
    ids = [s["sequence_id"] for s in shards]
    if (
        len(ids) != len(set(ids))
        or not ids
        or any(type(s["pairs"]) is not int or s["pairs"] < 4 for s in shards)
    ):
        raise ValueError("invalid or duplicate train shards")
    if set(partition) != {"passing", "failing"}:
        raise ValueError("expected passing/failing partition")
    grouped = partition["passing"] + partition["failing"]
    if (
        len(grouped) != len(set(grouped))
        or set(grouped) != set(ids)
        or min(map(len, partition.values())) < BATCH_SIZE // 2
    ):
        raise ValueError("partition must cover train exactly with at least four per group")


def frame_stream(length, start, count, sid, seed):
    """Address a random permutation stream without resetting at epoch boundaries."""
    stable = int.from_bytes(hashlib.sha256(sid.encode()).digest()[:8], "little")
    positions = np.arange(start, start + count, dtype=np.int64)
    cycles, offsets = np.divmod(positions, length)
    result = np.empty(count, dtype=np.int64)
    for cycle in np.unique(cycles):
        rng = np.random.default_rng(np.random.SeedSequence([seed, stable, int(cycle)]))
        mask = cycles == cycle
        result[mask] = rng.permutation(length)[offsets[mask]]
    return result.tolist()


def epoch_selection(shards, partition, arm, epoch, seed=0):
    """320 batches: uniform8, or failing4+passing4; both 2560 frame draws.

    Each group follows a seeded fixed sequence permutation. A sequence's frame
    stream advances by its actual cumulative draw count, including unequal
    epoch counts. Thus all its pairs are visited before the next shuffled cycle.
    Sampling frequencies differ intentionally; this is NOT matched-data training.
    """
    validate_partition(shards, partition)
    if arm not in ARMS or type(epoch) is not int or epoch < 0 or type(seed) is not int or seed < 0:
        raise ValueError("invalid sampler arm, epoch or seed")
    lengths = {s["sequence_id"]: s["pairs"] for s in shards}
    groups = (
        [sorted(lengths)]
        if arm == "uniform"
        else [sorted(partition["failing"]), sorted(partition["passing"])]
    )
    size = BATCH_SIZE // len(groups)
    batches = [[] for _ in range(UPDATES_PER_EPOCH)]
    for group_index, group in enumerate(groups):
        rng = np.random.default_rng(np.random.SeedSequence([seed, group_index, 719]))
        order = [group[i] for i in rng.permutation(len(group))]
        before = epoch * UPDATES_PER_EPOCH * size
        drawn = [order[i % len(order)] for i in range(before, before + UPDATES_PER_EPOCH * size)]
        counts = Counter(drawn)
        streams = {}
        for rank, sid in enumerate(order):
            start = before // len(order) + int(rank < before % len(order))
            streams[sid] = iter(frame_stream(lengths[sid], start, counts[sid], sid, seed))
        for i, sid in enumerate(drawn):
            batches[i // size].append((sid, next(streams[sid])))
    return [sample for batch in batches for sample in batch]
