import json
from collections import Counter, defaultdict

import pytest
import torch

from aero_ir.registration.replay_sampling import ARMS, epoch_selection, frame_stream
from aero_ir.utils.manifest import canonical_hash
from scripts.train_registration_replay import (
    EMPTY_CHAIN,
    build_model,
    load_checkpoint,
    save_checkpoint,
    schedule,
    verify_products,
    verify_trace,
)


def fixture():
    shards = [{"sequence_id": f"s{i:03d}", "pairs": 101} for i in range(160)]
    partition = {
        "failing": [s["sequence_id"] for s in shards[:43]],
        "passing": [s["sequence_id"] for s in shards[43:]],
    }
    return shards, partition


@pytest.mark.parametrize("arm", ARMS)
def test_balanced_stream_budget_determinism_and_coverage(arm):
    shards, partition = fixture()
    stream = defaultdict(list)
    for epoch in range(12):
        selected = epoch_selection(shards, partition, arm, epoch)
        assert len(selected) == 2560
        assert selected == epoch_selection(list(reversed(shards)), partition, arm, epoch)
        for i in range(0, len(selected), 8):
            batch = selected[i : i + 8]
            assert len({sid for sid, _ in batch}) == 8
            if arm == "failure_aware":
                assert sum(sid in partition["failing"] for sid, _ in batch) == 4
        counts = Counter(sid for sid, _ in selected)
        if arm == "uniform":
            assert set(counts.values()) == {16}
        else:
            assert set(counts[s] for s in partition["failing"]) == {29, 30}
            assert set(counts[s] for s in partition["passing"]) == {10, 11}
        for sid, index in selected:
            stream[sid].append(index)
    for sid, indices in stream.items():
        assert sorted(indices[:101]) == list(range(101))
        assert indices == frame_stream(101, 0, len(indices), sid, 0)
    assert epoch_selection(shards, partition, arm, 0, 0) != epoch_selection(
        shards, partition, arm, 0, 1
    )


def test_sampler_rejects_invalid_partition_and_epoch():
    shards, partition = fixture()
    bad = {"passing": partition["passing"] + ["not_train"], "failing": partition["failing"]}
    with pytest.raises(ValueError, match="cover train"):
        epoch_selection(shards, bad, "uniform", 0)
    with pytest.raises(ValueError, match="sampler"):
        epoch_selection(shards, partition, "uniform", -1)
    with pytest.raises(ValueError, match="duplicate"):
        epoch_selection(shards + shards[:1], partition, "uniform", 0)


def trace_fixture():
    shards, partition = fixture()
    spec = {
        "epochs": 2,
        "updates_per_epoch": 2,
        "batch_size": 8,
        "seed": 0,
        "arm": "failure_aware",
        "partition": partition,
    }
    rows = []
    for epoch in range(2):
        selected = schedule(spec, shards, epoch)
        for batch in range(2):
            rows.append(
                {
                    "epoch": epoch + 1,
                    "batch": batch + 1,
                    "selection": selected[batch * 8 : (batch + 1) * 8],
                    "array_sha256": "a" * 64,
                    "loss": {"total": 0.3, "gradient_norm": 0.5},
                }
            )
    return spec, shards, rows


def write_attempt(root, spec, attempt, start, rows, partial=False):
    (root / f"attempt_{attempt:03d}_runtime.json").write_text(
        json.dumps({"spec_sha256": canonical_hash(spec), "resume_from_epoch": start})
    )
    (root / f"attempt_{attempt:03d}.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows) + ('{"epoch":' if partial else "")
    )


def test_trace_resume_discards_only_uncommitted_updates(tmp_path):
    spec, shards, rows = trace_fixture()
    write_attempt(tmp_path, spec, 0, 0, rows[:3], partial=True)
    write_attempt(tmp_path, spec, 1, 1, rows[2:])
    result = verify_trace(tmp_path, spec, shards, 4)
    assert result["abandoned_logged_steps"] == 2
    chain = EMPTY_CHAIN
    for row in rows:
        chain = canonical_hash({"previous": chain, "row": row})
    assert result["committed_chain"] == chain
    assert result["verified_optimizer_steps"] == 4


def test_trace_rejects_wrong_samples_nonfinite_and_missing_updates(tmp_path):
    spec, shards, rows = trace_fixture()
    write_attempt(tmp_path, spec, 0, 0, rows[:3])
    with pytest.raises(ValueError, match="budget"):
        verify_trace(tmp_path, spec, shards, 4)
    rows[0]["selection"][0] = ("not_train", 0)
    write_attempt(tmp_path, spec, 0, 0, rows)
    with pytest.raises(ValueError, match="selection"):
        verify_trace(tmp_path, spec, shards, 4)
    spec, shards, rows = trace_fixture()
    rows[0]["loss"]["total"] = float("nan")
    write_attempt(tmp_path, spec, 0, 0, rows)
    with pytest.raises(ValueError, match="metric"):
        verify_trace(tmp_path, spec, shards, 4)


def test_checkpoint_roundtrip_and_next_update_matches_resume(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(0)
    spec = {"epochs": 2, "updates_per_epoch": 1}
    model = build_model("residual_head")
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 2)

    def update(m, o, s):
        o.zero_grad()
        sum(p.square().mean() for p in m.head.parameters()).backward()
        o.step()
        s.step()

    update(model, optimizer, scheduler)
    path = tmp_path / "latest.pth"
    save_checkpoint(path, model, optimizer, scheduler, 1, spec, torch.device("cpu"), "a" * 64)
    restored, state = load_checkpoint(path, spec, torch.device("cpu"))
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in model.state_dict().items())
    assert not any(p.requires_grad for p in restored.base_model.parameters())
    resumed = torch.optim.AdamW(restored.head.parameters(), lr=1e-4)
    resumed.load_state_dict(state["optimizer"])
    rs = torch.optim.lr_scheduler.CosineAnnealingLR(resumed, 2)
    rs.load_state_dict(state["scheduler"])
    update(model, optimizer, scheduler)
    update(restored, resumed, rs)
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in model.state_dict().items())
    with pytest.raises(ValueError, match="protocol"):
        load_checkpoint(path, {**spec, "epochs": 3}, torch.device("cpu"))


def test_smoke_cannot_be_certified_as_full_training(tmp_path):
    with pytest.raises(ValueError, match="full300"):
        verify_products(tmp_path, tmp_path, {"smoke": True}, [])
