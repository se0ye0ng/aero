from collections import Counter

import torch

from scripts.run_flir_mixture_gpu_smoke import batches, digest_state


def test_matched_complete_effective_batch():
    selected = batches()
    assert len(selected) == 8 and all(len(b) == 8 for b in selected)
    assert Counter(i for b in selected for i in b) == {i: 2 for i in range(32)}
    assert selected == batches()


def test_state_identity_binds_values_and_shapes():
    a = {"weight": torch.tensor([1.0, 2.0])}
    assert digest_state(a) == digest_state({"weight": a["weight"].clone()})
    assert digest_state(a) != digest_state({"weight": torch.tensor([[1.0, 2.0]])})
    assert digest_state(a) != digest_state({"weight": torch.tensor([1.0, 3.0])})
