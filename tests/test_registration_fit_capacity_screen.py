import pytest

from scripts.screen_registration_fit_capacity import KIND, verified_shared_head


def test_shared_screen_rejects_wrong_kind_and_unbound_spec(tmp_path):
    with pytest.raises(ValueError, match="not a fit-capacity"):
        verified_shared_head(tmp_path, {"kind": "other"}, {"kind": KIND})
    with pytest.raises(ValueError, match="specification"):
        verified_shared_head(tmp_path, {"kind": KIND, "spec_sha256": "bad"}, {"kind": KIND})
