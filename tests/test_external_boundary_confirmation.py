import numpy as np

from scripts.probe_external_boundary_confirmation import aggregate
from scripts.probe_external_registration_landmarks import summarize


def test_duplicate_labels_retained_in_primary_and_stratified():
    fail = summarize(np.array([np.inf, np.inf]), [1.0, 3.0, 5.0, 10.0])
    good = summarize(np.array([0.0, 0.0]), [1.0, 3.0, 5.0, 10.0])
    rows = [
        dict(duplicate_labels=True, baseline=fail, boundary=fail),
        dict(duplicate_labels=False, baseline=fail, boundary=good),
    ]
    result = aggregate(rows)
    assert result["all12"]["pairs"] == 2
    assert result["all12"]["variants"]["boundary"]["landmarks"] == 4
    assert result["all12"]["variants"]["boundary"]["unavailable"] == 2
    assert result["all12"]["variants"]["boundary"]["macro_pck"]["3.0"] == 0.5
    assert result["without_development_duplicate_labels"]["pairs"] == 1
    assert (
        result["without_development_duplicate_labels"]["variants"]["boundary"]["macro_pck"]["3.0"]
        == 1
    )
