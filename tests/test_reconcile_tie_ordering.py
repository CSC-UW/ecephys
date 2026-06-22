"""Regression test for `reconcile_labeled_intervals` tie-ordering.

When two intervals share a start time (e.g. the sub-millisecond NoData/Artifact
slivers produced when hypnograms are reconciled against filetable/artifact
boundaries), the result must still be ordered so that end times are monotonically
increasing -- otherwise downstream `FloatHypnogram` validation raises
"Hypnogram end times are not monotonically increasing." Previously the final sort
keyed on start_time alone, leaving the tie order to a non-stable sort, so an
unrelated change elsewhere in the frame could flip it and break the load.

See ecephys.utils.pandas.reconcile_labeled_intervals and
ecephys.hypnogram.core.FloatHypnogram._validate.
"""

import pandas as pd

from ecephys.hypnogram.core import FloatHypnogram, reconcile_hypnograms
from ecephys.utils.pandas import reconcile_labeled_intervals


def test_tied_start_intervals_sorted_by_end():
    """Two intervals sharing a start time must emerge end-sorted.

    The winner (df1) carries two equal-start intervals presented LONG-first. With a
    start-time-only final sort the long interval would stay first, giving a
    non-monotonic end_time; sorting by (start, end) puts the shorter one first.
    """
    df1 = pd.DataFrame(
        {
            "state": ["Artifact", "Artifact"],
            "start_time": [100.0, 100.0],
            "end_time": [400.0, 100.00002],  # long-first on purpose
        }
    )
    df2 = pd.DataFrame(
        {"state": ["NREM"], "start_time": [500.0], "end_time": [600.0]}
    )
    out = reconcile_labeled_intervals(
        df1, df2, "start_time", "end_time", "duration"
    ).reset_index(drop=True)
    assert out["end_time"].is_monotonic_increasing
    assert list(out["end_time"]) == [100.00002, 400.0, 600.0]


def test_reconcile_hypnograms_tied_start_constructs_floathypnogram():
    """The hypnogram-level wrapper must yield a valid FloatHypnogram on tied starts."""
    h1 = pd.DataFrame(
        {
            "state": ["Artifact", "Artifact"],
            "start_time": [100.0, 100.0],
            "end_time": [400.0, 100.00002],
            "duration": [300.0, 0.00002],
        }
    )
    h2 = pd.DataFrame(
        {
            "state": ["NREM"],
            "start_time": [500.0],
            "end_time": [600.0],
            "duration": [100.0],
        }
    )
    out = reconcile_hypnograms(h1, h2)
    assert out["end_time"].is_monotonic_increasing
    # Must construct without raising "end times are not monotonically increasing".
    FloatHypnogram(out.reset_index(drop=True))
