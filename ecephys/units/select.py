from pathlib import Path

import numpy as np
import pandas as pd
import spikeextractors as se


SELECTION_INTERVALS_DF = {
    "depth": (0.0, float("Inf")),
    "fr": (0.0, float("Inf")),
    # "isi_viol": (0.0, float("Inf")),
    # "cumulative_drift": (0.0, float("Inf")),
    # "isolation_distance": (0.0, float("Inf")),
}  # Should all be columns of `cluster_info.tsv` or `metrics.csv`


def get_selection_intervals_str(selection_intervals):
    if selection_intervals is None:
        return None
    return '_'.join(
        sorted([f'{k}={v1}-{v2}' for k, (v1, v2) in selection_intervals.items()])
    )


def _get_cluster_groups(kslabel, curated_group):
    """Cluster group. Revert to KSLabel if group is None or 'unsorted'"""
    assert len(kslabel) == len(curated_group)
    use_KSLabel = (curated_group == 'unsorted') | pd.isna(curated_group)
    use_KSLabel = use_KSLabel.values
    group = np.empty((len(kslabel),), dtype=object)
    group[use_KSLabel] = kslabel[use_KSLabel]
    group[~use_KSLabel] = curated_group[~use_KSLabel]
    group[np.where(group == 'nan')[0]] = np.nan
    return group

def subset_cluster_info(
    info,
    good_only=False,
    drop_noise=True,
    selection_intervals=None,
):
    """
    Subset a spikeinterface extractor object.

    Args:
        info (pd.DataFrame): As loaded from `cluster_info.tsv`.

    Kwargs:
        good_only (bool): Subselect `cluster_group == 'good'`.
            We use KSLabel assignment when curated `group` is None or 'unscored'
        drop_noise (bool): Subselect `cluster_group != 'noise'`
        selection_intervals (None or dict): Dictionary of {<col_name>: (<value_min>, <value_max>)} used
            to subset clusters based on depth, firing rate, metrics value, etc
            All keys should be columns of `cluster_info.tsv` or `metrics.csv`, and the values should be numrical.
    """
    if selection_intervals is None:
        selection_intervals = SELECTION_INTERVALS_DF
    validate_selection_intervals(info, selection_intervals)

    for col in ["cluster_id", "KSLabel", "group"]:
        assert col in info.columns

    print(
        f"Subset clusters: good_only={good_only}, drop_noise={drop_noise}, "
        f"selection_intervals={selection_intervals}"
    )

    # Group taking in account manual curation or reverting to automatic KS label otherwise
    curated_group = _get_cluster_groups(info["KSLabel"], info["group"])

    n_clusters = len(info)
    keep_cluster = np.ones((n_clusters,), dtype=bool)

    if drop_noise:
        not_noise_rows = curated_group != "noise"
        print(f" -> Drop N={len(np.where(~not_noise_rows)[0])}/{n_clusters} noise clusters")
        keep_cluster = keep_cluster & not_noise_rows

    if good_only:
        good_rows = curated_group == "good"
        print(f" -> Drop N={len(np.where(~good_rows)[0])}/{n_clusters} not 'good' clusters")
        keep_cluster = keep_cluster & good_rows

    for info_col, interval in selection_intervals.items():
        select_rows = info[info_col].between(*interval)
        print(f" -> Info column = `{info_col}`: Drop N={len(np.where(~select_rows)[0])}/{n_clusters} not within interval = `{interval}")
        keep_cluster = keep_cluster & select_rows

    info_subset = info.loc[keep_cluster].copy()
    print(f"Subselect N = {len(info_subset)}/{n_clusters} clusters", end='')
    return info_subset


def subset_clusters(
    extractor,
    info,
    good_only=False,
    drop_noise=True,
    selection_intervals=None,
):
    """
    Subset a spikeinterface extractor object.

    Args:
        extractor (spikeextractors.sortingextractor)
        info (pd.DataFrame): As loaded from `cluster_info.tsv`.

    Kwargs:
        good_only (bool): Subselect `cluster_group == 'good'`.
            We use KSLabel assignment when curated `group` is None or 'unscored'
        drop_noise (bool): Subselect `cluster_group != 'noise'`
        selection_intervals (None or dict): Dictionary of {<col_name>: (<value_min>, <value_max>)} used
            to subset clusters based on depth, firing rate, metrics value, etc
            All keys should be columns of `cluster_info.tsv` or `metrics.csv`, and the values should be numrical.
    """
    info_subset = subset_cluster_info(
        info,
        good_only=good_only,
        drop_noise=drop_noise,
        selection_intervals=selection_intervals,
    )

    subextractor = se.SubSortingExtractor(
        extractor, unit_ids=list(info_subset["cluster_id"])
    )
    assert sorted(info_subset["cluster_id"]) == sorted(subextractor.get_unit_ids())

    return subextractor


def validate_selection_intervals(info, selection_intervals):
    if not isinstance(selection_intervals, dict):
        raise ValueError(f"Wrong format for selection_intervals {selection_intervals}")
    for k, (v1, v2) in selection_intervals.items():
        if k not in info.columns:
            raise ValueError(f"Unrecognized key `selection_intervals`: {k}")
        if any([not isinstance(v, (float, int)) for v in [v1, v2]]):
            raise ValueError(f"Incompatible values for key `{k}` in `selection_intervals` `{selection_intervals}. \n Expecting numerical values.")
        from pandas.api.types import is_numeric_dtype
        if not is_numeric_dtype(info[k].dtype):
            raise ValueError(
                f"Non-numerical dtype in `cluster_info.tsv` for the following `selection_intervals` key: {k}"
            )