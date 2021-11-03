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
    output_string = '_'.join(sorted([f'{k}={v1}-{v2}' for k, (v1, v2) in selection_intervals.items()]))
    return output_string.replace('.','_')


def _get_cluster_groups(kslabel, curated_group, cluster_ids, cluster_group_overrides=None):
    """Cluster group. Revert to KSLabel if group is None or 'unsorted'. Allow overrides.
    
    Kwargs:
        cluster_group_overrides (None or dict): Dictionary of {<group>: <cluster_list>} used to
            override the groups saved in phy
    """
    assert len(kslabel) == len(curated_group)
    assert len(kslabel) == len(cluster_ids)
    use_KSLabel = (curated_group == 'unsorted') | pd.isna(curated_group)
    use_KSLabel = use_KSLabel.values
    group = np.empty((len(kslabel),), dtype=object)
    group[use_KSLabel] = kslabel[use_KSLabel]
    group[~use_KSLabel] = curated_group[~use_KSLabel]
    group[np.where(group == 'nan')[0]] = np.nan
    if cluster_group_overrides is not None:
        for override_group, override_list in cluster_group_overrides.items():
            print(f"Overriding cluster group from phy with group `{override_group}` for N={len(override_list)} clusters")
            for cluster_id in override_list:
                idx = np.where(cluster_ids.values == cluster_id)[0]
                if not len(idx):
                    raise ValueError(f"Unrecognized cluster_id in `{override_group}` group override list: {cluster_id}")
                assert len(idx) == 1
                idx = idx[0]
                if 'noise' in override_group and group[idx] != 'noise':
                    import warnings
                    warnings.warn(f"cluster `{cluster_id}`: Overriding group `{group[idx]}` with group `{override_group}`")
                group[idx] = override_group
    return group


def subset_cluster_info(
    info,
    selected_groups=None,
    good_only=False,
    drop_noise=True,
    selection_intervals=None,
    cluster_group_overrides=None,
):
    """
    Subset a spikeinterface extractor object.

    Args:
        info (pd.DataFrame): As loaded from `cluster_info.tsv`.

    Kwargs:
        selected_groups (list or None): List of subselected cluster groups. Affected by `drop_noise` and `good_only` overrides.
        good_only (bool): Subselect `cluster_group == 'good'`.
            We use KSLabel assignment when curated `group` is None or 'unscored'
        drop_noise (bool): Subselect `cluster_group != 'noise'`
        selection_intervals (None or dict): Dictionary of {<col_name>: (<value_min>, <value_max>)} used
            to subset clusters based on depth, firing rate, metrics value, etc
            All keys should be columns of `cluster_info.tsv` or `metrics.csv`, and the values should be numrical.
        cluster_group_overrides (None or dict): Dictionary of {<group>: <cluster_list>} used to
            override the groups saved in phy.
    """
    if selection_intervals is None:
        selection_intervals = SELECTION_INTERVALS_DF
    validate_selection_intervals(info, selection_intervals)

    for col in ["cluster_id", "KSLabel", "group"]:
        assert col in info.columns
    
    # Group taking in account manual curation or reverting to automatic KS label otherwise
    curated_group = _get_cluster_groups(info["KSLabel"], info["group"], info['cluster_id'], cluster_group_overrides=cluster_group_overrides)

    all_groups = list(set(list(np.unique(curated_group)) + list(cluster_group_overrides.keys())))
    if drop_noise:
        assert selected_groups is None or 'noise' not in selected_groups
    if selected_groups is not None:
        final_selected_groups = selected_groups
    else:
        final_selected_groups = all_groups
    if drop_noise:
        final_selected_groups = [g for g in final_selected_groups if g != 'noise']
    if good_only:
        assert selected_groups is None
        final_selected_groups = ['good']
        
    print(
        f"Subset clusters: \n"
        f"selected_groups={selected_groups}, good_only={good_only}, drop_noise={drop_noise} -> Return the following groups `{final_selected_groups}`\n"
        f"selection_intervals={selection_intervals}"
    )

    n_clusters = len(info)
    keep_cluster = np.ones((n_clusters,), dtype=bool)

    cluster_in_groups = np.zeros((n_clusters,), dtype=bool)
    for g in final_selected_groups:
        in_group = curated_group == g
        print(f" -> Select N={len(np.where(in_group)[0])}/{n_clusters} for group `{g}`")
        cluster_in_groups = cluster_in_groups | in_group
    keep_cluster = keep_cluster & cluster_in_groups

    # if drop_noise:
    #     not_noise_rows = curated_group != "noise"
    #     print(f" -> Drop N={len(np.where(~not_noise_rows)[0])}/{n_clusters} noise clusters")
    #     keep_cluster = keep_cluster & not_noise_rows

    # if good_only:
    #     good_rows = curated_group == "good"
    #     print(f" -> Drop N={len(np.where(~good_rows)[0])}/{n_clusters} not 'good' clusters")
    #     keep_cluster = keep_cluster & good_rows

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
    selected_groups=None,
    good_only=False,
    drop_noise=True,
    selection_intervals=None,
    cluster_group_overrides=None,
):
    """
    Subset a spikeinterface extractor object.

    Args:
        extractor (spikeextractors.sortingextractor)
        info (pd.DataFrame): As loaded from `cluster_info.tsv`.

    Kwargs:
        selected_groups (list or None): List of subselected cluster groups. Affected by `drop_noise` and `good_only` kwargs.
        good_only (bool): Subselect `cluster_group == 'good'`.
            We use KSLabel assignment when curated `group` is None or 'unscored'
        drop_noise (bool): Subselect `cluster_group != 'noise'`
        selection_intervals (None or dict): Dictionary of {<col_name>: (<value_min>, <value_max>)} used
            to subset clusters based on depth, firing rate, metrics value, etc
            All keys should be columns of `cluster_info.tsv` or `metrics.csv`, and the values should be numrical.
        cluster_group_overrides (None or dict): Dictionary of {<group>: <cluster_list>} used to
            override the groups saved in phy.
    """
    info_subset = subset_cluster_info(
        info,
        selected_groups=selected_groups,
        good_only=good_only,
        drop_noise=drop_noise,
        selection_intervals=selection_intervals,
        cluster_group_overrides=cluster_group_overrides,
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