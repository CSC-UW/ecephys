import pandas as pd
from pandas.api import types


def takes_spikes_frame(func):
    """A decorator you can use on your own functions to ensure that the first argument passed is a valid Spikes frame."""

    def wrapped_function(spikes, *args, **kwargs):
        if "t" not in spikes.columns:
            raise ValueError(
                f"Problem with spikes frame passed to {func.__name__}: `t` column not found."
            )
        if not types.is_numeric_dtype(spikes["t"].dtype):
            raise ValueError(
                f"Problem with spikes frame passed to {func.__name__}: `t` column dtype is not numeric."
            )
        if "cluster_id" not in spikes.columns:
            raise ValueError(
                f"Problem with spikes frame passed to {func.__name__}: `cluster_id` column not found."
            )
        return func(spikes, *args, **kwargs)

    return wrapped_function


@takes_spikes_frame
def between_time(spikes, start_time=-float("Inf"), end_time=float("Inf")):
    mask = (spikes["t"] >= start_time) & (spikes["t"] <= end_time)
    return spikes.loc[mask]


@takes_spikes_frame
def add_cluster_info(spikes, clusterInfo, propertiesToAdd):
    """Add column(s) with cluster properties to a spikes frame.

    Parameters:
    ===========
    spikes: DataFrame
    clusterInfo DataFrame
        One column per cluster property. Must contain a 'cluster_id' column.
    propertiesToAdd: DataFrame column indexer
        The properties from clusterInfo to add.
    """
    if isinstance(propertiesToAdd, str):
        propertiesToAdd = [propertiesToAdd]
    if not isinstance(propertiesToAdd, list):
        raise ValueError(f"Expected list, got {type(propertiesToAdd)}")
    return pd.merge(
        clusterInfo[propertiesToAdd + ["cluster_id"]],
        spikes,
        on="cluster_id",
        validate="one_to_many",
    )


@takes_spikes_frame
def as_trains(spikes, oneTrainPer="cluster_id"):
    return pd.DataFrame(
        spikes.groupby(
            oneTrainPer,
            observed=False,  # Represent all categories
        )["t"].unique()
    )
