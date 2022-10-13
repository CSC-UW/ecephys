import pandas as pd
from pandas.api import types


def takes_trains(func):
    """A decorator you can use on your own functions to ensure that the first argument passed is a valid trains frame."""

    def wrapped_function(trains, *args, **kwargs):
        if "t" not in trains.columns:
            raise ValueError(
                f"Problem with trains frame passed to {func.__name__}: `t` column not found."
            )
        if not types.is_object_dtype(trains["t"].dtype):
            raise ValueError(
                f"Problem with trains frame passed to {func.__name__}: `t` column dtype is not object."
            )
        return func(trains, *args, **kwargs)

    return wrapped_function


@takes_trains
def silent(trains):
    return trains["t"].isna()


def takes_unit_trains(func):
    """A decorator you can use on your own functions to ensure that the first argument passed is a valid unit trains frame."""

    @takes_trains
    def wrapped_function(unitTrains, *args, **kwargs):
        if unitTrains.index.name != "cluster_id":
            raise ValueError(
                f"Problem with unit trains frame passed to {func.__name__}: Not indexed by cluster_id."
            )

        return func(unitTrains, *args, **kwargs)

    return wrapped_function


# TODO: Allow many-to-1  units-to-train merge
@takes_unit_trains
def add_cluster_info(unitTrains, clusterInfo, propertiesToAdd):
    if isinstance(propertiesToAdd, str):
        propertiesToAdd = [propertiesToAdd]
    if not isinstance(propertiesToAdd, list):
        raise ValueError(f"Expected list, got {type(propertiesToAdd)}")
    return pd.merge(
        clusterInfo[propertiesToAdd + ["cluster_id"]],
        unitTrains,
        on="cluster_id",
        validate="one_to_one",
    ).set_index("cluster_id")
