import pandas as pd
import numpy as np


@pd.api.extensions.register_dataframe_accessor("spikes")
class SpikesAccessor:
    def __init__(self, df):
        self._df = df
        self.validate()

    def validate(self):
        if "t" not in self._df.columns:
            raise ValueError("`t` column not found.")
        if "cluster_id" not in self._df.columns:
            raise ValueError("`cluster_id` column not found.")
        if "spike" != self._df.index.name:
            raise ValueError("`spike` index not found.")

    def select_time(self, start_time, end_time):
        mask = (self._df["t"] >= start_time) & (self._df["t"] <= end_time)
        return self._df.loc[mask]

    def as_trains(self, start_time=-np.Inf, end_time=np.Inf, grouping_col="cluster_id"):
        return pd.DataFrame(
            self.select_time(start_time, end_time)
            .groupby(
                grouping_col,
                observed=False,  # Represent all categories
            )["t"]
            .unique()
        )

    def join_units(
        self, units, units_columns=None, start_time=-float("Inf"), end_time=float("Inf")
    ):
        """Return spikes df with added info from units.

        Use all units columns if units_columns is None.
        """
        if units_columns is None:
            units_columns = units.columns
        if not all([c in units.columns] for c in units_columns):
            raise ValueError("Could not find all requested columns in units df.")
        return pd.merge(
            units[units_columns].reset_index(),
            self.select_time(start_time, end_time).reset_index(),
            validate="one_to_many",
        ).set_index(
            "spike"
        )  # TODO: Any way to speed this up?
