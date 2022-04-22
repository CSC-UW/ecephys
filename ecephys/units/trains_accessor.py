import pandas as pd


@pd.api.extensions.register_series_accessor("trains")
class TrainsAccessor:
    def __init__(self, obj):
        self._obj = obj
        self.validate()

    def validate(self):
        if "cluster_id" != self._obj.index.name:
            raise ValueError("`cluster_id` index not found.")
        if "t" != self._obj.name:
            raise ValueError("Series name is not `t`.")


@pd.api.extensions.register_dataframe_accessor("trains")
class TrainsAccessor:
    def __init__(self, df):
        self._df = df
        self.validate()

    def validate(self):
        if "t" not in self._df.columns:
            raise ValueError("`t` column not found.")

    def silent(self):
        return self._df["t"].isna()

    # TODO: Allow many-to-1  units-to-train merge
    def join_units(self, units, units_columns=None):
        """Return trains df with added info from units.

        Use all units columns if `units_columns` is None.
        """
        if units_columns is None:
            units_columns = units.columns
        if not all([c in units.columns] for c in units_columns):
            raise ValueError(
                "Could not find all requested columns in units df."
            )
        return pd.merge(
            units[units_columns].reset_index(),
            self._df.reset_index(),
            validate='one_to_one',
        ).set_index(
            self._df.index.name
        ).sort_values(
            by=self._df.index.name
        )