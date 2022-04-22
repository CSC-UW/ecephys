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
