import pandas as pd


@pd.api.extensions.register_dataframe_accessor("units")
class UnitsAccessor:
    def __init__(self, df):
        self._df = df
        self.validate()

    # Should we enforce that cluster_id is always the index? Should we set it to be the index on a copy?
    def validate(self):
        if "cluster_id" != self._df.index.name:
            raise ValueError("`cluster_id` index not found.")
        if "depth" not in self._df.columns:
            raise ValueError("`depth` column not found.")

    def add_structure_from_sharptrack(self, sharptrack):
        """Add sharptrack structure info to self.units dataframe."""
        for structure in sharptrack.structures.itertuples():
            matches = (self._df["depth"] >= structure.lowerBorder_imec) & (
                self._df["depth"] <= structure.upperBorder_imec
            )
            self._df.loc[matches, "structure"] = structure.acronym
        self._df["structure"] = self._df["structure"].fillna("out")
