import ast
from pathlib import Path

import pandas as pd

from ecephys.sglx.external import readSGLX


def get_pandas_type_maps() -> dict:
    """Get the most suitable pandas type for each metadata field.

    Because of the way pandas handles missing values, many of these types may not be
    respected. For example, pandas will cast int columns to float if they contain NaN.
    """
    meta_types = dict()
    meta_types["always_present"] = {
        "appVersion": object,
        "fileCreateTime": "datetime64[s]",
        "fileName": object,
        "fileSHA1": object,
        "fileSizeBytes": int,
        "fileTimeSecs": float,
        "firstSample": int,
        "gateMode": object,
        "nSavedChans": int,
        "snsSaveChanSubset": object,
        "syncSourceIdx": int,
        "syncSourcePeriod": float,
        "trigMode": object,
        "typeImEnabled": int,
        "typeNiEnabled": int,
        "typeThis": object,
        "userNotes": object,
        "snsShankMap": object,
        "snsChanMap": object,
        "path": object,  # Not a SpikeGLX field. Added by the user.
    }
    meta_types["if_using_imec"] = {
        "acqApLfSy": object,
        "imAiRangeMax": float,
        "imAiRangeMin": float,
        "imCalibrated": bool,
        "imDatApi": object,
        "imDatBs_fw": object,
        "imDatBsc_fw": object,
        "imDatBsc_hw": object,
        "imDatBsc_pn": object,
        "imDatBsc_sn": object,
        "imDatFx_hw": object,
        "imDatFx_pn": object,
        "imDatFx_sn": object,
        "imDatHs_fw": object,  # Not documented in SpikeGLX manual, but present.
        "imDatHs_hw": object,
        "imDatHs_pn": object,
        "imDatHs_sn": object,
        "imDatPrb_dock": int,
        "imDatPrb_pn": object,
        "imDatPrb_port": int,
        "imDatPrb_slot": int,
        "imDatPrb_sn": object,
        "imDatPrb_type": int,
        "imLEDEnable": bool,
        "imRoFile": object,
        "imSampRate": float,
        "imTrgRising": bool,
        "imTrgSource": int,
        "snsApLfSy": object,
        "syncImInputSlot": int,
        "imroTbl": object,
    }
    meta_types["if_using_timed_trigger"] = {
        "trgTimIsHInf": bool,
        "trgTimIsNInf": bool,
        "trgTimNH": float,
        "trgTimTH": float,
        "trgTimTL": float,
        "trgTimTL0": float,
    }
    meta_types["maybe_present"] = {
        "nDataDirs": int,
        "rmt_USERTYPE": object,
        "typeObEnabled": int,
        "imIsSvyRun": bool,
        "imMaxInt": int,
        "imStdby": object,
        "imSvyMaxBnk": object,
    }
    return meta_types


def get_polars_type_maps() -> dict:
    """Get the most suitable polars type for each metadata field.

    - "fileCreateTime" really only has second precision, but there is no polars type for that.
      The different polars datetime types (ns, us, ms) only trade off precision vs. range,
      but all use the same space. We probably don't need to use ns, but why not?
    """
    import polars as pl

    meta_types = dict()
    meta_types["always_present"] = {
        "appVersion": pl.String,
        "fileCreateTime": pl.Datetime(time_unit="ns"),
        "fileName": pl.String,
        "fileSHA1": pl.String,
        "fileSizeBytes": pl.Int64,
        "fileTimeSecs": pl.Float64,
        "firstSample": pl.Int64,
        "gateMode": pl.String,
        "nSavedChans": pl.Int64,
        "snsSaveChanSubset": pl.String,
        "syncSourceIdx": pl.Int64,
        "syncSourcePeriod": pl.Float64,
        "trigMode": pl.String,
        "typeImEnabled": pl.Int64,
        "typeNiEnabled": pl.Int64,
        "typeThis": pl.String,
        "userNotes": pl.String,
        "snsShankMap": pl.String,
        "snsChanMap": pl.String,
    }
    meta_types["if_using_imec"] = {
        "acqApLfSy": pl.List(pl.Int64),
        "imAiRangeMax": pl.Float64,
        "imAiRangeMin": pl.Float64,
        "imCalibrated": pl.Boolean,
        "imDatApi": pl.String,
        "imDatBs_fw": pl.String,
        "imDatBsc_fw": pl.String,
        "imDatBsc_hw": pl.String,
        "imDatBsc_pn": pl.String,
        "imDatBsc_sn": pl.String,
        "imDatFx_hw": pl.String,
        "imDatFx_pn": pl.String,
        "imDatFx_sn": pl.String,
        "imDatHs_fw": pl.String,
        "imDatHs_hw": pl.String,
        "imDatHs_pn": pl.String,
        "imDatHs_sn": pl.String,
        "imDatPrb_dock": pl.Int64,
        "imDatPrb_pn": pl.String,
        "imDatPrb_port": pl.Int64,
        "imDatPrb_slot": pl.Int64,
        "imDatPrb_sn": pl.String,
        "imDatPrb_type": pl.Int64,
        "imLEDEnable": pl.Boolean,
        "imRoFile": pl.String,
        "imSampRate": pl.Float64,
        "imTrgRising": pl.Boolean,
        "imTrgSource": pl.Int64,
        "snsApLfSy": pl.List(pl.Int64),
        "syncImInputSlot": pl.Int64,
        "imroTbl": pl.String,
    }
    meta_types["if_using_timed_trigger"] = {
        "trgTimIsHInf": pl.Boolean,
        "trgTimIsNInf": pl.Boolean,
        "trgTimNH": pl.Float64,
        "trgTimTH": pl.Float64,
        "trgTimTL": pl.Float64,
        "trgTimTL0": pl.Float64,
    }
    meta_types["maybe_present"] = {
        "nDataDirs": pl.Int64,
        "rmt_USERTYPE": pl.String,
        "typeObEnabled": pl.Int64,
        "imIsSvyRun": pl.Boolean,
        "imMaxInt": pl.Int64,
        "imStdby": pl.String,
        "imSvyMaxBnk": pl.String,
    }
    return meta_types


def read_metadata_as_polars(files: list[Path]):
    """
    Takes a list of filepaths, reads the metadata for each file, and returns a
    summary dataframe with types cast. Also adds a 'path' column to the dataframe,
    that contains the source filepath.

    The returned dataframe can be written to parquet without modification.

    See https://billkarsh.github.io/SpikeGLX/Sgl_help/Metadata_30.html"""
    import polars as pl

    meta_dict = [readSGLX.readMeta(f) for f in files]
    df = pl.DataFrame(meta_dict)

    metadata_is_missing = df.select(
        pl.all_horizontal(pl.all().is_null())
    ).to_series()  # Sometimes a file exists but is empty, and no metadata is found

    df = df.with_columns(pl.Series("path", [str(f) for f in files]))
    df = df.filter(~metadata_is_missing)  # Drop the empty files

    if metadata_is_missing.any():  # Warn the user
        missing_files = [
            files[i] for i, is_missing in enumerate(metadata_is_missing) if is_missing
        ]
        for f in missing_files:
            print(
                f"No metadata found for {f}. SpikeGLX probably wrote an empty file. Dropping."
            )

    meta_types = get_polars_type_maps()
    for group, types in meta_types.items():
        missing = set(types) - set(df.columns)
        if missing:
            print(f"Metadata fields {missing} from group {group} not found.")
            for field in missing:
                types.pop(field, None)
        for col, dtype in types.items():
            if dtype == pl.Boolean:
                df = df.with_columns(
                    pl.col(col).replace_strict({"false": False, "true": True})
                )
            elif dtype == pl.List(pl.Int64):
                df = df.with_columns(
                    pl.col(col).map_elements(
                        ast.literal_eval, return_dtype=pl.List(pl.Int64)
                    )
                )
            else:
                df = df.with_columns(pl.col(col).cast(dtype))

    extra = (
        set(df.columns)
        - set.union(*(set(types) for group, types in meta_types.items()))
        - set(["path"])
    )
    if extra:
        print(f"Found unexpected metadata fields: {extra}")

    return df


def pd2pl(df: pd.DataFrame):
    """Convert a dataframe of SpikeGLX metadata (e.g., from read_metadata_as_pandas)
    from pandas to polars. The result can be written to parquet without modification.
    """
    import polars as pl

    for col in ["path", "fileName", "imRoFile"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
    df = pl.from_pandas(df)
    pl_types = get_polars_type_maps()
    for group, types in pl_types.items():
        for col, dtype in types.items():
            if col in df.columns:
                df = df.with_columns(pl.col(col).cast(dtype))
    return df


def pl2pd(df) -> pd.DataFrame:
    """Convert a dataframe of SpikeGLX metadata (e.g., from pl.scan_parquet)
    from polars to pandas. The result can NOT be written to parquet, and does NOT
    commute with pd2pl. This basically just exists to provide a backward compatible
    layer.
    """
    df = df.to_pandas()
    for col in ["path", "fileName", "imRoFile"]:
        if col in df.columns:
            df[col] = df[col].apply(Path)
    for col in ["acqApLfSy", "snsApLfSy"]:
        if col in df.columns:
            df[col] = df[col].apply(tuple)
    return df
