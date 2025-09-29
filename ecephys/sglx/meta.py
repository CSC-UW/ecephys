import polars as pl


def get_pandas_type_maps() -> dict:
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
