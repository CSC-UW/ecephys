import itertools as it
import pandas as pd
from ...sglx import file_mgmt

# TODO: Currently, there can be no clean `get_session_directory(subject_name, experiment_name) function,
#       because there is no single session directory -- it can be split across AP and LF locations, and we
#       don't know which might hold e.g. video files for that session.


def add_experiment_times(ftab: pd.DataFrame) -> pd.DataFrame:
    """Get a series containing the time of each file start, in seconds, relative to the beginning of the experiment.
    Remember that the beginning of the experiment is the beginning of the first session, and may not align perfectly with
    what you would normally /call/ the beginning of the experiment. This is why we have aliases.
    # TODO: I really should have named this construct a MultiSession instead of an Experiment... Oh well, one day maybe...

    Parameters
    ==========
    ftab: pd.DataFrame
        ALL files from one single experiment. We cannot check this for you, you must guarantee it.

    Returns
    =======
    new_ftab:
        Same length as ftab, but with additional columns added containing experiment times:
        - expmtPrbAcqFirstTime: First timestamp, measured from the start of the probe's first acquisition this experiment, in this probe's timebase.
            BEWARE: Its error can be as great as the error in involved in subtracting one fileCreateTime from another!!!!
        - expmtPrbAcqLastTime: Last timestamp, measured from the start of the probe's first acquisition this experiment, in this probe's timebase.
            BEWARE: Its error can be as great as the error in involved in subtracting one fileCreateTime from another!!!!
        - expmtPrbAcqFirstDatetime: Datetime of first timestamp, more accurate than fileCreateTime.
            BEWARE: Its error can be as great as the error in involved in subtracting one fileCreateTime from another!!!!
        - expmtPrbAcqLastDatetime: Datetime of last timestamp
            BEWARE: Its error can be as great as the error in involved in subtracting one fileCreateTime from another!!!!
    """
    ftab = ftab.sort_values("fileCreateTime")
    new_ftab = []

    # Do the rigourus method
    for probe, stream, ftype in it.product(
        ftab["probe"].unique(), ftab["stream"].unique(), ftab["ftype"].unique()
    ):
        # Select data. We cannot assume that different streams or probes having the same metadata.
        mask = (
            (ftab["probe"] == probe)
            & (ftab["stream"] == stream)
            & (ftab["ftype"] == ftype)
        )
        _ftab = ftab.loc[mask]
        # In case we changed session path for only one of the probes
        if not len(_ftab):
            continue
        # Get semicontinuous segments
        acqs, segs = file_mgmt.get_semicontinuous_segments(_ftab)
        firstAcquisitionDatetime = segs[0]["prbRecDatetime"] - pd.to_timedelta(
            segs[0]["firstTime"], "s"
        )
        # Combine the (super-precise) timestamps WITHIN each acquisition segment, with the (less precise)* offsets BETWEEN acquisitions segments.
        # *(These offsets have to be estimated using file creation times). Tests indicate that these are precise to within a few msec.)
        # TODO: Add expmtPrbAcqFirstSample and expmtPrbAcqLastSample fields?
        for acq, seg in zip(acqs, segs):
            segAcquisitionDatetime = seg["prbRecDatetime"] - pd.to_timedelta(
                seg["firstTime"], "s"
            )
            segAcqOffset = (
                segAcquisitionDatetime - firstAcquisitionDatetime
            ).total_seconds()
            acq["expmtPrbAcqFirstTime"] = acq["firstTime"] + segAcqOffset
            acq["expmtPrbAcqLastTime"] = acq["lastTime"] + segAcqOffset
            acq[
                "expmtPrbAcqFirstDatetime"
            ] = firstAcquisitionDatetime + pd.to_timedelta(
                acq["expmtPrbAcqFirstTime"], "s"
            )
            acq["expmtPrbAcqLastDatetime"] = firstAcquisitionDatetime + pd.to_timedelta(
                acq["expmtPrbAcqLastTime"], "s"
            )
        new_ftab.append(pd.concat(acqs))

    new_ftab = pd.concat(new_ftab)
    assert len(new_ftab) == len(ftab)
    return new_ftab
