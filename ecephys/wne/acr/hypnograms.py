import tdt
import numpy as np
import tdt
import pickle
import pandas as pd
from ecephys import hypnogram
from ecephys.wne import constants
from ecephys.tdt.utils import set_probe_and_locations
import spikeinterface as si
import acr

try:
    import acr
    import acr.io

    HAS_ACR_PACKAGE = True
except ImportError:
    HAS_ACR_PACKAGE = False

def get_experiment_start_timestamp(subject, experiment):
    first_rec = acr.info_pipeline.get_exp_recs(subject, experiment)[0]
    exp_times = acr.info_pipeline.subject_info_section(subject, 'rec_times')
    return pd.Timestamp(exp_times[first_rec]['start'])

def datetime_to_seconds_from_reference(dt, reference_timestamp):
    return (dt - reference_timestamp).total_seconds()

def dt2t(dt, subject, experiment):
    return datetime_to_seconds_from_reference(
        dt,
        get_experiment_start_timestamp(subject, experiment)
    )


def load_full_raw_hypnogram(experiment, subject, simplify: bool = True) -> hypnogram.FloatHypnogram:
    assert HAS_ACR_PACKAGE
    hgs = []
    recordings = acr.info_pipeline.get_exp_recs(subject, experiment)
    info_times = acr.info_pipeline.subject_info_section(subject, "rec_times")
    reference_timestamp = get_experiment_start_timestamp(subject, experiment)
    for rec in recordings:
        hg = acr.io.load_hypno(
            subject,
            rec,
            corrections=True,
            # update=True,
            update=False,
            float=True,
        )
        relative_start_time = datetime_to_seconds_from_reference(pd.Timestamp(info_times[rec]["start"]), reference_timestamp)
        hg["start_time"] = hg["start_time"] + relative_start_time
        hg["end_time"] = hg["end_time"] + relative_start_time
        hgs.append(hg)
    hg = hypnogram.FloatHypnogram(pd.concat(hgs).sort_values(by="start_time").reset_index(drop=True))
    if simplify:
        hg = hg.replace_states(constants.SIMPLIFIED_STATES)
        hg = hypnogram.FloatHypnogram.clean(hg._df)
    return hg

def get_light_dark_periods(experiment, subject, as_float: bool = True):
    """Get light/dark periods in chronological order.

    Examples:
    ---------
    "lightsOn": [t1, t3, t5],
    "lightsOff": [t2, t4],
    -->
    intervals = [(t1, t2), (t2, t3), (t3, t4), (t4, t5)]
    labels = ["on", "off", "on", "off"]
    """

    rec_times = acr.info_pipeline.subject_info_section(subject, 'rec_times')
    bl_start_actual = rec_times[f'{experiment}-bl']["start"]
    bl_day = bl_start_actual.split("T")[0]

    bl_start = pd.Timestamp(bl_day + "T09:00:00")
    bl_start = pd.Timestamp(bl_day + "T09:00:00")

    intervals = [
        (
            dt2t((bl_start + pd.Timedelta("0 h")), subject, experiment), 
            (dt2t((bl_start + pd.Timedelta("12 h")), subject, experiment))
        ),
        (
            dt2t((bl_start + pd.Timedelta("12 h")), subject, experiment), 
            (dt2t((bl_start + pd.Timedelta("24 h")), subject, experiment))
        ),
        (
            dt2t((bl_start + pd.Timedelta("24 h")), subject, experiment), 
            (dt2t((bl_start + pd.Timedelta("36 h")), subject, experiment))
        ),
        (
            dt2t((bl_start + pd.Timedelta("36 h")), subject, experiment), 
            (dt2t((bl_start + pd.Timedelta("48 h")), subject, experiment))
        ),
    ]
    labels = [
        "on",
        "off",
        "on",
        "off",
    ]
    return intervals, labels

def get_novel_objects_period(
    experiment, subject
) -> tuple[float, float]:

    rec_times = acr.info_pipeline.subject_info_section(subject, 'rec_times')

    if f'{experiment}-sd' in rec_times.keys():
        sd_rec = f'{experiment}-sd'
        sd_end = pd.Timestamp(rec_times[sd_rec]['end'])
    else:
        stim_start, stim_end = acr.stim.stim_bookends(subject, experiment)
        sd_rec = experiment
        sd_end = stim_start
    sd_start_actual = pd.Timestamp(rec_times[sd_rec]['start'])
    sd_day = rec_times[sd_rec]['start'].split("T")[0]

    return (
        dt2t(sd_start_actual, subject, experiment), 
        dt2t(sd_end, subject, experiment)
    )

def get_stimulation_period(
    experiment, subject
) -> tuple[float, float]:

    stim_start, stim_end = acr.stim.stim_bookends(subject, experiment)
    return (
        dt2t(stim_start, subject, experiment), 
        dt2t(stim_end, subject, experiment)
    )

def get_novel_objects_hypnogram(
    full_hg: hypnogram.FloatHypnogram, experiment: str, subject: str,
) -> hypnogram.FloatHypnogram:
    (nod_start, nod_end) = get_novel_objects_period(experiment, subject)
    return full_hg.trim(nod_start, nod_end)

def get_stimulation_hypnogram(
    full_hg: hypnogram.FloatHypnogram, experiment: str, subject: str,
) -> hypnogram.FloatHypnogram:
    return full_hg.trim(*get_stimulation_period(experiment, subject))

def get_day1_hypnogram(
    full_hg: hypnogram.FloatHypnogram, experiment: str, subject: str,
) -> hypnogram.FloatHypnogram:
    intervals, labels = get_light_dark_periods(experiment, subject)
    assert labels == ["on", "off", "on", "off"]
    return full_hg.trim(
        intervals[0][0],  # start of first light period,
        intervals[1][1],  # end of first dark period
    )

def get_day2_hypnogram(
    full_hg: hypnogram.FloatHypnogram, experiment: str, subject: str
) -> hypnogram.FloatHypnogram:
    intervals, labels = get_light_dark_periods(experiment, subject)
    assert labels == ["on", "off", "on", "off"]
    return full_hg.trim(
        intervals[2][0],  # start of first light period,
        intervals[3][1],  # end of first dark period
    )


def get_day1_light_period_hypnogram(
    full_hg: hypnogram.FloatHypnogram, experiment: str, subject: str,
) -> hypnogram.FloatHypnogram:
    intervals, labels = get_light_dark_periods(experiment, subject)
    assert labels == ["on", "off", "on", "off"]
    return full_hg.trim(
        intervals[0][0],  # start of first light period
        intervals[0][1],  # end of first light period
    )


def get_day1_dark_period_hypnogram(
    full_hg: hypnogram.FloatHypnogram, experiment: str, subject: str,
) -> hypnogram.FloatHypnogram:
    intervals, labels = get_light_dark_periods(experiment, subject)
    assert labels == ["on", "off", "on", "off"]
    return full_hg.trim(
        intervals[1][0],  # start of first dark period
        intervals[1][1],  # end of first dark period
    )


def get_day2_light_period_hypnogram(
    full_hg: hypnogram.FloatHypnogram, experiment: str, subject: str,
) -> hypnogram.FloatHypnogram:
    intervals, labels = get_light_dark_periods(experiment, subject)
    assert labels == ["on", "off", "on", "off"]
    return full_hg.trim(
        intervals[2][0],  # start of second light period
        intervals[2][1],  # end of second light period
    )


def get_day2_dark_period_hypnogram(
    full_hg: hypnogram.FloatHypnogram, experiment: str, subject: str
) -> hypnogram.FloatHypnogram:
    intervals, labels = get_light_dark_periods(experiment, subject)
    assert labels == ["on", "off", "on", "off"]
    return full_hg.trim(
        intervals[3][0],  # start of second dark period
        intervals[3][1],  # end of second dark period
    )


def get_post_deprivation_day2_light_period_hypnogram(
    full_hg: hypnogram.FloatHypnogram, experiment: str, subject: str
) -> hypnogram.FloatHypnogram:
    hg = get_day2_light_period_hypnogram(full_hg, experiment, subject)
    nod_hg = get_novel_objects_hypnogram(full_hg, experiment, subject)
    sleep_deprivation_end = nod_hg.end_time.max()
    return hg.trim(sleep_deprivation_end, hg["end_time"].max())


def get_circadian_match_hypnogram(
    full_hg: hypnogram.FloatHypnogram, experiment: str, subject: str, start: float, end: float
) -> hypnogram.FloatHypnogram:
    match_start = start - pd.to_timedelta("24h").total_seconds()
    match_end = end - pd.to_timedelta("24h").total_seconds()
    return full_hg.trim(match_start, match_end).keep_states(["NREM"])

def compute_basic_novel_objects_deprivation_experiment_hypnograms(
    experiment: str,
    subject: str, 
    duration="1:00:00",
) -> dict[str, hypnogram.FloatHypnogram]:
    """
    Load NOD FloatHypnograms

    Parameters:
    ===========
    experiment: str
    subject: str
    """
    duration = pd.to_timedelta(duration).total_seconds()
    hgs = dict()

    full_hg = load_full_raw_hypnogram(experiment, subject)

    # hgs["Full 48h"] = full_hg.keep_states(["NREM", "Wake", "REM"])

    nod_hg = get_novel_objects_hypnogram(full_hg, experiment, subject).keep_states(["Wake"])
    hgs["Early NOD"] = nod_hg.keep_first(duration)
    hgs["Late NOD"] = nod_hg.keep_last(duration)

    stim_hg = get_stimulation_hypnogram(full_hg, experiment, subject).keep_states(["Wake"])
    hgs["Stim NOD"] = stim_hg

    pdd2_hg = get_post_deprivation_day2_light_period_hypnogram(
        full_hg, experiment, subject
    )
    rslp_hg = pdd2_hg.keep_states(["NREM"])
    hgs["Early Recovery NREM"] = rslp_hg.keep_first(duration)
    hgs["Late Recovery NREM"] = rslp_hg.keep_last(duration)

    match_hg = get_circadian_match_hypnogram(
        full_hg,
        experiment,
        subject,
        hgs["Early Recovery NREM"]["start_time"].min(),
        hgs["Early Recovery NREM"]["end_time"].max(),
    ).keep_states(["NREM"])
    hgs["Early Recovery NREM match"] = match_hg

    hgs["Early Baseline NREM"] = (
        get_day1_light_period_hypnogram(full_hg, experiment, subject)
        .keep_states(["NREM"])
        .keep_first(duration)
    )
    return hgs