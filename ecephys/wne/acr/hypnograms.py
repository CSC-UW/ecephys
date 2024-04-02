import tdt
import numpy as np
import tdt
import pickle
import pandas as pd
from ecephys import hypnogram
from ecephys.wne import constants
from ecephys.tdt.utils import set_probe_and_locations
import spikeinterface as si

try:
    import acr
    import acr.io

    HAS_ACR_PACKAGE = True
except ImportError:
    HAS_ACR_PACKAGE = False


def load_full_raw_hypnogram(segments_info: list[dict], simplify: bool = True) -> hypnogram.FloatHypnogram:
    assert HAS_ACR_PACKAGE
    hgs = []
    reference_timestamp = pd.Timestamp(segments_info[0]["info_times"]["start"])
    for segment_info in segments_info:
        hg = acr.io.load_hypno(
            segment_info["subject"],
            segment_info["recording"],
            corrections=True,
            # update=True,
            update=False,
            float=True,
        )
        relative_start_time = _get_relative_recording_start_time(segment_info, reference_timestamp)
        hg["start_time"] = hg["start_time"] + relative_start_time
        hg["end_time"] = hg["end_time"] + relative_start_time
        hgs.append(hg)
    hg = hypnogram.FloatHypnogram(pd.concat(hgs).sort_values(by="start_time").reset_index(drop=True))
    if simplify:
        hg = hg.replace_states(constants.SIMPLIFIED_STATES)
        hg = hypnogram.FloatHypnogram.clean(hg._df)
    return hg


def get_light_dark_periods(experiment: str, subject: wne.sglx.SGLXSubject, as_float: bool = True):
    """Get light/dark periods in chronological order.

    Examples:
    ---------
    "lightsOn": [t1, t3, t5],
    "lightsOff": [t2, t4],
    -->
    intervals = [(t1, t2), (t2, t3), (t3, t4), (t4, t5)]
    labels = ["on", "off", "on", "off"]
    """
    params = PROJ.load_experiment_subject_params(experiment, subject.name)
    on = pd.DataFrame({"time": [pd.to_datetime(x) for x in params["lightsOn"]], "transition": "on"})
    off = pd.DataFrame({"time": [pd.to_datetime(x) for x in params["lightsOff"]], "transition": "off"})
    df = pd.concat([on, off]).sort_values("time").reset_index(drop=True)
    if as_float:
        df["time"] = subject.dt2t(experiment, params["hypnogram_probe"], df["time"].values)

    periods = list(it.pairwise(df.itertuples()))
    intervals = [(start.time, end.time) for start, end in periods]
    labels = [start.transition for start, _ in periods]
    return intervals, labels


def plot_lights_overlay(
    intervals: list[tuple],
    interval_labels: list[str],
    ax: plt.Axes,
    ymin=1.0,
    ymax=1.02,
    alpha=1.0,
    zorder=1000,
    colors={"on": "yellow", "off": "gray"},  # Keys are from interval_labels
):
    """Take intervals and labels, as returned by `get_light_dark_periods()`, and use these to overlay
    the light/dark cycle onto a plot.

    Examples
    --------
    >>> fig, ax = plt.subplots()
    Overlay into background, span whole plot axis
    >>> plot_lights_overlay(intervals, interval_labels, ax=ax, ymin=0, ymax=1, alpha=0.3)
    Overlay into foreground, outside of plot yaxis (but within plot xaxis)
    >>> plot_lights_overlay(intervals, interval_labels, ax=ax, ymin=1.0, ymax=1.02, alpha=1.0)
    """
    xlim = ax.get_xlim()

    for (tOn, tOff), lbl in zip(intervals, interval_labels):
        # clip manually on xaxis so we can set clip_on=False for yaxis
        tOn = max(tOn, xlim[0])
        tOff = min(tOff, xlim[1])
        ax.axvspan(
            tOn,
            tOff,
            alpha=alpha,
            color=colors[lbl],
            zorder=zorder,
            ec="none",
            ymin=ymin,
            ymax=ymax,
            clip_on=False,
        )

    ax.set_xlim(xlim)


def get_novel_objects_period(experiment: str, subject: wne.sglx.SGLXSubject) -> tuple[float, float]:
    params = PROJ.load_experiment_subject_params(experiment, subject.name)
    probe = params["hypnogram_probe"]

    start = pd.to_datetime(params["novel_objects_start"])
    start = subject.dt2t(experiment, probe, start)

    end = pd.to_datetime(params["novel_objects_end"])
    end = subject.dt2t(experiment, probe, end)

    return (start, end)


def compute_basic_hypnograms(
    subject: str,
    probe: str,
    experiment: str,
    duration="1:00:00",
) -> dict[str, hypnogram.FloatHypnogram]:
    duration = pd.to_timedelta(duration).total_seconds()
    hgs = dict()

    full_hg = load_full_raw_hypnogram(experiment, subject, probe)

    hgs["Full 48h"] = full_hg.keep_states(["NREM", "Wake", "REM"])

    nod_hg = get_novel_objects_hypnogram(full_hg, NOD, subject).keep_states(["Wake"])
    hgs["Early NOD"] = nod_hg.keep_first(duration)
    hgs["Late NOD"] = nod_hg.keep_last(duration)

    return hgs
