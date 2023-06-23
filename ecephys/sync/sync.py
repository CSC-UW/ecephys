import logging
import warnings

from difflib import SequenceMatcher
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
import tdt

from ecephys import utils
import ecephys.sglx
from ecephys.sglx.external import readSGLX

logger = logging.getLogger(__name__)

# TODO: System-specific packages should be in system-specific repos (e.g. tdt_xarray, sglxarray, acute, etc.)


def plot_pulses(onset_times, offset_times, nPulsesToPlot=20):
    """Use onset and offset times to plot pulse waveforms.
    Times should be in seconds. Plot only the first N pulses (default 100)"""

    # Interleave onset and offset times
    edge_times = np.empty(
        (onset_times.size + offset_times.size,), dtype=onset_times.dtype
    )
    edge_times[0::2] = onset_times
    edge_times[1::2] = offset_times

    # Figure out when pulses are high and when they are low
    sync_levels = np.empty(
        (onset_times.size + offset_times.size,), dtype=onset_times.dtype
    )
    sync_levels[0::2] = 1
    sync_levels[1::2] = 0

    plt.figure(num=None, figsize=(30, 6), dpi=80, facecolor="w", edgecolor="k")
    plt.step(edge_times[:nPulsesToPlot], sync_levels[:nPulsesToPlot], where="post")


def visualize_mapping(x, y, model, nPulsesToPlot=10, xname="X", yname="Y"):
    """Visualize remapped sync pulse times from x in y's time space, to verify correctness.
    Black lines = Pulse times measured by y.
    Red lines = Pulse times measured by x and remapped to y's time space."""
    y_pred = model.predict(x)

    plt.figure(num=None, figsize=(30, 1), dpi=80, facecolor="w", edgecolor="k")
    plt.vlines(y[:nPulsesToPlot], 0, 1, colors="black")
    plt.vlines(y_pred[:nPulsesToPlot], 0, 1, linestyle="dotted", colors="red")
    plt.title(
        f"Black: Events measured by {xname} \n Red: Events remapped from {xname} to {yname}'s time space."
    )


def check_edges(rising, falling):
    """Attempt to pair rising and falling edges, such that rising edges always precede falling edges.

    Returns
    =======
    rising, falling:
        The sizes of these arrays may be different than the sizes of the input arrays.
    """
    mismatch = rising.size - falling.size
    if mismatch > 0:  # More rising edges than falling edges
        logger.warning(
            f"Number of rising and falling edges do not match. Rising - falling = {mismatch}."
        )
        if all(
            np.less(rising[mismatch:], falling)
        ):  # Rising edges occur before falling edges
            return rising[mismatch:], falling
        elif all(np.less(rising[:-mismatch], falling)):
            return rising[:-mismatch], falling
        else:
            raise ValueError(
                "Could not reconcile mismatched numbers of rising and falling edges such that rising edges always precede falling edges."
            )
    elif mismatch < 0:  # More falling edges than rising edges
        logger.warning(
            f"Number of rising and falling edges do not match. Rising - falling = {mismatch}."
        )
        if all(np.less(rising, falling[:mismatch])):
            return rising, falling[:mismatch]
        elif all(np.less(rising, falling[-mismatch:])):
            return rising, falling[-mismatch:]
        else:
            raise ValueError(
                "Could not reconcile mismatched numbers of rising and falling edges such that rising edges always precede falling edges."
            )
    else:  # Number of rising edges matches number of falling edges
        if all(np.less(rising, falling)):
            return rising, falling
        elif all(
            np.less(rising[:-1], falling[1:])
        ):  # If first edge recorded was falling, and last was rising
            return rising[:-1], falling[1:]
        else:
            raise ValueError(
                "Could not find a rising-falling edge pairing such that rising edges always precede falling edges."
            )


def fit_times(x, y, visualize=True, xname="X", yname="Y"):
    """Using event times shared by both systems, fit the X's measured times to Y's measured times."""
    x = x.reshape((-1, 1))
    model = LinearRegression().fit(x, y)
    r_sq = model.score(x, y)

    logger.info(f"coefficient of determination: {r_sq}")
    logger.info(f"intercept: {model.intercept_}")
    logger.info(f"drift rate in msec/hr: {(model.coef_[0] - 1) * 60 * 60 * 1000}")
    logger.info("(assumes `x` and `y` are provided in seconds)")

    if visualize:
        visualize_mapping(x, y, model, xname=xname, yname=yname)

    return model


def remap_times(x_like, model):
    """Given a model returned by `fit_times`, remap a vector of times from the timespace of X to the timespace of Y.

    Note that x_like only has to have the same timebase/reference frame as X!
    For example, if X was a sync barcode recorded alongside camera frame TTLS on DAQ-X,
    and Y was the same sync barcode recorded alongside neural signals on DAQ-Y,
    this function could be used to convert camera frame times into the neural timebase.
    """
    return model.predict(x_like.reshape((-1, 1)))


def binarize(x, threshold):
    "Turn a non-binary vector (e.g. timeseries) into a binarized one."
    return np.where(x > threshold, 1, 0)


def get_rising_edges_from_binary_signal(x):
    """Assumes x is binarized"""
    return np.squeeze(np.where(np.diff(x, prepend=np.nan) == 1))


def get_falling_edges_from_binary_signal(x):
    """Assumes x is binarized"""
    return np.squeeze(np.where(np.diff(x, prepend=np.nan) == -1))


def get_shared_sequence(a, b):
    """Find the longest sequence of elements shared by both a and b."""
    s = SequenceMatcher(None, a, b)
    match = s.find_longest_match(alo=0, ahi=len(a), blo=0, bhi=len(b))
    shared = a[match.a : match.a + match.size]
    a_slice = slice(match.a, match.a + match.size)
    b_slice = slice(match.b, match.b + match.size)
    return shared, a_slice, b_slice


#####
# Generic system-to-system functions
# #####


def fit_barcode_times(
    sysX_times, sysX_values, sysY_times, sysY_values, sysX_name="X", sysY_name="Y"
):
    """Use barcodes to align streams."""
    shared_values, sysY_slice, sysX_slice = get_shared_sequence(
        sysY_values, sysX_values
    )
    print(
        "Length of longest barcode sequence common to both systems:\n",
        len(shared_values),
    )
    return fit_times(
        x=sysX_times[sysX_slice],
        y=sysY_times[sysY_slice],
        xname=sysX_name,
        yname=sysY_name,
    )


def fit_random_pulse_times(
    sysX_times, sysY_times, sysX_name="X", sysY_name="Y", max_shift=None, plot=True
):
    """Use random pulses to align streams.
    Apply _find_mapping twice, sliding the signals past each other in both forward and reverse directions to find the best fit.
    """
    sysX_times, sysY_times = _equalize_lengths(
        sysX_times, sysY_times
    )  # Is this necessary?

    model_f, r_sq_f = _find_mapping(
        sysX_times, sysY_times, "forward", max_shift=max_shift, plot=plot
    )
    model_b, r_sq_b = _find_mapping(
        sysX_times, sysY_times, "backward", max_shift=max_shift, plot=plot
    )

    if r_sq_f > r_sq_b:
        print(f"{sysY_name} times lead {sysX_name} times -- Forward shift selected")
        return model_f
    elif r_sq_b > r_sq_f:
        print(f"{sysY_name} times lag {sysX_name} times -- Backward shift selected")
        return model_b
    else:
        utils.warn(
            "Unexpected: Backward and forward fits are equivalent. Are signals already very close to aligned?"
        )
        assert (
            model_f.intercept_ == model_b.intercept_
        ), "Forward and backward fits have different intercepts."
        assert (
            model_f.coef_[0] == model_b.coef_[0]
        ), "Forward and backward fits have different slopes."
        return model_f


def _equalize_lengths(x, y):
    """Keep only the first N items in vectors x and y, where N is the size of the smallest vector.
    Makes subsequent computations simpler."""
    n = np.min([np.size(x), np.size(y)])
    return (x[:n], y[:n])


def _find_mapping(sysX_times, sysY_times, direction, max_shift=None, plot=True):
    """Iteratively fit signals to each other with different shifts to find the one that maximizes fit.
    max_shift can be used to avoid exhaustively searching all signal offsets. Defaults to 10% of total pulses.
    If your two systems start recording at very different times, you may need to play with this parameter.
    Reducing it will speed up computation, but also increase change of an error."""
    if max_shift is None:
        max_shift = np.int64(
            np.ceil(np.min([sysX_times.size, sysY_times.size]) * 0.1)
        )  # Always leaves at least 90% of pulses worth of overlap between the two signals

    shifts = np.arange(
        max_shift
    )  # Generate a list of all the shifts that we are going to try

    fits = np.array(
        [_eval_shift(sysX_times, sysY_times, n, direction) for n in shifts]
    )  # Evaluate each shift

    if plot:
        plt.figure(num=None, figsize=(30, 6), dpi=80, facecolor="w", edgecolor="k")
        plt.plot(shifts, fits)

    best_shift = shifts[np.argmax(fits)]
    model, r_sq = _shift(sysX_times, sysY_times, best_shift, direction)

    print(direction)
    print("number of pulses shifted:", best_shift)
    print("coefficient of determination:", r_sq)
    print("intercept:", model.intercept_)
    print("drift rate in msec/hr:", (model.coef_[0] - 1) * 60 * 60 * 1000)
    print(" ")

    return model, r_sq


def _eval_shift(sysX_times, sysY_times, n_pulses, direction):
    model, r_sq = _shift(sysX_times, sysY_times, n_pulses, direction)
    return r_sq


def _shift(sysX_times, sysY_times, n_pulses, direction):
    """Remove n_pulses from the start of one signal and the end of the other, then fit the remaining Y pulses to the remaining X pulses."""

    if direction == "forward":
        # Remove pulses from the start of the X signal and the end of the Y signal.
        y = sysX_times[n_pulses:]
        x = sysY_times[: np.size(y)].reshape((-1, 1))
    elif direction == "backward":
        # Remove pulses from the start of the Y signal and the end of the X signal.
        x = sysY_times[n_pulses:].reshape((-1, 1))
        y = sysX_times[: np.size(x)]
    else:
        raise ("Shift direction must be specified")

    model = LinearRegression().fit(x, y)
    r_sq = model.score(x, y)

    return model, r_sq


def fit_square_pulse_times(
    sysX_rising_times,
    sysX_falling_times,
    sysY_rising_times,
    sysY_falling_times,
    sysX_name="X",
    sysY_name="Y",
    visualize=True,
    expected_pulse_width=0.5,
):
    xdf = pd.DataFrame({"rising": sysX_rising_times, "falling": sysX_falling_times})
    ydf = pd.DataFrame({"rising": sysY_rising_times, "falling": sysY_falling_times})

    xdf = _check_pulse_widths(xdf, expected_pulse_width, sys_name=sysX_name)
    ydf = _check_pulse_widths(ydf, expected_pulse_width, sys_name=sysY_name)

    # After discarding abberant pulses, the number and sequence identity of pules may not match,
    # because for e.g. a good pulse may have been split in to by a bouncy edge on one probe and not another.
    # So we need to find the best mapping between the two sets of pulses.
    xt = xdf[["rising", "falling"]].values.flatten()
    assert np.all(np.diff(xt) > 0), f"{sysX_name} times must be increasing"
    yt = ydf[["rising", "falling"]].values.flatten()
    assert np.all(np.diff(yt) > 0), f"{sysY_name} times must be increasing"

    if len(xt) == len(yt):
        pass
    elif len(xt) < len(yt):
        xt, yt = _match_edges(xt, yt)
    elif len(xt) > len(yt):
        yt, xt = _match_edges(yt, xt)

    return fit_times(xt, yt, xname=sysX_name, yname=sysY_name, visualize=visualize)


def _check_pulse_widths(
    pulses: pd.DataFrame,
    expected_pulse_width: float,
    atol: float = 0.001,
    sys_name: str = "system",
) -> pd.DataFrame:
    pulses["width"] = pulses["falling"] - pulses["rising"]
    pulses["width_discrepancy"] = pulses["width"] - expected_pulse_width
    is_discrepant = pulses["width_discrepancy"].abs() > atol
    n_discrepant = is_discrepant.sum()
    if n_discrepant:
        logger.warning(
            f"Discarding {n_discrepant} {sys_name} pulses with a width discrepancy greater than {atol} seconds."
        )
        logger.warning(pulses[is_discrepant])
        pulses = pulses[~is_discrepant]
    return pulses


def _match_edges(less_edges, more_edges, atol=0.01):
    """Find the edges in `more_edges` that are closest to the edges in `less_edges`."""
    assert len(more_edges) > len(
        less_edges
    ), "First argument must have more edges than second argument"
    matched_less_edges = []
    matched_more_edges = []
    for less_edge in less_edges:
        ix = utils.find_nearest(more_edges, less_edge)
        if np.isclose(more_edges[ix], less_edge, atol=atol):
            matched_less_edges.append(less_edge)
            matched_more_edges.append(more_edges[ix])
    return np.asarray(matched_less_edges), np.asarray(matched_more_edges)


#####
# e3vision specific functions
#####
def get_e3v_pulse_width_threshold_from_fps(fps):
    """Calculate threshold for detecting frames written to disk from the camera's framerate.

    Cameras have a 10% duty cycle when streaming but not writing to disk, and a 50% duty cycle when writing to disk.
    So if you are recording at 15 fps, these pulse widths are 1/15 * 0.1 and 1/15 * 0.5, respectively.
    We use the average of the two as the threshold.

    See: https://docs.white-matter.com/docs/e3vision/system-details/hardware/hub/

    Returns
    -------
    threshold: (float)
        The minimum pulse width for frames written to disk, in seconds.
    """
    frame_duration = 1 / fps
    acq_width = 0.1 * frame_duration
    rec_width = 0.5 * frame_duration
    return (acq_width + rec_width) / 2


def get_frame_times_from_e3v(rising, falling, fps):
    """Get frame start times from frame capture pulse edges."""
    rising, falling = check_edges(rising, falling)
    threshold = get_e3v_pulse_width_threshold_from_fps(fps)
    pulse_width = falling - rising
    saved_frames = np.where(pulse_width > threshold)
    return rising[saved_frames]


# MUCH slower than the version above, which assumes certain constraints on the input edges.
# So slow, in fact, that it is probably better to spend time enforcing those conditions.
# This function is probably safe to delete.
def _get_frame_times_from_e3v(rising_edges, falling_edges, fps):
    threshold = get_e3v_pulse_width_threshold_from_fps(fps)
    frame_start_edges = list()
    for pulse_on in rising_edges:
        pulse_off = falling_edges[falling_edges > pulse_on].min()
        pulse_width = pulse_off - pulse_on
        if pulse_width > threshold:
            frame_start_edges.append(pulse_on)

    return np.asarray(frame_start_edges)


#####
# TDT specific functions
#####


def load_epoc_store_from_tdt(block_path, store_name, start_time=0, end_time=0):
    """Load a single epoc store from disk.

    Parameters:
    -----------
    block_path:
        Path to the TDT block directory which contains files of type .Tbk, .tev, .tsq, etc.
    store_name: string
        The name of the epoc store to load.
    start_time: float, optional, default: 0
        Start time of the data to load, relative to the file start, in seconds.
    end_time: float, optional, default: 0
        End time of the data to load, relative to the file start, in seconds.
        Passing `0` (default) will load until the end of the file.

    Returns:
    --------
    info: tdt.StructType
        The `info` field of a tdt `blk` struct, used to get the file start time.
    store: tdt.StructType
        The store field of a tdt `blk.epocs` struct, which contains the data.
    """
    assert store_name not in [
        "epocs",
        "snips",
        "streams",
        "scalars",
        "info",
        "time_ranges",
    ]

    read_block_kwargs = dict(store=["info", store_name], t1=start_time, t2=end_time)
    blk = tdt.read_block(block_path, **read_block_kwargs)

    # TDT will replace all trailing / with _
    if store_name[-1] == "/":
        store_name = store_name[:-1] + "_"

    return blk.info, blk.epocs[store_name]


def get_rising_edges_from_tdt(onset_times):
    """Assumes onset times follow TDT convention:

    Onsets are the first samples where a value is high after being low.
    If sample 0 is high, the first onset is sample 0. Consistent with TDT convention.
    """
    return onset_times[np.where(onset_times > 0)]


def get_falling_edges_from_tdt(offset_times):
    """Assumes offset times follow TDT convention:

    Offsets are the first samples where a value is low after being high.
    If the last sample is high, the last offset is Inf. Consistent with TDT convention.
    """
    return offset_times[np.where(offset_times < np.Inf)]


def extract_ttl_edges_from_tdt(block_path, store_name):
    _, store = load_epoc_store_from_tdt(block_path, store_name)
    # Sometimes TDT calls the last offset Inf even when it was not the last sample.
    # Breaking their own convention... maddening
    if (store.onset.size == store.offset.size) and (store.offset[-1] == np.inf):
        utils.warn(
            "TDT strongly suspected of messing up the last falling edge. Dropping final pulse."
        )
        store.onset = store.onset[:-1]
        store.offset = store.offset[:-1]
    rising = get_rising_edges_from_tdt(store.onset)
    falling = get_falling_edges_from_tdt(store.offset)
    return check_edges(rising, falling)


def get_tdt_barcodes(block_path, store_name, bar_duration=0.029):
    rising, falling = extract_ttl_edges_from_tdt(block_path, store_name)
    return extract_barcodes_from_times(rising, falling, bar_duration=bar_duration)


#####
# SpikeGLX specific functions
#####
# TODO: Pull these from CSC-UW/barcode_sync/counter_barcode_sync.ipynb, NOT e.g. e3vision_sync.ipynb
# TODO: Incorporate some of these into sglxarray


def load_sync_channel_from_sglx_imec(bin_path):
    """Load the sync channel from the specified binary file.
    The SpikeGLX metadata file must be present in the same directory as the binary file.
    """
    meta = readSGLX.readMeta(bin_path)
    rawData = readSGLX.makeMemMapRaw(bin_path, meta)
    fs = readSGLX.SampRate(meta)

    # Read the entire file
    firstSamp = 0
    lastSamp = rawData.shape[1] - 1

    # Get timestamps of each sample
    time = np.arange(firstSamp, lastSamp + 1)
    time = time / fs  # timestamps in seconds from start of file

    # Which digital word to read.
    # For imec, there is only 1 digital word, dw = 0.
    # For NI, digital lines 0-15 are in word 0, lines 16-31 are in word 1, etc.
    dw = 0
    # Which lines within the digital word, zero-based
    # Note that the SYNC line for PXI 3B is stored in line 6.
    dLineList = [6]
    sync = np.squeeze(
        readSGLX.ExtractDigital(rawData, firstSamp, lastSamp, dw, dLineList, meta)
    )

    return sync, time


def extract_ttl_edges_from_sglx_imec(bin_path):
    sglx_sync_in, sglx_times = load_sync_channel_from_sglx_imec(bin_path)

    rising_samples = get_rising_edges_from_binary_signal(sglx_sync_in)
    falling_samples = get_falling_edges_from_binary_signal(sglx_sync_in)

    rising = sglx_times[rising_samples]
    falling = sglx_times[falling_samples]

    return check_edges(rising, falling)


def _get_sglx_imec_barcodes(bin_path, bar_duration=0.029):
    """Since extracting the TTL edges is so costly, it is better to do that and save to disk first, then extract the barcodes from those saved TTLs, in case any barcodes are malformed."""
    rising, falling = extract_ttl_edges_from_sglx_imec(bin_path)
    return extract_barcodes_from_times(rising, falling, bar_duration=bar_duration)


def get_sglx_nidq_barcodes(bin_path, sync_channel, bar_duration=0.029, threshold=4000):
    sig = ecephys.sglx.load_nidq_analog(bin_path, channels=[sync_channel])
    sig = sig.sel(channel=sync_channel)
    nidq_barcode_in = binarize(sig.values, threshold=threshold)

    nidq_rising_edge_samples = get_rising_edges_from_binary_signal(nidq_barcode_in)
    nidq_falling_edge_samples = get_falling_edges_from_binary_signal(nidq_barcode_in)

    rising = sig.time.values[nidq_rising_edge_samples]
    falling = sig.time.values[nidq_falling_edge_samples]
    rising, falling = check_edges(rising, falling)
    return extract_barcodes_from_times(rising, falling, bar_duration=bar_duration)


#####
# Barcode parsing
#####


def extract_barcodes_from_times(
    on_times,
    off_times,
    inter_barcode_interval=10,
    bar_duration=0.03,
    barcode_duration_ceiling=2,
    nbits=32,
):
    """Read barcodes from timestamped rising and falling edges.

    Parameters
    ----------
    on_times : numpy.ndarray
        Timestamps of rising edges on the barcode line
    off_times : numpy.ndarray
        Timestamps of falling edges on the barcode line
    inter_barcode_interval : numeric, optional
        Minimun duration of time between barcodes.
    bar_duration : numeric, optional
        A value slightly shorter than the expected duration of each bar
    barcode_duration_ceiling : numeric, optional
        The maximum duration of a single barcode
    nbits : int, optional
        The bit-depth of each barcode

    Returns
    -------
    barcode_start_times : list of numeric
        For each detected barcode, the time at which that barcode started
    barcodes : list of int
        For each detected barcode, the value of that barcode as an integer.

    Notes
    -----
    ignores first code in prod (ok, but not intended)
    ignores first on pulse (intended - this is needed to identify that a barcode is starting)

    Original taken from Open Ephys code, modified 6/22/2023 by Graham Findlay, to handle cases where malfunctioning hardware produces occasional malformed barcodes.
    """

    def _extract_barcode(barcode_start_time):
        oncode = on_times[
            np.where(
                np.logical_and(
                    on_times > barcode_start_time,
                    on_times < barcode_start_time + barcode_duration_ceiling,
                )
            )[0]
        ]
        offcode = off_times[
            np.where(
                np.logical_and(
                    off_times > barcode_start_time,
                    off_times < barcode_start_time + barcode_duration_ceiling,
                )
            )[0]
        ]

        currTime = offcode[0]

        bits = np.zeros((nbits,))

        for bit in range(0, nbits):
            nextOn = np.where(oncode > currTime)[0]
            nextOff = np.where(offcode > currTime)[0]

            if nextOn.size > 0:
                nextOn = oncode[nextOn[0]]
            else:
                nextOn = barcode_start_time + inter_barcode_interval

            if nextOff.size > 0:
                nextOff = offcode[nextOff[0]]
            else:
                nextOff = barcode_start_time + inter_barcode_interval

            if nextOn < nextOff:
                bits[bit] = 1

            currTime += bar_duration

        barcode = 0

        # least sig left
        for bit in range(0, nbits):
            barcode += bits[bit] * pow(2, bit)

        return barcode

    start_indices = np.diff(on_times)
    a = np.where(start_indices > inter_barcode_interval)[0]
    barcode_start_times = on_times[a + 1]

    barcodes = []
    for i, t in enumerate(barcode_start_times):
        try:
            barcode = _extract_barcode(t)
        except:
            warnings.warn(
                f"Problem extracting barcode {i}, t={t}. It is likely that previous & subsequent barcodes in this file are malformed, indicating a hardware issue."
            )
            barcodes.append(np.NaN)
        else:
            barcodes.append(barcode)

    return barcode_start_times, barcodes
