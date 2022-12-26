import tdt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from difflib import SequenceMatcher
from ..sglx.external.readSGLX import readMeta, SampRate, makeMemMapRaw, ExtractDigital
from ..sglx import load_nidq_analog
from .external.barcodes import extract_barcodes_from_times
from ..utils import warn

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
    assert (
        rising.size == falling.size
    ), "Number of rising and falling edges do not match."
    assert all(np.less(rising, falling)), "Falling edges occur before rising edges."


def fit_times(x, y, visualize=True, xname="X", yname="Y"):
    """Using event times shared by both systems, fit the X's measured times to Y's measured times."""
    x = x.reshape((-1, 1))
    model = LinearRegression().fit(x, y)
    r_sq = model.score(x, y)

    print("coefficient of determination:", r_sq)
    print("intercept:", model.intercept_)
    print("drift rate in msec/hr:", (model.coef_[0] - 1) * 60 * 60 * 1000)
    print("(assumes `x` and `y` are provided in seconds)")

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


def get_sync_model(
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
    check_edges(rising, falling)
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
        warn(
            "TDT strongly suspected of messing up the last falling edge. Dropping final pulse."
        )
        store.onset = store.onset[:-1]
        store.offset = store.offset[:-1]
    rising = get_rising_edges_from_tdt(store.onset)
    falling = get_falling_edges_from_tdt(store.offset)
    check_edges(rising, falling)
    return rising, falling


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
    The SpikeGLX metadata file must be present in the same directory as the binary file."""
    meta = readMeta(bin_path)
    rawData = makeMemMapRaw(bin_path, meta)
    fs = SampRate(meta)

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
    sync = np.squeeze(ExtractDigital(rawData, firstSamp, lastSamp, dw, dLineList, meta))

    return sync, time


def get_sglx_imec_barcodes(bin_path, bar_duration=0.029):
    """Get SpikeGLX barcodes and times

    Returns
    --------
    (barcode_start_times, barcode_values)
    """
    sglx_barcode_in, sglx_times = load_sync_channel_from_sglx_imec(bin_path)

    sglx_rising_edge_samples = get_rising_edges_from_binary_signal(sglx_barcode_in)
    sglx_falling_edge_samples = get_falling_edges_from_binary_signal(sglx_barcode_in)

    sglx_rising_edge_times = sglx_times[sglx_rising_edge_samples]
    sglx_falling_edge_times = sglx_times[sglx_falling_edge_samples]

    return extract_barcodes_from_times(
        sglx_rising_edge_times, sglx_falling_edge_times, bar_duration=bar_duration
    )


def get_sglx_nidq_barcodes(bin_path, sync_channel, bar_duration=0.029, threshold=4000):
    sig = load_nidq_analog(bin_path, channels=[sync_channel])
    sig = sig.sel(channel=sync_channel)
    nidq_barcode_in = binarize(sig.values, threshold=threshold)

    nidq_rising_edge_samples = get_rising_edges_from_binary_signal(nidq_barcode_in)
    nidq_falling_edge_samples = get_falling_edges_from_binary_signal(nidq_barcode_in)

    nidq_rising_edge_times = sig.time.values[nidq_rising_edge_samples]
    nidq_falling_edge_times = sig.time.values[nidq_falling_edge_samples]

    return extract_barcodes_from_times(
        nidq_rising_edge_times, nidq_falling_edge_times, bar_duration=bar_duration
    )
