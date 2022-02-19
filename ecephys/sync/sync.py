import pyabf
import tdt
import numpy as np
import matplotlib.pyplot as plt
from warnings import warn
from sklearn.linear_model import LinearRegression


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


def visualize_mapping(x, y, model, nPulsesToPlot=10):
    """Visualize remapped sync pulse times from x in y's time space, to verify correctness.
    Black lines = Pulse times measured by y.
    Red lines = Pulse times measured by x and remapped to y's time space."""
    y_pred = model.predict(x)

    plt.figure(num=None, figsize=(30, 6), dpi=80, facecolor="w", edgecolor="k")
    plt.vlines(y[:nPulsesToPlot], 0, 1, colors="black")
    plt.vlines(y_pred[:nPulsesToPlot], 0, 1, linestyle="dotted", colors="red")


def check_edges(rising, falling):
    assert (
        rising.size == falling.size
    ), "Number of rising and falling edges do not match."
    assert all(np.less(rising, falling)), "Falling edges occur before rising edges."


def fit_times(x, y, visualize=True):
    """Using event times shared by both systems, fit the X's measured times to Y's measured times."""
    x = x.reshape((-1, 1))
    model = LinearRegression().fit(x, y)
    r_sq = model.score(x, y)

    print("coefficient of determination:", r_sq)
    print("intercept:", model.intercept_)
    print("drift rate in msec/hr:", (model.coef_[0] - 1) * 60 * 60 * 1000)
    print("(assumes `x` and `y` are provided in seconds)")

    if visualize:
        visualize_mapping(x, y, model)

    return model


##### ABF specific functions #####


def extract_ttl_abf(abf_path):
    abf = pyabf.ABF(abf_path)

    TTL_IN = 2
    assert abf.adcNames[TTL_IN] == "TTL_IN"
    assert abf.adcUnits[TTL_IN] == "V"
    ttl_sig = abf.data[TTL_IN, :]  # in volts
    time = abf.getAllXs()  # in seconds

    return ttl_sig, time


def get_rising_edges_abf(data, ttl_threshold, time=None):
    rising_edges = (
        np.flatnonzero((data[:-1] < ttl_threshold) & (data[1:] > ttl_threshold)) + 1
    )
    if time is not None:
        rising_edges = time[rising_edges]
    return rising_edges


def get_falling_edges_abf(data, ttl_threshold, time=None):
    falling_edges = (
        np.flatnonzero((data[:-1] > ttl_threshold) & (data[1:] < ttl_threshold)) + 1
    )
    if time is not None:
        falling_edges = time[falling_edges]
    return falling_edges


def extract_ttl_edges_abf(abf_path, ttl_threshold, time=None):
    data, time = extract_ttl_abf(abf_path)
    rising = get_rising_edges_abf(data, ttl_threshold, time)
    falling = get_falling_edges_abf(data, ttl_threshold, time)
    check_edges(rising, falling)
    return rising, falling


##### TDT specific functions #####


def load_tdt_epoc_store(block_path, store_name, start_time=0, end_time=0):
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


def get_rising_edges_tdt(onset_times):
    """Assumes onset times follow TDT convention:

    Onsets are the first samples where a value is high after being low.
    If sample 0 is high, the first onset is sample 0. Consistent with TDT convention.
    """
    return onset_times[np.where(onset_times > 0)]


def get_falling_edges_tdt(offset_times):
    """Assumes offset times follow TDT convention:

    Offsets are the first samples where a value is low after being high.
    If the last sample is high, the last offset is Inf. Consistent with TDT convention.
    """
    return offset_times[np.where(offset_times < np.Inf)]


def extract_ttl_edges_tdt(block_path, store_name):
    _, store = load_tdt_epoc_store(block_path, store_name)
    # Sometimes TDT calls the last offset Inf even when it was not the last sample.
    # Breaking their own convention... maddening
    if (store.onset.size == store.offset.size) and (store.offset[-1] == np.inf):
        warn(
            "TDT strongly suspected of messing up the last falling edge. Dropping final pulse."
        )
        store.onset = store.onset[:-1]
        store.offset = store.offset[:-1]
    rising = get_rising_edges_tdt(store.onset)
    falling = get_falling_edges_tdt(store.offset)
    check_edges(rising, falling)
    return rising, falling
