import pandas as pd
import numpy as np
from . import event_detection as evt
from ..utils import dt_series_to_seconds, round_to_values, all_arrays_equal, get_epocs
from statsmodels.nonparametric.smoothers_lowess import lowess

# -------------------- Detection related functions --------------------


def get_detection_series(
    csd, coarse_detection_chans=slice(None), n_fine_detection_chans=5
):
    _csd = (
        csd.sel(channel=coarse_detection_chans)
        .rolling(channel=n_fine_detection_chans, center=True)
        .mean()
        .dropna(dim="channel")
    )
    return -_csd[_csd.argmin(dim="channel")]


def get_peak_info(sig, spws):
    spws = spws.copy()

    def _get_peak_info(spw):
        spw_sig = sig.sel(time=slice(spw.start_time, spw.end_time))
        peak = spw_sig[spw_sig.argmax()]
        return peak.item(), peak.time.item(), peak.channel.item()

    info = list(map(_get_peak_info, spws.itertuples()))
    spws[["peak_amplitude", "peak_time", "peak_channel"]] = info
    return spws


def get_coarse_detection_chans(peak_channel, n_coarse_detection_chans, csd_chans):
    assert (
        n_coarse_detection_chans % 2
    ), "Must use an odd number of of detection channels."

    idx = csd_chans.index(peak_channel)
    first = idx - n_coarse_detection_chans // 2
    last = idx + n_coarse_detection_chans // 2 + 1

    assert first >= 0, "Cannot detect SPWs outside the bounds of your CSD."
    assert last < len(csd_chans), "Cannot detect SPWs outside the bounds of your CSD."

    return csd_chans[first:last]


def detect_by_value(
    csd,
    initial_peak_channel,
    n_coarse_detection_chans,
    detection_threshold,
    boundary_threshold,
    minimum_duration=0.005,
):
    csd = csd.swap_dims({"pos": "channel"})
    cdc = get_coarse_detection_chans(
        initial_peak_channel, n_coarse_detection_chans, csd.channel.values.tolist()
    )
    ser = get_detection_series(csd, cdc)

    spws = evt.detect_by_value(
        ser.values,
        ser.time.values,
        detection_threshold,
        boundary_threshold,
        minimum_duration,
    )
    spws.attrs["initial_peak_channel"] = initial_peak_channel
    spws.attrs["n_coarse_detection_chans"] = n_coarse_detection_chans
    if "datetime" in ser.coords:
        spws.attrs["t0"] = ser.datetime.values.min()
    return get_peak_info(ser, spws)


def detect_by_zscore(
    csd,
    initial_peak_channel,
    n_coarse_detection_chans,
    detection_threshold=2.5,
    boundary_threshold=1.5,
    minimum_duration=0.005,
):
    csd = csd.swap_dims({"pos": "channel"})
    cdc = get_coarse_detection_chans(
        initial_peak_channel, n_coarse_detection_chans, csd.channel.values.tolist()
    )
    ser = get_detection_series(csd, cdc)

    spws = evt.detect_by_zscore(
        ser.values,
        ser.time.values,
        detection_threshold,
        boundary_threshold,
        minimum_duration,
    )
    spws.attrs["initial_peak_channel"] = initial_peak_channel
    spws.attrs["n_coarse_detection_chans"] = n_coarse_detection_chans
    if "datetime" in ser.coords:
        spws.attrs["t0"] = ser.datetime.values.min()
    return get_peak_info(ser, spws)


# -------------------- Drift related functions --------------------


def _estimate_drift(t, pos, **kwargs):
    out = lowess(pos, t, **kwargs)
    return out[:, 0], out[:, 1]


def estimate_drift(spws, imec_map, frac=1 / 48, it=3):
    um_per_mm = 1000
    peak_times = dt_series_to_seconds(spws.peak_time)
    peak_ycoords = imec_map.chans2coords(spws.peak_channel)[:, 1] / um_per_mm
    t, y = _estimate_drift(peak_times, peak_ycoords, frac=frac, it=it)
    nearest_y = round_to_values(y, imec_map.y / um_per_mm)
    t0_chan = imec_map.lf_map.set_index("y").loc[nearest_y[0] * um_per_mm]
    nearest_chans = imec_map.lf_map.set_index("y").loc[nearest_y * um_per_mm]
    nearest_ids = nearest_chans.chan_id.values
    usr_order_shift = nearest_chans.usr_order.values - t0_chan.usr_order

    return pd.DataFrame(
        {
            "dt": spws.peak_time.values,
            "t": t,
            "y": y,
            "nearest_y": nearest_y,
            "nearest_id": nearest_ids,
            "shifts": usr_order_shift,
        }
    )


def get_drift_epocs(drift):
    cols = ["nearest_y", "nearest_id", "shifts"]
    dfs = list(get_epocs(drift, col, "dt") for col in cols)
    assert all_arrays_equal(
        df.index for df in dfs
    ), "Different columns yielded different epochs."

    return pd.concat(dfs, axis=1)


def get_shift_timeseries(epocs, times):
    shifts = np.full(len(times), np.nan)
    for epoc in epocs.reset_index().itertuples():
        times_in_epoc = (times >= epoc.start_dt) & (times <= epoc.end_dt)
        shifts[times_in_epoc] = epoc.shifts

    mask = np.isnan(shifts)
    shifts[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), shifts[~mask])

    return shifts.astype(np.int64)
