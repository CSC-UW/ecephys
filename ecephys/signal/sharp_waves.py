from . import event_detection as evt


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
