import logging
import os

import numpy as np
import pandas as pd
import xarray as xr

import ecephys.utils
import ecephys.utils.dask as dask_utils
from ecephys import dasig, emg_from_lfp, npsig

logger = logging.getLogger(__name__)


def validate_timeseries(
    da: xr.DataArray,
    timedim: str = "time",
    check_times: bool = False,
):
    if timedim not in da.dims:
        raise AttributeError(f"Timeseries DataArray must have dimension ({timedim})")
    if "fs" not in da.attrs:
        raise ValueError("Timeseries must have sampling rate attr `fs`")
    if check_times and not np.all(np.diff(da[timedim].values) >= 0):
        raise ValueError("Timeseries times must be monotonically increasing.")


def validate_2d_timeseries(
    da: xr.DataArray,
    timedim: str = "time",
    sigdim: str = "channel",
    check_times: bool = False,
):
    validate_timeseries(da, timedim=timedim, check_times=check_times)
    if not da.dims == (timedim, sigdim):
        raise AttributeError(
            f"Timeseries2D DataArray must have dimensions ({timedim}, {sigdim})"
        )


def validate_laminar(da: xr.DataArray, sigdim: str = "channel", lamdim: str = "y"):
    if sigdim not in da.dims:
        raise AttributeError("Laminar DataArray must include a channel dimension.")
    if lamdim not in da[sigdim].coords:
        raise AttributeError(
            f"Laminar DataArray must have {lamdim} coordinate on {sigdim} dimension."
        )


def get_pitch(da: xr.DataArray) -> xr.DataArray:
    """Get the vertical spacing between electrode sites, in microns"""
    validate_laminar(da)
    vals = np.diff(np.unique(da["y"].values))
    assert ecephys.utils.all_equal(vals), (
        f"Electrode pitch is not uniform. Pitches:\n {vals}"
    )
    return np.absolute(vals[0])


def _decompose_decimation_factor(
    q: int, max_stage: int = 12
) -> list[int]:
    """Break *q* into a list of factors each <= *max_stage*.

    For example ``_decompose_decimation_factor(50)`` returns ``[10, 5]``.
    Raises ``ValueError`` if *q* has a prime factor larger than *max_stage*.
    """
    if q <= max_stage:
        return [q]
    stages: list[int] = []
    remaining = q
    while remaining > max_stage:
        for f in range(max_stage, 1, -1):
            if remaining % f == 0:
                stages.append(f)
                remaining //= f
                break
        else:
            raise ValueError(
                f"Cannot decompose decimation factor {q} into stages "
                f"<= {max_stage}: remaining factor {remaining} is prime."
            )
    if remaining > 1:
        stages.append(remaining)
    return stages


def decimate_timeseries(da: xr.DataArray, q: int) -> xr.DataArray:
    """Decimate a 2-D timeseries by factor *q*.

    For *q* > 12 the decimation is automatically split into multiple
    stages with per-stage factors <= 12, following the
    ``scipy.signal.decimate`` recommendation.
    """
    validate_2d_timeseries(da)
    stages = _decompose_decimation_factor(q)
    for stage_q in stages:
        da = antialiasing_filter(da, stage_q)
        da = da.isel(time=slice(None, None, stage_q))
        da.attrs["fs"] = da.fs / stage_q
    return da


def antialiasing_filter(da: xr.DataArray, q: int) -> xr.DataArray:
    """This function is lazy and works on chunked data, but your chunks have to be long
    enough to accomodate the impulse response of the filter."""
    validate_2d_timeseries(da)
    res = da.copy()
    if da.chunks is None:
        res.values = npsig.antialiasing_filter(
            da.values, q, time_axis=da.get_axis_num("time")
        )
    else:
        res.data = dasig.antialiasing_filter(
            res.data,
            res.fs,
            q,
            time_axis=da.get_axis_num("time"),
        )
    return res.__class__(res)


def mne_filter(
    da: xr.DataArray, l_freq: float, h_freq: float, **kwargs
) -> xr.DataArray:
    import mne.filter

    validate_2d_timeseries(da)
    res = da.copy()
    if da.chunks is None:
        original_dtype = res.dtype
        res.values = mne.filter.filter_data(
            da.values.T.astype(np.float64), da.fs, l_freq, h_freq, **kwargs
        ).T.astype(original_dtype)
    else:
        res.data = dasig.mne_filter(res.data.T, res.fs, l_freq, h_freq, **kwargs).T
    return res.__class__(res)


def spatially_interpolate_timeseries(
    da: xr.DataArray,
    interp_me: list,  # The channels that should be interpolated
    inplace: bool = True,
) -> xr.DataArray:
    from ibldsp import voltage

    validate_2d_timeseries(da)
    validate_laminar(da)
    do_interp = np.isin(da["channel"], interp_me)
    if not do_interp.any():
        logger.debug(
            "None of the requested signals are present in the data. Doing nothing."
        )
        return

    # Get x coordinates if available, otherwise assume that channels are colinear
    if "x" in da["channel"].coords:
        x = da["x"].values
    else:
        print(
            "Data do not contain x coordinates on channel dimension. Assuming all electrodes are colinear."
        )
        x = np.zeros_like(da["y"])

    if inplace:
        da = da.copy()
    da.values = voltage.interpolate_bad_channels(
        da.values.T, do_interp.astype("int"), x=x, y=da["y"].values
    ).T
    return da


def dephase_neuropixels(
    pots: xr.DataArray, q: int = 1, inplace: bool = True
) -> xr.DataArray:
    import neuropixel
    from ibldsp import fourier

    validate_2d_timeseries(pots)
    hdr = neuropixel.trace_header(version=1)
    shifts = hdr["sample_shift"][pots["channel"].values] / q
    if not inplace:
        pots = pots.copy()
    pots.values = fourier.fshift(pots.values, shifts, axis=0)
    return pots


def preprocess_neuropixels_ibl_style(
    pots: xr.DataArray,
    bad_chans: list,
    downsample_factor: int = 4,
    chunk_size: int = 2**16,
    chunk_overlap: int = 2**10,
) -> xr.DataArray:
    import ibldsp.utils
    from tqdm.auto import tqdm

    validate_2d_timeseries(pots)
    validate_laminar(pots)
    wg = ibldsp.utils.WindowGenerator(
        ns=pots["time"].size, nswin=chunk_size, overlap=chunk_overlap
    )
    segments = list()
    for first, last in tqdm(list(wg.firstlast)):
        seg = pots.isel(time=slice(first, last))
        seg = decimate_timeseries(seg, q=downsample_factor)
        seg = dephase_neuropixels(seg, q=downsample_factor)
        seg = spatially_interpolate_timeseries(seg, bad_chans)
        first_valid = 0 if first == 0 else int(wg.overlap / 2 / downsample_factor)
        last_valid = (
            seg["time"].size
            if last == pots["time"].size
            else int(seg["time"].size - wg.overlap / 2 / downsample_factor)
        )
        segments.append(seg.isel(time=slice(first_valid, last_valid)))

    return xr.concat(segments, dim="time")


def get_synthetic_emg_defaults() -> dict:
    return dict(
        target_sf=20,
        window_size=25.0,
        wp=[300, 600],
        ws=[275, 625],
        gpass=1,
        gstop=60,
        ftype="butter",
        method="both",
    )


#: Per-method `units` attribute for the returned EMG.
_EMG_UNITS = {"per_window": "corr", "global": "corr (amplitude-weighted)"}


def synthetic_emg(pots: xr.DataArray, emg_kwargs: dict = None):
    """Estimate the EMG from LFP signals, using the `emg_from_lfp` subpackage.

    Parameters used for the computation are stored as attributes on the result.

    Parameters:
    -----------
    emg_kwargs:
        Keyword arguments passed to `emg_from_lfp.compute()`, overriding
        `get_synthetic_emg_defaults()`. `method` selects the estimator(s):
        "per_window" (exact per-window correlation), "global" (faster
        amplitude-weighted approximation; see emg_from_lfp._compute_global_corr),
        or "both" (default). "both" shares the band-pass filter, so it costs only
        marginally more than a single method.

    Returns:
    --------
    For a single `method`: a `DataArray` with a `time` dimension. For
    `method="both"`: a `Dataset` with one variable per method (`per_window`,
    `global`) sharing the `time` coordinate. Computation parameters are stored as
    attributes.
    """
    validate_2d_timeseries(pots)
    defaults = get_synthetic_emg_defaults()
    emg_kwargs = defaults if emg_kwargs is None else {**defaults, **emg_kwargs}
    assert pots.fs > (emg_kwargs["ws"][-1] * 2), (
        "EMG computation will fail trying to filter above the Nyquest frequency"
    )

    res = emg_from_lfp.compute(pots.values.T, pots.fs, **emg_kwargs)

    def _times(n):
        return np.linspace(
            float(pots["time"].min()), float(pots["time"].max()), n
        )

    if isinstance(res, dict):  # method="both"
        out = xr.Dataset(
            {m: ("time", arr.flatten()) for m, arr in res.items()},
            coords={"time": _times(next(iter(res.values())).size)},
        )
        for m in res:
            out[m].attrs["units"] = _EMG_UNITS[m]
    else:
        vals = res.flatten()
        out = xr.DataArray(
            vals,
            dims="time",
            coords={"time": _times(vals.size)},
            attrs={"units": _EMG_UNITS.get(emg_kwargs["method"], "corr")},
        )
    out.attrs.update(emg_kwargs)
    return out


def kernel_current_source_density(
    pots: xr.DataArray,
    drop=slice(None),
    do_lcurve=False,
    lcurve_kwargs: dict = None,
    **kcsd_kwargs,
) -> xr.DataArray:
    """Compute 1D kernel current source density.
    If signal units are in uV, then CSD units are in nA/mm.
    Evaluates eagerly, non-parallel.

    Required coords:
    ----------------
    y, with units in um

    Paramters:
    ----------
    drop_chans: list
        Channels (as they appear in `self`) to exclude when estimating the CSD.
    do_lcurve: Boolean
        Whether to perform L-Curve parameter estimation.
    **kcsd_kwargs:
        Keywords passed to KCSD1D.

    Returns:
    --------
    csd: KernelCurrentSourceDensity
        The CSD estimates. If the estimation locations requested of KCSD1D correspond
        exactly to electrode positions, a `channel` coordinate on the `pos` dimension
        will give corresponding channels for each estimate.
    """
    import kcsd

    validate_2d_timeseries(pots)
    validate_laminar(pots)
    umPerMm = 1000

    # Make sure we get CSD estimates at electrode locations, rather than say, in between electrodes.
    pitch_mm = get_pitch(pots) / umPerMm  # Convert um to mm for KCSD package.
    gdx = kcsd_kwargs.get("gdx", None)
    if (gdx is not None) and (gdx != pitch_mm):
        raise ValueError("Requested gdx does not match electrode pitch.")
    else:
        kcsd_kwargs.update(gdx=pitch_mm)

    # Drop bad signals and redundant signals
    good = pots.drop_sel({"channel": drop}, errors="ignore")
    u, ix = np.unique(good["y"], return_index=True)
    good = good.isel({"channel": ix})

    # Convert um to mm for KCSD package.
    elePosMm = good["y"].values / umPerMm

    # Compute kCSD
    k = kcsd.KCSD1D(
        elePosMm.reshape(-1, 1),
        good.transpose("channel", "time").values,
        **kcsd_kwargs,
    )
    if do_lcurve:
        print("Performing L-Curve parameter estimation...")
        k.L_curve(**(lcurve_kwargs if lcurve_kwargs is not None else dict()))

    # Check and format result
    estm_locs = np.round(k.estm_x * umPerMm)
    mask = pots["y"].isin(estm_locs)
    assert estm_locs.size == mask.sum(), (
        "CSD returned estimates that do not match original signal positions exactly."
    )
    csd = xr.zeros_like(pots.sel({"channel": mask}))
    csd.values = k.values("CSD").T
    csd.attrs = dict(kcsd=k, pitch_mm=pitch_mm, fs=pots.fs)
    csd.name = "csd"
    return csd


def lazy_mapped_kernel_current_source_density(
    pots: xr.DataArray, **kwargs
) -> xr.DataArray:
    """
    Intended for lazy, chunked, parallelization across time.
    Sadly, you cannot get attrs generated in the workhorse function this way.
    For ~2h of data, the computation time here is slightly higher than using the eager, non-parallel version.
    16-21s parallel from disk, vs 11-18s non-parallel from disk, vs 10s non-parallel from in-memory.

    Examples
    --------
    lf = xr.open_dataarray('my_lfps.zarr', engine='zarr', chunks='auto')
    csd = lazy_mapped_kernel_current_source_density(lf, drop=bad_chans, do_lcurve=False, gdx=123, r_init=456, lambd=789)
    lazy = csd.sel(channel=chans_of_interest, time=slice(t1, t2))
    dat = lazy.compute()
    """
    tmpl = pots.copy()  # Result will have same shape and dims as input
    tmpl.name = "csd"
    tmpl.attrs = (
        dict()
    )  # Input attrs may not be relevant, so stop them from being copies to the output.
    csd = pots.map_blocks(
        kernel_current_source_density, kwargs=kwargs, template=tmpl
    )  # No attrs :'(
    csd.encoding = (
        dict()
    )  # Prevent irrelevant pots encoding from carrying over and messing with to_zarr
    if "fs" in pots.attrs:
        csd.attrs["fs"] = pots.attrs["fs"]
    return csd


def get_segments(
    da: xr.DataArray, gap_tolerance: float = 0.001
) -> list[tuple[int, int]]:
    """Detect discontinuous segments of data that have been concatenated together.

    Parameters:
    -----------
    gap_tolerance: float
        Segments are defined by gaps in the data longer than this value, in milliseconds
    """
    validate_2d_timeseries(da)
    dt = np.diff(da["time"].values)
    assert np.all(dt >= 0), "The times must be increasing."
    jumps = dt > ((1 / da.fs) + gap_tolerance)
    jump_ixs = np.where(jumps)[0]  # The jumps are between ix and ix + 1
    segments = np.insert(jump_ixs + 1, 0, 0)
    segments = np.append(segments, da["time"].size + 1)
    segments = [(i, j) for i, j in zip(segments[:-1], segments[1:])]

    segment_sizes = [da["time"].values[i:j].size for i, j in segments]
    assert np.sum(segment_sizes) == da["time"].values.size, (
        "Every sample in the data must be accounted for."
    )

    return segments


def stft_psd(da: xr.DataArray, gap_tolerance: float = 0.001, **kwargs) -> xr.DataArray:
    """Perform STFT, works on discontinuous segments of data that have been concatenated together."""
    validate_2d_timeseries(da)
    segments = get_segments(da, gap_tolerance=gap_tolerance)
    return xr.concat(
        [_stft_psd(da.isel(time=slice(i, j)), **kwargs) for i, j in segments],
        dim="time",
    )


def _stft_psd(da: xr.DataArray, **kwargs) -> xr.DataArray:
    """Only works for continuous segments of evenly-sample data."""
    validate_2d_timeseries(da)
    dt = np.diff(da["time"].values)
    assert np.all(dt >= 0), "The times must be increasing."
    Sfs, stft_times, Sxx = npsig.stft_psd(
        da.values.T, da.fs, t0=float(da["time"][0]), **kwargs
    )
    return xr.DataArray(
        Sxx,
        dims=("channel", "frequency", "time"),
        coords={
            "frequency": Sfs,
            "time": stft_times,
            **da["channel"].coords,
        },
        attrs=da.attrs,
    )


def complex_stft(
    da: xr.DataArray, gap_tolerance: float = 0.001, **kwargs
) -> xr.DataArray:
    """Perform STFT, works on discontinuous segments of data that have been concatenated together."""
    validate_2d_timeseries(da)
    segments = get_segments(da, gap_tolerance=gap_tolerance)
    if "n_fft" in kwargs:
        segments = [s for s in segments if s[1] - s[0] >= kwargs["n_fft"]]
    return xr.concat(
        [_complex_stft(da.isel(time=slice(i, j)), **kwargs) for i, j in segments],
        dim="time",
    )


def _complex_stft(da: xr.DataArray, **kwargs) -> xr.DataArray:
    """Only works for continuous segments of evenly-sample data."""
    validate_2d_timeseries(da)
    dt = np.diff(da["time"].values)
    assert np.all(dt >= 0), "The times must be increasing."
    Sfs, stft_times, Sxx = npsig.complex_stft(
        da.values.T, da.fs, t0=float(da["time"][0]), **kwargs
    )
    return xr.DataArray(
        Sxx,
        dims=("channel", "frequency", "time"),
        coords={
            "frequency": Sfs,
            "time": stft_times,
            **da["channel"].coords,
        },
        attrs=da.attrs,
    )


def naive_rechunk(da: xr.DataArray) -> xr.DataArray:
    validate_2d_timeseries(da)
    chunkaxis = da.get_axis_num("time")
    chunks = da.chunks[chunkaxis]
    assert ecephys.utils.all_equal(chunks[:-1]), (
        "All but last chunk must be the same size"
    )
    chunksize = chunks[0]
    return da.chunk(chunks={"time": chunksize})


def get_timeseries_chunk(da: xr.DataArray, chunk_index: int) -> xr.DataArray:
    validate_2d_timeseries(da)
    axis = da.get_axis_num("time")
    chunk_bounds = dask_utils.get_dask_chunk_bounds(da.data, axis=axis)
    start_frame = chunk_bounds[chunk_index]
    end_frame = chunk_bounds[chunk_index + 1]
    return da.isel({"time": slice(start_frame, end_frame)})


def iterate_timeseries_chunks(da: xr.DataArray):
    validate_2d_timeseries(da)
    axis = da.get_axis_num("time")
    chunk_bounds = dask_utils.get_dask_chunk_bounds(da.data, axis=axis)
    n_chunks = len(chunk_bounds) - 1
    return (
        da.isel({"time": slice(chunk_bounds[i], chunk_bounds[i + 1])})
        for i in range(n_chunks)
    )


def make_trialed(
    da: xr.DataArray,
    pre: float,
    post: float,
    event_frames: np.ndarray[int] = None,
    event_times: np.ndarray[float] = None,
) -> xr.DataArray:
    # It is absolutely necessary to have the data loaded into memory for decent performance.
    # xrsig.validate_timeseries(da, check_times=True)
    n_frames_pre = int(pre * da.fs)
    n_frames_post = int(post * da.fs)
    if event_frames is None:
        in_da = (event_times >= da.time.values[0] + pre) & (
            event_times <= da.time.values[-1] - post
        )
        event_frames = np.searchsorted(da.time.values, event_times[in_da])
    else:
        in_da = (event_frames >= n_frames_pre) & (
            event_frames <= da.time.size - n_frames_post
        )
        event_frames = event_frames[in_da]
    event_times = da.time.values[event_frames]

    trial_start_frames = event_frames - n_frames_pre
    trial_end_frames = event_frames + n_frames_post

    trialinfo = pd.DataFrame(
        {
            "event_time": event_times,
            "event_frame": event_frames.astype(int),
            "start_frame": trial_start_frames.astype(int),
            "end_frame": trial_end_frames.astype(int),
        }
    )
    trialinfo = trialinfo[
        ~(trialinfo < 0).any(axis=1)
    ]  # Remove events whose window starts before the data
    trialinfo = trialinfo[
        ~(trialinfo >= da.time.size).any(axis=1)
    ]  # Remove events whose window ends after the data

    num_trial_frames = n_frames_pre + n_frames_post
    assert all(trialinfo["end_frame"] - trialinfo["start_frame"] == num_trial_frames), (
        "Trials are uniform length."
    )

    # Reshape the LFP, adding an event dimension as the last dimension
    trials = []
    time = np.linspace(-pre, post, num_trial_frames)
    for trl in trialinfo.itertuples():
        da_trial = (
            da.isel(time=slice(trl.start_frame, trl.end_frame))
            .drop_vars("time")
            .assign_coords(time=time, event=trl.event_time)
        )
        trials.append(da_trial)
    return xr.concat(trials, dim="event"), in_da


def assign_laminar_coordinate(
    da: xr.DataArray,
    table: pd.DataFrame,
    sigdim: str = "channel",
    lamdim: str = "y",
    fill_value="???",
) -> xr.DataArray:
    """Label channels based on depth. Useful for adding anatomy."""
    validate_laminar(da, sigdim, lamdim)
    coords_to_add = [c for c in table.columns if c not in ["lo", "hi"]]
    depths = da[lamdim].to_numpy()
    for coord_name in coords_to_add:
        coord_values = np.empty(depths.shape, dtype=object)
        for i in range(len(table)):
            mask = (depths >= table["lo"].iloc[i]) & (depths <= table["hi"].iloc[i])
            coord_values[np.where(mask)] = table[coord_name].iloc[i]
        coord_values[pd.isnull(coord_values)] = fill_value
        da = da.assign_coords({coord_name: (sigdim, coord_values)})
    return da


def cwt(da: xr.DataArray, sigdim: str = "channel", parallel=True, **cwt_kwargs):
    """Complex wavelet transform. Do you have pyfftw installed?"""
    validate_2d_timeseries(da, sigdim=sigdim)
    if parallel:
        os.environ["SSQ_PARALLEL"] = "1"
    Wx, freqs, scales = npsig.cwt(da.values.T, da.fs, **cwt_kwargs)
    return (
        xr.DataArray(
            np.atleast_3d(Wx),
            dims=(sigdim, "frequency", "time"),
            coords={
                "frequency": freqs,
                **da["time"].coords,
                **da[sigdim].coords,
            },
            attrs=da.attrs,
        )
        .assign_attrs(scales=scales)
        .sortby("frequency")
    )


def ssq_cwt(da: xr.DataArray, sigdim: str = "channel", parallel=True, **cwt_kwargs):
    """Synchrosqueezed complex wavelet transform. Do you have pyfftw installed?
    SSQ CWT may be unstable at low frequencies (<20Hz) when data length is limited.
    Memory footprint is much higher than for plain CWT, plus does differentiation.
    """
    import ssqueezepy as ssq

    validate_2d_timeseries(da, sigdim=sigdim)
    if parallel:
        os.environ["SSQ_PARALLEL"] = "1"
    Tx, Wx, freqs, scales, *_ = ssq.ssq_cwt(da.T.values, fs=da.fs, **cwt_kwargs)
    Tx = (
        xr.DataArray(
            np.atleast_3d(Tx),  # Do you want abs?
            dims=(sigdim, "frequency", "time"),
            coords={
                "frequency": freqs,
                **da["time"].coords,
                **da[sigdim].coords,
            },
            attrs=da.attrs,
        )
        .assign_attrs(scales=scales)
        .sortby("frequency")
    )  # Syncrhosqueezed transform
    Wx = (
        xr.DataArray(
            np.atleast_3d(Wx),  # Do you want abs?
            dims=(sigdim, "frequency", "time"),
            coords={
                "frequency": freqs,
                **da["time"].coords,
                **da[sigdim].coords,
            },
            attrs=da.attrs,
        )
        .assign_attrs(scales=scales)
        .sortby("frequency")
    )  # Regular transform
    return Tx, Wx


def butter_bandpass(
    da: xr.DataArray, lowcut: float, highcut: float, order: int, plot: bool = False
) -> xr.DataArray:
    validate_2d_timeseries(da)
    res = da.copy()
    if da.chunks is None:
        res.values = npsig.filt.butter_bandpass(
            res.values.T, lowcut, highcut, res.fs, order, plot
        ).T
    else:
        res.data = dasig.butter_bandpass(
            res.data,
            lowcut,
            highcut,
            res.fs,
            order,
            time_axis=da.get_axis_num("time"),
            plot=plot,
        )
    return res.__class__(res)


def hilbert(da: xr.DataArray) -> xr.DataArray:
    """
    To get instantaneous power with dask:
    da = butter_bandpass(da, lowcut, highcut, order)
    da = hilbert(da)
    da = dask.array.square(dask.array.abs(da))
    """
    validate_2d_timeseries(da)
    res = da.copy()
    if da.chunks is None:
        res.values = npsig.hilbert(res.values)
    else:
        res.data = dasig.hilbert(res.data)
    return res.__class__(res)


def moving_transform(
    da: xr.DataArray, window: float, step: float, method: str
) -> xr.DataArray:
    validate_2d_timeseries(da)
    res = da.copy()
    if da.chunks is None:
        res.values = npsig.moving_transform(res.values, res.fs, window, step, method)
    else:
        res.data = dasig.moving_transform(res.data, res.fs, window, step, method)
    return res.__class__(res)


def validate_3d_timeseries(
    da: xr.DataArray,
    evtdim: str = "event",
    sigdim: str = "channel",
    timedim: str = "time",
    check_times: bool = False,
):
    if not da.dims == (evtdim, sigdim, timedim):
        raise AttributeError(
            f"Timeseries3D DataArray must have dimensions ({timedim}, {sigdim}, {evtdim})"
        )
    if "fs" not in da.attrs:
        raise ValueError("Timeseries2D must have sampling rate attr `fs`")
    if check_times and not np.all(np.diff(da[timedim].values) >= 0):
        raise ValueError("Timeseries2D times must be monotonically increasing.")


def demean_trialed(
    da: xr.DataArray, mean_estimation_time=slice(None, None)
) -> xr.DataArray:
    validate_3d_timeseries(da)
    baseline_means = da.sel(time=mean_estimation_time).mean(dim="time")
    return (da - baseline_means).assign_attrs(**da.attrs)


def detrend_trialed(
    da: xr.DataArray, trend_estimation_time=slice(None, None)
) -> xr.DataArray:
    """For some reason, polyfit will segfault with large numbers of trials. It's not clear if this is a memory issue, or a numerical stability issue, or what."""
    prestim_lfps = da.sel(time=trend_estimation_time)
    print("Fitting detrend polynomial...")
    p = prestim_lfps.polyfit(dim="time", deg=1)
    print("Evaluating detrend polynomial...")
    fit = xr.polyval(da["time"], p.polyfit_coefficients)
    print("Subtracting detrend polynomial...")
    return (da - fit).assign_attrs(**da.attrs)


def get_channel_indices(da: xr.DataArray, channel_ids: np.ndarray) -> np.ndarray:
    return np.argwhere(da["channel"].isin(channel_ids).values).squeeze()


def bipolar_reference(da: xr.DataArray, shift: int) -> xr.DataArray:
    """Bipolar referencing.

    Args:
        da: The DataArray to bipolar reference.
        shift: The number of channels to shift the DataArray by.
            If your channels are index ordered from deeper to more superficial, a
            positive shift will subtract the deeper channels from the more superficial
            channels. In the result, `shift` deepest channels will be dropped.

    Returns:
        The bipolar referenced DataArray.
        The coordinates along the channel dimension are retained, and correspond to the
        channel that was rereferenced. E.g. "y" is the depth of each channel A in A - B.
        New coordinates are added to the channel dimension, e.g. "ref_y" is the depth of
        each channel B in A - B.

    Examples:
        >>> lfp_bi = bipolar_reference(lfp, shift=10)

        To retain only signals for which both poles (A and B) are in the same structure:
        >>> keep = lfp_bi["acronym"] == lfp_bi["ref_acronym"]
        >>> lfp_bi = lfp_bi.isel({"channel": keep})

        To get the separation of each pole:
        >>> lfp_bi["y"] - lfp_bi["ref_y"]
    """
    validate_2d_timeseries(da)
    bi = da - da.shift({"channel": shift})
    bi = bi.isel(channel=slice(shift, None))
    coords = [x for x in bi["channel"].coords.keys() if x != "channel"]
    for coord in coords:
        bi = bi.assign_coords(
            {
                f"ref_{coord}": (
                    "channel",
                    da[coord].shift({"channel": shift}).values[shift:],
                )
            }
        )
    bi.attrs["fs"] = da.fs
    return bi
