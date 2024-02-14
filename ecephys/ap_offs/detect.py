from ecephys.wne.sglx import utils as sglx_utils

from functools import partial
import pandas as pd
import wisc_ecephys_tools as wet
import numpy as np
from offproj import core
import dask.array
import dask_image
import dask_image.ndfilters
import xarray as xr
import zarr
from ecephys import utils
import dask.array
import scipy.stats
import xarray as xr

DEFAULT_OPTS = {
    "median_filter_N_chans": 5,
    "median_filter_N_samples": 20,
    "std_threshold": 0.085,
    "std_threshold_ratio": 0.5,
    "mad_threshold": 1.5,
    "min_off_median_duration": 0.01,
    "min_off_span": 100,
    # "min_off_convexity_ratio": 0.75,
    "min_off_convexity_ratio": 0.0,
}


def get_thresh(x, opts):
    return np.median(x) - opts["mad_threshold"] * scipy.stats.median_abs_deviation(x)

def get_thresholds(da, opts):
    if da.shape[1]:
        dat = dask.array.apply_along_axis(
            partial(get_thresh, opts=opts),
            0,
            da,
        )
    else:
        dat = []
    return xr.DataArray(
        data=dat,
        dims=("channel"),
        coords={
            "channel": da.channel
        },
        name="Detection threshold",
        attrs={"mad_threshold": opts["mad_threshold"]}
    )

def get_offs_df(da, lbl_ixs):
    """Generate offs dataframe.
    
    Args:
    da: xr.DataArray
        DataArray used for detection
    lbl_ixs: dict
        {<label>: (<col_indices>, <row_indices>)}
    """

    def _get_median_nframes(col_indices, row_indices):
        tmp = pd.DataFrame({'row': row_indices, 'col': col_indices})
        return tmp.groupby('row').count()['col'].median()

    _lbls = np.sort(list(lbl_ixs.keys()))
    start_frames = pd.DataFrame([lbl_ixs[lbl][0].min() for lbl in _lbls], columns=['start_frame'], index=_lbls)
    end_frames = pd.DataFrame([lbl_ixs[lbl][0].max() for lbl in _lbls], columns=['end_frame'], index=_lbls)
    median_nframes = pd.DataFrame([_get_median_nframes(*lbl_ixs[lbl]) for lbl in _lbls], columns=['median_nframes'], index=_lbls)
    lo_chan_ix = pd.DataFrame([lbl_ixs[lbl][1].min() for lbl in _lbls], columns=['lo_chan_idx'], index=_lbls)
    max_chan_idx = pd.DataFrame([lbl_ixs[lbl][1].max() for lbl in _lbls], columns=['max_chan_idx'], index=_lbls)

    # Can't compute area / convexity ratio this way if channels are not evenly spaced
    if len(set(np.diff(da.y.values))) > 1:
        raise NotImplementedError("Require evenly spaced channels")
    areas = pd.DataFrame([lbl_ixs[lbl][0].size for lbl in _lbls], columns=['area'], index=_lbls)

    df = pd.concat([areas, start_frames, end_frames, median_nframes, lo_chan_ix, max_chan_idx], axis=1).dropna()
    df['label'] = df.index

    y = da.y.values
    t = da.time.values

    df['start_time'] = t[df['start_frame'].values]
    df['end_time'] = t[df['end_frame'].values]
    df['duration'] = df['end_time'] - df['start_time']
    df['median_duration'] = df['median_nframes'] / da.attrs["fs"]
    df['lo'] = y[df['lo_chan_idx'].values]
    df['hi'] = y[df['max_chan_idx'].values]
    df['span'] = df['hi'] - df['lo']

    def _assign_convexity_metrics(df, lbl_ixs):
        from scipy.spatial import ConvexHull, QhullError
        try:
            convex_area = df.apply(
                lambda row: ConvexHull(np.transpose(np.array(lbl_ixs[row["label"]]))).volume,
                axis=1
            )
        except QhullError:
            convex_area = float("Inf")
        df["convexity_ratio"] = df["area"] / convex_area
        return df

    df = _assign_convexity_metrics(df, lbl_ixs)

    return df.astype({
        'start_frame': np.dtype('int64'),
        'end_frame': np.dtype('int64'),
        'median_nframes': np.dtype('int64'), 
        'lo_chan_idx': np.dtype('int64'),
        'max_chan_idx': np.dtype('int64'),
        'label': np.dtype('int64')
    })

def detect_ap_offs(da, thresholds, opts=None):

    if opts is None:
        opts = DEFAULT_OPTS
    assert set(DEFAULT_OPTS.keys()) <= set(opts.keys())

    # Binary mask of below threhsold pixels
    off_mask = da.copy()
    off_mask.data = dask.array.where(da < thresholds, True, False)

    # Labels for each contiguous blob
    lbl_da = da.copy()
    lbl_img, _ = dask_image.ndmeasure.label(off_mask)
    lbl_da.data = lbl_img
    del off_mask
    print("Done labelling")

    # {label: (col_indices, row_indices)}
    lbl_ixs = scipy.ndimage.value_indices(np.array(lbl_da.data), ignore_value=0) # {lbl: (row/time_indices, col/chan_indices)}

    offs_raw = get_offs_df(da, lbl_ixs, opts)
    offs_clean = offs_raw[
        (offs_raw["median_duration"] >= opts["min_off_median_duration"])
        & (offs_raw["span"] >= opts["min_off_span"])
        & (offs_raw["convexity_ratio"] >= opts["min_off_convexity_ratio"])
    ]

    return offs_raw, offs_clean, lbl_ixs