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
    # Clean binary mask
    "clean_binary_mask": True,
    "n_channels_clean": 3,
    "n_channels_connect": 5,
    "n_samples_connect": 10,
    "n_samples_clean": None,
}


def get_mad_thresh(x, opts=None, bins=None):
    mode = bins[np.argmax(np.histogram(x, bins=bins)[0])]
    return mode - opts["mad_threshold"] * scipy.stats.median_abs_deviation(x)

def get_quantile_thresh(x, opts=None):
    quantile = opts["quantile_threshold"]
    return np.quantile(x, quantile)

def get_thresholds(da, method="mad", opts=None):
    if opts is None:
        opts = DEFAULT_OPTS
    
    assert method in ["mad", "quantile"]
    if method == "mad":
        bins = np.linspace(float(da.data.min()), float(da.data.max()), 100)
        func = partial(get_mad_thresh, opts=opts, bins=bins)
    if method == "quantile":
        func = partial(get_quantile_thresh, opts=opts)

    dat = dask.array.apply_along_axis(
        func,
        0,
        da,
    )

    return xr.DataArray(
        data=dat,
        dims=("channel"),
        coords={
            "channel": da.channel,
            "y": ("channel", da.y.data)
        },
        name="Detection threshold",
        attrs={
            "method": method,
            "opts": opts,
        }
    )


def clean_binary_mask(off_mask, n_samples_connect=None, n_samples_clean=10, n_channels_clean=3, n_channels_connect=5):

    import dask_image.ndmorph

    # Horizontal: Connect across samples
    if n_samples_connect is not None:
        struct = np.ones((n_samples_connect, 1))
        print(f"Binary close: {struct.shape}")
        off_mask.data = dask_image.ndmorph.binary_closing(off_mask.data, structure=struct, iterations=1)
    
    # Vertical: Connect across bad channels
    if n_channels_clean is not None:
        struct = np.ones((1, n_channels_clean))
        print(f"Binary close: {struct.shape}")
        off_mask.data = dask_image.ndmorph.binary_closing(off_mask.data, structure=struct, iterations=1)
    
    # # # Horizontal  Remove shorter blobs
    if n_samples_clean is not None:
        struct = np.ones((n_samples_clean, 1))
        print(f"Binary open: {struct.shape}")
        off_mask.data = dask_image.ndmorph.binary_opening(off_mask.data, structure=struct, iterations=1)
    
    # # # vertical : Remove few-channel epochs
    if n_channels_clean is not None:
        struct = np.ones((1, n_channels_clean))
        print(f"Binary open: {struct.shape}")
        off_mask.data = dask_image.ndmorph.binary_opening(off_mask.data, structure=struct, iterations=1)
    
    # # vertical : Connect distant blobs vertically
    if n_channels_connect is not None:
        struct = np.ones((1, n_channels_connect))
        print(f"Binary close: {struct.shape}")
        off_mask.data = dask_image.ndmorph.binary_closing(off_mask.data, structure=struct, iterations=1)

    return off_mask


def get_offs_df(da, lbl_ixs):
    """Generate offs dataframe.
    
    Args:
    da: xr.DataArray
        DataArray used for detection
    lbl_ixs: dict
        {<label>: (<col_indices>, <row_indices>)}
    """

    if not lbl_ixs:
        return pd.DataFrame(
            columns=[
                "area",
                "start_frame",
                "end_frame",
                "median_nframes",
                "lo_chan_idx",
                "max_chan_idx",
                "label",
                "start_time",
                "end_time",
                "duration",
                "median_duration",
                "lo",
                "hi",
                "span",
                "convexity_ratio",
            ]
        )

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
        def _get_row_area(row):
            from scipy.spatial import ConvexHull, QhullError
            try:
                return ConvexHull(np.transpose(np.array(lbl_ixs[row["label"]]))).volume
            except QhullError:
                convex_area = float("Inf")
        convex_area = df.apply(
            lambda row: _get_row_area(row),
            axis=1
        )
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
    missing_keys = set(DEFAULT_OPTS.keys()) - set(opts.keys())
    if missing_keys:
        import warnings
        warnings.warn(f"Following keys might be missing from `opts` kwarg: {missing_keys}")

    # Binary mask of below threhsold pixels
    off_mask = da.copy()
    off_mask.data = dask.array.where(da < thresholds, True, False)
    off_mask.name = "OFF mask"

    # Morphological cleaning
    if opts["clean_binary_mask"]:
        off_mask = clean_binary_mask(
            off_mask,
            n_samples_connect=opts["n_samples_connect"],
            n_samples_clean=opts["n_samples_clean"],
            n_channels_clean=opts["n_channels_clean"],
            n_channels_connect=opts["n_channels_connect"],
        )

    # Labels for each contiguous blob
    lbl_da = da.copy()
    lbl_img, _ = dask_image.ndmeasure.label(off_mask)
    lbl_da.data = lbl_img
    lbl_da.name = "OFF label"
    del off_mask
    print("Done labelling")

    # {label: (col_indices, row_indices)}
    lbl_ixs = scipy.ndimage.value_indices(np.array(lbl_da.data), ignore_value=0) # {lbl: (row/time_indices, col/chan_indices)}

    offs_raw = get_offs_df(da, lbl_ixs)

    return offs_raw, lbl_da, lbl_ixs