import xarray as xr
import pandas as pd
import skimage.segmentation
import numpy as np


def _get_boundaries(mask_da):
    boundaries = mask_da.copy()
    boundaries.data = skimage.segmentation.find_boundaries(mask_da, mode="inner")
    return boundaries

def _get_mask_da(xs, ys, da):
    """Binary mask from set of coordinates."""

    xmin, xmax = max(0, xs.min() - 2), xs.max() + 2 # Margin to find borders
    mask_times = da.time.isel(time=slice(xmin, xmax)).data

    mask_chans = da.channel.data
    mask_chans_y = da.y.data
    
    # Mask just around coordinates of interest
    mask_da = xr.DataArray(
        np.zeros((len(mask_times), len(mask_chans))),
        dims=("time", "channel"),
        coords={
            "time": mask_times,
            "channel": mask_chans,
            "y": ("channel", mask_chans_y),
        }
    )
    
    # Fill mask
    for x, y in zip(xs - xmin, ys):
        mask_da[x, y] = True

    return mask_da

def add_ap_off_overlay(da: xr.DataArray, lbl_ixs: dict, offs_df: pd.DataFrame, ax, xlim, **plot_kwargs):

    offs_sel = offs_df[
        (offs_df.start_time.between(*xlim))
        & (offs_df.end_time.between(*xlim))
    ]

    for off_row in offs_sel.itertuples():

        xs, ys = lbl_ixs[off_row.label]
        mask_da = _get_mask_da(xs, ys, da)

        mask_da.where(mask_da).plot.imshow(
            y="y", 
            x="time", 
            ax=ax,
            # Hack for full yellow
            # cmap="winter",
            cmap="summer",
            vmin=0,
            vmax=1,
            add_colorbar=False,
            **plot_kwargs,
        )


def add_mean_ap_off_overlay(offs_df: pd.DataFrame, ax, xlim: tuple[float], ax_ylim: tuple[float], **plot_kwargs):

    offs_sel = offs_df[
        (offs_df.start_time.between(*xlim))
        & (offs_df.end_time.between(*xlim))
    ]

    def data_y_to_ax_y(data_y):
        ymin, ymax = ax_ylim
        return (data_y-ymin)/(ymax - ymin)

    for off_row in offs_sel.itertuples():

        ax.axvspan(
            off_row.start_time,
            off_row.end_time,
            data_y_to_ax_y(off_row.lo),
            data_y_to_ax_y(off_row.hi),
            facecolor="yellow",
            **plot_kwargs
        )