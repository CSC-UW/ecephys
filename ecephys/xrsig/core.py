import kcsd
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ecephys.emg_from_lfp as lfemg
from ecephys.signal import timefrequency as tfr
from ecephys.signal import filt

from .. import utils as ece_utils
from . import utils


class DataArrayWrapper:
    """Passes calls not otherwise overriden to an underlying DataArray object.
    This allows one to effectively extend the DataArray type without subclassing it directly,
    using the principle of composition over inheritance."""

    def __init__(self, obj):
        if isinstance(obj, DataArrayWrapper):
            da = obj._da
        elif isinstance(obj, xr.DataArray):
            da = obj
        else:
            raise ValueError(
                f"Cannot create {self.__class__.__name__} from object of type {type(obj)}."
            )
        self._da = da

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self._da, attr)

    def __getitem__(self, item):
        return self._da[item]

    def __setitem__(self, item, data):
        self._da[item] = data

    def __repr__(self):
        return repr(self._da)

    def __len__(self):
        return len(self._da)

    def __getstate__(self):
        return vars(self)

    def __setstate__(self, state):
        vars(self).update(state)


class Timeseries(DataArrayWrapper):
    """
    Required dimension: "time"
    Attributes:
        fs: Sampling rate

    'time' is seconds from a reference period (usually the start of the experiment).
    """

    def __init__(self, da):
        super().__init__(da)
        self._validate()

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self._da, attr)

    def __getitem__(self, item):
        return self._da[item]

    def __setitem__(self, item, data):
        self._da[item] = data

    def __repr__(self):
        return repr(self._da)

    def __len__(self):
        return len(self._da)

    def _validate(self):
        expected_attrs = {"fs"}
        if not "time" in self.dims:
            raise AttributeError(
                f"{self.__class__.__name__} must have dimension `time`."
            )
        for attr in expected_attrs:
            if not attr in self.attrs:
                raise AttributeError(
                    f"{self.__class__.__name__} must have attribute {attr}."
                )

    def butter_bandpass(self, lowcut, highcut, order, plot=True):
        _da = self._da.copy()
        _da.values = filt.butter_bandpass(
            _da.values.T, lowcut, highcut, _da.fs, order, plot
        ).T
        if plot:
            plotDur = 1  # seconds
            plotT = slice(self.time.min(), self.time.min() + plotDur)
            self.sel(time=plotT).plot.line(
                x="time", add_legend=False, figsize=(36, 5), alpha=0.1
            )
            plt.title(f"First {plotDur}s of original signal")
            _da.sel(time=plotT).plot.line(
                x="time", add_legend=False, figsize=(36, 5), alpha=0.1
            )
            plt.title(f"First {plotDur}s of filtered signal")
        return self.__class__(_da)


class TimeseriesND(Timeseries):
    """
    Dimensions: (..., 'time', ..., <signal>).
    Attributes:
        fs: Sampling rate

    'time' is seconds from a reference period (usually the start of the experiment).
    """

    def __init__(self, da):
        super().__init__(da)
        self._validate()
        self._lastdim = self.dims[-1]

    def _validate(self):
        if not len(self.dims) > 1:
            raise AttributeError(f"{self.__class__.__name__} must have >1 dimension.")
        if not "time" in self.dims[:-1]:
            raise AttributeError(
                f"{self.__class__.__name__} must include dimension `time` in the first N-1 dimensions."
            )


class Timeseries2D(TimeseriesND):
    """
    Dimensions: ('time', <other>).
    Attributes:
        fs: Sampling rate

    'time' is seconds from a reference period (usually the start of the experiment).
    """

    def __init__(self, da):
        super().__init__(da)
        self._validate()

    def _validate(self):
        if not len(self.dims) == 2:
            raise AttributeError(f"{self.__class__.__name__} must have 2 dimensions.")

    def spectrograms(self, **kwargs):
        """Get spectrograms for each channel using Welch's method.
        Extra keyword arguments are passed down to scipy.signal.welch.
        """
        # If a window length has not been specified, use 4s, to be sure we can capture 0.5Hz.
        if "nperseg" not in kwargs:
            fft_window_length = 4  # seconds
            kwargs.update({"nperseg": int(fft_window_length * self.fs)})

        freqs, spg_time, spg_vals = tfr.parallel_spectrogram_welch(
            self.transpose("time", self._lastdim).values, self.fs, **kwargs
        )
        time = (
            self.time.values.min() + spg_time
        )  # Account for the fact that t[0] might not == 0.

        spgs = xr.DataArray(
            np.atleast_3d(spg_vals),
            dims=("frequency", "time", self._lastdim),
            coords={
                "frequency": freqs,
                "time": time,
                **self[
                    self._lastdim
                ].coords,  # Copy any channel coord data, like x, y pos.
            },
        )
        spgs = spgs.assign_attrs(self.attrs)

        if "units" in spgs.attrs:
            spgs = spgs.assign_attrs({"units": f"{spgs.units}^2/Hz"})

        return spgs


class LFPs(Timeseries2D):
    """Provides methods for operating on LFPs.

    Dimensions: ('time', <signal>).
    Attributes:
        fs: Sampling rate

    'time' is seconds from a reference period (usually the start of the experiment).
    """

    def kCSD(self, drop=[], doLCurve=False, **kCsdKwargs):
        """Compute 1D kernel current source density.
        If signal units are in uV, then CSD units are in nA/mm.

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
        umPerMm = 1000
        if not "y" in self.coords:
            raise AttributeError(
                f"kCSD method requires a y coordinate on the last dimension."
            )

        def get_pitch(y):
            """Get the vertical spacing between electrode sites, in microns"""
            vals = np.diff(np.unique(y))
            assert ece_utils.all_equal(vals), "Electrode pitch is not uniform."
            return np.absolute(vals[0])

        # Make sure we get CSD estimates at electrode locations, rather than say, in between electrodes.
        pitchMm = (
            get_pitch(self["y"].values) / umPerMm
        )  # Convert um to mm for KCSD package.
        gdx = kCsdKwargs.get("gdx", None)
        if (gdx is not None) and (gdx != pitchMm):
            raise ValueError("Requested gdx does not match electrode pitch.")
        else:
            kCsdKwargs.update(gdx=pitchMm)

        # Drop bad signals and redundant signals
        sigs = self.drop_sel({self._lastdim: drop}, errors="ignore")
        sigs = sigs.groupby("y").first()

        # Convert um to mm for KCSD package.
        elePosMm = sigs["y"].values / umPerMm

        # Compute kCSD
        sigs = sigs.transpose("y", "time")
        k = kcsd.KCSD1D(elePosMm.reshape(-1, 1), sigs.values, **kCsdKwargs)
        if doLCurve:
            print("Performing L-Curve parameter estimation...")
            k.L_curve()

        # Check and format result
        assert (k.estm_x.size == self["y"].size) and np.allclose(
            k.estm_x * umPerMm, self["y"].values
        ), "CSD returned estimates that do not match original signal positions exactly."
        csd = self._da.copy()
        csd.values = k.values("CSD").T
        return csd.assign_attrs(kcsd=k)

    def synthetic_emg(self, **emgKwargs):
        """Estimate the EMG from LFP signals, using the `emg_from_lfp` subpackage.

        Parameters used for the computation are stored as attributes on the returned DataArray.

        Parameters:
        -----------
        **emg_kwargs:
            Keyword arguments passed to `emg_from_lfp.compute_emg()`

        Returns:
        --------
        DataArray:
            EMG with time dimension and timedelta, datetime coords.
        """
        values = lfemg.compute_emg(
            self.transpose(self._lastdim, "time").values, self.fs, **emgKwargs
        ).flatten()
        time = np.linspace(self.time.min(), self.time.max(), values.size)

        emg = xr.DataArray(
            values,
            dims="time",
            coords={
                "time": time,
            },
            attrs={"long_name": "emg_from_lfp", "units": "corr"},
        )
        for key in emgKwargs:
            emg.attrs[key] = emgKwargs[key]

        return emg


class LaminarScalars(DataArrayWrapper):
    """Any channelwise scalar data.

    Dimensions: ('channel',)."""

    def __init__(self, da):
        super().__init__(da)
        self._validate()

    def _validate(self):
        expected = ("channel",)
        if not self.dims == expected:
            raise AttributeError(
                f"{self.__class__.__name__} must have dimensions {expected}."
            )

    def plot_laminar(self, ax=None, figsize=(10, 15), **kwargs):
        """Plot a depth profile of values.
        Requires 'y' coordinate on 'channel' dimension."""
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        da = (
            self.sortby("y") if "y" in self.coords else self._da
        )  # Sort by depth if possible
        da.plot.line(y="y", ax=ax, **kwargs)
        ytick_labels = [
            x for x in zip(np.round(da["y"].values, 2), da["channel"].values)
        ]
        ax.set_yticks(da.y.values)
        ax.set_yticklabels(ytick_labels)

        if "structure" in self.coords:
            boundaries = da.isel(channel=utils.get_boundary_ilocs(da, "structure"))
            ax.set_yticks(boundaries.channel)
            ax.set_yticklabels(boundaries.structure.values)
            for ch in boundaries.channel:
                ax.axhline(ch, alpha=0.5, color="dimgrey", linestyle="--")
