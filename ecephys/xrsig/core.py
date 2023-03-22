import os
import kcsd
import logging
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import ecephys.emg_from_lfp as lfemg
import ssqueezepy as ssq
from scipy import signal
from neurodsp import voltage, fourier
import neuropixel
from ecephys.signal import timefrequency as tfr
from ecephys.signal import filt
from ecephys import plot as eplt

from .. import utils as ece_utils
from . import utils

logger = logging.getLogger("xrsig")


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


class Laminar(DataArrayWrapper):
    def __init__(self, da):
        super().__init__(da)
        self._validate()

    def _validate(self):
        if not "channel" in self.dims:
            raise AttributeError(
                f"{self.__class__.__name__} must include a channel dimension."
            )
        if not "y" in self["channel"].coords:
            raise AttributeError(
                f"{self.__class__.__name__} must have y coordinate on channel dimension."
            )

    def get_pitch(self):
        """Get the vertical spacing between electrode sites, in microns"""
        vals = np.diff(np.unique(self["y"].values))
        assert ece_utils.all_equal(
            vals
        ), f"Electrode pitch is not uniform. Pitches:\n {vals}"
        return np.absolute(vals[0])


class LaminarScalars(Laminar):
    """Any channelwise scalar data."""

    def __init__(self, da):
        super().__init__(da)
        self._validate()

    def _validate(self):
        if not len(self.dims) == 1:
            raise AttributeError(f"{self.__class__.__name__} must be 1D.")

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


class LaminarVectors(Laminar):
    """Any channelwise vector data.

    Dimensions: ('channel', <other>), or (<other>, 'channel').
    """

    def __init__(self, da):
        super().__init__(da)
        self._validate()
        self._otherdim = (set(self.dims) - {"channel"}).pop()

    def _validate(self):
        if not len(self.dims) == 2:
            raise AttributeError(f"{self.__class__.__name__} must be 2D.")

    def plt(
        self,
        db=False,
        xscale="log",
        xticks=1,
        aspect=(8.5 / 11),
        figsize="auto",
        ax=None,
        cmap="cividis",
        **kwargs,
    ):
        y = self.y.values
        x = self[self._otherdim].values

        if figsize == "auto":
            height = (y.size * 0.18) / xticks
            figsize = (aspect * height, height)
        ax = eplt.check_ax(ax, figsize)
        z = self.transpose("channel", self._otherdim).values
        if db:
            z = np.log10(z)
        ax.pcolormesh(x, y, z, cmap=cmap, **kwargs)
        ax.set_xscale(xscale)
        ax.set_xlabel(self._otherdim)
        ax.set_ylabel("Position [um]")
        ytick_labels = [
            y for y in zip(np.round(y[::xticks], 2), self["channel"].values[::xticks])
        ]
        ax.set_yticks(y[::xticks])
        ax.set_yticklabels(ytick_labels)
        if xscale == "log":
            ax.set_xlim(np.min(x[x > 0]), np.max(x))
        return ax


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


class Timeseries2D(Timeseries):
    """
    Dimensions: ('time', <other>).
    Attributes:
        fs: Sampling rate

    'time' is seconds from a reference period (usually the start of the experiment).
    """

    def __init__(self, da):
        super().__init__(da)
        self._validate()
        self._sigdim = self.dims[-1]

    def _validate(self):
        if not len(self.dims) == 2:
            raise AttributeError(f"{self.__class__.__name__} must have 2 dimensions.")
        if not self.dims[0] == "time":
            raise AttributeError(
                f"{self.__class__.__name__} must have time as the first dimension."
            )

    def butter_bandpass(self, lowcut, highcut, order, plot=True):
        da = self._da.copy()
        da.values = filt.butter_bandpass(
            da.values.T, lowcut, highcut, da.fs, order, plot
        ).T
        if plot:
            plotDur = 1  # seconds
            plotT = slice(self["time"].min(), self["time"].min() + plotDur)
            self.sel(time=plotT).plot.line(
                x="time", add_legend=False, figsize=(36, 5), alpha=0.1
            )
            plt.title(f"First {plotDur}s of original signal")
            da.sel(time=plotT).plot.line(
                x="time", add_legend=False, figsize=(36, 5), alpha=0.1
            )
            plt.title(f"First {plotDur}s of filtered signal")
        return self.__class__(da)

    def scipy_stft(self, **kwargs):
        """Get spectrograms for each channel using Welch's method.
        Extra keyword arguments are passed down to scipy.signal.welch.
        This is hella slow. Use ssq/pyfftw!
        """
        # If a window length has not been specified, use 4s, to be sure we can capture 0.5Hz.
        if "nperseg" not in kwargs:
            fft_window_length = 4  # seconds
            kwargs.update({"nperseg": int(fft_window_length * self.fs)})

        freqs, spg_time, spg_vals = tfr.parallel_spectrogram_welch(
            self.values, self.fs, **kwargs
        )
        time = (
            self["time"].values.min() + spg_time
        )  # Account for the fact that t[0] might not == 0.

        spgs = xr.DataArray(
            np.atleast_3d(spg_vals),
            dims=("frequency", "time", self._sigdim),
            coords={
                "frequency": freqs,
                "time": time,
                **self[
                    self._sigdim
                ].coords,  # Copy any channel coord data, like x, y pos.
            },
            attrs=self.attrs,
        )

        if "units" in spgs.attrs:
            spgs = spgs.assign_attrs({"units": f"{spgs.units}^2/Hz"})

        return spgs

    def ssq_stft(self, **kwargs):
        """
        See `help(ssqueezepy.stft)`.

        Useful kwargs:
        --------------
        window:
            If unspecified, use 'hamming'.
        nfft: int
            Number of points in the fft. If not specified, defaults to the next_power_of_two(int(self.fs * 4))
        hop_len:
            STFT stride, or number of samples to skip/hop over between subsequent
            windowings. Relates to 'overlap' as `overlap = n_fft - hop_len`
        dtype: str['float32', 'float64'] / None
            Compute precision; use 'float32` for speed & memory at expense of
            accuracy (negligible for most purposes).
            Defaults to 'float32'
        """
        # if n_fft is None:
        #    n_fft = ssq.p2up(int(self.fs * 4))[0]
        # if not "window" in kwargs:
        #    kwargs["window"] = "hamming"
        if not "n_fft" in kwargs:
            kwargs["n_fft"] = ssq.p2up(int(self.fs))[0]
        if not "hop_len" in kwargs:
            kwargs["hop_len"] = kwargs["n_fft"] // 4
        if not "dtype" in kwargs:
            kwargs["dtype"] = "float32"
        if ("fs" in kwargs) and (kwargs["fs"] != self.fs):
            raise ValueError(f"Was passed fs={kwargs['fs']}, but expected {self.fs}.")

        logger.debug(f"Calling SSQ STFT with the following kwargs: {kwargs}")
        _, Wx, freqs, _, *_ = ssq.ssq_stft(self.values.T, fs=self.fs, **kwargs)

        ns = self.time.size
        time = (
            np.linspace(0, ns, Wx.shape[-1], endpoint=False) + (kwargs["hop_len"] / 2)
        ) / float(self.fs) + self.time.values.min()

        return xr.DataArray(
            np.atleast_3d(np.abs(Wx)),
            dims=(self._sigdim, "frequency", "time"),
            coords={
                "frequency": freqs,
                "time": time,
                **self[self._sigdim].coords,
            },
            attrs=self.attrs,
        )

    def make_trialed(self, tTrials, tPre, tPost, tol=None):
        method = "nearest" if tol is not None else None
        # Get precise stamples of trial zero, start, and end times
        sTrial = (
            np.round((self.time - self.time.min()) * self.fs, 0).astype("int")
        ).sel(time=tTrials, tolerance=tol, method=method)
        sPre = sTrial - int(tPre * self.fs)
        sPost = sTrial + int(tPost * self.fs)

        trials = pd.DataFrame({"sTrial": sTrial, "sPre": sPre, "sPost": sPost}).astype(
            "int"
        )
        trials = trials[
            ~(trials < 0).any(axis=1)
        ]  # Remove events whose window starts before the data
        trials = trials[
            ~(trials >= self.time.size).any(axis=1)
        ]  # Remove events whose window ends after the data

        # Reshape the LFP, adding an third event dimension
        dat3d = np.dstack(
            [
                self.isel(time=slice(trl.sPre, trl.sPost)).values
                for trl in trials.itertuples()
            ]
        )

        # Get precise timestamps along which events are aligned
        (nTrialSamples, nChans, nTrials) = dat3d.shape
        winTime = np.linspace(-tPre, tPost, nTrialSamples)

        return xr.DataArray(
            data=dat3d,
            dims=(*self.dims, "event"),
            coords={
                "time": winTime,
                "event": tTrials,
                **self[self._sigdim].coords,
            },
            attrs=self.attrs,
        )

    def ssq_cwt(self, plot_filterbank=False, **kwargs):
        """Synchrosqueezed complex wavelet transform. Do you have pyfftw installed?"""
        os.environ["SSQ_PARALLEL"] = "1"
        logger.warn(
            "SSQ CWT may be unstable at low frequencies (<20Hz) when data length is limited."
        )
        if self.time.size > 1e6:
            logger.warn(
                f"You requested a synchrosqueezed complex wavelet transform with > {1e6} points. Maybe you want a STFT..."
            )

        Tx, Wx, freqs, scales, *_ = ssq.ssq_cwt(self.T.values, fs=self.fs, **kwargs)
        if plot_filterbank and kwargs.get("wavelet") is not None:
            plt.figure(figsize=(22, 6))
            ssq.Wavelet(kwargs.get("wavelet"), ssq.p2up(self.time.size)[0]).viz(
                "filterbank", scales=scales
            )

        cwtSq = xr.DataArray(
            np.atleast_3d(np.abs(Tx)),
            dims=(self._sigdim, "frequency", "time"),
            coords={
                "frequency": freqs,
                **self["time"].coords,
                **self[self._sigdim].coords,
            },
            attrs=self.attrs,
        )
        cwt = xr.DataArray(
            np.atleast_3d(np.abs(Wx)),
            dims=(self._sigdim, "frequency", "time"),
            coords={
                "frequency": freqs,
                **self["time"].coords,
                **self[self._sigdim].coords,
            },
            attrs=self.attrs,
        )
        return cwtSq, cwt

    def scipy_cwt(self, freqs=np.geomspace(0.5, 300, 300)):
        logger.warn("Scipy CWT is sloooooow. Try SSQ CWT next time.")
        nyquist = self.fs / 2
        assert np.all(freqs <= nyquist)
        w = 6
        widths = w * self.fs / (2 * freqs * np.pi)
        cwt = np.apply_along_axis(
            lambda x: signal.cwt(x, signal.morlet2, widths, w=w), 0, self.values
        )

        # if normalize:
        #    cwt = cwt / (1 / freqs)[:, None]

        return xr.DataArray(
            np.atleast_3d(np.abs(cwt)),
            dims=("frequency", "time", self._sigdim),
            coords={
                "frequency": freqs,
                **self["time"].coords,
                **self[self._sigdim].coords,
            },
            attrs=self.attrs,
        )

    def decimate(self, q=2):
        dat = signal.decimate(self.values, q=q, ftype="fir", axis=0)
        rs = xr.DataArray(
            dat,
            dims=self.dims,
            coords={
                **self["time"][::q].coords,
                **self[self._sigdim].coords,
            },
            attrs=self.attrs,
        )
        rs.attrs["fs"] = self.fs / q
        return self.__class__(rs)


class LFPs(Timeseries2D, Laminar):
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

        # Make sure we get CSD estimates at electrode locations, rather than say, in between electrodes.
        pitchMm = self.get_pitch() / umPerMm  # Convert um to mm for KCSD package.
        gdx = kCsdKwargs.get("gdx", None)
        if (gdx is not None) and (gdx != pitchMm):
            raise ValueError("Requested gdx does not match electrode pitch.")
        else:
            kCsdKwargs.update(gdx=pitchMm)

        # Drop bad signals and redundant signals
        sigs = self.drop_sel({self._sigdim: drop}, errors="ignore")
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
        return Timeseries2D(csd.assign_attrs(kcsd=k))

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
            self.transpose(self._sigdim, "time").values, self.fs, **emgKwargs
        ).flatten()
        time = np.linspace(self.time.min(), self.time.max(), values.size)

        emg = xr.DataArray(
            values,
            dims="time",
            coords={
                "time": time,
            },
            attrs={"units": "corr"},
        )
        for key in emgKwargs:
            emg.attrs[key] = emgKwargs[key]

        return emg

    def interpolate(self, signals, inplace=True):
        """signals: list of IDs of signals to interpolate"""
        do_interp = np.isin(self[self._sigdim], signals)
        if not do_interp.any():
            logger.debug(
                "None of the requested signals are present in the data. Doing nothing."
            )
            return
        else:
            logger.debug(
                f"Data contains bad signals: {self[self._sigdim].values[do_interp].squeeze()}. Interpolating..."
            )
        if "x" in self[self._sigdim].coords:
            x = self["x"].values
        else:
            print(
                "Data do not contain x coordinates on channel dimension. Assuming all electrodes are colinear."
            )
            x = np.zeros_like(self["y"])

        lf = self._da if inplace else self.copy()
        lf.values = voltage.interpolate_bad_channels(
            lf.values.T, do_interp.astype("int"), x=x, y=self["y"].values
        ).T
        return self.__class__(lf)


class NPX1LFPs(LFPs):
    def __init__(self, da):
        super().__init__(da)
        self._validate()

    def _validate(self):
        if not self.dims == ("time", "channel"):
            raise AttributeError(
                f"{self.__class__.__name__} must have dimensions (time, channel)"
            )

    def dephase(self, q=1, inplace=True):
        hdr = neuropixel.trace_header(version=1)
        lf = self._da if inplace else self.copy()
        shifts = hdr["sample_shift"][lf["channel"].values] / q
        lf.values = fourier.fshift(lf.values, shifts, axis=0)
        return self.__class__(lf)
