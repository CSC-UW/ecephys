import kcsd
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ecephys.emg_from_lfp as lfemg
from ecephys.signal import timefrequency as tfr

from . import dsp, utils

class DataArrayWrapper:
    """Passes calls not otherwise overriden to an underlying DataArray object.
    This allows one to effectively extend the DataArray type without subclassing it directly,
    using the principle of composition over inheritance."""
    def __init__(self, da):
        assert isinstance(da, xr.DataArray)
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

class LocalFieldPotentials(DataArrayWrapper):
    """Provides methods for operating on LFPs.

    Dimensions: ('time', 'channel').
    Attributes:
        fs: Sampling rate

    'time' is seconds from a reference period (usually the start of the experiment). """
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
        expected_dims = ('time', 'channel')
        expected_attrs = {'fs'}
        if not self.dims == expected_dims:
            raise AttributeError(
                f"{self.__class__.__name__} must have dimensions {expected_dims}."
            )
        for attr in expected_attrs:
            if not attr in self.attrs:
                raise AttributeError(
                    f"{self.__class__.__name__} must have attribute {attr}."
                )

    def spectrograms(self, **kwargs):
        """Get spectrograms for each channel using Welch's method.
        Extra keyword arguments are passed down to scipy.signal.welch.
        """
        # TODO: Checkut xrft package!
        # If a window length has not been specified, use 4s, to be sure we can capture 0.5Hz.
        if 'nperseg' not in kwargs:
            fft_window_length = 4 # seconds
            kwargs.update({'nperseg': int(fft_window_length * self.fs)})

        freqs, spg_time, spg = tfr.parallel_spectrogram_welch(
            self.transpose("time", "channel").values, self.fs, **kwargs
        )
        time = self.time.values.min() + spg_time # Account for the fact that t[0] might not == 0.

        spgda = xr.DataArray(
            np.atleast_3d(spg),
            dims=("frequency", "time", "channel"),
            coords={
                "frequency": freqs,
                "time": time,
                **self.channel.coords, # Copy any channel coord data, like x, y pos.
            },
        )
        spgda = spgda.assign_attrs(self.attrs)

        if "timedelta" in self.coords:
            timedelta = self.timedelta.values.min() + pd.to_timedelta(spg_time, "s")
            spgda = spgda.assign_coords({"timedelta": ("time", timedelta)})
        if "datetime" in self.coords:
            datetime = self.datetime.values.min() + pd.to_timedelta(spg_time, "s")
            spgda = spgda.assign_coords({"datetime": ("time", datetime)})

        if "units" in spgda.attrs:
            spgda = spgda.assign_attrs({"units": f"{spgda.units}^2/Hz"})

        return ChannelSpectrograms(spgda)


    def kCSD(self, drop_chans=[], do_lcurve=False, **kcsd_kwargs):
        """Compute 1D kernel current source density.
        If signal units are in uV, then CSD units are in nA/mm.
        Must have a y coordinate on the channel dimension, with units in um.

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
        um_per_mm = 1000  # Convert um to mm for KCSD package.
        chans_before_drop = (
            self.channel.values
        )  # Save for later, because KCSD will still give us estimates at dropped channels.
        ele_pos_before_drop = self.y.values / um_per_mm
        lfda = self.drop_sel(channel=drop_chans, errors="ignore")
        lfda = lfda.groupby("y").first()
        lfda = lfda.transpose("y", "time")
        ele_pos = lfda.y.values / um_per_mm

        k = kcsd.KCSD1D(ele_pos.reshape(-1, 1), lfda.values, **kcsd_kwargs)

        if do_lcurve:
            print("Performing L-curve parameter estimation...")
            k.L_curve()

        csd = xr.DataArray(
            k.values("CSD"),
            dims=("pos", "time"),
            coords={"pos": k.estm_x, "time": lfda.time.values},
        )
        if "timedelta" in lfda.coords:
            csd = csd.assign_coords({"timedelta": ("time", lfda.timedelta.values)})
        if "datetime" in lfda.coords:
            csd = csd.assign_coords({"datetime": ("time", lfda.datetime.values)})

        if (k.estm_x.size == ele_pos_before_drop.size) and np.allclose(
            k.estm_x, ele_pos_before_drop
        ):
            csd = csd.assign_coords({"channel": ("pos", chans_before_drop)})

        csd = csd.assign_attrs(kcsd=k, fs=lfda.fs)

        return KernelCurrentSourceDensity(csd)

    def synthetic_emg(self, **emg_kwargs):
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
            self.transpose("channel", "time").values, self.fs, **emg_kwargs
        ).flatten()
        time = np.linspace(self.time.min(), self.time.max(), values.size)
        timedelta = pd.to_timedelta(time, "s")
        datetime = self.datetime.values.min() + timedelta

        emg = xr.DataArray(
            values,
            dims="time",
            coords={
                "time": time,
                "timedelta": ("time", timedelta),
                "datetime": ("time", datetime),
            },
            attrs={"long_name": "emg_from_lfp", "units": "zero-lag correlation"},
        )
        for key in emg_kwargs:
            emg.attrs[key] = emg_kwargs[key]

        return emg

class ChannelSpectrograms(DataArrayWrapper):
    """Channelwise spectrograms.
    Dimensions: ('frequency', 'time', 'channel').
    'time' is seconds from a reference period (usually the start of the experiment). """
    def __init__(self, da):
        super().__init__(da)
        self._validate()

    def _validate(self):
        expected = ('frequency', 'time', 'channel')
        if not self.dims == expected:
            raise AttributeError(
                f"{self.__class__.__name__} must have dimensions {expected}."
            )

    def spectra(self):
        spectra = self.median(dim='time')
        return ChannelSpectra(spectra)

    def bandpowers(self, f_range):
        """Get band-limited power from a spectrogram.

        Parameters
        ----------
        f_range: (float, float)
            Frequency range to restrict to, as [f_low, f_high].

        Returns:
        --------
        bandpower: xr.DataArray (time, channel)
            Sum of the power in `f_range` at each point in time.
        """
        bandpower = self.sel(frequency=slice(*f_range)).sum(dim="frequency")
        bandpower.attrs["f_range"] = f_range
        return bandpower

    def multiband_powers(self, bands):
        """Get multiple bandpower series in a single Dataset object.

        Examples
        --------
            self.multiband_powers({'delta': (0.5, 4), 'theta': (5, 10)})
        """
        return xr.Dataset(
            {band_name: self.get_bandpower(f_range) for band_name, f_range in bands.items()}
        )

class ChannelSpectra(DataArrayWrapper):
    """Channelwise power spectral densities.
    Dimensions: ('frequency', 'channel')."""
    def __init__(self, da):
        super().__init__(da)
        self._validate()

    def _validate(self):
        expected = ('frequency', 'channel')
        if not self.dims == expected:
            raise AttributeError(
                f"{self.__class__.__name__} must have dimensions {expected}."
            )

    def bandpower(self, slice):
        pow = self.sel(frequency=slice).sum(dim='frequency')
        return ChannelScalars(pow)

class ChannelScalars(DataArrayWrapper):
    """Any channelwise scalar data.

    Dimensions: ('channel',)."""
    def __init__(self, da):
        super().__init__(da)
        self._validate()

    def _validate(self):
        expected = ('channel',)
        if not self.dims == expected:
            raise AttributeError(
                f"{self.__class__.__name__} must have dimensions {expected}."
            )

    def plot_laminar(self, ax=None, figsize=(10, 15), **kwargs):
        """Plot a depth profile of values.
        Requires 'y' coordinate on 'channel' dimension."""
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        da = self.sortby('y') if 'y' in self.coords else self._da # Sort by depth if possible
        da.plot.line(y='channel', ax=ax, **kwargs)

        if 'structure' in self.coords:
            boundaries = da.isel(channel=utils.get_boundary_ilocs(da, 'structure'))
            ax.set_yticks(boundaries.channel)
            ax.set_yticklabels(boundaries.structure.values)
            for ch in boundaries.channel:
                ax.axhline(ch, alpha=0.5, color="dimgrey", linestyle="--")

class KernelCurrentSourceDensity(DataArrayWrapper):
    """kCSD estimates.

    Dimensions: ('time', 'pos').
    Attributes:
        fs: Sampling rate
        kcsd: KCSD1D object

    If the estimation locations requested of KCSD1D correspond
    exactly to electrode positions, a `channel` coordinate on the `pos` dimension
    will give corresponding channels for each estimate.
    """
    def __init__(self, da):
        super().__init__(da)
        self._validate()

    def _validate(self):
        expected_dims = ('time', 'pos')
        expected_attrs = {'fs', 'kcsd'}
        if not self.dims == expected_dims:
            raise AttributeError(
                f"{self.__class__.__name__} must have dimensions {expected_dims}."
            )
        for attr in expected_attrs:
            if not attr in self.attrs:
                raise AttributeError(
                    f"{self.__class__.__name__} must have attribute {attr}."
                )
