import matplotlib.pyplot as plt
import xarray as xr
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
        expected = ('time', 'channel')
        if not self.dims == expected:
            raise AttributeError(
                f"{self.__class__.__name__} must have dimensions {expected}."
            )
        if not 'fs' in self.attrs:
            raise AttributeError(
                f"{self.__class__.__name__} must have fs attribute."
            )

    def spectrograms(self):
        # Get spectrograms for each channel
        fft_window_length = 4
        nperseg = int(fft_window_length * self.fs)  # 4 second window
        noverlap = nperseg // fft_window_length  # 1 second window overlap
        spgs = dsp.parallel_spectrogram_welch(
            self._da, nperseg=nperseg, noverlap=noverlap
        )
        return ChannelSpectrograms(spgs)

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