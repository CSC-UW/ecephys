import numpy as np
import xarray as xr

from ephyviewer.datasource.signals import BaseAnalogSignalSource


class DataArrayRecordingSource(BaseAnalogSignalSource):
    def __init__(
        self,
        da: xr.DataArray,
        timedim: str = "time",
        chandim: str = "channel",
        sample_rate: float = None,
    ):
        BaseAnalogSignalSource.__init__(self)
        if not ((timedim in da.dims) and (chandim in da.dims)):
            raise ValueError(
                f"Expected to find dimensions {timedim} and {chandim} among DataArray dims {da.dims}"
            )
        if not len(da.dims) == 2:
            raise ValueError("DataArray has more than two dimensions.")
        assert np.all(
            np.diff(da[timedim].values) >= 0
        ), "Times must be monotonically increasing."

        self.da = da
        self.timedim = timedim
        self.chandim = chandim

        sample_rate = (
            self.da.attrs.get("fs", None) if sample_rate is None else sample_rate
        )
        if sample_rate is None:
            raise ValueError(
                "Sample rate must be either be provided as a kwarg, or present as an attribute on the DataArray."
            )
        self._sample_rate = sample_rate

    @property
    def nb_channel(self) -> int:
        return self.da[self.chandim].size

    def get_channel_name(self, chan: int = 0) -> str:
        return self.da[self.chandim].values[chan]

    @property
    def t_start(self) -> float:
        return float(self.da[self.timedim].values[0])

    @property
    def sample_rate(self) -> float:
        return self._sample_rate

    @property
    def t_stop(self) -> float:
        return float(self.da[self.timedim].values[-1])

    def get_length(self) -> int:
        return self.da[self.timedim].size

    def get_shape(self) -> tuple[int, int]:
        return (self.get_length(), self.nb_channel)

    def get_chunk(self, i_start: int = None, i_stop: int = None) -> np.ndarray:
        return self.da.isel(time=slice(i_start, i_stop)).values

    def time_to_index(self, t: float) -> int:
        return np.searchsorted(self.da[self.timedim].values, t)

    def index_to_time(self, ind: int) -> float:
        return self.da[self.timedim].values[ind]
