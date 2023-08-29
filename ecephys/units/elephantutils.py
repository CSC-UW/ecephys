import neo
import numpy as np
from elephant.statistics import instantaneous_rate, time_histogram
from elephant import kernels
from ecephys.units.dtypes import SpikeTrain_Secs
from quantities import s, ms
import xarray as xr

def convert_to_neo_spiketrain(
    spike_train_sec: SpikeTrain_Secs,
    t_start_sec=None,
    t_stop_sec=None,
):
    if t_start_sec is None:
        t_start_sec = np.min(spike_train_sec)
    if t_stop_sec is None:
        t_stop_sec = np.max(spike_train_sec)

    return neo.SpikeTrain(
        spike_train_sec,
        t_start=t_start_sec * s,
        t_stop=t_stop_sec * s,
        units=s
    )

def compute_instantaneous_rate_xrsig(
    spiketrain_sec: SpikeTrain_Secs,
    sampling_frequency_hz: float = 300,
    gaussian_sigma_msec: float = None,
    t_start_sec: float = None,
    t_stop_sec : float = None,
    channel_name: str = "mua",
) -> xr.DataArray:
    """Elephant-style instantaneous rate with gaussian kernel."""
    if gaussian_sigma_msec is None:
        kernel="auto"
    else:
        kernel = kernels.GaussianKernel(
            sigma=gaussian_sigma_msec * ms,
        )

    neotrain = convert_to_neo_spiketrain(
        spiketrain_sec,
        t_start_sec=min(t_start_sec, np.min(spiketrain_sec)),
        t_stop_sec=max(t_stop_sec, np.max(spiketrain_sec)),
    )

    res = instantaneous_rate(
        neotrain,
        (1 / sampling_frequency_hz) * s,
        kernel=kernel
    )

    return xr.DataArray(
        res.magnitude,
        dims=["time", "channel"],
        coords={
            "channel": [channel_name],
            "time": res.times.magnitude,
        },
        name="Instantaneous rate (Hz)",
        attrs={"fs": sampling_frequency_hz}
    )
