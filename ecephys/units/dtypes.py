from typing import Union, Any

import numpy as np
import numpy.typing as npt
import xarray as xr

SpikeTrain_Samples = npt.NDArray[np.int64]  # Spike train, in samples
SpikeTrain_Secs = npt.NDArray[np.float64]  # Spike train, in seconds
SpikeTrain = Union[SpikeTrain_Samples, SpikeTrain_Secs]

ClusterTrains_Samples = dict[np.int64, SpikeTrain_Samples]  # Key: Cluster ID
ClusterTrains_Secs = dict[np.int64, SpikeTrain_Secs]  # Key: Cluster ID
ClusterTrains = Union[ClusterTrains_Samples, ClusterTrains_Secs]

SpikeTrainDict = dict[Any, SpikeTrain]

ClusterIDs = npt.NDArray[np.int64]
ClusterIXs = npt.NDArray[np.int64]

XArray = Union[xr.DataArray, xr.Dataset]
