import pathlib

import numpy as np
from spikeinterface.core import BaseRecording, BaseRecordingSegment
from spikeinterface.core.core_tools import define_function_from_class
import xarray as xr

from ecephys import utils
from ecephys import xrsig


class XrsigZarrExtractor(BaseRecording):
    def __init__(
        self,
        path: str,  # Not pathlib.Path, so that object can be created from serialization dict during multiprocessing
    ):
        """For converting WNE/xrsig-style zarr stores to zarr-backed Spikeinterface recordings.
        Each continuous block of data is one segment. So a recording with multiple stops/starts/crashes will be multisegment.

        Examples:
        ---------
        with xr.open_dataarray(input_zarr, engine='zarr', chunks='auto') as z:
            chunksize = z.encoding['preferred_chunks']['time']
            compressor = z.encoding['compressor']
        rec = XrsigZarrExtractor(input_zarr)
        rec.save(folder=output_zarr, format='zarr', chunk_size = z.encoding['preferred_chunks']['time'], compressor = z.encoding['compressor'], n_jobs=20, progress_bar=True, mp_context='spawn')
        """
        dat = xr.open_dataarray(path, engine="zarr", chunks="auto")
        if dat.chunks:
            try:
                dat.chunksizes
            except ValueError:
                print(
                    "Xarray claims that chunk sizes are inconsistent. Rechunking using encoding['preferred chunks']..."
                )
                dat = dat.chunk(dat.encoding["preferred_chunks"])
        utils.hotfix_times(dat.time.values)
        dat = dat.drop_duplicates(dim="time", keep="first")
        # Timestamps should really be computed once, and added as a kwarg, so that they get propgated and not recomputed.
        # Example: https://github.com/SpikeInterface/spikeinterface/blob/main/src/spikeinterface/preprocessing/whiten.py#L47-L90

        BaseRecording.__init__(
            self,
            channel_ids=dat.channel.values,
            sampling_frequency=dat.fs,
            dtype=dat.dtype,
        )
        for i, (start_frame, end_frame) in enumerate(xrsig.get_segments(dat)):
            seg_dat = dat.isel(time=slice(start_frame, end_frame))
            self.add_recording_segment(XrsigZarrRecordingSegment(seg_dat, seg_dat.fs))
            self.set_times(seg_dat.time.values, segment_index=i)

        assert dat.units == "uV", "Expected data to already be in uV"
        self.set_channel_gains(1)
        self.set_channel_offsets(0)

        self.set_channel_locations(np.vstack([dat["x"], dat["y"]]).T)
        # TODO: Set probe from a .meta file? What is to be gained, since we already have channel locs and scaled traces?
        # TODO: We don't need to set inter_sample_shift, since we have already accounted for this with tshift, right?
        #   See https://github.com/SpikeInterface/spikeinterface/blob/main/src/spikeinterface/extractors/neoextractors/spikeglx.py

        path = pathlib.Path(path)
        self._kwargs = {"path": str(path.absolute())}


class XrsigZarrRecordingSegment(BaseRecordingSegment):
    def __init__(self, da: xr.DataArray, sampling_frequency: float):
        BaseRecordingSegment.__init__(self, sampling_frequency=sampling_frequency)
        self._da = da

    def get_num_samples(self):
        return self._da.time.size

    def get_traces(self, start_frame, end_frame, channel_indices):
        return self._da.isel(
            time=slice(start_frame, end_frame), channel=channel_indices
        ).values
        # SI TODO: Is it okay to return a memmap, or does it have to return an in-memory numpy array? A lazy da does not work, nor does a dask array.
        # SI TODO: Document return shape (n_samples, n_traces)


read_wne_xr_zarr = define_function_from_class(
    source_class=XrsigZarrExtractor, name="read_xrsig_zarr"
)
