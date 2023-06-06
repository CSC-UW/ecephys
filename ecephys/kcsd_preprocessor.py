import warnings

import kcsd
import numpy as np

from spikeinterface.core.core_tools import define_function_from_class
from spikeinterface.preprocessing.basepreprocessor import (
    BasePreprocessor,
    BasePreprocessorSegment,
)


class KCSD1DRecording(BasePreprocessor):
    """
    Perform 1D Kernel Current Source Density

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor to which KCSD will be applied.
    kcsd_kwargs: dict
        Parameters taken by the kcsd.KCSD1D constructor.
        You probably want to estimate some of these (esp. `lambd`) beforehand, using e.g. cross-validation or L-curve.
        See: https://kcsd-python.readthedocs.io/en/latest/DOCUMENTATION.html#kcsd.KCSD.KCSD1D.__init__

    Returns
    -------
    kcsd: KCSD1D
        The KCSD recording extractor
    """

    name = "kcsd1d"

    def __init__(self, recording, kcsd_kwargs):
        x = recording.get_channel_locations(axes="x")
        if np.unique(x).size > 1:
            warnings.warn(
                "Your channels are not truly co-linear. Consider using a 2D CSD."
            )
        y = recording.get_channel_locations(axes="y")
        if not np.all(np.unique(y) == y.squeeze()):
            warnings.warn(
                "Your channel y locations are not unique and sorted by depth."
                "Consider using DepthOrderRecording and/or AverageAcrossDirectionRecording."
            )

        # Because the result of the KCSD may have a number of traces != extracelluar potentials, we need to create a new recording to hold the result.
        # This requires anticipating the shape and properties of the result, in particular the number and position of estimates, and the dtype.
        y_mm = y / 1000.00  # Convert um to mm
        dummy_traces = np.zeros_like(y_mm)
        k = kcsd.KCSD1D(
            y_mm,
            dummy_traces,
            **kcsd_kwargs,
        )
        y_kcsd = np.round(
            k.estm_pos.squeeze() * 1000.0
        )  # y-coords, in um, of kCSD estimates

        # If estimate locations match channel positions exactly, keep the channel labels
        # Otherwise, "channels" no longer have a meaning beyond position.
        if np.all(y_kcsd == y.squeeze()):
            new_channel_ids = recording.get_channel_ids()
        else:
            new_channel_ids = y_kcsd
        BasePreprocessor.__init__(
            self, recording, channel_ids=new_channel_ids, dtype=k.values("CSD").dtype
        )

        # New x coordinates are all 0.
        new_locs = np.vstack([np.zeros_like(y_kcsd), y_kcsd]).T
        self.set_channel_locations(new_locs)
        # This is safe, right? It won't unintended side-effects on `recording`? And should be done?
        self.set_channel_gains(1)
        self.set_channel_offsets(0)

        for parent_segment in recording._recording_segments:
            rec_segment = KCSD1DRecordingSegment(parent_segment, y_mm, kcsd_kwargs)
            self.add_recording_segment(rec_segment)
            # It should be safe to derive KCSD segments from LFP segments, since segments only really
            # store information about time, and not channel/position.

        self._kwargs = dict(recording=recording, kcsd_kwargs=kcsd_kwargs)


class KCSD1DRecordingSegment(BasePreprocessorSegment):
    def __init__(self, parent_recording_segment, y_mm, kcsd_kwargs):
        BasePreprocessorSegment.__init__(self, parent_recording_segment)
        self.y_mm = y_mm
        self.kcsd_kwargs = kcsd_kwargs

    def get_traces(self, start_frame, end_frame, channel_indices):
        # How can I properly check that `channel_indices` is not a subset
        traces = self.parent_recording_segment.get_traces(
            start_frame, end_frame, slice(None)
        )

        # Compute kCSD
        k = kcsd.KCSD1D(
            self.y_mm,
            traces.T,
            **self.kcsd_kwargs,
        )
        csd = k.values("CSD").T
        csd = csd[:, channel_indices]
        return csd


# function for API
whiten = define_function_from_class(source_class=KCSD1DRecording, name="kcsd1d")
