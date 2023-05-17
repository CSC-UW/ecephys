import kcsd
import numpy as np
from spikeinterface.core.core_tools import define_function_from_class
from spikeinterface.preprocessing.basepreprocessor import BasePreprocessor, BasePreprocessorSegment

class KCSD1DRecording(BasePreprocessor):
    """
    Perform 1D Kernel Current Source Density

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor to which KCSD will be applied.

    Returns
    -------
    kcsd: KCSD1D
        The KCSD recording extractor
    """
    name = 'kcsd1d'

    def __init__(
        self,
        recording,
    ):
        BasePreprocessor.__init__(self, recording)

        for parent_segment in recording._recording_segments:
            rec_segment = KCSD1DRecordingSegment(parent_segment)
            self.add_recording_segment(rec_segment)

        self._kwargs = dict(
            recording=recording
        )

class KCSD1DRecordingSegment(BasePreprocessorSegment):
    def __init__(self, parent_recording_segment):
        BasePreprocessorSegment.__init__(self, parent_recording_segment)

    def get_traces(self, start_frame, end_frame, channel_indices):
        """Very important: First the channels specified by `channel_indices` are fetched, then KCSD is applied."""
        traces = self.parent_recording_segment.get_traces(
            start_frame, end_frame, channel_indices)

        # Get from self
        ele_pos_um = None
        kcsd_kwargs = None
        expected_estm_pos_um = None

        # Convert um to mm for KCSD package.
        um_per_mm = 1000
        ele_pos_mm = ele_pos_um / um_per_mm

        # Compute kCSD
        k = kcsd.KCSD1D(
            ele_pos_mm.reshape(-1, 1),
            traces.T,
            **kcsd_kwargs,
        )

        # TODO: Check that estm_pos match expected. Or, okay to save to recoridng object?
        estm_pos_um = np.round(k.estm_x * um_per_mm)

        csd = k.values("CSD").T

        return csd


# function for API
whiten = define_function_from_class(source_class=KCSD1DRecording, name="kcsd1d")


def do_kcsd1d(traces, ele_pos_um, **kcsd_kwargs):
    # Convert um to mm for KCSD package.
    um_per_mm = 1000
    ele_pos_mm = ele_pos_um / um_per_mm

    # Compute kCSD
    k = kcsd.KCSD1D(
        ele_pos_mm.reshape(-1, 1),
        traces.T,
        **kcsd_kwargs,
    )

    # Check and format result
    estm_pos_um = np.round(k.estm_x * um_per_mm)
    mask = ele_pos_um.isin(estm_pos_um)
    assert (
        estm_pos_um.size == mask.sum()
    ), "CSD returned estimates that do not match original signal positions exactly."
    csd = xr.zeros_like(pots.sel({"channel": mask}))
    csd.values = k.values("CSD").T
    csd.attrs = dict(kcsd=k, pitch_mm=pitch_mm, fs=pots.fs)
    csd.name = "csd"
    return csd
