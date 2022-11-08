import spikeinterface.full as si
import logging


logger = logging.getLogger(__name__)


def _prepro_drift_correction(
    si_rec,
    **kwargs,
):
    return NotImplementedError()


def preprocess_si_recording(
    si_rec, 
    opts,
    output_dir=None,
):
    prepro_opts = opts['preprocessing']

    for step in prepro_opts:
        step_name = step['step_name']
        step_params = step['step_params']
        if step_name not in PREPRO_FUNCTIONS:
            raise ValueError(
                f"Unrecognized preprocessing step: {step_name}."
                f"Should be one of: {list(PREPRO_FUNCTIONS.keys())}"
            )
        # logger.info(
        print(
            f"Apply preprocessing step: `{step_name}` with params `{step_params}`",
        )
        si_rec = PREPRO_FUNCTIONS[step_name](
            si_rec,
            **step_params
        )

    return si_rec


PREPRO_FUNCTIONS = {
    "scale": si.scale,
    "phase_shift": si.phase_shift,
    # "bad_channels": _prepro_bad_channels,
    "bad_channels": si.remove_bad_channels,
    "bandpass_filter": si.bandpass_filter,
    "common_reference": si.common_reference,
    "whiten": si.whiten,
    "drift_correction": _prepro_drift_correction,
}
