import spikeinterface.full as si
from pathlib import Path
from horology import Timing
import logging
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spikeinterface.sortingcomponents.peak_localization import localize_peaks, LocalizeMonopolarTriangulation, LocalizeCenterOfMass
from spikeinterface.sortingcomponents.motion_estimation import estimate_motion
from spikeinterface.sortingcomponents.motion_correction import correct_motion_on_peaks, CorrectMotionRecording
from spikeinterface.widgets import plot_pairwise_displacement, plot_displacement
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)


def get_peak_displacement_fig(si_rec, peaks, peak_locations, peak_locations_corrected, motion, temporal_bins, spatial_bins, extra_check):
    fig, axes = plt.subplots(figsize=(15, 20), nrows=2)

    # Peak motion
    x = peaks['sample_ind'] / si_rec.get_sampling_frequency()
    y = peak_locations['y']
    axes[0].scatter(x, y, s=1, color='k', alpha=0.01)
    plot_displacement(motion, temporal_bins, spatial_bins, extra_check, with_histogram=False, ax=axes[0])
    axes[0].set_title("Original peaks and estimated motion")

    x = peaks['sample_ind'] / si_rec.get_sampling_frequency()
    y = peak_locations_corrected['y']
    axes[1].scatter(x, y, s=1, color='k', alpha=0.01)
    axes[0].set_title("Corrected peaks")

    return fig, axes


def _prepro_drift_correction(
    si_rec,
    output_dir=None,
    noise_level_params=None,
    peak_detection_params=None,
    peak_localization_method='center_of_mass',
    peak_localization_params=None,
    motion_method_params=None,
    non_rigid_params=None,
    clean_motion_params=None,
    motion_params=None,
    job_kwargs=None,
):
    # Input
    if noise_level_params is None:
        noise_level_params = {}
    if peak_detection_params is None:
        peak_detection_params = {}
    if peak_localization_method is None:
        peak_localization_method = {}
    if peak_localization_params is None:
        peak_localization_params = {}
    if motion_method_params is None:
        motion_method_params = {}
    if non_rigid_params is None:
        non_rigid_params = {}
    if motion_params is None:
        motion_params = {}
    if job_kwargs is None:
        job_kwargs = {}

    # Output
    drift_output_dir = Path(output_dir)/'drift_correction'
    drift_output_dir.mkdir(parents=True, exist_ok=True)

    # Steps
    with Timing(name="Get noise levels: "):
        noise_levels = si.get_noise_levels(
            si_rec,
            return_scaled=False,
            **noise_level_params,
        )

    with Timing(name="Detect peaks: "):
        peak_pipeline_steps = [
            PEAK_LOCALIZATION_FUNCTIONS[peak_localization_method](
                si_rec,
                **peak_localization_params
            )
        ]

        peaks, peak_locations = detect_peaks(
            si_rec,
            noise_levels=noise_levels,
            pipeline_steps=peak_pipeline_steps,
            **peak_detection_params,
            **job_kwargs,
        )

    with Timing(name="Estimate motion: "):
        motion, temporal_bins, spatial_bins, extra_check = estimate_motion(
            si_rec,
            peaks,
            peak_locations=peak_locations,
            method="decentralized_registration",
            method_kwargs=motion_method_params,
            non_rigid_kwargs=non_rigid_params,
            clean_motion_kwargs=clean_motion_params,
            upsample_to_histogram_bin=False,
            output_extra_check=True,
            progress_bar=True,
            verbose=False,
            direction="y",
            **motion_params,
        )
        print(motion.shape)
        print(spatial_bins)

    with Timing(name="Get corrected peaks (debugging figure): "):
        times = si_rec.get_times()
        peak_locations_corrected = correct_motion_on_peaks(
            peaks,
            peak_locations,
            times,
            motion,
            temporal_bins,
            spatial_bins,
            direction='y',
        )

    with Timing(name="Plot peak displacement and corrected peaks: "):
        fig, _ = get_peak_displacement_fig(
            si_rec, peaks, peak_locations, peak_locations_corrected,
            motion, temporal_bins, spatial_bins, extra_check,
        )
        savepath = drift_output_dir/"peak_displacement.png"
        print(f"Save debugging fig at {savepath}")
        fig.savefig(
            savepath,
            bbox_inches="tight",
        )

    with Timing(name="Correct motion on traces: "):
        print(si_rec.get_traces(start_frame=0, end_frame=100).shape)
        rec_corrected = CorrectMotionRecording(
            si_rec,
            motion,
            temporal_bins,
            spatial_bins,
            direction=1,
        )
        print(motion.shape)
        print(temporal_bins.shape)
        print(spatial_bins.shape)
        print(rec_corrected.get_traces(start_frame=0, end_frame=100).shape)
    
    return rec_corrected


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
        if step_name == 'drift_correction':
            # This one requires passing output_dir
            # And has a bunch of "sub-steps" params
            # What's the best way to pass ouput_dir only if recognized?
            # Better catching TypeError instead? 
            si_rec = PREPRO_FUNCTIONS[step_name](
                si_rec,
                output_dir=output_dir,
                noise_level_params = step_params.get('noise_level_params', None),
                peak_detection_params = step_params.get('peak_detection_params', None),
                peak_localization_method = step_params.get('peak_localization_method', None),
                peak_localization_params = step_params.get('peak_localization_params', None),
                motion_method_params = step_params.get('motion_method_params', None),
                non_rigid_params = step_params.get('non_rigid_params', None),
                clean_motion_params = step_params.get('clean_motion_params', None),
                motion_params = step_params.get('motion_params', None),
                job_kwargs=step_params.get('job_kwargs', None),
            )
        else:
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


PEAK_LOCALIZATION_FUNCTIONS = {
    "center_of_mass": LocalizeCenterOfMass,
    "monopolar_triangulation": LocalizeMonopolarTriangulation,
}