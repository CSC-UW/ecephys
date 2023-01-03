import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from horology import Timing
import spikeinterface.full as si
from spikeinterface.sortingcomponents.motion_correction import (CorrectMotionRecording, correct_motion_on_peaks)
from spikeinterface.sortingcomponents.motion_estimation import estimate_motion, clean_motion_vector
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spikeinterface.sortingcomponents.peak_localization import ( LocalizeCenterOfMass, LocalizeMonopolarTriangulation, localize_peaks)
from spikeinterface.widgets import (plot_displacement, plot_pairwise_displacement)

logger = logging.getLogger(__name__)


def get_peak_displacement_fig(si_rec, peaks, peak_locations, peak_locations_corrected, motion, temporal_bins, spatial_bins, extra_check):
    fig, axes = plt.subplots(figsize=(60, 20), ncols=3)
    ALPHA = 0.002 # >= 0.002 or invisible
    DECIMATE_RATIO = 10

    # Peak motion
    x = peaks[::DECIMATE_RATIO]['sample_ind'] / si_rec.get_sampling_frequency()
    y = peak_locations[::DECIMATE_RATIO]['y']
    y_corrected = peak_locations_corrected[::DECIMATE_RATIO]['y']

    axes[0].scatter(x, y, s=1, color='k', alpha=ALPHA)
    plot_displacement(motion, temporal_bins, spatial_bins, extra_check, with_histogram=False, ax=axes[0])
    axes[0].set_title(
        f"Original peaks and estimated motion \n"
        f"Total N={len(peaks)} peaks. Plot {100/DECIMATE_RATIO}% of peaks."
    )

    axes[1].scatter(x, y_corrected, s=1, color='k', alpha=ALPHA)
    axes[1].set_title("Corrected peaks")

    # Peak motion
    axes[2].plot(motion)
    axes[2].set_title("Motion estimates")

    return fig, axes


def _compute_peaks(
    si_rec,
    noise_level_params,
    peak_localization_method,
    peak_localization_params,
    peak_detection_params,
    job_kwargs,
):

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
    return peaks, peak_locations


def _compute_motion(
    si_rec,
    peaks,
    peak_locations,
    motion_method_params,
    non_rigid_params,
    motion_params,
):
    with Timing(name="Estimate motion: "):
        motion, temporal_bins, spatial_bins, extra_check = estimate_motion(
            si_rec,
            peaks,
            peak_locations=peak_locations,
            method="decentralized_registration",
            method_kwargs=motion_method_params,
            non_rigid_kwargs=non_rigid_params,
            clean_motion_kwargs=None,
            upsample_to_histogram_bin=False,  # Keep false if we clean motion separately
            output_extra_check=True,
            progress_bar=True,
            verbose=False,
            direction="y",
            **motion_params,
        )
    return motion, temporal_bins, spatial_bins, extra_check


def _clean_motion(
    motion,
    temporal_bins,
    clean_motion_params,
    bin_duration_s,
):
    with Timing(name="Clean motion: "):
        motion = clean_motion_vector(
            motion,
            temporal_bins,
            bin_duration_s,
            **clean_motion_params
        )
    return motion


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
    rerun_existing=True,
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
    output_dir.mkdir(parents=True, exist_ok=True)
    peaks_path = output_dir/'peaks.npy'
    peak_locations_path = output_dir/'peak_locations.npy'
    motion_path = output_dir/'motion_non_rigid.npz'
    clean_motion_path = output_dir/'motion_non_rigid_clean.npz'

    compute_peaks = (
        rerun_existing 
        or not peaks_path.exists()
        or not peak_locations_path.exists()
    )
    if compute_peaks:
        print("(Re)compute peaks")
        peaks, peak_locations = _compute_peaks(
            si_rec,
            noise_level_params,
            peak_localization_method,
            peak_localization_params,
            peak_detection_params,
            job_kwargs,
        )
        print("Save peaks/peak locations at :")
        print(peaks_path)
        print(peak_locations_path)
        np.save(peaks_path, peaks)
        np.save(peak_locations_path, peak_locations)
    else:
        print("Load peaks/peak locations from :")
        print(peaks_path)
        print(peak_locations_path)
        peaks = np.load(peaks_path)
        peak_locations = np.load(peak_locations_path)

    compute_motion = (
        rerun_existing
        or compute_peaks
        or not motion_path.exists()
    )
    if compute_motion:
        print("(Re)compute motion (no cleaning)")
        motion, temporal_bins, spatial_bins, extra_check = _compute_motion(
            si_rec,
            peaks,
            peak_locations,
            motion_method_params,
            non_rigid_params,
            motion_params,
        )
        print("Save uncleaned motion at :")
        print(motion_path)
        np.savez(
            motion_path, 
            motion=motion,
            temporal_bins=temporal_bins,
            spatial_bins=spatial_bins,
            **extra_check
        )
    else:
        print("Load uncleaned motion from :")
        print(motion_path)
        npz = np.load(motion_path)
        motion = npz['motion']
        temporal_bins = npz['temporal_bins']
        spatial_bins = npz['spatial_bins']
        extra_check=dict(
            motion_histogram=npz['motion_histogram'],
            spatial_hist_bins=npz['spatial_hist_bins'],
            temporal_hist_bins=npz['temporal_hist_bins'],
        )

    clean_motion = (
        rerun_existing
        or compute_motion
        or not clean_motion_path.exists()
    )
    if clean_motion:
        print("(Re)clean motion")
        if clean_motion_params is not None:
            assert "bin_duration_s" in motion_params
            motion_clean = _clean_motion(
                motion,
                temporal_bins,
                clean_motion_params,
                motion_params["bin_duration_s"]
            )
        else:
            motion_clean = motion
        print("Save clean motion at :")
        print(clean_motion_path)
        np.savez(
            clean_motion_path, 
            motion=motion_clean,
            temporal_bins=temporal_bins,
            spatial_bins=spatial_bins,
            **extra_check
        )
    else:
        print("Load clean motion from :")
        print(clean_motion_path)
        npz = np.load(motion_path)
        motion_clean = npz['motion']
        temporal_bins = npz['temporal_bins']
        spatial_bins = npz['spatial_bins']
        extra_check=dict(
            motion_histogram=npz['motion_histogram'],
            spatial_hist_bins=npz['spatial_hist_bins'],
            temporal_hist_bins=npz['temporal_hist_bins'],
        )
    motion = motion_clean

    with Timing(name="Get corrected peaks (debugging figure): "):
        print("Correct motion on peaks")
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
        print("Plot motion")
        fig, _ = get_peak_displacement_fig(
            si_rec, peaks, peak_locations, peak_locations_corrected,
            motion, temporal_bins, spatial_bins, extra_check,
        )
        savepath = output_dir/"peak_displacement.png"
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
            border_mode="remove_channels",
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
    rerun_existing=True,
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
                rerun_existing=rerun_existing,
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