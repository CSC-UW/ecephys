from horology import Timing
import logging
import matplotlib.pyplot as plt
import numpy as np
import spikeinterface.full as si
from spikeinterface import widgets
from spikeinterface.sortingcomponents import (
    motion_correction,
    motion_estimation,
    peak_detection,
    peak_pipeline,
    peak_localization,
)

logger = logging.getLogger(__name__)


def get_raw_peak_fig(
    si_rec, peaks, peak_locations, motion, temporal_bins, spatial_bins, extra_check
):

    ALPHA = 0.002  # >= 0.002 or invisible
    if len(peaks) <= 50e6:
        DECIMATE_RATIO = 1
    elif len(peaks) <= 400e6:
        DECIMATE_RATIO = 5
    else:
        DECIMATE_RATIO = 10

    fig, ax = plt.subplots(figsize=(20, 60))

    # Peak motion
    x = peaks[::DECIMATE_RATIO]["sample_ind"] / si_rec.get_sampling_frequency()
    y = peak_locations[::DECIMATE_RATIO]["y"]

    ax.scatter(x, y, s=1, color="k", alpha=ALPHA)

    widgets.plot_displacement(
        motion, temporal_bins, spatial_bins, extra_check, with_histogram=False, ax=ax
    )
    ax.set_title(
        f"Original peaks and estimated motion \n"
        f"Total N={len(peaks)} peaks. Plot {100/DECIMATE_RATIO}% of peaks."
    )

    return fig


def get_peak_displacement_fig(
    si_rec,
    peaks,
    peak_locations,
    peak_locations_corrected,
    motion,
    temporal_bins,
    spatial_bins,
    extra_check,
):
    ALPHA = 0.002  # >= 0.002 or invisible
    if len(peaks) <= 50e6:
        DECIMATE_RATIO = 1
    elif len(peaks) <= 400e6:
        DECIMATE_RATIO = 5
    else:
        DECIMATE_RATIO = 10

    fig = plt.figure(figsize=(60, 20), layout="constrained")
    spec = fig.add_gridspec(2, 3)

    # Peak motion
    x = peaks[::DECIMATE_RATIO]["sample_ind"] / si_rec.get_sampling_frequency()
    y = peak_locations[::DECIMATE_RATIO]["y"]
    y_corrected = peak_locations_corrected[::DECIMATE_RATIO]["y"]

    ax = fig.add_subplot(spec[:, 0])  # Left
    ax.scatter(x, y, s=1, color="k", alpha=ALPHA)
    widgets.plot_displacement(
        motion, temporal_bins, spatial_bins, extra_check, with_histogram=False, ax=ax
    )
    ax.set_title(
        f"Original peaks and estimated motion \n"
        f"Total N={len(peaks)} peaks. Plot {100/DECIMATE_RATIO}% of peaks."
    )

    ax = fig.add_subplot(spec[:, 1])
    ax.scatter(x, y_corrected, s=1, color="k", alpha=ALPHA)
    ax.set_title("Corrected peaks")

    # Peak motion
    ax = fig.add_subplot(spec[0, 2])  # Top right
    ax.plot(motion)
    ax.set_title("Motion estimates")

    ax = fig.add_subplot(spec[1, 2])  # Bottom right
    ax.plot(motion)
    ax.set_ylim(-100, 100)
    ax.set_title("Motion estimates")

    return fig


def _compute_peaks(
    si_rec,
    noise_level_params,
    extract_waveforms_params,
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
        extract_dense_waveforms = peak_pipeline.ExtractDenseWaveforms(si_rec, return_output=False, **extract_waveforms_params)
        pipeline_nodes = [
            extract_dense_waveforms,
            PEAK_LOCALIZATION_FUNCTIONS[peak_localization_method](
                si_rec, **peak_localization_params, parents=[extract_dense_waveforms]
            )
        ]

        peaks, peak_locations = peak_detection.detect_peaks(
            si_rec,
            noise_levels=noise_levels,
            pipeline_nodes=pipeline_nodes,
            **peak_detection_params,
            **job_kwargs,
        )
    return peaks, peak_locations


def _compute_motion(
    si_rec,
    peaks,
    peak_locations,
    motion_method_params,
    motion_params,
):
    """Estimate motion without cleaning."""
    with Timing(name="Estimate motion: "):
        (
            motion,
            temporal_bins,
            spatial_bins,
            extra_check,
        ) = motion_estimation.estimate_motion(
            si_rec,
            peaks,
            peak_locations=peak_locations,
            output_extra_check=True,
            progress_bar=True,
            verbose=False,
            direction="y",
            upsample_to_histogram_bin=False,  # Keep false if we clean motion separately
            post_clean=False,  # We clean in separate step
            **motion_params,
            **motion_method_params,
        )
    return motion, temporal_bins, spatial_bins, extra_check


def _clean_motion(
    motion,
    temporal_bins,
    clean_motion_params,
    bin_duration_s,
):
    with Timing(name="Clean motion: "):
        motion = motion_estimation.clean_motion_vector(
            motion, temporal_bins, bin_duration_s, **clean_motion_params
        )
    return motion


def _prepro_drift_correction(
    si_rec,
    output_dir=None,
    noise_level_params=None,
    peak_detection_params=None,
    extract_waveforms_params=None,
    peak_localization_method="localize_monopolar_location",
    peak_localization_params=None,
    motion_params=None,
    motion_method_params=None,
    clean_motion_params=None,
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
    if motion_params is None:
        motion_params = {}
    if job_kwargs is None:
        job_kwargs = {}

    # Output
    output_dir.mkdir(parents=True, exist_ok=True)
    peaks_path = output_dir / "peaks.npy"
    peak_locations_path = output_dir / "peak_locations.npy"
    motion_path = output_dir / "motion_non_rigid.npz"
    clean_motion_path = output_dir / "motion_non_rigid_clean.npz"

    compute_peaks = (
        rerun_existing or not peaks_path.exists() or not peak_locations_path.exists()
    )
    if compute_peaks:
        print("(Re)compute peaks")
        print(job_kwargs)
        peaks, peak_locations = _compute_peaks(
            si_rec,
            noise_level_params,
            extract_waveforms_params,
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

    compute_motion = rerun_existing or compute_peaks or not motion_path.exists()
    if compute_motion:
        print("(Re)compute motion (no cleaning)")
        motion, temporal_bins, spatial_bins, extra_check = _compute_motion(
            si_rec,
            peaks,
            peak_locations,
            motion_method_params,
            motion_params,
        )
        print("Save uncleaned motion at :")
        print(motion_path)
        np.savez(
            motion_path,
            motion=motion,
            temporal_bins=temporal_bins,
            spatial_bins=spatial_bins,
            **extra_check,
        )
    else:
        print("Load uncleaned motion from :")
        print(motion_path)
        npz = np.load(motion_path)
        motion = npz["motion"]
        temporal_bins = npz["temporal_bins"]
        spatial_bins = npz["spatial_bins"]
        extra_check = dict(
            # motion_histogram=npz['motion_histogram'],
            # spatial_hist_bins=npz['spatial_hist_bins'],
            # temporal_hist_bins=npz['temporal_hist_bins'],
        )

    clean_motion = rerun_existing or compute_motion or not clean_motion_path.exists()
    if clean_motion:
        print("(Re)clean motion")
        if clean_motion_params is not None:
            assert "bin_duration_s" in motion_params
            motion_clean = _clean_motion(
                motion,
                temporal_bins,
                clean_motion_params,
                motion_params["bin_duration_s"],
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
            **extra_check,
        )
    else:
        print("Load clean motion from :")
        print(clean_motion_path)
        npz = np.load(motion_path)
        motion_clean = npz["motion"]
        temporal_bins = npz["temporal_bins"]
        spatial_bins = npz["spatial_bins"]
        extra_check = dict(
            # motion_histogram=npz['motion_histogram'],
            # spatial_hist_bins=npz['spatial_hist_bins'],
            # temporal_hist_bins=npz['temporal_hist_bins'],
        )
    motion = motion_clean

    # Only plot if we changed the motions and can't find the plots already
    plot_filepaths = {
        "displacement": output_dir / "peak_displacement.png",
        "displacement_hq": output_dir / "peak_displacement_hq.png",
        "raw_peaks_hq": output_dir / "peak_uncorrected_hq.png",
    }
    make_debugging_plots = (
        rerun_existing
        or clean_motion
        or not all([f.exists() for f in plot_filepaths.values()])
    )
    if make_debugging_plots:

        with Timing(name="Get corrected peaks (debugging figure): "):
            print("Correct motion on peaks")
            times = si_rec.get_times()
            peak_locations_corrected = motion_correction.correct_motion_on_peaks(
                peaks,
                peak_locations,
                times,
                motion,
                temporal_bins,
                spatial_bins,
                direction="y",
            )

        with Timing(name="Plot peak displacement and corrected peaks: "):
            print(f"Save debugging figs at {plot_filepaths}")
            fig = get_peak_displacement_fig(
                si_rec,
                peaks,
                peak_locations,
                peak_locations_corrected,
                motion,
                temporal_bins,
                spatial_bins,
                extra_check,
            )
            fig.savefig(plot_filepaths["displacement"], bbox_inches="tight")
            fig.savefig(plot_filepaths["displacement_hq"], bbox_inches="tight", dpi=500)
            fig = get_raw_peak_fig(
                si_rec,
                peaks,
                peak_locations,
                motion,
                temporal_bins,
                spatial_bins,
                extra_check,
            )
            savepath = output_dir / "peak_uncorrected_hq.png"
            print(f"Save debugging fig at {savepath}")
            fig.savefig(plot_filepaths["raw_peaks_hq"], bbox_inches="tight", dpi=500)

    with Timing(name="Correct motion on traces: "):
        print(si_rec.get_traces(start_frame=0, end_frame=100).shape)
        rec_corrected = motion_correction.CorrectMotionRecording(
            si_rec,
            motion,
            temporal_bins,
            spatial_bins,
            direction=1,
            border_mode="remove_channels",
            spatial_interpolation_method="kriging",
            sigma_um=20.0,
            p=1,
            num_closest=3,
        )

    return rec_corrected


def preprocess_si_recording(
    si_rec,
    opts,
    output_dir=None,
    rerun_existing=True,
    job_kwargs=None,
) -> si.BaseRecording:
    if job_kwargs is None:
        job_kwargs = {
            "n_jobs": 1,
            "chunk_duration": "1s",
            "progress_bar": True,
        }
    prepro_opts = opts["preprocessing"]

    for step in prepro_opts:
        step_name = step["step_name"]
        step_params = step["step_params"]
        if step_name not in PREPRO_FUNCTIONS:
            raise ValueError(
                f"Unrecognized preprocessing step: {step_name}."
                f"Should be one of: {list(PREPRO_FUNCTIONS.keys())}"
            )
        # logger.info(
        print(
            f"Apply preprocessing step: `{step_name}` with params `{step_params}`",
        )
        if step_name == "drift_correction":
            # This one requires passing output_dir
            # And has a bunch of "sub-steps" params
            # What's the best way to pass ouput_dir only if recognized?
            # Better catching TypeError instead?
            si_rec = PREPRO_FUNCTIONS[step_name](
                si_rec,
                output_dir=output_dir,
                noise_level_params=step_params.get("noise_level_params", None),
                peak_detection_params=step_params.get("peak_detection_params", None),
                extract_waveforms_params=step_params.get("extract_waveforms_params", None),
                peak_localization_method=step_params.get(
                    "peak_localization_method", None
                ),
                peak_localization_params=step_params.get(
                    "peak_localization_params", None
                ),
                motion_params=step_params.get("motion_params", None),
                motion_method_params=step_params.get("motion_method_params", None),
                clean_motion_params=step_params.get("clean_motion_params", None),
                rerun_existing=rerun_existing,
                job_kwargs=job_kwargs,
            )
        elif step_name == "whiten":
            # Backwards compatibility
            step_params = step_params.copy()
            if not "dtype" in step_params:
                step_params["dtype"] = "float32"
            si_rec = PREPRO_FUNCTIONS[step_name](si_rec, **step_params)
        else:
            si_rec = PREPRO_FUNCTIONS[step_name](si_rec, **step_params)

    return si_rec


PREPRO_FUNCTIONS = {
    "scale": si.scale,
    "phase_shift": si.phase_shift,
    # "bad_channels": _prepro_bad_channels,
    "bandpass_filter": si.bandpass_filter,
    "common_reference": si.common_reference,
    "whiten": si.whiten,
    "zscore": si.zscore,
    "drift_correction": _prepro_drift_correction,
}


PEAK_LOCALIZATION_FUNCTIONS = {
    "center_of_mass": peak_localization.LocalizeCenterOfMass,
    "monopolar_triangulation": peak_localization.LocalizeMonopolarTriangulation,
}
