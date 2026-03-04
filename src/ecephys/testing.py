"""Test data generators for extracellular electrophysiology.

Provides functions to create synthetic Neuropixels recordings with realistic
probe geometry, inter-sample shifts, and injected spike templates. Useful as
surrogates for real si_recording.zarr files in tests and benchmarks.

Follows the pandas.testing / numpy.testing pattern: importable from any
downstream package without test-only dependencies.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import probeinterface.neuropixels_tools as npt
import spikeinterface.generation as sgen
import spikeinterface.preprocessing as sp

import spikeinterface as si

if TYPE_CHECKING:
    pass


def _compute_inter_sample_shift(
    num_channels: int,
    probe_part_number: str = "NP1000",
) -> np.ndarray:
    """Compute inter-sample shifts from Neuropixels ADC multiplexing.

    Uses the ADC sampling table from probeinterface's probe metadata
    to derive fractional sample delays for each channel.

    Args:
        num_channels: Number of recorded channels (first N of the probe).
        probe_part_number: Neuropixels probe SKU (e.g., "NP1000" for NP1.0).

    Returns:
        Array of shape (num_channels,) with fractional shifts in [0, 1).
    """
    full_probe = npt.build_neuropixels_probe(probe_part_number)
    adc_sampling_table = full_probe.annotations.get("adc_sampling_table")
    if adc_sampling_table is None:
        raise ValueError(
            f"Probe {probe_part_number} has no adc_sampling_table annotation"
        )

    num_adcs, num_channels_per_adc, mux_table = npt.make_mux_table_array(
        adc_sampling_table
    )
    total_contacts = full_probe.get_contact_count()

    # Build per-contact adc_sample_order from the mux routing table
    adc_sample_order = np.zeros(total_contacts, dtype=int)
    for adc_idx in range(num_adcs):
        for sample_idx in range(num_channels_per_adc):
            ch = mux_table[adc_idx, sample_idx]
            if ch < total_contacts:
                adc_sample_order[ch] = sample_idx

    # Compute number of ADC cycles per sample period
    ap_freq = full_probe.annotations["ap_sample_frequency_hz"]
    lf_freq = full_probe.annotations["lf_sample_frequency_hz"]
    num_cycles = int(num_channels_per_adc * (1 + lf_freq / ap_freq))

    shifts = adc_sample_order[:num_channels] / num_cycles
    return shifts


def generate_neuropixels_recording(
    num_units: int = 200,
    duration_s: float = 10.0,
    sampling_frequency: float = 30000.0,
    dtype: str = "float32",
    with_drift: bool = False,
    probe_name: str = "Neuropixels1-384",
    probe_part_number: str = "NP1000",
    seed: int | None = 42,
) -> tuple[si.BaseRecording, si.BaseSorting]:
    """Generate a synthetic Neuropixels recording with injected spikes.

    Wraps spikeinterface.generation.generate_drifting_recording() to produce
    a recording with realistic probe geometry, spatially correlated noise,
    spike templates, and correct inter-sample shifts. Useful as a surrogate
    for si_recording.zarr in tests and benchmarks.

    Args:
        num_units: Number of synthetic neurons to inject.
        duration_s: Duration in seconds.
        sampling_frequency: Sampling frequency in Hz.
        dtype: Output dtype. "float32" preserves native generation output.
            "int16" scales and casts to match raw SpikeGLX recordings.
        with_drift: If True, return the drifting recording. If False, return
            the static recording.
        probe_name: Name of the toy probe layout from spikeinterface.generation.
            Options include "Neuropixels1-384", "Neuropixels1-128",
            "Neuropixels2-384".
        probe_part_number: Neuropixels probe SKU for ADC metadata (used to
            compute inter_sample_shift). "NP1000" for NP1.0, "NP2000"
            for NP2.0.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (recording, sorting). The recording has attached probe
        geometry and inter_sample_shift property. The sorting contains
        ground truth spike times.
    """
    # Drift start must be less than duration; scale period to fit
    t_start_drift = min(0.1 * duration_s, 60.0)
    period_s = min(duration_s * 0.5, 200.0)

    static_rec, drifting_rec, sorting = sgen.generate_drifting_recording(
        num_units=num_units,
        duration=duration_s,
        sampling_frequency=sampling_frequency,
        probe_name=probe_name,
        generate_displacement_vector_kwargs=dict(
            displacement_sampling_frequency=5.0,
            drift_start_um=[0, 20],
            drift_stop_um=[0, -20],
            drift_step_um=1,
            motion_list=[
                dict(
                    drift_mode="zigzag",
                    non_rigid_gradient=None,
                    t_start_drift=t_start_drift,
                    t_end_drift=None,
                    period_s=period_s,
                ),
            ],
        ),
        generate_sorting_kwargs=dict(firing_rates=(2.0, 8.0), refractory_period_ms=4.0),
        generate_noise_kwargs=dict(noise_levels=(6.0, 8.0), spatial_decay=25.0),
        seed=seed,
    )

    rec = drifting_rec if with_drift else static_rec

    # Add inter_sample_shift from real Neuropixels ADC metadata
    actual_num_channels = rec.get_num_channels()
    inter_sample_shift = _compute_inter_sample_shift(
        actual_num_channels, probe_part_number
    )
    rec.set_property("inter_sample_shift", inter_sample_shift)

    # Set gain/offset so detect_bad_channels' coherence+psd method works
    # (it requires has_scaleable_traces()). The generated templates are
    # already in uV, so gain=1, offset=0 is correct.
    rec.set_channel_gains(1.0)
    rec.set_channel_offsets(0.0)

    # Convert dtype if needed
    if np.dtype(dtype) != rec.get_dtype():
        if np.issubdtype(np.dtype(dtype), np.integer):
            # Scale float traces to use ~80% of the integer range.
            # Estimate the current data range from a small sample, then
            # compute a gain that maps it to the target integer range.
            sample = rec.get_traces(
                start_frame=0, end_frame=min(30000, rec.get_num_frames())
            )
            current_max = np.percentile(np.abs(sample), 99.9)
            if current_max > 0:
                info = np.iinfo(np.dtype(dtype))
                gain = 0.8 * info.max / current_max
            else:
                gain = 1.0
            rec = sp.scale(rec, gain=gain, offset=0.0, dtype=dtype)
        else:
            rec = sp.astype(rec, dtype=dtype)

    return rec, sorting


def save_surrogate_zarr(
    path: str | Path,
    duration_s: float = 10.0,
    num_units: int = 50,
    dtype: str = "int16",
    probe_name: str = "Neuropixels1-384",
    chunk_duration: str = "1s",
    seed: int | None = 42,
    n_jobs: int = 1,
) -> si.BaseRecording:
    """Generate a surrogate Neuropixels recording and save as zarr.

    Convenience function for creating on-disk test data for benchmarking
    I/O-bound preprocessing chains.

    Args:
        path: Output path for the .zarr directory.
        duration_s: Duration in seconds.
        num_units: Number of synthetic neurons.
        dtype: Trace dtype (default "int16" to match raw SpikeGLX).
        probe_name: Probe layout name.
        chunk_duration: Zarr chunk duration for writing.
        seed: Random seed.
        n_jobs: Number of parallel write jobs.

    Returns:
        A ZarrRecordingExtractor loaded from the saved file.
    """
    rec, _ = generate_neuropixels_recording(
        num_units=num_units,
        duration_s=duration_s,
        dtype=dtype,
        probe_name=probe_name,
        seed=seed,
    )

    rec.save(
        folder=path,
        format="zarr",
        chunk_duration=chunk_duration,
        n_jobs=n_jobs,
        progress_bar=True,
        verbose=True,
    )

    return si.load(path)


def download_dataset(
    name: str,
    base_url: str = "",
    registry: dict[str, str] | None = None,
    local_folder: str | Path | None = None,
) -> Path:
    """Download a test dataset using pooch (stub for future hosted data).

    This is a minimal wrapper around pooch for fetching test datasets from
    a remote repository (e.g., gin.g-node.org). Currently ships with an
    empty registry — populate it when hosted datasets become available.

    Args:
        name: Name/path of the dataset file in the registry.
        base_url: Base URL of the data repository.
        registry: Mapping of filenames to SHA256 hashes. If None, uses
            the built-in (currently empty) registry.
        local_folder: Local directory for caching downloaded files.
            Defaults to pooch's default cache directory.

    Returns:
        Path to the downloaded file on local disk.

    Raises:
        ImportError: If pooch is not installed.
        ValueError: If the dataset name is not in the registry.
    """
    try:
        import pooch
    except ImportError as e:
        raise ImportError(
            "pooch is required for downloading test datasets. "
            "Install it with: pip install pooch"
        ) from e

    if registry is None:
        registry = {}

    if name not in registry:
        available = list(registry.keys()) if registry else ["(none)"]
        raise ValueError(
            f"Dataset {name!r} not found in registry. Available datasets: {available}"
        )

    fetcher = pooch.create(
        path=local_folder or pooch.os_cache("ecephys_testing_data"),
        base_url=base_url,
        registry=registry,
    )

    return Path(fetcher.fetch(name))
