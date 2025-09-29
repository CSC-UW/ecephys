from . import senzai, si_extractor
from .core import (
    antialiasing_filter,
    assign_laminar_coordinate,
    bipolar_reference,
    butter_bandpass,
    complex_stft,
    cwt,
    decimate_timeseries,
    demean_trialed,
    dephase_neuropixels,
    detrend_trialed,
    get_channel_indices,
    get_pitch,
    get_segments,
    get_synthetic_emg_defaults,
    get_timeseries_chunk,
    hilbert,
    iterate_timeseries_chunks,
    kernel_current_source_density,
    lazy_mapped_kernel_current_source_density,
    make_trialed,
    mne_filter,
    moving_transform,
    naive_rechunk,
    preprocess_neuropixels_ibl_style,
    spatially_interpolate_timeseries,
    ssq_cwt,
    stft_psd,
    synthetic_emg,
    validate_2d_timeseries,
    validate_3d_timeseries,
    validate_laminar,
    validate_timeseries,
)
from .plt import (
    add_structure_boundaries_to_laminar_plot,
    get_boundary_ilocs,
    plot_laminar_image_horizontal,
    plot_laminar_image_vertical,
    plot_laminar_scalars_horizontal,
    plot_laminar_scalars_vertical,
    plot_laminar_timeseries,
    plot_traces,
)

__all__ = [
    "add_structure_boundaries_to_laminar_plot",
    "antialiasing_filter",
    "assign_laminar_coordinate",
    "bipolar_reference",
    "butter_bandpass",
    "complex_stft",
    "cwt",
    "decimate_timeseries",
    "demean_trialed",
    "dephase_neuropixels",
    "detrend_trialed",
    "ephyviewer",
    "get_boundary_ilocs",
    "get_channel_indices",
    "get_pitch",
    "get_segments",
    "get_synthetic_emg_defaults",
    "get_timeseries_chunk",
    "hilbert",
    "iterate_timeseries_chunks",
    "kernel_current_source_density",
    "lazy_mapped_kernel_current_source_density",
    "make_trialed",
    "mne_filter",
    "moving_transform",
    "naive_rechunk",
    "plot_laminar_image_horizontal",
    "plot_laminar_image_vertical",
    "plot_laminar_scalars_horizontal",
    "plot_laminar_scalars_vertical",
    "plot_laminar_timeseries",
    "plot_traces",
    "preprocess_neuropixels_ibl_style",
    "senzai",
    "si_extractor",
    "spatially_interpolate_timeseries",
    "ssq_cwt",
    "stft_psd",
    "synthetic_emg",
    "validate_2d_timeseries",
    "validate_3d_timeseries",
    "validate_laminar",
    "validate_timeseries",
]


# Lazy imports for expensive visualization modules
_LAZY_IMPORTS = {
    "ephyviewer": ".ephyviewer",
}


def __getattr__(name):
    """Lazy import expensive modules only when accessed."""
    if name in _LAZY_IMPORTS:
        import importlib

        module = importlib.import_module(_LAZY_IMPORTS[name], package=__package__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
