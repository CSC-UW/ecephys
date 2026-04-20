# AGENTS.md — ecephys

## What This Is

Core Python library for extracellular electrophysiology. Generic tools that could be useful outside of WISC — signal processing, spike train analysis, sleep scoring, probe data loading, and project organization. Used by nearly every other package in the workspace.

**Version:** 0.0.6 | **Python:** >=3.9 | **Build:** hatchling | **No CLI entry points**

Cannot be published to PyPI due to custom GitHub fork dependencies (ephyviewer, kCSD-python, ssqueezepy).

## Build & Test

```bash
# No independent test suite currently exists
# Linting
uv run ruff check src/

# Install (editable, from workspace root)
cd gfys_workspace && uv sync --all-extras --group dev
```

## Module Catalog

### Data Loading

| Module | Purpose | Key exports |
|--------|---------|-------------|
| `sglx` | SpikeGLX metadata and file management | `catgt`, `file_mgmt`, `meta`, `load_nidq_analog()` |
| `sglxr` | Neuropixels probe reading and registration | `ImecMap`, `get_timestamps()`, `memmap_dask_array()` |
| `tdtxr` | TDT (Tucker-Davis) data loading | `load_stream_store()`, `stream_store_to_xarray()` |

### Signal Processing

| Module | Purpose | Key exports |
|--------|---------|-------------|
| `npsig` | NumPy-based signal processing | `butter_bandpass()`, `complex_stft()`, `decimate_timeseries()`, `get_pitts_csd()`, `stft_psd()` |
| `dasig` | Dask-based signal processing (scales to large data) | `antialiasing_filter()`, `butter_bandpass()`, `mne_filter()`, `hilbert()` |
| `xrsig` | xarray-based signal processing (lazy, heavy) | `core` (CSD, STFT, synthetic EMG), `plt` (visualization), `si_extractor` |

Signal abstraction levels: NumPy (`npsig`) → Dask (`dasig`) → xarray (`xrsig`), for scaling and metadata.

### Analysis

| Module | Purpose | Key exports |
|--------|---------|-------------|
| `units` | Spike train analysis | `bin_train()`, `bin_trains()`, `get_peths_from_trains()` |
| `units.siks_sorting` | SpikeInterface sorting wrapper | (lazy import — heavy) |
| `units.correlograms` | Autocorrelograms | (lazy import — heavy) |
| `hypnogram` | Sleep stage scoring | `Hypnogram`, `DatetimeHypnogram`, `FloatHypnogram`, `clean()`, `condense()` |
| `sync` | Cross-hardware synchronization | `extract_barcodes_from_times()`, `fit_barcode_times()`, `remap_times()` |

### Infrastructure

| Module | Purpose | Key exports |
|--------|---------|-------------|
| `wne` | Project/experiment organization (WNE convention) | `Project`, `ProjectLibrary`, `Subject` |
| `wne.sglx` | SpikeGLX-specific project management | `SGLXProject`, `SGLXSubject`, `SGLXSubjectLibrary` |
| `utils` | Utility functions | `read_htsv()`, `write_htsv()`, `hotfix_times()` |
| `sharptrack` | Probe position estimation | (standalone module) |
| `plot` | Matplotlib visualization utilities | (standalone module) |

## Key Architectural Constraints

- **`ecephys.wne` must not depend on `wisc_ecephys_tools`.** The dependency flows one way: `wisc_ecephys_tools` → `ecephys`, never the reverse.
- **Lazy imports for heavy modules.** Modules that pull in spikeinterface, sklearn, matplotlib, or PySide6 are imported on-demand, not at package init. This keeps `import ecephys` fast. Heavy modules include: `xrsig.core`, `xrsig.plt`, `xrsig.ephyviewer`, `units.siks_sorting`, `units.correlograms`, `wne.sglx`.
- **No backward compatibility guarantees.** Breaking changes are acceptable. Update dependent code in tandem.

## WNE Subsystem

The `ecephys.wne` module implements the Wisconsin Neurophysiology Environment convention for organizing multi-session recordings:

```python
from ecephys.wne import project

lib = project.ProjectLibrary.from_yaml("projects.yaml")
proj = lib.get_project("offproj")
proj.get_experiment_subject_directory("novel_objects_deprivation", "CNPIX12-Santiago")
```

`wisc_ecephys_tools` wraps this with WISC-specific YAML files and convenience functions. See `wisc_ecephys_tools/AGENTS.md` for details.

## See Also

- Root `AGENTS.md` — workspace overview, code style, recording discoveries
- `wisc_ecephys_tools/AGENTS.md` — WISC-specific infrastructure built on ecephys
- `gfys_workspace/docs/DATA.md` — data hierarchy details
