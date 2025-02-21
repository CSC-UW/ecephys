# ecephys
Python tools for extracellular electrophysiology at the Wisconsin Institute for Sleep and Consciousness.

## Installation

### Requirements:
*Updated 2/21/2025.*

Confirmed to work with Python 3.12. 
Python 3.13 should work. Previously, the bottleneck was `numba` (required by `spikeinterface`, and used in `unit` module code), but this now supports 3.13.

You need to use the `CSC-UW` fork of `spikeinterface`. The specific branch you need depends on your intent. For running spike sorting, use `wisc/sorting`. For everything else, use `wisc/dev`. If you are using `uv` as a package/environment manager (highly recommended), this should be handled for you when you install `ecephys[run-sorting]` or `ecephys[load-sorting]`.
If you are spike sorting, you probably also want `pytorch` for spikeinterface's drift correction.

### Apologies

Unfortuantely, this package cannot be published to PyPI so long as its dependencies include git URLs (e.g. our spikeinterface fork)
The parts of `ecephys` that are tied to our fork because of reliance on the old API need to be factored out into their own package. Then, hopefully updated and brought back in. Otherwise, they will block the package from being able to use the latest spikeinterface indefinitely.

Also, we desparately need to update from `pandas` `1.x` to `2.x` and/or `polars` (WIP).
