# ecephys
Python tools for extracellular electrophysiology at the Wisconsin Institute for Sleep and Consciousness.

## Installation

### Requirements:
*Updated 11/19/2024.*

I reccomend Python 3.12. Currently, the bottleneck is `numba` (required by `spikeinterface`, and used in `unit` module code). It should support 3.13 by the end of the year.

You need to use the `CSC-UW` fork of `spikeinterface`. The specific branch you need depends on your intent. For running spike sorting, use `wisc/sorting`. For everything else, use `wisc/dev`.
If you are spike sorting, you probably also want `pytorch` for spikeinterface's drift correction.

### Apologies

Unfortuantely, this package cannot be published to PyPI so long as its dependencies include git URLs (e.g. our spikeinterface fork)
Also, we desparately need to update from `pandas` `1.x` to `2.x` and/or `polars`.