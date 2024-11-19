# ecephys
Python tools for extracellular electrophysiology at the Wisconsin Institute for Sleep and Consciousness.

## Installation

### Requirements:
*Updated 11/19/2024.*

I reccomend Python 3.12. Currently, the bottleneck is `numba` (required by `spikeinterface`, `ibllib`, and used in `unit` module code). It should support 3.13 by the end of the year.

You need to use the `CSC-UW` fork of `spikeinterface`. The specific branch you need depends on your intent. For running spike sorting, use `wisc/sorting`. For everything else, use `wisc/dev`.
If you are spike sorting, you probably also want `pytorch` for spikeinterface's drift correction.

### Apologies

Unfortuantely, this package cannot be published to PyPI so long as its dependencies include git URLs (e.g. our spikeinterface fork)
Also, we desparately need to update from `pandas` `1.x` to `2.x` and/or `polars`.

#### From the past

If you are using `ephyviewer` and Python 3.11, I reccomend PySide6 + Qt6:
```
pip install PySide6
```
Note that `ibllib` depends indirectly on Qt5 (`PyQtWebEngine-Qt5` and `PyQt5-Qt5`). It probably doesn't really need these, but they're going to get installed anyways, and that should be fine. You'll never invoke them. If you need to, see below.

If for some reason you need to use Qt5, this is what used to work and is now broken. We have no working solution for Qt5 currently.
```
conda install -c conda-forge pyqt qtpy qtconsole
pip install pyqt5 ephyviewer
```

