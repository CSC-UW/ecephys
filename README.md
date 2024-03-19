# ecephys
Python tools for extracellular electrophysiology at the Wisconsin Institute for Sleep and Consciousness.

## Installation

### Requirements:
*Updated 1/22/2024 -- May be more recent than `CSC-UW/ece-env`.* 

I reccomend Python 3.11. `pyfftw` technically allows you to use Python >3.11, but only provides prebuilt wheels through 3.11, and if you are using mamba/conda and CPython (probable), `pyfftw`'s build will probably fail for 3.12.     

You need to use the `CSC-UW` fork of `spikeinterface`. The specific branch you need depends on your intent. For running spike sorting, use `wisc/sorting`. For everything else, use `wisc/dev`. 
If you are spike sorting, you probably also want `pytorch` for spikeinterface's drift correction. 
If you are using `ephyviewer` and Python 3.11, I reccomend PySide6 + Qt6:
```
pip install PySide6
```
Note that `ibllib` depends indirectly on Qt5 (`PyQtWebEngine-Qt5` and `PyQt5-Qt5`). It probably doesn't really need these, but they're going to get installed anyways, and that should be fine. You'll never invoke them. If you need to, see below. 

### Other

Unfortuantely, this package cannot be published to PyPI so long as its dependencies include git URLs (e.g. kCSD-python, our spikeinterface fork, etc.)

#### From the past

If for some reason you need to use Qt5, this is what used to work and is now broken. We have no working solution for Qt5 currently. 
```
conda install -c conda-forge pyqt qtpy qtconsole
pip install pyqt5 ephyviewer
```
