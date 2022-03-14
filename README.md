# ecephys
Python tools for extracellular electrophysiology.

## Installation
`tables` is required by `pandas` for writing HDF5 stores, and itself requires an installation of HDF5, which can be a pain.
Either the user (i.e. you) has to do this, or you can let `conda` do it for you (`conda install -c conda-forge pytables`).
For this reason, if installing with pip, `tables` is an extra (i.e. `pip install ecephys[tables]`).
> NB: HDF5 files should no longer be necessary. Everything once saved as HDF5 is now saved as NetCDF, provided through Xarray.

`phy` is required by a single function, but its dependencies are restrictive, so it is not included as a requirement. You will need to figure out how to deal with this if you plan to load spike sorting results (ask me).

## Contributing
If you wish to make any changes (e.g. add documentation, tests, continuous integration, etc.), please follow the [Shablona](https://github.com/uwescience/shablona) template.
