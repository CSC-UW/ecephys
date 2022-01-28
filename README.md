# ecephys
Python tools for extracellular electrophysiology.

## Installation
`tables` is required by `pandas` for writing HDF5 stores, and itself requires an installation of HDF5, which can be a pain.
Either the user (i.e. you) has to do this, or you can let `conda` do it for you (`conda install -c conda-forge pytables`).
For this reason, if installing with pip, `tables` is an extra (i.e. `pip install ecephys[tables]`).
> NB: HDF5 files should no longer be necessary. Everything once saved as HDF5 is now saved as NetCDF, provided through Xarray.

## Contributing
If you wish to make any changes (e.g. add documentation, tests, continuous integration, etc.), please follow the [Shablona](https://github.com/uwescience/shablona) template.

## Interactive plotting with Matplotlib in a JupyterLab notebook.
This is not specific to the `ecephys` module per se, but you may need to do this to plot some results.
```
conda create -n ecephys python=3.7
conda activate ecephys
conda install -c conda-forge jupyterlab
conda install -c conda-forge ipympl
conda install -c conda-forge nodejs
jupyter labextenstion install @jupyter-widgets/jupyterlab-manager
jupyter lab build
```

Now install this package.
```
cd path/to/cloned/ecephys/repo
pip install -e .
```