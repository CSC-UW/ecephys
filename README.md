## ecephys
Python tools for extracellular electrophysiology.

### Interactive plotting with Matplotlib in a JupyterLab notebook.
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

### Contributing
If you wish to make any changes (e.g. add documentation, tests, continuous integration, etc.), please follow the [Shablona](https://github.com/uwescience/shablona) template.