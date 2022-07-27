import xarray as xr
import sklearn
import matplotlib.pyplot as plt
import numpy as np

def fit_principal_components(lf, n_components=10, thresh=0.95, verbose=True):
    pca = sklearn.decomposition.PCA(n_components=n_components)
    pca.fit(lf.values)

    if verbose:
        fig, axes = plt.subplots(1, 2, figsize=(20, 5))
        print(f"Explained variance ratios: {pca.explained_variance_ratio_}")
        axes[0].plot(pca.explained_variance_ratio_)
        axes[0].set_title("Explained variance ratio")
        axes[1].plot(pca.singular_values_)
        axes[1].set_title("Singular values")

    evr = np.cumsum(pca.explained_variance_ratio_)
    nKeep = max(np.sum(evr <= thresh), 1)
    print(f"{nKeep} major components detected.")

    return pca, nKeep

def get_principal_components(lf, pca, plot=False, plot_time=slice(None)):
    pcs = xr.DataArray(pca.transform(lf.values), dims=("time", "component"), coords={**lf.time.coords})

    if plot:
        fig, axes = plt.subplots(pca.n_components, 1, figsize=(36, 1 * pca.n_components))
        for i, ax in enumerate(axes):
            pcs.sel(time=plot_time, component=i).plot.line(x='time', add_legend=False, ax=ax)

    return pcs

def ica_from_pca(pcs, plot=False, plot_time=slice(None)):
    ica = sklearn.decomposition.FastICA(random_state=42, whiten='unit-variance')
    ics = xr.DataArray(ica.fit_transform(pcs.values), dims=("time", "component"), coords={**pcs.time.coords})
    nPC = len(pcs.component)
    fig, axes = plt.subplots(nPC, 1, figsize=(36, 1 * nPC))
    for i, ax in enumerate(axes):
        ics.sel(time=plot_time, component=i).plot.line(x='time', add_legend=False, ax=ax)

    return ica, ics

def get_ic_loadings(lf, n_components=3, plot=False, plot_time=slice(None)):
    ica = sklearn.decomposition.FastICA(n_components=n_components, random_state=42, whiten='unit-variance')
    ics = xr.DataArray(ica.fit_transform(lf.values), dims=("time", "component"), coords={**lf.time.coords})
    if plot:
        fig, axes = plt.subplots(n_components, 1, figsize=(36, 1 * n_components))
        for i, ax in enumerate(axes):
            ics.sel(time=plot_time, component=i).plot.line(x='time', add_legend=False, ax=ax)

    return xr.DataArray(ica.mixing_, dims=("channel", "component"), coords={**lf.channel.coords})