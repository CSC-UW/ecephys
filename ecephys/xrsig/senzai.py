import sklearn
import numpy as np
import pandas as pd
import xarray as xr
import colorcet as cc
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.cluster.hierarchy as sch


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
    pcs = xr.DataArray(
        pca.transform(lf.values), dims=("time", "component"), coords={**lf.time.coords}
    )

    if plot:
        _, axes = plt.subplots(pca.n_components, 1, figsize=(36, 1 * pca.n_components))
        for i, ax in enumerate(axes):
            pcs.sel(time=plot_time, component=i).plot.line(
                x="time", add_legend=False, ax=ax
            )

    return pcs


def ica_from_pca(pcs, plot=False, plot_time=slice(None)):
    ica = sklearn.decomposition.FastICA(random_state=42, whiten="unit-variance")
    ics = xr.DataArray(
        ica.fit_transform(pcs.values),
        dims=("time", "component"),
        coords={**pcs.time.coords},
    )
    nPC = len(pcs.component)
    _, axes = plt.subplots(nPC, 1, figsize=(36, 1 * nPC))
    for i, ax in enumerate(axes):
        ics.sel(time=plot_time, component=i).plot.line(
            x="time", add_legend=False, ax=ax
        )

    return ica, ics


def get_ic_loadings(lf, n_components=3, plot=False, plot_time=slice(None)):
    ica = sklearn.decomposition.FastICA(
        n_components=n_components, random_state=42, whiten="unit-variance"
    )
    ics = xr.DataArray(
        ica.fit_transform(lf.values),
        dims=("time", "component"),
        coords={**lf.time.coords},
    )
    if plot:
        _, axes = plt.subplots(n_components, 1, figsize=(36, 1 * n_components))
        for i, ax in enumerate(axes):
            ics.sel(time=plot_time, component=i).plot.line(
                x="time", add_legend=False, ax=ax
            )

    return xr.DataArray(
        ica.mixing_, dims=("channel", "component"), coords={**lf.channel.coords}
    )


def plot_pairwise_band_coherence(band_coh):
    if len(np.unique(band_coh.structure)) > len(sns.color_palette("colorblind")):
        palette = cc.cm.glasbey.colors
    else:
        palette = sns.color_palette("colorblind")
    lut = dict(zip(np.unique(band_coh.structure), palette))
    structure_colors = band_coh.structure.to_series().map(lut)

    df = pd.DataFrame(
        band_coh.values, index=band_coh.chA.values, columns=band_coh.chB.values
    )
    assert (df.values == band_coh.values).all()
    sns.clustermap(
        df,
        row_cluster=False,
        col_cluster=False,
        row_colors=structure_colors,
        col_colors=structure_colors,
        cbar_pos=None,
        cmap="cividis",
    )
    legend_elements = [
        mpatches.Patch(label=structure, facecolor=color)
        for structure, color in lut.items()
    ]
    plt.legend(
        handles=legend_elements,
        handlelength=1,
        loc="upper left",
        ncol=1,
        shadow=True,
        bbox_to_anchor=(-0.3, 1.3),
    )


def cluster_pairwise_band_coherence(band_coh, plot=True):
    df = pd.DataFrame(
        band_coh.values, index=band_coh.chA.values, columns=band_coh.chB.values
    )
    assert (df.values == band_coh.values).all()
    Z = sch.linkage(df.values, method="average", metric="euclidean")

    if plot:
        if len(np.unique(band_coh.structure)) > len(sns.color_palette("colorblind")):
            palette = cc.cm.glasbey.colors
        else:
            palette = sns.color_palette("colorblind")

        lut = dict(zip(np.unique(band_coh.structure), palette))
        structure_colors = band_coh.structure.to_series().map(lut)

        sns.clustermap(
            df,
            row_linkage=Z,
            col_linkage=Z,
            row_colors=structure_colors,
            col_colors=structure_colors,
            cbar_pos=None,
            cmap="cividis",
        )
        legend_elements = [
            mpatches.Patch(label=structure, facecolor=color)
            for structure, color in lut.items()
        ]
        plt.legend(
            handles=legend_elements,
            handlelength=1,
            loc="upper left",
            ncol=1,
            shadow=True,
            bbox_to_anchor=(-0.3, 1.3),
        )

    return Z


def get_dendrogram(band_coh, Z):
    _, ax = plt.subplots(figsize=(30, 5))
    return sch.dendrogram(Z, labels=band_coh.structure.values, leaf_font_size=12, ax=ax)


def plot_pairwise_phase_lag(band_lag):
    if len(np.unique(band_lag.structure)) > len(sns.color_palette("colorblind")):
        palette = cc.cm.glasbey.colors
    else:
        palette = sns.color_palette("colorblind")

    lut = dict(zip(np.unique(band_lag.structure), palette))
    structure_colors = band_lag.structure.to_series().map(lut)

    pldf = pd.DataFrame(
        band_lag.values, index=band_lag.chA.values, columns=band_lag.chB.values
    )
    assert (pldf.values == band_lag.values).all()
    g = sns.clustermap(
        pldf,
        row_cluster=False,
        col_cluster=False,
        row_colors=structure_colors,
        col_colors=structure_colors,
        cbar_pos=None,
        cmap="cividis",
    )

    legend_elements = [
        mpatches.Patch(label=structure, facecolor=color)
        for structure, color in lut.items()
    ]
    lgd = plt.legend(
        handles=legend_elements,
        handlelength=1,
        loc="upper left",
        ncol=1,
        shadow=True,
        bbox_to_anchor=(-0.3, 1.3),
    )
