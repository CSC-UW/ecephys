import itertools
import xarray as xr
import ssqueezepy as ssq
import matplotlib.pyplot as plt
from brainbox import lfp as bblfp
from tqdm.auto import tqdm


def single_pair_coherence(lf, chA, chB):
    freqs, coherence, phase_lag = bblfp.coherence(
        lf.sel(channel=chA).values, lf.sel(channel=chB).values, fs=lf.fs
    )
    return freqs, coherence, phase_lag


# TODO: Parallelize, but first get rid of brainbox.lfp.coherence, which forces us to
# recompute the spectrogram of a given channel for time we compare it against a new one.


def sequential_pairwise_coherence(lf):
    freqs_, _, _ = bblfp.coherence(
        lf.isel(channel=0).values, lf.isel(channel=0).values, fs=lf.fs
    )
    coh = xr.DataArray(
        dims=("chA", "chB", "frequency"),
        coords={
            **lf.rename({"channel": "chA"})["chA"].coords,
            **lf.rename({"channel": "chB"})["chB"].coords,
            "frequency": freqs_,
        },
    )
    lag = coh.copy()
    for chA, chB in tqdm(itertools.combinations(lf.channel.values, 2)):
        freqs, coherence, phase_lag = single_pair_coherence(lf, chA, chB)
        assert all(freqs == freqs_)
        coh.loc[dict(chA=chA, chB=chB)] = coherence
        coh.loc[dict(chA=chB, chB=chA)] = coherence
        lag.loc[dict(chA=chA, chB=chB)] = phase_lag
        lag.loc[dict(chA=chB, chB=chA)] = -phase_lag

    for ch in lf.channel.values:
        coh.loc[dict(chA=ch, chB=ch)] = 1
        lag.loc[dict(chA=ch, chB=ch)] = 0

    return coh, lag


def ssq_cwt_scale_selection(
    N, wavelet="gmw", scaletype="log-piecewise", preset="maximal", nv=32, downsample=4
):
    M = ssq.utils.p2up(N)[0]
    wavelet = ssq.Wavelet(wavelet, N=M)
    min_scale, max_scale = ssq.utils.cwt_scalebounds(wavelet, N=N, preset=preset)
    scales = ssq.utils.make_scales(
        N,
        min_scale,
        max_scale,
        nv=nv,
        scaletype=scaletype,
        wavelet=wavelet,
        downsample=downsample,
    )

    plt.figure(figsize=(22, 6))
    wavelet.viz("filterbank", scales=scales)
