import numpy as np
import ssqueezepy as ssq


def get_n_fft(fs, s):
    n_fft = int(fs * s)  # Get requested n_fft
    n_fft_fast = ssq.p2up(n_fft)[0]  # Also get fast chunk size (i.e. next power of 2)
    s_fast = n_fft_fast / fs  # Get fast chunk size, in seconds
    return (n_fft, s), (n_fft_fast, s_fast)


def stft(
    x: np.ndarray,
    fs: float,
    window=None,
    n_fft=None,
    win_len=None,
    hop_len=None,
    padtype="reflect",
    dtype="float32",
    t0=0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    See `help(ssqueezepy.stft)`.
    Returns are ordered to match scipy.signal.spectrogram

    Parameters
    ----------
    x: (n_signals, n_samples)
    t0: The time of the first sample in `x`. Used to calcualte accurate STFT chunk times.

    Returns
    -------
    Sfs: (n_freqs)
    stft_times: (n_chunks)
    Sx: (n_signals, n_frequencies, n_chunks)
    """

    if n_fft is None:
        _, (n_fft, s_fft) = get_n_fft(fs, s=1)
        # print(f"n_fft not provided. Using STFT chunk size of {s_fft:.2f}s.")
    if hop_len is None:
        hop_len = n_fft // 4
        # print(f"hop_len not provided. Using STFT hop length of {(hop_len / fs):.2f}s")

    Sx = ssq.stft(
        x,
        window=window,
        n_fft=n_fft,
        win_len=win_len,
        hop_len=hop_len,
        fs=fs,
        padtype=padtype,
        modulated=False,
        derivative=False,
        dtype=dtype,
    )
    Sfs = ssq._ssq_stft._make_Sfs(Sx, fs)

    (n_signals, n_samples) = x.shape
    stft_times = (
        np.linspace(0, n_samples, Sx.shape[-1], endpoint=False) + (hop_len / 2)
    ) / float(fs) + t0

    return Sfs, stft_times, np.atleast_3d(np.abs(Sx))


def cwt(
    x,
    fs,
    wavelet="gmw",
    scales="log-piecewise",
    nv=None,
    padtype="reflect",
    vectorized=True,
    nan_checks=True,
    patience=0,
    cache_wavelet=None,
):
    """Compute a 0-th order continuous wavelet transform (CWT) of `x`.
    For more information, see :func:`ssqueezepy.ssq_cwt`.
    """
    if nv is None and not isinstance(scales, np.ndarray):
        nv = 32
    N = x.shape[-1]
    dt, fs, t = ssq.utils._process_fs_and_t(fs, None, N)
    wavelet = ssq.wavelets.Wavelet._init_if_not_isinstance(wavelet, N=N)
    scales, cwt_scaletype, *_ = ssq.utils.process_scales(
        scales, N, wavelet, nv=nv, get_params=True
    )
    Wx, scales = ssq.cwt(
        x,
        wavelet,
        scales=scales,
        fs=fs,
        nv=nv,
        l1_norm=True,
        derivative=False,
        padtype=padtype,
        rpadded=False,
        vectorized=vectorized,
        astensor=True,
        patience=patience,
        cache_wavelet=cache_wavelet,
        nan_checks=nan_checks,
    )
    was_padded = bool(padtype is not None)
    freqs = ssq.ssqueezing._compute_associated_frequencies(
        scales,
        N,
        wavelet,
        cwt_scaletype,
        maprange="peak",
        was_padded=was_padded,
        dt=dt,
        transform="cwt",
    )
    freqs = freqs[::-1]
    return Wx, freqs, scales
