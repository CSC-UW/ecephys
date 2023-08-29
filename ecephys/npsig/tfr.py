import numpy as np
import scipy.signal
import ssqueezepy as ssq


def get_n_fft(fs, s):
    n_fft = int(fs * s)  # Get requested n_fft
    n_fft_fast = ssq.p2up(n_fft)[0]  # Also get fast chunk size (i.e. next power of 2)
    s_fast = n_fft_fast / fs  # Get fast chunk size, in seconds
    return (n_fft, s), (n_fft_fast, s_fast)


def stft_psd(
    x: np.ndarray,
    fs: float,
    window="dpss",
    n_fft=None,
    hop_len=None,
    t0=0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    See `help(ssqueezepy.stft)`.
    Returns are ordered to match scipy.signal.spectrogram.
    Returns are scaled as if using scaling=density kwarg in scipy.signal.spectrogram.
    NO detrending is performed, as if setting detrend=False in scipy.signal.spectrogram.
    Positive frequncies (+ DC, very important!) are returned.
    To convert to dB, use 10 * np.log10(Sxx).

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
    if hop_len is None:
        hop_len = n_fft // 4
    if window == "dpss":
        window = scipy.signal.windows.dpss(n_fft, max(4, n_fft // 8), sym=False)
    elif window == "tukey":
        window = scipy.signal.windows.tukey(n_fft, alpha=0.25, sym=False)
    elif isinstance(window, np.ndarray):
        assert window.size == n_fft, "Window size must match n_fft"
    else:
        raise ValueError(f"Unsupported window type: {window}")

    Sx = ssq.stft(
        x,
        window=window,
        n_fft=n_fft,
        win_len=n_fft,
        hop_len=hop_len,
        fs=fs,
        padtype="reflect",
        modulated=False,
        derivative=False,
        dtype="float32",
    )
    Sfs = ssq._ssq_stft._make_Sfs(Sx, fs)  # This uses Sx dtype, so leave it here.

    # Convert complex specturm to PSD
    Sx = np.conjugate(Sx) * Sx  # (Sx.conj() * Sx).real == square(abs(Sx))
    win = window.astype(Sx.dtype)  # complex64
    scale = 1.0 / (fs * (win * win).sum())
    Sx = Sx * scale

    # Since we are using a one-sided spectrum, reassign power from negative frequencies
    idxr = [
        slice(None),
    ] * Sx.ndim
    freq_dim = Sx.ndim - 2
    if n_fft % 2:
        idxr[freq_dim] = slice(1, None)
        Sx[tuple(idxr)] *= 2
    else:
        # Last point is unpaired Nyquist freq point, don't double
        idxr[freq_dim] = slice(1, -1)
        Sx[tuple(idxr)] *= 2

    (n_signals, n_samples) = x.shape
    stft_times = (
        np.linspace(0, n_samples, Sx.shape[-1], endpoint=False) + (hop_len / 2)
    ) / float(fs) + t0

    return Sfs, stft_times, np.atleast_3d(Sx.real)


def complex_stft(
    x: np.ndarray,  # (n_signals, n_samples)
    fs: float,
    n_fft: int = None,
    hop_len: int = None,
    t0: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n_fft is None:
        _, (n_fft, s_fft) = get_n_fft(fs, s=1)
    if hop_len is None:
        hop_len = n_fft // 4

    fx, tx, Sxx = scipy.signal.spectrogram(
        x,
        fs=fs,
        nperseg=n_fft,
        noverlap=(n_fft - hop_len),
        mode="complex",
        detrend=False,
    )

    return fx, tx + t0, Sxx  # (n_signals, n_frequencies, n_segments)


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
