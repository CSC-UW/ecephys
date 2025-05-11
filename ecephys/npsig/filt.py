import matplotlib.pyplot as plt
import numpy as np
import scipy.signal


def plot_butter_bandpass_properties(sos, fs, order, xlim=None):
    # Plot the frequency response for a few different orders.
    plt.figure()
    plt.clf()
    w, h = scipy.signal.sosfreqz(sos, worN=2000, fs=fs)
    plt.plot(w, abs(h), label="order = %d" % order)
    if xlim is not None:
        plt.xlim(xlim)
    plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)], "--", label="sqrt(0.5)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Gain")
    plt.grid(True)
    plt.legend(loc="best")


def get_butter_bandpass_coefs(lowcut, highcut, fs, order, plot=True):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = scipy.signal.butter(order, [low, high], btype="band", output="sos")
    if plot:
        bw = highcut - lowcut
        plot_butter_bandpass_properties(
            sos, fs, order, xlim=(lowcut - bw, highcut + bw)
        )
    return sos


def butter_bandpass(data, lowcut, highcut, fs, order, plot=True):
    """Attenuation by order:
    order=1: 6dB/octave
    order=2: 12dB/octave
    order=3: 18dB/octave
    order=4: 24dB/octave
    order=5: 30dB/octave
    order=6: 36dB/octave

    Higher orders incur more computational cost (often negligible), increase potential
    for ringing and smearing, and may be numerically unstable. But, of coruse, come with
    narrower transition bands and better stopband attenuation.
    """
    sos = get_butter_bandpass_coefs(lowcut, highcut, fs, order, plot)
    y = scipy.signal.sosfiltfilt(sos, data)
    return y


def estimate_impulse_response_len(b, a, eps=1e-3):
    """From scipy filtfilt docs.
    Input:
         b, a : filter params
         eps  : How low must the signal drop to? (default 1e-2)
    """

    _, p, _ = scipy.signal.tf2zpk(b, a)
    r = np.max(np.abs(p))
    approx_impulse_len = int(np.ceil(np.log(eps) / np.log(r)))

    return approx_impulse_len


def antialiasing_filter(x: np.ndarray, q: int, time_axis=0) -> np.ndarray:
    result_type = x.dtype
    assert (result_type == np.float64) or (result_type == np.float32), (
        "Data must be float64 or float32."
    )
    assert q < 13, (
        "It is recommended to call `decimate` multiple times for downsampling factors higher than 13. See scipy.signal.decimate docs."
    )
    n = 8
    sos = scipy.signal.cheby1(n, 0.05, 0.8 / q, output="sos")
    sos = np.asarray(sos, dtype=result_type)
    return scipy.signal.sosfiltfilt(sos, x, axis=time_axis)
