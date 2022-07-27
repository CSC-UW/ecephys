import numpy as np
import matplotlib.pyplot  as plt
from scipy.signal import butter, filtfilt, freqz

def plot_butter_bandpass_properties(b, a, fs, order, xlim=None):
    # Plot the frequency response for a few different orders.
    plt.figure()
    plt.clf()
    w, h = freqz(b, a, worN=2000)
    plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)
    if xlim is not None:
        plt.xlim(xlim)

    plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)], '--', label='sqrt(0.5)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.grid(True)
    plt.legend(loc='best')

def get_butter_bandpass_coefs(lowcut, highcut, fs, order, plot=True):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    if plot:
        bw = highcut - lowcut
        plot_butter_bandpass_properties(b, a, fs, order, xlim=(lowcut - bw, highcut + bw))
    return b, a

def butter_bandpass(data, lowcut, highcut, fs, order, plot=True):
    b, a = get_butter_bandpass_coefs(lowcut, highcut, fs, order, plot)
    y = filtfilt(b, a, data)
    return y