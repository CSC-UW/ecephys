import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch, spectrogram, find_peaks

def rms(data):

    """
    Computes root-mean-squared voltage of a signal
    Input:
    -----
    data - numpy.ndarray
    Output:
    ------
    rms_value - float
    
    """

    return np.power(np.mean(np.power(data.astype('float32'),2)),0.5)

class Recording:
    def __init__(self, file, datafmt, fs, gain):
        self.file = file
        self.datafmt = datafmt
        self.fs = fs
        self.gain = gain
        
        self.numNeuralChannels = 384
        if self.datafmt == "OpenEphys":
            self.numRowsInFile = self.numNeuralChannels
        elif self.datafmt == "SpikeGLX":
            self.numRowsInFile = self.numNeuralChannels + 1
        else:
            raise("Unexpected datafmt")
        
        self._data = None
        self._rms_signal = None
        self._psd = None
        self._spectrogram = dict()

        
    @property
    def data(self):
        if self._data is None:
            self.load()
            
        return self._data

    
    @property
    def rms_signal(self):
        if self._rms_signal is None:
            self.compute_rms_signal()
            
        return self._rms_signal
    
    
    @property
    def psd(self):
        if self._psd is None:
            self.compute_psd()
            
        return self._psd
    
    
    @property
    def psd_peaks(self):
        fft_freqs, power = self.psd
        mean_power_dbs = np.mean(np.log10(power), 1)
        ipks, pkinfo = find_peaks(mean_power_dbs)
        return fft_freqs[ipks]
    
    
    def spectrogram(self, channel):
        if self._spectrogram[channel] is None:
            self.compute_spectrogram(channel)
            
        return self._spectrogram[channel]
    
    
    def load(self):
        rawData = np.memmap(self.file, dtype='int16', mode='r')
        rawData = np.reshape(rawData, (int(rawData.size/self.numRowsInFile), self.numRowsInFile))
        self._data = rawData[:, 0:(self.numNeuralChannels)]
        
        
    def compute_rms_signal(self, start_time=0, end_time=None):
        data = self.data
        numChannels = self.numNeuralChannels
        
        if not end_time:
            end_time = int(data.shape[0] / self.fs)

        numIterations = 10
        params = dict()
        params['start_time'] = start_time
        params['time_interval'] = min((end_time - start_time) / numIterations, 5)
        params['skip_s_per_pass'] = int((end_time - start_time) / numIterations)
        ephys_params = dict()
        ephys_params['sample_rate'] = self.fs
        if self.datafmt == "OpenEphys":
            ephys_params['bit_volts'] = 0.195
        elif self.datafmt == "SpikeGLX":
            ephys_params['bit_volts'] = (0.6 / 512 / self.gain) * 1e6
        else: 
            raise("Unexpected datafmt")

        offsets = np.zeros((numChannels, numIterations), dtype = 'int16')
        rms_signal = np.zeros((numChannels, numIterations), dtype='float')

        for i in range(numIterations):

            start_sample = int((params['start_time'] + params['skip_s_per_pass'] * i)* ephys_params['sample_rate'])
            end_sample = start_sample + int(params['time_interval'] * ephys_params['sample_rate'])

            for ch in range(numChannels):
                chunk_data = data[start_sample:end_sample, ch]
                offsets[ch,i] = np.median(chunk_data)
                median_subtr = chunk_data - offsets[ch,i]
                rms_signal[ch,i] = rms(median_subtr) * ephys_params['bit_volts'] # Put data in units of microvolts

        self._rms_signal = rms_signal
    
    
    def compute_psd(self, start_time, end_time):
        data = self.data
        nchannels = self.numNeuralChannels
        sample_frequency = self.fs
        window_start = start_time
        window_length = end_time - start_time

        nfft = 4096*2
        if self.fs == 2500:
            nperseg = 2048
        elif self.fs == 30000:
            nperseg = 512
        else:
            raise
        mask_chans = 192

        startPt = int(sample_frequency*window_start)
        endPt = startPt + int(sample_frequency*window_length)

        channels = np.arange(nchannels).astype('int')

        chunk = np.copy(data[startPt:endPt,channels])

        for ch in np.arange(nchannels):
            chunk[:,ch] = chunk[:,ch] - np.median(chunk[:,ch])

        power = np.zeros((int(nfft/2+1), nchannels))

        for ch in np.arange(nchannels):

            sample_frequencies, Pxx_den = welch(chunk[:,ch], fs=sample_frequency, nfft=nfft, nperseg=nperseg)
            power[:,ch] = Pxx_den

        self._psd = (sample_frequencies, power)
    
    
    def compute_spectrogram(self, ch, start_time, end_time):
        nchannels = self.numNeuralChannels
        window_start = start_time
        window_length = end_time - start_time
        
        nfft = 4096*2
        if self.fs == 2500:
            nperseg = 2048
        elif self.fs == 30000:
            nperseg=512
        else:
            raise
        mask_chans = 192

        startPt = int(self.fs*window_start)
        endPt = startPt + int(self.fs*window_length)

        iCh = ch - 1
        chunk = np.copy(self.data[startPt:endPt,iCh])
        chunk = chunk - np.median(chunk)

        self._spectrogram[ch] = spectrogram(chunk, fs=self.fs, nfft=nfft, noverlap=0, nperseg=nperseg)
    
    
    def plot_rms_signal(self):
        fig = plt.figure()
        plt.plot(np.median(self.rms_signal, 1))
        plt.xlabel('Channel')
        plt.ylabel('RMS Signal (uV)')
        plt.show()
    
    
    def plot_psd(self):
        fft_freqs, power = self.psd
        
        fig = plt.figure()
        plt.subplot(1, 4, 1)
        plt.semilogx(fft_freqs, np.log10(power))
        plt.xlabel('frequency [Hz]')
        plt.ylabel('PSD [V**2/Hz]')

        plt.subplot(1, 4, 2)
        plt.plot(fft_freqs, np.log10(power))
        plt.xlabel('frequency [Hz]')
        plt.ylabel('PSD [V**2/Hz]')

        plt.subplot(1, 4, 3)
        plt.pcolormesh(fft_freqs, np.arange(power.shape[1]), np.log10(power).T)
        plt.xlim(np.min(fft_freqs[fft_freqs > 0]), np.max(fft_freqs))
        plt.xscale('log')
        plt.xlabel('frequency [Hz]')
        plt.ylabel('PSD [V**2/Hz]')

        plt.subplot(1, 4, 4)
        plt.pcolormesh(fft_freqs, np.arange(power.shape[1]), np.log10(power).T)
        plt.xlim(np.min(fft_freqs[fft_freqs > 0]), np.max(fft_freqs))
        plt.xlabel('frequency [Hz]')
        plt.ylabel('PSD [V**2/Hz]')

        plt.show()
    
    
    def plot_psd_peaks(self):
        fft_freqs, power = self.psd
        pks = self.psd_peaks

        mean_power_dbs = np.mean(np.log10(power), 1)

        fig = plt.figure()
        plt.plot(fft_freqs, mean_power_dbs)
        plt.xlabel('frequency [Hz]')
        plt.ylabel('PSD [V**2/Hz]')
        plt.vlines(pks, np.min(mean_power_dbs), np.max(mean_power_dbs), linestyles='dashed')
        plt.show()
    
    
    def plot_spectrogram(self, channel):
        f, t, Sxx = self.spectrogram(channel)
        
        fig = plt.figure()
        plt.subplot(2, 1, 1)
        plt.pcolormesh(t, f, np.log10(Sxx))
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')

        plt.subplot(2, 1, 2)
        plt.pcolormesh(t, f, np.log10(Sxx))
        plt.ylim(np.min(f[f > 0]), np.max(f))
        plt.yscale('log')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')

        plt.show()
    

class LFP_Recording(Recording):
    def __init__(self, file, datafmt):
        super().__init__(file, datafmt, 2500, 250)
        
class AP_Recording(Recording):
    def __init__(self, file, datafmt):
        super().__init__(file, datafmt, 30000, 500)

        
class Dataset:
    def __init__(self, lfp_file, ap_file, datafmt):
        self.lfp = LFP_Recording(lfp_file, datafmt)
        self.ap = AP_Recording(ap_file, datafmt)    