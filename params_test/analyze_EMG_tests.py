import numpy as np
import itertools
import matplotlib.pyplot as plt
import os.path
import EMGfromLFP
import sleepscore.load
import scipy.stats
import tqdm

from run_EMG_tests import BINPATH, DATATYPE, EXPLORED_PARAMS, DATA_DIR, get_filename, TEND, TARGET_SF


PLOT_ALL_DERIVEDEMGS = True
PLOT_RMSEMG_BEST_DERIVED_EMG = True

RMS_WINSIZE = 2000


def main():

    # Compute RMS of true EMG
    def window_rms(a, window_size):
        a2 = np.power(a, 2).flatten()
        window = np.ones(window_size)/float(window_size)
        return np.sqrt(np.convolve(a2, window, 'same'))
    EMG, emg_sf, _ = sleepscore.load.loader_switch(
        BINPATH, datatype=DATATYPE, tEnd=TEND, chanList=['EMGs-1']
    )
    print('RMS')
    # plt.figure()
    # for winsize in 50, 100, 200, 1000:
    #     EMG_rms = window_rms(EMG, winsize) # 1-dim
    #     plt.plot(EMG_rms)
    # plt.show()
    EMG_rms = window_rms(EMG, RMS_WINSIZE) # 1-dim
    print('done')

    # Downsample rms EMG
    
    dsf, emg_downsample = sleepscore.load.utils.get_dsf(TARGET_SF, emg_sf)
    EMG_rms_ds = EMG_rms[::dsf]
    nsamples = len(EMG_rms_ds)
    # count nan
    print('number of NaN in EMG_RMS: {np.sum(np.isnan(EMG_rms_ds))}')
    print('replace Nan with 0')
    EMG_rms_ds[np.isnan(EMG_rms_ds)] = 0

    # Load downsampled EMG for comparison with RMS
    EMG_ds, _, _ = sleepscore.load.read_TDT(
        BINPATH, tEnd=TEND, chanList=['EMGs-1'], downSample=TARGET_SF
    )

    # # PLot EMG and rms EMG
    # nsamp_plot = nsamples
    # plt.figure()
    # x = np.arange(0, nsamples)/target_sf
    # plt.plot(x[:nsamp_plot], EMG_ds.flatten()[:nsamp_plot], label='EMG')
    # plt.plot(x[:nsamp_plot], EMG_rms_ds.flatten()[:nsamp_plot], label='rms EMG')
    # plt.legend()
    # plt.xlabel('time (s)')
    # plt.show()

    
    # Load all derived EMGs (same number of samples as RMS EMG)
    labels_list = []
    emg_list = []
    for chanlist, ftype, gpass, gstop, (wp, ws), window_size in tqdm.tqdm(
        itertools.product(EXPLORED_PARAMS)
    ):
        print(chanlist, ftype, gpass, gstop, wp, ws, window_size)
        filename = get_filename(chanlist, ftype, gpass, gstop, wp, ws, window_size)
        path = DATA_DIR + filename + '.npy'
        if not os.path.exists(path):
            continue
        print(filename)

        emg, emgmeta = EMGfromLFP.load_EMG(path, tEnd=TEND, desired_length=nsamples)
        emg_list.append(emg)
        labels_list.append(filename)

    emgs = np.concatenate(emg_list)


    if PLOT_ALL_DERIVEDEMGS:
        plt.figure()
        for i in range(emgs.shape[0]):
            x = np.arange(0, emgs.shape[1])/TARGET_SF
            plt.plot(x, emgs[i, :], label=labels_list[i])
        plt.legend()
        plt.xlabel('time (s)')
        plt.title("derived EMGs")
        plt.show()
        


    #  correlation with RMS 
    corrs = []
    for i in range(emgs.shape[0]):
        print(labels_list[i], end=": ")
        corr, pv = scipy.stats.pearsonr(emgs[i, :], EMG_rms_ds)
        print(corr, pv)
        corrs.append(corr)
        
    # Max
    i_max = np.argmax(corrs)
    maxcorr = corrs[i_max]
    lab_max = labels_list[i_max]
    print(f'\nHighest correlation: {maxcorr}, filename={lab_max}')
    # Min
    i_min = np.argmin(corrs)
    mincorr = corrs[i_min]
    lab_min = labels_list[i_min]
    print(f'\nLowest correlation: {mincorr}, filename={lab_min}')
    
    if PLOT_RMSEMG_BEST_DERIVED_EMG:
        # Plot with RMS
        nsamp_plot = 20000
        fig, ax1 = plt.subplots()
        x = np.arange(0, nsamples)/TARGET_SF
        color = 'tab:red'
        ax1.set_xlabel('time (s)')
        ax1.set_ylabel('rms', color=color)
        ax1.plot(x[:nsamp_plot], EMG_rms_ds[:nsamp_plot], color=color,
                 label=f'RMS of (true) EMG, winsize=2000')
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:blue'
        ax2.set_ylabel('derived_EMG', color=color)
        ax2.plot(x[:nsamp_plot], emgs[i_max, :nsamp_plot], color=color,
                 label=labels_list[i_max])
        plt.legend()
        plt.title(f"RMS of EMG and derived EMG with best-fitting params. R={maxcorr}")
        plt.show()


if __name__ == '__main__':
    main()
