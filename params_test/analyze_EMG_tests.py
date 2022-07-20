import itertools
import os
import os.path
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import tqdm

import emg_from_lfp
import pandas as pd
from run_EMG_tests import (
    BINPATH,
    DATA_DIR,
    DATATYPE,
    EMG_CHANNAME,
    EXPLORED_PARAMS,
    RESULTS_DIR,
    TARGET_SF,
    TEND,
    get_filename,
)

PLOT_ALL_DERIVEDEMGS = True
PLOT_RMSEMG_BEST_DERIVED_EMG = True

RMS_WINSIZE = 2000


def main():

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Compute RMS of true EMG
    def window_rms(a, window_size):
        a2 = np.power(a, 2).flatten()
        window = np.ones(window_size) / float(window_size)
        return np.sqrt(np.convolve(a2, window, "same"))

    print("Load true EMG\n")
    EMG, emg_sf, _ = emg_from_lfp.load.loader_switch(
        BINPATH, datatype=DATATYPE, tEnd=TEND, chanList=[EMG_CHANNAME]
    )
    print("done")
    print("\n\n\n")

    print("Compute RMS of True EMG: \n")
    EMG_rms = window_rms(EMG, RMS_WINSIZE)  # 1-dim
    print("done")
    print("\n\n\n")

    # Downsample rms EMG
    print("Downsample RMS of true EMG:\n")
    dsf, emg_downsample = emg_from_lfp.load.utils.get_dsf(TARGET_SF, emg_sf)
    EMG_rms_ds = EMG_rms[::dsf]
    nsamples = len(EMG_rms_ds)
    # count nan
    print(f"number of NaN in EMG_RMS: {np.sum(np.isnan(EMG_rms_ds))}")
    print("replace Nan with 0")
    EMG_rms_ds[np.isnan(EMG_rms_ds)] = 0
    print("Done. \n\n\n")

    # Load all derived EMGs (same number of samples as RMS EMG)
    print("Load all derived EMGs: \n")
    labels_list = []
    emg_list = []
    params_list = []
    for chanlist, ftype, gpass, gstop, (wp, ws), window_size in tqdm.tqdm(
        list(itertools.product(*EXPLORED_PARAMS))
    ):
        params = {
            "chanlist": chanlist,
            "ftype": ftype,
            "gpass": gpass,
            "gstop": gstop,
            "wp": wp,
            "ws": ws,
            "window_size": window_size,
        }
        print(params)
        filename = get_filename(chanlist, ftype, gpass, gstop, wp, ws, window_size)
        path = Path(DATA_DIR) / (filename + ".npy")
        if not os.path.exists(path):
            continue
        print(filename)

        emg, emgmeta = emg_from_lfp.load_emg(path, tEnd=TEND, desired_length=nsamples)
        emg_list.append(emg)
        labels_list.append(filename)

        params_list.append(params)

    emgs = np.concatenate(emg_list)
    print("Done \n\n\n")

    if PLOT_ALL_DERIVEDEMGS:
        print("Plot all derived EMGs \n")
        fig = plt.figure()
        for i in range(emgs.shape[0]):
            x = np.arange(0, emgs.shape[1]) / TARGET_SF
            plt.plot(x, emgs[i, :], label=labels_list[i])
        lgd = plt.legend(loc=(1.04, 0))
        plt.xlabel("time (s)")
        plt.title("derived EMGs")
        # plt.show()

        fig.savefig(
            RESULTS_DIR / "all_derived_EMG",
            bbox_extra_artists=(lgd,),
            bbox_inches="tight",
        )
        print("Done\n\n\n")

    #  correlation with RMS
    print("Compute correlation between derived EMGs and RMS of true EMG\n")
    corrs = []
    for i in range(emgs.shape[0]):
        print(labels_list[i], end=": ")
        corr, pv = scipy.stats.pearsonr(emgs[i, :], EMG_rms_ds)
        print(corr, pv)
        corrs.append(corr)
    # Save correlations
    print("Done\n")

    # Max
    i_max = np.argmax(corrs)
    maxcorr = corrs[i_max]
    lab_max = labels_list[i_max]
    print(f"\nHighest correlation: {maxcorr}, filename={lab_max}")
    # Min
    i_min = np.argmin(corrs)
    mincorr = corrs[i_min]
    lab_min = labels_list[i_min]
    print(f"\nLowest correlation: {mincorr}, filename={lab_min}")

    if PLOT_RMSEMG_BEST_DERIVED_EMG:
        print("Plot best fitting derived EMG.\n")
        print(f"Best parameters: {labels_list[i_max]}")

        # Plot with RMS
        nsamp_plot = 20000
        fig, ax1 = plt.subplots()
        x = np.arange(0, nsamples) / TARGET_SF
        color = "tab:red"
        ax1.set_xlabel("time (s)")
        ax1.set_ylabel("rms", color=color)
        ax1.plot(
            x[:nsamp_plot],
            EMG_rms_ds[:nsamp_plot],
            color=color,
            label=f"RMS of (true) EMG, winsize=2000",
        )
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = "tab:blue"
        ax2.set_ylabel("derived_EMG", color=color)
        ax2.plot(
            x[:nsamp_plot],
            emgs[i_max, :nsamp_plot],
            color=color,
            label=labels_list[i_max],
        )
        lgd = plt.legend(loc=(1.04, 0))
        plt.title(f"RMS of EMG and derived EMG with best-fitting params. R={maxcorr}")
        plt.show()

        fig.savefig(
            RESULTS_DIR / "best_derived_EMG",
            bbox_extra_artists=(lgd,),
            bbox_inches="tight",
        )

        print("Done")

    # Save results:
    results_df = pd.DataFrame(params_list)
    results_df["labels"] = labels_list
    results_df["corr"] = corrs
    results_df.to_csv(RESULTS_DIR / "corr_to_true_EMG.csv")


if __name__ == "__main__":
    main()
