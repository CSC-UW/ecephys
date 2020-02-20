import numpy as np
import itertools
import matplotlib.pyplot as plt
import os.path
import yaml
import EMGfromLFP
import tqdm
import os


BINPATH = "/Users/tom/data/Jazz_GHB_SW/Jazz07recordings/Block-1"
DATATYPE = 'TDT'
TEND = 10000  # Compute shorter EMGs
TARGET_SF = 20
EXPLORED_PARAMS = [
    [
        ["LFPs-1", "LFPs-32"],
        ["LFPs-1", "LFPs-16", "LFPs-32"]
    ],  # LFP_chanList
    ['butt', 'cheby2'],  # ftype
    [1], # gpass
    [30, 60], # gstop
    [
        [(300, 600), (275, 625)],
        [(300, 1200), (275, 1300)],
        [(600, 1200), (300, 1300)],
        [300, 275],
        [600, 300],
    ], # (wp, ws)
    [25, 50] # window_size
]
DATA_DIR = './data'


def get_filename(chanlist, ftype, gpass, gstop, wp, ws, window_size):
    return f"EMG_chan={chanlist}_{ftype}_gpass{gpass}_gstop{gstop}_wp{wp}_ws{ws}_winsize{window_size}"
    

def main():

    with open('./EMG_config.yml', 'r') as f:
        EMG_config = yaml.load(f)

    for chanlist, ftype, gpass, gstop, (wp, ws), window_size in tqdm.tqdm(
        itertools.product(EXPLORED_PARAMS)
    ):
        print(chanlist, ftype, gpass, gstop, wp, ws, window_size)

        filename = get_filename(chanlist, ftype, gpass, gstop, wp, ws, window_size)
        path = DATA_DIR + filename

        if not os.path.exists('./data'):
            os.makedirs('./data', exists_ok=True)
        if os.path.exists(path+'.npy'):
            print('next')
            continue

        cfg = EMG_config.copy()
        cfg = {
            'overwrite': False,  # Do we recompute and overwrite preexisting EMG data
            'LFP_downsample': None,  # sf of LFP used when computing xcorr. 
        }
        cfg.update({
            'EMGdata_savePath': path,
            'LFP_binPath': BINPATH,
            'LFP_datatype': DATATYPE,
            'LFP_chanList': chanlist,
            'ftype': ftype,
            'gpass': gpass,
            'gstop': gstop,
            'wp': wp,
            'ws': ws,
            'window_size': window_size,
            'LFP_tEnd': TEND,
            'sf': TARGET_SF,
        })

        EMGfromLFP.run(cfg)


if __name__ == '__main__':
    main()
