import pandas as pd

def get_start_time(H, row):
    if row.name == 0:
        start_time = 0.0
    else:
        start_time = H.loc[row.name - 1].end_time

    return start_time

def load_visbrain_hypnogram(hypno_path):
    H = pd.read_csv(hypno_path, sep='\t', names=['state', 'end_time'], skiprows=1)
    H['start_time'] = H.apply(lambda row: get_start_time(H, row), axis=1)

    return H

def filter_states(H, states):
    return H[H['state'].isin(states)]