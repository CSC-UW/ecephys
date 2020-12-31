import os
from pathlib import Path
import pandas as pd

MODULE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
DATAPATHS_FILE = os.path.join(MODULE_DIRECTORY, "datapaths.csv")


def load_datapaths():
    return pd.read_csv(DATAPATHS_FILE)


def get_datapath(**kwargs):
    df = load_datapaths()
    mask = pd.Series(True, index=df.index)
    for column, value in kwargs.items():
        mask = mask & (df[column] == value)

    return Path(df[mask]["path"].iloc[0])
