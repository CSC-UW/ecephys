import os
import pandas as pd

MODULE_DIRECTORY = os.path.dirname(os.path(abspath(__file__)))
DATAPATHS_FILE = os.path.join(MODULE_DIRECTORY, "datapaths.csv")


def load_datapaths():
    df = pd.load(DATAPATHS_FILE)