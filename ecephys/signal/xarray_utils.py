import pandas as pd


def rebase_time(sig, in_place=True):
    if not in_place:
        sig = sig.copy()
    sig["timedelta"] = sig.datetime - sig.datetime.min()
    sig["time"] = sig["timedelta"] / pd.to_timedelta(1, "s")
    return sig