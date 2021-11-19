import pandas as pd


def rebase_time(sig, in_place=True):
    """Rebase time and timedelta coordinates so that t=0 corresponds to the beginning
    of the datetime dimension."""
    if not in_place:
        sig = sig.copy()
    sig["timedelta"] = sig.datetime - sig.datetime.min()
    sig["time"] = sig["timedelta"] / pd.to_timedelta(1, "s")
    return sig


# TODO: Remove this function. It is ugly and dangerous. Function graveyard gist?
def dtdim(dat):
    assert "datetime" in dat.coords, "Datetime coordinate not found."
    if "time" in dat.dims:
        return dat.swap_dims({"time": "datetime"})
    elif "timedelta" in dat.dims:
        return dat.swap_dims({"timedelta": "datetime"})
    else:
        raise ValueError(
            "Exactly one of `time` or `timedelta` must be present as a dimension."
        )
