"""Utility functions for data loading and transformation."""

import yaml
import numpy as np


def get_dsf(downsample, sf):
    """Get the downsampling factor.

    Parameters
    ----------
    downsample : float
        The down-sampling frequency.
    sf : float
        The sampling frequency
    
    Return:
        dsf, target_sf
    """
    if downsample is None:
        return 1, sf
    else:
        assert all([isinstance(k, (int, float)) for k in (downsample, sf)])
        dsf = int(np.round(sf / downsample))
        downsample = float(sf / dsf)
        return dsf, downsample


def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.load(f)


def save_yaml(path, data):
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
