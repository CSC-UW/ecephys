# emg_from_lfp

Derive EMG from LFP through correlation of highfrequency activity.

This code is basically a Python translation and readaptation of the
bz_EMGFromLFP.m function from Buzsaki's lab.
(`https://github.com/buzsakilab/buzcode/blob/master/detectors/bz_EMGFromLFP.m`).
Based on Erik Schomburg's and code and work, published in `Theta Phase
Segregation of Input-Specific Gamma Patterns in Entorhinal-Hippocampal Networks,
Schomburg et al., Neuron 2014.``


Tom Bugnon, 01/2020


## Installation

- Clone, fork, or download this package `emg_from_lfp` (<https://github.com/CSC-UW/emg_from_lfp>)

```
# From the `emg_from_lfp` directory you just downloaded
pip install -e .
```

## Command-line usage:


1.  Copy the default configuration file (`EMG_config_df`)

2.  Manually set the parameters for the copied config file.


- From the command line (make sure you're in your virtualenvironment)

`python -m emg_from_lfp <path_to_config_file>`

- From python:

```python
import emg_from_lfp

emg_from_lfp.run({config_dict}) # See function docstring
```

3. Load the computed data with

```python
import emg_from_lfp

# Load
emg_from_lfp.load_EMG(
  <path_to_EMGdata>, tStart=None, tEnd=None, desired_length=None
)
```
