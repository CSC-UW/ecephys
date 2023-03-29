# ecephys
Python tools for extracellular electrophysiology at the Wisconsin Institute for Sleep and Consciousness.

## Installation

### Requirements:

 - spikeinterface, ``CSC-UW`` github fork, `wisc/dev` branch:

```
pip install "git+ssh://git@github.com/CSC-UW/spikeinterface.git@wisc/dev
```
or (editable mode)
```
git clone https://github.com/CSC-UW/spikeinterface.git
cd spikeinterface
pip install -e .
```

  - For spike-interface drift correction: `pytorch``

### Ecephys


```
git clone https://github.com/CSC-UW/ecephys.git
cd ecephys
```
Then `pip install -e .` or `poetry install` (preferred)

Unfortuantely, this package cannot be published to PyPI so long as its dependencies include git URLs (e.g. kCSD-python, our spikeinterface fork, etc.)
